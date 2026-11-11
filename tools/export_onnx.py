"""
export_onnx.py — S1.1: Export LLaMA-3.2-1B-Instruct to ONNX for the llm_server engine.

One script does everything (replaces llama_jetson's export_onnx + strip_lm_head +
save_lm_head_bin three-step):

  1. optimum export, task="text-generation-with-past"
     → merged model.onnx with DYNAMIC seq_len AND past_len. The same graph serves
       prefill (seq=N, past=0) and decode (seq=1, past=t) — one engine, two TRT
       optimization profiles.
  2. Strip lm_head → backbone outputs last_hidden_state [batch, seq, 2048].
     Kept external so the engine outputs hidden states for ALL prompt positions
     but lm_head (2048×128256 = 501 MB matmul) runs only on the LAST position.
  3. Save lm_head_weight.bin — raw fp16 [HIDDEN=2048, VOCAB=128256] row-major,
     loaded by the C++ runtime via fread (same format as before).

Run from llm_server/:
    python tools/export_onnx.py [--model meta-llama/Llama-3.2-1B-Instruct]

Output → llm_server/onnx/
    model_backbone.onnx  (+ .onnx_data)   lm_head_weight.bin   tokenizer.json
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HIDDEN = 2048
VOCAB = 128256


def find_producer(graph, tensor_name):
    for node in graph.node:
        if tensor_name in node.output:
            return node
    return None


def print_io(model, label):
    print(f"\n{label} I/O:")
    for io_list, tag in ((model.graph.input, "IN "), (model.graph.output, "OUT")):
        for x in io_list:
            shape = [d.dim_value if d.dim_value > 0 else d.dim_param
                     for d in x.type.tensor_type.shape.dim]
            print(f"  {tag} {x.name:50s} {shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--out", default=None, help="output dir (default: llm_server/onnx)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(__file__).parent.parent / "onnx"
    tmp_dir = out_dir / "_export_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Export with optimum (dynamic seq + past axes) ─────────────────────
    print(f"Exporting {args.model} → {tmp_dir}")
    from optimum.exporters.onnx import main_export
    main_export(
        model_name_or_path=args.model,
        output=tmp_dir,
        task="text-generation-with-past",
        dtype="fp16",
        device="cpu",
    )

    # optimum may emit model.onnx (merged) or decoder_model_merged.onnx
    candidates = [tmp_dir / "model.onnx", tmp_dir / "decoder_model_merged.onnx"]
    src = next((c for c in candidates if c.exists()), None)
    if src is None:
        raise SystemExit(f"No merged ONNX found in {tmp_dir}: {list(tmp_dir.glob('*.onnx'))}")

    print(f"Loading {src.name} with external data...")
    model = onnx.load(str(src), load_external_data=True)
    graph = model.graph
    print_io(model, "EXPORTED")

    # ── 2. Strip lm_head ─────────────────────────────────────────────────────
    logits_name = next(o.name for o in graph.output if "logits" in o.name.lower())
    producer = find_producer(graph, logits_name)
    print(f"\nlogits producer: {producer.op_type}  inputs={list(producer.input)}")

    lm_matmul = None
    if producer.op_type == "MatMul":
        lm_matmul = producer
    elif producer.op_type in ("Add", "Gemm", "Cast"):
        for inp in producer.input:
            mm = find_producer(graph, inp)
            if mm and mm.op_type == "MatMul":
                lm_matmul = mm
                break
    if lm_matmul is None:
        raise SystemExit(f"Cannot locate lm_head MatMul before '{logits_name}'")

    hidden_name = lm_matmul.input[0]
    weight_name = lm_matmul.input[1]
    print(f"lm_head: hidden='{hidden_name}'  weight='{weight_name}'")

    # Extract lm_head weight → raw fp16 bin [HIDDEN, VOCAB]
    w = None
    for init in graph.initializer:
        if init.name == weight_name:
            w = numpy_helper.to_array(init)
            break
    if w is None:
        raise SystemExit(f"lm_head weight '{weight_name}' not in initializers")
    w = np.asarray(w)
    if w.shape == (VOCAB, HIDDEN):
        w = np.ascontiguousarray(w.T)
    elif w.shape != (HIDDEN, VOCAB):
        raise SystemExit(f"Unexpected lm_head shape {w.shape}")
    w.astype(np.float16).tofile(str(out_dir / "lm_head_weight.bin"))
    print(f"Saved lm_head_weight.bin  [{HIDDEN},{VOCAB}] fp16 "
          f"({w.nbytes / 1e6:.0f} MB)")

    # Rewire outputs: logits → last_hidden_state
    hidden_vi = None
    for vi in list(graph.value_info) + list(graph.input):
        if vi.name == hidden_name:
            hidden_vi = vi
            break
    if hidden_vi is None:
        hidden_vi = helper.make_tensor_value_info(
            hidden_name, TensorProto.FLOAT16, ["batch", "sequence_length", HIDDEN])

    new_outputs = [hidden_vi if o.name == logits_name else o for o in graph.output]
    del graph.output[:]
    graph.output.extend(new_outputs)

    # Drop lm_head node chain (MatMul + producer if distinct)
    drop = {id(lm_matmul), id(producer)}
    kept = [n for n in graph.node if id(n) not in drop]
    removed = len(graph.node) - len(kept)
    del graph.node[:]
    graph.node.extend(kept)
    print(f"Removed {removed} lm_head node(s)")
    print_io(model, "STRIPPED")

    # ── 3. Save backbone + tokenizer, clean up ───────────────────────────────
    out_onnx = out_dir / "model_backbone.onnx"
    print(f"\nSaving {out_onnx} ...")
    onnx.save_model(
        model, str(out_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model_backbone.onnx_data",
        size_threshold=1024,
    )

    for name in ("tokenizer.json", "config.json"):
        if (tmp_dir / name).exists():
            shutil.copy(tmp_dir / name, out_dir / name)

    shutil.rmtree(tmp_dir)

    print("\n── Done ─────────────────────────────────────")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:30s} {f.stat().st_size / 1e6:8.1f} MB")
    print("\nNext: bash tools/build_engine.sh   (or python tools/build_engine.py)")


if __name__ == "__main__":
    main()
