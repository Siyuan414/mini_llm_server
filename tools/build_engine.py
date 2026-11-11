"""
build_engine.py — S1.1: Build the TRT engine with TWO optimization profiles.

Why Python API instead of trtexec: trtexec's multi-profile syntax is awkward and
version-dependent; the builder API states exactly what we mean.

Profiles (batch=1 for Stage 1; Stage 2 rebuilds with a batch dim):
  Profile 0 — PREFILL : input_ids [1, 1..2048], past_len = 0
  Profile 1 — DECODE  : input_ids [1, 1],       past_len = 0..2047

The C++ runtime picks the profile per phase:
  context->setOptimizationProfileAsync(0, stream)  // prefill enqueue
  context->setOptimizationProfileAsync(1, stream)  // decode loop

Run from llm_server/ (on the 5070 Ti box, TRT 10.x python bindings):
    python tools/build_engine.py [--max-seq 2048] [--onnx onnx/model_backbone.onnx]

Output: engine/llama1b_fp16.trt
"""

import argparse
import time
from pathlib import Path

import tensorrt as trt

NUM_KV_HEADS = 8
HEAD_DIM = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--workspace-gb", type=float, default=8.0)
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    onnx_path = Path(args.onnx) if args.onnx else root / "onnx" / "model_backbone.onnx"
    out_path = Path(args.out) if args.out else root / "engine" / "llama1b_fp16.trt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_seq = args.max_seq

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # Strongly typed: precision comes from the ONNX itself (already fp16).
    # Newer TRT removed BuilderFlag.FP16 weak-typing in favor of this.
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing {onnx_path} ...")
    # parse_from_file resolves the external .onnx_data next to the model
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(f"  parser error: {parser.get_error(i)}")
        raise SystemExit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))

    inputs = {network.get_input(i).name: network.get_input(i)
              for i in range(network.num_inputs)}
    print(f"Network inputs ({len(inputs)}):")
    for name, t in inputs.items():
        print(f"  {name:45s} {t.shape}")

    def kv_shape(batch,past):
        return (batch, NUM_KV_HEADS, past, HEAD_DIM)

    def make_profile(seq_rng, past_rng):
        """seq_rng/past_rng: (min, opt, max)"""
        prof = builder.create_optimization_profile()
        batch_rng = (1,4,32)
        for name in inputs:
            if name == "input_ids" or name == "position_ids":
                prof.set_shape(name, (batch_rng[0], seq_rng[0]), (batch_rng[1], seq_rng[1]), (batch_rng[2], seq_rng[2]))
            elif name == "attention_mask":
                # mask covers past + current: total = past_len + seq_len
                tot = tuple(p + s for p, s in zip(past_rng, seq_rng))
                prof.set_shape(name, (batch_rng[0], tot[0]), (batch_rng[1], tot[1]), (batch_rng[2], tot[2]))
            elif name.startswith("past_key_values"):
                prof.set_shape(name, kv_shape(batch_rng[0], past_rng[0]),
                               kv_shape(batch_rng[1], past_rng[1]), kv_shape(batch_rng[2], past_rng[2]))
            else:
                raise SystemExit(f"Unhandled input: {name}")
        config.add_optimization_profile(prof)
        return prof

    # Profile 0 — PREFILL: seq 1..max_seq, past fixed at 0
    make_profile(seq_rng=(1, 512, max_seq), past_rng=(0, 0, 0))
    # Profile 1 — DECODE: seq fixed at 1, past 0..max_seq-1
    make_profile(seq_rng=(1, 1, 1), past_rng=(0, 512, max_seq - 1))
    print(f"\nProfiles: 0=prefill(seq 1..{max_seq}, past=0)  "
          f"1=decode(seq=1, past 0..{max_seq - 1})")

    print("Building engine (few minutes on 5070 Ti)...")
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit("Engine build FAILED")
    out_path.write_bytes(blob)
    print(f"\nEngine saved: {out_path}  "
          f"({out_path.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")
    print("\nSmoke test:")
    print("  python tools/smoke_test.py   (verifies both profiles execute)")


if __name__ == "__main__":
    main()
