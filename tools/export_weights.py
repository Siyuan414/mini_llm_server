"""
export_weights.py — dump a LLaMA checkpoint to raw fp16 binaries (S2.7).

The custom decode path needs plain weight matrices, not an ONNX graph. This
writes one .bin per tensor plus a manifest, so C++ can `fread` them with no
parsing.

  weights/
    manifest.json              shapes, dtypes, file names, model dims
    embed_tokens.bin           [vocab, hidden]
    model.norm.bin             [hidden]
    lm_head.bin                [hidden, vocab]   ← TRANSPOSED, see below
    layer00.input_layernorm.bin        [hidden]
    layer00.q_proj.bin                 [hidden, n_heads*head_dim]     transposed
    layer00.k_proj.bin                 [hidden, n_kv_heads*head_dim]  transposed
    layer00.v_proj.bin                 [hidden, n_kv_heads*head_dim]  transposed
    layer00.o_proj.bin                 [n_heads*head_dim, hidden]     transposed
    layer00.post_attention_layernorm.bin [hidden]
    layer00.gate_proj.bin              [hidden, intermediate]  transposed
    layer00.up_proj.bin                [hidden, intermediate]  transposed
    layer00.down_proj.bin              [intermediate, hidden]  transposed
    ... layer01 ... layer15

── Why the projections are transposed ───────────────────────────────────────
PyTorch stores nn.Linear weight as [out_features, in_features] and computes
y = x @ W.T. cuBLAS is column-major, so a row-major [in, out] matrix is read
directly as a column-major [out, in] one — which is exactly the operand
cublasHgemm wants for y[out] = W[out,in] @ x[in], with no transpose flag and no
runtime copy. Doing the transpose here, once, keeps the hot path free of it.

Run from llm_server/:
    python tools/export_weights.py --model <hf_model_dir> [--out weights]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def save(t: torch.Tensor, path: Path, transpose: bool) -> list:
    """Write fp16 row-major. Returns the on-disk shape."""
    a = t.detach().to(torch.float16).cpu()
    if transpose:
        a = a.t().contiguous()          # [out, in] -> [in, out]
    arr = a.numpy()
    arr.tofile(path)
    return list(arr.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir or hub id")
    ap.add_argument("--out", default="weights")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(args.model)
    print(f"Loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    sd = model.state_dict()

    head_dim = getattr(cfg, "head_dim", None) or \
               cfg.hidden_size // cfg.num_attention_heads

    # rope_theta moved into a `rope_parameters` dict in newer transformers, so
    # getattr(cfg, "rope_theta") silently returns the default — which wrote
    # 10000.0 into every manifest while the real value is 500000.0. Harmless so
    # far only because model_config.h reads config.json instead of the manifest.
    cfgd = cfg.to_dict()
    rope_theta = cfgd.get("rope_theta")
    if rope_theta is None:
        rope_theta = (cfgd.get("rope_parameters") or {}).get("rope_theta", 10000.0)

    # LLaMA-3.1/3.2 apply frequency-dependent RoPE scaling on top of rope_theta.
    # The kernel implements plain RoPE, which is why this is recorded but not
    # consumed — see the note in benchmarks/ about where that approximation
    # starts to matter.
    rope_scaling = cfgd.get("rope_scaling") or cfgd.get("rope_parameters")

    manifest = {
        "num_layers":       cfg.num_hidden_layers,
        "hidden_size":      cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim":         head_dim,
        "vocab_size":       cfg.vocab_size,
        "rms_norm_eps":     cfg.rms_norm_eps,
        "rope_theta":       rope_theta,
        "rope_scaling":     rope_scaling,
        "max_position_embeddings": cfg.max_position_embeddings,
        "tensors": {},
    }

    def emit(name, tensor, transpose):
        fn = f"{name}.bin"
        shape = save(tensor, out / fn, transpose)
        manifest["tensors"][name] = {"file": fn, "shape": shape,
                                     "transposed": transpose}
        print(f"  {name:44s} {str(shape):>20s}")

    print("\nGlobal:")
    emit("embed_tokens", sd["model.embed_tokens.weight"], False)
    emit("model.norm",   sd["model.norm.weight"], False)
    # LLaMA-3.2-1B ties lm_head to the embedding table.
    lm = sd.get("lm_head.weight", sd["model.embed_tokens.weight"])
    emit("lm_head", lm, True)          # -> [hidden, vocab]

    print("\nLayers:")
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        tag = f"layer{i:02d}"
        emit(f"{tag}.input_layernorm", sd[p + "input_layernorm.weight"], False)
        emit(f"{tag}.q_proj", sd[p + "self_attn.q_proj.weight"], True)
        emit(f"{tag}.k_proj", sd[p + "self_attn.k_proj.weight"], True)
        emit(f"{tag}.v_proj", sd[p + "self_attn.v_proj.weight"], True)
        emit(f"{tag}.o_proj", sd[p + "self_attn.o_proj.weight"], True)
        emit(f"{tag}.post_attention_layernorm",
             sd[p + "post_attention_layernorm.weight"], False)
        emit(f"{tag}.gate_proj", sd[p + "mlp.gate_proj.weight"], True)
        emit(f"{tag}.up_proj",   sd[p + "mlp.up_proj.weight"], True)
        emit(f"{tag}.down_proj", sd[p + "mlp.down_proj.weight"], True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(f.stat().st_size for f in out.glob("*.bin"))
    print(f"\nWrote {len(manifest['tensors'])} tensors, "
          f"{total/1e9:.2f} GB → {out}/")
    print("Manifest: " + str(out / "manifest.json"))


if __name__ == "__main__":
    main()
