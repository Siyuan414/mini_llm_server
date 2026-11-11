"""
dump_reference.py — HuggingFace ground truth for the custom decode path (S2.7).

Runs the model on a fixed token sequence and saves the hidden state after EVERY
layer, plus the final logits, as raw fp32.

Why per-layer: wiring 16 layers and comparing only the final logits tells you
nothing about where it broke. Compare layer 0 first; when that matches, layer 1
can only fail for its own reasons.

Output → reference/
    tokens.txt              the input ids, one per line
    embed.bin               [seq, hidden]   after embedding, before layer 0
    layer00.bin ...         [seq, hidden]   after each decoder layer
    final_norm.bin          [seq, hidden]   after model.norm
    logits.bin              [seq, vocab]

Every file is fp32 row-major. The C++ side compares only the LAST position,
since that is what a decode step produces.

Run from llm_server/:
    python tools/dump_reference.py --model <hf_model_dir> \
        [--prompt "The capital of France is"] [--out reference]
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", default="reference")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16).cuda().eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    (out / "tokens.txt").write_text(
        "\n".join(str(int(t)) for t in ids[0]) + "\n")
    print(f"prompt : {args.prompt!r}")
    print(f"tokens : {ids.shape[1]}  {[int(t) for t in ids[0]]}")

    # output_hidden_states gives us [embedding, layer0, layer1, ..., layer15],
    # where the LAST entry has already had model.norm applied.
    with torch.no_grad():
        o = model(ids, output_hidden_states=True)

    hs = o.hidden_states          # tuple of len num_layers+1
    n_layers = model.config.num_hidden_layers

    def save(name, t):
        a = t[0].detach().float().cpu().numpy()      # [seq, ...]
        a.astype(np.float32).tofile(out / f"{name}.bin")
        print(f"  {name:14s} {list(a.shape)}")

    print("\nsaving:")
    save("embed", hs[0])
    for i in range(1, n_layers):
        save(f"layer{i-1:02d}", hs[i])

    # hs[-1] is post-model.norm; re-derive the pre-norm last layer output so the
    # C++ side can compare BOTH (its `x` before the final norm, and `s.h` after).
    save(f"layer{n_layers-1:02d}", hs[-1])   # NOTE: normalized, see below
    save("final_norm", hs[-1])
    save("logits", o.logits)

    print(f"\nNOTE: hidden_states[-1] has model.norm applied, so layer"
          f"{n_layers-1:02d}.bin == final_norm.bin. Compare your pre-norm x "
          f"against layer{n_layers-2:02d}.bin and your post-norm against "
          f"final_norm.bin.")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
