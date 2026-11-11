"""
verify_logits.py — S1.6 correctness gate (logit-level).

Comparing generated *text* against HuggingFace is a bad gate for an fp16 engine:
one near-tie flips a token and the sequences diverge into two equally-valid
continuations forever. That measures chaos, not correctness.

This compares the post-prefill logit vector instead — one forward pass, no
accumulated drift — and reports:

  top-1 agree     do both pick the same next token?
  top-5 overlap   how much of the candidate set is shared
  cosine sim      overall shape of the distribution
  max |diff|      worst-case elementwise error
  KL(hf || trt)   distributional distance after softmax
  top-1 margin    gap between HF's #1 and #2 logit; a small margin means a
                  disagreement is a coin-flip, not a bug

Run from llm_server/:
    python tools/verify_logits.py --model <hf_model_dir>
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROMPTS = [
    "The key insight about transformers is",
    "Once upon a time",
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, humans first",
]


def trt_logits(runtime, engine, lm_head, tokenizer, prompt, tmp):
    subprocess.run(
        [runtime, "--engine", engine, "--lm-head", lm_head,
         "--tokenizer", tokenizer, "--prompt", prompt,
         "--max-new-tokens", "1", "--temperature", "0",
         "--quiet", "--warmup", "0", "--dump-logits", str(tmp)],
        capture_output=True, text=True, check=True)
    return np.fromfile(tmp, dtype=np.float32)


def hf_logits(model, tok, prompt):
    import torch
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model(ids).logits[0, -1, :]
    return out.float().cpu().numpy()


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runtime", default="./build/runtime")
    ap.add_argument("--engine", default="engine/llama1b_fp16.trt")
    ap.add_argument("--lm-head", default="onnx/lm_head_weight.bin")
    ap.add_argument("--tokenizer", default="onnx/tokenizer.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16).cuda().eval()

    print(f"{'prompt':<40} {'top1':>5} {'top5':>5} {'cos':>7} "
          f"{'maxdiff':>8} {'KL':>8} {'margin':>7}")
    print("-" * 84)

    agree = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "logits.bin"
        for p in PROMPTS:
            a = trt_logits(args.runtime, args.engine, args.lm_head,
                           args.tokenizer, p, tmp)
            b = hf_logits(model, tok, p)
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]

            ta, tb = int(a.argmax()), int(b.argmax())
            top5a, top5b = set(a.argsort()[-5:]), set(b.argsort()[-5:])
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            maxd = float(np.abs(a - b).max())
            pa, pb = softmax(a), softmax(b)
            kl = float((pb * np.log((pb + 1e-12) / (pa + 1e-12))).sum())
            srt = np.sort(b)[::-1]
            margin = float(srt[0] - srt[1])

            ok = ta == tb
            agree += ok
            print(f"{p[:38]:<40} {'yes' if ok else 'NO':>5} "
                  f"{len(top5a & top5b):>5} {cos:>7.4f} {maxd:>8.3f} "
                  f"{kl:>8.4f} {margin:>7.3f}")
            if not ok:
                print(f"{'':<40} trt={tok.decode([ta])!r} hf={tok.decode([tb])!r}")

    print("-" * 84)
    print(f"top-1 agreement: {agree}/{len(PROMPTS)}")
    print("\nHealthy fp16 engine: cosine > 0.999, KL < 0.01, top-5 overlap 5/5.")
    print("A top-1 disagreement with a small HF margin (< ~0.05) is a numerical "
          "tie, not a bug.")
    return 0 if agree >= len(PROMPTS) - 1 else 1


if __name__ == "__main__":
    sys.exit(main())
