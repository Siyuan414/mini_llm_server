"""
verify_tokenizer.py — S1.6: compare the C++ BPE tokenizer against HF `tokenizers`.

Why this matters: a pre-tokenizer mismatch feeds the model a *different prompt*,
which shows up as large logit differences that look like an engine bug but
aren't. LLaMA-3's real pre-tokenizer is a Unicode regex that, among other
things, splits digit runs into groups of at most 3 and has specific
contraction/punctuation rules. The hand-rolled scanner in tokenizer.h
approximates it.

Run from llm_server/:
    python tools/verify_tokenizer.py [--model <hf_model_dir>]
"""

import argparse
import re
import subprocess
import sys

PROMPTS = [
    "The key insight about transformers is",
    "Once upon a time",
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, humans first",
    "Cost: $1,234.56 (approx.)",
    "it's a test — don't fail",
    "x = [1, 2, 3]; y = x[0]",
]


def cpp_tokens(runtime, engine, lm_head, tokenizer, prompt):
    out = subprocess.run(
        [runtime, "--engine", engine, "--lm-head", lm_head,
         "--tokenizer", tokenizer, "--prompt", prompt,
         "--max-new-tokens", "1", "--warmup", "0", "--quiet",
         "--print-tokens"],
        capture_output=True, text=True, check=True).stdout
    m = re.search(r"^TOKENS:(.*)$", out, re.M)
    if not m:
        raise SystemExit(f"no TOKENS line:\n{out}")
    return [int(x) for x in m.group(1).split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="HF model dir (defaults to onnx/tokenizer.json)")
    ap.add_argument("--runtime", default="./build/runtime")
    ap.add_argument("--engine", default="engine/llama1b_fp16.trt")
    ap.add_argument("--lm-head", default="onnx/lm_head_weight.bin")
    ap.add_argument("--tokenizer", default="onnx/tokenizer.json")
    args = ap.parse_args()

    if args.model:
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(args.model)
        encode = lambda s: hf(s).input_ids
        decode = lambda ids: [hf.decode([i]) for i in ids]
    else:
        from tokenizers import Tokenizer
        hf = Tokenizer.from_file(args.tokenizer)
        # tokenizer.json carries a post-processor that already prepends BOS —
        # do NOT add it again here.
        encode = lambda s: hf.encode(s).ids
        decode = lambda ids: [hf.decode([i]) for i in ids]

    passed = 0
    for p in PROMPTS:
        a = cpp_tokens(args.runtime, args.engine, args.lm_head,
                       args.tokenizer, p)
        b = encode(p)
        ok = a == b
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {p!r}")
        if not ok:
            print(f"      cpp ({len(a)}): {a}")
            print(f"      hf  ({len(b)}): {b}")
            print(f"      cpp: {decode(a)}")
            print(f"      hf : {decode(b)}")
    print(f"\n{passed}/{len(PROMPTS)} prompts tokenized identically.")
    return 0 if passed == len(PROMPTS) else 1


if __name__ == "__main__":
    sys.exit(main())
