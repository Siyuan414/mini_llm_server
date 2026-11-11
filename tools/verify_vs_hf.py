"""
verify_vs_hf.py — S1.6 correctness gate.

Runs the C++ runtime in GREEDY mode and HuggingFace transformers with
do_sample=False on the same prompts, then compares the generated text.

Greedy decoding is deterministic, so a correct engine + KV cache + lm_head +
tokenizer must reproduce HF exactly (modulo fp16 rounding, which can flip a
token when the top-2 logits are within ~1e-3 — reported as a near-tie, not a
hard failure).

Run from llm_server/:
    python tools/verify_vs_hf.py \
        --model /path/to/Llama-3.2-1B-Instruct \
        [--tokens 48] [--runtime ./build/runtime]

Exit code 0 = all prompts matched.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROMPTS = [
    "The key insight about transformers is",
    "Once upon a time",
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, humans first",
]


def run_cpp(runtime, engine, lm_head, tokenizer, prompt, n_tokens):
    out = subprocess.run(
        [runtime,
         "--engine", engine, "--lm-head", lm_head, "--tokenizer", tokenizer,
         "--prompt", prompt, "--max-new-tokens", str(n_tokens),
         "--temperature", "0"],
        capture_output=True, text=True, check=True).stdout
    m = re.search(r'Full output: "(.*)"\s*$', out, re.S)
    if not m:
        raise SystemExit(f"Could not parse runtime output:\n{out}")
    return m.group(1)


def run_hf(model_dir, prompt, n_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16).cuda().eval()

    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir or hub id")
    ap.add_argument("--runtime", default="./build/runtime")
    ap.add_argument("--engine", default="engine/llama1b_fp16.trt")
    ap.add_argument("--lm-head", default="onnx/lm_head_weight.bin")
    ap.add_argument("--tokenizer", default="onnx/tokenizer.json")
    ap.add_argument("--tokens", type=int, default=48)
    args = ap.parse_args()

    print(f"Comparing C++ runtime vs HF transformers — {args.tokens} greedy tokens\n")

    passed = 0
    for prompt in PROMPTS:
        cpp = run_cpp(args.runtime, args.engine, args.lm_head, args.tokenizer,
                      prompt, args.tokens)
        hf = run_hf(args.model, prompt, args.tokens)

        # The runtime prints escaped newlines inside the quoted output
        cpp_n = cpp.replace("\\n", "\n").strip()
        hf_n = hf.strip()

        ok = cpp_n == hf_n
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  \"{prompt}\"")
        if not ok:
            d = first_divergence(cpp_n, hf_n)
            print(f"      diverges at char {d}")
            print(f"      cpp: ...{cpp_n[max(0, d-40):d+40]!r}")
            print(f"      hf : ...{hf_n[max(0, d-40):d+40]!r}")

    print(f"\n{passed}/{len(PROMPTS)} prompts matched exactly.")
    if passed < len(PROMPTS):
        print("Note: fp16 near-ties can legitimately flip one token and then "
              "diverge. Check whether the divergence point is a plausible tie "
              "before treating it as a bug.")
    return 0 if passed == len(PROMPTS) else 1


if __name__ == "__main__":
    sys.exit(main())
