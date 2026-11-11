"""
bench.py — S1.6 benchmark sweep → benchmarks/STAGE1.md

Sweeps:
  A. Prefill scaling   — prompt 8..2048 tokens, measures prefill tok/s and TTFT
  B. Decode vs context — decode throughput at increasing KV depth
  C. Sampling overhead — greedy vs top-k/top-p at a fixed context

Every run does an untimed warmup inside the runtime (--warmup), so the numbers
exclude one-time TRT kernel-selection and profile-switch costs.

Run from llm_server/:
    python tools/bench.py [--out benchmarks/STAGE1.md]
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

BASE = ["--engine", "engine/llama1b_fp16.trt",
        "--lm-head", "onnx/lm_head_weight.bin",
        "--tokenizer", "onnx/tokenizer.json",
        "--quiet", "--json", "--warmup", "2"]


def run(runtime, extra):
    out = subprocess.run([runtime] + BASE + extra,
                         capture_output=True, text=True, check=True).stdout
    for line in reversed(out.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit(f"No JSON in output:\n{out}")


def gpu_name():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown GPU"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default="./build/runtime")
    # STAGE1.md is the curated report; this script writes raw regenerable tables.
    ap.add_argument("--out", default="benchmarks/perf_raw.md")
    args = ap.parse_args()

    rt = args.runtime
    print("A. prefill scaling")
    prefill = []
    # 2048 is the full context — leave headroom for the decode steps
    for n in [8, 32, 128, 512, 1024, 1900]:
        r = run(rt, ["--prompt-tokens", str(n), "--max-new-tokens", "8"])
        prefill.append(r)
        print(f"   {n:5d} tok  {r['prefill_ms']:8.2f} ms  "
              f"{r['prefill_tok_s']:9.0f} tok/s  TTFT {r['ttft_ms']:.1f} ms")

    print("B. decode vs context depth")
    decode = []
    for ctx in [8, 128, 512, 1024]:
        r = run(rt, ["--prompt-tokens", str(ctx), "--max-new-tokens", "64"])
        decode.append((ctx, r))
        print(f"   ctx {ctx:5d}  p50 {r['decode_ms_p50']:6.2f} ms  "
              f"{r['decode_tok_s']:6.1f} tok/s")

    print("C. sampling overhead (ctx 512)")
    modes = [("greedy", ["--temperature", "0"]),
             ("top-k 50 / top-p 0.95",
              ["--temperature", "0.8", "--top-k", "50", "--top-p", "0.95"])]
    sampling = []
    for label, flags in modes:
        r = run(rt, ["--prompt-tokens", "512", "--max-new-tokens", "64"] + flags)
        sampling.append((label, r))
        print(f"   {label:22s} p50 {r['decode_ms_p50']:6.2f} ms  "
              f"{r['decode_tok_s']:6.1f} tok/s")

    # ── Write report ─────────────────────────────────────────────────────────
    g = gpu_name()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("# Stage 1 — Single-request engine benchmarks\n")
    L.append(f"**Model** LLaMA-3.2-1B-Instruct FP16 (TensorRT, 2 optimization profiles)  ")
    L.append(f"**GPU** {g}  ")
    L.append(f"**Host** {platform.platform()}  ")
    L.append(f"**Date** {date.today().isoformat()}\n")
    L.append("All runs include 2 untimed warmup cycles; decode figures are p50 "
             "over 64 steps.\n")

    L.append("## A. Prefill scaling\n")
    L.append("| Prompt tokens | Prefill (ms) | Prefill (tok/s) | TTFT (ms) |")
    L.append("|---:|---:|---:|---:|")
    for r in prefill:
        L.append(f"| {r['prompt_tokens']} | {r['prefill_ms']:.2f} | "
                 f"{r['prefill_tok_s']:.0f} | {r['ttft_ms']:.1f} |")
    L.append("")

    L.append("## B. Decode throughput vs context depth\n")
    L.append("| Context (tokens) | p50 (ms/tok) | p95 (ms/tok) | Decode (tok/s) |")
    L.append("|---:|---:|---:|---:|")
    for ctx, r in decode:
        L.append(f"| {ctx} | {r['decode_ms_p50']:.2f} | "
                 f"{r['decode_ms_p95']:.2f} | {r['decode_tok_s']:.1f} |")
    L.append("")

    L.append("## C. Sampling overhead (context 512)\n")
    L.append("| Mode | p50 (ms/tok) | Decode (tok/s) |")
    L.append("|---|---:|---:|")
    for label, r in sampling:
        L.append(f"| {label} | {r['decode_ms_p50']:.2f} | {r['decode_tok_s']:.1f} |")
    if len(sampling) == 2:
        d = sampling[1][1]['decode_ms_p50'] - sampling[0][1]['decode_ms_p50']
        pct = 100 * d / sampling[0][1]['decode_ms_p50']
        L.append(f"\nSampling costs **{d:.2f} ms/token ({pct:.0f}%)** — a full "
                 f"128k-entry radix sort per step. A radix top-k would avoid "
                 f"sorting the tail.\n")

    L.append("## Notes\n")
    L.append("- **Prefill vs decode profiles.** One engine, two TRT optimization "
             "profiles (prefill: seq 1..2048/past=0; decode: seq=1/past 0..2047). "
             "Switching costs a `setOptimizationProfileAsync` once per phase.")
    L.append("- **KV cache is ping-pong, not spliced.** TRT reads "
             "`past_key_values` as a contiguous `[1, H, past, D]` tensor, so a "
             "fixed-capacity buffer strided by `max_seq` silently misreads every "
             "head after head 0. The `present` output is already the exact layout "
             "the next step needs, so buffers alternate instead of being copied — "
             "correct *and* removes 32 `cudaMemcpy2DAsync` calls per step.")
    L.append("- **Decode is memory-bound.** Throughput falls with context depth "
             "because attention rescans a longer KV cache each step.")
    out.write_text("\n".join(L) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    sys.exit(main())
