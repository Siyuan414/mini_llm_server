"""
bench_vs_vllm.py — same workload, same client, two servers (Stage 4).

Both llm_server and vLLM expose /v1/completions with SSE streaming, so this
script is server-agnostic: point it at a URL and it measures what a real client
would experience.

Metrics (per request, then aggregated):
    TTFT   time to the FIRST streamed token — what a user waits before text moves
    TPOT   mean inter-token time after the first — how fast text then flows
    E2E    submit to last token
    throughput = total generated tokens / wall time of the whole run

Concurrency is real: N requests are issued simultaneously from a thread pool, so
the server's batching and scheduling are what is being measured.

Usage:
    # 1. start llm_server
    ./build/serve --weights weights --tokenizer onnx/tokenizer.json \
                  --config onnx/config.json --kv-budget-mb 1024 --max-batch 64

    python tools/bench_vs_vllm.py --url http://localhost:8080/v1/completions \
        --name llm_server --concurrency 32 --requests 128 --max-tokens 64 \
        --out benchmarks/vllm_llm_server.json

    # 2. stop it, start vLLM on the same GPU
    vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8000 --dtype float16

    python tools/bench_vs_vllm.py --url http://localhost:8000/v1/completions \
        --name vllm --model meta-llama/Llama-3.2-1B-Instruct \
        --concurrency 32 --requests 128 --max-tokens 64 \
        --out benchmarks/vllm_vllm.json

    # 3. compare
    python tools/bench_vs_vllm.py --compare benchmarks/vllm_*.json
"""

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

WORDS = ("transformer attention gradient tensor kernel inference latency "
         "throughput cache memory scheduler batching").split()


def make_prompts(n, target_tokens=0, seed=0):
    """n prompts. If target_tokens > 0, pad each to roughly that many tokens.

    Prompt length is the axis that exposes chunked prefill: a server that
    prefills one token per step pays target_tokens decode steps before it can
    emit anything, while a server that packs the prompt into one forward pass
    pays ~one. Short prompts hide that difference entirely.
    """
    rng = random.Random(seed)
    if target_tokens <= 0:
        return ["Explain how " + " ".join(rng.choices(WORDS, k=3)) + " works:"
                for _ in range(n)]

    # Every word here is 1-2 BPE pieces, so ~0.75 words per token is close.
    # --report-prompt-tokens measures the real count rather than trusting this.
    n_words = max(1, int(target_tokens * 0.75))
    return ["Summarize the following notes:\n"
            + " ".join(rng.choices(WORDS, k=n_words))
            + "\nSummary:" for _ in range(n)]


def count_tokens(prompt, model_dir):
    """Exact prompt length, when transformers + the model dir are available."""
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(model_dir)
        return len(tk.encode(prompt))
    except Exception:
        return None


def one_request(url, model, prompt, max_tokens, timeout):
    """Returns (ttft_s, e2e_s, n_tokens) or None on failure."""
    body = {"prompt": prompt, "max_tokens": max_tokens,
            "stream": True, "temperature": 0}
    if model:
        body["model"] = model

    t0 = time.perf_counter()
    ttft = None
    n = 0
    try:
        with requests.post(url, json=body, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # OpenAI shape; llm_server's /generate shape is {"token": ...}
                text = ""
                if "choices" in obj and obj["choices"]:
                    text = obj["choices"][0].get("text", "")
                    if obj["choices"][0].get("finish_reason"):
                        continue          # final chunk carries no text
                elif "token" in obj:
                    text = obj["token"]
                elif obj.get("done"):
                    break
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n += 1
    except Exception as e:
        print(f"   request failed: {e}", file=sys.stderr)
        return None

    if ttft is None:
        return None
    return ttft, time.perf_counter() - t0, n


def run(args):
    prompts = make_prompts(args.requests, args.prompt_tokens)
    n_prompt = count_tokens(prompts[0], args.tokenizer_dir) \
        if args.tokenizer_dir else None
    print(f"{args.name}: {args.requests} requests, concurrency "
          f"{args.concurrency}, {args.max_tokens} tokens each"
          + (f", prompt ~{n_prompt} tok" if n_prompt else
             f", prompt target {args.prompt_tokens} tok"
             if args.prompt_tokens else ""))

    # Warm the server (first request pays any lazy init / CUDA graph capture).
    one_request(args.url, args.model, "Hello", 8, args.timeout)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(
            lambda p: one_request(args.url, args.model, p,
                                  args.max_tokens, args.timeout),
            prompts))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r]
    if not ok:
        print("all requests failed", file=sys.stderr)
        return 1

    ttfts = sorted(r[0] * 1000 for r in ok)
    tokens = sum(r[2] for r in ok)
    # TPOT: generation time after the first token, per token.
    tpots = sorted(((e2e - ttft) / (n - 1) * 1000)
                   for ttft, e2e, n in ok if n > 1)

    def p(v, q):
        return v[min(len(v) - 1, int(len(v) * q))] if v else 0.0

    out = {
        "name": args.name,
        "url": args.url,
        "requests": len(ok),
        "failed": len(results) - len(ok),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "prompt_tokens": n_prompt or args.prompt_tokens,
        "tokens": tokens,
        "wall_s": wall,
        "throughput_tok_s": tokens / wall,
        "ttft_p50_ms": p(ttfts, 0.50),
        "ttft_p95_ms": p(ttfts, 0.95),
        "tpot_p50_ms": p(tpots, 0.50),
        "tpot_mean_ms": statistics.mean(tpots) if tpots else 0.0,
    }

    print(f"   throughput {out['throughput_tok_s']:8.1f} tok/s")
    print(f"   TTFT  p50  {out['ttft_p50_ms']:8.1f} ms   "
          f"p95 {out['ttft_p95_ms']:.1f} ms")
    print(f"   TPOT  p50  {out['tpot_p50_ms']:8.2f} ms")
    if out["failed"]:
        print(f"   {out['failed']} FAILED")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"   wrote {args.out}")
    return 0


def compare(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    runs.sort(key=lambda r: -r["throughput_tok_s"])
    w = max(len(r["name"]) for r in runs)
    print(f"\n{'server':<{w}} {'prompt':>7} {'tok/s':>10} {'TTFT p50':>10} "
          f"{'TTFT p95':>10} {'TPOT p50':>10}")
    print("-" * (w + 52))
    for r in runs:
        print(f"{r['name']:<{w}} {r.get('prompt_tokens', 0):7d} "
              f"{r['throughput_tok_s']:10.1f} "
              f"{r['ttft_p50_ms']:9.1f}m {r['ttft_p95_ms']:9.1f}m "
              f"{r['tpot_p50_ms']:9.2f}m")
    lens = {r.get("prompt_tokens", 0) for r in runs}
    if len(lens) > 1:
        print("\nWARNING: prompt lengths differ across runs — not comparable.")
    if len(runs) == 2:
        a, b = runs
        print(f"\n{a['name']} is {a['throughput_tok_s']/b['throughput_tok_s']:.2f}x "
              f"{b['name']} on throughput, "
              f"{b['ttft_p50_ms']/a['ttft_p50_ms']:.2f}x on TTFT p50")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080/v1/completions")
    ap.add_argument("--name", default="llm_server")
    ap.add_argument("--model", default=None, help="required by vLLM")
    ap.add_argument("--requests", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="pad prompts to ~N tokens (0 = short prompts). "
                         "This is the axis that exposes chunked prefill.")
    ap.add_argument("--tokenizer-dir", default=None,
                    help="HF model dir, to report exact prompt length")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
