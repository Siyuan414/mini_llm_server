# llm_server vs vLLM 0.11.0

**Model** LLaMA-3.2-1B-Instruct, FP16 · **GPU** RTX 5070 Ti · WSL2, CUDA 13
**Client** identical for both — `tools/bench_vs_vllm.py` over `/v1/completions` SSE
Both servers speak the OpenAI API, so one client measures both. Only one server
runs at a time. vLLM: `--gpu-memory-utilization 0.5 --max-model-len 4096`.
llm_server: `--kv-budget-mb 1024 --max-batch 64`.

Two operating points, because **one of them alone is misleading**.

## Short prompts (~10 tokens) — 128 requests, concurrency 32, 64 out

| | throughput | TTFT p50 | TTFT p95 | TPOT p50 |
|---|---:|---:|---:|---:|
| **llm_server** | **5,755 tok/s** | 61.3 ms | 66.5 ms | **4.57 ms** |
| vLLM 0.11.0 | 5,633 tok/s | **21.6 ms** | **31.0 ms** | 5.49 ms |
| ratio | **1.02×** | 0.35× | 0.47× | **1.20×** |

## Realistic prompts (256 tokens) — 64 requests, concurrency 32, 64 out

| | throughput | TTFT p50 | TTFT p95 | TPOT p50 |
|---|---:|---:|---:|---:|
| llm_server | 1,294 tok/s | 1,172 ms | 1,182 ms | 6.64 ms |
| **vLLM 0.11.0** | **4,182 tok/s** | **123 ms** | **161 ms** | **5.61 ms** |
| ratio | **0.31×** | **0.10×** | 0.14× | 0.84× |

**vLLM wins decisively at realistic prompt lengths: 3.2× throughput, 9.6× TTFT.**
The short-prompt parity is a real measurement of a narrow regime, not a general
claim, and quoting it alone would be dishonest.

## What the two rows together show

**TTFT is linear in prompt length — no chunked prefill.** llm_server consumes one
prompt token per forward pass, so TTFT ≈ `prompt_tokens × step_time`:
predicted 256 × 4.6 ms ≈ 1.18 s, measured **1.172 s**. vLLM packs a prompt into
one pass (chunked across steps when large), so its TTFT grows far slower — 21.6 ms
at 10 tokens, 123 ms at 256. This is the single largest gap in the system and it
widens with every additional prompt token.

**Throughput collapse follows from the same cause.** A prefilling sequence occupies
a batch slot and emits no output token, so with 256-token prompts and 64-token
generations, ~80% of the steps a sequence spends resident produce nothing. Hence
5,755 → 1,294 tok/s. vLLM only drops 5,633 → 4,182, because its prefill costs one
step, not 256.

**TPOT crosses over, and the reason is different from the above.** llm_server is
17% *faster* per output token at short context and 18% *slower* at long context.
Two costs trade places:

- *Per-step dispatch* — llm_server is a C++ loop issuing cuBLAS calls; vLLM
  re-enters Python to schedule every step. At 1B with ~70 cached tokens the step
  is short enough that this overhead dominates, and llm_server wins.
- *Attention kernel quality* — at ~300 cached tokens the decode attention kernel
  dominates the step. `paged_attention.cuh` uses 64 threads per (sequence, head),
  scalar `__half` loads and an fp32 accumulator; vLLM uses FlashAttention-style
  vectorized loads and tensor cores. There llm_server loses.

So the TPOT win at short prompts is a *framework-overhead* win, not a kernel win,
and it does not survive contact with realistic context lengths. Worth stating
plainly: the fair summary is "competitive on dispatch overhead, behind on kernel
quality and missing chunked prefill entirely."

## What this still does NOT show

- **1B is small, which flatters llm_server.** vLLM's CUDA graphs and fused kernels
  pay off more at 7B+, while its per-step Python overhead matters less.
- **One run per point.** No repeats, no variance, no concurrency sweep.
- **vLLM was not tuned** — near-defaults, and given only half the GPU.
- **Feature coverage is not comparable.** vLLM has chunked prefill, prefix caching,
  CUDA graphs, quantization, LoRA, tensor parallelism, dozens of architectures.
  This project implements the core serving loop, not a product.

## The two fixes this identifies, in priority order

1. **Chunked prefill** — process the prompt in one (or a few) forward passes
   instead of one token per pass. Needs varlen attention in the kernel (query
   length > 1, causal mask within the chunk) plus a scheduler decision splitting
   each step's token budget between prefill and decode. This is worth ~9× TTFT
   and ~3× throughput at 256-token prompts — by far the largest single win
   available, and the reason vLLM added it.
2. **Vectorized attention loads** — `__half2`/`float4` loads and more threads per
   head in `paged_attention.cuh`, to close the TPOT gap at long context.

## Reproduce

```bash
SNAP="/mnt/c/Users/siyua/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/<hash>"

# llm_server
./build/serve --weights weights --tokenizer onnx/tokenizer.json \
              --config onnx/config.json --kv-budget-mb 1024 --max-batch 64
python3 tools/bench_vs_vllm.py --name llm_server_p256 \
  --url http://localhost:8080/v1/completions \
  --prompt-tokens 256 --tokenizer-dir "$SNAP" \
  --requests 64 --concurrency 32 --max-tokens 64 \
  --out benchmarks/p256_llm_server.json

# vLLM (stop llm_server first — one server per GPU)
vllm serve "$SNAP" --served-model-name llama-3.2-1b-instruct \
  --port 8000 --dtype float16 --max-model-len 4096 \
  --gpu-memory-utilization 0.5
python3 tools/bench_vs_vllm.py --name vllm_p256 \
  --url http://localhost:8000/v1/completions --model llama-3.2-1b-instruct \
  --prompt-tokens 256 --tokenizer-dir "$SNAP" \
  --requests 64 --concurrency 32 --max-tokens 64 \
  --out benchmarks/p256_vllm.json

python3 tools/bench_vs_vllm.py --compare benchmarks/p256_*.json
```

Drop `--prompt-tokens` for the short-prompt point.
