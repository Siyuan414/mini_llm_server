# Stage 3 — Chunked prefill

**Model** LLaMA-3.2-1B-Instruct, FP16 · **GPU** RTX 5070 Ti · WSL2, CUDA 13
**Workload** 64 requests, concurrency 32, 256-token prompts, 64 tokens out, greedy
**Client** `tools/bench_vs_vllm.py` over `/v1/completions` SSE — identical for every row

Stage 2B consumed a prompt one token per forward pass, so TTFT was
`prompt_len × step_time`. At 256-token prompts that measured **1,172 ms** against
vLLM's 123 ms ([VS_VLLM.md](VS_VLLM.md)). Stage 3 batches prompt tokens with
decode tokens in a single pass under a per-step token budget.

## Headline

| | tok/s | TTFT p50 | TTFT p95 | TPOT |
|---|---:|---:|---:|---:|
| Stage 2B (one token per pass) | 1,294 | 1,172 ms | 1,182 ms | 6.64 ms |
| **Stage 3 (chunked prefill)** | **3,261** | **240 ms** | **253 ms** | **5.97 ms** |
| improvement | **2.5×** | **4.9×** | **4.7×** | 1.11× |
| vLLM 0.11.0 | 4,182 | 123 ms | 161 ms | 5.61 ms |

TTFT improved 4.9× and throughput 2.5× from the same hardware and the same
kernels — the change is that a prefilling sequence no longer occupies a batch
slot for 256 steps while producing nothing.

## What was built

| | |
|---|---|
| S3.1 | `paged_attention_varlen.cuh` — attention for `q_len >= 1`. Mask is a loop bound (`n_keys = q_pos[g] + 1`), not a branch; `lens[]` disappears. Decode is the degenerate case. |
| S3.2 | `BlockAllocator::append_tokens` — multi-token growth, all-or-nothing, plus `fits_ever` to reject prompts that can never fit (lazy allocation loses eager allocation's natural guard and would otherwise livelock). |
| S3.3 | `chunk_scheduler.h` — decodes first, prefill FCFS, per-sequence cap, block-aware. No CUDA. |
| S3.4 | `varlen_layer.h` + `serve_chunked.cpp` — mixed-batch forward pass and engine loop. |

### Correctness

- **Varlen vs decode**, exact zero on three cases: `q_len == 1` reproducing the
  decode kernel; ragged `q_len` 5/1/3 where global query index and sequence index
  diverge; shuffled physical blocks. The oracle is the decode kernel itself —
  a decode call at length L *is* "attend to keys 0..L-1", so matching N decode
  calls proves the mask by construction and cancels bugs common to both kernels.
- **Chunk-size invariance**: the same prompt served with `--max-chunk 512`
  (one pass) and `--max-chunk 2` (three passes) produces token-identical greedy
  output. If `q_pos` or the mask were off by one at a boundary, the text would
  diverge.
- **Scheduler**: 23 assertions in `test_chunk_scheduler.cpp`, including that a
  plan is *affordable* — applying it with `append_tokens` never fails.
- **Allocator**: `append_tokens(seq, 0, 40)` produces the same table as 40
  successive `append_token` calls.

## The configuration sweep, and what it showed

All five rows: same model, same 1 GB KV pool, same workload. Only scheduler
flags differ.

| budget / chunk / admits | tok/s | TTFT p50 | TTFT p95 | TPOT |
|---|---:|---:|---:|---:|
| 512 / 512 / ∞ | 3,159 | 123 ms | 287 ms | 7.96 ms |
| **2048 / 512 / ∞**  *(default)* | **3,261** | 240 ms | **253 ms** | **5.97 ms** |
| 2048 / 128 / ∞ | 3,202 | 246 ms | 251 ms | 6.13 ms |
| 2048 / 512 / 2 | 3,105 | **96.8 ms** | 297 ms | 8.26 ms |

**Throughput is 3,105–3,261 across every configuration — a 5% spread.** Scheduler
policy redistributes latency; it does not change how much work the GPU gets
through. That is the single most useful result here, because it says the
remaining 1.3× gap to vLLM is *kernel* work, not policy work, and that further
scheduler tuning is motion without progress.

Three findings, each of which contradicted a prediction:

**1. A bigger budget made p50 TTFT worse, not better** (123 → 240 ms), while
tightening the distribution (p95 287 → 253) and improving TPOT. A 2048-token
budget admits ~8 prompts per step, so nearly all in-flight requests prefill
simultaneously and finish together. A 512-token budget admits 2 per step, so
completions stagger: early ones fast, late ones slow. Both are work-conserving,
so the *last* completion lands at the same time either way — this is the
processor-sharing result from the FCFS analysis, reappearing across requests
instead of within a step.

**2. `max_chunk` did nothing** (240 vs 246 p50, 253 vs 251 p95). The cap exists to
bound head-of-line blocking, but at a 2048 budget there is room for everyone and
it never binds. Correct in principle, irrelevant at this operating point.

**3. Capping admissions gave the best p50 (96.8 ms, beating vLLM's 123) but the
worst TPOT (8.26 ms).** The prediction was that budget and admission rate were
independent knobs. They are not: mass admission finishes all prefill in ~4 steps
and every step afterwards is pure decode — short and uniform. Capped admission
*spreads* prefill across many more steps, so sequences that started decoding
early keep sharing steps with other sequences' chunks. Staggering does not
reduce prefill interference; it stretches it out.

So the real axis is **concentrated vs spread prefill interference**, and it is an
SLO choice rather than a tuning bug:

- `2048 / 512 / ∞` — best throughput, TPOT and tail. The default.
- `2048 / 512 / 2` — best p50 TTFT, ~38% worse TPOT. Use when
  time-to-first-token dominates.

## Remaining gap to vLLM

| | llm_server | vLLM | ratio |
|---|---:|---:|---:|
| throughput | 3,261 | 4,182 | 0.78× |
| TTFT p50 | 240 (96.8 tuned) | 123 | 0.51× (1.27× tuned) |
| TTFT p95 | 253 | 161 | 0.64× |
| TPOT | 5.97 | 5.61 | 0.94× |

TPOT is within 6% and TTFT p50 can be made better than vLLM's. The throughput
and p95 gaps are attributable to kernel quality: `paged_attention_varlen.cuh`
uses 64 threads per (query token, head) with scalar `__half` loads and an fp32
accumulator, and every query block re-reads K/V from HBM. FlashAttention tiles
over queries *and* keys so a K/V tile is loaded once into shared memory and
reused across a tile of queries, and it uses tensor cores.

Next, in order of expected return:

1. **Vectorized loads** — `__half2`/`float4` in Phases 1 and 3, more threads per
   head. Cheapest real throughput win.
2. **Query/key tiling** — stop re-reading K/V per query block. This is the
   structural fix and the larger win for long contexts.
3. **CUDA graphs** for the pure-decode step, to remove per-launch overhead.

Speculative decoding is deliberately *not* on this list: it buys single-stream
latency by spending idle arithmetic capacity, and at concurrency 32 there is
none to spend — it would likely reduce throughput here. It belongs with a
low-concurrency benchmark ([VS_VLLM.md](VS_VLLM.md) covers why).

## Reproduce

```bash
cmake --build build -j

# correctness
nvcc -std=c++17 -I kernels -I src tools/test_paged_attention_varlen.cu \
     -o build/test_varlen --extended-lambda -arch=sm_120 && ./build/test_varlen
g++ -std=c++17 -I src tools/test_chunk_scheduler.cpp -o build/test_chunk && ./build/test_chunk
g++ -std=c++17 -I src tools/test_block_allocator.cpp -o build/test_alloc && ./build/test_alloc

# chunk-size invariance: these two must produce identical greedy output
./build/serve_chunked ... --max-chunk 512   # one pass
./build/serve_chunked ... --max-chunk 2     # three passes

# benchmark (default config)
./build/serve_chunked --weights weights --tokenizer onnx/tokenizer.json \
  --config onnx/config.json --kv-budget-mb 1024 --max-batch 64 \
  --max-batched-tokens 2048 --max-chunk 512

SNAP="/mnt/c/Users/siyua/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/<hash>"
python3 tools/bench_vs_vllm.py --name chunked \
  --url http://localhost:8080/v1/completions \
  --prompt-tokens 256 --tokenizer-dir "$SNAP" \
  --requests 64 --concurrency 32 --max-tokens 64 \
  --out benchmarks/p256_chunked.json
```

## Caveats

- One run per configuration. No repeats, no variance, no concurrency sweep.
- 256-token prompts only. The gap to vLLM should widen at longer contexts, where
  kernel quality matters more.
- 1B is a small model, which flatters llm_server: vLLM's CUDA graphs and fused
  kernels pay off more at 7B+, while its per-step Python overhead matters less.
- vLLM was not tuned — near-defaults, given half the GPU.
