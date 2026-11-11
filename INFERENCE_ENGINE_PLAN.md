# Mini Inference Server — Senior ML Inference Engineer Plan

**Goal:** Build a production-style LLM inference server ("mini-vLLM"), demonstrating
end-to-end ownership from kernel to serving layer.

**Timeline:** ~3–4 months | **Platform:** RTX 5070 Ti (WSL) for development; Jetson Orin
Nano as the edge-deployment story.

**Why this plan:** At 3 YOE + MS, the gap to senior isn't breadth — it's one deep,
coherent system whose every design decision you can defend. Continuous batching, paged
KV cache, and speculative decoding are what senior inference interviews probe; almost no
mid-level candidate has built them.

**Working mode (from Stage 2 onward):** coaching, not delivery. For each piece: design
and tradeoffs get explained, comprehension gets checked with questions, then *you* write
the code and it gets reviewed. Stage 1 was written for you, and the lesson learned was
that code you didn't write isn't code you can defend.

The split that emerged in practice: anything an interviewer would ask about — kernels,
scheduler policy, cache design — you write. Plumbing — CLI parsing, buffer allocation,
test harnesses, benchmark scripts — gets written for you.

**Ordering principle (learned the hard way at the vLLM comparison):** measure before
building. The comparison was planned for Stage 4 and run early instead; it replaced the
planned next feature (speculative decoding) with the one the numbers demanded (chunked
prefill), and gave a before-number to prove the after against. Every stage below now
ends with a measurement, and the next stage is chosen from it rather than from the plan.

---

## Baseline before this project

- Quantization study: GPTQ, AWQ, SmoothQuant, mixed precision, QAT fundamentals
- CUDA kernels: tiled matmul, flash attention (Triton), Nsight profiling
- Jetson Orin Nano: TRT Edge-LLM (28.2 tok/s decode, Qwen3-4B) vs llama.cpp (18.1 tok/s)
- Single-request TRT runtime in `../llama_jetson` — the seed for this project

---

## Stage 1 — Single-request engine ✅ COMPLETE

→ [benchmarks/STAGE1.md](benchmarks/STAGE1.md)

| | Delivered |
|---|---|
| S1.1 | One engine, two TRT optimization profiles: prefill (seq 1–2048, past=0), decode (seq=1, past 0–2047) |
| S1.2 | `KVCache` class — ping-pong buffers, zero copies per step |
| S1.3 | True batched prefill — whole prompt in one `enqueueV3` |
| S1.4 | GPU sampling (temperature / top-k / top-p), 1.7% overhead |
| S1.5 | Dimensions from `config.json`; device-resident attention mask |
| S1.6 | Verified vs HF: cosine 1.0000, 5/5 top-1; benchmark sweep with warmup |

**Results:** 41k tok/s prefill (512 tokens), 235 tok/s decode, 2048 context.

**Two bugs worth remembering:**

1. **KV must be contiguous.** TRT reads `past_key_values` as `[1, H, past, D]` with head
   *h* at `h·past·D` — a stride that changes every step. A fixed-capacity buffer strided
   by `max_seq` misreads every head after head 0. Symptom: fluent, wrong output. Fix:
   ping-pong, since `present` is already the layout `past` needs next.
2. **The pre-tokenizer is a regex with ordered alternatives**, not a categorizer.
   `1969` → `196`+`9` (digits capped at 3), `(n` → one chunk (rule 2 precedes rule 4).
   Getting this wrong shifted logits by KL 0.52 while still producing fluent text.

---

## Stage 2 — Continuous batching ← the core of the project

**Split into 2A and 2B deliberately.** Paged KV requires a custom attention kernel,
because TRT's attention demands a contiguous `past` and paging scatters KV across blocks.
Building the scheduler and that kernel simultaneously means every bug is ambiguous, with
no reference to compare against. So:

- **2A** — fixed slots, TRT's stock attention → a working multi-request server and a
  correctness baseline. Teaches continuous batching.
- **2B** — swap in paged KV + custom attention as a pure memory optimization, validated
  against 2A. Teaches paged attention.

### S2.0 — Batched engine spike ✅ COMPLETE

Rebuilt the engine with a batch range (min 1 / opt 4 / max 32) on every input, and
verified ragged batched decode with `lens = [500, 30, 200, 1]`.

**Verification design.** `isfinite()` is a worthless gate — it passes even when masking
leaks. Three invariance tests instead:

| Test | Question | Result |
|---|---|---|
| A | slot 1 batched vs alone | 0.234 |
| B | same batch, padding poisoned +1e4 vs −1e4 | **0.000000** |
| C | slot 1 at past=500 (padded) vs past=30 (not) | 0.234 |

B is the correctness gate and it is *exact*: flipping 470 positions of masked memory
changed nothing, so padding provably never reaches the output. A and C being identical
showed batch size doesn't affect numerics at all — only reduction length does. 0.234 on a
signal of max 14.66 is 1.6%: fp16 rounding from a 501-term softmax vs a 31-term one.

**Lesson: "numerically different" is not "incorrect."** A and C measure batch-composition
reproducibility, a real property of every batched server (vLLM included), not correctness.

**What the spike established for the design:**

- Per-row `position_ids` are mandatory — a slot's new token is at *its own* next position,
  not at `past`; passing `past` misrotates RoPE.
- The mask has three regions per row: `[1 × len_i][0 × (past − len_i)][1 × seq]`.
  TRT appends new KV at index `past`, so the new columns are always last.
- **The shared `past` is the fundamental limitation.** One `past = max(lens)` for the whole
  batch means every slot carries the longest sequence's length, and `past` grows by 1 each
  step for every slot regardless of its real length. One 2000-token request forces all 8
  slots to carry 2000 tokens of KV. *This is the motivation for 2B, derived rather than
  asserted.*

### Stage 2A — Continuous batching, fixed slots ✅ COMPLETE

→ [benchmarks/STAGE2.md](benchmarks/STAGE2.md)

| | Delivered |
|---|---|
| S2.1 | `BatchKVCache` — N slots with independent lengths, scatter kernel, per-row mask and positions |
| S2.2 | `Request` + `Scheduler` — waiting queue, running set, admission control, eviction, batch rebuilt every step so requests join mid-flight |
| S2.3 | Batched sampling — one segmented argmax for B sequences (`batched_pick.cuh`) |
| S2.4 | HTTP front-end — OpenAI-compatible SSE streaming, cancellation on disconnect |
| S2.5 | Verify + benchmark — 24/24 outputs identical to sequential |

**Results:** 947 tok/s peak at B=16 (5.2× single-request).

**Unresolved:** throughput collapses above B=16 (152 tok/s at B=32). Suspected cause
is TRT context memory sized for profile maxima (batch 32 × seq 2048) across two
execution contexts; `cudaMemGetInfo` reports 16302/16302 MB at every batch size on
WSL, making it useless for confirming this. Superseded by 2B rather than fixed.

### Stage 2B — Paged KV + custom decode path ✅ COMPLETE

→ [benchmarks/STAGE2B.md](benchmarks/STAGE2B.md)

| | Delivered |
|---|---|
| S2.6 | `BlockAllocator` — 16-token blocks, free list, block tables, no CUDA, 10 tests |
| S2.7 | Custom CUDA decode path instead of a TRT plugin: RMSNorm/RoPE/SiLU-mul/embedding kernels + 7 cuBLAS GEMMs per layer, verified layer-by-layer vs HF |
| S2.8 | Paged attention kernel — block-table indirection, 3 invariance tests, block-major restructure after profiling |
| S2.9 | `paged_runtime` — preemption (newest-first, recompute), KV budget sweep |

**Results:** 15,090 tok/s (**15.9×** Stage 2A's 947), TTFT p50 524 → 83 ms,
128 concurrent requests in 512 MB. KV utilization 86% vs 11%.

**Why a custom decode path rather than a TRT plugin (S2.7):** replacing attention
inside the engine needs ONNX graph surgery plus `IPluginV3` — a fiddly,
version-fragile API where a bug looks like a TRT error rather than a math error.
Writing the forward pass directly means everything is unit-testable and verifiable
against HF layer by layer. Cost: ~16% slower per step at equal batch (no kernel
fusion), which is what buys the ability to page.

**Findings worth carrying forward:**

- Paged attention is at **parity per call**, not faster — indirection costs about
  what the reduced scanning saves. The win is 8–40× memory. (Matches vLLM.)
- The naive Phase-3 loop was **2.3× slower** than the kernel it replaced, from
  per-token integer division; block-major iteration fixed it. Only the benchmark
  revealed this.
- **Raw throughput overstates under preemption** — at 64 MB, 32% of generated
  tokens were recompute. Goodput is the honest metric.
- Prefill and decode are the **same operation** in this path (no admission stall),
  at the cost of P steps for a P-token prompt. Chunked prefill is the fix.
  *This was recorded as a note here and cost 1,172 ms of TTFT in the vLLM
  comparison below — the gap between noticing a weakness and measuring it.*

## vLLM comparison ✅ COMPLETE — *ran early, and it rewrote the plan*

→ [benchmarks/VS_VLLM.md](benchmarks/VS_VLLM.md)

Originally scheduled as Stage 4, after speculative decoding. Doing it first was the
best decision in the project, because it replaced a guess about what to build next
with a measurement.

Same client, same GPU, same model, two prompt lengths:

| prompt | | tok/s | TTFT p50 | TPOT |
|---|---|---:|---:|---:|
| ~10 tok | llm_server | 5,755 | 61 ms | **4.57** |
| | vLLM | 5,633 | **22 ms** | 5.49 |
| 256 tok | llm_server | 1,294 | 1,172 ms | 6.64 |
| | vLLM | **4,182** | **123 ms** | **5.61** |

**Parity at short prompts, 3.2× behind at realistic ones.** Quoting the first row
alone would have been dishonest. TTFT proved to be exactly `prompt_len × step_time`
— predicted 256 × 4.6 ms = 1.18 s, measured 1.172 s — which identified the missing
feature precisely rather than by intuition.

TPOT also *crosses over*: ahead at short context (framework dispatch overhead
dominates, and a C++ loop beats a Python scheduler) and behind at long context
(the attention kernel dominates, and scalar fp16 loads lose to FlashAttention).
A win in one regime does not survive into the other.

**This reordered the roadmap.** Speculative decoding was planned next; the
measurement said chunked prefill, and said so with a number. Speculative decoding
also targets TPOT, which at 256-token prompts was the *smallest* gap (0.84×) while
TTFT sat at 0.10×.

## Stage 3 — Chunked prefill ✅ COMPLETE

→ [benchmarks/STAGE3.md](benchmarks/STAGE3.md)

| | Delivered |
|---|---|
| S3.1 | `paged_attention_varlen.cuh` — attention for `q_len >= 1`. Mask is a loop bound (`n_keys = q_pos[g] + 1`), not a branch; `lens[]` disappears entirely. Decode is the degenerate case. |
| S3.2 | `BlockAllocator::append_tokens` — multi-token growth, all-or-nothing, plus `fits_ever` to reject prompts that can never fit |
| S3.3 | `chunk_scheduler.h` — decodes first, prefill FCFS, per-sequence cap, block-aware. No CUDA, 23 tests. |
| S3.4 | `varlen_layer.h` + `serve_chunked.cpp` — mixed-batch forward pass and engine loop |
| S3.5 | Chunk-size invariance + configuration sweep |

**Results at 256-token prompts:** TTFT 1,172 → **240 ms (4.9×)**, throughput
1,294 → **3,261 tok/s (2.5×)**, TPOT 6.64 → 5.97 ms. Tuned for latency,
**96.8 ms TTFT p50 — better than vLLM's 123 ms.**

**Why varlen came first, and what it also unlocks.** Speculative decoding verifies
K draft tokens in one forward pass, which *is* `q_len = K` with a causal mask among
the drafts — the same kernel chunked prefill needs. Building it for prefill leaves
speculative decoding as acceptance sampling plus cache rollback, not a second kernel
project. That's why the order changed rather than the scope.

**Findings worth carrying forward:**

- **Chunk-size invariance is the cheap correctness gate.** The same prompt served
  with `--max-chunk 512` (one pass) and `--max-chunk 2` (three passes) must produce
  token-identical greedy output. No reference model needed; catches any off-by-one
  in `q_pos` or the mask at a chunk boundary.
- **Scheduler policy redistributes latency but not throughput.** Across five
  configurations throughput varied 3,105–3,261 tok/s — a 5% spread — while TTFT p50
  ranged 97–246 ms. This localizes the residual gap to vLLM as *kernel* work and
  makes further scheduler tuning provably pointless.
- **A bigger token budget made p50 TTFT worse** (123 → 240 ms) while tightening the
  tail. Admitting ~8 prompts per step means they all prefill together and finish
  together; a smaller budget staggers completions. The processor-sharing result from
  the FCFS analysis, reappearing across requests instead of within a step.
- **Budget and admission rate are not independent knobs.** Capping admissions gave
  the best p50 (96.8 ms) and the *worst* TPOT (8.26 ms): staggering doesn't reduce
  prefill interference, it stretches it over more steps. The real axis is
  concentrated vs spread interference, and it's an SLO choice.
- **`max_chunk` never bound** at a 2048 budget (240 vs 246 ms). Correct in
  principle, irrelevant at this operating point — worth knowing which knobs are
  live before tuning them.

## Stage 4 — Attention kernel: profiled, scoped, deliberately not built

→ [benchmarks/STAGE4_FINDINGS.md](benchmarks/STAGE4_FINDINGS.md)

Investigated as the measurement indicated, then stopped on purpose. What the
profiling established:

- Attention is **47.4%** of a decode step (gemm 36.9%, elementwise 7.5%, lm_head 8.1%)
- **11.6% load efficiency** — 3.7 useful bytes per 32-byte sector, an 8.8× overfetch
  verified by hand against the ideal sector count
- **Not** bandwidth-bound (15–19% of peak on every shape) and **not** compute-bound
  (20% SM throughput). Latency-bound, stalling on requests it mostly wastes.
- Cause: Phase 1 assigns one thread per token, so adjacent lanes read addresses
  128 bytes apart. The loop *looks* strided and is still uncoalesced — the stride is
  over the wrong axis.
- Amdahl ceiling: 2× faster attention → 31% faster step → ~4,200 tok/s, which is
  vLLM's number.

The fix (cooperative coalesced load of a K tile into shared memory, then
thread-per-token math out of shared) is specified in the findings doc. Left
unimplemented: the remaining work is implementation grind, and this project exists
to understand the concepts rather than to win a benchmark. The concepts — profile
before optimizing, compute the ceiling first, and why tiling *forces* online softmax
— are recorded.

Two methodology rules came out of it and now apply to everything: **report min of N
trials, not a mean** (the same binary measured 113 µs idle and 179 µs with a game
open; minima reproduce across sessions to ~1%, medians drift 20–30% from thermal
droop within one run), and **verify the binary changed before explaining a result**
(an entire mechanism was theorized for a slowdown that turned out to be an
unrebuilt kernel plus a background game).

## Stage 5 — Speculative decoding

Deliberately *after* the kernel work, and for a reason the benchmarks made explicit:
speculative decoding spends idle arithmetic capacity to buy single-stream latency,
and at concurrency 32 the GPU is already compute-saturated — it would likely *reduce*
throughput. It is a low-concurrency latency feature, not a throughput feature.

- **S5.1** Draft model + its own KV pool. One `BlockAllocator` and one block table
  serve both models (logical→physical mapping is model-independent); only the
  per-block byte stride differs.
- **S5.2** `BlockAllocator::truncate(seq_id, new_len)` — rollback. The first
  operation in this system that makes a sequence *shrink*; everything so far only
  grows, which is why `append_tokens` may assume `have >= blocks_for(cur_len)`.
- **S5.3** Acceptance sampling — accept draft token `x` with probability
  `min(1, p_target(x)/p_draft(x))`, resample from `max(0, p_target − p_draft)` on
  rejection. This makes the output distribution provably identical to plain
  sampling; speculative decoding is exact, not an approximation.
- **S5.4** Benchmark at concurrency **1 and 4**, not 32. Measuring at 32 would
  correctly implement the feature and record a regression.

## Stage 6 — Write-up

- **S6.1** One deep technical post: the scheduler, the paged cache, and the
  measure-then-build ordering that reshaped the plan.
- **S6.2** Repo polish: architecture diagram, consolidated benchmark tables.

---

## Open questions

- Stage 2A throughput collapse above B=16 — never root-caused; superseded by 2B.
- Prefill throughput peaked at 512 tokens (41k tok/s) and fell to 10k at 1900 in the
  TRT path. Moot now that prefill runs on the custom kernel, but the same O(n²)
  question applies to `paged_attention_varlen` and S4.2 should measure it.
- TTFT p95 (253 ms) remains 1.6× vLLM's (161 ms) in every configuration tried. Not
  chunk monopolization — `max_chunk` didn't move it. Suspect admission queueing;
  needs a concurrency sweep to confirm.
- Every benchmark is a single run. No repeats, no variance, no error bars.
- Only 256-token prompts were swept. The gap to vLLM should widen at longer contexts.

## Explicitly deprioritized

- More quantization work (enough depth already)
- MLIR/TVM (different career track)
- k8s (learn on the job if a role requires it)
- Full Unicode pre-tokenizer (needs PCRE2/ICU; ASCII approximation verified 8/8)
