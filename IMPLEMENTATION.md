# llm_server — what was built and how

An LLM inference server written from scratch in C++/CUDA: tokenizer, forward
pass, KV cache, scheduler and HTTP front-end, with no inference framework in the
hot path. Target model LLaMA-3.2-1B-Instruct (now also 3B), FP16, RTX 5070 Ti.

This document is the *how*. [INFERENCE_ENGINE_PLAN.md](INFERENCE_ENGINE_PLAN.md)
is the roadmap and the reasoning behind the ordering; `benchmarks/` holds the
measurements.

---

## Contents

1. [Where the code lives](#1-where-the-code-lives)
2. [Stage 1 — single-request engine on TensorRT](#2-stage-1--single-request-engine-on-tensorrt)
3. [Stage 2A — continuous batching with fixed slots](#3-stage-2a--continuous-batching-with-fixed-slots)
4. [Stage 2B — paged KV and our own forward pass](#4-stage-2b--paged-kv-and-our-own-forward-pass)
5. [HTTP serving layer](#5-http-serving-layer)
6. [vLLM comparison — the measurement that redirected the project](#6-vllm-comparison--the-measurement-that-redirected-the-project)
7. [Stage 3 — chunked prefill](#7-stage-3--chunked-prefill)
8. [Stage 4 — profiling, and the decision not to optimize](#8-stage-4--profiling-and-the-decision-not-to-optimize)
9. [How correctness was established](#9-how-correctness-was-established)
10. [Bugs that taught something](#10-bugs-that-taught-something)
11. [Current state](#11-current-state)

---

## 1. Where the code lives

```
src/
  runtime.cpp          Stage 1   single-request engine over TensorRT
  kv_cache.h           Stage 1   ping-pong KV buffers
  batch_runtime.cpp    Stage 2A  continuous batching over TensorRT
  batch_kv_cache.h     Stage 2A  N fixed slots + scatter
  scheduler.h          Stage 2A  request lifecycle, admission  (no CUDA)
  block_allocator.h    Stage 2B  paged blocks, free list, block tables (no CUDA)
  decode_layer.h       Stage 2B  hand-written decoder layer, one token/sequence
  paged_runtime.cpp    Stage 2B  continuous batching on the paged cache
  varlen_layer.h       Stage 3   decoder layer over a mixed prefill+decode batch
  chunk_scheduler.h    Stage 3   per-step token budget policy  (no CUDA)
  serve.cpp            Stage 2B  HTTP server, one prompt token per pass
  serve_chunked.cpp    Stage 3   HTTP server with chunked prefill
  tokenizer.h                    byte-level BPE, LLaMA-3 pre-tokenizer
  model_config.h                 dimensions from config.json
  weights.h                      raw fp16 tensor loader
  http.h                         minimal HTTP/1.1 + SSE, no dependencies

kernels/
  layer_kernels.cuh    RMSNorm, RoPE, SiLU-mul, embedding, residual add
  paged_attention.cuh         decode attention, block-table indirection
  paged_attention_varlen.cuh  the same for q_len >= 1
  sampling.cuh         temperature / top-k / top-p on device
  batched_pick.cuh     one segmented argmax for B sequences
  kv_scatter.cuh       Stage 2A fixed-slot KV scatter

tools/                 export, verification, benchmarks, unit tests
benchmarks/            STAGE1.md STAGE2.md STAGE2B.md STAGE3.md
                       VS_VLLM.md STAGE4_FINDINGS.md
```

Three files contain **no CUDA at all** — `scheduler.h`, `block_allocator.h`,
`chunk_scheduler.h`. That was deliberate: the policy is the interesting part and
it should be unit-testable on the host without a GPU. It also mirrors how vLLM
separates `scheduler.py` from the model runner.

---

## 2. Stage 1 — single-request engine on TensorRT

**Goal:** one prompt in, tokens out, correct and fast, with TensorRT doing the
matrix work.

### Two optimization profiles in one engine

Prefill and decode have incompatible shapes. Prefill feeds the whole prompt with
an empty cache; decode feeds one token against a large cache. A single TRT
profile covering both would be optimized for neither, so the engine carries two:

| profile | seq_len | past_len |
|---|---|---|
| prefill | 1–2048 | 0 |
| decode | 1 | 0–2047 |

`setOptimizationProfileAsync` switches between them. Two execution contexts, each
pinned to one profile, avoid re-selecting per step.

Built as a `STRONGLY_TYPED` network — in TRT 10 the old `BuilderFlag::kFP16` and
`EXPLICIT_BATCH` are gone, and precision comes from the ONNX file's own dtypes.

### Ping-pong KV cache (`kv_cache.h`)

This is where the project's first real bug lived, and the fix is the design.

TRT reads `past_key_values` as a **contiguous** `[B, H, past, D]` tensor, so head
*h* starts at offset `h · past · D`. That stride depends on `past`, which grows
every step. A natural implementation — allocate `max_seq` capacity once, write
new KV at the end — produces a buffer strided by `max_seq` while TRT reads it
strided by `past`. Head 0 is correct and every later head reads the wrong
memory. The output stays fluent and is wrong, which is the worst failure mode.

The fix exploits a property of the graph: the `present_key_values` **output** is
already exactly the layout `past_key_values` needs next step. So keep two
buffers and swap:

```
step N:    past = buffer A  ->  present = buffer B
step N+1:  past = buffer B  ->  present = buffer A
```

No copies, no stride arithmetic, and the invariant is structural rather than
maintained by hand. Interface: `fits / bind / commit / length / reset` — which
later became the seam Stage 2 replaced.

### Tokenizer (`tokenizer.h`)

Byte-level BPE in C++, loading `tokenizer.json` directly. Two details cost real
time:

- **Merge format varies.** Older `tokenizer.json` stores merges as `"a b"`
  strings; current ones store `[["a","b"], ...]`. The parser handles both.
- **The pre-tokenizer is an ordered regex alternation, not a classifier.**
  LLaMA-3's pattern tries alternatives in order, and the ordering is load-bearing:
  digits are capped at 3 per token, so `1969` → `196` + `9`, and the
  contraction rule precedes the punctuation rule, so `(n` is one chunk. Treating
  it as "classify each character" shifted logits by KL 0.52 while still producing
  fluent text. Now 8/8 token sequences match HF.

### Sampling (`sampling.cuh`)

Temperature, top-k, top-p entirely on device: radix sort the logits, prefix-scan
the probabilities, inverse-CDF sample. Keeping it on the GPU avoids a
`[vocab]` copy to host per token — 1.7% overhead measured.

**Results:** 41k tok/s prefill (512 tokens), 235 tok/s decode, cosine 1.0000 and
5/5 top-1 against HF.

---

## 3. Stage 2A — continuous batching with fixed slots

**Goal:** many concurrent requests, each joining and leaving mid-flight.
TensorRT keeps doing the matrix work, so the new ideas are isolated to scheduling
and cache layout.

### Why a spike came first

Before writing the scheduler, a batched engine was built and probed with
`lens = [500, 30, 200, 1]` to find out what batching actually requires. Three
findings shaped everything after:

- **Per-row `position_ids` are mandatory.** A slot's new token sits at *its own*
  next position, not at the batch's shared `past`. Passing `past` misrotates RoPE.
- **The mask has three regions per row:** `[1 × len_i][0 × (past − len_i)][1 × seq]`.
  TRT appends new KV at index `past`, so the new columns are always last.
- **The shared `past` is the fundamental limit.** One `past = max(lens)` for the
  whole batch means every slot carries the longest sequence's length. One
  2000-token request forces all 8 slots to carry 2000 tokens of KV. *This is why
  Stage 2B exists* — derived from measurement rather than asserted.

### `BatchKVCache` (`batch_kv_cache.h`)

N independent slots, each with its own length. Every step it builds the padded
KV tensor, the per-row mask and the per-row positions. New KV is written with a
scatter kernel (`kv_scatter.cuh`) rather than N small `cudaMemcpy`s.

`B = batch_size()` returns *highest active slot + 1*, not the count of active
slots — the tensor must be contiguous from slot 0, so a hole costs a wasted row
rather than a reshuffle.

### `Scheduler` (`scheduler.h`)

Pure policy, no CUDA: `waiting_` (deque), `running_` (vector, found by `.slot`),
`done_`. Lifecycle is `submit → next_admission → mark_running → on_token →
retire`.

The interesting knob is `max_admits_per_step`. Admitting a request requires a
separate batch-1 prefill, which stalls the decode batch (~12 ms for a 512-token
prompt versus ~5 ms per decode step), so each admission costs running requests
~2.5 steps of latency. Three policies — eager, lazy, capped — with capped at 1 as
the default. Stage 2.5 measured this rather than assuming it.

**Results:** 947 tok/s at B=16, 24/24 outputs identical to sequential execution.

**Unresolved:** throughput collapses above B=16 (152 tok/s at B=32). Suspected
TRT context memory sized for profile maxima across two contexts;
`cudaMemGetInfo` reports 16302/16302 MB at every batch size under WSL, so it
couldn't be confirmed. Superseded by 2B rather than fixed.

---

## 4. Stage 2B — paged KV and our own forward pass

**Goal:** stop reserving a full `max_seq` window per slot. Requests averaged ~55
tokens but reserved 512, so ~89% of the KV cache was dead space.

This required replacing TensorRT's attention, because TRT demands a contiguous
`past` and paging scatters KV across blocks.

### Decision: write the forward pass rather than a TRT plugin

Replacing attention inside the engine means ONNX graph surgery plus an
`IPluginV3` implementation — a fiddly, version-fragile API where a bug surfaces
as a TensorRT error rather than a math error. Writing the forward pass directly
means every piece is unit-testable and verifiable against HF layer by layer.

Cost: ~16% slower per step at equal batch, because there is no kernel fusion.
That 16% is what buys the ability to page.

### `BlockAllocator` (`block_allocator.h`) — the OS page table

No CUDA. Fixed 16-token blocks, a free list, and a per-sequence block table.

```
physical:  [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ][ 7 ] ...   16 tokens each
seq A (35 tok) -> [3, 0, 6]      48 token-slots, not 512
seq B (20 tok) -> [1, 5]         32 token-slots
```

Uniform block size means **no external fragmentation** — any free block satisfies
any request, so N free blocks are always fully usable. The only waste is the
partially-filled last block, averaging 8 tokens per sequence versus 457 in
Stage 2A.

Key operations:

- `blocks_for(n)` — ceiling division
- `allocate(seq, n)` / `release(seq)` — pop from / push to the free stack
- `locate(seq, pos)` → `(physical_block, offset)` — the translation the kernel does
- `flatten(seq_ids, max_blocks)` → row-major `[n_seqs, max_blocks]` padded with
  **−1**, so a bug reads an obviously invalid id rather than silently hitting
  block 0
- `utilization(lens)` — measured 86% paged versus 11% in Stage 2A

**Division of labour with the GPU:** this class is the page table, the attention
kernel is the MMU. Allocation runs about once per 16 tokens over a few dozen ints
— free lists, hash maps, branching, all things CPUs are good at. Translation runs
every step over every cached token, on the GPU, via the flattened table.

### Running out mid-generation

A running sequence needs a new block every 16 tokens and cannot be failed
halfway. Two defences:

1. **Watermark** — refuse to admit a new request unless a reserve remains.
2. **Preemption** — when it happens anyway, evict a victim, free its blocks,
   requeue it for re-prefill. This is vLLM's "recompute" policy.

Victim selection is **newest-first (LIFO)**: least work invested. The victim is
reinserted at the *front* of the waiting queue so it is re-admitted **last**,
which prevents immediate re-eviction thrash.

Preemption was initially absent, and the failure was instructive: at a 64 MB
budget the server deadlocked because the watermark (128) exceeded the total pool
(122 blocks), and at 128/256 MB it simply reported "out of blocks".

### The forward pass (`decode_layer.h`)

One LLaMA decoder layer, by hand:

```
h = rmsnorm(x, w_in)
q,k,v = Wq·h, Wk·h, Wv·h
rope(q, pos); rope(k, pos)
write k,v into the paged cache at position len-1      <- BEFORE attention
a = paged_attention(q, cache)
x += Wo·a
h = rmsnorm(x, w_post)
x += Wdown · silu_mul(Wgate·h, Wup·h)
```

Seven cuBLAS GEMMs plus five custom kernels per layer.

**All seven GEMMs have the same form**, which is worth the trick that makes it
so. `export_weights.py` writes projections transposed to `[in, out]` row-major.
cuBLAS is column-major, so a row-major `[in, out]` matrix *is* a column-major
`[out, in]` matrix — exactly the operand `cublasHgemm` wants for
`y[out] = W[out,in] · x[in]`. No transpose flag, no runtime copy:

```cpp
cublasHgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
            out, B, in, &one, W, out, x, in, &zero, y, out);
```

The transpose happens once, offline, in the exporter.

### Custom kernels (`layer_kernels.cuh`)

- **`rmsnorm`** — warp-shuffle reduction for the sum of squares, accumulated in
  fp32 (fp16 overflows on 2048 squared values), scale broadcast via shared memory.
- **`rope_inplace`** — one block per `(sequence, head)`, `blockIdx` for identity
  and `threadIdx` for elements. **LLaMA-3 pairs element `i` with `i + D/2`**, not
  with `i+1`; adjacent pairing produces fluent-and-wrong output. `rope_theta` is
  500000, not the older 10000.
- **`silu_mul`**, **`embedding`**, **`residual_add`** — straightforward, but the
  coalescing rule bit twice: iterate `for (i = tid; i < n; i += blockDim.x)`, so
  adjacent threads touch adjacent addresses. Giving each thread a contiguous
  chunk is the intuitive version and is several times slower.

### Paged attention (`paged_attention.cuh`)

Grid `(Hq, B)` — one block per `(sequence, query head)`. Dynamic shared memory
holds `[ scores(len) | reduce(kThreads) | table(max_blocks) ]`.

Three phases:

1. `scores[t] = dot(q, K[t]) * scale`, with `physical = table[t / BS]`
2. softmax (block-wide max, then block-wide sum)
3. `out[d] = Σ scores[t] · V[t][d]`, thread `d` owning output dimension `d`

GQA: `kv_head = h / (Hq / Hkv)`, so 4 query heads share one KV head on the 1B.

**No online softmax**, and the reason matters: the score vector is `1 × N`, not
`N × N`. It fits in shared memory, so two passes over it are free. (Stage 3's
tiling discussion revisits this — once scores *don't* fit, online softmax becomes
mandatory. It is what makes tiling possible, not primarily a stability device.)

**Phase 3 is block-major, and that was a 2.3× fix.** The obvious version derives
the address from `t`:

```
logical = t / BS;  offset = t % BS;  physical = table[logical];
```

In Phase 1 that's fine — threads stride over `t`, so each division happens once.
In Phase 3 *every* one of the D threads walks *every* token, so the same
divisions repeat D times: ~D·len ≈ 3200 integer divisions per block at D=64,
len=50. At ~20 cycles each, this made paged attention **slower than the
fixed-slot kernel it replaced** despite scanning 10× fewer tokens. Looping
block-major — outer over logical blocks, inner over offsets — removes the
division entirely and reads the table once per 16 tokens. Only the benchmark
revealed this.

**Results:** 15,090 tok/s (15.9× Stage 2A), TTFT p50 524 → 83 ms, 128 concurrent
requests in 512 MB.

**Finding worth carrying:** paged attention is at **parity per call**, not faster
— indirection costs about what reduced scanning saves. The win is 8–40× memory.
This matches vLLM's own result.

**Second finding:** raw throughput overstates performance under preemption. At a
64 MB budget, 14,097 tokens were generated but only 9,600 were useful — 32%
recompute. **Goodput** is the honest metric.

---

## 5. HTTP serving layer

`http.h` is a minimal HTTP/1.1 server with no dependencies: parse the request
line and headers, route, stream. Two response shapes behind a `Format` enum —
OpenAI-compatible SSE on `/v1/completions` (with the `data: [DONE]` sentinel) and
a simpler `{"token": ...}` shape on `/generate` for curl.

OpenAI compatibility is worth ~40 lines: `openai-python`, LangChain and any chat
UI work unmodified, and it is what vLLM, TGI and llama.cpp all speak — which is
what later made a single benchmark client able to measure both servers.
`/v1/chat/completions` is **not** implemented; it needs the model's chat template
applied to a messages array.

**Threading:** one engine thread owns all CUDA. HTTP connection threads only
touch a mutex-protected queue and their own socket. A `Stream` is a `shared_ptr`
so the engine can outlive the connection thread that accepted it, and a client
disconnect is detected as cancellation.

---

## 6. vLLM comparison — the measurement that redirected the project

Originally planned as the final stage. Running it early was the best decision in
the project.

`tools/bench_vs_vllm.py` is server-agnostic: both servers speak OpenAI SSE, so
one client measures TTFT, TPOT and throughput for either. Only one server holds
the GPU at a time.

| prompt | | tok/s | TTFT p50 | TPOT |
|---|---|---:|---:|---:|
| ~10 tok | llm_server | 5,755 | 61 ms | **4.57** |
| | vLLM 0.11 | 5,633 | **22 ms** | 5.49 |
| 256 tok | llm_server | 1,294 | 1,172 ms | 6.64 |
| | vLLM 0.11 | **4,182** | **123 ms** | **5.61** |

Parity at short prompts; **3.2× behind at realistic ones**. Quoting the first row
alone would have been dishonest, which is why both are recorded.

The diagnostic that mattered: TTFT was exactly `prompt_len × step_time` —
predicted 256 × 4.6 ms = 1.18 s, measured 1.172 s. That identified the missing
feature precisely rather than by intuition.

**TPOT also crosses over**, ahead at short context and behind at long. At short
context the step is dominated by framework dispatch, where a C++ loop beats
vLLM's per-step Python; at long context the attention kernel dominates, where
scalar fp16 loads lose to FlashAttention. A win in one regime does not survive
into the other.

**This reordered the roadmap.** Speculative decoding was planned next; the
measurement said chunked prefill, with a number attached. Speculative decoding
also targets TPOT, which at 256-token prompts was the *smallest* gap (0.84×)
while TTFT sat at 0.10×.

---

## 7. Stage 3 — chunked prefill

**Goal:** stop spending one forward pass per prompt token.

### The varlen attention kernel (`paged_attention_varlen.cuh`)

Generalizes attention from "one query token per sequence" to "any number, varying
per sequence". The same change covers three cases with no branching:

| | query tokens per sequence |
|---|---|
| decode | 1 |
| prefill chunk | N |
| speculative verification | K |

Query tokens from all sequences are concatenated into `q = [total_q, Hq, D]`, and
`blockIdx.y` is a **global** query index `g`. Two host-built arrays carry what `g`
alone cannot say:

```
q_seq[g]   which sequence -> which block_table row
q_pos[g]   ABSOLUTE position in that sequence
```

**The mask is a loop bound, not a branch.** Causality says the query at absolute
position `p` sees keys `0..p`, so the block scans exactly `q_pos[g] + 1` keys and
never computes a score it would discard.

Note what that removes: **`lens[]` disappears entirely.** It existed only to say
how many keys to scan, and `q_pos[g] + 1` says it per query token rather than per
sequence — strictly more precise. Decode is the case `q_pos[g] == len − 1`, which
scans the whole cache, exactly as before. The empty-slot guard also vanishes,
because `n_keys = p + 1 ≥ 1` always.

Flat arrays rather than FlashAttention's `cu_seqlens` prefix sum: `total_q` ints
are already being built on the host each step, and O(1) indexing is one less
thing to get wrong while the mask is unverified.

**Ordering requirement:** query token `p` attends to key `p` — itself. So
`write_kv_paged_varlen` must store K/V for *every* query token in the batch
before attention launches.

### Multi-token block append (`block_allocator.h`)

```
new_blocks = blocks_for(cur_len + n) − blocks_for(cur_len)
```

**Not** `ceil(n / block_size)` — the partially-filled last block absorbs tokens
for free. At `cur_len = 20` you hold 2 blocks (32 slots) with 12 spare, so n = 4
costs **zero** new blocks while `ceil(4/16)` says 1. The single-token
`append_token` encodes the same rule as `cur_len % block_size == 0`.

`append_tokens` is **all-or-nothing**: if the pool cannot supply every block,
allocate none. A partial append leaves tokens with nowhere to live, the kernel
reads −1 from the table, Phase 1 turns that into `-FLT_MAX`, and the sequence is
silently corrupted instead of failing loudly.

Sizing is measured against **what the sequence actually holds**
(`it->second.size()`), not what `cur_len` implies. Those differ whenever blocks
were reserved ahead of use — e.g. a prompt allocated eagerly at admission then
consumed token by token — so this is correct in both regimes.

`fits_ever(n)` guards against livelock. Lazy per-chunk allocation loses the
natural check eager allocation had for free: an oversized prompt would be
admitted, prefill until blocks run out, be preempted, requeued, admitted again,
forever. A prompt failing `fits_ever` is rejected outright as a client error.

### The chunk scheduler (`chunk_scheduler.h`)

No CUDA. Input: running sequences with current and prompt lengths, plus a token
budget. Output: a per-sequence token count summing to at most the budget.

Four rules, each with a reason:

**1. Decodes first, one token each, in a full pass before any prefill.** A
generating sequence needs exactly 1 token and its user is watching text stream; a
prefilling sequence needs many but its user sees nothing either way. Decode is
latency-critical and cheap, prefill is elastic and expensive. Walking the list
once and handling each sequence as encountered would let a 236-token prompt at
position 0 eat the budget the decoders behind it needed.

Decode tokens are granted **unconditionally**, with no block check — starving a
mid-stream decode is worse than a long step. But the block *is* counted, so the
prefill pass does not spend it twice.

**2. Prefill in FCFS order, not split evenly.** With 492 tokens spare and three
sequences each needing 256: serving them in order finishes one this step (TTFT 1,
2, 2 → mean 1.67); splitting evenly finishes none until the next (2, 2, 2 → mean
2.0). Both are work-conserving, so the *last* sequence finishes at the same time
either way — reordering cannot improve the makespan, only move completions
earlier. Processor sharing maximizes mean completion time among work-conserving
policies, so splitting evenly is strictly worse.

**3. Cap each sequence's chunk.** Pure FCFS lets a 4096-token prompt block a
16-token one for eight steps. The cap bounds that without paying rule 2's
mean-latency tax. (Measured: at a 2048 budget the cap never binds. Correct in
principle, irrelevant at that operating point.)

**4. Respect block availability, by shrinking rather than skipping.** A chunk
that cannot be backed is not a plan, it is a preemption waiting to happen. The
largest affordable chunk is computed in O(1) by inverting the block arithmetic:
with `avail` blocks spare a sequence can reach `(have + avail) × block_size`
total length, hence that many more tokens. Shrinking is what makes degradation
gradual instead of a cliff.

Crucially, **blocks committed by earlier entries are tracked**. Two sequences
each asking for 256 tokens against a 20-block pool must get 256 and 64, not 256
and 256 — testing each against the same `num_free()` would promise 32 blocks and
force a preemption the scheduler should have avoided.

Planning and committing are separate: `plan()` takes `const BlockAllocator&` and
only reads it. The caller applies the plan with `append_tokens` and handles
failure by preempting. So the policy is testable with no pool and rejectable with
no rollback.

### Mixed-batch forward pass (`varlen_layer.h`)

Three things change beyond the attention kernel:

1. **RoPE takes absolute positions.** `launch_rope` already took a per-row
   position array, so passing `q_pos` works unchanged — but passing
   chunk-relative indices would produce fluent-and-wrong text.
2. **`write_kv_paged_varlen`** stores n tokens per sequence at their own
   positions, one thread block per `(query token, kv_head)`.
3. **`lm_head` runs on the last query token of each sequence only.** A prefill
   chunk's interior tokens exist to populate the KV cache and predict nothing we
   keep. Running `lm_head` over all 512 rows would be a 512 × 128256 GEMM
   producing 262 MB of fp16 logits to discard 511/512 of it, so `gather_rows`
   compacts the final hidden states to `[n_out, hidden]` first. Gathering *after*
   the final norm keeps the norm a single launch over contiguous memory.

A sequence still mid-prefill belongs in neither list — it gets more prompt next
step. `serve_chunked.cpp` decides this as `len + grant >= prompt_len`.

### The engine loop (`serve_chunked.cpp`)

```
DRAIN    take newly-arrived requests off the HTTP queue
ADMIT    waiting -> running, lazily (no blocks reserved for the whole prompt),
         rejecting prompts that fail fits_ever
PREEMPT  guarantee the decoders can grow; evict newest-first if not
PLAN     ChunkScheduler
COMMIT   append_tokens for every grant  (affordable by construction)
BUILD    flat q_seq / q_pos / token-id arrays + the block table
FORWARD  one varlen pass over total_q tokens
SAMPLE   logits only for sequences whose chunk finished their prompt
RETIRE   EOS, token limit, or client hung up
```

**PREEMPT runs before PLAN**, which is not obvious. The scheduler grants decode
tokens unconditionally but counts their blocks; if the pool cannot back the
decoders at all, no plan is affordable. Freeing space first is what makes every
`append_tokens` in COMMIT succeed — so a failure there is logged as a `BUG:`,
because it means the scheduler and allocator disagree.

`q_seq` stores the **index in `running`**, not the sequence id, because it indexes
a row of the flattened table and `flatten()` is built from `running` order.

Preempting a half-prefilled sequence needs no special case: `len = 0` discards
partial prefill and restarts the prompt, the same recompute policy. `len` tracks
prefill and generation progress uniformly, which is why this falls out.

**Results at 256-token prompts:** TTFT 1,172 → 240 ms (4.9×), throughput
1,294 → 3,261 tok/s (2.5×), TPOT 6.64 → 5.97 ms. Tuned for latency
(`--max-admits-per-step 2`): **96.8 ms TTFT p50, better than vLLM's 123**.

### What the configuration sweep showed

Five configurations, same model and pool, only scheduler flags differing:

| budget / chunk / admits | tok/s | TTFT p50 | TTFT p95 | TPOT |
|---|---:|---:|---:|---:|
| 512 / 512 / ∞ | 3,159 | 123 | 287 | 7.96 |
| **2048 / 512 / ∞** *(default)* | **3,261** | 240 | **253** | **5.97** |
| 2048 / 128 / ∞ | 3,202 | 246 | 251 | 6.13 |
| 2048 / 512 / 2 | 3,105 | **96.8** | 297 | 8.26 |

**Throughput varied 5% across every configuration.** Scheduler policy
redistributes latency; it does not change how much work the GPU gets through.
That localizes the residual gap to vLLM as *kernel* work and makes further
scheduler tuning provably pointless.

Two predictions failed, informatively:

- **A bigger budget made p50 TTFT worse** (123 → 240 ms) while tightening the
  tail. Admitting ~8 prompts per step means they all prefill together and finish
  together; a smaller budget staggers completions. Rule 2's processor-sharing
  result, reappearing across requests instead of within a step.
- **Budget and admission rate are not independent.** Capping admissions gave the
  best p50 and the *worst* TPOT: mass admission finishes prefill in ~4 steps and
  every step afterwards is pure decode, short and uniform, while capped admission
  *spreads* prefill across many more steps. Staggering doesn't reduce prefill
  interference, it stretches it out. The real axis is concentrated versus spread
  interference, and it is an SLO choice.

---

## 8. Stage 4 — profiling, and the decision not to optimize

Investigated as the measurement indicated, then stopped deliberately. Full detail
in [benchmarks/STAGE4_FINDINGS.md](benchmarks/STAGE4_FINDINGS.md).

- Attention is **47.4%** of a decode step (gemm 36.9%, elementwise 7.5%,
  lm_head 8.1%)
- **11.6% load efficiency** — 3.7 useful bytes per 32-byte sector. Verified by
  hand: the launch should need 2.1 M sectors and issued 18.5 M, an 8.8×
  overfetch that inverts to 11.4%.
- **Not** bandwidth-bound (15–19% of peak on every shape) and **not**
  compute-bound (20% SM throughput). Latency-bound.
- Cause: Phase 1 assigns one thread per token, so adjacent lanes read addresses
  128 bytes apart. The loop *looks* strided and is still uncoalesced — the stride
  is over the wrong axis. Phase 3 is already coalesced, which is why the
  aggregate is 11.6% rather than ~6%.
- Amdahl ceiling: 2× faster attention → 31% faster step → ~4,200 tok/s, vLLM's
  number.

The fix — cooperative coalesced load of a K tile into shared memory, then
thread-per-token math out of shared — is specified in the findings doc and left
unimplemented. The remaining work was implementation grind; the concepts
(profile before optimizing, compute the ceiling first, and why tiling *forces*
online softmax) are recorded.

Two methodology rules came out of it:

1. **Report min of N trials, not a mean.** The same binary measured 113 µs idle
   and 179 µs with a game open. Minima reproduce across sessions to ~1%; medians
   drift 20–30% within one run from thermal droop.
2. **Verify the binary changed before explaining a result.** An entire mechanism
   was theorized for a slowdown that turned out to be an unrebuilt kernel plus a
   background game. The launch configuration was in the profiler output the whole
   time.

---

## 9. How correctness was established

The methodology mattered more than any single test.

**Layer-by-layer, not end-to-end.** `dump_reference.py` saves HF's hidden states
after every layer; `verify_decode_path.cu` compares ours at each one. An
end-to-end check tells you *that* something is wrong; this tells you *where*.
Expected ~0.005 relative error from fp16. The error curve's **shape** matters
too: flat means rounding noise, monotonically growing means drift.

**Invariance tests, with one exact gate.** In the Stage 2A spike:

| test | question | result |
|---|---|---|
| A | slot 1 batched vs alone | 0.234 |
| B | padding poisoned +1e4 vs −1e4 | **0.000000** |
| C | slot 1 at past=500 vs past=30 | 0.234 |

B is the correctness gate and it is *exact*: flipping 470 positions of masked
memory changed nothing, so padding provably never reaches the output. A and C
being identical showed batch size doesn't affect numerics at all — only reduction
length does. 0.234 on a signal of max 14.66 is 1.6%, i.e. fp16 rounding from a
501-term softmax versus a 31-term one.

**"Numerically different" is not "incorrect."** A and C measure batch-composition
reproducibility, a real property of every batched server including vLLM, not
correctness.

**Self-consistency tests need no reference model.** Three were decisive:

- **Shuffled block placement** — the same logical sequence stored in scattered
  physical blocks must give a bit-identical answer. Exact zero required. If it
  fails, the indirection is wrong.
- **Varlen versus decode** — varlen with `q_len = N` must equal N separate decode
  calls. A decode call at length L *is* "attend to keys 0..L−1", so matching it
  proves the mask by construction, and bugs common to both kernels cancel out.
- **Chunk-size invariance** — the same prompt served with `--max-chunk 512` (one
  pass) and `--max-chunk 2` (three passes) must produce token-identical greedy
  output. Catches any off-by-one in `q_pos` or the mask at a chunk boundary, with
  no reference model at all.

**Diagnose by error magnitude.** When a layer showed 0.129711, the number itself
identified the class: too large for fp16 rounding, too small for a completely
broken layer. That pointed at an off-by-one rather than a wrong formula.

**Unit tests for the no-CUDA components.** `block_allocator` and
`chunk_scheduler` have ~50 host assertions between them, including the one that
matters most — that a plan is *affordable*, i.e. applying it with `append_tokens`
never fails.

---

## 10. Bugs that taught something

| bug | symptom | lesson |
|---|---|---|
| KV buffer strided by `max_seq`, TRT reads stride `past` | fluent, wrong output | A layout contract that changes every step should be structural (ping-pong), not maintained by hand |
| Pre-tokenizer treated as a classifier, not ordered alternation | fluent text, logits off by KL 0.52 | `1969` → `196`+`9`; rule order is load-bearing |
| `lens` excluded the current token | layer00 rel err **0.129711** | Query `p` attends to key `p` — *itself*. Error magnitude identified the bug class |
| Phase 3 derived the address per token | paged attention **2.3× slower** than what it replaced | ~3200 integer divisions per block. Only the benchmark revealed it |
| `num_blocks` (pool size) conflated with `max_blocks` (table width) | would read the wrong pool region for every layer past 0 | Equal in the single-sequence harness, thousands vs ~128 in the server |
| Preemption absent | deadlock at 64 MB, "out of blocks" at 128/256 MB | A running sequence cannot be failed halfway |
| `out[b]` instead of `out[g]` in the varlen kernel | all of a sequence's query tokens race on one address | Invisible at `q_len == 1`; needed the ragged test to surface |
| Test expectations written from intuition | asserted `ceil(12/16) = 1` new block; the answer is **0** | 35+12 = 47 fits in 3 blocks. Derive expectations from the rule, not the gut |
| `getattr(cfg, "rope_theta", 10000.0)` | every manifest recorded 10000 instead of 500000 | Harmless only because `model_config.h` reads `config.json`. Silent defaults hide |
| Benchmark run with a game open | 113 µs → 179 µs, same binary | Min of N, and check the launch config before theorizing |

---

## 11. Current state

| | |
|---|---|
| Stage 1 | ✅ 41k tok/s prefill, 235 tok/s decode, cosine 1.0000 vs HF |
| Stage 2A | ✅ 947 tok/s, 24/24 identical to sequential |
| Stage 2B | ✅ 15,090 tok/s, 128 concurrent in 512 MB, own forward pass |
| HTTP serving | ✅ OpenAI-compatible SSE, cancellation |
| vs vLLM 0.11 | ✅ parity at short prompts, 0.78× throughput at 256-token prompts |
| Stage 3 | ✅ TTFT 1,172 → 240 ms, 1,294 → 3,261 tok/s |
| Stage 4 | profiled, scoped, deliberately not implemented |
| Stage 5 | in progress — model prep done |

**Stage 5 (speculative decoding) status.** The target was switched from 1B to
**Llama-3.2-3B** so the existing 1B can serve as the draft model — a draft must
share the target's tokenizer, and nothing smaller than the 1B exists with
LLaMA-3's 128,256-token vocabulary. 3B weights exported (7.21 GB, 28 layers,
hidden 3072, 24Q/8KV heads × 128) and **verified layer-by-layer against HF: rel
err 0.005–0.008, flat across all 28 layers, argmax matches.** `head_dim = 128`
and the non-power-of-two GQA ratio of 3 both worked without kernel changes.

Remaining:

- `BlockAllocator::truncate(seq, new_len)` — rollback. The first operation that
  makes a sequence **shrink**; everything so far only grows, which is why
  `append_tokens` may assume `have >= blocks_for(cur_len)`.
- Acceptance sampling — accept draft token `x` with probability
  `min(1, p_target(x)/p_draft(x))`, resample from `max(0, p_target − p_draft)` on
  rejection. This makes the output distribution **provably identical** to plain
  sampling; speculative decoding is exact, not an approximation.
- Two KV pools, one block table. The logical→physical mapping is
  model-independent, so a single `BlockAllocator` serves both models; only the
  per-block byte stride differs.
- Benchmark at concurrency **1 and 4**, not 32. Speculative decoding spends idle
  arithmetic capacity, and continuous batching has already consumed it at high
  concurrency — measuring at 32 would correctly implement the feature and record
  a regression.

**Memory is the live constraint:** 7.21 GB (3B) + 3.00 GB (1B) = 10.2 GB of
weights on a 16 GB card, leaving ~3–4 GB for two KV pools and scratch. An easy
788 MB is recoverable — LLaMA-3.2 ties `lm_head` to the embedding table and the
exporter writes both.

**Known limitations**, stated rather than discovered:

- RoPE is plain; LLaMA-3.1/3.2 apply frequency-dependent scaling
  (`rope_type: llama3`, factor 32). The two agree at short positions because the
  scaled dimensions have wavelengths in the millions and barely rotate, which is
  why verification at ~50 tokens passes. They diverge somewhere in the low
  thousands — so short-context verification *cannot* catch this.
- `/v1/chat/completions` is not implemented (needs the chat template).
- Pre-tokenizer is an ASCII approximation (full Unicode needs PCRE2/ICU);
  verified 8/8 on test prompts.
- Stage 2A's throughput collapse above B=16 was never root-caused.
- Every benchmark is a single run, min-of-5 only for the kernel microbenchmarks.
