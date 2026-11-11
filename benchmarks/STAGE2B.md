# Stage 2B — Paged KV cache and a custom CUDA decode path

**Model** LLaMA-3.2-1B-Instruct, FP16 · **GPU** RTX 5070 Ti (sm_120) · WSL2, CUDA 13

TensorRT is not used in this path. The forward pass is seven cuBLAS GEMMs per
layer plus hand-written kernels; KV memory comes from a block allocator; attention
reads through a block table.

Reproduce:

```bash
g++  -std=c++17 -I src tools/test_block_allocator.cpp -o build/test_alloc
nvcc -std=c++17 -I kernels -I src tools/test_paged_attention.cu  -o build/test_paged  --extended-lambda -arch=sm_120
nvcc -std=c++17 -I kernels          tools/test_layer_kernels.cu  -o build/test_layers -arch=sm_120
nvcc -std=c++17 -I src -I kernels   tools/verify_decode_path.cu  -o build/verify_decode -lcublas --extended-lambda -arch=sm_120
nvcc -std=c++17 -I kernels -I src tools/bench_paged_attention.cu -o build/bench_paged --extended-lambda -arch=sm_120 -O3
cmake --build build -j                       # → build/paged_runtime
```

---

## 1. Why: what Stage 2A left on the table

Stage 2A gave every batch slot a full `max_seq` window. On a measured workload
(prompts ~8 tokens, 48 generated) that meant **~55 real tokens in a 512-token
reservation — 11% utilization**, and attention scanned all 512 positions per slot
regardless. Throughput efficiency fell from 81% at batch 4 to **32% at batch 16**.

Paging allocates 16-token blocks on demand instead.

| | fixed slots (2A) | paged (2B) |
|---|---:|---:|
| reservation per request | 512 tokens | ⌈len/16⌉ × 16 |
| waste per request | ~457 tokens | ≤ 15 tokens |
| utilization (55-token request) | 10.7% | **85.9%** |

---

## 2. Correctness

### Block allocator — 10/10 (`test_alloc`)

Ceiling division, growth only at block boundaries, free-list accounting,
watermark behaviour, exhaustion returning `false` rather than throwing, `locate()`
translation, and the flattened GPU table.

One test is about the property that motivates uniform block size: allocate 8
blocks, free 3 *non-adjacent* ones, then successfully allocate a 3-block request.
**Fixed-size blocks have no external fragmentation** — any free block satisfies any
need, so N free blocks are always fully usable.

### Layer kernels — 5/5 (`test_layers`)

RMSNorm, RoPE, SiLU-mul, embedding, residual add, each against a
double-precision CPU reference. Max |diff| ≤ 0.0018; embedding is exact.

### Paged attention — 3/3 (`test_paged`)

| test | question | result |
|---|---|---|
| vs CPU reference | is the maths right | max diff 0.0003 |
| shuffled physical blocks | does output depend on *where* blocks live | **0.00000000** |
| ragged batch isolation | do neighbours contaminate each other | **0.00000000** |

Test 2 is the one that matters: the same logical sequences stored in *different*
physical blocks must produce bit-identical output. That is the property paging
depends on, and the only bug this kernel can really have.

### Decode path vs HuggingFace — all 16 layers (`verify_decode`)

Prompt fed one token at a time (what a decode step does), hidden state compared
after every layer:

| after | rel err | | after | rel err |
|---|---:|---|---|---:|
| layer00 | 0.0040 | | layer08 | 0.0063 |
| layer01 | 0.0053 | | layer09 | 0.0056 |
| layer02 | 0.0066 | | layer10 | 0.0055 |
| layer03 | 0.0062 | | layer11 | 0.0049 |
| layer04 | 0.0065 | | layer12 | 0.0048 |
| layer05 | 0.0074 | | layer13 | 0.0049 |
| layer06 | 0.0072 | | layer14 | 0.0051 |
| layer07 | 0.0070 | | final_norm | 0.0053 |

logits rel err 0.0078, **argmax matches**.

Error **peaks at layer 5 and then decreases** — it does not compound. The residual
stream plus RMSNorm rescales away accumulated drift each layer, which is why fp16
inference is stable at depth.

---

## 3. Paged vs contiguous attention (kernel only)

`bench_paged` runs both kernels on identical ragged workloads. Same arithmetic;
the differences are addressing and how many tokens get scanned. Physical block
placement is fragmented (from the real allocator), i.e. the pessimistic case.

max_seq = 512:

| B | mean len | paged µs | fixed µs | speedup | paged MB | fixed MB | saved |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 85.5 | 15.5 | 15.4 | 0.99× | 0.8 | 4.2 | 5.6× |
| 8 | 51.2 | 16.9 | 15.8 | 0.93× | 1.0 | 8.4 | 8.5× |
| 16 | 42.6 | 19.9 | 19.1 | 0.96× | 1.7 | 16.8 | 10.0× |
| 32 | 54.2 | 31.3 | 35.6 | 1.14× | 4.1 | 33.6 | 8.2× |
| 64 | 60.8 | 71.3 | 71.8 | 1.01× | 8.9 | 67.1 | 7.6× |

At max_seq = 2048 the memory saving reaches **22–40×**; latency is unchanged.

**Paged attention is at parity per call, not faster.** It scans ~10× fewer tokens
but reads them in 16-token fragments instead of one long stream, so it is
latency-bound where the contiguous version is bandwidth-bound. The two roughly
cancel. This matches vLLM's own result — the paper's win is capacity, not kernel
speed. Anyone claiming paged attention is "faster per request" has the mechanism
wrong.

### The optimization the benchmark forced

The first implementation was **2.3× slower** than contiguous (0.43× at B=4).
Cause: Phase 3 assigns thread *d* to output dimension *d*, so every one of the D
threads walked every token and repeated the same `t / BS`, `t % BS`, and table
lookup — ~D·len = 3200 integer divisions per block at D=64, len=50, at ~20 cycles
each.

Restructuring Phase 3 **block-major** (outer loop over logical blocks, inner over
offsets within them) removes the division entirely and reads the table once per 16
tokens: **0.43× → 0.79×**, then parity after caching the table row in shared
memory. Thread-count sweep afterwards (128/64/32) moved nothing beyond noise, so
tuning stopped there.

The general lesson: *the indirection can cost more than the work it saves, and only
a benchmark tells you.*

---

## 4. Concurrency at a fixed KV budget

`paged_runtime`, 200 requests, 48 tokens each, `--max-batch 128`, block size 16.
Goodput = 9,600 useful tokens ÷ wall time; raw tok/s counts tokens regenerated
after preemption.

| KV budget | blocks | peak batch | raw tok/s | **goodput** | TTFT p50 | TTFT p95 | preemptions |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 MB | 122 | 61 | 5,720 | **3,895** | 1246 ms | 2025 ms | 743 |
| 128 MB | 244 | 122 | 8,802 | **6,163** | 692 ms | 1068 ms | 304 |
| 256 MB | 488 | 128 | 11,133 | **9,731** | 416 ms | 416 ms | 47 |
| 512 MB | 976 | 128 | 13,890 | **13,890** | 109 ms | 420 ms | 0 |
| 1024 MB | 1953 | 128 | 15,090 | **15,090** | 83 ms | 392 ms | 0 |

**vs Stage 2A's best of 947 tok/s: 15.9×**, with TTFT p50 524 → 83 ms.

Three things this curve shows:

- **The knee is 512 MB** — the first budget with zero preemptions. Below it,
  throughput is traded for memory at a worsening rate; above it, 2× the memory
  buys 8%.
- **Raw throughput overstates under pressure.** At 64 MB the run reports 14,097
  tokens but only 9,600 are useful — 32% is recompute after preemption. Goodput is
  the honest number.
- **Degradation is graceful.** At 64 MB, Stage 2A could not fit *two* requests
  (33.5 MB each) and would simply fail. Paged serves all 200, slowly and with
  thrashing, but it completes.

Peak batch reaches the `--max-batch 128` cap at ≥256 MB, not a memory limit — the
976-block pool at 512 MB had room for ~244 concurrent 56-token sequences.

---

## 5. Design notes

**The block allocator has no CUDA in it.** Allocation runs about once per 16
tokens over a few dozen integers — free lists, hash maps, branching, all things
CPUs are good at. Translation runs every step over every cached token, on the GPU,
through the flattened table. Same split as an OS: the kernel manages page tables in
software, the MMU translates in hardware.

**Prefill and decode are the same operation here.** Stage 2A needed a separate
batch-1 prefill enqueue because the TRT batch shares one `seq` dimension. In this
path every sequence advances exactly one token per step, so a sequence still
reading its prompt just takes its input from the prompt instead of the sampler.
Consequence: **no admission stall** — but a P-token prompt costs P steps rather than
one enqueue. Chunked prefill (many prompt tokens per step, varlen attention) is the
standard fix and is not implemented.

**Admission on prompt length alone is insufficient.** A request admitted for its
8-token prompt grows to 56. Without preemption, runs at 128 and 256 MB failed
mid-generation with "out of blocks", and a 128-block watermark deadlocked the 64 MB
run outright. Two fixes exist: reserve `prompt + max_new` up front (never fails,
wastes capacity since most requests finish early), or admit optimistically and
evict on failure. This uses the second.

**Preemption: newest-first, recompute.** The victim is the most recently admitted
running sequence — least work invested, so the least recompute is discarded. Its
blocks return to the pool and it is requeued at the *back* of the admission order;
requeueing it as next-to-admit thrashes (admit, preempt, admit, preempt, no
progress). This is vLLM's *recompute* policy; the alternative, *swap*, copies blocks
to host memory and restores them — less wasted compute, more PCIe traffic.

**Cost vs TensorRT.** At equal batch the custom path is ~16% slower than the TRT
fixed-slot path (499 vs 593 tok/s at B=4) — the price of losing kernel fusion. That
is what buys the ability to page, and 15.9× aggregate throughput at scale.

---

## 6. Bugs worth remembering

| bug | symptom | how it was caught |
|---|---|---|
| Token could not attend to itself (`lens` off by one) | layer-0 rel err 0.13 — too big for fp16, too small for a broken layer | layer-by-layer comparison; the *magnitude* named it before any code was read |
| Per-token integer division in Phase 3 | paged 2.3× slower than the fixed-slot kernel it replaced | benchmark, not the test suite |
| `num_blocks` vs `max_blocks` conflated in the pool offset | every layer past 0 would read the wrong region | integration — the two are *equal* in the single-sequence verify harness |
| Admission ignoring future growth | "out of blocks" mid-run at 128/256 MB | the budget sweep |

The first and third are the instructive ones. A bug that makes output *fluent but
wrong* is far more dangerous than one that crashes, and two names that mean
different things but coincide in a test will survive a green suite until
integration.
