# Stage 1 — Single-request engine

**Model** LLaMA-3.2-1B-Instruct, FP16, TensorRT 10.14 (one engine, two optimization profiles)
**GPU** NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120) · **Host** WSL2 / Ubuntu · CUDA 13

Reproduce: `python tools/bench.py` (perf) · `tools/verify_logits.py`,
`tools/verify_tokenizer.py` (correctness).

---

## Correctness

Verified at three levels, weakest gate last — because comparing *generated text*
against HuggingFace is a bad primary gate for an fp16 engine: a single near-tie
flips one token and the two sequences diverge into equally-valid continuations
forever. That measures chaos, not correctness.

### 1. Tokenizer — 8/8 exact (`verify_tokenizer.py`)

Token ids identical to HF `tokenizers` across prose, code, digits, currency,
contractions and em-dashes.

### 2. Logits — 5/5 top-1, cosine 1.0000 (`verify_logits.py`)

One forward pass, no accumulated drift:

| Prompt | top-1 | top-5 | cosine | max diff | KL(hf‖trt) |
|---|:--:|:--:|---:|---:|---:|
| The key insight about transformers is | yes | 5/5 | 1.0000 | 0.074 | 0.0001 |
| Once upon a time | yes | 5/5 | 1.0000 | 0.078 | 0.0001 |
| The capital of France is | yes | 5/5 | 1.0000 | 0.078 | 0.0001 |
| def fibonacci(n): | yes | 5/5 | 1.0000 | 0.082 | 0.0001 |
| In 1969, humans first | yes | 4/5 | 1.0000 | 0.062 | 0.0003 |

### 3. Generated text — 2/5 exact over 48 greedy tokens

Expected, and not a defect. The three non-matching prompts reproduce HF exactly
for 10–15 tokens, then split at a point where both continuations are fluent and
sensible. At KL ≈ 1e-4 per step, a near-tie eventually flips; greedy decoding
never reconverges afterwards.

### The bug this process caught

Two prompts initially showed cosine 0.976 / 0.995 — ~5000× the KL of the others.
Too large for fp16 rounding, so it was a real defect: the hand-rolled
pre-tokenizer approximated LLaMA-3's regex instead of implementing its ordered
alternation.

```
def fibonacci(n):     ours: '(' 'n'      HF: '(n'
In 1969, humans...    ours: '19' '69'    HF: '196' '9'
```

Rewriting `pretokenize()` to follow the alternation in order fixed both:

- rule 2 (`[^\r\n\p{L}\p{N}]?\p{L}+`) is tried **before** rule 4, which is why
  `(n` is one chunk rather than `(` + `n`
- rule 3 (`\p{N}{1,3}`) caps digit runs at three, greedy left-to-right, which is
  why `1969` splits `196`+`9` and not `19`+`69`

After the fix: 8/8 tokenizer, cosine 1.0000 on all prompts, and `In 1969, humans
first` now matches HF for all 48 generated tokens.

Remaining limitation: `\p{L}`/`\p{N}` are approximated for ASCII (any byte > 127
is treated as a letter). Correct for accented Latin and em-dashes; full Unicode
needs PCRE2 or ICU.

---

## Performance

All runs include 2 untimed warmup cycles. Decode figures are p50 over 64 steps.

### A. Prefill scaling

| Prompt tokens | Prefill (ms) | Prefill (tok/s) | TTFT (ms) |
|---:|---:|---:|---:|
| 8 | 6.08 | 1,317 | 6.9 |
| 32 | 5.92 | 5,404 | 6.7 |
| 128 | 6.34 | 20,187 | 7.1 |
| 512 | 12.47 | **41,062** | 13.5 |
| 1024 | 37.62 | 27,218 | 38.5 |
| 1900 | 185.09 | 10,265 | 186.3 |

Below ~128 tokens prefill is launch-latency bound — 8 and 128 tokens cost nearly
the same wall time. Throughput peaks at 512 and then falls. Part of that is
attention's O(n²) term, but the profile's `opt` shape is also 512, so kernel
selection is tuned for that length and longer prompts run tactics chosen for a
shorter one. Worth re-testing with `opt=1024`.

### B. Decode vs context depth

| Context | p50 (ms/tok) | Decode (tok/s) |
|---:|---:|---:|
| 8 | 4.26 | 234.6 |
| 128 | 4.34 | 230.3 |
| 512 | 4.82 | 207.5 |
| 1024 | 5.32 | 188.0 |

Smooth degradation — decode is memory-bound and each step rescans a longer KV
cache.

### C. Sampling overhead (context 512)

| Mode | p50 (ms/tok) | Decode (tok/s) |
|---|---:|---:|
| greedy (argmax) | 4.77 | 209.8 |
| temperature 0.8 + top-k 50 + top-p 0.95 | 4.85 | 206.3 |

**0.08 ms/token (1.7%)** — a full 128k-entry radix sort per step, and it barely
registers because the step is already memory-bound. An earlier uncontrolled
comparison suggested 13%; the warmup-and-matched-context harness corrected it. A
radix top-k is not worth implementing at this scale.

---

## Design notes

**Two profiles, one engine.** Prefill (seq 1..2048, past=0) and decode (seq=1,
past 0..2047) are separate TRT optimization profiles; the runtime switches with
`setOptimizationProfileAsync` once per phase.

**The KV cache is ping-pong, not spliced — and that was a bug fix, not an
optimization.** TRT reads `past_key_values.i.key` as a *contiguous*
`[1, H, past, D]` tensor: head *h* begins at `h·past·D`. A fixed-capacity buffer
strided by `max_seq` therefore cannot be handed to TRT at all — every head after
head 0 is misread. The symptom was fluent-but-wrong output, which is far more
misleading than a crash. The `present` output is already exactly the layout the
next step needs as `past`, so the two buffers alternate. This is correct *and*
removes 32 `cudaMemcpy2DAsync` calls per step.

**Zero per-step allocation.** All device memory — KV, hidden states, logits,
lm_head, sampler scratch — is allocated at init. The decode loop makes no
`cudaMalloc` calls, and only 4 bytes (the token id) return to the host per step.

**Model-agnostic.** Dimensions load from `config.json`; nothing about the model
is compiled in.

---

## Stage 2 seam

The runtime touches the KV cache through five methods only —
`fits / bind / commit / length / reset`. A paged implementation (fixed-size
blocks, free list, per-sequence block table) replaces the body of `KVCache`
without the engine loop changing; `bind()` becomes where a block table is
published instead of raw pointers.
