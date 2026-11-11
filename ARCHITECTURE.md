# llm_server — repo architecture

A map of what is in this repo and why. For the narrative of how each piece was
built, see [IMPLEMENTATION.md](IMPLEMENTATION.md).

**The first thing to know:** there are **five executables**, and only one is
current. The other four are the rungs that were climbed to get there, kept
deliberately as correctness oracles and performance baselines — not dead code,
but not the thing you run either.

---

## 1. The five executables

| target | stage | engine | KV cache | status |
|---|---|---|---|---|
| `runtime` | 1 | TensorRT | ping-pong, 1 sequence | historical |
| `batch_runtime` | 2A | TensorRT | N fixed slots | historical, **baseline** |
| `paged_runtime` | 2B | own CUDA | paged blocks | historical, **oracle** |
| `serve` | 2B + HTTP | own CUDA | paged blocks | historical, **baseline** |
| **`serve_chunked`** | **3** | **own CUDA** | **paged blocks** | **← run this** |

Why each is kept:

- **`runtime`** — the only path that still uses TensorRT end to end. Reference
  for "what does a framework achieve on this hardware": 41k tok/s prefill,
  235 tok/s decode.
- **`batch_runtime`** — produced the 947 tok/s figure that Stage 2B's 15.9× is
  measured against. Also the source of the 24/24 "batched output identical to
  sequential" correctness result.
- **`paged_runtime`** — no HTTP, so it isolates engine performance from socket
  overhead. The 15,090 tok/s headline came from here.
- **`serve`** — the pre-chunked-prefill server. Its 1,294 tok/s / 1,172 ms TTFT
  at 256-token prompts is the before-number for Stage 3's 4.9× TTFT improvement.
  Delete it and the Stage 3 claim loses its baseline.
- **`serve_chunked`** — current. Everything new goes here.

```bash
./build/serve_chunked --weights weights_3b --tokenizer onnx_3b/tokenizer.json \
  --config onnx_3b/config.json --kv-budget-mb 512 --max-batch 8 \
  --max-batched-tokens 2048 --max-chunk 512
```

---

## 2. The architectural line: host policy vs device math

Three headers contain **no CUDA at all**, and that is the main structural
decision in the repo:

| file | what it decides |
|---|---|
| `scheduler.h` | which waiting request becomes running (Stage 2A) |
| `block_allocator.h` | which physical block backs which logical position |
| `chunk_scheduler.h` | how many tokens each sequence gets this step |

These are the parts an interviewer asks about, and they are the parts that are
hard to debug on a GPU. Keeping them host-only means ~50 unit assertions run in
milliseconds with no device — `tools/test_block_allocator.cpp` and
`tools/test_chunk_scheduler.cpp` compile with plain `g++`.

It also mirrors vLLM's split between `scheduler.py` and the model runner, and it
reflects a real division of labour: allocation runs about once per 16 tokens over
a few dozen ints (free lists, hash maps, branching — CPU work), while translation
runs every step over every cached token (GPU work).

**What crosses the boundary** each step is four flat int arrays plus one tensor
back:

```
host -> device    d_ids      [total_q]   token id per query token
                  d_q_seq    [total_q]   which sequence  -> block-table row
                  d_q_pos    [total_q]   absolute position  -> RoPE + mask
                  d_out_rows [n_out]     which rows need logits
                  d_table    [n_seqs, max_blocks]   flattened, -1 padded
device -> host    logits     [n_out, vocab]
```

That's the whole interface. Note `n_out ≤ n_seqs ≤ total_q` — only the last query
token of a sequence whose chunk finished its prompt gets logits.

---

## 3. `src/` — file by file

### Current path (Stage 3)

| file | role | why it exists |
|---|---|---|
| `serve_chunked.cpp` | the engine loop | DRAIN → ADMIT → PREEMPT → PLAN → COMMIT → BUILD → FORWARD → SAMPLE → RETIRE. Owns the only CUDA stream |
| `chunk_scheduler.h` | per-step token budget | decodes first, prefill FCFS, per-sequence cap, block-aware. No CUDA |
| `varlen_layer.h` | mixed-batch forward pass | one decoder layer over prefill chunks *and* decode tokens together |
| `block_allocator.h` | paged KV bookkeeping | free list, per-sequence block tables, multi-token append, `fits_ever` |

### Shared infrastructure

| file | role | notes |
|---|---|---|
| `http.h` | HTTP/1.1 + SSE | no dependencies. `Format::OpenAI` and `Format::Simple` |
| `tokenizer.h` | byte-level BPE | LLaMA-3 ordered-regex pre-tokenizer; the subtle part |
| `model_config.h` | dims from `config.json` | **this is what the runtime reads** — not the manifest |
| `weights.h` | raw fp16 tensor loader | loads by filename convention with per-file size checks |

### Historical, kept as baselines

| file | belongs to | superseded by |
|---|---|---|
| `runtime.cpp`, `kv_cache.h` | Stage 1, TensorRT | — |
| `batch_runtime.cpp`, `batch_kv_cache.h`, `scheduler.h` | Stage 2A | `block_allocator.h` + `chunk_scheduler.h` |
| `paged_runtime.cpp` | Stage 2B | `serve_chunked.cpp` |
| `decode_layer.h` | Stage 2B | `varlen_layer.h` (which still `#include`s it for `gemm`, `DecodeScratch`) |
| `serve.cpp` | Stage 2.4 | `serve_chunked.cpp` |

`decode_layer.h` is the one with a live dependency — `varlen_layer.h` reuses its
`gemm()` helper and `DecodeScratch` struct, so it isn't purely historical.

---

## 4. `kernels/`

| file | grid | notes |
|---|---|---|
| `paged_attention_varlen.cuh` | `(Hq, total_q)` | **current.** Mask is a loop bound (`n_keys = q_pos[g] + 1`); `lens[]` doesn't exist |
| `paged_attention.cuh` | `(Hq, B)` | Stage 2B decode-only. Still the **oracle** for the varlen tests |
| `layer_kernels.cuh` | varies | rmsnorm (warp shuffle), rope (`i` pairs with `i + D/2`), silu_mul, embedding, residual_add |
| `batched_pick.cuh` | — | `cub::DeviceSegmentedReduce::ArgMax`, one launch for B sequences |
| `sampling.cuh` | — | temperature / top-k / top-p on device. **Unused by `serve_chunked`**, which is greedy |
| `kv_scatter.cuh` | — | Stage 2A fixed-slot scatter. Historical |
| `_paged_attention_ref.cuh` | — | the original version, kept aside when the kernel was rewritten by hand |

The leading underscore on `_paged_attention_ref.cuh` marks it as "moved aside,
not built".

---

## 5. `tools/` — grouped by purpose

**Model preparation**

| | |
|---|---|
| `export_onnx.py`, `build_engine.py` | the TensorRT path (Stages 1, 2A) |
| `export_weights.py` | raw fp16 tensors for the custom path. Transposes projections to `[in, out]` so every GEMM is N/N |
| `dump_reference.py` | HF hidden states after every layer, for verification |

**Correctness**

| | checks |
|---|---|
| `verify_tokenizer.py` | 8/8 token sequences vs HF |
| `verify_logits.py`, `verify_vs_hf.py` | the TensorRT path |
| `verify_decode_path.cu` | **the custom forward pass, layer by layer.** Takes *positional* args: `weights_dir reference_dir config_dir` |
| `test_block_allocator.cpp` | block arithmetic, all-or-nothing append, `fits_ever` |
| `test_chunk_scheduler.cpp` | the four scheduling rules + plan affordability |
| `test_layer_kernels.cu` | rmsnorm / rope / silu individually |
| `test_paged_attention.cu` | CPU reference, shuffled blocks, ragged batch |
| `test_paged_attention_varlen.cu` | varlen vs N decode calls — exact zero on three cases |

**Performance**

| | measures |
|---|---|
| `bench.py`, `bench_batch.py` | the TensorRT paths |
| `bench_paged_attention.cu` | the decode kernel in isolation |
| `bench_varlen_attention.cu` | the varlen kernel; min-of-5, reports achieved bandwidth vs peak |
| `profile_step.cu` | per-category share of a forward step + Amdahl ceiling |
| `bench_vs_vllm.py` | **server-agnostic** OpenAI-SSE client. Measures llm_server and vLLM with identical code; `--compare` diffs the JSON |

Note the argument-style inconsistency: `verify_decode_path.cu` takes positional
arguments while everything else takes flags. Worth fixing next time that file is
touched.

---

## 6. Data directories (not in git)

| dir | size | what | needed by |
|---|---:|---|---|
| `weights/` | 2.8 G | 1B raw fp16 | the **draft** model for Stage 5 |
| `weights_3b/` | 6.8 G | 3B raw fp16 | the **target** model |
| `onnx/` | 3.5 G | 1B ONNX + tokenizer + config | TensorRT path; `tokenizer.json` and `config.json` also used by the custom path |
| `onnx_3b/` | 8.7 M | 3B tokenizer + config only | `serve_chunked` with the 3B |
| `engine/` | 2.5 G | serialized TRT engine | `runtime`, `batch_runtime` |
| `reference/`, `reference_3b/` | 9 M | HF per-layer hidden states | `verify_decode_path` |

**~6 GB is reclaimable** if you're done with the TensorRT comparison: `onnx/`
minus its `tokenizer.json` and `config.json`, plus `engine/`. Keep those two
small files — the custom path reads them.

The 3B export duplicates an 788 MB tensor: LLaMA-3.2 ties `lm_head` to the
embedding table and `export_weights.py` writes both. Reclaimable by aliasing them
in `weights.h` when the manifest shows identical shapes.

---

## 7. Documents

| | |
|---|---|
| `README.md` | status table + quick start |
| `IMPLEMENTATION.md` | what was built and how, with the bug catalogue and verification methodology |
| `INFERENCE_ENGINE_PLAN.md` | roadmap, and why the order changed when measurements contradicted it |
| `ARCHITECTURE.md` | this file |
| `benchmarks/STAGE{1,2,2B,3}.md` | per-stage results |
| `benchmarks/VS_VLLM.md` | the head-to-head, at two prompt lengths |
| `benchmarks/STAGE4_FINDINGS.md` | attention profiling; the optimization that was scoped and deliberately skipped |
| `benchmarks/*.json` | raw benchmark output, consumed by `bench_vs_vllm.py --compare` |

---

## 8. Reading order, if you're coming back cold

1. `README.md` — where things stand
2. **`block_allocator.h`** — start here. No CUDA, heavily commented, and it
   contains the central idea (logical→physical paging) in ~200 lines
3. `chunk_scheduler.h` — the four scheduling rules, each with its reason
4. `paged_attention_varlen.cuh` — the kernel; the header comment explains the
   mask-as-loop-bound and the block-major Phase 3
5. `serve_chunked.cpp` — how the above are wired into a loop
6. `IMPLEMENTATION.md` §9–10 — the verification methodology and the bug list

If you only re-read one thing before an interview, make it §10 of
IMPLEMENTATION.md. The bugs are where the understanding is.

---

## 9. Build

```bash
cmake -B build && cmake --build build -j
```

Five targets. The CUDA architecture is set in `CMakeLists.txt` by platform:
`sm_120` for the RTX 5070 Ti (Blackwell), `sm_87` for Jetson Orin. TensorRT
include and library paths switch on `CMAKE_SYSTEM_PROCESSOR` too.

The standalone tests aren't CMake targets — they compile directly:

```bash
g++  -std=c++17 -I src tools/test_block_allocator.cpp  -o build/test_alloc
g++  -std=c++17 -I src tools/test_chunk_scheduler.cpp  -o build/test_chunk
nvcc -std=c++17 -I kernels -I src tools/test_paged_attention_varlen.cu \
     -o build/test_varlen --extended-lambda -arch=sm_120
nvcc -std=c++17 -I src -I kernels tools/verify_decode_path.cu \
     -o build/verify_decode -lcublas --extended-lambda -arch=sm_120
```

`--extended-lambda` is required wherever `block_reduce` is used — it takes a
device lambda as its reduction operator.
