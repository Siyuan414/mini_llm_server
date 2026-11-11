# llm_server — an LLM inference server written from scratch

Tokenizer, forward pass, KV cache, scheduler and HTTP front-end in C++/CUDA, with
no inference framework in the hot path. LLaMA-3.2, FP16, one RTX 5070 Ti.

**Benchmarked head-to-head against vLLM 0.11 on identical hardware, model and
workload** (`tools/bench_vs_vllm.py` drives both over the same OpenAI SSE client):

| 256-token prompts, 32 concurrent | llm_server | vLLM 0.11 |
|---|---:|---:|
| throughput | 3,261 tok/s | **4,182 tok/s** |
| TTFT p50 | **96.8 ms** *(tuned)* / 240 ms | 123 ms |
| TPOT p50 | 5.97 ms | **5.61 ms** |

Better time-to-first-token, within 6% on per-token latency, 78% of throughput.
The gap is localized rather than guessed: throughput varied **5% across five
scheduler configurations**, which places it in the attention kernel — profiled at
**11.6% memory load efficiency**, with the fix scoped in
[benchmarks/STAGE4_FINDINGS.md](benchmarks/STAGE4_FINDINGS.md).

Built in stages, each one chosen from the previous one's measurements:
single-request engine → continuous batching → paged KV with a hand-written CUDA
forward pass → chunked prefill → speculative decoding (in progress).

- [ARCHITECTURE.md](ARCHITECTURE.md) — **what is in this repo and why**: the five
  executables (only one is current), the host/device split, file-by-file roles,
  and a reading order for coming back cold. Start here.
- [IMPLEMENTATION.md](IMPLEMENTATION.md) — **how it was built**: design decisions,
  verification methodology, and the bugs worth remembering.
- [INFERENCE_ENGINE_PLAN.md](INFERENCE_ENGINE_PLAN.md) — roadmap, and why the
  ordering changed when measurements contradicted it.
- `benchmarks/` — the numbers.

## Status

- [x] **Stage 1** — Single-request engine → [benchmarks/STAGE1.md](benchmarks/STAGE1.md)
      41k tok/s prefill, 235 tok/s decode, verified against HF (cosine 1.0000, 5/5 top-1)
- [x] **Stage 2A** — Continuous batching → [benchmarks/STAGE2.md](benchmarks/STAGE2.md)
      947 tok/s peak (5.2× over single-request), 24/24 outputs identical to sequential
- [x] **Stage 2B** — Paged KV + custom decode path → [benchmarks/STAGE2B.md](benchmarks/STAGE2B.md)
      **15,090 tok/s (15.9× over Stage 2A)**, TTFT 524 → 83 ms, 128 concurrent
      requests in 512 MB of KV. No TensorRT: hand-written forward pass verified
      layer-by-layer against HF (rel err ~0.005), paged attention kernel
      bit-identical under shuffled block placement, preemption with recompute.
- [x] **vs vLLM 0.11** → [benchmarks/VS_VLLM.md](benchmarks/VS_VLLM.md)
      Same client, same GPU, same model, two prompt lengths.
      **10-token prompts: parity** (5,755 vs 5,633 tok/s, TPOT 4.57 vs 5.49 ms).
      **256-token prompts: vLLM 3.2× throughput, 9.6× TTFT** (1,294 vs 4,182
      tok/s; 1,172 vs 123 ms). TTFT is linear in prompt length — predicted
      256 × 4.6 ms = 1.18 s, measured 1.172 s — because prefill runs one token
      per forward pass. TPOT also crosses over: ahead on dispatch overhead at
      short context, behind on kernel quality at long context.
- [x] **Stage 3** — Chunked prefill → [benchmarks/STAGE3.md](benchmarks/STAGE3.md)
      **TTFT 1,172 → 240 ms (4.9×), throughput 1,294 → 3,261 tok/s (2.5×)** at
      256-token prompts. Varlen attention (`q_len >= 1`, mask as a loop bound),
      multi-token block append, decodes-first token-budget scheduler. Verified by
      chunk-size invariance: the same prompt in 1 pass and in 3 passes gives
      token-identical output. Tuned for latency it reaches **96.8 ms TTFT p50,
      beating vLLM's 123 ms**; throughput across all five scheduler
      configurations varied only 5%, which localizes the remaining gap to kernel
      quality rather than policy.
- [~] **Attention kernel profiling** → [benchmarks/STAGE4_FINDINGS.md](benchmarks/STAGE4_FINDINGS.md)
      Investigated and deliberately stopped. Attention is **47% of a step** and runs
      at **11.6% load efficiency** (3.7 useful bytes per 32-byte sector) because
      Phase 1 puts one thread per token, so adjacent lanes read 128 bytes apart.
      Not bandwidth-bound (15–19% of peak) and not compute-bound (20%) — it is
      latency-bound. The fix (shared-memory tile staging) is scoped and costed in
      the doc; left unimplemented since this project is about the concepts, and the
      concept — profile before optimizing, and why tiling forces online softmax —
      is captured.
- [ ] Stage 5 — Speculative decoding (low-concurrency latency; rides on the
      varlen kernel)
- [ ] Stage 3 — Speculative decoding

## Quick start

```bash
python tools/export_onnx.py --model <hf_model_dir>   # → onnx/
python tools/build_engine.py                         # → engine/
cmake -B build && cmake --build build -j

./build/runtime \
  --engine engine/llama1b_fp16.trt \
  --lm-head onnx/lm_head_weight.bin \
  --tokenizer onnx/tokenizer.json \
  --prompt "The key insight about transformers is" \
  --max-new-tokens 64 [--temperature 0.8 --top-k 50 --top-p 0.95]
```

Verify and benchmark:

```bash
python tools/verify_tokenizer.py                     # 8/8 vs HF tokenizers
python tools/verify_logits.py --model <hf_model_dir> # 5/5 top-1, cosine 1.0000
python tools/bench.py                                # perf sweep
```

## Layout

```
src/   runtime.cpp          Stage 1: single-request TRT runtime
       batch_runtime.cpp    Stage 2A: continuous batching over TRT
       kv_cache.h           single-sequence KV (ping-pong)
       batch_kv_cache.h     N fixed slots + scatter
       scheduler.h          request lifecycle + admission policy (no CUDA)
       block_allocator.h    paged blocks, free list, block tables (no CUDA)
       decode_layer.h       Stage 2B: hand-written forward pass, no TensorRT
       varlen_layer.h       Stage 3: forward pass over a mixed prefill+decode batch
       chunk_scheduler.h    Stage 3: per-step token budget policy (no CUDA)
       serve_chunked.cpp    Stage 3: engine loop with chunked prefill
       weights.h            raw fp16 tensor loader
       tokenizer.h          byte-level BPE, LLaMA-3 pre-tokenizer rules
       model_config.h       dimensions from config.json
       argmax.cuh           CUB greedy argmax

kernels/ sampling.cuh       temperature / top-k / top-p on device
         batched_pick.cuh   one segmented argmax for B sequences
         kv_scatter.cuh     fixed-slot KV scatter (Stage 2A)
         layer_kernels.cuh  RMSNorm, RoPE, SiLU-mul, embedding, residual
         paged_attention.cuh  decode attention with block-table indirection
         paged_attention_varlen.cuh  same, for q_len >= 1 (prefill chunks,
                              speculative verification); decode is the q_len==1 case

tools/  export_onnx.py, build_engine.py       build the TRT engine
        export_weights.py, dump_reference.py  build the custom decode path
        bench.py, bench_batch.py, bench_paged_attention.cu
        verify_logits.py, verify_tokenizer.py, verify_decode_path.cu
        test_block_allocator.cpp, test_paged_attention.cu, test_layer_kernels.cu
```

## Tests

```bash
g++  -std=c++17 -I src tools/test_block_allocator.cpp -o build/test_alloc
nvcc -std=c++17 -I kernels -I src tools/test_paged_attention.cu \
     -o build/test_paged --extended-lambda -arch=sm_120
nvcc -std=c++17 -I kernels tools/test_layer_kernels.cu -o build/test_layers -arch=sm_120
nvcc -std=c++17 -I src -I kernels tools/verify_decode_path.cu \
     -o build/verify_decode -lcublas --extended-lambda -arch=sm_120
```

## Origin

Starting point was the single-request TRT runtime in `../llama_jetson`
(C++ BPE tokenizer, GPU argmax, zero per-step malloc). That project remains the
edge-deployment reference (Jetson Orin Nano, TRT Edge-LLM vs llama.cpp).

Requires CUDA 12+, TensorRT 10.x (the pip `tensorrt` version must match the
system `libnvinfer` the binary links, or engine deserialization fails), cuBLAS.
