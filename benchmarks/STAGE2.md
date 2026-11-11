# Stage 2A — Continuous batching benchmarks

**Model** LLaMA-3.2-1B-Instruct FP16, TensorRT · ctx 512  
**GPU** NVIDIA GeForce RTX 5070 Ti  
**Host** Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39  
**Date** 2026-08-16  
**Workload** 24 requests, 48 tokens each, greedy

## Correctness

24/24 requests produced identical text at batch 8 and batch 1.

Fixed-slot batching passes `past = max_seq` for every slot, so the attention reduction length does not depend on batch composition — results are bit-identical regardless of what else is running. (Contrast S2.0's ragged test, where `past = max(lens)` varied and fp16 rounding shifted results by ~1.6% of signal.)

## A. Throughput vs batch size

| Batch | tok/s | vs B=1 | Efficiency | TTFT p50 | TTFT p95 | TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 192 | 1.00x | 100% | 3024 ms | 5518 ms | 5.14 ms |
| 2 | 342 | 1.78x | 89% | 1705 ms | 3094 ms | 5.74 ms |
| 4 | 595 | 3.11x | 78% | 980 ms | 1638 ms | 6.50 ms |
| 8 | 832 | 4.34x | 54% | 525 ms | 995 ms | 8.92 ms |
| 16 | 933 | 4.86x | 30% | 222 ms | 801 ms | 12.62 ms |
| 32 | 152 | 0.79x | 2% | 393 ms | 1084 ms | 127.97 ms |
| 64 | 47 | 0.24x | 0% | 1379 ms | 6926 ms | 383.10 ms |
| 128 | 66 | 0.35x | 0% | 1413 ms | 3608 ms | 266.19 ms |
| 256 | 70 | 0.37x | 0% | 1126 ms | 3245 ms | 254.21 ms |
| 512 | 62 | 0.32x | 0% | 1360 ms | 3809 ms | 288.35 ms |

Efficiency is speedup / batch size — the fraction of ideal linear scaling. The gap is the fixed-slot tax: every slot scans the full `max_seq` window regardless of its real length, so a 35-token request does the same KV traffic as a 500-token one. Eliminating that is Stage 2B.

## B. Admission policy (batch 8)

| Admits/step | tok/s | TTFT p50 | TTFT p95 | TPOT |
|---:|---:|---:|---:|---:|
| 1 | 820 | 533 ms | 1009 ms | 9.06 ms |
| 2 | 836 | 522 ms | 990 ms | 9.01 ms |
| 4 | 841 | 519 ms | 980 ms | 8.95 ms |

Admission requires a separate batch-1 prefill that stalls the decode batch. Admitting more per step fills slots sooner (lower queueing delay) at the cost of longer stalls for requests already running.

## Design notes

- **Prefill cannot join the decode batch.** The batch shares one `seq` dimension: a joining request wants `seq=N` while decoding slots want `seq=1`, and no single shape serves both. Real chunked prefill packs both into one flat sequence with varlen attention, which needs a custom kernel (Stage 2B). Here prefill runs alone.
- **Ping-pong does not survive batching.** It worked in Stage 1 because `past` grew by exactly 1 per step, matching the stride TRT writes. With N slots, admitting a long request changes `past` for everyone and invalidates every slot's layout. So the cache owns memory at a constant `max_seq` stride and a scatter kernel moves each new token to `cache[slot, :, lens[slot], :]`.
- **Batched sampling removed B-1 syncs per step.** One `cublasHgemm` with `n=B` (the row-major/column-major layouts already line up — no strided-batch API needed) plus one `cub::DeviceSegmentedReduce::ArgMax` replaced a per-row loop. Measured: ~0.7 ms per sync eliminated, +24% aggregate throughput at B=4.
