# Stage 4 — profiling findings (not implemented)

**Status:** investigated, then deliberately stopped. The optimization was scoped,
measured and costed; implementing it was judged not worth the time for a project
whose purpose is understanding the concepts. Recorded here so the measurement
isn't lost, and because the *decision* is part of the result.

**GPU** RTX 5070 Ti (70 SMs, 896 GB/s peak HBM) · LLaMA-3.2-1B FP16

## Where a forward step goes

`tools/profile_step.cu`, 32 sequences × 1 query token, context 512, 16 layers:

| category | ms/step | share |
|---|---:|---:|
| attention | 4.04 | **47.4%** |
| gemm (7 per layer) | 3.14 | 36.9% |
| elementwise (norm, RoPE, SiLU, residual, write_kv) | 0.64 | 7.5% |
| lm_head | 0.69 | 8.1% |

Attention is the largest single consumer, which was not obvious beforehand — at
`q_len = 1` the GEMMs run against 2048×2048 weight matrices while attention scans
only 32×512 tokens.

Amdahl ceiling for an attention-only optimization: 1.5× → 18.8% overall,
2× → 31.1%, 4× → 55.2%. A 2× would have taken the server from 3,261 to roughly
4,200 tok/s, i.e. vLLM's measured 4,182.

*Caveat:* `profile_step` synchronizes at every category boundary (~384 round trips
per step), which inflates absolute times — `sum` 8.5 ms vs a real step time of
~6 ms implied by the measured TPOT. The shares are sound; the milliseconds are not.

## The kernel is not memory-bound, and not compute-bound

`tools/bench_varlen_attention.cu`, min of 5 trials:

| shape | min µs | achieved | % of peak |
|---|---:|---:|---:|
| decode, batch 32, ctx 256 | 113.3 | 151 GB/s | 16.8% |
| decode, batch 32, ctx 2048 | 990.1 | 136 GB/s | 15.2% |
| prefill, 1×256, ctx 0 | 418.0 | 166 GB/s | 18.5% |
| prefill, 1×512, ctx 0 | 1689.3 | 162 GB/s | 18.0% |
| mixed, 30 decode + 1×256 | 2158.9 | 157 GB/s | 17.5% |

**15–19% of peak bandwidth on every shape.** Uniformly low regardless of `q_len`
or context, which rules out the bandwidth story the benchmark was written to test.

Nsight Compute confirms why:

```
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct   11.60 %
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum              18,491,392 sectors
sm__throughput.avg.pct_of_peak_sustained_elapsed                20.40 %
```

**11.60% load efficiency** — of every 32-byte sector fetched, 3.7 bytes are used.
Cross-checked by hand: that launch should need 67 MB = 2.1 M sectors; it issued
18.5 M. An 8.8× overfetch, which inverts to 11.4%, matching the reported 11.60%.

Compute throughput is also only 20%, so the kernel is **latency-bound**: stalling
on memory requests that mostly fetch bytes it discards.

## The cause: Phase 1's access pattern

```cuda
for (int i = tid; i < n_keys; i += blockDim.x) {   // thread per TOKEN
    for (int d = 0; d < D; ++d)                    // each thread walks all of D
        acc += __half2float(qv[d]) * __half2float(kv[d]);
}
```

Thread `t` owns token `t`. At a given `d`, lane 0 reads `k_pool[token0*64 + d]` and
lane 1 reads `k_pool[token1*64 + d]` — **128 bytes apart**. Each lane wants 2 bytes,
so a 32-lane warp triggers 32 separate sector fetches to use 64 bytes of them.

Phase 3 is already coalesced (`base[o*D + d]` with thread `d` → adjacent lanes read
adjacent `d` of the same token), which is why the aggregate figure is 11.6% rather
than ~6%.

Note the loop *looks* strided and is nonetheless uncoalesced: the stride is over
the wrong axis. "Strided is good" was never the rule — adjacent threads must touch
adjacent addresses.

This affects `paged_attention.cuh` identically, which reframes a Stage 2B finding:
"paged attention is at parity with the fixed-slot kernel" was parity between two
equally uncoalesced kernels. Neither was close to the hardware.

## The scoped fix, and why it was not written

Stage K in shared memory with a *cooperative coalesced load*, then compute
thread-per-token out of shared:

```
for each tile of 64 keys (= 4 logical blocks):
    for each logical block in the tile:
        all 64 threads copy that block's contiguous 2 KB into k_tile
        thread tid reads src[tid], src[tid+64], ... -> adjacent halves, coalesced
    __syncthreads()
    thread j computes dot(q, k_tile + j*D) from SHARED memory
    __syncthreads()
```

Global reads coalesce because threads cooperate on contiguous addresses; the math
keeps thread-per-token parallelism, so no warp reduction is needed and
memory-level parallelism is preserved. Shared memory ≈ 16.5 KB at
`max_scan = 2048`, comfortably under the 48 KB default.

Expected: load efficiency 11.6% → 90%+, attention perhaps 2×, step ~31% faster.
**Not implemented** — the concept (coalescing, shared-memory staging, profiling
before optimizing) was the point, and it is captured above.

### What a full version would need

Tiling forces **online softmax**. With separate Phase 1 / Phase 3 loops, Phase 2
sits between them and needs the whole score vector, so `max_scan` must stay the
full scan length. Shrinking it requires fusing the phases and accumulating output
per tile with a running max and a rescaled accumulator.

Worth recording because earlier in this project the conclusion was that online
softmax is unnecessary for decode — true while the whole 1×N score vector fits in
shared memory, so two passes over it are free. Tiling changes that premise. Online
softmax is not primarily a numerical-stability device; it is **what makes tiling
possible**, and that is why FlashAttention has it.

## Benchmark methodology, learned the hard way

The same binary measured **113 µs and then 179 µs** — 58% apart. Cause: a video
game was open, competing for SMs, bandwidth and the power budget.

Worse, an elaborate explanation had already been constructed for the "slowdown"
(collapsed memory-level parallelism, warp-reduction overhead) before anyone checked
that the kernel had even been rebuilt. It hadn't: the ncu output still showed
`(64,1,1)` threads per block and a byte-identical sector count.

Two rules adopted:

1. **Report min of N trials, not a mean.** The minimum is the trial least polluted
   by throttling and contention, and it reproduces across sessions to within ~1%
   (113.5 → 113.3 µs, 490.1 → 488.4, 988.5 → 990.1). Medians drift 20–30% within a
   single run from thermal droop alone.
2. **Verify the binary changed before explaining a result.** Launch configuration
   and instruction-level counters are in the profiler output; check them.

## Reproduce

```bash
nvcc -std=c++17 -O3 -I kernels -I src tools/bench_varlen_attention.cu \
     -o build/bench_varlen --extended-lambda -arch=sm_120
./build/bench_varlen

nvcc -std=c++17 -O3 -I src -I kernels tools/profile_step.cu \
     -o build/profile_step -lcublas --extended-lambda -arch=sm_120
./build/profile_step --weights weights --config onnx/config.json \
                     --n-seqs 32 --q-len 1 --context 512

ncu --kernel-name paged_attention_varlen -s 25 -c 2 \
  --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./build/bench_varlen 2>&1 | tail -20
```

Check `nvidia-smi` shows an idle GPU first.
