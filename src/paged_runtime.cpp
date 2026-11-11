/**
 * paged_runtime.cpp — continuous batching over the PAGED cache (S2.9).
 *
 * No TensorRT. The forward pass is decode_layer.h; memory comes from
 * BlockAllocator; attention is paged_attention.cuh.
 *
 * ── Prefill and decode are the same operation here ───────────────────────────
 * Stage 2A needed a separate batch-1 prefill enqueue because the TRT batch
 * shares one `seq` dimension — a joining request wants seq=N while decoding
 * slots want seq=1. In this path every sequence advances exactly ONE token per
 * step regardless, so a sequence still consuming its prompt simply takes its
 * next input from the prompt instead of from the sampler.
 *
 *   consequence 1: no admission stall — a request joins the batch immediately
 *   consequence 2: a P-token prompt costs P steps, not one enqueue
 *
 * (2) is why real systems do chunked prefill: pack many prompt tokens into one
 * step with varlen attention. Not implemented; the honest cost is measured.
 *
 * ── What this measures ───────────────────────────────────────────────────────
 * Given a fixed KV budget, how many requests fit and what throughput results.
 * Stage 2A reserved max_seq per slot (11% utilization on a real workload);
 * here a request holds ceil(len/16) blocks and returns them on completion.
 *
 * Build: see CMakeLists (target `paged_runtime`)
 */

#include "model_config.h"
#include "weights.h"
#include "block_allocator.h"
#include "decode_layer.h"
#include "tokenizer.h"
#include "batched_pick.cuh"

#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){                    \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,                      \
            cudaGetErrorString(e)); exit(1);} } while(0)

// ── One in-flight sequence ───────────────────────────────────────────────────
struct Seq {
    uint64_t id = 0;
    std::vector<int> prompt;
    std::vector<int> output;
    int len = 0;                 // tokens committed to the cache
    int max_new = 64;
    std::chrono::steady_clock::time_point t_submit, t_first, t_done;

    // While len < prompt.size() the next input comes from the prompt and the
    // sampled token is discarded — that is what makes prefill just decode.
    bool prefilling() const { return len < (int)prompt.size(); }
    int  next_input() const { return prefilling() ? prompt[len] : output.back(); }
    int  position()   const { return len; }
    bool finished(int eos, int eot) const {
        return !output.empty() &&
               ((int)output.size() >= max_new ||
                output.back() == eos || output.back() == eot);
    }
};

int main(int argc, char** argv) {
    std::string wdir = "weights", tok_path = "onnx/tokenizer.json",
                cfg_path = "onnx/config.json", prompts_path;
    int    block_size   = 16;
    int    max_batch    = 32;      // rows per forward pass
    int    max_new      = 64;
    int    max_seq      = 2048;
    double kv_budget_mb = 2048;    // the knob the whole experiment turns on
    bool   json_out     = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--weights"    && i+1<argc) wdir         = argv[++i];
        else if (a == "--tokenizer"  && i+1<argc) tok_path     = argv[++i];
        else if (a == "--config"     && i+1<argc) cfg_path     = argv[++i];
        else if (a == "--prompts"    && i+1<argc) prompts_path = argv[++i];
        else if (a == "--block-size" && i+1<argc) block_size   = std::stoi(argv[++i]);
        else if (a == "--max-batch"  && i+1<argc) max_batch    = std::stoi(argv[++i]);
        else if (a == "--max-new-tokens" && i+1<argc) max_new  = std::stoi(argv[++i]);
        else if (a == "--max-seq"    && i+1<argc) max_seq      = std::stoi(argv[++i]);
        else if (a == "--kv-budget-mb" && i+1<argc) kv_budget_mb = std::stod(argv[++i]);
        else if (a == "--json")                   json_out     = true;
    }

    ModelConfig cfg;
    cfg.load(cfg_path, max_seq);
    if (!json_out) cfg.print();

    Tokenizer tok;
    tok.load(tok_path);

    Weights weights;
    weights.load(wdir, cfg);

    // ── Size the pool from the budget ────────────────────────────────────────
    // bytes per block = layers * 2(K,V) * Hkv * block_size * head_dim * 2
    const size_t bytes_per_block = (size_t)cfg.num_layers * 2 * cfg.num_kv_heads
                                 * block_size * cfg.head_dim * sizeof(__half);
    const int num_blocks = (int)((kv_budget_mb * 1e6) / bytes_per_block);
    if (num_blocks < 8) { fprintf(stderr, "KV budget too small\n"); return 1; }
    const int max_blocks_per_seq = (max_seq + block_size - 1) / block_size;

    const size_t pool_per_layer = (size_t)num_blocks * cfg.num_kv_heads
                                * block_size * cfg.head_dim;
    const size_t pool_elems     = pool_per_layer * cfg.num_layers;

    __half *k_pool, *v_pool, *x, *logits;
    CK(cudaMalloc(&k_pool, pool_elems * sizeof(__half)));
    CK(cudaMalloc(&v_pool, pool_elems * sizeof(__half)));
    CK(cudaMalloc(&x,      (size_t)max_batch * cfg.hidden_dim * sizeof(__half)));
    CK(cudaMalloc(&logits, (size_t)max_batch * cfg.vocab_size * sizeof(__half)));

    int *d_table, *d_lens, *d_pos, *d_ids;
    CK(cudaMalloc(&d_table, (size_t)max_batch * max_blocks_per_seq * sizeof(int)));
    CK(cudaMalloc(&d_lens,  (size_t)max_batch * sizeof(int)));
    CK(cudaMalloc(&d_pos,   (size_t)max_batch * sizeof(int)));
    CK(cudaMalloc(&d_ids,   (size_t)max_batch * sizeof(int)));

    DecodeScratch scratch;
    scratch.alloc(cfg, max_batch);

    BlockAllocator alloc;
    // Watermark 0: preemption (below) is what protects running sequences now.
    // A large watermark makes admission fail on small pools without preventing
    // the real problem — a sequence admitted on its prompt still grows.
    alloc.configure({block_size, num_blocks, 0});

    BatchedPicker picker;
    picker.alloc(cfg.vocab_size, max_batch);

    cublasHandle_t blas;
    cublasCreate(&blas);

    if (!json_out) {
        printf("KV pool:   %.0f MB → %d blocks of %d tokens "
               "(%zu KB/block, all layers)\n",
               kv_budget_mb, num_blocks, block_size, bytes_per_block >> 10);
        printf("Capacity:  %d tokens total; at ~55 tok/request ≈ %d concurrent\n",
               num_blocks * block_size,
               num_blocks / ((55 + block_size - 1) / block_size));
    }

    // ── Workload ─────────────────────────────────────────────────────────────
    std::vector<std::string> prompts;
    if (!prompts_path.empty()) {
        std::ifstream pf(prompts_path);
        for (std::string line; std::getline(pf, line); )
            if (!line.empty()) prompts.push_back(line);
    } else {
        prompts = {"The capital of France is",
                   "Once upon a time",
                   "def fibonacci(n):",
                   "In 1969, humans first"};
    }

    std::vector<Seq> waiting, running, done;
    uint64_t next_id = 1;
    for (const auto& p : prompts) {
        Seq s;
        s.id       = next_id++;
        s.prompt   = tok.encode(p, /*add_bos=*/true);
        s.max_new  = max_new;
        s.t_submit = std::chrono::steady_clock::now();
        waiting.push_back(std::move(s));
    }
    std::reverse(waiting.begin(), waiting.end());   // pop_back = FIFO

    // ── The loop ─────────────────────────────────────────────────────────────
    const auto t0 = std::chrono::steady_clock::now();
    long long steps = 0, tokens_out = 0, preemptions = 0;
    int peak_batch = 0;
    double occupancy_sum = 0;

    // TODO
    // while (!waiting.empty() || !running.empty()) {
    //
    //   1. ADMIT while there is room:
    //        while (!waiting.empty() && (int)running.size() < max_batch
    //               && alloc.can_admit(waiting.back().prompt.size())) {
    //            Seq s = std::move(waiting.back()); waiting.pop_back();
    //            alloc.allocate(s.id, (int)s.prompt.size());
    //            running.push_back(std::move(s));
    //        }
    //        No prefill stall: the sequence just joins with len = 0.
    //
    //   2. BUILD the batch (B = running.size()):
    //        ids[r]  = running[r].next_input()
    //        pos[r]  = running[r].position()
    //        lens[r] = running[r].len + 1     <- INCLUDES the token being added
    //        table   = alloc.flatten(ids_of_running, max_blocks_per_seq)
    //        upload all four
    //
    //   3. GROW the cache for each row BEFORE the forward pass — write_kv_paged
    //      stores at lens-1, so the block must already exist:
    //        if (!alloc.append_token(seq.id, seq.len)) { /* preempt */ }
    //
    //   4. FORWARD:
    //        launch_embedding(...); for each layer forward_layer(...);
    //        launch_rmsnorm(final); gemm(lm_head) -> logits
    //        picker.argmax_batched(logits, B, out_tokens, 0)
    //
    //   5. COMMIT per row:
    //        seq.len += 1
    //        if (seq.prefilling())  discard out_tokens[r]   <- still reading prompt
    //        else { seq.output.push_back(out_tokens[r]); ++tokens_out; }
    //        note: the FIRST generated token appears when len reaches
    //        prompt.size(); stamp t_first there for TTFT
    //
    //   6. RETIRE finished rows: alloc.release(seq.id), move to `done`,
    //      erase from `running` (iterate backwards so indices stay valid)
    //
    //   ++steps; peak_batch = max(peak_batch, B);
    //   occupancy_sum += alloc.utilization(lengths_of_running);
    // }
    while (!waiting.empty() || !running.empty()) {
        // 1. ADMIT
        while (!waiting.empty() && (int)running.size() < max_batch
               && alloc.can_admit(waiting.back().prompt.size())) {
            Seq s = std::move(waiting.back()); waiting.pop_back();
            alloc.allocate(s.id, (int)s.prompt.size());
            running.push_back(std::move(s));
        }

        // Nothing admissible and nothing running: the budget cannot fit even
        // one request. Fail loudly rather than spin.
        if (running.empty()) {
            fprintf(stderr, "deadlock: %zu waiting, %d blocks free "
                    "— KV budget cannot hold one request\n",
                    waiting.size(), alloc.num_free());
            break;
        }

        // ── 1b. PREEMPT ──────────────────────────────────────────────────────
        // A sequence is admitted on its PROMPT length, but it keeps growing.
        // Under pressure the pool fills with half-finished work and the next
        // append_token fails. Rather than reserve prompt+max_new up front
        // (which wastes capacity, since most requests finish early), admit
        // optimistically and evict when growth would fail.
        //
        // Victim policy: newest first (LIFO) — least work invested, so the
        // least recompute is thrown away. This is vLLM's "recompute" strategy;
        // the alternative is "swap" (copy blocks to host and restore).
        //
        // Done BEFORE building the batch so B and the row order stay stable
        // through the rest of the step.
        {
            auto blocks_needed = [&]() {
                int n = 0;
                for (const auto& s : running)
                    if (s.len % block_size == 0) ++n;   // this row starts a block
                return n;
            };
            while (alloc.num_free() < blocks_needed() && running.size() > 1) {
                Seq victim = std::move(running.back());
                running.pop_back();
                alloc.release(victim.id);          // blocks return to the pool
                victim.len = 0;                    // recompute from scratch
                victim.output.clear();
                // Insert at the FRONT of `waiting`, which is popped from the
                // back — so it is admitted LAST. Requeueing it as next-to-admit
                // would thrash: admit, preempt, admit, preempt.
                waiting.insert(waiting.begin(), std::move(victim));
                ++preemptions;
            }
        }

        // 2. BUILD the batch
        const int B = (int)running.size();
        std::vector<int> h_ids(B), h_pos(B), h_lens(B);
        for (int r = 0; r < B; ++r) {
            h_ids[r]  = running[r].next_input();
            h_pos[r]  = running[r].position();
            h_lens[r] = running[r].len + 1;
        }
        CK(cudaMemcpy(d_ids,  h_ids.data(),  B * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos,  h_pos.data(),  B * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_lens, h_lens.data(), B * sizeof(int), cudaMemcpyHostToDevice));

        // 3. GROW the cache — must happen BEFORE flatten, or the block holding
        //    this token would not be in the table write_kv_paged consults.
        for (int r = 0; r < B; ++r) {
            // Preemption above guaranteed enough free blocks, so this cannot
            // fail — assert rather than handle.
            if (!alloc.append_token(running[r].id, running[r].len)) {
                fprintf(stderr, "BUG: growth failed after preemption "
                        "(seq %llu, %d free)\n",
                        (unsigned long long)running[r].id, alloc.num_free());
                exit(1);
            }
        }

        std::vector<uint64_t> ids_of_running;
        ids_of_running.reserve(B);
        for (const auto& s : running) ids_of_running.push_back(s.id);
        const std::vector<int> table =
            alloc.flatten(ids_of_running, max_blocks_per_seq);
        CK(cudaMemcpy(d_table, table.data(), table.size() * sizeof(int),
                      cudaMemcpyHostToDevice));

        // 4. FORWARD
        const int max_len = *std::max_element(h_lens.begin(), h_lens.end());
        forward_decode(cfg, weights, blas, scratch,
                       d_ids, x, logits,
                       k_pool, v_pool,
                       d_table, d_lens, d_pos,
                       B, block_size, num_blocks, max_blocks_per_seq, max_len,
                       /*stream=*/0);

        // 5. COMMIT
        std::vector<int> h_out;
        picker.argmax_batched(logits, B, h_out, 0);
        for (int r = 0; r < B; ++r) {
            Seq& s = running[r];
            s.len += 1;
            if (s.prefilling()) {
                // still reading the prompt — the sampled token is discarded
            } else {
                if (s.output.empty()) s.t_first = std::chrono::steady_clock::now();
                s.output.push_back(h_out[r]);
                ++tokens_out;
            }
        }

        // Stats before retirement, so occupancy reflects what was resident.
        ++steps;
        peak_batch = std::max(peak_batch, B);
        {
            std::vector<int> lens_of_running;
            lens_of_running.reserve(running.size());
            for (const auto& s : running) lens_of_running.push_back(s.len);
            occupancy_sum += alloc.utilization(lens_of_running);
        }

        // 6. RETIRE — iterate backwards so erasing does not shift rows we have
        //    not visited yet.
        for (int r = B - 1; r >= 0; --r) {
            Seq& s = running[r];
            if (s.finished(tok.eos_id(), tok.eot_id())) {
                s.t_done = std::chrono::steady_clock::now();
                alloc.release(s.id);              // blocks go back to the pool
                done.push_back(std::move(s));
                running.erase(running.begin() + r);
            }
        }
    }
    const double wall_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // ── Report ───────────────────────────────────────────────────────────────
    std::vector<double> ttfts;
    for (const auto& s : done)
        ttfts.push_back(std::chrono::duration<double,std::milli>(
            s.t_first - s.t_submit).count());
    auto pct = [](std::vector<double> v, double p) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, (size_t)(v.size() * p))];
    };

    if (json_out) {
        printf("{\"kv_budget_mb\":%.0f,\"num_blocks\":%d,\"block_size\":%d,"
               "\"requests\":%zu,\"steps\":%lld,\"tokens\":%lld,"
               "\"wall_ms\":%.1f,\"tok_s\":%.1f,\"peak_batch\":%d,"
               "\"mean_occupancy\":%.3f,\"ttft_p50\":%.1f,\"ttft_p95\":%.1f,"
               "\"preemptions\":%lld}\n",
               kv_budget_mb, num_blocks, block_size, done.size(), steps,
               tokens_out, wall_ms, tokens_out / (wall_ms / 1000.0),
               peak_batch, steps ? occupancy_sum / steps : 0.0,
               pct(ttfts, 0.5), pct(ttfts, 0.95), preemptions);
    } else {
        printf("\n──────────────────────────────────────────────\n");
        for (const auto& s : done)
            printf("[%llu] %d tok  %s\n", (unsigned long long)s.id,
                   (int)s.output.size(), tok.decode(s.output).c_str());
        printf("──────────────────────────────────────────────\n");
        printf("  %zu requests, %lld steps, %lld tokens\n",
               done.size(), steps, tokens_out);
        printf("  peak batch %d, mean KV utilization %.1f%%, %lld preemptions\n",
               peak_batch, 100.0 * (steps ? occupancy_sum / steps : 0),
               preemptions);
        printf("  wall %.0f ms → %.1f tok/s aggregate\n",
               wall_ms, tokens_out / (wall_ms / 1000.0));
        printf("──────────────────────────────────────────────\n\n");
    }

    scratch.free();
    weights.free();
    picker.free();
    cublasDestroy(blas);
    return 0;
}
