/**
 * serve_chunked.cpp — HTTP server with CHUNKED PREFILL (S3.4).
 *
 * serve.cpp consumes a prompt one token per forward pass, so TTFT is
 * prompt_len x step_time: measured 1,172 ms for a 256-token prompt against
 * vLLM's 123 ms (benchmarks/VS_VLLM.md). This version batches prompt tokens
 * with decode tokens in a single pass, under a per-step token budget.
 *
 * ── The step ─────────────────────────────────────────────────────────────────
 *   DRAIN    take newly-arrived requests off the HTTP queue
 *   ADMIT    move waiting -> running (lazy: no blocks reserved for the prompt)
 *   PREEMPT  guarantee the decoders can grow; evict newest-first if not
 *   PLAN     ChunkScheduler: 1 token per decode, remainder to prefill, FCFS
 *   COMMIT   append_tokens for every grant  (affordable by construction)
 *   BUILD    flat q_seq / q_pos / token-id arrays + the block table
 *   FORWARD  one varlen pass over total_q tokens
 *   SAMPLE   logits only for sequences whose chunk finished their prompt
 *   RETIRE   EOS, token limit, or client hung up
 *
 * ── Why PREEMPT runs before PLAN ─────────────────────────────────────────────
 * The scheduler grants decode tokens unconditionally (starving a mid-stream
 * decode is worse than a long step) but counts the blocks they need. If the
 * pool cannot cover the decoders at all, no plan is affordable — so free space
 * first, then plan against what is actually available. That ordering is what
 * makes every append_tokens below succeed.
 */

#include "model_config.h"
#include "weights.h"
#include "block_allocator.h"
#include "chunk_scheduler.h"
#include "varlen_layer.h"
#include "tokenizer.h"
#include "http.h"
#include "batched_pick.cuh"

#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){                    \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,                      \
            cudaGetErrorString(e)); exit(1);} } while(0)

using Clock = std::chrono::steady_clock;

struct Seq {
    uint64_t id = 0;
    std::vector<int> prompt;
    std::vector<int> output;
    int len     = 0;             // tokens already in the KV cache
    int max_new = 64;
    std::shared_ptr<http::Stream> stream;
    Clock::time_point t_submit, t_first;
    bool header_sent = false;
    bool hit_eos     = false;

    int  prompt_len() const { return (int)prompt.size(); }
    bool prefilling() const { return len < prompt_len(); }

    /// Token fed at absolute position `pos`.
    int token_at(int pos) const {
        return pos < prompt_len() ? prompt[pos] : output[pos - prompt_len()];
    }
};

int main(int argc, char** argv) {
    std::string wdir = "weights", tok_path = "onnx/tokenizer.json",
                cfg_path = "onnx/config.json",
                model_name = "llama-3.2-1b-instruct";
    int    port = 8080, block_size = 16, max_batch = 64, max_seq = 2048;
    int    max_batched_tokens = 512, max_chunk = 512;
    // How many WAITING requests may become RUNNING in one step. Separate knob
    // from the token budget, and measurement showed why they must be separate:
    // a large budget admits many prompts at once, so nearly all in-flight
    // requests prefill simultaneously and finish together — p50 TTFT 123 -> 240
    // ms even though throughput was unchanged (processor sharing across
    // requests). Capping admissions staggers prefill completions instead.
    // 0 = unlimited.
    int    max_admits_per_step = 0;
    double kv_budget_mb = 1024;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--weights"    && i+1<argc) wdir       = argv[++i];
        else if (a == "--tokenizer"  && i+1<argc) tok_path   = argv[++i];
        else if (a == "--config"     && i+1<argc) cfg_path   = argv[++i];
        else if (a == "--model-name" && i+1<argc) model_name = argv[++i];
        else if (a == "--port"       && i+1<argc) port       = std::stoi(argv[++i]);
        else if (a == "--block-size" && i+1<argc) block_size = std::stoi(argv[++i]);
        else if (a == "--max-batch"  && i+1<argc) max_batch  = std::stoi(argv[++i]);
        else if (a == "--max-seq"    && i+1<argc) max_seq    = std::stoi(argv[++i]);
        else if (a == "--max-batched-tokens" && i+1<argc)
            max_batched_tokens = std::stoi(argv[++i]);
        else if (a == "--max-chunk"  && i+1<argc) max_chunk  = std::stoi(argv[++i]);
        else if (a == "--max-admits-per-step" && i+1<argc)
            max_admits_per_step = std::stoi(argv[++i]);
        else if (a == "--kv-budget-mb" && i+1<argc) kv_budget_mb = std::stod(argv[++i]);
    }
    // Decodes are granted unconditionally, so the budget must at least cover a
    // full batch of them or every step overruns it.
    max_batched_tokens = std::max(max_batched_tokens, max_batch);

    ModelConfig cfg;  cfg.load(cfg_path, max_seq);  cfg.print();
    Tokenizer   tok;  tok.load(tok_path);
    Weights weights;  weights.load(wdir, cfg);

    const size_t bytes_per_block = (size_t)cfg.num_layers * 2 * cfg.num_kv_heads
                                 * block_size * cfg.head_dim * sizeof(__half);
    const int num_blocks = (int)((kv_budget_mb * 1e6) / bytes_per_block);
    const int max_blocks_per_seq = (max_seq + block_size - 1) / block_size;
    const size_t pool_per_layer = (size_t)num_blocks * cfg.num_kv_heads
                                * block_size * cfg.head_dim;

    // Row-count buffers are sized by TOKENS now, not sequences — a step may
    // carry max_batched_tokens query tokens. Only logits and out_rows stay
    // per-sequence, because only one row per sequence is ever sampled.
    __half *k_pool, *v_pool, *x, *logits;
    CK(cudaMalloc(&k_pool, pool_per_layer * cfg.num_layers * sizeof(__half)));
    CK(cudaMalloc(&v_pool, pool_per_layer * cfg.num_layers * sizeof(__half)));
    CK(cudaMalloc(&x,      (size_t)max_batched_tokens * cfg.hidden_dim * sizeof(__half)));
    CK(cudaMalloc(&logits, (size_t)max_batch * cfg.vocab_size * sizeof(__half)));

    int *d_table, *d_q_seq, *d_q_pos, *d_ids, *d_out_rows;
    CK(cudaMalloc(&d_table,    (size_t)max_batch * max_blocks_per_seq * sizeof(int)));
    CK(cudaMalloc(&d_q_seq,    (size_t)max_batched_tokens * sizeof(int)));
    CK(cudaMalloc(&d_q_pos,    (size_t)max_batched_tokens * sizeof(int)));
    CK(cudaMalloc(&d_ids,      (size_t)max_batched_tokens * sizeof(int)));
    CK(cudaMalloc(&d_out_rows, (size_t)max_batch * sizeof(int)));

    DecodeScratch scratch;  scratch.alloc(cfg, max_batched_tokens, max_batch);
    BlockAllocator alloc;   alloc.configure({block_size, num_blocks, 0});
    BatchedPicker picker;   picker.alloc(cfg.vocab_size, max_batch);
    ChunkScheduler sched;   sched.configure({max_batched_tokens, max_chunk});
    cublasHandle_t blas;    cublasCreate(&blas);

    std::mutex mu;
    std::condition_variable cv;
    std::deque<Seq> incoming;
    uint64_t next_id = 1;
    bool shutting_down = false;

    http::Server server;
    server.set_model_name(model_name);
    server.listen_and_serve(port,
        [&](http::Request req, std::shared_ptr<http::Stream> st) {
            Seq s;
            s.prompt   = tok.encode(req.prompt, /*add_bos=*/true);
            s.max_new  = std::max(1, std::min(req.max_tokens, max_seq - 8));
            s.stream   = std::move(st);
            s.t_submit = Clock::now();
            {
                std::lock_guard<std::mutex> lk(mu);
                s.id = next_id++;
                s.stream->set_id("cmpl-" + std::to_string(s.id));
                incoming.push_back(std::move(s));
            }
            cv.notify_one();
        });

    printf("Serving:   http://0.0.0.0:%d   (chunked prefill)\n", port);
    printf("KV pool:   %.0f MB -> %d blocks of %d tokens\n",
           kv_budget_mb, num_blocks, block_size);
    printf("Budget:    %d tokens/step, max chunk %d, max batch %d\n",
           max_batched_tokens, max_chunk, max_batch);
    if (max_admits_per_step > 0)
        printf("Admissions: %d per step\n", max_admits_per_step);
    printf("\n");

    std::vector<Seq> waiting, running;
    long long preemptions = 0;

    while (true) {
        // ── DRAIN ────────────────────────────────────────────────────────────
        {
            std::unique_lock<std::mutex> lk(mu);
            if (incoming.empty() && waiting.empty() && running.empty())
                cv.wait(lk, [&] { return !incoming.empty() || shutting_down; });
            if (shutting_down) break;
            while (!incoming.empty()) {
                waiting.push_back(std::move(incoming.front()));
                incoming.pop_front();
            }
        }

        // ── ADMIT (lazy: no blocks reserved for the whole prompt) ─────────────
        int admitted = 0;
        while (!waiting.empty() && (int)running.size() < max_batch) {
            if (max_admits_per_step > 0 && admitted >= max_admits_per_step) break;
            Seq& front = waiting.front();

            // Lazy allocation lost the guard eager allocation had for free: an
            // oversized prompt would be admitted, prefill until blocks run out,
            // be preempted, requeued, admitted again — forever. Reject it.
            if (!alloc.fits_ever(front.prompt_len())) {
                if (front.stream->alive()) {
                    front.stream->begin();
                    front.stream->token("[error: prompt exceeds KV cache capacity]");
                    front.stream->done(0, 0.0, 0.0, /*stop=*/true);
                }
                waiting.erase(waiting.begin());
                continue;
            }
            // Room for at least the first chunk?
            const int first = std::min(front.prompt_len(), max_chunk);
            if (!alloc.can_admit(first)) break;

            Seq s = std::move(front);
            waiting.erase(waiting.begin());
            alloc.allocate(s.id, 0);              // empty table; grows per chunk
            running.push_back(std::move(s));
            ++admitted;
        }
        if (running.empty()) continue;

        // ── PREEMPT: make sure every decoder can take its one token ───────────
        // Must run BEFORE planning: the scheduler grants decode tokens
        // unconditionally, so if the pool cannot back them no plan is
        // affordable. Newest-first (LIFO) — least work invested.
        {
            auto decode_blocks_needed = [&] {
                int n = 0;
                for (const auto& s : running)
                    if (!s.prefilling())
                        n += alloc.blocks_to_append(s.id, s.len, 1);
                return n;
            };
            while (alloc.num_free() < decode_blocks_needed() && running.size() > 1) {
                Seq victim = std::move(running.back());
                running.pop_back();
                alloc.release(victim.id);
                victim.len = 0;                   // discard partial prefill too
                victim.output.clear();
                alloc.allocate(victim.id, 0);
                waiting.push_back(std::move(victim));   // re-admitted last
                ++preemptions;
            }
        }

        // ── PLAN ─────────────────────────────────────────────────────────────
        std::vector<SeqView> views;
        views.reserve(running.size());
        for (const auto& s : running)
            views.push_back({s.id, s.len, s.prompt_len()});
        const ChunkPlan plan = sched.plan(views, alloc);
        if (plan.entries.empty()) continue;

        // Map sequence id -> index in `running`, so plan entries (which carry
        // ids, not positions) can be applied.
        std::vector<int> slot_of(running.size());
        auto index_of = [&](uint64_t id) {
            for (size_t i = 0; i < running.size(); ++i)
                if (running[i].id == id) return (int)i;
            return -1;
        };

        // ── COMMIT + BUILD ───────────────────────────────────────────────────
        std::vector<int> h_ids, h_q_seq, h_q_pos, h_out_rows;
        std::vector<int> sample_of;               // out_row index -> running index
        std::vector<uint64_t> ids_of_running;
        for (const auto& s : running) ids_of_running.push_back(s.id);

        h_ids.reserve(plan.total_tokens);
        h_q_seq.reserve(plan.total_tokens);
        h_q_pos.reserve(plan.total_tokens);

        bool commit_failed = false;
        for (const auto& e : plan.entries) {
            const int r = index_of(e.id);
            if (r < 0) continue;
            Seq& s = running[r];

            // Affordable by construction — the scheduler planned against the
            // same allocator state. A failure here means the plan and the pool
            // disagree, which is a bug rather than back-pressure.
            if (!alloc.append_tokens(s.id, s.len, e.n_tokens)) {
                fprintf(stderr, "BUG: unaffordable plan (seq %llu, n=%d, free=%d)\n",
                        (unsigned long long)s.id, e.n_tokens, alloc.num_free());
                commit_failed = true;
                break;
            }

            for (int i = 0; i < e.n_tokens; ++i) {
                const int pos = s.len + i;
                h_ids.push_back(s.token_at(pos));
                h_q_seq.push_back(r);             // row of the flattened table
                h_q_pos.push_back(pos);
            }

            // Sample only if this chunk reaches the end of the prompt. A
            // sequence still mid-prefill predicts nothing we keep — its
            // interior tokens exist to populate the KV cache.
            if (s.len + e.n_tokens >= s.prompt_len()) {
                h_out_rows.push_back((int)h_ids.size() - 1);   // last query token
                sample_of.push_back(r);
            }
        }
        if (commit_failed) continue;

        const int total_q = (int)h_ids.size();
        const int n_out   = (int)h_out_rows.size();
        if (total_q == 0) continue;

        const std::vector<int> table =
            alloc.flatten(ids_of_running, max_blocks_per_seq);
        int max_scan = 0;
        for (int p : h_q_pos) max_scan = std::max(max_scan, p + 1);

        CK(cudaMemcpy(d_ids,   h_ids.data(),   total_q*sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_q_seq, h_q_seq.data(), total_q*sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_q_pos, h_q_pos.data(), total_q*sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_table, table.data(), table.size()*sizeof(int),
                      cudaMemcpyHostToDevice));
        if (n_out)
            CK(cudaMemcpy(d_out_rows, h_out_rows.data(), n_out*sizeof(int),
                          cudaMemcpyHostToDevice));

        // ── FORWARD ──────────────────────────────────────────────────────────
        forward_varlen(cfg, weights, blas, scratch, d_ids, x, logits,
                       k_pool, v_pool, d_table, d_q_seq, d_q_pos, d_out_rows,
                       total_q, n_out, block_size, num_blocks,
                       max_blocks_per_seq, max_scan, 0);

        // Advance lengths for every granted token, sampled or not.
        for (const auto& e : plan.entries) {
            const int r = index_of(e.id);
            if (r >= 0) running[r].len += e.n_tokens;
        }

        // ── SAMPLE + stream ──────────────────────────────────────────────────
        if (n_out) {
            std::vector<int> picked;
            picker.argmax_batched(logits, n_out, picked, 0);

            for (int o = 0; o < n_out; ++o) {
                Seq& s = running[sample_of[o]];
                if (s.output.empty()) {
                    s.t_first = Clock::now();
                    if (s.stream->alive()) { s.stream->begin(); s.header_sent = true; }
                }
                s.output.push_back(picked[o]);
                s.hit_eos = (picked[o] == tok.eos_id() || picked[o] == tok.eot_id());
                if (!s.hit_eos && s.stream->alive())
                    s.stream->token(tok.decode({picked[o]}));
            }
        }

        // ── RETIRE ───────────────────────────────────────────────────────────
        for (int r = (int)running.size() - 1; r >= 0; --r) {
            Seq& s = running[r];
            const bool at_limit = (int)s.output.size() >= s.max_new;
            const bool gone     = !s.stream->alive();
            if (!s.hit_eos && !at_limit && !gone) continue;

            if (!gone) {
                const double ttft = std::chrono::duration<double,std::milli>(
                    s.t_first - s.t_submit).count();
                const double gen = std::chrono::duration<double,std::milli>(
                    Clock::now() - s.t_first).count();
                if (!s.header_sent) s.stream->begin();
                s.stream->done((int)s.output.size(), ttft,
                               gen > 0 ? s.output.size() / (gen / 1000.0) : 0.0,
                               s.hit_eos);
            }
            alloc.release(s.id);
            running.erase(running.begin() + r);
        }
    }

    server.stop();
    printf("preemptions: %lld\n", preemptions);
    scratch.free(); weights.free(); picker.free();
    cublasDestroy(blas);
    return 0;
}
