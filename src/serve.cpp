/**
 * serve.cpp — HTTP front-end over the paged engine (S2.4).
 *
 * One engine thread runs the continuous-batching loop from paged_runtime; HTTP
 * connection threads push requests into a queue and hand over a socket. Tokens
 * are streamed back as they are produced.
 *
 * ── Endpoints ────────────────────────────────────────────────────────────────
 *   POST /v1/completions   OpenAI-compatible (SSE, `data: [DONE]` sentinel)
 *   POST /generate         simpler shape for curl
 *
 * OpenAI compatibility is worth the ~40 lines: openai-python, LangChain and any
 * chat UI work against it unmodified, and it is what vLLM/TGI/llama.cpp all
 * speak. /v1/chat/completions is NOT implemented — it needs the model's chat
 * template applied to a messages array.
 *
 * ── Threading ────────────────────────────────────────────────────────────────
 * All CUDA lives on the engine thread. HTTP threads only touch the request queue
 * (mutex) and the socket. A Stream is shared_ptr so the engine can outlive the
 * connection thread that accepted it.
 *
 *   curl -N localhost:8080/v1/completions \
 *     -d '{"prompt":"The capital of France is","max_tokens":32}'
 */

#include "model_config.h"
#include "weights.h"
#include "block_allocator.h"
#include "decode_layer.h"
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
    int len = 0;
    int max_new = 64;
    std::shared_ptr<http::Stream> stream;
    Clock::time_point t_submit, t_first;
    bool header_sent = false;
    bool hit_eos = false;

    bool prefilling() const { return len < (int)prompt.size(); }
    int  next_input() const { return prefilling() ? prompt[len] : output.back(); }
};

int main(int argc, char** argv) {
    std::string wdir = "weights", tok_path = "onnx/tokenizer.json",
                cfg_path = "onnx/config.json",
                model_name = "llama-3.2-1b-instruct";
    int    port = 8080, block_size = 16, max_batch = 32, max_seq = 2048;
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
        else if (a == "--kv-budget-mb" && i+1<argc) kv_budget_mb = std::stod(argv[++i]);
    }

    ModelConfig cfg;  cfg.load(cfg_path, max_seq);  cfg.print();
    Tokenizer   tok;  tok.load(tok_path);
    Weights weights;  weights.load(wdir, cfg);

    const size_t bytes_per_block = (size_t)cfg.num_layers * 2 * cfg.num_kv_heads
                                 * block_size * cfg.head_dim * sizeof(__half);
    const int num_blocks = (int)((kv_budget_mb * 1e6) / bytes_per_block);
    const int max_blocks_per_seq = (max_seq + block_size - 1) / block_size;
    const size_t pool_per_layer = (size_t)num_blocks * cfg.num_kv_heads
                                * block_size * cfg.head_dim;

    __half *k_pool, *v_pool, *x, *logits;
    CK(cudaMalloc(&k_pool, pool_per_layer * cfg.num_layers * sizeof(__half)));
    CK(cudaMalloc(&v_pool, pool_per_layer * cfg.num_layers * sizeof(__half)));
    CK(cudaMalloc(&x,      (size_t)max_batch * cfg.hidden_dim * sizeof(__half)));
    CK(cudaMalloc(&logits, (size_t)max_batch * cfg.vocab_size * sizeof(__half)));
    int *d_table, *d_lens, *d_pos, *d_ids;
    CK(cudaMalloc(&d_table, (size_t)max_batch * max_blocks_per_seq * sizeof(int)));
    CK(cudaMalloc(&d_lens,  (size_t)max_batch * sizeof(int)));
    CK(cudaMalloc(&d_pos,   (size_t)max_batch * sizeof(int)));
    CK(cudaMalloc(&d_ids,   (size_t)max_batch * sizeof(int)));

    DecodeScratch scratch;  scratch.alloc(cfg, max_batch);
    BlockAllocator alloc;   alloc.configure({block_size, num_blocks, 0});
    BatchedPicker picker;   picker.alloc(cfg.vocab_size, max_batch);
    cublasHandle_t blas;    cublasCreate(&blas);

    // ── Shared queue between HTTP threads and the engine ─────────────────────
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

    printf("Serving:   http://0.0.0.0:%d\n", port);
    printf("           POST /v1/completions   (OpenAI-compatible, streaming)\n");
    printf("           POST /generate         (simple)\n");
    printf("KV pool:   %.0f MB → %d blocks of %d tokens\n\n",
           kv_budget_mb, num_blocks, block_size);

    // ── Engine loop ──────────────────────────────────────────────────────────
    std::vector<Seq> waiting, running;
    while (true) {
        // Drain the queue; block only when there is nothing at all to do.
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

        // ADMIT
        while (!waiting.empty() && (int)running.size() < max_batch
               && alloc.can_admit((int)waiting.front().prompt.size())) {
            Seq s = std::move(waiting.front());
            waiting.erase(waiting.begin());
            alloc.allocate(s.id, (int)s.prompt.size());
            running.push_back(std::move(s));
        }
        if (running.empty()) continue;

        // PREEMPT — evict newest until this step's growth fits (see STAGE2B.md)
        {
            auto need = [&] {
                int n = 0;
                for (const auto& s : running) if (s.len % block_size == 0) ++n;
                return n;
            };
            while (alloc.num_free() < need() && running.size() > 1) {
                Seq victim = std::move(running.back());
                running.pop_back();
                alloc.release(victim.id);
                victim.len = 0;
                victim.output.clear();
                waiting.push_back(std::move(victim));   // re-admitted later
            }
        }

        const int B = (int)running.size();
        std::vector<int> h_ids(B), h_pos(B), h_lens(B);
        for (int r = 0; r < B; ++r) {
            h_ids[r]  = running[r].next_input();
            h_pos[r]  = running[r].len;
            h_lens[r] = running[r].len + 1;
        }
        CK(cudaMemcpy(d_ids,  h_ids.data(),  B*sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos,  h_pos.data(),  B*sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_lens, h_lens.data(), B*sizeof(int), cudaMemcpyHostToDevice));

        for (int r = 0; r < B; ++r) alloc.append_token(running[r].id, running[r].len);

        std::vector<uint64_t> ids_of_running;
        for (const auto& s : running) ids_of_running.push_back(s.id);
        const std::vector<int> table = alloc.flatten(ids_of_running, max_blocks_per_seq);
        CK(cudaMemcpy(d_table, table.data(), table.size()*sizeof(int),
                      cudaMemcpyHostToDevice));

        const int max_len = *std::max_element(h_lens.begin(), h_lens.end());
        forward_decode(cfg, weights, blas, scratch, d_ids, x, logits,
                       k_pool, v_pool, d_table, d_lens, d_pos,
                       B, block_size, num_blocks, max_blocks_per_seq, max_len, 0);

        std::vector<int> out;
        picker.argmax_batched(logits, B, out, 0);

        // COMMIT + stream
        for (int r = 0; r < B; ++r) {
            Seq& s = running[r];
            s.len += 1;
            if (s.prefilling()) continue;          // still reading the prompt

            if (s.output.empty()) {
                s.t_first = Clock::now();
                if (s.stream->alive()) { s.stream->begin(); s.header_sent = true; }
            }
            s.output.push_back(out[r]);
            s.hit_eos = (out[r] == tok.eos_id() || out[r] == tok.eot_id());
            // Stream the token's own text, not a re-decode of the whole output —
            // BPE pieces concatenate, so per-token decode is correct here.
            if (!s.hit_eos && s.stream->alive())
                s.stream->token(tok.decode({out[r]}));
        }

        // RETIRE — finished, or the client hung up (cancellation)
        for (int r = B - 1; r >= 0; --r) {
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
    scratch.free(); weights.free(); picker.free();
    cublasDestroy(blas);
    return 0;
}
