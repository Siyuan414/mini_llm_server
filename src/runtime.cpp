/**
 * runtime.cpp — llm_server Stage 1: single-request engine with TRUE batched prefill.
 *
 * Changes vs the llama_jetson runtime it started from:
 *   ✓ Batched prefill — whole prompt in ONE enqueueV3 (was: token-by-token loop)
 *   ✓ Two-profile engine: profile 0 = prefill (seq 1..2048, past=0),
 *                          profile 1 = decode  (seq=1, past 0..2047)
 *   ✓ Context 64 → 2048 tokens
 *   ✓ Attention mask lives on device, filled with 1s once at init
 *     (was: host std::vector rebuilt + uploaded every step)
 *
 * Build:  cmake -B build && cmake --build build -j
 *
 * Run (from llm_server/):
 *   ./build/runtime \
 *       --engine    engine/llama1b_fp16.trt \
 *       --lm-head   onnx/lm_head_weight.bin \
 *       --tokenizer onnx/tokenizer.json \
 *       --prompt    "The key insight about transformers is" \
 *       --max-new-tokens 128
 */

#include "tokenizer.h"
#include "argmax.cuh"
#include "kv_cache.h"
#include "model_config.h"
#include "sampling.cuh"

#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Model dimensions come from config.json at startup (see model_config.h).
static constexpr int DEFAULT_MAX_SEQ = 2048;

static constexpr int PROFILE_PREFILL = 0;
static constexpr int PROFILE_DECODE  = 1;

// ── Helpers ───────────────────────────────────────────────────────────────────
#define CUDA_CHECK(x)                                                     \
    do { cudaError_t e=(x); if(e!=cudaSuccess){                           \
        fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,              \
                cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x)                                                   \
    do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){              \
        fprintf(stderr,"cuBLAS error %d %s:%d\n",(int)s,__FILE__,__LINE__);\
        exit(1); } } while(0)

using hptr = __half*;

// ── TRT Logger ────────────────────────────────────────────────────────────────
struct Logger : nvinfer1::ILogger {
    void log(Severity sev, const char* msg) noexcept override {
        if (sev <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
} gLogger;

// ── Pre-allocated buffer pool ─────────────────────────────────────────────────
// ALL device memory allocated at startup; prefill + decode make zero mallocs.
struct Buffers {
    int64_t* d_input_ids    = nullptr;   // [max_seq]
    int64_t* d_position_ids = nullptr;   // [max_seq]
    int64_t* d_attn_mask    = nullptr;   // [max_seq] — all 1s, written ONCE

    // KV memory lives in KVCache (see kv_cache.h) — not here.

    hptr d_hidden  = nullptr;            // [max_seq, hidden_dim]
    hptr d_logits  = nullptr;            // [vocab_size]
    hptr d_lm_head = nullptr;            // [hidden_dim, vocab_size]

    int hidden_dim = 0, vocab_size = 0;  // cached for lm_head_argmax

    void alloc(const ModelConfig& cfg, const char* lmh_path) {
        hidden_dim = cfg.hidden_dim;
        vocab_size = cfg.vocab_size;
        const int max_seq = cfg.max_seq;

        CUDA_CHECK(cudaMalloc(&d_input_ids,    max_seq * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_position_ids, max_seq * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_attn_mask,    max_seq * sizeof(int64_t)));

        // Mask is all-ones for batch=1 (no padding): fill once, never touch again.
        {
            std::vector<int64_t> ones(max_seq, 1LL);
            CUDA_CHECK(cudaMemcpy(d_attn_mask, ones.data(),
                                  max_seq * sizeof(int64_t),
                                  cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMalloc(&d_hidden,
                              (size_t)max_seq * hidden_dim * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_logits, (size_t)vocab_size * sizeof(__half)));

        // lm_head weights (raw fp16 [hidden_dim, vocab_size])
        size_t lmh_n = (size_t)hidden_dim * vocab_size;
        std::vector<__half> h_w(lmh_n);
        FILE* f = fopen(lmh_path, "rb");
        if (!f) throw std::runtime_error(std::string("Cannot open ") + lmh_path);
        if (fread(h_w.data(), sizeof(__half), lmh_n, f) != lmh_n)
            throw std::runtime_error("lm_head file size mismatch — does it match "
                                     "config.json hidden_size x vocab_size?");
        fclose(f);
        CUDA_CHECK(cudaMalloc(&d_lm_head, lmh_n * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_lm_head, h_w.data(), lmh_n * sizeof(__half),
                              cudaMemcpyHostToDevice));

        printf("Buffers:   lm_head %.0f MB — 0 mallocs after init\n",
               (double)(lmh_n * sizeof(__half)) / (1 << 20));
    }
};

// ── TRT Engine wrapper ────────────────────────────────────────────────────────
struct Engine {
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    std::string                  hidden_out;
    std::vector<std::string>     pres_k, pres_v;   // outputs
    std::vector<std::string>     past_k, past_v;   // inputs
    int                          cur_profile = -1;

    void load(const char* path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error(std::string("Cannot open engine: ") + path);
        std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});
        auto* rt  = nvinfer1::createInferRuntime(gLogger);
        engine    = rt->deserializeCudaEngine(buf.data(), buf.size());
        if (!engine) throw std::runtime_error("deserializeCudaEngine failed");
        context = engine->createExecutionContext();
        CUDA_CHECK(cudaStreamCreate(&stream));

        int n = engine->getNbIOTensors();
        for (int i = 0; i < n; ++i) {
            std::string nm = engine->getIOTensorName(i);
            const bool is_out = engine->getTensorIOMode(nm.c_str())
                                == nvinfer1::TensorIOMode::kOUTPUT;
            const bool is_key = nm.find(".key")   != std::string::npos;
            const bool is_val = nm.find(".value") != std::string::npos;
            if (is_out) {
                if      (is_key) pres_k.push_back(nm);
                else if (is_val) pres_v.push_back(nm);
                else             hidden_out = nm;
            } else {
                if      (is_key) past_k.push_back(nm);
                else if (is_val) past_v.push_back(nm);
            }
        }
        auto by_layer = [](const std::string& a, const std::string& b) {
            auto idx = [](const std::string& s) {
                auto d1 = s.find('.') + 1, d2 = s.find('.', d1);
                return std::stoi(s.substr(d1, d2 - d1));
            };
            return idx(a) < idx(b);
        };
        std::sort(pres_k.begin(), pres_k.end(), by_layer);
        std::sort(pres_v.begin(), pres_v.end(), by_layer);
        std::sort(past_k.begin(), past_k.end(), by_layer);
        std::sort(past_v.begin(), past_v.end(), by_layer);
        printf("Engine:    loaded — %d profiles, hidden='%s', %zu KV pairs\n",
               engine->getNbOptimizationProfiles(), hidden_out.c_str(),
               pres_k.size());
    }

    void use_profile(int p) {
        if (p == cur_profile) return;
        if (!context->setOptimizationProfileAsync(p, stream))
            throw std::runtime_error("setOptimizationProfileAsync failed");
        cur_profile = p;
    }
};

// ── Unified forward: n tokens appended at the cache's current position ────────
// kv.length()==0 → prefill profile (any n).  >0 → decode profile (n must be 1).
void forward(Engine& eng, Buffers& buf, KVCache& kv, const int* tokens, int n) {
    const int kv_t = kv.length();
    if (!kv.fits(n))
        throw std::runtime_error("KV cache overflow");
    if (kv_t > 0 && n != 1)
        throw std::runtime_error("decode profile requires n==1");

    eng.use_profile(kv_t == 0 ? PROFILE_PREFILL : PROFILE_DECODE);
    auto* ctx = eng.context;
    int total = kv_t + n;

    // Upload token ids + positions (mask is already all-1s on device)
    std::vector<int64_t> h_ids(n), h_pos(n);
    for (int i = 0; i < n; ++i) { h_ids[i] = tokens[i]; h_pos[i] = kv_t + i; }
    CUDA_CHECK(cudaMemcpyAsync(buf.d_input_ids, h_ids.data(),
                               n * sizeof(int64_t),
                               cudaMemcpyHostToDevice, eng.stream));
    CUDA_CHECK(cudaMemcpyAsync(buf.d_position_ids, h_pos.data(),
                               n * sizeof(int64_t),
                               cudaMemcpyHostToDevice, eng.stream));

    ctx->setInputShape("input_ids",      nvinfer1::Dims2{1, n});
    ctx->setInputShape("position_ids",   nvinfer1::Dims2{1, n});
    ctx->setInputShape("attention_mask", nvinfer1::Dims2{1, total});
    ctx->setTensorAddress("input_ids",      buf.d_input_ids);
    ctx->setTensorAddress("position_ids",   buf.d_position_ids);
    ctx->setTensorAddress("attention_mask", buf.d_attn_mask);

    kv.bind(ctx, eng.past_k, eng.past_v, eng.pres_k, eng.pres_v);
    ctx->setTensorAddress(eng.hidden_out.c_str(), buf.d_hidden);

    if (!ctx->enqueueV3(eng.stream))
        throw std::runtime_error("enqueueV3 failed");

    kv.commit(n);   // present becomes next step's past — no copy
}

// ── lm_head + GPU argmax on ONE position of the hidden buffer ─────────────────
// Projects hidden[position] through lm_head, then picks the next token:
// greedy (argmax) or temperature/top-k/top-p sampling. Either way only the
// token id returns to the host.
int next_token(cublasHandle_t blas, GpuArgmax& argmax, GpuSampler& sampler,
               const SamplingParams& sp, Buffers& buf, int position,
               cudaStream_t stream) {
    const int H = buf.hidden_dim, V = buf.vocab_size;
    __half* hidden = buf.d_hidden + (size_t)position * H;
    __half one  = __float2half(1.f);
    __half zero = __float2half(0.f);
    CUBLAS_CHECK(cublasHgemm(blas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        V, 1, H,
        &one,  buf.d_lm_head, V,
               hidden,        H,
        &zero, buf.d_logits,  V));
    return sp.greedy() ? argmax.run(buf.d_logits, V, stream)
                       : sampler.sample(buf.d_logits, sp, stream);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::string engine_path, lmhead_path, tokenizer_path, config_path, prompt;
    int max_new_tokens = 32;
    int max_seq        = DEFAULT_MAX_SEQ;
    SamplingParams sp;
    sp.temperature = 0.0f;   // default greedy — deterministic, matches HF temp=0
    int  warmup        = 1;  // untimed full cycles before measuring
    int  prompt_tokens = 0;  // >0: synthesize a prompt of exactly this many tokens
    bool quiet         = false;
    bool json_out      = false;
    std::string dump_logits;   // write post-prefill logits here (fp32 binary)
    bool print_tokens  = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--engine"         && i+1<argc) engine_path    = argv[++i];
        else if (a == "--lm-head"        && i+1<argc) lmhead_path    = argv[++i];
        else if (a == "--tokenizer"      && i+1<argc) tokenizer_path = argv[++i];
        else if (a == "--config"         && i+1<argc) config_path    = argv[++i];
        else if (a == "--prompt"         && i+1<argc) prompt         = argv[++i];
        else if (a == "--max-new-tokens" && i+1<argc) max_new_tokens = std::stoi(argv[++i]);
        else if (a == "--max-seq"        && i+1<argc) max_seq        = std::stoi(argv[++i]);
        else if (a == "--temperature"    && i+1<argc) sp.temperature = std::stof(argv[++i]);
        else if (a == "--top-k"          && i+1<argc) sp.top_k       = std::stoi(argv[++i]);
        else if (a == "--top-p"          && i+1<argc) sp.top_p       = std::stof(argv[++i]);
        else if (a == "--seed"           && i+1<argc) sp.seed        = std::stoull(argv[++i]);
        else if (a == "--warmup"         && i+1<argc) warmup         = std::stoi(argv[++i]);
        else if (a == "--prompt-tokens"  && i+1<argc) prompt_tokens  = std::stoi(argv[++i]);
        else if (a == "--quiet")                      quiet          = true;
        else if (a == "--json")                       json_out       = true;
        else if (a == "--dump-logits"    && i+1<argc) dump_logits    = argv[++i];
        else if (a == "--print-tokens")               print_tokens   = true;
    }
    if (prompt_tokens > 0 && prompt.empty()) prompt = "x";  // placeholder
    // Default: config.json next to tokenizer.json
    if (config_path.empty() && !tokenizer_path.empty()) {
        size_t slash = tokenizer_path.find_last_of("/\\");
        config_path = (slash == std::string::npos)
                    ? "config.json"
                    : tokenizer_path.substr(0, slash + 1) + "config.json";
    }
    if (engine_path.empty()||lmhead_path.empty()||tokenizer_path.empty()||prompt.empty()){
        fprintf(stderr,
            "Usage: %s --engine <.trt> --lm-head <.bin>"
            " --tokenizer <tokenizer.json> --prompt \"<text>\"\n"
            "          [--config <config.json>] [--max-new-tokens N] [--max-seq N]\n"
            "          [--temperature T] [--top-k K] [--top-p P] [--seed S]\n"
            "          (temperature 0 = greedy, the default)\n",
            argv[0]);
        return 1;
    }

    printf("\n=== llm_server Stage 1 — batched prefill runtime ===\n\n");

    ModelConfig cfg;
    cfg.load(config_path, max_seq);
    cfg.print();

    Tokenizer tokenizer;
    tokenizer.load(tokenizer_path);

    Engine engine;
    engine.load(engine_path.c_str());

    Buffers buf;
    buf.alloc(cfg, lmhead_path.c_str());

    KVCache kv;
    kv.alloc({cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq});
    printf("KV cache:  %zu MB (ctx %d, ping-pong, zero-copy)\n",
           kv.bytes() >> 20, kv.capacity());

    if ((int)engine.past_k.size() != cfg.num_layers)
        fprintf(stderr, "[warn] engine has %zu KV layers, config says %d\n",
                engine.past_k.size(), cfg.num_layers);

    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));
    CUBLAS_CHECK(cublasSetStream(blas, engine.stream));

    GpuArgmax argmax;
    argmax.alloc(cfg.vocab_size);

    GpuSampler sampler;
    if (!sp.greedy()) {
        sampler.alloc(cfg.vocab_size, sp.seed);
        if (!quiet)
            printf("Sampler:   temp %.2f, top-k %d, top-p %.2f, seed %llu (%zu MB)\n",
                   sp.temperature, sp.top_k, sp.top_p,
                   (unsigned long long)sp.seed, sampler.bytes() >> 20);
    } else if (!quiet) {
        printf("Sampler:   greedy (argmax)\n");
    }

    if (!quiet) printf("\n");

    std::vector<int> prompt_ids;
    if (prompt_tokens > 0) {
        // Synthetic prompt of an exact token count, for prefill scaling curves.
        prompt_ids.push_back(tokenizer.bos_id());
        static const int filler[] = {791, 1401, 20616, 922, 87970, 374, 430, 814};
        for (int i = 1; i < prompt_tokens; ++i)
            prompt_ids.push_back(filler[i % 8]);
    } else {
        prompt_ids = tokenizer.encode(prompt, /*add_bos=*/true);
    }
    int n_prompt = (int)prompt_ids.size();
    if (print_tokens) {
        printf("TOKENS:");
        for (int id : prompt_ids) printf(" %d", id);
        printf("\n");
    }
    if (!kv.fits(n_prompt)) {
        fprintf(stderr, "Prompt too long: %d > %d\n", n_prompt, kv.capacity());
        return 1;
    }
    if (!quiet) {
        printf("Prompt : \"%s\"\n", prompt_tokens > 0 ? "<synthetic>" : prompt.c_str());
        printf("Tokens : %d  (with BOS)\n\n", n_prompt);
    }

    // ── Warmup: untimed full cycles (first enqueue per profile pays one-time
    // kernel selection + allocation costs that would otherwise pollute prefill)
    for (int w = 0; w < warmup; ++w) {
        kv.reset();
        forward(engine, buf, kv, prompt_ids.data(), n_prompt);
        int t = next_token(blas, argmax, sampler, sp, buf, n_prompt - 1,
                           engine.stream);
        for (int i = 0; i < 4 && kv.fits(1); ++i) {
            forward(engine, buf, kv, &t, 1);
            t = next_token(blas, argmax, sampler, sp, buf, 0, engine.stream);
        }
    }
    kv.reset();

    // ── Prefill: whole prompt, ONE enqueue ───────────────────────────────────
    auto t_pre0 = std::chrono::steady_clock::now();
    forward(engine, buf, kv, prompt_ids.data(), n_prompt);
    CUDA_CHECK(cudaStreamSynchronize(engine.stream));
    double t_prefill_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t_pre0).count();

    int tok = next_token(blas, argmax, sampler, sp, buf, n_prompt - 1,
                         engine.stream);
    double ttft_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t_pre0).count();

    // Post-prefill logits — the cleanest signal to compare against a reference
    // implementation, since it isolates one forward pass from any drift that
    // accumulates over a generated sequence.
    if (!dump_logits.empty()) {
        std::vector<__half> h(cfg.vocab_size);
        CUDA_CHECK(cudaMemcpy(h.data(), buf.d_logits,
                              (size_t)cfg.vocab_size * sizeof(__half),
                              cudaMemcpyDeviceToHost));
        std::vector<float> f(cfg.vocab_size);
        for (int i = 0; i < cfg.vocab_size; ++i) f[i] = __half2float(h[i]);
        FILE* fp = fopen(dump_logits.c_str(), "wb");
        if (!fp) throw std::runtime_error("cannot write " + dump_logits);
        fwrite(f.data(), sizeof(float), f.size(), fp);
        fclose(fp);
    }
    if (!quiet)
        printf("Prefill : %.1f ms  (%.0f tok/s)   TTFT: %.1f ms\n\n",
               t_prefill_ms, n_prompt / (t_prefill_ms / 1000.0), ttft_ms);

    // ── Decode loop ──────────────────────────────────────────────────────────
    std::vector<int>    generated = {tok};
    std::vector<double> step_ms;

    if (!quiet) {
        printf("Output  : %s", tokenizer.decode({tok}).c_str());
        fflush(stdout);
    }

    for (int s = 0; s < max_new_tokens - 1; ++s) {
        if (!kv.fits(1)) { printf("\n[KV cache full]"); break; }

        auto t0 = std::chrono::steady_clock::now();
        forward(engine, buf, kv, &tok, 1);
        tok = next_token(blas, argmax, sampler, sp, buf, 0, engine.stream);
        step_ms.push_back(std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now() - t0).count());

        generated.push_back(tok);
        if (!quiet) {
            printf("%s", tokenizer.decode({tok}).c_str());
            fflush(stdout);
        }

        if (tok == tokenizer.eos_id() || tok == tokenizer.eot_id())
            break;
    }
    if (!quiet) printf("\n\n");

    // ── Report ────────────────────────────────────────────────────────────────
    {
        int    n      = (int)step_ms.size();
        double avg = 0, p50 = 0, p95 = 0;
        if (n > 0) {
            avg = std::accumulate(step_ms.begin(), step_ms.end(), 0.0) / n;
            auto sorted = step_ms;
            std::sort(sorted.begin(), sorted.end());
            p50 = sorted[n / 2];
            p95 = sorted[std::max(0, (int)(n * 0.95) - 1)];
        }

        if (json_out) {
            printf("{\"prompt_tokens\":%d,\"decode_tokens\":%d,"
                   "\"prefill_ms\":%.3f,\"prefill_tok_s\":%.1f,\"ttft_ms\":%.3f,"
                   "\"decode_ms_avg\":%.3f,\"decode_ms_p50\":%.3f,"
                   "\"decode_ms_p95\":%.3f,\"decode_tok_s\":%.1f,"
                   "\"sampling\":\"%s\"}\n",
                   n_prompt, n, t_prefill_ms,
                   n_prompt / (t_prefill_ms / 1000.0), ttft_ms,
                   avg, p50, p95, p50 > 0 ? 1000.0 / p50 : 0.0,
                   sp.greedy() ? "greedy" : "top-k/top-p");
            return 0;
        }
        if (n == 0) { printf("Full output: \"%s\"\n",
                             tokenizer.decode(generated).c_str()); return 0; }

        printf("==============================================\n");
        printf("  Stage 1 runtime (prefill=%d  decode=%d)\n", n_prompt, n);
        printf("----------------------------------------------\n");
        printf("  prefill : %7.1f ms   (%.0f tok/s)\n", t_prefill_ms,
               n_prompt / (t_prefill_ms / 1000.0));
        printf("  TTFT    : %7.1f ms\n", ttft_ms);
        printf("  decode  : %7.2f ms/tok  →  %5.1f tok/s\n", avg, 1000.0 / avg);
        printf("  p50     : %7.2f ms/tok\n", p50);
        printf("  p95     : %7.2f ms/tok\n", p95);
        printf("==============================================\n\n");
    }

    printf("Full output: \"%s\"\n", tokenizer.decode(generated).c_str());
    return 0;
}
