/**
 * batch_runtime.cpp — continuous-batching runtime (Stage 2A).
 *
 * Stage 1's runtime.cpp serves one request start-to-finish. This one keeps N
 * requests in flight and lets new ones join mid-decode.
 *
 * ── The loop ─────────────────────────────────────────────────────────────────
 *   begin_step()
 *   if scheduler wants a request in AND a slot is free:
 *       prefill it       (batch 1, profile 0)   ← stalls the decode batch
 *       install its KV into the slot
 *   decode one token for every slot (batch B, profile 1)
 *   sample B tokens, record them, retire anyone who finished
 *   commit_step()        (scatter kernel + lens += 1)
 *
 * ── Why prefill is a separate enqueue ────────────────────────────────────────
 * The batch shares one `seq` dimension. A joining request wants seq=N while the
 * decoding slots want seq=1, and no single shape serves both. Real chunked
 * prefill packs both into one flat sequence with varlen attention — which needs
 * a custom kernel, i.e. Stage 2B. So in 2A prefill runs alone and the batch
 * waits (~12 ms for 512 tokens vs ~5 ms per decode step).
 *
 * Build:  cmake --build build -j    →  ./build/batch_runtime
 */

#include "tokenizer.h"
#include "argmax.cuh"
#include "model_config.h"
#include "batch_kv_cache.h"
#include "scheduler.h"
#include "sampling.cuh"
#include "batched_pick.cuh"

#include <NvInfer.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int PROFILE_PREFILL = 0;
static constexpr int PROFILE_DECODE  = 1;

#define CUDA_CHECK(x)                                                     \
    do { cudaError_t e=(x); if(e!=cudaSuccess){                           \
        fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,              \
                cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(x)                                                   \
    do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){              \
        fprintf(stderr,"cuBLAS %d %s:%d\n",(int)s,__FILE__,__LINE__);     \
        exit(1); } } while(0)

struct Logger : nvinfer1::ILogger {
    void log(Severity sev, const char* msg) noexcept override {
        if (sev <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
    }
} gLogger;

// ── Engine wrapper (same as Stage 1, plus two contexts) ──────────────────────
struct Engine {
    nvinfer1::ICudaEngine*       engine = nullptr;
    // Two contexts so prefill and decode can each hold their own profile —
    // switching profiles on one context every step would thrash.
    nvinfer1::IExecutionContext* prefill_ctx = nullptr;
    nvinfer1::IExecutionContext* decode_ctx  = nullptr;
    cudaStream_t                 stream = nullptr;
    std::string                  hidden_out;
    std::vector<std::string>     pres_k, pres_v, past_k, past_v;

    void load(const char* path) {
        // TODO: same as runtime.cpp Engine::load — deserialize, collect and sort
        //       the past_/pres_ tensor names, create BOTH contexts, and call
        //       setOptimizationProfileAsync(PROFILE_PREFILL / PROFILE_DECODE)
        //       on the respective context once here (not per step).
        
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error(std::string("Cannot open engine: ") + path);
        std::vector<char> buf((std::istreambuf_iterator<char>(f)), {});
        auto* rt  = nvinfer1::createInferRuntime(gLogger);
        engine = rt->deserializeCudaEngine(buf.data(), buf.size());
        if (!engine) throw std::runtime_error("deserializeCudaEngine failed");
        prefill_ctx = engine->createExecutionContext();
        decode_ctx  = engine->createExecutionContext();
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
        prefill_ctx->setOptimizationProfileAsync(PROFILE_PREFILL, stream);
        decode_ctx->setOptimizationProfileAsync(PROFILE_DECODE, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Each execution context reserves activation memory for the LARGEST
        // shape in its profile — here batch 32 x seq 2048 — not for the shape
        // you actually run. Two contexts pay it twice.
        printf("Engine:    loaded — %d profiles, hidden='%s', %zu KV pairs\n",
               engine->getNbOptimizationProfiles(), hidden_out.c_str(),
               pres_k.size());
        printf("Engine:    context memory %zu MB x2 contexts\n",
               engine->getDeviceMemorySize() >> 20);
    }
};

// ── Buffers shared by both phases ────────────────────────────────────────────
struct Buffers {
    int64_t* d_input_ids    = nullptr;   // [max_batch * max_seq]
    int64_t* d_position_ids = nullptr;   // [max_batch * max_seq]
    int64_t* d_attn_mask    = nullptr;   // [max_batch * (max_seq + max_seq)]
    __half*  d_hidden       = nullptr;   // [max_batch * max_seq * H]
    __half*  d_logits       = nullptr;   // [max_batch * V]
    __half*  d_lm_head      = nullptr;   // [H * V]
    __half*  d_dummy        = nullptr;   // 1 element; a valid non-null address
                                         // for the EMPTY past KV during prefill

    // Scratch for the standalone prefill enqueue: TRT writes the new request's
    // KV here, then BatchKVCache::install_prefill copies it into the slot.
    // Sized for batch 1, max_seq tokens.
    std::vector<__half*> prefill_k, prefill_v;

    int hidden_dim = 0, vocab_size = 0;

    void alloc(const ModelConfig& cfg, int max_batch, const char* lmh_path) {
        // TODO
        //  - the six device buffers above, sized for max_batch
        //  - prefill_k/v: one cudaMalloc per layer of
        //      1 * num_kv_heads * max_seq * head_dim * sizeof(__half)
        //  - load lm_head_weight.bin exactly as runtime.cpp does
        int max_seq = cfg.max_seq;
        int mb = max_batch;
        hidden_dim = cfg.hidden_dim;
        vocab_size = cfg.vocab_size;
        CUDA_CHECK(cudaMalloc(&d_input_ids, mb * max_seq * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_position_ids, mb * max_seq * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_attn_mask, mb * (max_seq + max_seq) * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_hidden, mb * max_seq * cfg.hidden_dim * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_logits, mb * cfg.vocab_size * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_lm_head, (size_t)cfg.hidden_dim * cfg.vocab_size * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(__half)));

        const size_t lmh_n = (size_t)cfg.hidden_dim * cfg.vocab_size;
        std::vector<__half> h_w(lmh_n);
        FILE* f = fopen(lmh_path, "rb");
        if (!f) throw std::runtime_error(std::string("Cannot open ") + lmh_path);
        if (fread(h_w.data(), sizeof(__half), lmh_n, f) != lmh_n)
            throw std::runtime_error("lm_head size mismatch");
        fclose(f);
        CUDA_CHECK(cudaMemcpy(d_lm_head, h_w.data(), lmh_n * sizeof(__half),
                              cudaMemcpyHostToDevice));
        prefill_k.resize(cfg.num_layers);
        prefill_v.resize(cfg.num_layers);
        for (int l = 0; l < cfg.num_layers; ++l) {
            size_t sz = (size_t)cfg.num_kv_heads * cfg.max_seq * cfg.head_dim * sizeof(__half);
            CUDA_CHECK(cudaMalloc(&prefill_k[l], sz));
            CUDA_CHECK(cudaMalloc(&prefill_v[l], sz));  


        }
    }
};

// ── lm_head + pick a token from one row of d_hidden ──────────────────────────
int pick_token(cublasHandle_t blas, GpuArgmax& argmax, GpuSampler& sampler,
               const SamplingParams& sp, Buffers& buf, int row,
               cudaStream_t stream) {
    const int H = buf.hidden_dim, V = buf.vocab_size;
    __half* hidden = buf.d_hidden + (size_t)row * H;
    __half one = __float2half(1.f), zero = __float2half(0.f);
    CUBLAS_CHECK(cublasHgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
        V, 1, H,
        &one,  buf.d_lm_head, V,
               hidden,        H,
        &zero, buf.d_logits,  V));
    return sp.greedy() ? argmax.run(buf.d_logits, V, stream)
                       : sampler.sample(buf.d_logits, sp, stream);
}

// ── Prefill one request: batch 1, profile 0, past = 0 ────────────────────────
// Returns the sampled first token. Leaves the request's KV in buf.prefill_k/v,
// laid out as [1, H, n, D] — stride n, which is what install_prefill expects.
int prefill(Engine& eng, Buffers& buf, const ModelConfig& cfg,
            const std::vector<int>& prompt, cublasHandle_t blas,
            GpuArgmax& argmax, GpuSampler& sampler, const SamplingParams& sp) {
    const int n = (int)prompt.size();
    auto* ctx = eng.prefill_ctx;

    // 1. inputs — batch 1. The mask is [1, n] because past = 0; only decode
    //    needs the max_seq + 1 form.
    std::vector<int64_t> ids(n), pos(n), mask(n, 1LL);
    for (int i = 0; i < n; ++i) { ids[i] = prompt[i]; pos[i] = i; }
    CUDA_CHECK(cudaMemcpyAsync(buf.d_input_ids, ids.data(),
                               n * sizeof(int64_t),
                               cudaMemcpyHostToDevice, eng.stream));
    CUDA_CHECK(cudaMemcpyAsync(buf.d_position_ids, pos.data(),
                               n * sizeof(int64_t),
                               cudaMemcpyHostToDevice, eng.stream));
    CUDA_CHECK(cudaMemcpyAsync(buf.d_attn_mask, mask.data(),
                               n * sizeof(int64_t),
                               cudaMemcpyHostToDevice, eng.stream));

    ctx->setInputShape("input_ids",      nvinfer1::Dims2{1, n});
    ctx->setInputShape("position_ids",   nvinfer1::Dims2{1, n});
    ctx->setInputShape("attention_mask", nvinfer1::Dims2{1, n});
    ctx->setTensorAddress("input_ids",      buf.d_input_ids);
    ctx->setTensorAddress("position_ids",   buf.d_position_ids);
    ctx->setTensorAddress("attention_mask", buf.d_attn_mask);

    // 2. past is EMPTY ([1,H,0,D]) but still needs a valid non-null address, and
    //    it must NOT alias the present buffers — TRT would read and write the
    //    same memory. d_dummy exists purely to satisfy the address check.
    const nvinfer1::Dims4 past_shape{1, cfg.num_kv_heads, 0, cfg.head_dim};
    for (int l = 0; l < cfg.num_layers; ++l) {
        ctx->setInputShape(eng.past_k[l].c_str(), past_shape);
        ctx->setInputShape(eng.past_v[l].c_str(), past_shape);
        ctx->setTensorAddress(eng.past_k[l].c_str(), buf.d_dummy);
        ctx->setTensorAddress(eng.past_v[l].c_str(), buf.d_dummy);
    }

    // 3. present → the scratch install_prefill will copy out of
    for (int l = 0; l < cfg.num_layers; ++l) {
        ctx->setTensorAddress(eng.pres_k[l].c_str(), buf.prefill_k[l]);
        ctx->setTensorAddress(eng.pres_v[l].c_str(), buf.prefill_v[l]);
    }
    ctx->setTensorAddress(eng.hidden_out.c_str(), buf.d_hidden);

    if (!ctx->enqueueV3(eng.stream))
        throw std::runtime_error("prefill enqueueV3 failed");

    // 4. only the LAST position predicts the next token
    return pick_token(blas, argmax, sampler, sp, buf, n - 1, eng.stream);
}

 
// ── One decode step over all active slots ────────────────────────────────────
// Fills `out_tokens[slot]` for every slot in [0, B).
void decode_step(Engine& eng, Buffers& buf, BatchKVCache& kv,
                 const ModelConfig& cfg, const std::vector<int>& in_tokens,
                 std::vector<int>& out_tokens, cublasHandle_t blas,
                 GpuArgmax& argmax, GpuSampler& sampler,
                 BatchedPicker& picker, const SamplingParams& sp) {
    const int B = kv.batch_size();
    if (B == 0) return;
    auto* ctx = eng.decode_ctx;

    // 1. inputs. Every one of these is [B, ...] and B must match what bind()
    //    declares — a disagreement is either a TRT rejection or a silent
    //    misread of another slot's data.
    std::vector<int64_t> ids(B);
    for (int b = 0; b < B; ++b) ids[b] = in_tokens[b];
    const std::vector<int64_t> pos  = kv.build_positions(1);   // [B, 1]
    const std::vector<int64_t> mask = kv.build_mask(1);        // [B, max_seq+1]

    CUDA_CHECK(cudaMemcpyAsync(buf.d_input_ids, ids.data(),
                               ids.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, eng.stream));
    CUDA_CHECK(cudaMemcpyAsync(buf.d_position_ids, pos.data(),
                               pos.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, eng.stream));
    CUDA_CHECK(cudaMemcpyAsync(buf.d_attn_mask, mask.data(),
                               mask.size() * sizeof(int64_t),
                               cudaMemcpyHostToDevice, eng.stream));

    ctx->setInputShape("input_ids",      nvinfer1::Dims2{B, 1});
    ctx->setInputShape("position_ids",   nvinfer1::Dims2{B, 1});
    ctx->setInputShape("attention_mask", nvinfer1::Dims2{B, cfg.max_seq + 1});
    ctx->setTensorAddress("input_ids",      buf.d_input_ids);
    ctx->setTensorAddress("position_ids",   buf.d_position_ids);
    ctx->setTensorAddress("attention_mask", buf.d_attn_mask);

    // 2. past = our cache (always [B,H,max_seq,D]), present = TRT scratch
    kv.bind(ctx, eng.past_k, eng.past_v, eng.pres_k, eng.pres_v);
    ctx->setTensorAddress(eng.hidden_out.c_str(), buf.d_hidden);

    // 3. run
    if (!ctx->enqueueV3(eng.stream))
        throw std::runtime_error("decode enqueueV3 failed");

    // 4. ONE lm_head GEMM for all B rows, then ONE segmented argmax.
    //
    //    The GEMM needs no batching API: cuBLAS column-major C(v,b) at v + b*V
    //    is exactly row-major [B,V] at b*V + v, and likewise for d_hidden. So
    //    the only change from the single-row version is n = 1 → n = B.
    if (sp.greedy()) {
        const int H = buf.hidden_dim, V = buf.vocab_size;
        __half one = __float2half(1.f), zero = __float2half(0.f);
        CUBLAS_CHECK(cublasHgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
            V, B, H,                       // n = B, not 1
            &one,  buf.d_lm_head, V,
                   buf.d_hidden,  H,
            &zero, buf.d_logits,  V));
        picker.argmax_batched(buf.d_logits, B, out_tokens, eng.stream);
    } else {
        // Sampling still goes row-by-row (segmented sort + scan is future work).
        for (int b = 0; b < B; ++b)
            out_tokens[b] = pick_token(blas, argmax, sampler, sp, buf, b,
                                       eng.stream);
    }

    // 5. move each slot's new KV entry from present into our cache at lens[b],
    //    then advance lens.
    kv.commit_step(eng.stream);
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::string engine_path, lmhead_path, tokenizer_path, config_path, prompts_path;
    int max_batch = 4, max_seq = 512, max_new_tokens = 64, admits_per_step = 1;
    SamplingParams sp;
    sp.temperature = 0.0f;   // greedy by default
    bool json_out = false, dump_outputs = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--engine"          && i+1<argc) engine_path     = argv[++i];
        else if (a == "--lm-head"         && i+1<argc) lmhead_path     = argv[++i];
        else if (a == "--tokenizer"       && i+1<argc) tokenizer_path  = argv[++i];
        else if (a == "--config"          && i+1<argc) config_path     = argv[++i];
        else if (a == "--prompts"         && i+1<argc) prompts_path    = argv[++i];
        else if (a == "--max-batch"       && i+1<argc) max_batch       = std::stoi(argv[++i]);
        else if (a == "--max-seq"         && i+1<argc) max_seq         = std::stoi(argv[++i]);
        else if (a == "--max-new-tokens"  && i+1<argc) max_new_tokens  = std::stoi(argv[++i]);
        else if (a == "--admits-per-step" && i+1<argc) admits_per_step = std::stoi(argv[++i]);
        else if (a == "--temperature"     && i+1<argc) sp.temperature  = std::stof(argv[++i]);
        else if (a == "--top-k"           && i+1<argc) sp.top_k        = std::stoi(argv[++i]);
        else if (a == "--top-p"           && i+1<argc) sp.top_p        = std::stof(argv[++i]);
        else if (a == "--json")                       json_out        = true;
        else if (a == "--dump-outputs")               dump_outputs    = true;
    }
    if (config_path.empty() && !tokenizer_path.empty()) {
        size_t slash = tokenizer_path.find_last_of("/\\");
        config_path = (slash == std::string::npos)
                    ? "config.json"
                    : tokenizer_path.substr(0, slash + 1) + "config.json";
    }
    if (engine_path.empty() || lmhead_path.empty() || tokenizer_path.empty()) {
        fprintf(stderr,
            "Usage: %s --engine <.trt> --lm-head <.bin> --tokenizer <tokenizer.json>\n"
            "       [--config <config.json>] [--prompts <file>] [--max-batch N]\n"
            "       [--max-seq N] [--max-new-tokens N] [--admits-per-step N]\n"
            "       [--temperature T] [--top-k K] [--top-p P]\n", argv[0]);
        return 1;
    }

    printf("\n=== llm_server Stage 2A — continuous batching ===\n\n");

    ModelConfig cfg;
    cfg.load(config_path, max_seq);
    cfg.print();

    Tokenizer tok;
    tok.load(tokenizer_path);

    Engine eng;
    eng.load(engine_path.c_str());

    Buffers buf;
    buf.alloc(cfg, max_batch, lmhead_path.c_str());

    BatchKVCache kv;
    kv.alloc({cfg.num_layers, cfg.num_kv_heads, cfg.head_dim,
              cfg.max_seq, max_batch});
    printf("KV cache:  %zu MB (%d slots x %d ctx)\n",
           kv.bytes() >> 20, max_batch, cfg.max_seq);

    Scheduler sched;
    sched.configure({admits_per_step, max_batch});

    // Device memory after everything is allocated. If `used` approaches total,
    // WDDM will page device memory to host RAM over PCIe — which shows up as a
    // sudden collapse in TPOT rather than an allocation failure.
    {
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        printf("VRAM:      %zu / %zu MB used\n",
               (total_b - free_b) >> 20, total_b >> 20);
    }

    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));
    CUBLAS_CHECK(cublasSetStream(blas, eng.stream));
    GpuArgmax argmax;  argmax.alloc(cfg.vocab_size);
    GpuSampler sampler;
    if (!sp.greedy()) sampler.alloc(cfg.vocab_size, sp.seed);
    BatchedPicker picker;  picker.alloc(cfg.vocab_size, max_batch);

    // ── Submit the workload ──────────────────────────────────────────────────
    std::vector<std::string> prompts;
    if (!prompts_path.empty()) {
        std::ifstream pf(prompts_path);
        if (!pf) { fprintf(stderr, "Cannot open %s\n", prompts_path.c_str()); return 1; }
        for (std::string line; std::getline(pf, line); )
            if (!line.empty()) prompts.push_back(line);
    } else {
        prompts = {
            "The key insight about transformers is",
            "Once upon a time",
            "The capital of France is",
            "def fibonacci(n):",
            "In 1969, humans first",
            "The three laws of robotics are",
        };
    }
    for (const auto& p : prompts) {
        Request r;
        r.prompt         = tok.encode(p, /*add_bos=*/true);
        r.sp             = sp;
        r.max_new_tokens = max_new_tokens;
        sched.submit(std::move(r));
    }
    printf("Workload:  %zu requests, %d slots, %d admit/step\n\n",
           prompts.size(), max_batch, admits_per_step);

    // ── The loop ─────────────────────────────────────────────────────────────
    const auto t0 = std::chrono::steady_clock::now();
    long long steps = 0, tokens_out = 0;

    while (sched.has_work()) {
        sched.begin_step();

        // Admission — the scheduler decides (it enforces its own per-step
        // budget), the cache supplies the slot.
        while (Request* r = sched.next_admission(kv.has_free())) {
            const int slot = kv.acquire();      // != -1: has_free() was true
            sched.mark_running(r, slot);

            const int first = prefill(eng, buf, cfg, r->prompt,
                                      blas, argmax, sampler, sp);
            kv.install_prefill(slot, (int)r->prompt.size(),
                               buf.prefill_k, buf.prefill_v, eng.stream);
            ++tokens_out;

            // A short request can finish on its prefill token alone.
            if (sched.on_token(slot, first, tok.eos_id(), tok.eot_id())) {
                kv.release(slot);
                sched.retire(slot);
            }
        }

        const int B = kv.batch_size();
        if (B == 0) continue;                   // nothing running

        auto in = sched.next_tokens(B, /*filler=*/tok.bos_id());
        std::vector<int> out(B, 0);
        decode_step(eng, buf, kv, cfg, in, out, blas, argmax, sampler,
                    picker, sp);
        ++steps;

        // kv.active(s) is the hole check — the cache is the authority on which
        // slots are live.
        for (int s = 0; s < B; ++s) {
            if (!kv.active(s)) continue;        // hole in the batch
            ++tokens_out;
            if (sched.on_token(s, out[s], tok.eos_id(), tok.eot_id())) {
                kv.release(s);
                sched.retire(s);
            }
        }
    }

    const double wall_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // ── Report ───────────────────────────────────────────────────────────────
    std::vector<double> ttfts, tpots;
    for (const auto& r : sched.done()) {
        ttfts.push_back(std::chrono::duration<double,std::milli>(
            r.t_first_token - r.t_submit).count());
        const double gen = std::chrono::duration<double,std::milli>(
            r.t_done - r.t_first_token).count();
        tpots.push_back(r.generated() > 1 ? gen / (r.generated() - 1) : 0.0);
    }
    auto pct = [](std::vector<double> v, double p) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, (size_t)(v.size() * p))];
    };
    const double mean_tpot = tpots.empty() ? 0.0
        : std::accumulate(tpots.begin(), tpots.end(), 0.0) / tpots.size();

    if (dump_outputs) {
        // One line per request, id \t text — for the correctness comparison.
        for (const auto& r : sched.done()) {
            std::string t = tok.decode(r.output);
            for (auto& c : t) if (c == '\n' || c == '\t') c = ' ';
            printf("OUT\t%llu\t%s\n", (unsigned long long)r.id, t.c_str());
        }
    }

    if (json_out) {
        printf("{\"max_batch\":%d,\"admits_per_step\":%d,\"requests\":%zu,"
               "\"steps\":%lld,\"tokens\":%lld,\"wall_ms\":%.1f,"
               "\"tok_s\":%.1f,\"ttft_p50\":%.1f,\"ttft_p95\":%.1f,"
               "\"tpot_mean\":%.2f,\"max_seq\":%d}\n",
               max_batch, admits_per_step, sched.done().size(), steps,
               tokens_out, wall_ms, tokens_out / (wall_ms / 1000.0),
               pct(ttfts, 0.50), pct(ttfts, 0.95), mean_tpot, cfg.max_seq);
        kv.free();
        CUBLAS_CHECK(cublasDestroy(blas));
        return 0;
    }

    printf("\n──────────────────────────────────────────────────────\n");
    for (size_t i = 0; i < sched.done().size(); ++i) {
        const auto& r = sched.done()[i];
        printf("[%llu] TTFT %7.1f ms  TPOT %5.2f ms  %d tok\n     %s\n",
               (unsigned long long)r.id, ttfts[i], tpots[i], r.generated(),
               tok.decode(r.output).c_str());
    }
    printf("──────────────────────────────────────────────────────\n");
    printf("  %zu requests, %lld decode steps, %lld tokens\n",
           sched.done().size(), steps, tokens_out);
    printf("  TTFT p50 %.1f ms  p95 %.1f ms   TPOT mean %.2f ms\n",
           pct(ttfts, 0.50), pct(ttfts, 0.95), mean_tpot);
    printf("  wall %.0f ms  →  %.1f tok/s aggregate\n",
           wall_ms, tokens_out / (wall_ms / 1000.0));
    printf("──────────────────────────────────────────────────────\n\n");

    kv.free();
    CUBLAS_CHECK(cublasDestroy(blas));
    return 0;
}
