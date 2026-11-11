/**
 * model_config.h — Load model dimensions from a HuggingFace config.json.
 *
 * Replaces the hardcoded constexpr block in runtime.cpp so the runtime works for
 * any LLaMA-family checkpoint (1B / 3B / 8B, Qwen, Mistral, ...) without a rebuild.
 *
 * Reads only the fields the runtime needs:
 *   num_hidden_layers, hidden_size, num_key_value_heads,
 *   num_attention_heads, vocab_size, max_position_embeddings
 *
 * head_dim is taken from config if present (some models set it explicitly),
 * otherwise derived as hidden_size / num_attention_heads.
 *
 * Hand-rolled scanner, same approach as tokenizer.h — no JSON dependency.
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

struct ModelConfig {
    int num_layers   = 0;
    int hidden_dim   = 0;
    int num_q_heads  = 0;
    int num_kv_heads = 0;
    int head_dim     = 0;
    int vocab_size   = 0;
    int max_pos      = 0;   // model's trained context length
    int inter_dim    = 0;   // MLP intermediate size (S2.7)
    float rms_eps    = 1e-5f;
    float rope_theta = 500000.f;   // LLaMA-3.2; older LLaMA used 10000

    // max_seq: what we actually allocate. Clamped to the model's max_pos.
    int max_seq      = 0;

    void load(const std::string& path, int requested_max_seq) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Cannot open config: " + path);
        std::stringstream ss; ss << f.rdbuf();
        const std::string src = ss.str();

        num_layers   = get_int(src, "num_hidden_layers");
        hidden_dim   = get_int(src, "hidden_size");
        num_q_heads  = get_int(src, "num_attention_heads");
        num_kv_heads = get_int(src, "num_key_value_heads", num_q_heads); // MHA fallback
        vocab_size   = get_int(src, "vocab_size");
        max_pos      = get_int(src, "max_position_embeddings", 0);
        inter_dim    = get_int(src, "intermediate_size", 0);
        rms_eps      = get_float(src, "rms_norm_eps", 1e-5f);
        rope_theta   = get_float(src, "rope_theta", 500000.f);

        int cfg_head_dim = get_int(src, "head_dim", 0);
        head_dim = cfg_head_dim > 0 ? cfg_head_dim
                                    : hidden_dim / std::max(num_q_heads, 1);

        if (num_layers <= 0 || hidden_dim <= 0 || vocab_size <= 0 || head_dim <= 0)
            throw std::runtime_error("config.json missing required fields");

        max_seq = requested_max_seq;
        if (max_pos > 0 && max_seq > max_pos) max_seq = max_pos;
    }

    void print() const {
        printf("Config:    %d layers, hidden %d, %dQ/%dKV heads x %d, "
               "vocab %d, ctx %d\n",
               num_layers, hidden_dim, num_q_heads, num_kv_heads, head_dim,
               vocab_size, max_seq);
    }

private:
    // Same scan as get_int but parses a float (rms_norm_eps is 1e-5, rope_theta
    // can be written as 500000.0).
    static float get_float(const std::string& s, const std::string& key,
                           float def) {
        const std::string pat = "\"" + key + "\"";
        size_t p = s.find(pat);
        if (p == std::string::npos) return def;
        p = s.find(':', p + pat.size());
        if (p == std::string::npos) return def;
        ++p;
        while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
        try { return std::stof(s.substr(p)); } catch (...) { return def; }
    }

    // Finds "key" then the next number after the following ':'.
    static int get_int(const std::string& s, const std::string& key, int def = -1) {
        const std::string pat = "\"" + key + "\"";
        size_t p = s.find(pat);
        if (p == std::string::npos) {
            if (def >= 0) return def;
            throw std::runtime_error("config.json: missing key " + key);
        }
        p = s.find(':', p + pat.size());
        if (p == std::string::npos) return def;
        ++p;
        while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
        if (p >= s.size() || (!std::isdigit(static_cast<unsigned char>(s[p]))
                              && s[p] != '-')) {
            if (def >= 0) return def;                 // e.g. null
            throw std::runtime_error("config.json: non-numeric " + key);
        }
        return std::stoi(s.substr(p));
    }
};
