/**
 * tokenizer.h  —  Byte-level BPE tokenizer for LLaMA 3 (header-only)
 *
 * Reads tokenizer.json produced by Hugging Face `transformers`.
 * Implements:
 *   - bytes_to_unicode mapping (GPT-2 / LLaMA 3 byte encoding)
 *   - Simplified pre-tokenizer: splits on whitespace/punctuation boundaries
 *     (matches the ByteLevel pre-tokenizer for common ASCII text; a production
 *      build would use PCRE2 for the full GPT-4 Unicode regex)
 *   - Standard BPE merge algorithm
 *   - Byte-level decode back to UTF-8
 *
 * NOTE: This parser extracts only `model.vocab` and `model.merges` from the
 * tokenizer.json.  It is a hand-rolled scanner — not a general JSON parser.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <climits>
#include <cctype>
#include <cstdio>

class Tokenizer {
public:
    // ── Public API ────────────────────────────────────────────────────────────
    void load(const std::string& tokenizer_json_path) {
        build_byte_maps();
        parse_json(tokenizer_json_path);
        printf("Tokenizer: %zu tokens, %zu merges  (bos=%d eos=%d)\n",
               token_to_id_.size(), merges_.size(), bos_id_, eos_id_);
    }

    std::vector<int> encode(const std::string& text, bool add_bos = true) const {
        std::vector<int> ids;
        if (add_bos) ids.push_back(bos_id_);
        for (const std::string& word : pretokenize(text)) {
            // byte-encode the word, then BPE-merge
            std::vector<std::string> pieces;
            for (unsigned char c : word)
                pieces.push_back(b2u_[c]);
            bpe_merge(pieces);
            for (const std::string& p : pieces) {
                auto it = token_to_id_.find(p);
                if (it != token_to_id_.end())
                    ids.push_back(it->second);
                // unknown pieces: silently skip (rare for ASCII text)
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        // Concatenate token strings (in byte-unicode space), then map back to bytes
        std::string raw;
        for (int id : ids) {
            if (id < 0 || id >= (int)id_to_token_.size()) continue;
            const std::string& tok = id_to_token_[id];
            // skip control/special tokens like <|begin_of_text|>
            if (!tok.empty() && tok.front() == '<' && tok.back() == '>') continue;
            raw += tok;
        }
        return bytes_from_unicode(raw);
    }

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int eot_id() const { return eot_id_; }

private:
    // ── Byte ↔ Unicode mappings ───────────────────────────────────────────────
    //
    // GPT-2 / LLaMA 3 byte encoding maps each byte 0-255 to a Unicode codepoint:
    //   printable ASCII 33-126  → same codepoint
    //   Latin supplement 161-172, 174-255 → same codepoint
    //   everything else (incl. 0-32 = control + space) → 256+n  (Ā, ā, Ă, … Ġ…)
    //
    // space (0x20 = 32): 32 bytes below it are non-printable, none of which are
    // in the initial list, so n counts from 0 for byte 0, giving 0x20 → 256+32 = 288 = 'Ġ'

    std::array<std::string, 256> b2u_;                   // byte → utf8 string
    std::unordered_map<std::string, uint8_t> u2b_;       // utf8 string → byte

    void build_byte_maps() {
        // collect the "pass-through" bytes
        std::vector<int> bs, cs;
        for (int b = '!'; b <= '~'; ++b)    { bs.push_back(b); cs.push_back(b); }
        for (int b = 0xA1; b <= 0xAC; ++b)  { bs.push_back(b); cs.push_back(b); }
        for (int b = 0xAE; b <= 0xFF; ++b)  { bs.push_back(b); cs.push_back(b); }

        // remaining bytes → 256, 257, 258, …
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n++);
            }
        }

        // now build the maps (codepoint → UTF-8 string)
        for (int i = 0; i < 256; ++i) {
            int byte_val  = bs[i];
            int codepoint = cs[i];
            std::string utf8 = to_utf8(codepoint);
            b2u_[byte_val] = utf8;
            u2b_[utf8]     = (uint8_t)byte_val;
        }
    }

    // codepoint → UTF-8 (handles BMP range 0-65535, sufficient here)
    static std::string to_utf8(int cp) {
        if (cp < 0x80)
            return std::string(1, (char)cp);
        if (cp < 0x800)
            return {(char)(0xC0 | (cp >> 6)), (char)(0x80 | (cp & 0x3F))};
        return {(char)(0xE0 | (cp >> 12)),
                (char)(0x80 | ((cp >> 6) & 0x3F)),
                (char)(0x80 | (cp & 0x3F))};
    }

    // Convert a byte-unicode encoded string back to raw UTF-8 bytes
    std::string bytes_from_unicode(const std::string& s) const {
        std::string out;
        size_t i = 0;
        while (i < s.size()) {
            bool found = false;
            // Try longest match first (UTF-8 chars are 1-3 bytes in our range)
            for (int len : {3, 2, 1}) {
                if (i + len > s.size()) continue;
                std::string key = s.substr(i, len);
                auto it = u2b_.find(key);
                if (it != u2b_.end()) {
                    out += (char)it->second;
                    i += len;
                    found = true;
                    break;
                }
            }
            if (!found) out += s[i++];   // pass through unknown chars
        }
        return out;
    }

    // ── Pre-tokenizer ─────────────────────────────────────────────────────────
    //
    // Splits text into "words" following the ByteLevel pre-tokenization rules.
    // A space is glued to the next word as a Ġ prefix in the byte-unicode space.
    // This simplified version handles ASCII text; Unicode letters/numbers are
    // passed through as single words.
    //
    // ── Pre-tokenizer ────────────────────────────────────────────────────────
    // Follows the ordered alternation of LLaMA-3's pre-tokenizer regex:
    //
    //   1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    //   2. [^\r\n\p{L}\p{N}]?\p{L}+      one optional non-alnum prefix, then letters
    //   3. \p{N}{1,3}                    digits in groups of AT MOST THREE
    //   4.  ?[^\s\p{L}\p{N}]+[\r\n]*     optional space, then symbols
    //   5. \s*[\r\n]+                    newline runs
    //   6. \s+(?!\S) | \s+               whitespace
    //
    // Order matters: rule 2 before rule 4 is why "(n" is one chunk rather than
    // "(" + "n", and rule 3's 3-digit cap is why "1969" splits as "196"+"9"
    // (greedy, left-to-right) rather than "19"+"69".
    //
    // LIMITATION: \p{L} and \p{N} are approximated for ASCII — any byte > 127 is
    // treated as a letter. That is correct for accented Latin and punctuation
    // like em-dashes, but not for scripts where the distinction matters (e.g.
    // Devanagari digits). Verified against HF `tokenizers` by
    // tools/verify_tokenizer.py; full Unicode support needs PCRE2 or ICU.
    std::vector<std::string> pretokenize(const std::string& text) const {
        std::vector<std::string> words;
        const size_t n = text.size();
        size_t i = 0;

        auto is_letter = [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c > 127;
        };
        auto is_digit = [](unsigned char c) { return c >= '0' && c <= '9'; };
        auto is_space = [](unsigned char c) {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                   c == '\v' || c == '\f';
        };
        auto is_sym = [&](unsigned char c) {
            return !is_space(c) && !is_letter(c) && !is_digit(c);
        };

        while (i < n) {
            const unsigned char c = text[i];

            // 1. contractions
            if (c == '\'' && i + 1 < n) {
                static const char* kContractions[] = {
                    "'re","'ve","'ll","'s","'t","'m","'d", nullptr
                };
                bool matched = false;
                for (int k = 0; kContractions[k]; ++k) {
                    size_t len = strlen(kContractions[k]);
                    if (text.compare(i, len, kContractions[k]) == 0) {
                        words.push_back(kContractions[k]);
                        i += len;
                        matched = true;
                        break;
                    }
                }
                if (matched) continue;
            }

            // 2. [^\r\n\p{L}\p{N}]? \p{L}+
            {
                size_t start = i, j = i;
                if (!is_letter(c) && !is_digit(c) && c != '\n' && c != '\r' &&
                    j + 1 < n && is_letter((unsigned char)text[j + 1]))
                    ++j;                              // consume the prefix
                if (j < n && is_letter((unsigned char)text[j])) {
                    while (j < n && is_letter((unsigned char)text[j])) ++j;
                    words.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
            }

            // 3. \p{N}{1,3}
            if (is_digit(c)) {
                size_t j = i;
                while (j < n && is_digit((unsigned char)text[j]) && j - i < 3) ++j;
                words.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }

            // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
            {
                size_t start = i, j = i;
                if (c == ' ' && j + 1 < n && is_sym((unsigned char)text[j + 1]))
                    ++j;
                if (j < n && is_sym((unsigned char)text[j])) {
                    while (j < n && is_sym((unsigned char)text[j])) ++j;
                    while (j < n && (text[j] == '\r' || text[j] == '\n')) ++j;
                    words.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
            }

            // 5/6. whitespace runs.
            //
            // `\s+(?!\S)` keeps back the final space of a run when a word
            // follows, so rule 2 can pick it up as that word's prefix — this is
            // how " key" becomes a single chunk. Reaching here with a run of 1
            // means the next char is NOT a letter (rule 2 would have taken it),
            // so the whole run is emitted, e.g. the standalone " " before a
            // digit in "In 1969".
            {
                size_t j = i;
                while (j < n && is_space((unsigned char)text[j])) ++j;
                const bool next_is_letter =
                    j < n && is_letter((unsigned char)text[j]);
                const size_t end = (next_is_letter && j - i >= 2) ? j - 1 : j;
                words.push_back(text.substr(i, end - i));
                i = end;   // always > i, so the loop always advances
            }
        }
        return words;
    }

    // ── BPE merge ─────────────────────────────────────────────────────────────
    void bpe_merge(std::vector<std::string>& pieces) const {
        while (pieces.size() > 1) {
            // Find the pair with the lowest merge rank
            int best_rank = INT_MAX;
            int best_i    = -1;
            for (int i = 0; i + 1 < (int)pieces.size(); ++i) {
                auto it = merge_rank_.find({pieces[i], pieces[i + 1]});
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i    = i;
                }
            }
            if (best_i == -1) break;
            pieces[best_i] += pieces[best_i + 1];
            pieces.erase(pieces.begin() + best_i + 1);
        }
    }

    // ── JSON parser (hand-rolled, extracts vocab + merges only) ───────────────
    struct PairHash {
        size_t operator()(const std::pair<std::string,std::string>& p) const {
            size_t h = std::hash<std::string>{}(p.first);
            return h ^ (std::hash<std::string>{}(p.second) + 0x9e3779b9 + (h << 6));
        }
    };

    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string>             id_to_token_;
    std::vector<std::pair<std::string,std::string>> merges_;
    std::unordered_map<std::pair<std::string,std::string>,int,PairHash> merge_rank_;

    int bos_id_ = 128000, eos_id_ = 128001, eot_id_ = 128009;

    // Minimal JSON string unescaper
    static std::string unescape(const std::string& s) {
        std::string out;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '\\' && i + 1 < s.size()) {
                ++i;
                switch (s[i]) {
                    case '"':  out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/'; break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'u': {
                        // \uXXXX → encode as UTF-8
                        if (i + 4 < s.size()) {
                            int cp = std::stoi(s.substr(i+1, 4), nullptr, 16);
                            out += to_utf8(cp);
                            i += 4;
                        }
                        break;
                    }
                    default: out += s[i]; break;
                }
            } else {
                out += s[i];
            }
        }
        return out;
    }

    // Read the next JSON string token from pos, advancing pos past the closing "
    static std::string read_json_string(const std::string& src, size_t& pos) {
        // pos should be at the opening "
        assert(src[pos] == '"');
        ++pos;
        std::string s;
        while (pos < src.size() && src[pos] != '"') {
            if (src[pos] == '\\') { s += src[pos++]; }
            s += src[pos++];
        }
        ++pos;  // skip closing "
        return unescape(s);
    }

    void parse_json(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Cannot open " + path);
        std::string src((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

        // ── Find "vocab": { ... } ─────────────────────────────────────────
        {
            size_t p = src.find("\"vocab\"");
            if (p == std::string::npos) throw std::runtime_error("No vocab in tokenizer.json");
            p = src.find('{', p);
            ++p;  // skip '{'
            // Scan key: value pairs until '}'
            while (p < src.size()) {
                while (p < src.size() && src[p] != '"' && src[p] != '}') ++p;
                if (src[p] == '}') break;
                std::string key = read_json_string(src, p);
                // skip colon
                while (p < src.size() && src[p] != ':') ++p; ++p;
                // read integer value
                while (p < src.size() && src[p] == ' ') ++p;
                size_t num_start = p;
                while (p < src.size() && src[p] >= '0' && src[p] <= '9') ++p;
                int id = std::stoi(src.substr(num_start, p - num_start));
                token_to_id_[key] = id;
            }
        }

        // Build id_to_token (reverse)
        int max_id = 0;
        for (auto& [k,v] : token_to_id_) max_id = std::max(max_id, v);
        id_to_token_.resize(max_id + 1);
        for (auto& [k,v] : token_to_id_) id_to_token_[v] = k;

        // Resolve special token IDs
        auto lookup = [&](const std::string& name, int def) {
            auto it = token_to_id_.find(name);
            return (it != token_to_id_.end()) ? it->second : def;
        };
        bos_id_ = lookup("<|begin_of_text|>", 128000);
        eos_id_ = lookup("<|end_of_text|>",   128001);
        eot_id_ = lookup("<|eot_id|>",        128009);

        // ── Find "merges": [ ... ] ────────────────────────────────────────
        // Two formats in the wild:
        //   legacy  : "merges": ["a b", "c d", ...]        (space-joined)
        //   current : "merges": [["a","b"], ["c","d"], ...] (pair arrays)
        // tokenizers >= 0.20 emits the pair-array form.
        {
            size_t p = src.find("\"merges\"");
            if (p == std::string::npos) return;  // merges optional if no BPE
            p = src.find('[', p); ++p;

            size_t probe = p;
            while (probe < src.size() &&
                   std::isspace(static_cast<unsigned char>(src[probe]))) ++probe;
            const bool pair_arrays = (probe < src.size() && src[probe] == '[');

            int rank = 0;
            auto add = [&](const std::string& a, const std::string& b) {
                merges_.push_back({a, b});
                merge_rank_[{a, b}] = rank++;
            };

            if (pair_arrays) {
                while (p < src.size()) {
                    while (p < src.size() &&
                           (std::isspace(static_cast<unsigned char>(src[p])) ||
                            src[p] == ',')) ++p;
                    if (p >= src.size() || src[p] == ']') break;  // outer close
                    if (src[p] != '[') break;                     // malformed
                    ++p;                                          // past inner '['
                    while (p < src.size() && src[p] != '"') ++p;
                    std::string a = read_json_string(src, p);
                    while (p < src.size() && src[p] != '"') ++p;
                    std::string b = read_json_string(src, p);
                    while (p < src.size() && src[p] != ']') ++p;  // inner close
                    ++p;
                    add(a, b);
                }
            } else {
                while (p < src.size()) {
                    while (p < src.size() && src[p] != '"' && src[p] != ']') ++p;
                    if (p >= src.size() || src[p] == ']') break;
                    std::string merge_str = read_json_string(src, p);
                    size_t sp = merge_str.find(' ');
                    if (sp == std::string::npos) continue;
                    add(merge_str.substr(0, sp), merge_str.substr(sp + 1));
                }
            }
        }
    }
};
