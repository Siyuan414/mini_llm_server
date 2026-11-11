/**
 * block_allocator.h — paged KV block management (Stage 2B, S2.6).
 *
 * NO CUDA. Pure bookkeeping, unit-testable on the host.
 *
 * ── Why ──────────────────────────────────────────────────────────────────────
 * Stage 2A gave every slot a full max_seq window. Measured on a real workload:
 * requests averaged ~55 tokens but reserved 512, so ~9x of the KV cache was
 * dead space, and throughput efficiency collapsed to 32% at batch 16 because
 * attention scanned all 512 positions per slot regardless.
 *
 * Paging replaces the per-slot window with fixed-size blocks handed out on
 * demand:
 *
 *   physical:  [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ][ 7 ] ...   (16 tokens each)
 *   seq A (35 tok) -> [3, 0, 6]      48 token-slots, not 512
 *   seq B (20 tok) -> [1, 5]         32 token-slots
 *
 * ── Division of labour ───────────────────────────────────────────────────────
 * This class is the OS page table; the S2.8 kernel is the MMU. Allocation runs
 * ~once per 16 tokens over a few dozen ints (free lists, hash maps, branching —
 * all things CPUs are good at). Translation runs every step over every cached
 * token, on the GPU, via the flattened table this class produces.
 *
 * ── Fragmentation ────────────────────────────────────────────────────────────
 * Uniform block size means NO external fragmentation: any free block satisfies
 * any request, so N free blocks are always fully usable. The only waste is
 * internal — the partially-filled last block, averaging block_size/2 = 8 tokens
 * per sequence, versus 457 wasted per sequence in Stage 2A.
 *
 * ── Running out mid-generation ───────────────────────────────────────────────
 * A running sequence needs a new block every `block_size` tokens; it cannot be
 * failed halfway. Two defenses:
 *   1. WATERMARK — refuse to admit a new request unless a reserve remains for
 *      running sequences to grow into. Prevents most starvation.
 *   2. PREEMPTION — when it happens anyway, evict a victim (newest first: least
 *      work invested), free its blocks, requeue it for re-prefill. This is
 *      vLLM's "recompute" policy; the alternative is "swap".
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

class BlockAllocator {
public:
    struct Config {
        int block_size = 16;   // tokens per block
        int num_blocks = 0;    // total physical blocks (pool bytes / block bytes)
        int watermark  = 0;    // blocks held in reserve for running sequences
    };

    void configure(const Config& c) {
        if (c.block_size <= 0 || c.num_blocks <= 0)
            throw std::invalid_argument("BlockAllocator: bad config");
        cfg_ = c;
        tables_.clear();
        // Reverse order so pop_back() hands out 0, 1, 2, ... — makes tests and
        // memory dumps readable. Any order is functionally correct.
        free_.resize(cfg_.num_blocks);
        std::iota(free_.rbegin(), free_.rend(), 0);
    }

    // ── Pool state ───────────────────────────────────────────────────────────
    int num_free()   const { return (int)free_.size(); }
    int num_total()  const { return cfg_.num_blocks; }
    int block_size() const { return cfg_.block_size; }
    int num_seqs()   const { return (int)tables_.size(); }

    /// Blocks needed to hold n_tokens (ceiling division).
    int blocks_for(int n_tokens) const {
        if (n_tokens <= 0) return 0;
        return (n_tokens + cfg_.block_size - 1) / cfg_.block_size;
    }

    /// May a NEW request be admitted, given that its FIRST step will write
    /// `first_chunk_tokens`? Under lazy (per-chunk) allocation this is NOT the
    /// whole prompt — a 2000-token prompt admitted with a 512-token chunk needs
    /// 32 blocks now, not 125. The rest is allocated as the prompt is consumed,
    /// which is what lets long prompts start promptly instead of waiting for
    /// the entire reservation to come free.
    ///
    /// The watermark stays reserved so already-running sequences can still grow.
    /// Pair this with fits_ever(): can_admit answers "room right now", fits_ever
    /// answers "could this ever fit at all".
    bool can_admit(int first_chunk_tokens) const {
        return blocks_for(first_chunk_tokens) <= num_free() - cfg_.watermark;
    }

    // ── Per-sequence tables ──────────────────────────────────────────────────
    /// Allocate blocks for a prompt and start a table for seq_id.
    bool allocate(uint64_t seq_id, int n_tokens) {
        if (tables_.count(seq_id)) throw std::runtime_error("seq already allocated");
        const int need = blocks_for(n_tokens);
        if (need > num_free()) return false;

        std::vector<int> t;
        t.reserve(need);
        for (int i = 0; i < need; ++i) { t.push_back(free_.back()); free_.pop_back(); }
        tables_.emplace(seq_id, std::move(t));
        return true;
    }

    

    /// Return every block a sequence owns to the pool.
    void release(uint64_t seq_id) {
        auto it = tables_.find(seq_id);
        if (it == tables_.end()) return;
        for (int b : it->second) free_.push_back(b);
        tables_.erase(it);
    }

    bool has(uint64_t seq_id) const { return tables_.count(seq_id) != 0; }

    /// Physical block ids for a sequence, in logical order.
    const std::vector<int>& table(uint64_t seq_id) const {
        auto it = tables_.find(seq_id);
        if (it == tables_.end()) throw std::runtime_error("no such sequence");
        return it->second;
    }

    /// Where token `pos` of `seq_id` physically lives:
    /// (physical_block, offset_within_block). This is the translation the
    /// kernel performs per access.
    std::pair<int,int> locate(uint64_t seq_id, int pos) const {
        const auto& t = table(seq_id);
        const int logical = pos / cfg_.block_size;
        const int offset  = pos % cfg_.block_size;
        if (logical < 0 || logical >= (int)t.size())
            throw std::out_of_range("locate: position beyond allocation");
        return {t[logical], offset};
    }

    // ── Flattened block table for the GPU ────────────────────────────────────
    /// Row-major [n_seqs, max_blocks_per_seq], padded with -1. The kernel reads
    /// block_table[s * max_blocks_per_seq + logical]. Unknown sequences and
    /// unallocated logical blocks are -1 so a bug shows up as an obvious
    /// out-of-range id rather than silently reading block 0.
    std::vector<int> flatten(const std::vector<uint64_t>& seq_ids,
                             int max_blocks_per_seq) const {
        std::vector<int> flat((size_t)seq_ids.size() * max_blocks_per_seq, -1);
        for (size_t s = 0; s < seq_ids.size(); ++s) {
            auto it = tables_.find(seq_ids[s]);
            if (it == tables_.end()) continue;               // hole in the batch
            const auto& t = it->second;
            const int n = std::min((int)t.size(), max_blocks_per_seq);
            for (int b = 0; b < n; ++b)
                flat[s * max_blocks_per_seq + b] = t[b];
        }
        return flat;
    }

    /// Fraction of allocated token-slots holding real tokens.
    /// Stage 2A's equivalent was ~55/512 = 11%.
    double utilization(const std::vector<int>& lengths) const {
        long long real = 0;
        for (int l : lengths) real += l;
        long long slots = 0;
        for (const auto& kv : tables_)
            slots += (long long)kv.second.size() * cfg_.block_size;
        return slots ? (double)real / (double)slots : 0.0;
    }

    // ── Growing a sequence ───────────────────────────────────────────────────
    /// Blocks a sequence of length `cur_len` must gain to hold `n` more tokens.
    ///
    /// NOT ceil(n / block_size): the partially-filled last block absorbs tokens
    /// for free. At cur_len 20 with 16-token blocks you hold 2 blocks = 32 slots
    /// with 12 spare, so n = 4 costs ZERO new blocks while ceil(4/16) says 1.
    /// Differencing blocks_for() is what accounts for that.
    ///
    /// Pure arithmetic — usable by the scheduler to size a chunk before the
    /// sequence exists.
    int blocks_to_append(int cur_len, int n) const {
        return blocks_for(cur_len + n) - blocks_for(cur_len);
    }

    /// As above, but measured against what the sequence ACTUALLY holds rather
    /// than what cur_len implies. Those differ whenever blocks were reserved
    /// ahead of use — e.g. a prompt allocated eagerly at admission, then
    /// consumed token by token. Sizing from the table is correct in both
    /// regimes; sizing from cur_len alone would re-allocate blocks the sequence
    /// already owns.
    int blocks_to_append(uint64_t seq_id, int cur_len, int n) const {
        const int have = (int)table(seq_id).size();
        const int want = blocks_for(cur_len + n);
        return want > have ? want - have : 0;
    }

    bool can_append(uint64_t seq_id, int cur_len, int n) const {
        return blocks_to_append(seq_id, cur_len, n) <= num_free();
    }

    /// Grow a sequence by `n` tokens, where `cur_len` is its length BEFORE the
    /// append. ALL-OR-NOTHING: if the pool cannot supply every block needed,
    /// nothing is allocated and false is returned (the caller must preempt).
    /// A partial append would leave tokens with nowhere to live, and the kernel
    /// reads -1 from the block table — which Phase 1 turns into -FLT_MAX,
    /// silently corrupting that sequence instead of failing loudly.
    bool append_tokens(uint64_t seq_id, int cur_len, int n) {
        auto it = tables_.find(seq_id);
        if (it == tables_.end()) throw std::runtime_error("no such sequence");

        const int have = (int)it->second.size();
        // The table must already cover cur_len. If it doesn't, the caller's
        // notion of this sequence's length has drifted from the allocator's —
        // cheapest possible place to catch a scheduler bug, versus discovering
        // it as garbage logits several layers deep.
        if (have < blocks_for(cur_len))
            throw std::runtime_error("append_tokens: table does not cover cur_len");

        const int want = blocks_for(cur_len + n);
        const int need = want > have ? want - have : 0;
        if (need > num_free()) return false;

        for (int i = 0; i < need; ++i) {
            it->second.push_back(free_.back());
            free_.pop_back();
        }
        return true;
    }

    /// Decode's one-token step is just the n == 1 case.
    bool append_token(uint64_t seq_id, int cur_len) {
        return append_tokens(seq_id, cur_len, 1);
    }

    /// Could a prompt of this size EVER be served, with an empty pool?
    ///
    /// Lazy allocation loses the natural guard eager allocation had: an
    /// oversized prompt is admitted, prefills until blocks run out, is
    /// preempted, requeued, admitted again — forever, burning the GPU on work
    /// it throws away. A prompt failing this check is rejected outright (a
    /// client error), never preempted.
    bool fits_ever(int n_tokens) const {
        return blocks_for(n_tokens) <= cfg_.num_blocks - cfg_.watermark;
    }

private:
    Config cfg_{};
    std::vector<int> free_;                                  // stack of block ids
    std::unordered_map<uint64_t, std::vector<int>> tables_;  // seq -> blocks
};
