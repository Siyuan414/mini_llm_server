/**
 * chunk_scheduler.h — how many query tokens each sequence gets this step (S3.3).
 *
 * NO CUDA. Pure policy, unit-testable on the host, like block_allocator.h.
 *
 * ── What this decides ────────────────────────────────────────────────────────
 * The varlen kernel (S3.1) made it POSSIBLE for sequence A to contribute 236
 * query tokens while B contributes 1. This class decides whether it SHOULD.
 *
 * Input:  the running sequences, each with a current length and a prompt length
 * Output: a per-sequence token count for this step, summing to <= the budget
 *
 * ── The rules, and why ───────────────────────────────────────────────────────
 * 1. DECODES FIRST. A generating sequence needs exactly 1 token and its user is
 *    watching text stream; a prefilling sequence needs many but its user sees
 *    nothing either way until the first token lands. Decode is latency-critical
 *    and cheap, prefill is elastic and expensive — so decodes are never starved
 *    and prefill soaks up whatever budget remains. This is why vLLM exposes
 *    max_num_batched_tokens rather than a prefill/decode ratio.
 *
 * 2. PREFILL IN FCFS ORDER, NOT SPLIT EVENLY. With 492 tokens spare and three
 *    sequences each needing 256: serving them in order finishes one this step
 *    (TTFT 1, 2, 2 → mean 1.67); splitting evenly finishes none until the next
 *    (2, 2, 2 → mean 2.0). Both are work-conserving, so the LAST sequence
 *    finishes at the same time either way — reordering cannot improve the
 *    makespan, only move completions earlier. Splitting evenly is strictly
 *    worse: it drags everyone to the latest possible finish and gains nothing.
 *    (Processor sharing maximizes mean completion time among work-conserving
 *    policies.)
 *
 * 3. CAP EACH SEQUENCE'S CHUNK. Pure FCFS lets a 4096-token prompt block a
 *    16-token one behind it for eight steps — head-of-line blocking, and it
 *    does wreck p99. The cap bounds that without paying rule 2's mean-latency
 *    tax. vLLM's long_prefill_token_threshold is exactly this.
 *
 * 4. RESPECT BLOCK AVAILABILITY. A chunk that cannot be backed by KV blocks is
 *    not a plan, it is a preemption waiting to happen. Shrink the chunk to what
 *    the allocator can actually supply.
 *
 * ── What this does NOT do ────────────────────────────────────────────────────
 * No admission (which waiting request becomes running), no preemption, no
 * allocation. It reads allocator state but never mutates it: the caller applies
 * the plan with append_tokens() and handles a false return by preempting.
 * Planning and committing stay separate so the plan can be tested without a
 * pool and rejected without rollback.
 */

#pragma once

#include "block_allocator.h"

#include <algorithm>
#include <cstdint>
#include <vector>

/// A running sequence, as the scheduler sees it. Deliberately not the engine's
/// Seq struct — the policy needs four numbers, not sockets and timestamps.
struct SeqView {
    uint64_t id         = 0;
    int      len        = 0;   // tokens already in the KV cache
    int      prompt_len = 0;   // total prompt length

    /// Still reading the prompt? Once len == prompt_len the sequence has
    /// consumed its prompt and every later step is a 1-token decode.
    bool prefilling() const { return len < prompt_len; }

    /// Prompt tokens not yet in the cache.
    int prefill_remaining() const {
        return prefilling() ? prompt_len - len : 0;
    }
};

/// Tokens granted to each sequence this step. A sequence absent from `entries`
/// gets nothing — it was preempted out of the budget, not finished.
struct ChunkPlan {
    struct Entry {
        uint64_t id       = 0;
        int      n_tokens = 0;   // query tokens this step (1 for a decode)
    };
    std::vector<Entry> entries;
    int total_tokens = 0;        // sum of n_tokens; <= cfg.max_batched_tokens

    /// True if any entry is a multi-token prefill chunk — i.e. this step needs
    /// the varlen kernel rather than the decode fast path.
    bool has_prefill() const {
        for (const auto& e : entries) if (e.n_tokens > 1) return true;
        return false;
    }
};

class ChunkScheduler {
public:
    struct Config {
        /// Total query tokens per forward pass. Bigger = better prefill
        /// throughput, longer worst-case step, so worse TPOT jitter.
        int max_batched_tokens = 512;

        /// Most tokens any ONE sequence may take in a step (rule 3).
        int max_chunk = 512;
    };

    void configure(const Config& c) { cfg_ = c; }
    const Config& config() const { return cfg_; }

    /// Plan one step. `running` is in admission (FCFS) order — the caller keeps
    /// it that way; this class does not reorder.
    ///
    /// `alloc` is consulted read-only for capacity (rule 4).
    ChunkPlan plan(const std::vector<SeqView>& running,
                   const BlockAllocator& alloc) const {
        ChunkPlan p;
        int blocks_committed = 0;   // blocks this plan has already promised

        // ── Rule 1: decodes first, one token each ────────────────────────────
        // A full pass BEFORE any prefill is considered. Walking the list once
        // and handling each sequence as encountered would let a 236-token
        // prompt at position 0 eat the budget the decoders behind it needed.
        //
        // The token is granted UNCONDITIONALLY — no block check. Decodes must
        // never be starved; if the pool genuinely cannot back the growth, the
        // caller preempts. But the block IS counted, so the prefill pass below
        // does not spend it twice (a decode crossing a block boundary costs a
        // block, and with 20 decoders mid-stride that is 20 blocks).
        for (const auto& seq : running) {
            if (!seq.prefilling()) {
                p.entries.push_back({seq.id, 1});
                p.total_tokens += 1;
                blocks_committed += alloc.blocks_to_append(seq.id, seq.len, 1);
            }
        }

        // ── Rules 2-4: prefill in FCFS order, capped, within block capacity ──
        for (const auto& seq : running) {
            if (!seq.prefilling()) continue;

            const int budget_left = cfg_.max_batched_tokens - p.total_tokens;
            if (budget_left <= 0) break;          // nothing left to hand out

            // Largest chunk the pool can still back, in TOKENS. Inverting
            // blocks_to_append is O(1): with `avail` blocks spare a sequence can
            // reach a total length of (have + avail) * block_size, so this many
            // more tokens fit. Shrinking to this — rather than skipping the
            // sequence when the full chunk does not fit — is what makes
            // degradation gradual instead of a cliff.
            const int avail    = alloc.num_free() - blocks_committed;
            const int have     = (int)alloc.table(seq.id).size();
            const int room     = (have + avail) * alloc.block_size() - seq.len;

            const int grant = std::min({seq.prefill_remaining(),
                                        cfg_.max_chunk,
                                        budget_left,
                                        room});
            if (grant <= 0) continue;             // no zero entries

            blocks_committed += alloc.blocks_to_append(seq.id, seq.len, grant);
            p.entries.push_back({seq.id, grant});
            p.total_tokens += grant;
        }

        return p;
    }

private:
    Config cfg_{};
};
