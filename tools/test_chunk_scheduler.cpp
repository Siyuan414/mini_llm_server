/**
 * test_chunk_scheduler.cpp — S3.3 policy tests, no CUDA, no GPU.
 *
 * The scheduler's job is to answer "how many query tokens does each sequence
 * get this step". These tests pin down the four rules, and then the invariant
 * that matters most in production: a plan must be AFFORDABLE — applying it with
 * append_tokens() must never fail, or the engine preempts someone for a promise
 * the scheduler should not have made.
 *
 * Build:
 *   g++ -std=c++17 -I src tools/test_chunk_scheduler.cpp -o build/test_chunk
 * Run:
 *   ./build/test_chunk
 */

#include "chunk_scheduler.h"

#include <cstdio>
#include <map>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg) do {                                               \
    if (!(cond)) { printf("  FAIL  %s\n", msg); ++failures; }               \
    else         { printf("  ok    %s\n", msg); }                           \
} while (0)

#define CHECK_EQ(a, b, msg) do {                                            \
    const auto _a = (a); const auto _b = (b);                               \
    if (_a != _b) {                                                         \
        printf("  FAIL  %s  (got %lld, want %lld)\n", msg,                  \
               (long long)_a, (long long)_b);                               \
        ++failures;                                                         \
    } else { printf("  ok    %s\n", msg); }                                 \
} while (0)

namespace {

constexpr int kBlockSize = 16;

/// Grant for a sequence, or 0 if the plan gave it nothing.
int granted(const ChunkPlan& p, uint64_t id) {
    for (const auto& e : p.entries) if (e.id == id) return e.n_tokens;
    return 0;
}

/// Builds an allocator whose state matches `seqs`: every sequence owns exactly
/// the blocks its current len requires.
BlockAllocator make_alloc(int num_blocks, const std::vector<SeqView>& seqs) {
    BlockAllocator a;
    a.configure({kBlockSize, num_blocks, 0});
    for (const auto& s : seqs) a.allocate(s.id, s.len);
    return a;
}

/// THE invariant: every grant in the plan must be backed by real blocks.
/// Applies the plan for real and reports whether anything failed.
bool plan_is_affordable(const ChunkPlan& p, BlockAllocator& a,
                        const std::vector<SeqView>& seqs) {
    std::map<uint64_t, int> len;
    for (const auto& s : seqs) len[s.id] = s.len;
    for (const auto& e : p.entries)
        if (!a.append_tokens(e.id, len[e.id], e.n_tokens)) return false;
    return true;
}

}  // namespace

int main() {
    // ── Rule 1: decodes are never starved by a long prompt ──────────────────
    {
        printf("rule 1: decodes first\n");
        std::vector<SeqView> seqs;
        for (uint64_t i = 1; i <= 20; ++i)
            seqs.push_back({i, /*len*/16, /*prompt_len*/16});   // decoding
        seqs.push_back({99, /*len*/0, /*prompt_len*/1000});     // long prefill

        auto a = make_alloc(200, seqs);
        ChunkScheduler s;
        s.configure({/*max_batched_tokens*/512, /*max_chunk*/512});
        const auto p = s.plan(seqs, a);

        int decodes = 0;
        for (uint64_t i = 1; i <= 20; ++i) decodes += granted(p, i);
        CHECK_EQ(decodes, 20, "all 20 decoders got exactly 1 token");
        CHECK_EQ(p.total_tokens, 512, "budget fully used");
        CHECK_EQ(granted(p, 99), 492, "prefill got only the remainder");
        CHECK(p.has_prefill(), "step needs the varlen kernel");
    }

    // ── Rule 2: FCFS, not split evenly ──────────────────────────────────────
    {
        printf("rule 2: FCFS beats processor sharing\n");
        std::vector<SeqView> seqs{{1, 0, 256}, {2, 0, 256}, {3, 0, 256}};
        auto a = make_alloc(200, seqs);
        ChunkScheduler s;
        s.configure({512, 512});
        const auto p = s.plan(seqs, a);

        // Splitting evenly would give 170/170/170 and finish nobody this step.
        CHECK_EQ(granted(p, 1), 256, "seq 1 fully prefilled this step");
        CHECK_EQ(granted(p, 2), 256, "seq 2 fully prefilled this step");
        CHECK_EQ(granted(p, 3), 0,   "seq 3 waits (no partial split)");
        CHECK_EQ(p.total_tokens, 512, "budget exactly consumed");
    }

    // ── Rule 3: per-sequence cap bounds head-of-line blocking ───────────────
    {
        printf("rule 3: max_chunk cap\n");
        std::vector<SeqView> seqs{{1, 0, 4096}, {2, 0, 16}};
        auto a = make_alloc(400, seqs);
        ChunkScheduler s;
        s.configure({/*budget*/512, /*max_chunk*/128});
        const auto p = s.plan(seqs, a);

        CHECK_EQ(granted(p, 1), 128, "huge prompt capped at 128");
        CHECK_EQ(granted(p, 2), 16,  "small prompt not blocked behind it");
    }

    // ── Rule 4a: shrink to available blocks, do not skip ─────────────────────
    {
        printf("rule 4a: degrade gradually under memory pressure\n");
        // 20 decoders at len 10 hold 1 block each and do NOT cross a boundary
        // (blocks_for(11) == blocks_for(10)), so they cost 0 new blocks.
        // 24 - 20 = 4 blocks free = 64 tokens of prefill room.
        std::vector<SeqView> seqs;
        for (uint64_t i = 1; i <= 20; ++i) seqs.push_back({i, 10, 10});
        seqs.push_back({99, 0, 1000});

        auto a = make_alloc(24, seqs);
        ChunkScheduler s;
        s.configure({512, 512});
        const auto p = s.plan(seqs, a);

        CHECK_EQ(granted(p, 99), 64,
                 "prefill shrunk to what blocks allow, not skipped");

        int decodes = 0;
        for (uint64_t i = 1; i <= 20; ++i) decodes += granted(p, i);
        CHECK_EQ(decodes, 20, "decodes unaffected by memory pressure");
        CHECK(plan_is_affordable(p, a, seqs), "and the plan is affordable");
    }

    // ── Rule 4b: decode-driven block growth is counted ───────────────────────
    {
        printf("rule 4b: prefill does not double-spend the decoders' blocks\n");
        // Now len 16: every decoder DOES cross a boundary, so the 20 decodes
        // cost 20 blocks. Pool 44 - 20 held = 24 free, minus 20 for the decodes
        // leaves 4 = 64 tokens. A scheduler that ignored decode growth would
        // see 24 blocks free and hand prefill 384 tokens it cannot back.
        std::vector<SeqView> seqs;
        for (uint64_t i = 1; i <= 20; ++i) seqs.push_back({i, 16, 16});
        seqs.push_back({99, 0, 1000});

        auto a = make_alloc(44, seqs);
        ChunkScheduler s;
        s.configure({512, 512});
        const auto p = s.plan(seqs, a);

        CHECK_EQ(granted(p, 99), 64, "prefill got 64, not 384");
        CHECK(plan_is_affordable(p, a, seqs),
              "decodes and prefill both fit exactly");
        CHECK_EQ(a.num_free(), 0, "pool drained to exactly zero");
    }

    // ── Rule 4c: decodes alone can over-subscribe; prefill yields ────────────
    {
        printf("rule 4c: when decodes alone exhaust the pool, prefill gets 0\n");
        // 20 decoders each needing a new block, but only 4 free. The decodes are
        // granted anyway (never starve a mid-stream decode) and prefill is
        // squeezed to nothing. The engine preempts to resolve this — the
        // scheduler's job is only to stop making it worse.
        std::vector<SeqView> seqs;
        for (uint64_t i = 1; i <= 20; ++i) seqs.push_back({i, 16, 16});
        seqs.push_back({99, 0, 1000});

        auto a = make_alloc(24, seqs);
        ChunkScheduler s;
        s.configure({512, 512});
        const auto p = s.plan(seqs, a);

        CHECK_EQ(granted(p, 99), 0, "no prefill when decodes need everything");
        int decodes = 0;
        for (uint64_t i = 1; i <= 20; ++i) decodes += granted(p, i);
        CHECK_EQ(decodes, 20, "decodes still granted (caller must preempt)");
    }

    // ── The affordability invariant ──────────────────────────────────────────
    {
        printf("plans are affordable (no promise the pool cannot keep)\n");

        // Two prefills that would each individually fit, but not together:
        // 16 blocks apiece against a 20-block pool. A scheduler that checks
        // each sequence against the same num_free() promises 32 blocks.
        std::vector<SeqView> seqs{{1, 0, 256}, {2, 0, 256}};
        auto a = make_alloc(20, seqs);
        ChunkScheduler s;
        s.configure({512, 512});
        const auto p = s.plan(seqs, a);

        CHECK(plan_is_affordable(p, a, seqs),
              "every grant backed by real blocks");
        CHECK(a.num_free() >= 0, "pool never over-drawn");
        printf("        seq1=%d seq2=%d, pool had 20 blocks\n",
               granted(p, 1), granted(p, 2));
    }

    // ── Degenerate cases ─────────────────────────────────────────────────────
    {
        printf("degenerate cases\n");
        ChunkScheduler s;
        s.configure({512, 512});

        BlockAllocator empty;
        empty.configure({kBlockSize, 100, 0});
        const auto p0 = s.plan({}, empty);
        CHECK_EQ(p0.total_tokens, 0, "no sequences -> empty plan");
        CHECK(!p0.has_prefill(), "empty plan needs no varlen kernel");

        // All decoding: a pure decode step must not claim to need varlen.
        std::vector<SeqView> dec{{1, 5, 5}, {2, 7, 7}};
        auto a = make_alloc(100, dec);
        const auto p1 = s.plan(dec, a);
        CHECK_EQ(p1.total_tokens, 2, "two decoders, two tokens");
        CHECK(!p1.has_prefill(), "pure decode step: fast path still usable");

        // Budget smaller than the number of decoders: they are granted anyway,
        // because starving a mid-stream decode is worse than a long step.
        std::vector<SeqView> many;
        for (uint64_t i = 1; i <= 10; ++i) many.push_back({i, 5, 5});
        auto a2 = make_alloc(100, many);
        ChunkScheduler tight;
        tight.configure({/*budget*/4, /*max_chunk*/4});
        const auto p2 = tight.plan(many, a2);
        CHECK_EQ(p2.total_tokens, 10, "decodes exceed budget but are not dropped");
    }

    printf("\n%s\n", failures ? "FAILED" : "all tests passed");
    return failures ? 1 : 0;
}
