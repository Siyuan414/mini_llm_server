/**
 * scheduler.h — request lifecycle + admission policy (Stage 2A).
 *
 * DELIBERATELY CONTAINS NO CUDA. This is pure policy: queues, state machines,
 * and the decision of when to admit. The runtime owns the GPU work and calls
 * into here. That split makes the interesting logic unit-testable without a
 * device, and mirrors how vLLM separates scheduler.py from the model runner.
 *
 * ── Lifecycle ────────────────────────────────────────────────────────────────
 *   submit()          → Waiting  (in queue, no slot)
 *   next_admission()  → Running  (slot assigned, prefilled by the runtime)
 *   on_token()        → Running  (accumulating output)
 *                     → Done     (EOS, or max_new_tokens reached)
 *
 * ── The admission tradeoff ───────────────────────────────────────────────────
 * Admitting requires a separate batch-1 prefill (profile 0), which stalls the
 * decode batch — ~12 ms for a 512-token prompt vs ~5 ms per decode step. So
 * every admission costs running requests ~2.5 steps of latency.
 *
 *   eager  : admit whenever a slot is free  → best TTFT, stuttering TPOT
 *   lazy   : admit only when the batch drains → smooth TPOT, queueing delay
 *   capped : at most `max_admits_per_step`    → bounded stall (default 1)
 *
 * S2.5 measures this rather than assuming it.
 */

#pragma once

#include "sampling.cuh"   // SamplingParams (host-side struct only)

#include <chrono>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

enum class ReqState { Waiting, Running, Done };

struct Request {
    uint64_t id = 0;
    std::vector<int> prompt;      // token ids, set at submit
    std::vector<int> output;      // generated token ids
    SamplingParams   sp;
    int max_new_tokens = 128;

    ReqState state = ReqState::Waiting;
    int      slot  = -1;          // BatchKVCache slot, -1 while waiting

    // Metrics (S2.5 reports TTFT / TPOT from these)
    std::chrono::steady_clock::time_point t_submit, t_first_token, t_done;

    int  generated()  const { return (int)output.size(); }
    bool at_limit()   const { return generated() >= max_new_tokens; }
};

class Scheduler {
public:
    struct Config {
        int max_admits_per_step = 1;   // bound the prefill stall
        int max_batch           = 4;   // must match BatchKVCache
    };

    void configure(const Config& c) { cfg_ = c; }

    // ── Submission ───────────────────────────────────────────────────────────
    /// Queue a new request. Returns its id.
    uint64_t submit(Request r) {
        // TODO
        //  - assign r.id = next_id_++
        //  - r.state = Waiting, r.t_submit = now
        //  - push onto waiting_
        //  - return the id
        const uint64_t id = r.id = next_id_++;
        r.state = ReqState::Waiting;
        r.t_submit = std::chrono::steady_clock::now();
        waiting_.push_back(std::move(r));
        return id;
    }

    // ── Admission (called by the runtime before each decode step) ───────────
    /// Pop the next request to prefill, or nullptr if none should be admitted
    /// this step. The runtime prefills it, calls BatchKVCache::acquire() +
    /// install_prefill(), then hands the slot back via mark_running().
    ///
    /// Returns nullptr when: nothing waiting, no free slot, or this step's
    /// admission budget is already spent.
    Request* next_admission(bool cache_has_free) {
        // TODO
        //  - if (!cache_has_free || waiting_.empty()
        //        || admits_this_step_ >= cfg_.max_admits_per_step) return nullptr
        //  - move front of waiting_ into running_, return pointer to it
        //  - ++admits_this_step_
        if (!cache_has_free || waiting_.empty() || admits_this_step_ >= cfg_.max_admits_per_step) {
            return nullptr;
        }
        running_.push_back(std::move(waiting_.front()));
        waiting_.pop_front();
        ++admits_this_step_;    
        return &running_.back();
    }

    /// Called once per decode step, before admissions, to reset the budget.
    void begin_step() { admits_this_step_ = 0; }

    /// The runtime reports which slot the admitted request landed in.
    void mark_running(Request* r, int slot) {
        // TODO: r->slot = slot; r->state = Running;
        if(r){
            r->slot = slot;
            r->state = ReqState::Running;

        }

    }

    // ── Token accounting ─────────────────────────────────────────────────────
    /// Record a generated token for the request in `slot`.
    /// Returns true if the request just finished (EOS or hit its token limit),
    /// in which case the runtime must release the slot.
    bool on_token(int slot, int token, int eos_id, int eot_id) {
        // TODO
        //  - find the running request with this slot
        //  - if output empty, stamp t_first_token (this is TTFT)
        //  - append token
        //  - if token == eos_id || token == eot_id || at_limit():
        //        state = Done, t_done = now, return true
        //  - return false
        Request* r = find_slot(slot);
        if (r) {
            if (r->output.empty()) {
                r->t_first_token = std::chrono::steady_clock::now();
            }
            r->output.push_back(token);
            if (token == eos_id || token == eot_id || r->at_limit()) {
                r->state = ReqState::Done;
                r->t_done = std::chrono::steady_clock::now();
                return true;
            }           
        }
        return false;
    }

    /// Move a finished request out of `running_` into `done_`. The runtime calls
    /// this after BatchKVCache::release(slot).
    void retire(int slot) {
        // TODO
        //  - find the running request with this slot
        //  - move it into done_ (erase from running_)
        //  - reset its slot to -1 (not strictly necessary)
       for(auto it = running_.begin(); it != running_.end(); ++it) {
            if (it->slot == slot) {
                it->slot = -1; // reset slot
                done_.push_back(std::move(*it));
                running_.erase(it);
                break;
            }
        }

    }

    // ── Queries ──────────────────────────────────────────────────────────────
    bool has_work()      const { return !waiting_.empty() || !running_.empty(); }
    int  n_waiting()     const { return (int)waiting_.size(); }
    int  n_running()     const { return (int)running_.size(); }
    const std::vector<Request>& done() const { return done_; }

    /// Token to feed each running slot next step: the last generated token.
    /// Slots with no running request get a filler (their output is discarded).
    std::vector<int> next_tokens(int batch_size, int filler) const {
        // TODO: [batch_size], indexed BY SLOT (not by request order)
        std::vector<int> tokens(batch_size, filler);
        for (const auto& req : running_) {
            if (req.slot >= 0 && req.slot < batch_size) {
                if (!req.output.empty()) {
                    tokens[req.slot] = req.output.back();
                }
            }
        }
        return tokens;
    }

private:
    // TODO: helper to find a running request by slot
    Request* find_slot(int slot) { 
        for (auto& req : running_) {
            if (req.slot == slot) {
                return &req;
            }
        }
        return nullptr;
    }

    Config cfg_{};
    uint64_t next_id_ = 1;
    int admits_this_step_ = 0;

    std::deque<Request>  waiting_;   // FIFO; a priority queue would go here
    std::vector<Request> running_;   // active requests; find by .slot
    std::vector<Request> done_;
};
