#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace anytime {

/// A fixed set of worker threads for splitting one bulk copy across cores.
///
/// Deliberately the smallest thing that does the job. There is exactly one caller --
/// the KV gather in `DecoderSession::decode_batch` -- and it wants a blocking
/// `parallel_for` over a few dozen independent slots, not a general task system. No
/// work stealing: every slot of a decode step copies the same number of bytes, so a
/// static split is already balanced and a deque per worker would only add contention.
///
/// Persistent rather than spawned per call, because the work is sometimes small. At
/// batch 1 the whole gather is 0.18 ms and starting eight threads costs more than that,
/// so a pool that spawned per step would be a slowdown at exactly the sizes where the
/// gather is already cheap.
///
/// Not a general-purpose pool in one important way: `parallel_for` must not be called
/// from a worker. Nothing does, and a nested call would deadlock rather than
/// misbehave quietly, so it is documented rather than detected.
class ThreadPool {
  public:
    /// `threads` is the total number of runners including the calling thread, so a
    /// pool of N owns N-1 workers. One or fewer owns none and runs everything inline.
    explicit ThreadPool(int threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int threads() const { return threads_; }

    /// Run `body(i)` for every i in [0, count), and return once all of them have.
    ///
    /// The calling thread takes a share rather than blocking, so a pool of one costs
    /// nothing but a loop.
    ///
    /// An exception from any index is captured and rethrown here, because letting one
    /// escape a worker calls std::terminate. Today's only caller cannot reach it --
    /// `decode_batch` totals a batch's shortfall and refuses before it reserves
    /// anything, so the gather never runs against a span that cannot hold it -- so
    /// this is a guard against a future caller rather than a path under test. If
    /// several throw, the first by index wins and the rest are dropped; they are
    /// symptoms of one call.
    void parallel_for(std::size_t count, const std::function<void(std::size_t)>& body);

  private:
    void worker_loop();

    int threads_;
    std::vector<std::thread> workers_;

    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable work_done_;

    const std::function<void(std::size_t)>* body_ = nullptr;
    std::size_t count_ = 0;
    std::size_t next_ = 0;
    std::size_t outstanding_ = 0;
    // Bumped once per parallel_for so a worker waking spuriously can tell whether the
    // batch it already finished is the one being announced.
    std::size_t generation_ = 0;
    std::exception_ptr error_;
    std::size_t error_index_ = 0;
    bool stopping_ = false;
};

}  // namespace anytime
