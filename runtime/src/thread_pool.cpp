#include "anytime/thread_pool.hpp"

#include <algorithm>

namespace anytime {

ThreadPool::ThreadPool(int threads) : threads_(std::max(1, threads)) {
    // A pool of one owns no workers at all, so `parallel_for` becomes a plain loop
    // with no lock taken and no condition variable touched. That is the path every
    // number recorded before this change was measured on, and it has to stay free.
    workers_.reserve(static_cast<std::size_t>(threads_ - 1));
    for (int i = 1; i < threads_; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    work_ready_.notify_all();
    for (std::thread& worker : workers_) {
        worker.join();
    }
}

void ThreadPool::parallel_for(std::size_t count,
                              const std::function<void(std::size_t)>& body) {
    if (count == 0) {
        return;
    }
    if (threads_ <= 1 || count == 1) {
        for (std::size_t i = 0; i < count; ++i) {
            body(i);
        }
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        body_ = &body;
        count_ = count;
        next_ = 0;
        // Every index is outstanding until someone finishes it. Counting claims
        // rather than indices would let the caller return while a worker was still
        // inside `body`, which is the whole thing this has to prevent.
        outstanding_ = count;
        error_ = nullptr;
        error_index_ = 0;
        ++generation_;
    }
    work_ready_.notify_all();

    // The caller is a runner too, so a pool of N splits the work N ways rather than
    // N-1 and does not sit idle waiting for its own workers.
    for (;;) {
        std::size_t index = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (next_ >= count_) {
                break;
            }
            index = next_++;
        }
        std::exception_ptr failure;
        try {
            body(index);
        } catch (...) {
            failure = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (failure && (!error_ || index < error_index_)) {
                error_ = failure;
                error_index_ = index;
            }
            if (--outstanding_ == 0) {
                work_done_.notify_all();
            }
        }
    }

    std::unique_lock<std::mutex> lock(mutex_);
    work_done_.wait(lock, [this] { return outstanding_ == 0; });
    body_ = nullptr;
    if (error_) {
        std::exception_ptr error = error_;
        error_ = nullptr;
        // Rethrown on the thread that asked for the work rather than swallowed, so a
        // future caller that can fail inside a task gets an exception instead of a
        // terminated interpreter.
        std::rethrow_exception(error);
    }
}

void ThreadPool::worker_loop() {
    std::size_t seen = 0;
    for (;;) {
        const std::function<void(std::size_t)>* body = nullptr;
        std::size_t index = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_ready_.wait(lock, [this, &seen] {
                return stopping_ || (generation_ != seen && next_ < count_);
            });
            if (stopping_) {
                return;
            }
            if (next_ >= count_) {
                // This batch is exhausted; wait for the next announcement rather than
                // spinning on a generation that has nothing left in it.
                seen = generation_;
                continue;
            }
            body = body_;
            index = next_++;
        }

        std::exception_ptr failure;
        try {
            (*body)(index);
        } catch (...) {
            failure = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (failure && (!error_ || index < error_index_)) {
                error_ = failure;
                error_index_ = index;
            }
            if (--outstanding_ == 0) {
                work_done_.notify_all();
            }
        }
    }
}

}  // namespace anytime
