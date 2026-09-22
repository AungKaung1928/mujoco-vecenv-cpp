#include "vecenv/vecenv.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define VECENV_PAUSE() _mm_pause()
#else
#define VECENV_PAUSE() ((void)0)
#endif

namespace vecenv {

namespace {
constexpr int kSpinIterations = 20000;   // tens of microseconds before blocking
}

VecEnv::VecEnv(const EnvConfig& cfg, int n, std::uint64_t seed, int threads, bool allow_overcommit)
    : n_(n), threads_(threads), cfg_(cfg) {
    if (n <= 0) throw std::invalid_argument("n must be positive");
    if (threads < 0) throw std::invalid_argument("threads must be >= 0");
    if (threads_ == 0) threads_ = std::min(n, kMaxWorkers);
    threads_ = std::min(threads_, n);
    if (threads_ > kMaxWorkers && !allow_overcommit) {
        throw std::invalid_argument(std::to_string(threads_) + " threads requested; the budget on the machine this "
                                    "was written on is " + std::to_string(kMaxWorkers) +
                                    " of 14 cores. Pass allow_overcommit only if you own the box.");
    }
    check_library_version();
    const std::string assets = cfg_.assets_dir.empty() ? default_assets_dir() : cfg_.assets_dir;
    prototype_ = load_variant(assets, cfg_.variant);
    envs_.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) envs_.push_back(std::make_unique<Env>(cfg_, prototype_.get(), seed + static_cast<std::uint64_t>(i)));

    actions_.assign(static_cast<std::size_t>(n) * kNAct, 0.0);
    obs_.assign(static_cast<std::size_t>(n) * kObsDim, 0.0f);
    terminal_obs_.assign(static_cast<std::size_t>(n) * kObsDim, 0.0f);
    rew_.assign(static_cast<std::size_t>(n), 0.0);
    done_.assign(static_cast<std::size_t>(n), 0);
    info_.assign(static_cast<std::size_t>(n), StepInfo{});

    // The calling thread is worker 0; threads 1..T-1 live in the pool.
    for (int tid = 1; tid < threads_; ++tid) pool_.emplace_back(&VecEnv::worker_, this, tid);
}

VecEnv::~VecEnv() { close(); }

void VecEnv::close() {
    if (closed_) return;
    closed_ = true;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
    for (auto& t : pool_) if (t.joinable()) t.join();
    pool_.clear();
}

void VecEnv::reset() {
    for (int i = 0; i < n_; ++i) {
        envs_[static_cast<std::size_t>(i)]->reset();
        envs_[static_cast<std::size_t>(i)]->observe(obs_.data() + static_cast<std::size_t>(i) * kObsDim);
        done_[static_cast<std::size_t>(i)] = 0;
        rew_[static_cast<std::size_t>(i)] = 0.0;
    }
}

void VecEnv::run_env_(int i) {
    const auto u = static_cast<std::size_t>(i);
    Env& e = *envs_[u];
    float* o = obs_.data() + u * kObsDim;
    StepInfo& inf = info_[u];
    e.step(actions_.data() + u * kNAct, o, &inf);
    rew_[u] = inf.reward;
    done_[u] = e.done() ? 1 : 0;
    if (done_[u]) {
        // Autoreset. The policy needs the new episode's observation; the value
        // function needs the one the episode ended on. Both are kept.
        std::copy(o, o + kObsDim, terminal_obs_.data() + u * kObsDim);
        e.reset();
        e.observe(o);
    }
}

void VecEnv::wait_generation_(std::uint64_t expected, int) {
    for (int k = 0; k < kSpinIterations; ++k) {
        if (generation_.load(std::memory_order_acquire) >= expected || stop_.load(std::memory_order_acquire)) return;
        VECENV_PAUSE();
    }
    std::unique_lock<std::mutex> lk(mtx_);
    cv_.wait(lk, [&] {
        return generation_.load(std::memory_order_acquire) >= expected || stop_.load(std::memory_order_acquire);
    });
}

void VecEnv::worker_(int tid) {
    std::uint64_t seen = 0;
    while (true) {
        wait_generation_(seen + 1, tid);
        if (stop_.load(std::memory_order_acquire)) return;
        seen = generation_.load(std::memory_order_acquire);
        for (int i = tid; i < n_; i += threads_) run_env_(i);
        remaining_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void VecEnv::step(const double* actions) {
    if (closed_) throw std::runtime_error("VecEnv is closed");
    std::copy(actions, actions + static_cast<std::size_t>(n_) * kNAct, actions_.begin());
    if (threads_ <= 1) {
        for (int i = 0; i < n_; ++i) run_env_(i);
        return;
    }
    remaining_.store(threads_ - 1, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(mtx_);
        generation_.fetch_add(1, std::memory_order_acq_rel);
    }
    cv_.notify_all();
    for (int i = 0; i < n_; i += threads_) run_env_(i);
    // The caller has nothing else to do, so it spins; the workers are the ones
    // that must not spin between steps.
    while (remaining_.load(std::memory_order_acquire) != 0) VECENV_PAUSE();
}

}  // namespace vecenv
