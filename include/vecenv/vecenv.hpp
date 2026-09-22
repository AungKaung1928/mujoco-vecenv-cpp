// N Microducks on T threads, one mjModel + mjData per environment.
//
// microduck-rl runs its N environments in N forked processes because the
// Python interpreter cannot step two environments at once. Here the step
// loop has no interpreter in it, so the same N environments run on T
// threads of one process: no pickling, no pipes, one shared address space
// for the observation batch. Env i is handled by thread i % T, always, so
// the result does not depend on T -- a test asserts 1 thread == 8 threads
// to the bit.
//
// Synchronisation is a generation counter: the caller publishes actions,
// bumps the generation, and waits for T workers to report done. Workers spin
// briefly on the counter and then block on a condition variable, so an idle
// VecEnv (the policy is thinking) does not burn eight cores.
//
// Autoreset matches vec_env.py: when an episode ends, the observation
// returned is the NEW episode's first one and the ended episode's final
// observation is kept in terminal_obs() for the value bootstrap.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "vecenv/env.hpp"

namespace vecenv {

constexpr int kMaxWorkers = 8;   // the 8-of-14 thread budget on the box this was written on

class VecEnv {
public:
    // threads == 0 -> one per env, capped at kMaxWorkers. threads == 1 -> no
    // worker threads at all; everything runs on the caller's thread.
    VecEnv(const EnvConfig& cfg, int n, std::uint64_t seed, int threads = 0,
           bool allow_overcommit = false);
    ~VecEnv();
    VecEnv(const VecEnv&) = delete;
    VecEnv& operator=(const VecEnv&) = delete;

    int num_envs() const { return n_; }
    int num_threads() const { return threads_; }

    void reset();                              // every env: reset(), observe()
    void step(const double* actions);          // n * 14, row-major

    const float* obs() const { return obs_.data(); }                 // n * 48
    const double* rewards() const { return rew_.data(); }            // n
    const std::uint8_t* dones() const { return done_.data(); }       // n
    const StepInfo* infos() const { return info_.data(); }           // n
    const float* terminal_obs() const { return terminal_obs_.data(); }   // n * 48, valid where dones()

    Env& env(int i) { return *envs_[i]; }
    const Env& env(int i) const { return *envs_[i]; }

    void close();

private:
    void worker_(int tid);
    void run_env_(int i);
    void wait_generation_(std::uint64_t expected, int tid);

    int n_;
    int threads_;
    EnvConfig cfg_;
    ModelPtr prototype_;
    std::vector<std::unique_ptr<Env>> envs_;
    std::vector<std::thread> pool_;

    std::vector<double> actions_;
    std::vector<float> obs_, terminal_obs_;
    std::vector<double> rew_;
    std::vector<std::uint8_t> done_;
    std::vector<StepInfo> info_;

    std::atomic<std::uint64_t> generation_{0};
    std::atomic<int> remaining_{0};
    std::atomic<bool> stop_{false};
    std::mutex mtx_;
    std::condition_variable cv_;
    bool closed_ = false;
};

}  // namespace vecenv
