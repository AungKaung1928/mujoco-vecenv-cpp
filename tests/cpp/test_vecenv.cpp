// The threaded layer must be invisible: the same numbers at any thread count,
// and the same numbers as N environments stepped one after another.
#include <atomic>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <dlfcn.h>
#include <pthread.h>

#include "doctest.h"
#include "helpers.hpp"
#include "vecenv/vecenv.hpp"

using namespace vecenv;

namespace {
struct Trace {
    std::vector<float> obs, term;
    std::vector<double> rew;
    std::vector<std::uint8_t> done;
};

Trace run(VecEnv& v, int steps, std::uint64_t seed) {
    const int n = v.num_envs();
    PCG64 r(seed);
    std::vector<double> a(static_cast<std::size_t>(n) * kNAct);
    Trace t;
    v.reset();
    for (int k = 0; k < steps; ++k) {
        for (auto& x : a) x = r.uniform(-1.0, 1.0);
        v.step(a.data());
        t.obs.insert(t.obs.end(), v.obs(), v.obs() + static_cast<std::size_t>(n) * kObsDim);
        t.rew.insert(t.rew.end(), v.rewards(), v.rewards() + n);
        t.done.insert(t.done.end(), v.dones(), v.dones() + n);
        for (int i = 0; i < n; ++i)
            if (v.dones()[i]) t.term.insert(t.term.end(), v.terminal_obs() + static_cast<std::size_t>(i) * kObsDim,
                                            v.terminal_obs() + static_cast<std::size_t>(i + 1) * kObsDim);
    }
    return t;
}
}  // namespace

TEST_CASE("1, 2 and 4 threads produce identical observations, rewards, dones and terminal observations") {
    const auto c = th::cfg();
    VecEnv v1(c, 4, 100, 1), v2(c, 4, 100, 2), v4(c, 4, 100, 4);
    const auto t1 = run(v1, 300, 9), t2 = run(v2, 300, 9), t4 = run(v4, 300, 9);
    CHECK(t1.obs == t2.obs);
    CHECK(t2.obs == t4.obs);
    CHECK(t1.rew == t4.rew);
    CHECK(t1.done == t4.done);
    CHECK(t1.term == t4.term);
    CHECK(!t1.term.empty());          // 300 steps crossed an episode boundary
    CHECK(v1.num_threads() == 1);
    CHECK(v4.num_threads() == 4);
}

TEST_CASE("N threaded envs reproduce N sequential envs seeded seed+i, autoreset included") {
    const auto c = th::cfg();
    const int n = 3, steps = 260;
    VecEnv v(c, n, 100, 3);
    const auto tv = run(v, steps, 21);

    std::vector<Env> seq;
    seq.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) seq.emplace_back(c, th::proto(), 100 + static_cast<std::uint64_t>(i));
    for (auto& e : seq) e.reset();
    PCG64 r(21);
    std::vector<double> a(static_cast<std::size_t>(n) * kNAct);
    Trace ts;
    for (int k = 0; k < steps; ++k) {
        for (auto& x : a) x = r.uniform(-1.0, 1.0);
        for (int i = 0; i < n; ++i) {
            float o[kObsDim];
            StepInfo info;
            Env& e = seq[static_cast<std::size_t>(i)];
            e.step(a.data() + static_cast<std::size_t>(i) * kNAct, o, &info);
            ts.rew.push_back(info.reward);
            ts.done.push_back(e.done() ? 1 : 0);
            if (e.done()) {
                ts.term.insert(ts.term.end(), o, o + kObsDim);
                e.reset();
                e.observe(o);
            }
            ts.obs.insert(ts.obs.end(), o, o + kObsDim);
        }
    }
    CHECK(tv.obs == ts.obs);
    CHECK(tv.rew == ts.rew);
    CHECK(tv.done == ts.done);
    CHECK(tv.term == ts.term);
}

TEST_CASE("thread budget: more than 8 threads is refused unless overcommit is explicit; 0 means one per env, capped") {
    const auto c = th::cfg();
    CHECK_THROWS_AS(VecEnv(c, 9, 0, 9), std::invalid_argument);
    VecEnv v(c, 9, 0, 0);
    CHECK(v.num_threads() == 8);
    CHECK(v.num_envs() == 9);
    VecEnv w(c, 2, 0, 0);
    CHECK(w.num_threads() == 2);
    // more threads than envs: capped to one per env before the budget check, so 9 for 3 envs is accepted
    VecEnv x(c, 3, 0, 9);
    CHECK(x.num_threads() == 3);
}

TEST_CASE("reset clears the dones and rewards left by the last step") {
    const auto c = th::cfg();
    VecEnv v(c, 4, 100, 2);
    v.reset();
    PCG64 r(9);
    std::vector<double> a(4 * kNAct);
    bool any_done = false;
    for (int k = 0; k < 300 && !any_done; ++k) {
        for (auto& x : a) x = r.uniform(-1.0, 1.0);
        v.step(a.data());
        for (int i = 0; i < 4; ++i) any_done = any_done || v.dones()[i] != 0;
    }
    REQUIRE(any_done);                // same seeds and actions as the parity test, which crosses an episode end in 300 steps
    v.reset();
    for (int i = 0; i < 4; ++i) {
        CHECK(v.dones()[i] == 0);
        CHECK(v.rewards()[i] == 0.0);
    }
}

TEST_CASE("close is idempotent and stepping a closed VecEnv throws") {
    const auto c = th::cfg();
    VecEnv v(c, 2, 0, 2);
    v.reset();
    v.close();
    v.close();
    std::vector<double> a(2 * kNAct, 0.0);
    CHECK_THROWS_AS(v.step(a.data()), std::runtime_error);
}

TEST_CASE("stepping before reset throws on the calling thread, also with worker threads") {
    // Env::step throws before reset(); on a pool thread that exception would be std::terminate.
    const auto c = th::cfg();
    std::vector<double> a(4 * kNAct, 0.0);
    for (int threads : {1, 4}) {
        VecEnv v(c, 4, 0, threads);
        CHECK_THROWS_AS(v.step(a.data()), std::runtime_error);
        v.reset();
        CHECK_NOTHROW(v.step(a.data()));
    }
}

#if !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
// Sanitizers interpose pthread_create themselves, so this seam runs in the plain build only.
// The test binary's own pthread_create wins over libc's (symbol interposition), also for
// calls from libstdc++. It fails std::thread starts once armed; MuJoCo's internal pool
// (libc++ inside libmujoco) is told apart by where its start routine lives.
namespace {
std::atomic<int> g_std_threads_left{-1};   // -1 = disarmed
}
extern "C" int pthread_create(pthread_t* t, const pthread_attr_t* a, void* (*f)(void*), void* arg) {
    using Fn = int (*)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
    static const Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "pthread_create"));
    Dl_info info{};
    const bool from_std_thread = dladdr(reinterpret_cast<void*>(f), &info) && info.dli_fname &&
                                 std::strstr(info.dli_fname, "libstdc++");
    if (from_std_thread && g_std_threads_left.load() >= 0 && g_std_threads_left.fetch_sub(1) == 0) return EAGAIN;
    return real(t, a, f, arg);
}

TEST_CASE("a worker thread that cannot be started is an exception, not std::terminate") {
    const auto c = th::cfg();
    g_std_threads_left = 1;                     // the 1st pool thread starts, the 2nd fails
    CHECK_THROWS_AS(VecEnv(c, 4, 0, 4), std::system_error);
    g_std_threads_left = -1;
    VecEnv ok(c, 4, 0, 4);                      // disarmed: the pool starts normally
    CHECK(ok.num_threads() == 4);
}
#endif
