// The threaded layer must be invisible: the same numbers at any thread count,
// and the same numbers as N environments stepped one after another.
#include <stdexcept>
#include <vector>

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
