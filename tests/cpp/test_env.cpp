// The environment contract, in C++. Mirrors microduck-rl/test_env.py where a
// check makes sense without the Python side; the cross-language equality
// itself lives in tests/py/test_contract.py.
#include <cmath>
#include <stdexcept>

#include "doctest.h"
#include "helpers.hpp"
#include "vecenv/env.hpp"

using namespace vecenv;

TEST_CASE("observation is 48 finite floats and the action space is 14") {
    Env e(th::cfg(), th::proto(), 0);
    e.reset();
    float o[kObsDim];
    e.observe(o);
    for (float v : o) CHECK(std::isfinite(v));
    CHECK(e.model()->nu == kNAct);
    CHECK(Env::obs_scale()[14] == static_cast<double>(0.05f));
}

TEST_CASE("zero action commands STAND exactly; out-of-range actions clip to the joint limits") {
    auto c = th::cfg();
    c.init_noise = 0.0;
    Env e(c, th::proto(), 0);
    e.reset();
    double a[kNAct];
    Env::zero_action(a);
    float o[kObsDim];
    StepInfo info;
    e.step(a, o, &info);
    for (int i = 0; i < kNAct; ++i) CHECK(e.data()->ctrl[i] == e.index().default_pose[i]);
    for (double& v : a) v = 5.0;
    e.step(a, o, &info);
    for (int i = 0; i < kNAct; ++i) {
        CHECK(e.data()->ctrl[i] >= e.index().lo[i] - 1e-12);
        CHECK(e.data()->ctrl[i] <= e.index().hi[i] + 1e-12);
    }
    // prev_action block is the CLIPPED action just taken
    for (int i = 28; i < 42; ++i) CHECK(o[i] == 1.0f);
}

TEST_CASE("projected gravity reads (0,0,-1) upright and stays a unit vector while thrashing") {
    auto c = th::cfg();
    c.init_noise = 0.0;
    Env e(c, th::proto(), 0);
    e.reset();
    float o[kObsDim];
    e.observe(o);
    CHECK(std::fabs(o[45]) < 1e-6f);
    CHECK(std::fabs(o[46]) < 1e-6f);
    CHECK(std::fabs(o[47] + 1.0f) < 1e-6f);
    const auto acts = th::actions(120, 1);
    StepInfo info;
    for (const auto& a : acts) {
        e.step(a.data(), o, &info);
        const double n = std::sqrt(double(o[45]) * o[45] + double(o[46]) * o[46] + double(o[47]) * o[47]);
        CHECK(std::fabs(n - 1.0) < 1e-5);
    }
}

TEST_CASE("done fires exactly once, on the last step, and is always a truncation") {
    auto c = th::cfg();
    c.episode_steps = 50;
    Env e(c, th::proto(), 0);
    e.reset();
    const auto acts = th::actions(50, 2);
    int dones = 0;
    StepInfo info;
    float o[kObsDim];
    for (int k = 0; k < 50; ++k) {
        e.step(acts[static_cast<std::size_t>(k)].data(), o, &info);
        dones += e.done();
        CHECK(info.truncated == e.done());
        CHECK_FALSE(info.terminated);
        CHECK(info.t == k + 1);
    }
    CHECK(dones == 1);
    CHECK(e.done());
}

TEST_CASE("reward is the ordered sum of its terms, both versions") {
    for (Reward r : {Reward::V1, Reward::V2}) {
        auto c = th::cfg();
        c.reward = r;
        Env e(c, th::proto(), 3);
        e.reset();
        const auto acts = th::actions(60, 3);
        StepInfo info;
        float o[kObsDim];
        for (const auto& a : acts) {
            e.step(a.data(), o, &info);
            CHECK(info.n_terms == Env::n_terms(r));
            double s = 0.0;
            for (int k = 0; k < info.n_terms; ++k) s += info.terms[k];
            CHECK(s == info.reward);
            CHECK(info.terms[0] >= 0.0);           // upright is floored at 0
            CHECK(info.terms[1] >= 0.0);           // height is bounded [0, 1]
            CHECK(info.terms[1] <= 1.0);
        }
    }
}

TEST_CASE("same seed replays bit-for-bit across three constructions") {
    const auto acts = th::actions(200, 4);
    Env a(th::cfg(), th::proto(), 11), b(th::cfg(), th::proto(), 11), c(th::cfg(), th::proto(), 11);
    a.reset(); b.reset(); c.reset();
    const auto ra = th::rollout(a, acts, 200), rb = th::rollout(b, acts, 200), rc = th::rollout(c, acts, 200);
    CHECK(ra.obs == rb.obs);
    CHECK(rb.obs == rc.obs);
    CHECK(ra.rew == rb.rew);
    CHECK(rb.rew == rc.rew);
}

TEST_CASE("step before reset is refused") {
    Env e(th::cfg(), th::proto(), 0);
    double a[kNAct] = {};
    float o[kObsDim];
    StepInfo info;
    CHECK_THROWS_AS(e.step(a, o, &info), std::runtime_error);
}

TEST_CASE("different seeds draw different push schedules; pushes land on the scheduled step") {
    Env e1(th::cfg(), th::proto(), 1), e2(th::cfg(), th::proto(), 2);
    e1.reset(); e2.reset();
    CHECK(e1.push_at() != e2.push_at());
    CHECK(e1.push_at().size() == 3);
    for (int t : e1.push_at()) { CHECK(t >= 25); CHECK(t < 225); }

    // Same schedule, zero impulse on the second: identical until the push step.
    auto c = th::cfg();
    c.init_noise = 0.0;
    c.n_pushes = 1;
    Env ea(c, th::proto(), 7), eb(c, th::proto(), 7);
    ea.reset();
    EpisodeParams p;
    p.push_at = ea.push_at();
    p.push_vel = ea.push_vel();
    p.push_vel[0] = {0.0, 0.0};
    eb.reset(p);
    const int at = ea.push_at()[0];
    const auto acts = th::actions(at + 5, 5);
    const auto ra = th::rollout(ea, acts, at + 5), rb = th::rollout(eb, acts, at + 5);
    int first = -1;
    for (int k = 0; k < at + 5 && first < 0; ++k)
        if (ra.obs[static_cast<std::size_t>(k)] != rb.obs[static_cast<std::size_t>(k)]) first = k;
    CHECK(first == at);
}

TEST_CASE("the observation and the reward are one instant: a second forward pass changes nothing") {
    Env e(th::cfg(), th::proto(), 3);
    e.reset();
    const auto acts = th::actions(120, 6);
    float o[kObsDim], o2[kObsDim];
    StepInfo info;
    double worst_g = 0.0, worst_h = 0.0;
    for (const auto& a : acts) {
        e.step(a.data(), o, &info);
        mj_forward(e.model(), e.data());
        e.observe(o2);
        for (int i = 42; i < 45; ++i) worst_g = std::max(worst_g, std::fabs(double(o[i]) - o2[i]));
        worst_h = std::max(worst_h, std::fabs(trunk_height(e.model(), e.data(), e.index()) - info.trunk_height));
    }
    CHECK(worst_g < 1e-9);
    CHECK(worst_h < 1e-9);
}

TEST_CASE("reset clamps the initial joint state to the limits even at init_noise 0.6") {
    auto c = th::cfg();
    c.init_noise = 0.6;
    Env e(c, th::proto(), 5);
    double worst = -1.0;
    for (int k = 0; k < 30; ++k) {
        e.reset();
        for (int i = 0; i < kNAct; ++i) {
            const double q = e.data()->qpos[e.index().qpos_i[i]];
            worst = std::max(worst, std::max(e.index().lo[i] - q, q - e.index().hi[i]));
        }
    }
    CHECK(worst <= 1e-12);
}

TEST_CASE("latency delays the applied action by exactly `latency` control steps") {
    auto c = th::cfg();
    c.init_noise = 0.0;
    c.latency = 2;
    c.n_pushes = 0;
    Env e(c, th::proto(), 0);
    e.reset();
    double a[kNAct];
    for (double& v : a) v = 1.0;
    float o[kObsDim];
    StepInfo info;
    // steps 1 and 2 apply the zero actions the buffer was primed with
    e.step(a, o, &info);
    for (int i = 0; i < kNAct; ++i) CHECK(e.data()->ctrl[i] == e.index().default_pose[i]);
    e.step(a, o, &info);
    for (int i = 0; i < kNAct; ++i) CHECK(e.data()->ctrl[i] == e.index().default_pose[i]);
    // step 3 applies the action from step 1
    e.step(a, o, &info);
    bool moved = false;
    for (int i = 0; i < kNAct; ++i) moved |= (e.data()->ctrl[i] != e.index().default_pose[i]);
    CHECK(moved);
    // while prev_action in the observation was the emitted one all along
    for (int i = 28; i < 42; ++i) CHECK(o[i] == 1.0f);
}

TEST_CASE("the PD hold-pose baseline topples: a fallen flag appears within the episode") {
    Env e(th::cfg(), th::proto(), 0);
    e.reset();
    double a[kNAct];
    Env::zero_action(a);
    float o[kObsDim];
    StepInfo info;
    bool fell = false;
    for (int k = 0; k < 250; ++k) { e.step(a, o, &info); fell |= info.fallen; }
    CHECK(fell);
}

TEST_CASE("held-out variants: 48-dim observation through the strided index, same actuator order") {
    for (const char* v : {"groundcontact_backlash", "rollers", "walk_backlash"}) {
        Env e(th::cfg(v), th::proto(v), 0);
        e.reset();
        float o[kObsDim];
        e.observe(o);
        for (float x : o) CHECK(std::isfinite(x));
        CHECK(e.model()->nu == kNAct);
    }
    Env eb(th::cfg("groundcontact_backlash"), th::proto("groundcontact_backlash"), 0);
    CHECK(eb.model()->nq == 35);
    bool contiguous = true;
    for (int i = 1; i < kNAct; ++i) contiguous &= (eb.index().qpos_i[i] == eb.index().qpos_i[i - 1] + 1);
    CHECK_FALSE(contiguous);   // qpos[7:21] would read the wrong numbers here
    Env eg(th::cfg(), th::proto(), 0);
    contiguous = true;
    for (int i = 1; i < kNAct; ++i) contiguous &= (eg.index().qpos_i[i] == eg.index().qpos_i[i - 1] + 1);
    CHECK(contiguous);
}

TEST_CASE("unknown variant, bad reward string and malformed episode parameters are rejected") {
    CHECK_THROWS_AS(variant_file("no_such_variant"), std::invalid_argument);
    CHECK_THROWS_AS(parse_reward("v3"), std::invalid_argument);
    Env e(th::cfg(), th::proto(), 0);
    EpisodeParams p;
    p.push_at = {300};
    p.push_vel = {{0.1, 0.0}};
    CHECK_THROWS_AS(e.reset(p), std::invalid_argument);        // outside the episode
    p.push_at = {50, 40};
    p.push_vel = {{0.1, 0.0}, {0.1, 0.0}};
    CHECK_THROWS_AS(e.reset(p), std::invalid_argument);        // unsorted
    p.push_at = {40};
    CHECK_THROWS_AS(e.reset(p), std::invalid_argument);        // lengths differ
    p.push_vel = {{0.1, 0.0}};
    CHECK_THROWS_AS(e.reset(p), std::invalid_argument);        // init_noise > 0 but no state
}
