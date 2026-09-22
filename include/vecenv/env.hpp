// One Microduck: the same contract as microduck-rl/env.py, in C++.
//
// 48 numbers in (joint pos 14, joint vel 14, previous action 14, gyro 3,
// projected gravity 3), 14 position residuals out, 50 Hz over a 500 Hz
// physics step, fixed-length episodes, three seeded pushes, two reward
// versions. Every quantity in the observation is one the real robot can
// measure; that decision and its reasons are documented in the Python file
// and are not repeated here.
//
// Two ways to start an episode:
//   reset()          draws pushes and initial noise from this Env's own PCG64.
//                    NOT the same draw as the Python env for the same seed
//                    (see rng.hpp).
//   reset(params)    takes the drawn values from outside. The contract test
//                    runs the Python env, reads what it drew, hands it over,
//                    and then the two must agree to the bit.
#pragma once

#include "vecenv/mujoco.hpp"

#include <array>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "vecenv/dr.hpp"
#include "vecenv/model.hpp"
#include "vecenv/rng.hpp"

namespace vecenv {

enum class Reward { V1, V2 };
Reward parse_reward(const std::string& s);
const char* reward_name(Reward r);

struct EnvConfig {
    std::string variant = "groundcontact";
    std::string assets_dir;              // empty -> default_assets_dir()
    int episode_steps = 250;
    double action_scale = kActionScale;
    int n_pushes = 3;
    double push_speed_lo = 0.15, push_speed_hi = 0.45;
    double init_noise = 0.02;
    Reward reward = Reward::V1;
    int latency = 0;
    bool sensor_noise = false;
    bool dr = false;
    DRConfig dr_cfg;
};

// What an episode needs from outside when the draw is not this Env's own.
struct EpisodeParams {
    bool has_state = false;
    std::array<double, kNAct> qpos{};    // absolute values at the actuated joints, post-clip
    std::array<double, kNAct> qvel{};
    std::vector<int> push_at;            // sorted control steps
    std::vector<std::array<double, 2>> push_vel;
    std::optional<DRFactors> dr;         // applied before the keyframe reset if present
};

struct StepInfo {
    double reward = 0.0;
    std::array<double, 6> terms{};       // in term_names() order
    int n_terms = 0;
    double trunk_height = 0.0;
    double upright_cos = 0.0;
    bool fallen = false;
    bool pushed = false;
    bool truncated = false;              // done is ALWAYS a truncation here
    bool terminated = false;
    int t = 0;
};

class Env {
public:
    Env(const EnvConfig& cfg, const mjModel* prototype, std::uint64_t seed);

    void reset();
    void reset(const EpisodeParams& p);
    void reseed(std::uint64_t seed);     // replaces the RNG, like reset(seed=...) in Python
    // `action` is clipped to [-1, 1] here, as in Python. Writes obs (48 f32).
    void step(const double* action, float* obs, StepInfo* info);
    void observe(float* obs);            // consumes RNG when sensor noise is on

    const mjModel* model() const { return model_.get(); }
    mjModel* model() { return model_.get(); }
    const mjData* data() const { return data_.get(); }
    mjData* data() { return data_.get(); }
    const ModelIndex& index() const { return ix_; }
    const EnvConfig& config() const { return cfg_; }

    bool started() const { return started_; }
    bool done() const { return t_ >= cfg_.episode_steps; }
    int t() const { return t_; }
    int latency() const { return latency_; }
    double init_noise() const { return init_noise_; }
    bool sensor_noise() const { return sensor_noise_; }
    const std::vector<int>& push_at() const { return push_at_; }
    const std::vector<std::array<double, 2>>& push_vel() const { return push_vel_; }
    const std::optional<DRFactors>& dr_factors() const { return dr_current_; }
    const std::array<double, kNAct>& prev_action() const { return prev_action_; }
    PCG64& rng() { return rng_; }

    static int n_terms(Reward r) { return r == Reward::V1 ? 6 : 4; }
    static const char* const* term_names(Reward r);
    static const std::array<double, kObsDim>& obs_scale();

    // Zero-action PD baseline: hold STAND.
    static void zero_action(double* a) { for (int i = 0; i < kNAct; ++i) a[i] = 0.0; }

private:
    void begin_episode_(const std::optional<DRFactors>& f);
    void finish_reset_();
    void schedule_pushes_();
    void reward_terms_(const std::array<double, kNAct>& action, StepInfo* info) const;
    void proj_gravity_(double* g);

    EnvConfig cfg_;
    ModelPtr model_;
    DataPtr data_;
    ModelIndex ix_;
    std::optional<DomainRandomizer> dr_;
    std::optional<DRFactors> dr_current_;
    PCG64 rng_;

    int latency_;
    double init_noise_;
    bool sensor_noise_;
    std::deque<std::array<double, kNAct>> act_buf_;
    std::array<double, kNAct> prev_action_{};
    int t_ = 0;
    std::vector<int> push_at_;
    std::vector<std::array<double, 2>> push_vel_;
    std::size_t pushes_done_ = 0;
    bool started_ = false;
};

}  // namespace vecenv
