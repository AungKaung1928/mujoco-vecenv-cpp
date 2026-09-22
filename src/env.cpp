#include "vecenv/env.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "vecenv/numpy_compat.hpp"

namespace vecenv {

namespace {
const char* const kTermsV1[6] = {"upright", "height", "posture", "action_rate", "joint_vel", "effort"};
const char* const kTermsV2[4] = {"upright", "height", "action_rate", "joint_vel"};

// REWARD_V1 / REWARD_V2 in env.py, same order as the term names.
constexpr double kW1[6] = {1.0, 1.0, -0.10, -0.05, -2.0e-4, -0.02};
constexpr double kW2[4] = {1.0, 1.0, -0.05, -2.0e-3};
}  // namespace

Reward parse_reward(const std::string& s) {
    if (s == "v1") return Reward::V1;
    if (s == "v2") return Reward::V2;
    throw std::invalid_argument("reward must be 'v1' or 'v2', got '" + s + "'");
}

const char* reward_name(Reward r) { return r == Reward::V1 ? "v1" : "v2"; }

const char* const* Env::term_names(Reward r) { return r == Reward::V1 ? kTermsV1 : kTermsV2; }

const std::array<double, kObsDim>& Env::obs_scale() {
    // OBS_SCALE is a float32 array in Python and raw is float64, so the product
    // is raw * double(float32(scale)). 0.05f is not 0.05.
    static const std::array<double, kObsDim> s = [] {
        std::array<double, kObsDim> a{};
        for (int i = 0; i < 14; ++i) a[i] = 1.0;
        for (int i = 14; i < 28; ++i) a[i] = static_cast<double>(0.05f);
        for (int i = 28; i < 42; ++i) a[i] = 1.0;
        for (int i = 42; i < 45; ++i) a[i] = static_cast<double>(0.25f);
        for (int i = 45; i < 48; ++i) a[i] = 1.0;
        return a;
    }();
    return s;
}

Env::Env(const EnvConfig& cfg, const mjModel* prototype, std::uint64_t seed)
    : cfg_(cfg),
      model_(copy_model(prototype)),
      data_(make_data(model_.get())),
      ix_(index_model(model_.get())),
      rng_(seed),
      latency_(cfg.latency),
      init_noise_(cfg.init_noise),
      sensor_noise_(cfg.sensor_noise) {
    if (cfg_.episode_steps <= 0) throw std::invalid_argument("episode_steps must be positive");
    if (cfg_.latency < 0) throw std::invalid_argument("latency must be >= 0");
    if (cfg_.dr) dr_.emplace(model_.get(), ix_);
    mj_resetDataKeyframe(model_.get(), data_.get(), ix_.stand_key);
    mj_forward(model_.get(), data_.get());
    const double h = trunk_height(model_.get(), data_.get(), ix_);
    if (std::fabs(h - kStandHeight) > 1e-3) {
        throw std::runtime_error("STAND trunk height is " + std::to_string(h) + ", expected " +
                                 std::to_string(kStandHeight));
    }
}

void Env::reseed(std::uint64_t seed) { rng_ = PCG64(seed); }

void Env::begin_episode_(const std::optional<DRFactors>& f) {
    // Same order as MicroduckEnv.reset: DR first (it may change latency and
    // init_noise for this episode), then the keyframe, then the latency buffer.
    if (f) {
        if (!dr_) dr_.emplace(model_.get(), ix_);
        dr_->apply(model_.get(), *f);
        latency_ = f->latency;
        init_noise_ = f->init_noise;
        sensor_noise_ = f->sensor_noise;
        dr_current_ = *f;
    }
    mj_resetDataKeyframe(model_.get(), data_.get(), ix_.stand_key);
    mj_forward(model_.get(), data_.get());
    act_buf_.assign(static_cast<std::size_t>(latency_), std::array<double, kNAct>{});
}

void Env::finish_reset_() {
    prev_action_.fill(0.0);
    t_ = 0;
    pushes_done_ = 0;
    started_ = true;
}

void Env::schedule_pushes_() {
    // Kept half a second clear of both ends, as in Python: a push at t=0 is an
    // initial condition, one at the buzzer is never recovered from.
    const int margin = static_cast<int>(0.5 * kControlHz);
    const int lo = margin, hi = cfg_.episode_steps - margin;
    push_at_.clear();
    push_vel_.clear();
    if (cfg_.n_pushes <= 0 || hi <= lo) return;
    const int k = std::min(cfg_.n_pushes, hi - lo);
    // k distinct steps in [lo, hi): partial Fisher-Yates, then sorted.
    std::vector<int> pool(static_cast<std::size_t>(hi - lo));
    std::iota(pool.begin(), pool.end(), lo);
    for (int i = 0; i < k; ++i) {
        const auto j = static_cast<std::size_t>(rng_.integers(i, static_cast<std::int64_t>(pool.size()) - 1));
        std::swap(pool[static_cast<std::size_t>(i)], pool[j]);
    }
    push_at_.assign(pool.begin(), pool.begin() + k);
    std::sort(push_at_.begin(), push_at_.end());
    std::vector<double> speed(static_cast<std::size_t>(k)), theta(static_cast<std::size_t>(k));
    for (auto& s : speed) s = rng_.uniform(cfg_.push_speed_lo, cfg_.push_speed_hi);
    for (auto& th : theta) th = rng_.uniform(0.0, 2.0 * M_PI);
    push_vel_.resize(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        const auto u = static_cast<std::size_t>(i);
        push_vel_[u] = {speed[u] * std::cos(theta[u]), speed[u] * std::sin(theta[u])};
    }
}

void Env::reset() {
    std::optional<DRFactors> f;
    if (cfg_.dr) f = sample_dr(cfg_.dr_cfg, rng_);
    begin_episode_(f);
    if (init_noise_ > 0.0) {
        const double n = init_noise_;
        mjData* d = data_.get();
        for (int i = 0; i < kNAct; ++i) d->qpos[ix_.qpos_i[i]] += rng_.uniform(-n, n);
        for (int i = 0; i < kNAct; ++i) d->qvel[ix_.qvel_i[i]] += rng_.uniform(-5 * n, 5 * n);
        for (int i = 0; i < kNAct; ++i) d->qpos[ix_.qpos_i[i]] = np::clip(d->qpos[ix_.qpos_i[i]], ix_.lo[i], ix_.hi[i]);
        mj_forward(model_.get(), d);
    }
    finish_reset_();
    schedule_pushes_();
}

void Env::reset(const EpisodeParams& p) {
    if (p.push_at.size() != p.push_vel.size()) throw std::invalid_argument("push_at and push_vel differ in length");
    if (!std::is_sorted(p.push_at.begin(), p.push_at.end())) throw std::invalid_argument("push_at must be sorted");
    for (int t : p.push_at)
        if (t < 0 || t >= cfg_.episode_steps)
            throw std::invalid_argument("push step " + std::to_string(t) + " is outside the episode");
    begin_episode_(p.dr);
    // The Python env only touches the state (and only runs its second
    // mj_forward) when init_noise > 0. Mirror that exactly: a second forward
    // pass on an unchanged state is NOT a no-op, because the solver warm-starts
    // from the previous acceleration and lands a few ulps elsewhere.
    if (init_noise_ > 0.0) {
        if (!p.has_state) throw std::invalid_argument("init_noise > 0 but EpisodeParams carries no state");
        mjData* d = data_.get();
        for (int i = 0; i < kNAct; ++i) d->qpos[ix_.qpos_i[i]] = p.qpos[i];
        for (int i = 0; i < kNAct; ++i) d->qvel[ix_.qvel_i[i]] = p.qvel[i];
        mj_forward(model_.get(), d);
    }
    finish_reset_();
    push_at_ = p.push_at;
    push_vel_ = p.push_vel;
}

void Env::proj_gravity_(double* g) {
    // World -Z in the trunk frame; (0, 0, -1) when upright. Same three MuJoCo
    // calls as the Python: normalise (noise denormalises), conjugate, rotate.
    mjtNum q[4];
    for (int i = 0; i < 4; ++i) q[i] = data_->sensordata[ix_.quat_adr + i];
    if (sensor_noise_ && ix_.quat_sd > 0.0)
        for (int i = 0; i < 4; ++i) q[i] += rng_.normal(0.0, ix_.quat_sd);
    mju_normalize4(q);
    mjtNum qinv[4];
    mju_negQuat(qinv, q);
    const mjtNum down[3] = {0.0, 0.0, -1.0};
    mju_rotVecQuat(g, down, qinv);
}

void Env::observe(float* obs) {
    const mjData* d = data_.get();
    double raw[kObsDim];
    double gyro[3];
    for (int i = 0; i < 3; ++i) gyro[i] = d->sensordata[ix_.gyro_adr + i];
    if (sensor_noise_ && ix_.gyro_sd > 0.0)
        for (int i = 0; i < 3; ++i) gyro[i] += rng_.normal(0.0, ix_.gyro_sd);
    for (int i = 0; i < kNAct; ++i) raw[i] = d->qpos[ix_.qpos_i[i]] - ix_.default_pose[i];
    for (int i = 0; i < kNAct; ++i) raw[14 + i] = d->qvel[ix_.qvel_i[i]];
    for (int i = 0; i < kNAct; ++i) raw[28 + i] = prev_action_[i];
    for (int i = 0; i < 3; ++i) raw[42 + i] = gyro[i];
    proj_gravity_(raw + 45);
    const auto& s = obs_scale();
    for (int i = 0; i < kObsDim; ++i) obs[i] = static_cast<float>(raw[i] * s[i]);
}

void Env::reward_terms_(const std::array<double, kNAct>& action, StepInfo* info) const {
    const mjData* d = data_.get();
    double dq2[kNAct], da2[kNAct];
    for (int i = 0; i < kNAct; ++i) {
        const double dq = d->qvel[ix_.qvel_i[i]];
        dq2[i] = dq * dq;
        const double da = action[i] - prev_action_[i];
        da2[i] = da * da;
    }
    const double h = trunk_height(model_.get(), d, ix_);
    const double cos = upright_cos(model_.get(), d, ix_);
    if (cfg_.reward == Reward::V1) {
        double q2[kNAct], f2[kNAct];
        for (int i = 0; i < kNAct; ++i) {
            const double q = d->qpos[ix_.qpos_i[i]] - ix_.default_pose[i];
            q2[i] = q * q;
            const double f = d->actuator_force[i];
            f2[i] = f * f;
        }
        const double dh = h - kStandHeight;
        info->n_terms = 6;
        info->terms[0] = kW1[0] * std::max(0.0, cos);
        info->terms[1] = kW1[1] * std::exp(-np::py_pow(dh / 0.03, 2.0));
        info->terms[2] = kW1[2] * np::mean(q2, kNAct);
        info->terms[3] = kW1[3] * np::mean(da2, kNAct);
        info->terms[4] = kW1[4] * np::mean(dq2, kNAct);
        info->terms[5] = kW1[5] * np::mean(f2, kNAct);
    } else {
        info->n_terms = 4;
        info->terms[0] = kW2[0] * std::max(0.0, cos);
        info->terms[1] = kW2[1] * np::clip(h / kStandHeight, 0.0, 1.0);
        info->terms[2] = kW2[2] * np::mean(da2, kNAct);
        info->terms[3] = kW2[3] * np::mean(dq2, kNAct);
    }
    // Python: float(sum(terms.values())), left to right from 0.
    double r = 0.0;
    for (int k = 0; k < info->n_terms; ++k) r += info->terms[k];
    info->reward = r;
    info->trunk_height = h;
    info->upright_cos = cos;
}

void Env::step(const double* action, float* obs, StepInfo* info) {
    if (!started_) {
        throw std::runtime_error(
            "call reset() before step(). A freshly constructed env has an empty push "
            "schedule, so stepping it produces a push-free episode and raises nothing.");
    }
    std::array<double, kNAct> a;
    for (int i = 0; i < kNAct; ++i) a[i] = np::clip(action[i], -1.0, 1.0);

    // Latency: apply the action emitted `latency` steps ago; prev_action in
    // the observation stays the one just emitted, which is what the policy knows.
    const double* applied = a.data();
    std::array<double, kNAct> popped;
    if (latency_ > 0) {
        act_buf_.push_back(a);
        popped = act_buf_.front();
        act_buf_.pop_front();
        applied = popped.data();
    }
    mjModel* m = model_.get();
    mjData* d = data_.get();
    for (int i = 0; i < kNAct; ++i)
        d->ctrl[i] = np::clip(ix_.default_pose[i] + cfg_.action_scale * applied[i], ix_.lo[i], ix_.hi[i]);

    bool pushed = false;
    if (pushes_done_ < push_at_.size() && t_ == push_at_[pushes_done_]) {
        d->qvel[0] += push_vel_[pushes_done_][0];
        d->qvel[1] += push_vel_[pushes_done_][1];
        ++pushes_done_;
        pushed = true;
    }

    for (int k = 0; k < kSubsteps; ++k) mj_step(m, d);
    // mj_step leaves sensordata/xpos one substep behind qpos/qvel. One forward
    // pass so the observation and the reward are a single instant. ~8% of the
    // control step, and not an optimisation target.
    mj_forward(m, d);

    reward_terms_(a, info);
    prev_action_ = a;
    ++t_;
    const bool done = t_ >= cfg_.episode_steps;
    info->fallen = info->trunk_height < kFallHeight;
    info->pushed = pushed;
    info->t = t_;
    info->truncated = done;
    info->terminated = false;
    observe(obs);
}

}  // namespace vecenv
