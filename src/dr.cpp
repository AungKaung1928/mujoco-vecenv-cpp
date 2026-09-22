#include "vecenv/dr.hpp"

#include <algorithm>

namespace vecenv {

namespace {
std::array<double, 5> as_array(const ActuatorClass& c) {
    return {c.damping, c.frictionloss, c.armature, c.kp, c.forcerange};
}
}  // namespace

DRConfig DRConfig::none() {
    DRConfig c;
    c.actuator = false;
    c.mass_lo = c.mass_hi = 1.0;
    c.friction_lo = c.friction_hi = 1.0;
    c.latency_lo = c.latency_hi = 0;
    c.init_noise_lo = c.init_noise_hi = 0.02;
    c.sensor_noise = false;
    return c;
}

std::array<double, 5> act_param_lo() {
    std::array<double, 5> lo = as_array(kActuatorClasses[0].second);
    for (const auto& kv : kActuatorClasses) {
        const auto a = as_array(kv.second);
        for (int k = 0; k < 5; ++k) lo[k] = std::min(lo[k], a[k]);
    }
    return lo;
}

std::array<double, 5> act_param_hi() {
    std::array<double, 5> hi = as_array(kActuatorClasses[0].second);
    for (const auto& kv : kActuatorClasses) {
        const auto a = as_array(kv.second);
        for (int k = 0; k < 5; ++k) hi[k] = std::max(hi[k], a[k]);
    }
    return hi;
}

ActuatorClass nominal_class() { return kActuatorClasses[0].second; }   // chosen_actuator

DRFactors sample_dr(const DRConfig& cfg, PCG64& rng) {
    DRFactors f;
    if (cfg.actuator) {
        const auto lo = act_param_lo(), hi = act_param_hi();
        double v[5];
        for (int k = 0; k < 5; ++k) v[k] = rng.uniform(lo[k], hi[k]);
        f.actuator = {v[0], v[1], v[2], v[3], v[4]};
    } else {
        f.actuator = nominal_class();
    }
    f.mass_scale = rng.uniform(cfg.mass_lo, cfg.mass_hi);
    f.floor_friction = rng.uniform(cfg.friction_lo, cfg.friction_hi);
    f.latency = static_cast<int>(rng.integers(cfg.latency_lo, cfg.latency_hi));
    f.init_noise = rng.uniform(cfg.init_noise_lo, cfg.init_noise_hi);
    f.sensor_noise = cfg.sensor_noise;
    return f;
}

DomainRandomizer::DomainRandomizer(const mjModel* m, const ModelIndex& ix) : ix_(ix) {
    dof_damping_.assign(m->dof_damping, m->dof_damping + m->nv);
    dof_frictionloss_.assign(m->dof_frictionloss, m->dof_frictionloss + m->nv);
    dof_armature_.assign(m->dof_armature, m->dof_armature + m->nv);
    gainprm_.assign(m->actuator_gainprm, m->actuator_gainprm + m->nu * mjNGAIN);
    biasprm_.assign(m->actuator_biasprm, m->actuator_biasprm + m->nu * mjNBIAS);
    forcerange_.assign(m->actuator_forcerange, m->actuator_forcerange + m->nu * 2);
    body_mass_.assign(m->body_mass, m->body_mass + m->nbody);
    body_inertia_.assign(m->body_inertia, m->body_inertia + m->nbody * 3);
    geom_friction_.assign(m->geom_friction, m->geom_friction + m->ngeom * 3);
}

void DomainRandomizer::restore(mjModel* m) const {
    std::copy(dof_damping_.begin(), dof_damping_.end(), m->dof_damping);
    std::copy(dof_frictionloss_.begin(), dof_frictionloss_.end(), m->dof_frictionloss);
    std::copy(dof_armature_.begin(), dof_armature_.end(), m->dof_armature);
    std::copy(gainprm_.begin(), gainprm_.end(), m->actuator_gainprm);
    std::copy(biasprm_.begin(), biasprm_.end(), m->actuator_biasprm);
    std::copy(forcerange_.begin(), forcerange_.end(), m->actuator_forcerange);
    std::copy(body_mass_.begin(), body_mass_.end(), m->body_mass);
    std::copy(body_inertia_.begin(), body_inertia_.end(), m->body_inertia);
    std::copy(geom_friction_.begin(), geom_friction_.end(), m->geom_friction);
}

void DomainRandomizer::apply(mjModel* m, const DRFactors& f) const {
    // Every write is relative to the captured nominal arrays, so applying the
    // same factors twice gives the same model: nothing compounds.
    restore(m);
    const auto& a = f.actuator;
    for (int i = 0; i < kNAct; ++i) {
        const int dof = ix_.qvel_i[i];
        m->dof_damping[dof] = a.damping;
        m->dof_frictionloss[dof] = a.frictionloss;
        m->dof_armature[dof] = a.armature;
    }
    // position actuator: gain = kp, bias = (0, -kp, -kv). Scale kp in both.
    for (int u = 0; u < m->nu; ++u) {
        const double kp_nom = gainprm_[u * mjNGAIN + 0];
        const double ratio = a.kp / kp_nom;
        m->actuator_gainprm[u * mjNGAIN + 0] = a.kp;
        m->actuator_biasprm[u * mjNBIAS + 1] = biasprm_[u * mjNBIAS + 1] * ratio;
        m->actuator_forcerange[2 * u] = -a.forcerange;
        m->actuator_forcerange[2 * u + 1] = a.forcerange;
    }
    for (int b = 0; b < m->nbody; ++b) {
        m->body_mass[b] = body_mass_[b] * f.mass_scale;
        for (int k = 0; k < 3; ++k) m->body_inertia[3 * b + k] = body_inertia_[3 * b + k] * f.mass_scale;
    }
    m->geom_friction[3 * ix_.floor_geom + 0] = geom_friction_[3 * ix_.floor_geom + 0] * f.floor_friction;
}

}  // namespace vecenv
