// Domain randomisation, mirroring microduck-rl/dr.py.
//
// The sampled values are applied to a model that this Env owns outright
// (one mjModel copy per environment), because they change the model. A
// shared model would make DR a data race and every environment identical.
#pragma once

#include "vecenv/mujoco.hpp"

#include <array>
#include <vector>

#include "vecenv/model.hpp"
#include "vecenv/rng.hpp"

namespace vecenv {

struct DRConfig {
    bool actuator = true;                 // draw the 5 servo params between the 4 fits
    double mass_lo = 0.85, mass_hi = 1.15;
    double friction_lo = 0.6, friction_hi = 1.2;
    int latency_lo = 0, latency_hi = 2;   // control steps, inclusive
    double init_noise_lo = 0.02, init_noise_hi = 0.3;
    bool sensor_noise = true;
    static DRConfig none();
};

struct DRFactors {
    ActuatorClass actuator{};
    double mass_scale = 1.0;
    double floor_friction = 1.0;
    int latency = 0;
    double init_noise = 0.02;
    bool sensor_noise = false;
};

std::array<double, 5> act_param_lo();
std::array<double, 5> act_param_hi();
ActuatorClass nominal_class();
// Draw order matches dr.py: actuator (5), mass, friction, latency, init_noise.
DRFactors sample_dr(const DRConfig& cfg, PCG64& rng);

class DomainRandomizer {
public:
    DomainRandomizer(const mjModel* m, const ModelIndex& ix);   // captures the nominal arrays
    void apply(mjModel* m, const DRFactors& f) const;           // model arrays only
    void restore(mjModel* m) const;
private:
    ModelIndex ix_;
    std::vector<double> dof_damping_, dof_frictionloss_, dof_armature_;
    std::vector<double> gainprm_, biasprm_, forcerange_;
    std::vector<double> body_mass_, body_inertia_, geom_friction_;
};

}  // namespace vecenv
