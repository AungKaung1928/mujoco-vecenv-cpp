#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#include "doctest.h"
#include "helpers.hpp"
#include "vecenv/dr.hpp"
#include "vecenv/env.hpp"

using namespace vecenv;

TEST_CASE("actuator ranges are the min/max over the four measured fits") {
    const auto lo = act_param_lo(), hi = act_param_hi();
    CHECK(lo[0] == doctest::Approx(0.041));
    CHECK(hi[0] == doctest::Approx(0.053));
    CHECK(lo[1] == doctest::Approx(0.0048));
    CHECK(hi[1] == doctest::Approx(0.032));
    CHECK(hi[1] / lo[1] == doctest::Approx(6.667).epsilon(0.01));   // friction loss disagrees 6.7x
    for (int k : {0, 2, 3, 4}) CHECK(hi[k] / lo[k] < 1.5);
    CHECK(nominal_class().kp == 0.55);
}

TEST_CASE("300 samples stay inside the declared ranges; none() is the nominal class") {
    DRConfig cfg;
    PCG64 rng(0);
    const auto lo = act_param_lo(), hi = act_param_hi();
    bool latency_seen[3] = {false, false, false};
    // observed min/max per factor: 5 actuator params, mass, friction, init noise
    double mn[8], mx[8];
    for (int k = 0; k < 8; ++k) { mn[k] = 1e300; mx[k] = -1e300; }
    for (int i = 0; i < 300; ++i) {
        const DRFactors f = sample_dr(cfg, rng);
        const double a[5] = {f.actuator.damping, f.actuator.frictionloss, f.actuator.armature, f.actuator.kp, f.actuator.forcerange};
        const double all[8] = {a[0], a[1], a[2], a[3], a[4], f.mass_scale, f.floor_friction, f.init_noise};
        for (int k = 0; k < 8; ++k) { mn[k] = std::min(mn[k], all[k]); mx[k] = std::max(mx[k], all[k]); }
        for (int k = 0; k < 5; ++k) { CHECK(a[k] >= lo[k]); CHECK(a[k] <= hi[k]); }
        CHECK(f.mass_scale >= 0.85); CHECK(f.mass_scale <= 1.15);
        CHECK(f.floor_friction >= 0.6); CHECK(f.floor_friction <= 1.2);
        CHECK(f.latency >= 0); CHECK(f.latency <= 2);
        if (f.latency >= 0 && f.latency <= 2) latency_seen[f.latency] = true;
        CHECK(f.init_noise >= 0.02); CHECK(f.init_noise <= 0.3);
        CHECK(f.sensor_noise);
    }
    // microduck-rl samples latency with integers(0, 2 + 1): 2 is included, so it must occur here too
    CHECK(latency_seen[0]); CHECK(latency_seen[1]); CHECK(latency_seen[2]);
    // inside the range is not enough: a range collapsed to its lower bound passes the
    // checks above. 300 uniform draws cover > 98 % of a range with near certainty.
    const double rlo[8] = {lo[0], lo[1], lo[2], lo[3], lo[4], 0.85, 0.6, 0.02};
    const double rhi[8] = {hi[0], hi[1], hi[2], hi[3], hi[4], 1.15, 1.2, 0.3};
    for (int k = 0; k < 8; ++k) CHECK(mx[k] - mn[k] > 0.9 * (rhi[k] - rlo[k]));
    const DRFactors f0 = sample_dr(DRConfig::none(), rng);
    CHECK(f0.actuator.kp == 0.55);
    CHECK(f0.latency == 0);
    CHECK(f0.init_noise == 0.02);
    CHECK_FALSE(f0.sensor_noise);
}

TEST_CASE("apply is exact, restore is exact, nothing compounds") {
    Env e(th::cfg(), th::proto(), 0);
    mjModel* m = e.model();
    const std::vector<double> mass0(m->body_mass, m->body_mass + m->nbody);
    const std::vector<double> damp0(m->dof_damping, m->dof_damping + m->nv);
    const std::vector<double> fric0(m->geom_friction, m->geom_friction + 3 * m->ngeom);
    DomainRandomizer rz(m, e.index());
    PCG64 rng(1);
    const DRFactors f = sample_dr(DRConfig{}, rng);
    rz.apply(m, f);
    for (int i = 0; i < kNAct; ++i) {
        CHECK(m->dof_damping[e.index().qvel_i[i]] == f.actuator.damping);
        CHECK(m->dof_frictionloss[e.index().qvel_i[i]] == f.actuator.frictionloss);
        CHECK(m->dof_armature[e.index().qvel_i[i]] == f.actuator.armature);
    }
    for (int d = 0; d < 6; ++d) CHECK(m->dof_damping[d] == damp0[static_cast<std::size_t>(d)]);   // free joint untouched
    for (int u = 0; u < m->nu; ++u) {
        CHECK(m->actuator_gainprm[u * mjNGAIN] == f.actuator.kp);
        CHECK(m->actuator_biasprm[u * mjNBIAS + 1] == doctest::Approx(-f.actuator.kp));
        CHECK(m->actuator_forcerange[2 * u] == -f.actuator.forcerange);
        CHECK(m->actuator_forcerange[2 * u + 1] == f.actuator.forcerange);
    }
    for (int b = 0; b < m->nbody; ++b) CHECK(m->body_mass[b] == mass0[static_cast<std::size_t>(b)] * f.mass_scale);
    const int fl = e.index().floor_geom;
    CHECK(m->geom_friction[3 * fl] == fric0[static_cast<std::size_t>(3 * fl)] * f.floor_friction);
    for (int g = 0; g < m->ngeom; ++g)
        if (g != fl) for (int k = 0; k < 3; ++k) CHECK(m->geom_friction[3 * g + k] == fric0[static_cast<std::size_t>(3 * g + k)]);
    for (int k = 0; k < 5; ++k) rz.apply(m, f);
    for (int b = 0; b < m->nbody; ++b) CHECK(m->body_mass[b] == mass0[static_cast<std::size_t>(b)] * f.mass_scale);
    rz.restore(m);
    for (int b = 0; b < m->nbody; ++b) CHECK(m->body_mass[b] == mass0[static_cast<std::size_t>(b)]);
    for (int d = 0; d < m->nv; ++d) CHECK(m->dof_damping[d] == damp0[static_cast<std::size_t>(d)]);
}

TEST_CASE("an env with DR draws a new servo gain at every reset and takes latency/init_noise from it") {
    auto c = th::cfg();
    c.dr = true;
    Env e(c, th::proto(), 0);
    std::set<double> kps;
    for (int k = 0; k < 5; ++k) {
        e.reset();
        kps.insert(e.model()->actuator_gainprm[0]);
        REQUIRE(e.dr_factors().has_value());
        CHECK(e.latency() == e.dr_factors()->latency);
        CHECK(e.init_noise() == e.dr_factors()->init_noise);
        CHECK(e.sensor_noise());
    }
    CHECK(kps.size() == 5);
}

TEST_CASE("each env owns its model: DR on one leaves the others and the prototype untouched") {
    const mjModel* p = th::proto();
    const double proto_mass = p->body_mass[1];
    auto c = th::cfg();
    Env plain(c, p, 0);
    c.dr = true;
    Env randomised(c, p, 1);
    randomised.reset();
    CHECK(randomised.model()->body_mass[1] != proto_mass);
    CHECK(plain.model()->body_mass[1] == proto_mass);
    CHECK(p->body_mass[1] == proto_mass);
}

TEST_CASE("sensor noise perturbs the gyro and orientation reads but gravity stays a unit vector") {
    auto c = th::cfg();
    c.init_noise = 0.0;
    c.n_pushes = 0;
    Env quiet(c, th::proto(), 0);
    c.sensor_noise = true;
    Env noisy(c, th::proto(), 0);
    quiet.reset(); noisy.reset();
    float a[kObsDim], b[kObsDim];
    quiet.observe(a);
    noisy.observe(b);
    bool differs = false;
    for (int i = 42; i < 48; ++i) differs |= (a[i] != b[i]);
    CHECK(differs);
    const double n = std::sqrt(double(b[45]) * b[45] + double(b[46]) * b[46] + double(b[47]) * b[47]);
    CHECK(std::fabs(n - 1.0) < 1e-5);
    for (int i = 0; i < 42; ++i) CHECK(a[i] == b[i]);    // noise touches only the IMU blocks
    // each block draws with its own sensor's sd from the XML (robot_groundcontact.xml: gyro 0.005, framequat 0.001)
    const ModelIndex ix = index_model(th::proto());
    CHECK(ix.gyro_sd == 0.005);
    CHECK(ix.quat_sd == 0.001);
}
