// pybind11 surface. Thin: numpy in, numpy out, the info dict shaped like
// vec_env.py's so microduck-rl's ppo.py runs on it unchanged.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <string>

#include "vecenv/dr.hpp"
#include "vecenv/env.hpp"
#include "vecenv/model.hpp"
#include "vecenv/rng.hpp"
#include "vecenv/vecenv.hpp"

namespace py = pybind11;
using namespace vecenv;

namespace {

py::array_t<float> obs_array(const float* src, int n) {
    py::array_t<float> out({n, kObsDim});
    std::memcpy(out.mutable_data(), src, sizeof(float) * static_cast<std::size_t>(n) * kObsDim);
    return out;
}

py::array_t<float> obs_row(const float* src) {
    py::array_t<float> out(kObsDim);
    std::memcpy(out.mutable_data(), src, sizeof(float) * kObsDim);
    return out;
}

py::dict info_dict(const StepInfo& i, Reward r, bool with_terms) {
    py::dict d;
    d["fallen"] = i.fallen;
    d["upright_cos"] = i.upright_cos;
    d["trunk_height"] = i.trunk_height;
    d["pushed"] = i.pushed;
    d["t"] = i.t;
    d["truncated"] = i.truncated;
    d["terminated"] = i.terminated;
    if (with_terms) {
        py::dict terms;
        const char* const* names = Env::term_names(r);
        for (int k = 0; k < i.n_terms; ++k) terms[names[k]] = i.terms[k];
        d["terms"] = terms;
    }
    return d;
}

std::array<double, kNAct> to_action(const py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
    if (a.size() != kNAct) throw std::invalid_argument("action must have 14 elements");
    std::array<double, kNAct> out;
    std::memcpy(out.data(), a.data(), sizeof(double) * kNAct);
    return out;
}

py::array_t<double> model_array(const Env& e, const std::string& name) {
    const mjModel* m = e.model();
    const mjData* d = e.data();
    auto vec = [](const double* p, int n) {
        py::array_t<double> out(n);
        std::memcpy(out.mutable_data(), p, sizeof(double) * static_cast<std::size_t>(n));
        return out;
    };
    auto mat = [](const double* p, int r, int c) {
        py::array_t<double> out({r, c});
        std::memcpy(out.mutable_data(), p, sizeof(double) * static_cast<std::size_t>(r) * c);
        return out;
    };
    if (name == "dof_damping") return vec(m->dof_damping, m->nv);
    if (name == "dof_frictionloss") return vec(m->dof_frictionloss, m->nv);
    if (name == "dof_armature") return vec(m->dof_armature, m->nv);
    if (name == "actuator_gainprm") return mat(m->actuator_gainprm, m->nu, mjNGAIN);
    if (name == "actuator_biasprm") return mat(m->actuator_biasprm, m->nu, mjNBIAS);
    if (name == "actuator_forcerange") return mat(m->actuator_forcerange, m->nu, 2);
    if (name == "body_mass") return vec(m->body_mass, m->nbody);
    if (name == "body_inertia") return mat(m->body_inertia, m->nbody, 3);
    if (name == "geom_friction") return mat(m->geom_friction, m->ngeom, 3);
    if (name == "qpos") return vec(d->qpos, m->nq);
    if (name == "qvel") return vec(d->qvel, m->nv);
    if (name == "ctrl") return vec(d->ctrl, m->nu);
    if (name == "actuator_force") return vec(d->actuator_force, m->nu);
    if (name == "sensordata") return vec(d->sensordata, m->nsensordata);
    if (name == "xpos") return mat(d->xpos, m->nbody, 3);
    throw std::invalid_argument("unknown array '" + name + "'");
}

Env* make_env(const EnvConfig& cfg, std::uint64_t seed) {
    check_library_version();
    const std::string assets = cfg.assets_dir.empty() ? default_assets_dir() : cfg.assets_dir;
    ModelPtr proto = load_variant(assets, cfg.variant);
    return new Env(cfg, proto.get(), seed);   // Env copies the model; the prototype dies here
}

}  // namespace

PYBIND11_MODULE(_vecenv_cpp, m) {
    m.doc() = "Threaded C++ MuJoCo environment for the Microduck, same contract as microduck-rl/env.py";
    m.attr("OBS_DIM") = kObsDim;
    m.attr("ACT_DIM") = kNAct;
    m.attr("MAX_WORKERS") = kMaxWorkers;
    m.attr("ACTION_SCALE") = kActionScale;
    m.attr("FALL_HEIGHT") = kFallHeight;
    m.attr("STAND_HEIGHT") = kStandHeight;
    m.def("mujoco_version", [] { return mj_version(); });
    m.def("header_version", [] { return static_cast<int>(mjVERSION_HEADER); });
    m.def("check_library_version", &check_library_version);
    m.def("default_assets_dir", &default_assets_dir);
    m.def("act_param_lo", [] { return act_param_lo(); });
    m.def("act_param_hi", [] { return act_param_hi(); });
    m.def("obs_scale", [] {
        py::array_t<double> out(kObsDim);
        std::memcpy(out.mutable_data(), Env::obs_scale().data(), sizeof(double) * kObsDim);
        return out;
    });
    m.def("term_names", [](const std::string& r) {
        const Reward rw = parse_reward(r);
        std::vector<std::string> out;
        for (int k = 0; k < Env::n_terms(rw); ++k) out.emplace_back(Env::term_names(rw)[k]);
        return out;
    });

    py::class_<PCG64>(m, "PCG64")
        .def(py::init<std::uint64_t>(), py::arg("seed"))
        .def_static("from_words", &PCG64::from_words, py::arg("state_hi"), py::arg("state_lo"),
                    py::arg("inc_hi"), py::arg("inc_lo"))
        .def("random_raw", [](PCG64& r, std::size_t n) {
            py::array_t<std::uint64_t> out(static_cast<py::ssize_t>(n));
            auto* p = out.mutable_data();
            for (std::size_t i = 0; i < n; ++i) p[i] = r.next_u64();
            return out;
        }, py::arg("n"))
        .def("random", [](PCG64& r, std::size_t n) {
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            auto* p = out.mutable_data();
            for (std::size_t i = 0; i < n; ++i) p[i] = r.next_double();
            return out;
        }, py::arg("n"))
        .def("uniform", [](PCG64& r, double lo, double hi, std::size_t n) {
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            auto* p = out.mutable_data();
            for (std::size_t i = 0; i < n; ++i) p[i] = r.uniform(lo, hi);
            return out;
        }, py::arg("lo"), py::arg("hi"), py::arg("n"))
        .def("normal", [](PCG64& r, double mean, double sd, std::size_t n) {
            py::array_t<double> out(static_cast<py::ssize_t>(n));
            auto* p = out.mutable_data();
            for (std::size_t i = 0; i < n; ++i) p[i] = r.normal(mean, sd);
            return out;
        }, py::arg("mean"), py::arg("sd"), py::arg("n"))
        .def("integers", [](PCG64& r, std::int64_t lo, std::int64_t hi, std::size_t n) {
            py::array_t<std::int64_t> out(static_cast<py::ssize_t>(n));
            auto* p = out.mutable_data();
            for (std::size_t i = 0; i < n; ++i) p[i] = r.integers(lo, hi);
            return out;
        }, py::arg("lo"), py::arg("hi"), py::arg("n"));

    py::class_<ActuatorClass>(m, "ActuatorClass")
        .def(py::init<>())
        .def(py::init([](double damping, double frictionloss, double armature, double kp, double forcerange) {
            return ActuatorClass{damping, frictionloss, armature, kp, forcerange};
        }), py::arg("damping"), py::arg("frictionloss"), py::arg("armature"), py::arg("kp"), py::arg("forcerange"))
        .def_readwrite("damping", &ActuatorClass::damping)
        .def_readwrite("frictionloss", &ActuatorClass::frictionloss)
        .def_readwrite("armature", &ActuatorClass::armature)
        .def_readwrite("kp", &ActuatorClass::kp)
        .def_readwrite("forcerange", &ActuatorClass::forcerange);

    py::class_<DRConfig>(m, "DRConfig")
        .def(py::init<>())
        .def_static("none", &DRConfig::none)
        .def_readwrite("actuator", &DRConfig::actuator)
        .def_readwrite("mass_lo", &DRConfig::mass_lo).def_readwrite("mass_hi", &DRConfig::mass_hi)
        .def_readwrite("friction_lo", &DRConfig::friction_lo).def_readwrite("friction_hi", &DRConfig::friction_hi)
        .def_readwrite("latency_lo", &DRConfig::latency_lo).def_readwrite("latency_hi", &DRConfig::latency_hi)
        .def_readwrite("init_noise_lo", &DRConfig::init_noise_lo).def_readwrite("init_noise_hi", &DRConfig::init_noise_hi)
        .def_readwrite("sensor_noise", &DRConfig::sensor_noise);

    py::class_<DRFactors>(m, "DRFactors")
        .def(py::init<>())
        .def_readwrite("actuator", &DRFactors::actuator)
        .def_readwrite("mass_scale", &DRFactors::mass_scale)
        .def_readwrite("floor_friction", &DRFactors::floor_friction)
        .def_readwrite("latency", &DRFactors::latency)
        .def_readwrite("init_noise", &DRFactors::init_noise)
        .def_readwrite("sensor_noise", &DRFactors::sensor_noise);

    py::class_<EnvConfig>(m, "EnvConfig")
        .def(py::init<>())
        .def_readwrite("variant", &EnvConfig::variant)
        .def_readwrite("assets_dir", &EnvConfig::assets_dir)
        .def_readwrite("episode_steps", &EnvConfig::episode_steps)
        .def_readwrite("action_scale", &EnvConfig::action_scale)
        .def_readwrite("n_pushes", &EnvConfig::n_pushes)
        .def_readwrite("push_speed_lo", &EnvConfig::push_speed_lo)
        .def_readwrite("push_speed_hi", &EnvConfig::push_speed_hi)
        .def_readwrite("init_noise", &EnvConfig::init_noise)
        .def_property("reward", [](const EnvConfig& c) { return std::string(reward_name(c.reward)); },
                      [](EnvConfig& c, const std::string& s) { c.reward = parse_reward(s); })
        .def_readwrite("latency", &EnvConfig::latency)
        .def_readwrite("sensor_noise", &EnvConfig::sensor_noise)
        .def_readwrite("dr", &EnvConfig::dr)
        .def_readwrite("dr_cfg", &EnvConfig::dr_cfg);

    py::class_<EpisodeParams>(m, "EpisodeParams")
        .def(py::init<>())
        .def("set_state", [](EpisodeParams& p, py::array_t<double, py::array::c_style | py::array::forcecast> qpos,
                             py::array_t<double, py::array::c_style | py::array::forcecast> qvel) {
            if (qpos.size() != kNAct || qvel.size() != kNAct) throw std::invalid_argument("qpos/qvel must have 14 elements");
            std::memcpy(p.qpos.data(), qpos.data(), sizeof(double) * kNAct);
            std::memcpy(p.qvel.data(), qvel.data(), sizeof(double) * kNAct);
            p.has_state = true;
        }, py::arg("qpos"), py::arg("qvel"))
        .def("set_pushes", [](EpisodeParams& p, const std::vector<int>& at,
                              py::array_t<double, py::array::c_style | py::array::forcecast> vel) {
            if (vel.ndim() != 2 || vel.shape(1) != 2 || static_cast<std::size_t>(vel.shape(0)) != at.size())
                throw std::invalid_argument("push_vel must be (len(push_at), 2)");
            p.push_at = at;
            p.push_vel.resize(at.size());
            for (std::size_t i = 0; i < at.size(); ++i) p.push_vel[i] = {vel.at(i, 0), vel.at(i, 1)};
        }, py::arg("push_at"), py::arg("push_vel"))
        .def_property("dr", [](const EpisodeParams& p) -> py::object {
            return p.dr ? py::cast(*p.dr) : py::none();
        }, [](EpisodeParams& p, py::object o) {
            if (o.is_none()) p.dr.reset(); else p.dr = o.cast<DRFactors>();
        })
        .def_readonly("has_state", &EpisodeParams::has_state);

    py::class_<Env>(m, "Env")
        .def(py::init(&make_env), py::arg("config"), py::arg("seed") = 0)
        .def("reset", [](Env& e) {
            e.reset();
            float o[kObsDim];
            e.observe(o);
            return obs_row(o);
        })
        .def("reset", [](Env& e, const EpisodeParams& p) {
            e.reset(p);
            float o[kObsDim];
            e.observe(o);
            return obs_row(o);
        }, py::arg("params"))
        .def("reseed", &Env::reseed, py::arg("seed"))
        .def("step", [](Env& e, py::array_t<double, py::array::c_style | py::array::forcecast> action, bool with_terms) {
            const auto a = to_action(action);
            float o[kObsDim];
            StepInfo info;
            e.step(a.data(), o, &info);
            return py::make_tuple(obs_row(o), info.reward, e.done(), info_dict(info, e.config().reward, with_terms));
        }, py::arg("action"), py::arg("with_terms") = true)
        .def("observe", [](Env& e) {
            float o[kObsDim];
            e.observe(o);
            return obs_row(o);
        })
        .def("zero_action", [](const Env&) {
            py::array_t<double> a(kNAct);
            Env::zero_action(a.mutable_data());
            return a;
        })
        .def("model_array", &model_array, py::arg("name"))
        .def_property_readonly("t", &Env::t)
        .def_property_readonly("done", &Env::done)
        .def_property_readonly("started", &Env::started)
        .def_property_readonly("latency", &Env::latency)
        .def_property_readonly("init_noise", &Env::init_noise)
        .def_property_readonly("sensor_noise", &Env::sensor_noise)
        .def_property_readonly("push_at", [](const Env& e) { return e.push_at(); })
        .def_property_readonly("push_vel", [](const Env& e) {
            const auto& v = e.push_vel();
            py::array_t<double> out({static_cast<py::ssize_t>(v.size()), static_cast<py::ssize_t>(2)});
            for (std::size_t i = 0; i < v.size(); ++i) { out.mutable_at(i, 0) = v[i][0]; out.mutable_at(i, 1) = v[i][1]; }
            return out;
        })
        .def_property_readonly("prev_action", [](const Env& e) {
            py::array_t<double> out(kNAct);
            std::memcpy(out.mutable_data(), e.prev_action().data(), sizeof(double) * kNAct);
            return out;
        })
        .def_property_readonly("dr_factors", [](const Env& e) -> py::object {
            return e.dr_factors() ? py::cast(*e.dr_factors()) : py::none();
        })
        .def_property_readonly("qpos_index", [](const Env& e) { return std::vector<int>(e.index().qpos_i.begin(), e.index().qpos_i.end()); })
        .def_property_readonly("qvel_index", [](const Env& e) { return std::vector<int>(e.index().qvel_i.begin(), e.index().qvel_i.end()); })
        .def_property_readonly("default_pose", [](const Env& e) {
            py::array_t<double> out(kNAct);
            std::memcpy(out.mutable_data(), e.index().default_pose.data(), sizeof(double) * kNAct);
            return out;
        })
        .def_property_readonly("nq", [](const Env& e) { return e.model()->nq; })
        .def_property_readonly("nv", [](const Env& e) { return e.model()->nv; })
        .def_property_readonly("nu", [](const Env& e) { return e.model()->nu; });

    py::class_<VecEnv>(m, "VecEnv")
        .def(py::init([](const EnvConfig& cfg, int n, std::uint64_t seed, int threads, bool allow_overcommit) {
            return new VecEnv(cfg, n, seed, threads, allow_overcommit);
        }), py::arg("config"), py::arg("n"), py::arg("seed") = 0, py::arg("threads") = 0,
            py::arg("allow_overcommit") = false)
        .def_property_readonly("num_envs", &VecEnv::num_envs)
        .def_property_readonly("num_threads", &VecEnv::num_threads)
        .def("reset", [](VecEnv& v) {
            v.reset();
            return obs_array(v.obs(), v.num_envs());
        })
        .def("reseed_all", [](VecEnv& v, std::uint64_t seed) {
            for (int i = 0; i < v.num_envs(); ++i) v.env(i).reseed(seed + static_cast<std::uint64_t>(i));
        }, py::arg("seed"))
        .def("step", [](VecEnv& v, py::array_t<double, py::array::c_style | py::array::forcecast> actions, bool with_terms) {
            const int n = v.num_envs();
            if (actions.size() != static_cast<py::ssize_t>(n) * kNAct)
                throw std::invalid_argument("actions must have shape (n, 14)");
            {
                py::gil_scoped_release release;
                v.step(actions.data());
            }
            py::array_t<double> rew(n);
            py::array_t<bool> done(n);
            py::list infos;
            const Reward r = v.env(0).config().reward;
            for (int i = 0; i < n; ++i) {
                rew.mutable_at(i) = v.rewards()[i];
                done.mutable_at(i) = v.dones()[i] != 0;
                py::dict d = info_dict(v.infos()[i], r, with_terms);
                if (v.dones()[i]) d["terminal_obs"] = obs_row(v.terminal_obs() + static_cast<std::size_t>(i) * kObsDim);
                infos.append(d);
            }
            return py::make_tuple(obs_array(v.obs(), n), rew, done, infos);
        }, py::arg("actions"), py::arg("with_terms") = false)
        .def("obs", [](const VecEnv& v) { return obs_array(v.obs(), v.num_envs()); })
        .def("env", [](VecEnv& v, int i) -> Env& {
            if (i < 0 || i >= v.num_envs()) throw std::out_of_range("env index");
            return v.env(i);
        }, py::arg("i"), py::return_value_policy::reference_internal)
        .def("close", &VecEnv::close);
}
