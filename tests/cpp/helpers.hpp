// Shared by the C++ tests: one loaded prototype per variant, a config that
// points at the assets, random actions.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vecenv/env.hpp"
#include "vecenv/model.hpp"
#include "vecenv/rng.hpp"

namespace th {

inline const mjModel* proto(const std::string& variant = "groundcontact") {
    static std::map<std::string, vecenv::ModelPtr> cache;
    auto it = cache.find(variant);
    if (it == cache.end()) {
        vecenv::check_library_version();
        it = cache.emplace(variant, vecenv::load_variant(vecenv::default_assets_dir(), variant)).first;
    }
    return it->second.get();
}

inline vecenv::EnvConfig cfg(const std::string& variant = "groundcontact") {
    vecenv::EnvConfig c;
    c.variant = variant;
    c.assets_dir = vecenv::default_assets_dir();
    return c;
}

inline std::vector<std::array<double, vecenv::kNAct>> actions(int n, std::uint64_t seed = 0) {
    vecenv::PCG64 r(seed);
    std::vector<std::array<double, vecenv::kNAct>> out(static_cast<std::size_t>(n));
    for (auto& a : out) for (auto& v : a) v = r.uniform(-1.0, 1.0);
    return out;
}

struct Rollout {
    std::vector<std::array<float, vecenv::kObsDim>> obs;
    std::vector<double> rew;
    std::vector<bool> done;
};

inline Rollout rollout(vecenv::Env& e, const std::vector<std::array<double, vecenv::kNAct>>& acts, int n) {
    Rollout r;
    for (int k = 0; k < n; ++k) {
        std::array<float, vecenv::kObsDim> o{};
        vecenv::StepInfo info;
        e.step(acts[static_cast<std::size_t>(k)].data(), o.data(), &info);
        r.obs.push_back(o);
        r.rew.push_back(info.reward);
        r.done.push_back(e.done());
    }
    return r;
}

}  // namespace th
