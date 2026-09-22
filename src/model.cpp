#include "vecenv/model.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>

#ifndef VECENV_SOURCE_DIR
#define VECENV_SOURCE_DIR "."
#endif

namespace vecenv {

// Order fixed by the <actuator> block in robot_walk.xml; index_model() checks
// the loaded model agrees, so an upstream reordering fails loudly instead of
// permuting every action a trained policy emits.
const std::array<const char*, kNAct> kActuatorNames = {
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
};

//                                              damping  frictionloss armature  kp     forcerange
const std::array<std::pair<const char*, ActuatorClass>, 4> kActuatorClasses = {{
    {"chosen_actuator",         {0.053, 0.0048, 0.0018, 0.55,  0.96}},
    {"chosen_actuator_old",     {0.048, 0.006,  0.002,  0.52,  0.91}},
    {"chosen_actuator_new",     {0.041, 0.032,  0.002,  0.386, 0.67}},
    {"chosen_actuator_antoine", {0.044, 0.013,  0.0017, 0.43,  0.75}},
}};

std::string variant_file(const std::string& variant) {
    static const std::map<std::string, std::string> table = {
        {"walk", "scene_walk.xml"},
        {"walk_backlash", "scene_walk_backlash.xml"},
        {"rollers", "scene_rollers.xml"},
        {"groundcontact", "scene.xml"},
        {"groundcontact_backlash", "scene_backlash.xml"},
    };
    const auto it = table.find(variant);
    if (it == table.end()) {
        std::string have;
        for (const auto& kv : table) have += (have.empty() ? "" : ", ") + kv.first;
        throw std::invalid_argument("unknown variant '" + variant + "', have: " + have);
    }
    return it->second;
}

std::string default_assets_dir() {
    if (const char* e = std::getenv("MICRODUCK_ASSETS"); e && *e) return e;
    return std::string(VECENV_SOURCE_DIR) + "/../microduck-rl/assets";
}

void check_library_version() {
    const int lib = mj_version();
    if (lib != mjVERSION_HEADER) {
        throw std::runtime_error(
            "libmujoco version " + std::to_string(lib) + " does not match the header this was "
            "compiled against (" + std::to_string(mjVERSION_HEADER) + "). Build against the "
            "mujoco wheel the Python side imports; the equality test means nothing otherwise.");
    }
}

ModelPtr load_variant(const std::string& assets_dir, const std::string& variant) {
    const std::string path = assets_dir + "/" + variant_file(variant);
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            path + " is missing. assets/ is gitignored in microduck-rl (the meshes are "
            "CC BY-SA-NC); run microduck-rl/fetch_assets.sh, or point MICRODUCK_ASSETS at "
            "a directory that has scene.xml.");
    }
    char err[1000] = {0};
    mjModel* m = mj_loadXML(path.c_str(), nullptr, err, sizeof err);
    if (!m) throw std::runtime_error("mj_loadXML(" + path + "): " + err);
    return ModelPtr(m);
}

ModelPtr copy_model(const mjModel* src) {
    mjModel* m = mj_copyModel(nullptr, src);
    if (!m) throw std::runtime_error("mj_copyModel failed");
    return ModelPtr(m);
}

DataPtr make_data(const mjModel* m) {
    mjData* d = mj_makeData(m);
    if (!d) throw std::runtime_error("mj_makeData failed");
    return DataPtr(d);
}

namespace {
int require_id(const mjModel* m, mjtObj type, const char* name, const char* what) {
    const int id = mj_name2id(m, type, name);
    if (id < 0) throw std::runtime_error(std::string("no ") + what + " named '" + name + "' in this variant");
    return id;
}
}  // namespace

ModelIndex index_model(const mjModel* m) {
    ModelIndex ix;
    if (m->nu != kNAct) throw std::runtime_error("model has " + std::to_string(m->nu) + " actuators, expected 14");
    for (int i = 0; i < kNAct; ++i) {
        const char* actual = mj_id2name(m, mjOBJ_ACTUATOR, i);
        if (!actual || std::strcmp(actual, kActuatorNames[i]) != 0) {
            throw std::runtime_error(std::string("actuator ") + std::to_string(i) + " is '" +
                                     (actual ? actual : "<unnamed>") + "', expected '" + kActuatorNames[i] +
                                     "': upstream reordered the action space");
        }
        const int j = require_id(m, mjOBJ_JOINT, kActuatorNames[i], "joint");
        ix.qpos_i[i] = m->jnt_qposadr[j];
        ix.qvel_i[i] = m->jnt_dofadr[j];
        ix.lo[i] = m->jnt_range[2 * j];
        ix.hi[i] = m->jnt_range[2 * j + 1];
    }
    ix.stand_key = require_id(m, mjOBJ_KEY, "STAND", "keyframe");
    for (int i = 0; i < kNAct; ++i) ix.default_pose[i] = m->key_qpos[ix.stand_key * m->nq + ix.qpos_i[i]];

    const int g = require_id(m, mjOBJ_SENSOR, "angular-velocity", "sensor");
    const int q = require_id(m, mjOBJ_SENSOR, "orientation", "sensor");
    if (m->sensor_dim[g] != 3 || m->sensor_dim[q] != 4) throw std::runtime_error("unexpected sensor dims");
    ix.gyro_adr = m->sensor_adr[g];
    ix.quat_adr = m->sensor_adr[q];
    ix.gyro_sd = m->sensor_noise[g];
    ix.quat_sd = m->sensor_noise[q];
    ix.trunk_body = require_id(m, mjOBJ_BODY, "trunk_base", "body");
    ix.floor_geom = require_id(m, mjOBJ_GEOM, "floor", "geom");
    if (m->jnt_type[0] != mjJNT_FREE) throw std::runtime_error("joint 0 is not the free joint; the push code assumes it is");
    return ix;
}

double trunk_height(const mjModel*, const mjData* d, const ModelIndex& ix) {
    return d->xpos[3 * ix.trunk_body + 2];
}

double upright_cos(const mjModel*, const mjData* d, const ModelIndex& ix) {
    return d->xmat[9 * ix.trunk_body + 8];
}

}  // namespace vecenv
