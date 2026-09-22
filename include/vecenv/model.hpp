// Model loading and the name -> index resolution the environment depends on.
//
// Everything that reads joint state goes through ModelIndex rather than
// assuming qpos[7:21]. On the backlash variants a passive joint sits in
// series with every actuated one, so the actuated joints are at qpos
// 7, 9, 11 ... 33 and the contiguous slice would read a mixture of joint
// angles and backlash deflections with no error. This mirrors
// common.actuated_qpos_index in microduck-rl.
#pragma once

#include "vecenv/mujoco.hpp"

#include <array>
#include <memory>
#include <string>
#include <utility>

namespace vecenv {

constexpr int kNAct = 14;
constexpr int kObsDim = 48;
constexpr int kSubsteps = 10;          // 500 Hz physics, 50 Hz control
constexpr int kControlHz = 50;
constexpr double kActionScale = 0.35;
constexpr double kFallHeight = 0.04;   // m
constexpr double kStandHeight = 0.12;  // m, asserted at construction

extern const std::array<const char*, kNAct> kActuatorNames;

struct ActuatorClass {
    double damping, frictionloss, armature, kp, forcerange;
};
// The four system-identification fits of the same XL330 servo shipped in
// joints_properties.xml. The DR range is their min..max per parameter.
extern const std::array<std::pair<const char*, ActuatorClass>, 4> kActuatorClasses;

struct ModelDeleter { void operator()(mjModel* m) const { if (m) mj_deleteModel(m); } };
struct DataDeleter  { void operator()(mjData* d) const { if (d) mj_deleteData(d); } };
using ModelPtr = std::unique_ptr<mjModel, ModelDeleter>;
using DataPtr  = std::unique_ptr<mjData, DataDeleter>;

// variant name -> scene file, same table as microduck-rl/common.py.
std::string variant_file(const std::string& variant);
// $MICRODUCK_ASSETS, else <this repo>/../microduck-rl/assets.
std::string default_assets_dir();
// Throws if the header this was compiled against and the loaded library
// disagree. They must be the SAME libmujoco.so the Python side imports, or
// the equality test has no meaning.
void check_library_version();
ModelPtr load_variant(const std::string& assets_dir, const std::string& variant);
ModelPtr copy_model(const mjModel* src);
DataPtr make_data(const mjModel* m);

struct ModelIndex {
    std::array<int, kNAct> qpos_i{};
    std::array<int, kNAct> qvel_i{};     // also the dof addresses DR writes to
    std::array<double, kNAct> lo{}, hi{};
    std::array<double, kNAct> default_pose{};   // STAND keyframe at the actuated joints
    int gyro_adr = -1;    // sensor "angular-velocity", 3 values
    int quat_adr = -1;    // sensor "orientation", 4 values
    double gyro_sd = 0.0, quat_sd = 0.0;   // model.sensor_noise of those two
    int trunk_body = -1;
    int floor_geom = -1;
    int stand_key = -1;
};
ModelIndex index_model(const mjModel* m);

double trunk_height(const mjModel* m, const mjData* d, const ModelIndex& ix);
double upright_cos(const mjModel* m, const mjData* d, const ModelIndex& ix);

}  // namespace vecenv
