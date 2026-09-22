"""Threaded C++ Microduck environments, same contract as microduck-rl.

Two layers. `_vecenv_cpp` is the pybind11 module: Env, VecEnv, EnvConfig,
DRConfig, EpisodeParams, PCG64. This file adds the drop-in: a `VecEnv` class
with the constructor and step signature of microduck-rl's `vec_env.VecEnv`,
so `ppo.py` runs on it with one attribute swap (see scripts/ppo_cpp.py).

Build first: cmake configures the module into this directory, see README.
"""
import numpy as np

from ._vecenv_cpp import (  # noqa: F401
    ACT_DIM, ACTION_SCALE, FALL_HEIGHT, MAX_WORKERS, OBS_DIM, STAND_HEIGHT,
    ActuatorClass, DRConfig, DRFactors, Env, EnvConfig, EpisodeParams, PCG64,
    act_param_hi, act_param_lo, check_library_version, default_assets_dir,
    header_version, mujoco_version, obs_scale, term_names,
)
from ._vecenv_cpp import VecEnv as _CVecEnv

_ACT_PARAM_NAMES = ("damping", "frictionloss", "armature", "kp", "forcerange")


def dr_config(obj):
    """A C++ DRConfig from: None/False (nominal), True (defaults), a C++
    DRConfig, microduck-rl's dr.DRConfig dataclass, or its DomainRandomizer
    (which carries one as `.cfg`)."""
    if obj is None or obj is False:
        return None
    if obj is True:
        return DRConfig()
    if isinstance(obj, DRConfig):
        return obj
    if hasattr(obj, "cfg"):
        obj = obj.cfg
    c = DRConfig()
    c.actuator = bool(obj.actuator)
    c.mass_lo, c.mass_hi = (float(v) for v in obj.mass_scale)
    c.friction_lo, c.friction_hi = (float(v) for v in obj.floor_friction)
    c.latency_lo, c.latency_hi = (int(v) for v in obj.latency)
    c.init_noise_lo, c.init_noise_hi = (float(v) for v in obj.init_noise)
    c.sensor_noise = bool(obj.sensor_noise)
    return c


def dr_factors(f):
    """A C++ DRFactors from the dict microduck-rl's DomainRandomizer draws
    (`randomizer.current` after a reset)."""
    if f is None:
        return None
    if isinstance(f, DRFactors):
        return f
    out = DRFactors()
    a = f["actuator"]
    out.actuator = ActuatorClass(*(float(a[k]) for k in _ACT_PARAM_NAMES))
    out.mass_scale = float(f["mass_scale"])
    out.floor_friction = float(f["floor_friction"])
    out.latency = int(f["latency"])
    out.init_noise = float(f["init_noise"])
    out.sensor_noise = bool(f["sensor_noise"])
    return out


def make_config(variant="groundcontact", episode_steps=250, action_scale=ACTION_SCALE,
                n_pushes=3, push_speed=(0.15, 0.45), init_noise=0.02, reward="v1",
                latency=0, sensor_noise=False, dr=None, assets_dir=None):
    """Same keyword arguments as MicroduckEnv.__init__ in microduck-rl."""
    cfg = EnvConfig()
    cfg.variant = str(variant)
    cfg.assets_dir = "" if assets_dir is None else str(assets_dir)
    cfg.episode_steps = int(episode_steps)
    cfg.action_scale = float(action_scale)
    cfg.n_pushes = int(n_pushes)
    cfg.push_speed_lo, cfg.push_speed_hi = (float(v) for v in push_speed)
    cfg.init_noise = float(init_noise)
    cfg.reward = str(reward)
    cfg.latency = int(latency)
    cfg.sensor_noise = bool(sensor_noise)
    d = dr_config(dr)
    cfg.dr = d is not None
    if d is not None:
        cfg.dr_cfg = d
    return cfg


def episode_params(qpos, qvel, push_at, push_vel, dr=None):
    """Hand a C++ Env exactly what a Python MicroduckEnv drew for an episode.

    qpos/qvel: the 14 actuated joint values AFTER the Python reset (post noise,
    post clip). push_at/push_vel: its `_push_at` / `_push_vel`. dr: the
    factors dict its randomiser drew, or None.
    """
    p = EpisodeParams()
    p.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))
    at = [int(v) for v in np.asarray(push_at).ravel()]
    vel = np.asarray(push_vel, dtype=np.float64).reshape(len(at), 2)
    p.set_pushes(at, vel)
    p.dr = dr_factors(dr)
    return p


class VecEnv:
    """Drop-in for microduck-rl `vec_env.VecEnv`: N envs, env i seeded seed+i,
    autoreset with `terminal_obs` in the info dict on every done.

    Differences that are deliberate: `threads` (default one per env, capped at
    8) chooses how many OS threads step the N envs; and the pushes an env
    draws for a given seed are NOT the ones the Python env draws for that
    seed (different seeding, see the README), so a Python run and a C++ run
    with the same seed are two samples of the same distribution, not replays.
    """

    MAX_WORKERS = MAX_WORKERS

    def __init__(self, n=4, seed=0, with_terms=False, threads=None, assets_dir=None,
                 allow_overcommit=False, **env_kwargs):
        self.closed = True
        cfg = make_config(assets_dir=assets_dir, **env_kwargs)
        self._v = _CVecEnv(cfg, int(n), int(seed), 0 if threads is None else int(threads),
                           bool(allow_overcommit))
        self.closed = False
        self.n = int(n)
        self.with_terms = bool(with_terms)
        self._last_obs = self._v.reset()

    @property
    def num_threads(self):
        return self._v.num_threads

    def reset(self, seed=None):
        if seed is None:
            return self._last_obs.copy()
        self._v.reseed_all(int(seed))
        self._last_obs = self._v.reset()
        return self._last_obs.copy()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float64).reshape(self.n, ACT_DIM)
        obs, rew, done, infos = self._v.step(actions, self.with_terms)
        self._last_obs = obs
        return obs.copy(), rew, done, infos

    def env(self, i):
        return self._v.env(i)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._v.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:      # noqa: BLE001
            pass
