"""The equality test between the C++ Env and microduck-rl's MicroduckEnv.

One episode: the Python env is reset, what it drew (initial joint state,
push schedule, DR factors) is handed to a C++ Env, and both are stepped with
the same actions. Everything that comes back is compared: the 48-dim float32
observation, the float64 reward and every logged term, and the info fields.

Used by tests/py/test_contract.py (asserts) and scripts/contract_report.py
(writes the table the README quotes).
"""
import numpy as np

from . import Env, episode_params, make_config


def compare_episode(py, kw, seed, steps=250, dr=False, action_seed=None):
    """`py` is a namespace with microduck-rl's env/common/dr modules.
    `kw` are MicroduckEnv keyword arguments (variant, reward, latency, ...).
    Returns a dict of measured differences; nothing is asserted here."""
    pykw = dict(kw)
    if dr:
        pykw["dr"] = py.dr.DomainRandomizer(py.dr.DRConfig(sensor_noise=False))
    pe = py.env.MicroduckEnv(seed=seed, **pykw)
    o_py = pe.reset(seed=seed)
    qi, vi = py.common.actuated_qpos_index(pe.model), py.common.actuated_qvel_index(pe.model)
    params = episode_params(pe.data.qpos[qi], pe.data.qvel[vi], pe._push_at, pe._push_vel,
                            dr=pe.dr.current if dr else None)
    ce = Env(make_config(**kw), seed)
    o_cpp = ce.reset(params)

    rng = np.random.default_rng(seed if action_seed is None else action_seed)
    obs_max = float(np.abs(o_py.astype(np.float64) - o_cpp.astype(np.float64)).max())
    obs_exact = int(np.array_equal(o_py, o_cpp))
    rew_rel = 0.0
    term_abs = 0.0
    info_mismatch = []
    height_abs = cos_abs = 0.0
    done_py = done_cpp = None
    for t in range(steps):
        a = rng.uniform(-1.0, 1.0, 14)
        o1, r1, d1, i1 = pe.step(a)
        o2, r2, d2, i2 = ce.step(a, True)
        obs_max = max(obs_max, float(np.abs(o1.astype(np.float64) - o2.astype(np.float64)).max()))
        obs_exact += int(np.array_equal(o1, o2))
        rew_rel = max(rew_rel, abs(r1 - r2) / max(abs(r1), 1e-12))
        for k, v in i1["terms"].items():
            term_abs = max(term_abs, abs(v - i2["terms"][k]))
        for k in ("pushed", "fallen", "t", "truncated", "terminated"):
            if i1[k] != i2[k]:
                info_mismatch.append((t, k, i1[k], i2[k]))
        height_abs = max(height_abs, abs(i1["trunk_height"] - i2["trunk_height"]))
        cos_abs = max(cos_abs, abs(i1["upright_cos"] - i2["upright_cos"]))
        if d1 != d2:
            info_mismatch.append((t, "done", d1, d2))
        done_py, done_cpp = d1, d2
    return {
        "kw": {k: str(v) for k, v in kw.items()}, "dr": dr, "seed": seed, "steps": steps,
        "obs_max_abs_diff": obs_max, "obs_exact_steps": obs_exact, "obs_compared": steps + 1,
        "reward_max_rel_diff": rew_rel, "term_max_abs_diff": term_abs,
        "trunk_height_max_abs_diff": height_abs, "upright_cos_max_abs_diff": cos_abs,
        "info_mismatches": info_mismatch, "done_py": bool(done_py), "done_cpp": bool(done_cpp),
        "pushes": [int(v) for v in pe._push_at], "latency": int(pe.latency),
        "init_noise": float(pe.init_noise),
    }


CASES = [
    dict(variant="groundcontact", reward="v1"),
    dict(variant="groundcontact", reward="v2"),
    dict(variant="groundcontact", reward="v2", latency=2),
    dict(variant="groundcontact", reward="v2", init_noise=0.0),
    dict(variant="groundcontact", reward="v1", action_scale=0.5, n_pushes=5),
    dict(variant="groundcontact_backlash", reward="v2"),
    dict(variant="rollers", reward="v1"),
]
DR_CASES = [
    dict(variant="groundcontact", reward="v2"),
    dict(variant="groundcontact_backlash", reward="v1"),
]
