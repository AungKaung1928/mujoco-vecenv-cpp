"""C++ Env == Python MicroduckEnv, to the bit where the arithmetic allows it.

Tolerances: observations are float32; 1e-6 absolute is about 16 ulps at
magnitude 1 and the expectation is exact equality, which the report script
counts. Rewards are float64 and compared at 1e-9 relative."""
import pytest

from vecenv_cpp import contract


def _run(py, kw, seed, dr):
    r = contract.compare_episode(py, kw, seed, dr=dr)
    assert r["obs_max_abs_diff"] <= 1e-6, r
    assert r["reward_max_rel_diff"] <= 1e-9, r
    assert r["term_max_abs_diff"] <= 1e-12, r
    assert r["trunk_height_max_abs_diff"] <= 1e-12, r
    assert r["upright_cos_max_abs_diff"] <= 1e-12, r
    assert r["info_mismatches"] == [], r
    assert r["done_py"] and r["done_cpp"]
    return r


@pytest.mark.parametrize("kw", contract.CASES, ids=lambda k: "-".join(f"{a}={b}" for a, b in k.items()))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_episode_matches_python(py, cpp, kw, seed):
    _run(py, kw, seed, dr=False)


@pytest.mark.parametrize("kw", contract.DR_CASES, ids=lambda k: "-".join(f"{a}={b}" for a, b in k.items()))
@pytest.mark.parametrize("seed", [3, 4])
def test_episode_matches_python_under_domain_randomisation(py, cpp, kw, seed):
    r = _run(py, kw, seed, dr=True)
    assert r["latency"] in (0, 1, 2)
    assert r["init_noise"] > 0.02 or r["init_noise"] == pytest.approx(0.02)


def test_a_wrong_push_is_detected(py, cpp):
    """The comparison is sensitive: shift one push by a step and it fails."""
    import numpy as np
    from vecenv_cpp import Env, episode_params, make_config
    kw = dict(variant="groundcontact", reward="v2")
    pe = py.env.MicroduckEnv(seed=5, **kw)
    pe.reset(seed=5)
    qi, vi = py.common.actuated_qpos_index(pe.model), py.common.actuated_qvel_index(pe.model)
    at = pe._push_at.copy()
    at[0] += 1
    ce = Env(make_config(**kw), 5)
    ce.reset(episode_params(pe.data.qpos[qi], pe.data.qvel[vi], at, pe._push_vel))
    rng = np.random.default_rng(5)
    diverged = False
    for _ in range(int(at[0]) + 3):
        a = rng.uniform(-1, 1, 14)
        o1 = pe.step(a)[0]
        o2 = ce.step(a)[0]
        diverged |= not np.array_equal(o1, o2)
    assert diverged
