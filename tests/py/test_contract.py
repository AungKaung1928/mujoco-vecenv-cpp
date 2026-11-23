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


def test_a_nan_from_cpp_is_not_a_match(py, cpp, monkeypatch):
    """A NaN must fail the tolerances: builtin max(running, nan) keeps the running value."""
    import numpy as np
    real_env = contract.Env

    class NanAt10:
        def __init__(self, *a):
            self.e, self.t = real_env(*a), 0

        def reset(self, *a):
            return self.e.reset(*a)

        def step(self, a, terms):
            o, r, d, i = self.e.step(a, terms)
            self.t += 1
            if self.t == 10:
                o, r = np.full_like(o, np.nan), float("nan")
                i["terms"] = {k: float("nan") for k in i["terms"]}
            return o, r, d, i

    monkeypatch.setattr(contract, "Env", NanAt10)
    r = contract.compare_episode(py, dict(variant="groundcontact", reward="v2"), 0, steps=20)
    for k in ("obs_max_abs_diff", "reward_max_rel_diff", "term_max_abs_diff"):
        assert not r[k] <= 1e-6, (k, r[k])


def _perturbed(py, monkeypatch, at_reset=False, **change):
    """compare_episode with the C++ env wrapped: at step 10 (or at reset) one output
    is changed by a known finite amount. Every field the report claims must see it."""
    import numpy as np
    real_env = contract.Env

    class Perturb:
        def __init__(self, *a):
            self.e, self.t = real_env(*a), 0

        def reset(self, *a):
            o = self.e.reset(*a)
            if at_reset:
                o = o.copy()
                o[0] += np.float32(1e-3)
            return o

        def step(self, a, terms):
            o, r, d, i = self.e.step(a, terms)
            self.t += 1
            if self.t == 10:
                if "obs" in change:
                    o = o.copy()
                    o[3] += np.float32(change["obs"])
                if "reward" in change:
                    r = r * (1 + change["reward"])
                if "info" in change:
                    i[change["info"]] = not i[change["info"]]
                if "done" in change:
                    d = not d
                if "height" in change:
                    i["trunk_height"] += change["height"]
            return o, r, d, i

    monkeypatch.setattr(contract, "Env", Perturb)
    return contract.compare_episode(py, dict(variant="groundcontact", reward="v2"), 0, steps=20)


def test_report_counts_a_finite_obs_difference(py, cpp, monkeypatch):
    r = _perturbed(py, monkeypatch, obs=1e-3)
    assert r["obs_exact_steps"] == r["obs_compared"] - 1 and r["obs_max_abs_diff"] > 5e-4


def test_report_counts_a_reset_obs_difference(py, cpp, monkeypatch):
    r = _perturbed(py, monkeypatch, at_reset=True)
    assert r["obs_exact_steps"] == r["obs_compared"] - 1


def test_report_sees_a_one_percent_reward_difference(py, cpp, monkeypatch):
    r = _perturbed(py, monkeypatch, reward=0.01)
    assert abs(r["reward_max_rel_diff"] - 0.01) < 1e-6


@pytest.mark.parametrize("key", ["pushed", "fallen", "truncated", "terminated"])
def test_report_lists_an_info_flag_mismatch(py, cpp, monkeypatch, key):
    r = _perturbed(py, monkeypatch, info=key)
    assert [m[:2] for m in r["info_mismatches"]] == [(9, key)]


def test_report_lists_a_done_mismatch(py, cpp, monkeypatch):
    r = _perturbed(py, monkeypatch, done=True)
    assert [m[:2] for m in r["info_mismatches"]] == [(9, "done")]


def test_report_measures_trunk_height(py, cpp, monkeypatch):
    r = _perturbed(py, monkeypatch, height=1e-3)
    assert abs(r["trunk_height_max_abs_diff"] - 1e-3) < 1e-9
