"""The drop-in VecEnv has vec_env.VecEnv's surface and semantics."""
import numpy as np
import pytest


def test_constructor_step_and_autoreset_shape_like_vec_env(cpp, py):
    with cpp.VecEnv(n=3, seed=100, reward="v2") as v:
        o0 = v.reset()
        assert o0.shape == (3, 48) and o0.dtype == np.float32
        a = np.zeros((3, 14))
        seen_terminal = 0
        for t in range(250):
            obs, rew, done, infos = v.step(a)
            assert obs.shape == (3, 48) and rew.shape == (3,) and rew.dtype == np.float64
            assert done.shape == (3,) and done.dtype == bool
            assert len(infos) == 3
            for i in infos:
                assert set(i) >= {"fallen", "upright_cos", "trunk_height", "pushed", "t",
                                  "truncated", "terminated"}
                assert i["t"] == t + 1
                assert i["truncated"] == (t == 249) and not i["terminated"]
            if t < 249:
                assert not done.any()
            else:
                assert done.all()
                for i, info in enumerate(infos):
                    assert "terminal_obs" in info and info["terminal_obs"].shape == (48,)
                    # the returned obs is the NEW episode's first observation
                    assert not np.array_equal(info["terminal_obs"], obs[i])
                    seen_terminal += 1
        assert seen_terminal == 3


def test_with_terms_carries_the_reward_decomposition(cpp, py):
    with cpp.VecEnv(n=2, seed=0, with_terms=True, reward="v1") as v:
        _, rew, _, infos = v.step(np.zeros((2, 14)))
        for r, i in zip(rew, infos):
            assert set(i["terms"]) == set(py.env.REWARD_V1)
            assert sum(i["terms"].values()) == r


def test_thread_count_does_not_change_the_numbers(cpp, py):
    rng = np.random.default_rng(0)
    acts = rng.uniform(-1, 1, (300, 4, 14))
    out = []
    for threads in (1, 4):
        with cpp.VecEnv(n=4, seed=100, threads=threads) as v:
            assert v.num_threads == threads
            tr = [v.step(a)[:3] for a in acts]
            out.append(tr)
    for (o1, r1, d1), (o2, r2, d2) in zip(*out):
        assert np.array_equal(o1, o2) and np.array_equal(r1, r2) and np.array_equal(d1, d2)


def test_reset_with_seed_replays_and_bare_reset_returns_the_current_obs(cpp, py):
    with cpp.VecEnv(n=2, seed=0) as v:
        a = np.random.default_rng(1).uniform(-1, 1, (20, 2, 14))
        o1 = v.reset(seed=5)
        t1 = [v.step(x)[0] for x in a]
        o2 = v.reset(seed=5)
        t2 = [v.step(x)[0] for x in a]
        assert np.array_equal(o1, o2)
        assert all(np.array_equal(p, q) for p, q in zip(t1, t2))
        cur = v.step(a[0])[0]
        assert np.array_equal(v.reset(), cur)


def test_accepts_microduck_rl_domain_randomizer_and_refuses_bad_kwargs(cpp, py):
    with cpp.VecEnv(n=2, seed=0, dr=py.dr.DomainRandomizer(py.dr.DRConfig())) as v:
        e = v.env(0)
        assert e.dr_factors is not None
        assert 0.85 <= e.dr_factors.mass_scale <= 1.15
        assert e.sensor_noise is True
    with pytest.raises(TypeError):
        cpp.VecEnv(n=2, seed=0, no_such_kwarg=1)
    with pytest.raises(ValueError):
        cpp.VecEnv(n=9, seed=0, threads=9)
    with pytest.raises(ValueError):
        cpp.VecEnv(n=2, seed=0, variant="no_such_variant")


def test_single_env_refuses_step_before_reset(cpp):
    e = cpp.Env(cpp.make_config(), 0)
    with pytest.raises(RuntimeError):
        e.step(np.zeros(14))


def test_same_seed_is_not_a_replay_of_the_python_env(cpp, py):
    """Stated in the README: different seeding, different pushes. Both are
    valid samples of the same push distribution."""
    pe = py.env.MicroduckEnv(seed=3)
    pe.reset(seed=3)
    ce = cpp.Env(cpp.make_config(), 3)
    ce.reset()
    assert len(ce.push_at) == 3 and all(25 <= t < 225 for t in ce.push_at)
    assert list(ce.push_at) != list(pe._push_at)
