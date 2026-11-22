"""make_config / dr_config carry every keyword through, read back from the C++ structs.

Every other test runs with default values, where a dropped keyword falls back to
a C++ default that happens to be the same number, and the contract injects pushes
directly. Non-default values here make a dropped or swapped field visible.
"""
import pytest


def test_make_config_keeps_every_env_keyword(cpp):
    c = cpp.make_config(variant="rollers", episode_steps=37, action_scale=0.5, n_pushes=5,
                        push_speed=(0.1, 0.6), init_noise=0.07, reward="v2", latency=2,
                        sensor_noise=True)
    assert (c.variant, c.episode_steps, c.n_pushes, c.reward, c.latency) == ("rollers", 37, 5, "v2", 2)
    assert (c.action_scale, c.push_speed_lo, c.push_speed_hi, c.init_noise) == (0.5, 0.1, 0.6, 0.07)
    assert c.sensor_noise is True and c.dr is False


def test_dr_config_translates_microduck_ranges(cpp, py):
    src = py.dr.DRConfig(actuator=False, mass_scale=(0.7, 1.3), floor_friction=(0.4, 1.0),
                         latency=(1, 3), init_noise=(0.05, 0.2), sensor_noise=False)
    for obj in (src, py.dr.DomainRandomizer(src)):          # dataclass or the randomiser holding it
        d = cpp.make_config(dr=obj).dr_cfg
        assert d.actuator is False and d.sensor_noise is False
        assert (d.mass_lo, d.mass_hi, d.friction_lo, d.friction_hi) == (0.7, 1.3, 0.4, 1.0)
        # microduck's latency range is inclusive (integers(lo, hi + 1)); the C++ draw is inclusive too
        assert (d.latency_lo, d.latency_hi) == (1, 3)
        assert (d.init_noise_lo, d.init_noise_hi) == (0.05, 0.2)


def test_episode_steps_truncates_the_episode(cpp):
    env = cpp.VecEnv(n=1, seed=0, episode_steps=5)
    try:
        dones = [bool(env.step([[0.0] * 14])[2][0]) for _ in range(5)]
    finally:
        env.close()
    assert dones == [False] * 4 + [True]
