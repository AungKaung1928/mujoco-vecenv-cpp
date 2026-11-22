"""After reset, the C++ model holds the same numbers as microduck-rl's, field by field.

test_contract compares what the two envs emit (observations, rewards) over 250 steps.
A parameter that randomisation writes but an episode barely exercises (a force limit
never reached, a small frictionloss) can differ there unseen. Here every field that
microduck-rl's DomainRandomizer writes (dr.py: damping, frictionloss, armature, kp
gain/bias, force range, mass, inertia, floor friction) is read back through
Env.model_array and must be bit-identical, plus the reset state.
"""
import numpy as np
import pytest

from vecenv_cpp.contract import episode_params, make_config

MODEL_FIELDS = ("dof_damping", "dof_frictionloss", "dof_armature", "actuator_gainprm", "actuator_biasprm",
                "actuator_forcerange", "body_mass", "body_inertia", "geom_friction")
DATA_FIELDS = ("qpos", "qvel", "ctrl")
CASES = [dict(variant="groundcontact", reward="v2"), dict(variant="groundcontact_backlash", reward="v2"),
         dict(variant="rollers", reward="v1")]


def _pair(py, cpp, kw, seed, dr):
    pykw = dict(kw)
    if dr:
        pykw["dr"] = py.dr.DomainRandomizer(py.dr.DRConfig(sensor_noise=False))
    pe = py.env.MicroduckEnv(seed=seed, **pykw)
    pe.reset(seed=seed)
    qi, vi = py.common.actuated_qpos_index(pe.model), py.common.actuated_qvel_index(pe.model)
    ce = cpp.Env(make_config(**kw), seed)
    ce.reset(episode_params(pe.data.qpos[qi], pe.data.qvel[vi], pe._push_at, pe._push_vel,
                            dr=pe.dr.current if dr else None))
    return pe, ce


@pytest.mark.parametrize("dr", [False, True], ids=["nominal", "dr"])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("kw", CASES, ids=[c["variant"] for c in CASES])
def test_model_fields_match_python(py, cpp, kw, seed, dr):
    pe, ce = _pair(py, cpp, kw, seed, dr)
    for f in MODEL_FIELDS:
        got = ce.model_array(f)
        want = np.asarray(getattr(pe.model, f)).reshape(got.shape)
        assert np.array_equal(got, want), (f, float(np.abs(got - want).max()))
    for f in DATA_FIELDS:
        got, want = ce.model_array(f), np.asarray(getattr(pe.data, f))
        assert got.shape == want.shape and np.array_equal(got, want), f


def test_randomisation_moves_every_field_group(py, cpp):
    # without this, a DR path that wrote nothing on either side would still pass above
    kw = CASES[0]
    _, nominal = _pair(py, cpp, kw, 0, False)
    _, rand = _pair(py, cpp, kw, 0, True)
    for f in MODEL_FIELDS:
        assert not np.array_equal(nominal.model_array(f), rand.model_array(f)), f


def test_unknown_field_is_an_error(py, cpp):
    _, ce = _pair(py, cpp, CASES[0], 0, False)
    with pytest.raises(ValueError):
        ce.model_array("body_massx")
