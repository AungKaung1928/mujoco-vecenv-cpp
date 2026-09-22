def test_header_library_and_python_wheel_are_one_version(cpp):
    import mujoco
    major, minor, patch = (int(v) for v in mujoco.__version__.split(".")[:3])
    assert cpp.mujoco_version() == cpp.header_version()
    assert cpp.header_version() == major * 1_000_000 + minor * 1_000 + patch
    cpp.check_library_version()


def test_constants_match_the_python_env(cpp, py):
    assert cpp.OBS_DIM == py.env.OBS_DIM == 48
    assert cpp.ACT_DIM == py.env.ACT_DIM == 14
    assert cpp.MAX_WORKERS == py.vec_env.MAX_WORKERS == 8
    assert cpp.ACTION_SCALE == py.env.ACTION_SCALE
    assert cpp.FALL_HEIGHT == py.env.FALL_HEIGHT
    assert cpp.STAND_HEIGHT == py.env.STAND_HEIGHT
    import numpy as np
    assert np.array_equal(cpp.obs_scale(), py.env.OBS_SCALE.astype(np.float64))
    assert cpp.term_names("v1") == list(py.env.REWARD_V1)
    assert cpp.term_names("v2") == list(py.env.REWARD_V2)
    assert np.allclose(cpp.act_param_lo(), py.dr.ACT_PARAM_LO)
    assert np.allclose(cpp.act_param_hi(), py.dr.ACT_PARAM_HI)
