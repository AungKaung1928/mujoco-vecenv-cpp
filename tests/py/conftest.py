"""Test wiring. Needs microduck-rl next to this repo (or $MICRODUCK_RL) with
its assets fetched, and the built module in python/vecenv_cpp/."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MICRODUCK_RL = os.path.abspath(os.environ.get("MICRODUCK_RL", os.path.join(REPO, "..", "microduck-rl")))
ASSETS = os.environ.setdefault("MICRODUCK_ASSETS", os.path.join(MICRODUCK_RL, "assets"))

sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, MICRODUCK_RL)


@pytest.fixture(scope="session")
def py():
    """microduck-rl's modules, imported in the order that sets OMP first."""
    if not os.path.exists(os.path.join(ASSETS, "scene.xml")):
        pytest.skip(f"no assets at {ASSETS}: run microduck-rl/fetch_assets.sh")
    import vec_env    # noqa: F401  sets OMP_NUM_THREADS before mujoco
    import common
    import dr
    import env
    import types
    return types.SimpleNamespace(vec_env=vec_env, common=common, dr=dr, env=env)


@pytest.fixture(scope="session")
def cpp():
    try:
        import vecenv_cpp
    except ImportError as exc:      # pragma: no cover
        pytest.skip(f"vecenv_cpp not built: {exc}")
    if not os.path.exists(os.path.join(ASSETS, "scene.xml")):
        pytest.skip(f"no assets at {ASSETS}: run microduck-rl/fetch_assets.sh")
    return vecenv_cpp
