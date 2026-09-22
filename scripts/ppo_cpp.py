#!/usr/bin/env python3
"""Run microduck-rl's ppo.py on the C++ vector env. Zero changes to microduck-rl.

ppo.py does `import vec_env` and later `vec_env.VecEnv(n=..., seed=..., **kw)`.
This script puts vecenv_cpp under that module name first, then runs ppo.py as
__main__ with the same command line. Everything else -- PPO, normaliser,
checkpoints, evaluation, throughput logging -- is ppo.py's own code.

    OMP_NUM_THREADS=1 nice -n 10 python scripts/ppo_cpp.py --total-steps 50000000 \\
        --chunk-steps 25000000 --tag v2cpp --reward v2

The box check, the chunking and the resume flags are ppo.py's. The one
knob that is new lives in the environment variable VECENV_THREADS (default:
one thread per env, capped at 8).
"""
import os
import runpy
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
MICRODUCK_RL = os.path.abspath(os.environ.get("MICRODUCK_RL", os.path.join(REPO, "..", "microduck-rl")))
os.environ.setdefault("MICRODUCK_ASSETS", os.path.join(MICRODUCK_RL, "assets"))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, MICRODUCK_RL)

import vecenv_cpp   # noqa: E402

threads = os.environ.get("VECENV_THREADS")
if threads:
    _Base = vecenv_cpp.VecEnv

    class VecEnv(_Base):
        def __init__(self, *a, **kw):
            kw.setdefault("threads", int(threads))
            super().__init__(*a, **kw)

    vecenv_cpp.VecEnv = VecEnv

sys.modules["vec_env"] = vecenv_cpp
print(f"  backend: vecenv_cpp (C++ threads), MuJoCo {vecenv_cpp.mujoco_version()}")
runpy.run_path(os.path.join(MICRODUCK_RL, "ppo.py"), run_name="__main__")
