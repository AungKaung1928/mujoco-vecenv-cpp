#!/usr/bin/env python3
"""Run every contract case and write the table the README quotes.

    python scripts/contract_report.py            # -> runs/contract.json, prints markdown

Single core, a minute. Nothing here is a throughput number.
"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
MICRODUCK_RL = os.path.abspath(os.environ.get("MICRODUCK_RL", os.path.join(REPO, "..", "microduck-rl")))
os.environ.setdefault("MICRODUCK_ASSETS", os.path.join(MICRODUCK_RL, "assets"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, MICRODUCK_RL)

import vec_env  # noqa: E402,F401  sets OMP before mujoco
import common   # noqa: E402
import dr       # noqa: E402
import env      # noqa: E402
import mujoco   # noqa: E402

from vecenv_cpp import contract, mujoco_version   # noqa: E402

py = types.SimpleNamespace(vec_env=vec_env, common=common, dr=dr, env=env)
rows = []
for kw in contract.CASES:
    for seed in (0, 1, 2):
        rows.append(contract.compare_episode(py, kw, seed))
for kw in contract.DR_CASES:
    for seed in (3, 4):
        rows.append(contract.compare_episode(py, kw, seed, dr=True))

steps = sum(r["obs_compared"] for r in rows)
exact = sum(r["obs_exact_steps"] for r in rows)
worst_obs = max(r["obs_max_abs_diff"] for r in rows)
worst_rew = max(r["reward_max_rel_diff"] for r in rows)
worst_term = max(r["term_max_abs_diff"] for r in rows)
mismatch = sum(len(r["info_mismatches"]) for r in rows)

print(f"| episodes | observations compared | bit-identical | max abs diff (f32) | max rel reward diff | max abs term diff | info mismatches |")
print(f"|---|---|---|---|---|---|---|")
print(f"| {len(rows)} | {steps:,} | {exact:,} ({100*exact/steps:.2f}%) | {worst_obs:.1e} | {worst_rew:.1e} | {worst_term:.1e} | {mismatch} |")
print()
print("| case | seed | pushes | latency | init_noise | obs exact | max abs obs diff | max rel reward diff |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    case = ", ".join(f"{k}={v}" for k, v in r["kw"].items()) + (", DR" if r["dr"] else "")
    print(f"| {case} | {r['seed']} | {r['pushes']} | {r['latency']} | {r['init_noise']:.3f} | "
          f"{r['obs_exact_steps']}/{r['obs_compared']} | {r['obs_max_abs_diff']:.1e} | {r['reward_max_rel_diff']:.1e} |")

os.makedirs(os.path.join(REPO, "runs"), exist_ok=True)
out = os.path.join(REPO, "runs", "contract.json")
with open(out, "w") as f:
    json.dump({"mujoco": mujoco.__version__, "mujoco_lib": mujoco_version(),
               "episodes": len(rows), "observations": steps, "bit_identical": exact,
               "obs_max_abs_diff": worst_obs, "reward_max_rel_diff": worst_rew,
               "term_max_abs_diff": worst_term, "info_mismatches": mismatch, "rows": rows}, f, indent=1)
print(f"\nwrote {out}")
