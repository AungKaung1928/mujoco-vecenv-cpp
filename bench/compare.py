#!/usr/bin/env python3
"""Python processes vs C++ threads, same hour, same box, same protocol.

Alternates the two backends round by round so a machine that changes state
during the run (the only thing WSL lets you see) biases both the same way.
Each measurement is a timed window of N environments driven by uniform
random actions, autoreset on, exactly what the C++ bench's `env` mode does.

    nice -n 10 python bench/compare.py --workers 1 2 4 8 --seconds 20 --rounds 2 --tag main

Writes runs/compare_<tag>.json and prints a markdown table. Refuses to start
on a busy box, like every other benchmark in this pair of repos.
"""
import argparse
import datetime
import json
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
MICRODUCK_RL = os.path.abspath(os.environ.get("MICRODUCK_RL", os.path.join(REPO, "..", "microduck-rl")))
os.environ.setdefault("MICRODUCK_ASSETS", os.path.join(MICRODUCK_RL, "assets"))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, MICRODUCK_RL)

import numpy as np          # noqa: E402
import vec_env              # noqa: E402  Python backend, sets OMP first
import bench as pybench     # noqa: E402  box_is_busy, reference_trend
import vecenv_cpp           # noqa: E402


def window(make, n, seconds, seed):
    """env-steps/s of `make(n)` over one timed window, warm-up excluded."""
    v = make(n)
    rng = np.random.default_rng(seed)
    try:
        for _ in range(100):
            v.step(rng.uniform(-1, 1, (n, 14)))
        steps = 0
        t0 = time.perf_counter()
        while True:
            v.step(rng.uniform(-1, 1, (n, 14)))
            steps += 1
            if (steps & 0xF) == 0 and time.perf_counter() - t0 > seconds:
                break
        return steps * n / (time.perf_counter() - t0)
    finally:
        v.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--reward", default="v2")
    ap.add_argument("--variant", default="groundcontact")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if max(a.workers) > 8:
        raise SystemExit("refusing >8 workers: the thread budget on this machine is 8 of 14 cores")

    warnings = pybench.box_is_busy()
    for w in warnings:
        print(f"  WARNING  {w} -- numbers below are contention, not capability")

    kw = dict(reward=a.reward, variant=a.variant)
    backends = {
        "python (processes)": lambda n: vec_env.VecEnv(n=n, seed=1234, **kw),
        "cpp (threads)": lambda n: vecenv_cpp.VecEnv(n=n, seed=1234, threads=n, **kw),
    }
    rates = {b: {n: [] for n in a.workers} for b in backends}
    refs = []
    print(f"\n{a.seconds:.0f} s windows, {a.rounds} round(s), backends alternate inside each round\n")
    for r in range(a.rounds):
        for n in a.workers:
            for b, make in backends.items():
                rate = window(make, n, a.seconds, seed=100 * r + n)
                rates[b][n].append(rate)
                print(f"  round {r}  {b:<20} {n} workers  {rate:10,.0f} env-steps/s")
                sys.stdout.flush()
            # a 1-worker Python reference after every configuration, the drift detector
            refs.append(window(backends["python (processes)"], 1, max(3.0, a.seconds / 3), seed=999 + r))

    print("\n| workers | python processes | cpp threads | cpp / python | cpp efficiency |")
    print("|---|---|---|---|---|")
    rows = []
    py1 = np.mean(rates["python (processes)"][a.workers[0]]) / a.workers[0]
    cpp1 = np.mean(rates["cpp (threads)"][a.workers[0]]) / a.workers[0]
    for n in a.workers:
        p, c = float(np.mean(rates["python (processes)"][n])), float(np.mean(rates["cpp (threads)"][n]))
        eff = c / (cpp1 * n)
        rows.append({"workers": n, "python": p, "cpp": c, "ratio": c / p, "cpp_efficiency": eff,
                     "python_efficiency": p / (py1 * n)})
        print(f"| {n} | {p:,.0f} | {c:,.0f} | {c/p:.2f}x | {100*eff:.0f}% |")
    run_len, span, direction = pybench.reference_trend(refs)
    worst = max(abs(x / refs[0] - 1) for x in refs)
    trending = run_len >= 3 and span > 0.04
    verdict = "UNSTABLE" if worst > 0.10 else "TRENDING" if trending else "stable"
    print(f"\n  1-process Python reference across the run: {', '.join(f'{x:,.0f}' for x in refs)} -> {verdict} "
          f"(worst drift {100*worst:+.1f}%, longest {direction or 'flat'} run {run_len})")
    os.makedirs(os.path.join(REPO, "runs"), exist_ok=True)
    out = os.path.join(REPO, "runs", f"compare{('_' + a.tag) if a.tag else ''}.json")
    with open(out, "w") as f:
        json.dump({"timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                   "seconds": a.seconds, "rounds": a.rounds, "reward": a.reward, "variant": a.variant,
                   "startup_warnings": warnings, "clean": not warnings,
                   "rates": {b: {str(n): v for n, v in d.items()} for b, d in rates.items()},
                   "rows": rows, "refs": refs, "worst_ref_drift": worst, "trending": trending,
                   "stable": worst <= 0.10 and not trending and not warnings}, f, indent=1)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
