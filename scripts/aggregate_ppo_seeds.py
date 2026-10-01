"""Aggregate the PPO backend comparison over training seeds.

Reads one final-evaluation JSON (microduck-rl eval_policy.py, Python env,
100 episodes x 5 evaluation seeds) and one training-run JSON per training seed
and backend. Reports per policy the evaluation return, survival, upright
fraction and median trunk height; per backend the mean and sample sd of those
over training seeds; the backend difference with a Welch t; and the
in-training evaluation survival per 12.5M-step block. Writes the result as
JSON so every number in the README comes from this file.

    python scripts/aggregate_ppo_seeds.py \
        --cpp-evals runs/eval_v2cpp_groundcontact.json runs/eval_v2cpp_s1_groundcontact.json \
                    runs/eval_v2cpp_s2_groundcontact.json \
        --cpp-runs runs/ppo_v2cpp.json runs/ppo_v2cpp_s1.json runs/ppo_v2cpp_s2.json \
        --py-evals ../microduck-rl/runs/eval_v2_groundcontact.json \
                   ../microduck-rl/runs/eval_v2_s1_groundcontact.json \
                   ../microduck-rl/runs/eval_v2_s2_groundcontact.json \
        --py-runs ../microduck-rl/runs/ppo_v2.json ../microduck-rl/runs/ppo_v2_s1.json \
                  ../microduck-rl/runs/ppo_v2_s2.json \
        --out runs/ppo_backend_seeds.json
"""
import argparse, json, math, statistics as st

METRICS = ("return_mean", "survival_mean", "upright_mean", "height_p50")


def block_survival(evals, block, total):
    out = []
    for lo in range(0, total, block):
        rows = [e for e in evals if lo < e["step"] <= lo + block]
        if rows:
            s = [e["survival_mean"] for e in rows]
            out.append({"steps": [lo, lo + block], "evals": len(rows),
                        "survival_mean": st.fmean(s), "survival_min": min(s), "survival_max": max(s)})
    return out


def policy(eval_path, run_path):
    ev, run = json.load(open(eval_path)), json.load(open(run_path))
    ln = ev["learned"]
    total = run["config"]["total_steps"]
    return {"eval_file": eval_path, "run_file": run_path, "seed": run["config"]["seed"],
            "finished": run["finished"], "global_step": run["global_step"],
            **{m: ln[m] for m in METRICS},
            "seed_std_return": ln["seed_std_return"], "per_seed_return": ln["per_seed_return"],
            "pushes_unrecoverable": ln["pushes_unrecoverable"],
            "train_survival_blocks": block_survival(run["evals"], 12_500_000, total)}, ev


def summary(pols):
    out = {}
    for m in METRICS + ("pushes_unrecoverable",):
        v = [p[m] for p in pols]
        out[m] = {"mean": st.fmean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0,
                  "min": min(v), "max": max(v), "values": v}
    return out


def welch(a, b):
    ma, mb, va, vb = st.fmean(a), st.fmean(b), st.variance(a), st.variance(b)
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    t = (ma - mb) / math.sqrt(se2) if se2 > 0 else math.copysign(math.inf, ma - mb)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) if se2 > 0 else float("nan")
    return {"diff": ma - mb, "t": t, "df": df}


def main():
    ap = argparse.ArgumentParser()
    for k in ("cpp-evals", "cpp-runs", "py-evals", "py-runs"):
        ap.add_argument(f"--{k}", nargs="+", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res, pd = {}, None
    for name, evs, runs in (("cpp", a.cpp_evals, a.cpp_runs), ("python", a.py_evals, a.py_runs)):
        assert len(evs) == len(runs), f"{name}: {len(evs)} evals vs {len(runs)} runs"
        pols = []
        for e, r in zip(evs, runs):
            p, ev = policy(e, r)
            pols.append(p)
            pd = pd or ev["pd"]
        res[name] = {"policies": pols, "summary": summary(pols)}
    res["pd"] = {m: pd[m] for m in METRICS if m in pd} | {"seed_std_return": pd["seed_std_return"]}
    if min(len(res[n]["policies"]) for n in ("cpp", "python")) > 1:
        res["cpp_minus_python"] = {m: welch(res["cpp"]["summary"][m]["values"],
                                            res["python"]["summary"][m]["values"]) for m in METRICS}
    for name in ("cpp", "python"):
        print(name)
        for p in res[name]["policies"]:
            tb = " ".join(f"{b['survival_mean']:.2f}" for b in p["train_survival_blocks"])
            print(f"  seed {p['seed']}  ret {p['return_mean']:6.1f} ± {p['seed_std_return']:.1f}"
                  f"  surv {p['survival_mean']:.3f}  up {p['upright_mean']:.3f}"
                  f"  h50 {p['height_p50']*100:.1f} cm  train-surv/block {tb}")
        s = res[name]["summary"]
        print("  mean ± sd  " + "  ".join(f"{m} {s[m]['mean']:.3f} ± {s[m]['sd']:.3f}" for m in METRICS))
    for m, w in res.get("cpp_minus_python", {}).items():
        print(f"cpp - python  {m}: {w['diff']:+.3f}  Welch t {w['t']:.2f}  df {w['df']:.1f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
