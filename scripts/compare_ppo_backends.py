"""Compare the PPO run on the C++ env with the Python-env run it mirrors.

Reads both run JSONs (trace = one row per update) and reports, per 12.5M-step
block: mean return50, mean clip fraction, mean/max approx KL, and the mean
per-update env-steps/s. Writes the result as JSON so every number in the
README comes from this file.

    python scripts/compare_ppo_backends.py \
        --py ../microduck-rl/runs/ppo_v2.json --cpp runs/ppo_v2cpp.json \
        --out runs/ppo_backend_compare.json
"""
import argparse, json, statistics as st


def blocks(trace, block, total):
    out = []
    for lo in range(0, total, block):
        rows = [r for r in trace if lo < r["step"] <= lo + block]
        if not rows:
            continue
        kl = [r["approx_kl"] for r in rows]
        out.append({
            "steps": [lo, lo + block], "updates": len(rows),
            "return50_mean": st.fmean(r["return50"] for r in rows),
            "return50_last": rows[-1]["return50"],
            "clipfrac_mean": st.fmean(r["clipfrac"] for r in rows),
            "approx_kl_mean": st.fmean(kl), "approx_kl_max": max(kl),
            "frac_kl_over_0.02": sum(k > 0.02 for k in kl) / len(kl),
            "rate_mean": st.fmean(r["rate"] for r in rows),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", required=True)
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--block", type=int, default=12_500_000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {"block_steps": a.block}
    for name, path in (("python", a.py), ("cpp", a.cpp)):
        d = json.load(open(path))
        total = d["config"]["total_steps"]
        res[name] = {"file": path, "global_step": d["global_step"], "finished": d["finished"],
                     "updates": len(d["trace"]), "final_return50": d["final_return50"],
                     "blocks": blocks(d["trace"], a.block, total)}
    for name in ("python", "cpp"):
        print(name, res[name]["global_step"], "finished", res[name]["finished"])
        for b in res[name]["blocks"]:
            print(f"  {b['steps'][1]/1e6:5.1f}M  ret50 {b['return50_mean']:6.1f}  clip {b['clipfrac_mean']:.3f}"
                  f"  kl {b['approx_kl_mean']:.3f}/{b['approx_kl_max']:.2f}  >0.02 {b['frac_kl_over_0.02']:.2f}"
                  f"  rate {b['rate_mean']:,.0f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
