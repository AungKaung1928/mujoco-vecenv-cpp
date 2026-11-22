"""README "At a glance" rows against the committed JSONs they were copied from:
the bit-identity contract (runs/contract.json), Python processes vs C++ threads at
8 workers (runs/compare_main.json, compare_contended.json), the C++ thread sweeps and the
sustained run (runs/bench_cpp_*.json), and the PPO / PD return and survival rows
(runs/ppo_backend_seeds.json)."""
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _load(name):
    with open(os.path.join(REPO, "runs", name)) as f:
        return json.load(f)


def _row(prefix):
    with open(os.path.join(REPO, "README.md"), encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"| {prefix} |"):
                return line
    raise AssertionError(f"README row not found: {prefix}")


def test_processes_vs_threads_row():
    r8 = next(r for r in _load("compare_main.json")["rows"] if r["workers"] == 8)
    want = f"{r8['python']:,.0f} vs {r8['cpp']:,.0f}, {r8['ratio']:.2f}x"
    assert want in _row("Python processes vs C++ threads"), want


def test_ppo_rows():
    s = _load("ppo_backend_seeds.json")
    for backend, label in (("cpp", "PPO 50M, C++ env"), ("python", "PPO 50M, Python env")):
        x = s[backend]["summary"]
        want = (f"{x['return_mean']['mean']:.1f} ± {x['return_mean']['sd']:.1f} · "
                f"{x['survival_mean']['mean']:.2f} ± {x['survival_mean']['sd']:.2f}")
        assert want in _row(label), (label, want)


def test_pd_row():
    pd = _load("ppo_backend_seeds.json")["pd"]
    want = f"{pd['return_mean']:.1f} ± {pd['seed_std_return']:.1f}"
    assert want in _row("PD hold-pose"), want


def test_contract_row():
    c = _load("contract.json")
    want = (f"{c['bit_identical']:,} / {c['observations']:,} "
            f"({100 * c['bit_identical'] / c['observations']:.2f}%), "
            f"max abs diff {c['obs_max_abs_diff']:.1e}, {c['info_mismatches']} info mismatches | "
            f"{c['episodes']} episodes")
    assert want in _row("C++ env vs Python env"), want


def test_contended_rerun_in_the_processes_vs_threads_row():
    r8 = next(r for r in _load("compare_contended.json")["rows"] if r["workers"] == 8)
    want = f"contended rerun {r8['python']:,.0f} vs {r8['cpp']:,.0f}, {r8['ratio']:.2f}x"
    assert want in _row("Python processes vs C++ threads"), want


def _sweep(name):
    rows = {r["threads"]: r for r in _load(name)["rows"]}
    return (f"{rows[1]['env_steps_per_s']:,.0f} → {rows[8]['env_steps_per_s']:,.0f}, "
            f"{rows[8]['speedup']:.2f}x, {100 * rows[8]['efficiency']:.0f}% efficiency")


def test_cpp_sweep_rows():
    assert _sweep("bench_cpp_env_main.json") in _row("C++ sweep, `env` mode")
    bare = _row("C++ sweep, `bare` mode")
    assert _sweep("bench_cpp_bare_main.json") in bare
    assert _load("bench_cpp_bare_main.json")["trending"] and "TRENDING verdict" in bare


def test_sustained_row():
    s = _load("bench_cpp_sustained.json")
    band = 100 * (s["band_high"] - s["band_low"]) / s["band_low"]
    want = (f"{s['band_low']:,.0f}–{s['band_high']:,.0f} ({band:.0f}% band), peak {s['peak']:,.0f} | "
            f"{len(s['rates'])} × {s['window_s']} s windows")
    assert want in _row(f"C++ sustained, {s['sustained_threads']} threads"), want


def test_cpp_case_count():
    """The C++ case count README.md and docs/RESULTS.md state is the number of TEST_CASEs in tests/cpp."""
    import glob
    import re
    n = sum(len(re.findall(r"^TEST_CASE\(", open(f, encoding="utf-8").read(), re.M))
            for f in glob.glob(os.path.join(REPO, "tests", "cpp", "*.cpp")))
    readme = ""
    for doc in ("README.md", os.path.join("docs", "RESULTS.md")):
        with open(os.path.join(REPO, doc), encoding="utf-8") as f:
            readme += f.read()
    for claim in (f"{n} cases /", f"{n} C++ cases", f"doctest, {n} cases"):
        assert claim in readme, claim
