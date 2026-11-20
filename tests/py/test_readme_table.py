"""README "At a glance" rows against the committed JSONs they were copied from:
Python processes vs C++ threads at 8 workers (runs/compare_main.json) and the
PPO / PD return and survival rows (runs/ppo_backend_seeds.json)."""
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
