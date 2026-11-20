"""scripts/aggregate_ppo_seeds.py: Welch t by hand, 12.5M-step block edges, and
the saved runs/ppo_backend_seeds.json recomputed from its own per-seed values."""
import importlib.util
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "aggregate_ppo_seeds", os.path.join(REPO, "scripts", "aggregate_ppo_seeds.py"))
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def test_welch_hand_computed():
    # a: mean 2, var 1 ; b: mean 4, var 4 ; se2 = 1/3 + 4/3 = 5/3
    w = agg.welch([1, 2, 3], [2, 4, 6])
    assert w["diff"] == -2
    assert math.isclose(w["t"], -2 / math.sqrt(5 / 3))                # -1.549
    assert math.isclose(w["df"], (25 / 9) / ((1 / 9 + 16 / 9) / 2))  # 50/17 = 2.941


def test_block_survival_edges_are_left_open_right_closed():
    b = 12_500_000
    evals = [{"step": s, "survival_mean": v} for s, v in
             ((0, 9.0), (10, 0.2), (b, 0.4), (b + 1, 0.6), (2 * b, 1.0))]
    out = agg.block_survival(evals, b, 2 * b)
    assert [o["evals"] for o in out] == [2, 2]          # step 0 belongs to no block
    assert math.isclose(out[0]["survival_mean"], 0.3) and out[0]["survival_max"] == 0.4
    assert math.isclose(out[1]["survival_mean"], 0.8) and out[1]["survival_min"] == 0.6


def test_saved_result_matches_its_own_per_seed_values():
    res = json.load(open(os.path.join(REPO, "runs", "ppo_backend_seeds.json")))
    for name in ("cpp", "python"):
        again = agg.summary(res[name]["policies"])
        for m, s in res[name]["summary"].items():
            for k in ("mean", "sd", "min", "max"):
                assert math.isclose(again[m][k], s[k], rel_tol=1e-12, abs_tol=1e-15), (name, m, k)
    for m, w in res["cpp_minus_python"].items():
        again = agg.welch(res["cpp"]["summary"][m]["values"], res["python"]["summary"][m]["values"])
        assert all(math.isclose(again[k], w[k], rel_tol=1e-12) for k in ("diff", "t", "df")), m
