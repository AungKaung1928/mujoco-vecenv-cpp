#!/usr/bin/env bash
# Reproduce the README's claims from a clean checkout.
#
# Tiers 1-3 are cheap and single-core apart from a few seconds of 4-thread
# tests: build, the C++ suite, the Python suite including the equality test
# against microduck-rl's environment. Tier 4 is the benchmark and needs the
# machine to itself.
set -u
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import mujoco, numpy, pybind11' 2>/dev/null; then
  echo "mujoco/numpy/pybind11 are not importable with '$PY'. From the repo root:" >&2
  echo "    python3 -m venv .venv && . .venv/bin/activate" >&2
  echo "    pip install -r requirements.txt" >&2
  exit 1
fi
MICRODUCK_RL="${MICRODUCK_RL:-$(pwd)/../microduck-rl}"
export MICRODUCK_ASSETS="${MICRODUCK_ASSETS:-$MICRODUCK_RL/assets}"
if [ ! -f "$MICRODUCK_ASSETS/scene.xml" ]; then
  cat <<MSG
No Microduck assets at $MICRODUCK_ASSETS. This repo reuses microduck-rl's
model (the meshes are CC BY-SA-NC and are not committed anywhere):

    git clone https://github.com/AungKaung1928/microduck-rl.git ../microduck-rl
    (cd ../microduck-rl && ./fetch_assets.sh)

or set MICRODUCK_RL / MICRODUCK_ASSETS.
MSG
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
hr() { printf '\n=== %s ===\n' "$1"; }

hr "1/4  build (Release, no fast-math, no FMA contraction)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v "$PY")" >/dev/null || exit 1
cmake --build build -j"${JOBS:-4}" || exit 1

hr "2/4  C++ suite -- contract, DR, thread-count independence, RNG, numpy arithmetic"
./build/vecenv_tests || exit 1

hr "3/4  Python suite -- equality against microduck-rl's env, PCG64 vs numpy, the drop-in"
"$PY" -m pytest tests/py -q -p no:cacheprovider || exit 1

hr "4/4  the runs that load the machine"
cat <<MSG
Everything above is single-core or a few seconds of 4 threads. The rest holds
8 threads for minutes and is only worth running on an idle box (load average
under 1.5; both benchmarks record the box state in their JSON).

Equality report, one core, a minute:
    $PY scripts/contract_report.py

C++ sweep and sustained run, same protocol as microduck-rl/bench.py:
    nice -n 10 ./build/bench_vecenv --threads 1 2 4 8 --seconds 20 --ref-seconds 10 --tag main
    nice -n 10 ./build/bench_vecenv --mode bare --threads 1 2 4 8 --seconds 20 --ref-seconds 10 --tag main
    nice -n 10 ./build/bench_vecenv --sustained 8 --windows 18 --seconds 20 --tag sustained

Python processes vs C++ threads, alternating, same hour:
    nice -n 10 $PY bench/compare.py --workers 1 2 4 8 --seconds 20 --rounds 2 --tag main

PPO on the C++ env (ppo.py unchanged, see scripts/ppo_cpp.py):
    OMP_NUM_THREADS=1 nice -n 10 $PY scripts/ppo_cpp.py --total-steps 50000000 --chunk-steps 25000000 --tag v2cpp --reward v2
MSG
for f in runs/contract.json runs/compare_main.json runs/bench_cpp_env_main.json; do
  [ -f "$f" ] && "$PY" - "$f" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
if "bit_identical" in d:
    print(f"\n{p}: {d['bit_identical']:,} of {d['observations']:,} observations bit-identical, "
          f"max |diff| {d['obs_max_abs_diff']:.1e}, max rel reward diff {d['reward_max_rel_diff']:.1e}")
elif "rows" in d and d["rows"] and "ratio" in d["rows"][0]:
    print(f"\n{p}: clean={d['clean']} stable={d['stable']}")
    for r in d["rows"]:
        print(f"  {r['workers']} workers  python {r['python']:>9,.0f}  cpp {r['cpp']:>9,.0f}  {r['ratio']:.2f}x")
elif "rows" in d:
    print(f"\n{p}: clean={d['clean']} stable={d['stable']} pass={d['pass']}")
    for r in d["rows"]:
        print(f"  {r['threads']} threads  {r['env_steps_per_s']:>9,.0f} env-steps/s  {100*r['efficiency']:3.0f}% efficient")
PY
done
exit 0
