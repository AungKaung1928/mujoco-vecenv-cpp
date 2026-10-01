# mujoco-vecenv-cpp — threaded MuJoCo environments in C++, bit-identical to the Python reference

[microduck-rl](https://github.com/AungKaung1928/microduck-rl) trains a balance-and-recover
policy for the Microduck biped on a laptop with no GPU. Its environment is Python, and
because the interpreter cannot step two environments at once, the vector env runs N
environments in N forked processes. Two things were measured there and left open: the Python
wrapper costs 25% over bare `mj_step` on one core, and scaling across 8 processes settles at
55% efficiency on the `walk` model and 42% on `groundcontact`, with the loss attributed, but
not proven, to a shared resource outside the core.

This repo is the same environment in C++17: one `mjModel` and one `mjData` per environment,
N environments on T threads of one process, a pybind11 module, and a drop-in `VecEnv` that
runs microduck-rl's `ppo.py` unchanged. The claim that matters is not "faster". It is that
the C++ environment produces **the same 48 numbers, the same reward and the same flags as the
Python one, to the bit**, and that this is tested on every commit. A faster environment that
is not the same environment trains a different policy and proves nothing about the Python
result it replaces.

Everything here is measured. The equality table and the throughput tables carry the
benchmark's own stability verdicts next to them. The full 50M-step PPO run on this backend is
measured too, over three training seeds per backend: the policies it trains score 339.4 ± 2.3
of 500 on the Python env, against 348.1 ± 8.1 for the policies trained there, in about 3.1 h
per run against 4.0–4.4 h.

**Walkthrough:** https://aungkaung1928.github.io/projects/mujoco-vecenv-cpp.html — the same project explained end to end, file by file.

## What is in the box

| piece | where | what it does |
|---|---|---|
| `Env` | `include/vecenv/env.hpp`, `src/env.cpp` | one Microduck: 48-dim observation, 14 position residuals, 50 Hz over 500 Hz physics, fixed 250-step episodes, three seeded pushes, rewards v1 and v2, latency buffer, sensor noise, DR |
| `VecEnv` | `include/vecenv/vecenv.hpp`, `src/vecenv.cpp` | N envs on T threads, autoreset with `terminal_obs`, results independent of T |
| `DomainRandomizer` | `include/vecenv/dr.hpp`, `src/dr.cpp` | the five servo parameters between the four measured fits, mass, floor friction, latency, initial noise, sensor noise; exact apply and restore |
| `PCG64` | `include/vecenv/rng.hpp` | numpy's generator core, verified against numpy's internal state |
| `np::` | `include/vecenv/numpy_compat.hpp` | the three arithmetic details that decide bit equality, see below |
| `_vecenv_cpp`, `vecenv_cpp` | `python/` | pybind11 module and the drop-in `VecEnv`; `scripts/ppo_cpp.py` runs microduck-rl's PPO on it |
| `bench_vecenv` | `bench/bench_vecenv.cpp` | the sweep and sustained protocol of microduck-rl's `bench.py`, plus a `bare` mode for the physics alone |
| `compare.py` | `bench/compare.py` | Python processes vs C++ threads, alternating, same hour, same box |
| tests | `tests/cpp`, `tests/py` | 34 C++ cases (417,482 assertions), 41 Python tests; CI runs them under ASan+UBSan and TSan too |

3,594 lines of own code. doctest is vendored (MIT). MuJoCo comes from the pip wheel: the
headers and `libmujoco.so` ship inside it, so there is nothing to install system-wide, and
the C++ side links **the same library file the Python side imports**. That is a requirement,
not a convenience: the equality test is only meaningful against the same physics binary, and
`check_library_version()` refuses to run if the header and library disagree.

## The equality claim

### How it is tested

The Python environment draws its randomness from `numpy.random.default_rng(seed)`: initial
joint noise, then the push schedule (`choice` without replacement, then speeds and angles),
and under DR the five actuator parameters, mass, friction, latency and noise. Reproducing
that draw in C++ means reimplementing SeedSequence, numpy's `choice` and its ziggurat normal,
and getting every one of them right. It is possible and it is not the point.

Instead the test runs the Python env, **reads what it drew**, and hands it to the C++ env:

```python
pe = MicroduckEnv(seed=s, **kw); o_py = pe.reset(seed=s)
params = vecenv_cpp.episode_params(pe.data.qpos[qi], pe.data.qvel[vi],   # post-noise, post-clip
                                   pe._push_at, pe._push_vel, dr=pe.dr.current)
ce = vecenv_cpp.Env(make_config(**kw), s); o_cpp = ce.reset(params)
# then 250 identical actions into both, compare everything that comes back
```

`Env::reset(const EpisodeParams&)` is that entry point. `Env::reset()` with no argument uses
the env's own PCG64 and is what training uses. The two share every line of code after the
draw.

### What had to be matched, and how it was found

Three things in the Python env are not what they look like, and each one was worth a
last-bit disagreement until it was measured:

- **`np.mean` over 14 joints is not a left-to-right sum.** numpy's `add.reduce` keeps eight
  partial sums and combines them as a tree for 8 ≤ n ≤ 128. Measured on 20,000 random
  14-vectors: a sequential loop matched `np.mean` on 15,831; the tree starting from the
  identity 0 matched on 20,000 of 20,000; the tree starting from the first element matched on
  11,776. `np::pairwise_sum` is the first of those, and a C++ test uses an input where the two
  orders provably differ (`{1e16, 1×7, -1e16, 0×5}`: sequential 0, tree 6).
- **Python's `x ** 2` is `pow(x, 2.0)` from libm, not `x * x`.** Measured: 1,621 mismatches in
  2,000,000 random doubles. The v1 height term uses it. GCC folds `pow(x, 2.0)` into `x * x`
  unless stopped, so `np::py_pow` routes the exponent through a `volatile`.
- **`OBS_SCALE` is float32, `raw` is float64.** The product is `raw * double(float32(0.05))`,
  and `double(0.05f)` is 0.05000000074505806. The C++ scale table is built from `0.05f`.

And one that is not arithmetic: the Python `reset` runs `mj_forward` once on the keyframe and
a second time only after applying initial noise. A second forward pass on an unchanged state
is **not** a no-op in MuJoCo, because the constraint solver warm-starts from the previous
acceleration and lands a few ulps elsewhere. The C++ reset runs the second pass under exactly
the same condition (`init_noise > 0`).

The build sets `-ffp-contract=off`: a fused multiply-add rounds once where numpy rounds twice.

### Result

`python scripts/contract_report.py`, one core, about a minute. MuJoCo 3.12.0, numpy 2.2.6,
Python 3.10.12, g++ 11.4, this laptop, 2026-09-22 (`runs/contract.json`):

| episodes | observations compared | bit-identical | max abs diff (f32) | max rel reward diff | max abs term diff | info mismatches |
|---|---|---|---|---|---|---|
| 25 | 6,275 | 6,275 (100.00%) | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0 |

The 25 episodes cover both reward versions, latency 0 and 2, initial noise 0 and 0.02, an
action scale of 0.5 with five pushes, the `groundcontact_backlash` and `rollers` held-out
variants (35 and 25 qpos entries, strided joint indices), and two DR cases with injected
factors (latency 0 and 1, initial noise 0.226 and 0.264). Every observation is the same
float32 array, every reward the same float64, every logged term identical, and `pushed`,
`fallen`, `t`, `truncated`, `terminated`, `trunk_height` and `upright_cos` agree on every
step. `tests/py/test_contract.py` asserts this at 1e-6 / 1e-9 tolerances on every push, and a
further test shifts one push by one step and checks the comparison notices.

What this does **not** cover: sensor noise (Gaussian per step; the C++ draws its own, so the
contract cases run with it off), and the draw itself (see next section).

### What is deliberately different

- **Seeding.** `Env(cfg, seed=7)` and `MicroduckEnv(seed=7)` draw different pushes. The C++
  seeds PCG64 from a 64-bit integer via splitmix; numpy runs SeedSequence. A test states this.
  A Python run and a C++ run with the same seed are two samples of the same distribution,
  not a replay of each other. Compare them across seeds, not step by step.
- **`normal()`** is Marsaglia polar, not numpy's ziggurat. **`integers()`** and the partial
  Fisher–Yates in `schedule_pushes_` are not numpy's algorithms either.
- **The PCG64 core, `random()` and `uniform()` are numpy's exactly.** `tests/py/test_pcg64.py`
  hands numpy's internal `(state, inc)` to the C++ class and checks 1,000 raw outputs, 1,000
  doubles and 1,000 uniforms per seed. So the building block is right; the seeding and the
  higher-level draws are the deliberate gap, recorded in `docs/ISSUES.md`.

## Design

**One model per environment.** Domain randomisation writes into `mjModel` (damping, gains,
masses, friction). A shared model would make DR a data race and every environment identical.
`Env` takes a prototype and `mj_copyModel`s it; a test mutates one env's masses and checks
the prototype and its neighbours are untouched. Sixteen bodies, 82 geoms: the copies are
kilobytes.

**Threads, not processes.** The step loop has no interpreter in it, so N environments run on
T threads of one process: no pickling, no pipes, one shared observation batch. Env i always
runs on thread i mod T, so **the result does not depend on T**. `test_vecenv.cpp` runs the
same 300 steps on 1, 2 and 4 threads and compares observations, rewards, dones and terminal
observations with `==`, and runs the same 260 steps as three sequential `Env`s, autoreset
included. That test is the one the TSan job in CI exists for.

**Generation counter, spin then block.** The caller publishes the action batch, increments an
atomic generation and acts as worker 0; the T−1 pool threads spin on the counter for tens of
microseconds and then block on a condition variable. An idle `VecEnv` (the policy is thinking)
burns nothing, and a busy one pays no futex per step. Acquire/release on the counter and on
the completion count is what makes the action and observation buffers visible without a lock.

**Autoreset the way `vec_env.py` does it.** When an episode ends, the observation returned is
the new episode's first, and the ended episode's last observation is kept in `terminal_obs`.
Every `done` in this task is a truncation, never a terminal state, and `ppo.py` bootstraps
`V(terminal_obs)` on it; the flags are carried so a stock GAE loop cannot get this wrong
silently.

**The thread budget is in the code.** More than 8 threads is refused unless the caller says
`allow_overcommit`. Every number in microduck-rl was measured inside 8 of 14 cores on a
laptop that has other work to do, and this repo inherits the rule.

## Throughput — measured 2026-09-22

The protocol is microduck-rl's: a single-thread reference window before the sweep and after
every configuration (UNSTABLE if it drifts more than 10%, TRENDING if it moves monotonically
across three windows by more than 4%), a box check recorded in the JSON, and a sustained run
whose plateau band, not its mean, is the budget. Every JSON named below is in `runs/`. The
host is the laptop in *Measured environment*; its power state cannot be read from inside WSL,
so two sweeps taken hours apart are not comparable to better than a few percent, and the
tables say when they were taken.

### Sweep, `groundcontact`, reward v2, `env` mode (the full environment)

`runs/bench_cpp_env_main.json`, 14:51, freshly rebooted, 1-min load 0.22, reference drift
within 1.3%, **stable**.

| threads | env-steps/s | per thread | speedup | efficiency | ref drift |
|---|---|---|---|---|---|
| 1 | 5,639 | 5,639 | 1.00x | 100% | +1.3% |
| 2 | 10,471 | 5,236 | 1.86x | 93% | −0.1% |
| 4 | 16,601 | 4,150 | 2.94x | 74% | −0.2% |
| 8 | 20,287 | 2,536 | 3.60x | 45% | −0.8% |

`nice -n 10 ./build/bench_vecenv --threads 1 2 4 8 --seconds 20 --ref-seconds 10 --tag main`

45% at 8 threads is the same figure the Python vector env reached with 8 processes on this
model (42%). Process isolation was not the loss; whatever caps this machine at 8 workers caps
threads in one address space the same way. The `groundcontact` model has five times the
floor-collidable geoms of `walk`, and the CPU is a Core Ultra 5 225H with 4 performance, 8
efficiency and 2 low-power cores and no SMT, so heterogeneous cores are at least as good a
reading as the shared-cache one; neither is provable from inside the guest.

### The same sweep in `bare` mode (10 × `mj_step`, no environment)

`runs/bench_cpp_bare_main.json`, 21:03, freshly rebooted, 1-min load 1.28.

| threads | env-steps/s | per thread | speedup | efficiency | ref drift |
|---|---|---|---|---|---|
| 1 | 5,404 | 5,404 | 1.00x | 100% | −0.7% |
| 2 | 10,788 | 5,394 | 2.00x | 100% | +0.4% |
| 4 | 20,142 | 5,036 | 3.73x | 93% | +3.7% |
| 8 | 27,011 | 3,376 | 5.00x | 62% | +2.7% |

`nice -n 10 ./build/bench_vecenv --mode bare --threads 1 2 4 8 --seconds 20 --ref-seconds 10 --cooldown 25 --tag main`

Three things to know about this table. A sweep three minutes earlier read 5,390 / 10,823 /
19,956 / 27,087, so every row reproduced within 1%. The guard nevertheless marked both sweeps
TRENDING: the single-thread reference rose 4.4% across three consecutive windows against a
4% bar. On this laptop the single-thread reference scatters about 5% from window to window
with nothing else running, so the trend detector, written for the Python repo's steadier
process reference, fires on scatter here. The rows are quoted with that verdict attached,
not hidden. And a third sweep started right after the second, on a warm package, read
5,340 / 10,496 / 18,909 / 24,384 (`runs/bench_cpp_bare_chained.json`): 10% down at 8 threads,
which is the chained-benchmark effect microduck-rl documented, reproduced in C++.

The bare rows scale better than the env rows (62% against 45% at 8 threads) because the
physics alone is a smaller working set per step. The 1-thread bare row (5,404) sits *below*
the 1-thread env row (5,639) measured six hours earlier; the two sweeps were taken in
different host states, so the env-over-bare cost on one core cannot be read from these two
tables to better than about 4%. The Python gap was 25%. Whatever the C++ gap is, it is
inside the day-to-day scatter of the host.

### Sustained, 8 threads, 18 windows

`runs/bench_cpp_sustained.json`, 21:26, `env` mode, 60 s cooldown first.

`nice -n 10 ./build/bench_vecenv --sustained 8 --windows 18 --seconds 20 --cooldown 60 --tag sustained`

| window | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| env-steps/s | 18,724 | 20,829 | 19,440 | 18,965 | 18,548 | 17,605 | 16,724 | 16,339 | 16,001 | 15,547 | 13,782 | 13,327 | 13,329 | 15,590 | 16,238 | 14,393 | 15,074 | 14,908 |

Peak 20,829, a monotonic fall to 13,327 at window 11, a climb to 16,238 at window 14, and
14,908 at the end. Plateau over the last six windows **13,329 to 16,238, a 22% band**; last
three windows average 14,792, 29% under the peak. The Python process-based run of the same
workload had the same shape: a bottom at window 11, then a climb. Two runtimes with no code
in common dip at the same minute, so the dip belongs to the host's power management, not to
either implementation. Budget on the low end of the band.

### Python processes vs C++ threads, same hour

`runs/compare_main.json`, 21:42, started after the 1-min load fell under 0.8.

`nice -n 10 python bench/compare.py --workers 1 2 4 8 --seconds 20 --rounds 2 --tag main`

| workers | Python processes | C++ threads | ratio | C++ efficiency |
|---|---|---|---|---|
| 1 | 2,336 | 4,867 | 2.08x | 100% |
| 2 | 3,463 | 7,429 | 2.14x | 76% |
| 4 | 4,270 | 10,370 | 2.43x | 53% |
| 8 | 6,385 | 12,327 | 1.93x | 32% |

A first run of the same command, started while the 1-min load was still 3.06 from the
sustained test (`runs/compare_contended.json`), gave 8 workers 6,668 vs 13,612, a 2.04x
ratio. The multi-worker rows of the two runs agree within 10% and the ratio at 8 is 1.9x to
2.0x either way. Both runs were marked UNSTABLE by the single-process Python reference
(scatter up to 39% between windows): one forked worker behind a pipe has a jittery
round-trip that the reference guard, designed for bare `mj_step`, reads as a moving machine.
The guard's verdict is recorded in both JSONs.

So the decision rule is settled: the C++ environment is worth its build step for training.
The efficiency question is not improved by it: 32% at 8 workers under this Python-driven
protocol, 45% in the pure C++ sweep, 42% for Python processes. Threads did not buy back the
scaling loss; they bought a 2x constant factor.

**The finding that matters for microduck-rl:** its Python vector env delivers about 6,400
env-steps/s at 8 processes under random actions. The 13,300 env-steps/s that repo budgeted
from was composed as bare physics × wrapper factor, and the inter-process round trip was
never in the composition. The first PPO run on that env, started tonight, reports about
3,450–3,750 env-steps/s with the policy forward pass and the update in the loop. That is the
fifth revision of the same number, downward again, and it is the first one measured on the
workload itself.

### PPO on the C++ env — measured 2026-09-29/30

microduck-rl's step-3 run, repeated on this backend: same configuration, same training seed,
same two-chunk schedule (0–25M without a KL stop, 25–50M resumed with `--target-kl 0.02`),
same 8 envs and 256-step rollouts. `ppo.py` runs unchanged as `__main__` with `vecenv_cpp`
installed as `vec_env`. The C++ run was cut into four 12.5M-step invocations so the laptop
could cool between them; the Python run was two 25M chunks, the second of them in three
invocations.

```
OMP_NUM_THREADS=1 nice -n 10 python scripts/ppo_cpp.py --total-steps 50000000 --chunk-steps 12500000 --tag v2cpp --reward v2
OMP_NUM_THREADS=1 nice -n 10 python scripts/ppo_cpp.py --resume runs/ppo_v2cpp.ckpt.pt --tag v2cpp                   # 12.5M -> 25M
OMP_NUM_THREADS=1 nice -n 10 python scripts/ppo_cpp.py --resume runs/ppo_v2cpp.ckpt.pt --tag v2cpp --target-kl 0.02  # twice, to 50M
cd ../microduck-rl && python eval_policy.py ../mujoco-vecenv-cpp/runs/ppo_v2cpp.pt --variant groundcontact \
    --out ../mujoco-vecenv-cpp/runs/eval_v2cpp_groundcontact.json
python scripts/compare_ppo_backends.py --py ../microduck-rl/runs/ppo_v2.json --cpp runs/ppo_v2cpp.json \
    --out runs/ppo_backend_compare.json
# seeds 1 and 2: --seed N --tag v2cpp_sN here, --seed N --tag v2_sN in microduck-rl, then
python scripts/aggregate_ppo_seeds.py --cpp-evals runs/eval_v2cpp{,_s1,_s2}_groundcontact.json \
    --cpp-runs runs/ppo_v2cpp{,_s1,_s2}.json \
    --py-evals ../microduck-rl/runs/eval_v2{,_s1,_s2}_groundcontact.json \
    --py-runs ../microduck-rl/runs/ppo_v2{,_s1,_s2}.json --out runs/ppo_backend_seeds.json
```

**Final evaluation**, both policies on microduck-rl's *Python* env, 100 episodes × 5 evaluation
seeds (the same five for both), ± is the standard deviation over the five seed means:

| policy | trained on | return of 500 | survival | upright | median trunk height | pushes on a fallen robot |
|---|---|---|---|---|---|---|
| `runs/ppo_v2cpp.pt` | C++ env | 341.9 ± 4.1 | 0.95 | 0.50 | 5.9 cm | 616 |
| microduck-rl `runs/ppo_v2.pt` | Python env | 344.8 ± 2.9 | 0.73 | 0.50 | 5.9 cm | 629 |
| PD hold-pose | — | 175.8 ± 0.2 | 0.47 | — | — | 1443 |

This first pair is training seed 0. Seeds 1 and 2 follow below.

Paired by evaluation seed, the C++-trained policy scores −3.5, −0.9, −3.7, −8.3 and +2.4
against the Python-trained one: mean −2.8, sd 3.9 over five seeds, about 1.6 standard errors.
That is inside the noise of the evaluation. The evaluation noise is not the large term,
though; the run-to-run spread is, so the run was repeated.

**Three training seeds per backend** (measured 2026-09-30/10-01). Seeds 1 and 2 on each
backend repeat the seed-0 protocol exactly: four 12.5M-step invocations, the first two without
a KL stop, the last two with `--target-kl 0.02`, the same evaluation. One deviation: the
Python seed-0 run did its first 25M as one invocation, the new Python seeds in two, which only
changes the env reseed at the resume. `scripts/aggregate_ppo_seeds.py` writes every number
below to `runs/ppo_backend_seeds.json`:

| training seed | C++ return | C++ survival | Python return | Python survival |
|---|---|---|---|---|
| 0 | 341.9 ± 4.1 | 0.948 | 344.8 ± 2.9 | 0.727 |
| 1 | 338.5 ± 8.1 | 0.998 | 357.3 ± 7.6 | 0.999 |
| 2 | 337.7 ± 7.6 | 0.873 | 342.2 ± 5.1 | 0.895 |
| **mean ± sd over seeds** | **339.4 ± 2.3** | **0.94 ± 0.06** | **348.1 ± 8.1** | **0.87 ± 0.14** |

Upright fraction is 0.49–0.50 for all six policies. Median trunk height is 5.9–6.0 cm for five
of them; Python seed 1 holds 8.8 cm, the one policy that found a taller posture, and also the
highest return. C++ minus Python, Welch over training seeds: return −8.7 (t −1.8, df 2.3),
survival +0.07 (t 0.75, df 2.8). Neither is a backend difference at n = 3.

**Survival is the column that looked different, and the extra seeds took the difference
away.** Survival counts steps with the trunk above 4 cm; standing height is 12 cm, so it says
how often the trunk dips under 4 cm from a low crouch. With one seed each it read 0.95 against
0.73, and that gap was real for those two policies (500 episodes, standard errors 0.005 and
0.013). Across seeds it is run-to-run spread: the Python seeds span 0.73–1.00 and the C++
seeds 0.87–1.00. The in-training evaluations say the same thing; each run settles into its
survival level from the first 12.5M-step block, and a second Python seed sits at 1.00 in
every block where seed 0 sat at 0.68–0.80. The seed-0 Python run was the low draw.

**Training curves**, training seed 0, per 12.5M-step block (`runs/ppo_backend_compare.json`: mean over the
block's updates of return50, clip fraction and approx KL; "rate" is the per-update env-steps/s
ppo.py logs):

| block | return50 C++ / Python | clip fraction C++ / Python | KL mean (max) C++ | KL mean (max) Python | rate C++ / Python |
|---|---|---|---|---|---|
| 0–12.5M | 345.1 / 337.9 | 0.56 / 0.58 | 0.21 (11.0) | 0.22 (25.1) | 4,725 / 3,396 |
| 12.5–25M | 331.5 / 330.4 | 0.69 / 0.73 | 0.76 (65.7) | 0.84 (173.1) | 4,755 / 3,123 |
| 25–37.5M, KL stop | 342.3 / 332.2 | 0.30 / 0.34 | 0.039 (0.53) | 0.042 (0.57) | 7,103 / 4,342 |
| 37.5–50M, KL stop | 349.5 / 344.3 | 0.21 / 0.23 | 0.022 (0.08) | 0.022 (0.08) | 5,998 / 3,735 |

The step-size failure microduck-rl found in its chunk 1 reproduced here. Without a KL stop,
the clip fraction climbed to 0.69 and the KL spiked to 66 in the second block, and return50
fell the same way (345 → 332 here, 338 → 330 there). The KL stop pulled both runs back to the
same clip fraction and KL within a few hundredths. A failure that shows on two backends with
different push draws is a property of the configuration, not of one unlucky run.

**Wall-clock for 50M steps:** 11,148 s (3.1 h) on the C++ env, the sum of ppo.py's
`chunk_wall_s` over the four invocations. The Python run took 8,786 s for its first 25M
(`chunk_wall_s`). Its second 25M took 7,122 s by the job runner's start/end stamps (58 and
60 min), plus a ~4-minute invocation stopped after a checkpoint whose 3 updates were redone.
That is about 15,900 s (4.4 h) in total, and the C++ env took about 0.70 of that time:

| stage | C++ | Python | ratio |
|---|---|---|---|
| 0–25M, no KL stop | 6,246 s | 8,786 s | 1.41x |
| 25–50M, KL stop | 4,914 s (runner stamps) | 7,122 s (runner stamps) | 1.45x |

Seeds 1 and 2, four invocations each by the runner's stamps: C++ 11,579 s and 10,994 s
(3.2 h, 3.1 h), Python 14,473 s and 13,803 s (4.0 h, 3.8 h): 1.25x for both pairs, against
1.43x for seed 0. The Python seed-0 run was the slow one; why is not measured. Take the
end-to-end training speed-up as 1.25x–1.43x on this laptop.

The per-update rates say the same with more detail. The gain is 1.39–1.52x while every update
runs its full 5 epochs, and 1.61–1.64x once the KL stop ends most updates after fewer epochs.
The shorter the update, the larger the share of time the env gets, and the more the faster env
shows. Neither figure reaches the ~2x of the random-action comparison above, because PPO puts
the policy forward pass and the gradient update, both single-threaded Python/PyTorch, on the
same critical path. A `top` snapshot 15 minutes into the last chunk shows the one training
process at 470% CPU with 8 env threads, so the threads sat idle a good part of every
rollout. Both backends' rates also moved by 14–16% between invocations of the same stage
(C++ 7,103 then 5,998; Python 4,342 then 3,735), on the same laptop, in one evening each. That
spread is why the ratio is quoted per stage and not as a single number.

## Tests

C++ (`./build/vecenv_tests`, doctest, 34 cases, 417,482 assertions): the observation is 48 finite floats with the
float32 scale; zero action commands STAND exactly and out-of-range actions clip to the joint
limits; projected gravity reads (0, 0, −1) upright and stays a unit vector through 120 steps of
thrashing; `done` fires once, on step 250, as a truncation; reward equals the ordered sum of
its terms under both versions; same seed replays across three constructions; `step` before
`reset` throws; pushes land on the scheduled step and diverge exactly there; a second
`mj_forward` changes nothing; reset clamps the initial state at `init_noise = 0.6`; latency
delays the applied action by exactly `latency` steps while `prev_action` stays current; the PD
baseline falls; the held-out variants give 48 dims through strided indices; DR ranges are the
min/max of the four fits, samples stay inside them, apply/restore are exact and nothing
compounds; each env owns its model; sensor noise touches only the IMU blocks; 1, 2 and 4
threads agree; N threaded envs equal N sequential ones; the thread budget is enforced; PCG64
determinism, ranges and moments; the numpy summation order with a provably different input.

Python (`pytest tests/py`, 41 tests): the equality cases above at three seeds each plus two
DR cases at two seeds; the push-shift sensitivity check; PCG64 raw/double/uniform against
numpy's internal state at three seeds; header, library and wheel are one MuJoCo version;
constants, scale table, term names and DR ranges equal the Python module's; the drop-in
`VecEnv` has `vec_env.VecEnv`'s shapes, dtypes, info keys, `terminal_obs` on every done,
`with_terms`, `reset(seed=)` replay and bare `reset()` semantics; thread count does not change
the numbers; microduck-rl's `DomainRandomizer` object is accepted as `dr=`; bad kwargs,
nine threads and an unknown variant are refused.

CI: the Release build and both suites against microduck-rl at a pinned commit with the
upstream assets fetched; the C++ suite again under `-fsanitize=address,undefined` and under
`-fsanitize=thread`.

## What this does not prove

- **2x is a constant factor on one laptop.** The Python-vs-C++ ratio was measured on one
  14-core machine in one evening, with the single-process reference flagged unstable both
  times; the multi-worker rows reproduced within 10% across two runs, and that is the whole
  evidence. It says nothing about a machine with more cores or a steadier power budget.
- **The sweep tables carry the guard's verdict, not a clean pass.** The bare sweep is
  TRENDING by a trend detector that fires on this box's 5% single-thread scatter; the rows
  reproduced within 1% across two sweeps and are quoted on that basis.
- **Bit equality holds on one machine, one compiler, one libm.** Python's `pow` and `exp` and
  the C++ ones are the same libm calls on the same host, which is why they agree; a runner
  with a different glibc or a numpy built with a different SIMD `exp` could differ in the last
  bit on the height term. The test tolerances (1e-6 on float32 observations, 1e-9 relative on
  rewards) are set so that a last-bit libm difference passes and a real defect does not; the
  report script counts exact matches separately so the distinction stays visible.
- **The draw is not replicated.** Same seed, different pushes. A training-run replay across
  backends is not available; comparisons are statistical.
- **Sensor noise is not contract-tested**, for the same reason.
- **Three training seeds per backend.** Enough to show the seed-0 survival gap was
  run-to-run spread, not enough to resolve a backend difference smaller than that spread
  (survival sd 0.06–0.14, return sd 2–8 over seeds). The training curves table is seed 0 only.
- **One robot.** `Env` is Microduck-shaped: 14 actuators, a `STAND` keyframe, a `trunk_base`
  body, `angular-velocity` and `orientation` sensors. The threading layer is not, and a second
  model behind the same interface is the natural next step once this one has numbers.
- **No rendering.** Image observations through the MuJoCo C rendering API are deferred until
  a machine with a working EGL context exists; the laptop this was written on has GLFW through
  WSLg and nothing else.

## Reproducing

Needs a C++17 compiler, CMake ≥ 3.18, Python ≥ 3.8, and microduck-rl with its assets next to
this repo (the Microduck meshes are CC BY-SA-NC and are not committed anywhere; its
`fetch_assets.sh` pulls them from a pinned upstream commit).

```bash
git clone https://github.com/AungKaung1928/microduck-rl.git ../microduck-rl
(cd ../microduck-rl && python3 -m venv .venv && . .venv/bin/activate \
   && pip install -r requirements.txt && ./fetch_assets.sh)

pip install -r requirements.txt            # mujoco (headers + library inside), numpy, pybind11, pytest
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/vecenv_tests
python -m pytest tests/py -q
python scripts/contract_report.py          # the equality table, one core, a minute
```

`./verify.sh` does all of that in order and prints the benchmark commands at the end. Set
`MICRODUCK_RL` or `MICRODUCK_ASSETS` if microduck-rl lives somewhere else. `python3 -m venv`
is broken on some machines; `uv venv` works the same way.

The same thing from nothing, inside a container: `docker build -t mujoco-vecenv-cpp . && docker run --rm
mujoco-vecenv-cpp` clones microduck-rl at the pinned commit, fetches the meshes, builds, and runs the
C++ and Python tests. Image built and its default test command passed inside it on 2026-09-23
(doctest suite plus 41 pytest cases), image size 1.05 GB.

Sanitizer builds: `cmake -S . -B build-san -DVECENV_SANITIZE=address,undefined
-DVECENV_BUILD_PYTHON=OFF -DVECENV_BUILD_BENCH=OFF` (or `thread`). Two things worth knowing.
`mujoco.h` includes a header meant for building MuJoCo itself under ASan that g++ rejects in
C++; `include/vecenv/mujoco.hpp` skips it, which is why every file includes MuJoCo through
that wrapper. And MuJoCo's XML compiler runs its own thread pool inside the prebuilt,
uninstrumented library, which TSan reports as a race on a mutex it cannot see into;
`tests/cpp/tsan.supp` suppresses reports from inside `libmujoco` and nothing else, so run
with `TSAN_OPTIONS=suppressions=$PWD/tests/cpp/tsan.supp`. On a recent kernel TSan may also
abort with "unexpected memory mapping"; `setarch $(uname -m) -R ./build-san/vecenv_tests`
disables address-space randomisation for that one process and needs no privileges.

Using it from microduck-rl's PPO, with `ppo.py` untouched:

```bash
OMP_NUM_THREADS=1 nice -n 10 python scripts/ppo_cpp.py --total-steps 50000000 \
    --chunk-steps 25000000 --tag v2cpp --reward v2       # VECENV_THREADS=8 is the default
```

Or in your own code:

```python
import vecenv_cpp
envs = vecenv_cpp.VecEnv(n=8, seed=0, reward="v2", dr=True)   # same kwargs as MicroduckEnv
obs = envs.reset()
obs, rew, done, infos = envs.step(actions)                     # (8,48) f32, (8,) f64, (8,) bool, list of dicts
```

## Measured environment

Intel Core Ultra 5 225H, 14 cores, WSL2 (kernel 6.18), Ubuntu 22.04, g++ 11.4.0, CMake 3.22.1,
Python 3.10.12, MuJoCo 3.12.0 (pip wheel), numpy 2.2.6, pybind11 3.1.0. No CUDA anywhere.

## Licence

MIT. `third_party/doctest/doctest.h` is MIT (Viktor Kirilov). MuJoCo is Apache-2.0 and is
used through its pip wheel. The Microduck model and meshes belong to Pollen Robotics
(code Apache-2.0, 3D files CC BY-SA-NC) and are fetched, never redistributed, by microduck-rl.
