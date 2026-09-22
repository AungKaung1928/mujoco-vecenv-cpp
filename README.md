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

Everything here is measured or marked `TODO(measure)`. The equality table is measured. The
throughput tables are not yet: they need an idle machine and the run phase has not started.

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

## Throughput — `TODO(measure)`

Nothing below is measured yet. The protocol is microduck-rl's, so the rows will be comparable
to the ones already published there: a single-thread reference window before the sweep and
after every configuration (UNSTABLE if it drifts more than 10%, TRENDING if it moves
monotonically across three windows by more than 4%), a box check recorded in the JSON, and a
sustained run whose plateau band, not its mean, is the budget.

### Sweep, `groundcontact`, reward v2, `env` mode (the full environment)

| threads | env-steps/s | per thread | speedup | efficiency | ref drift |
|---|---|---|---|---|---|
| 1 | TODO(measure) | | 1.00x | 100% | |
| 2 | TODO(measure) | | | | |
| 4 | TODO(measure) | | | | |
| 8 | TODO(measure) | | | | |

`nice -n 10 ./build/bench_vecenv --threads 1 2 4 8 --seconds 20 --ref-seconds 10 --tag main`

### The same sweep in `bare` mode (10 × `mj_step`, no environment)

Same rows, `--mode bare`. The gap between the two tables is the cost of everything that is not
physics: observation assembly, reward, pushes, bookkeeping and the one `mj_forward` that keeps
the observation on a single timestamp. In Python that gap was 25% (4,175 vs 5,223 env-steps/s
on one core, of which about 7 points is the `mj_forward`). The prediction here is a few
percent, and the `mj_forward` share stays, because it is a correctness cost and not overhead.

### Sustained, 8 threads, 18 windows

`nice -n 10 ./build/bench_vecenv --sustained 8 --windows 18 --seconds 20 --tag sustained`

microduck-rl's process-based run of the same workload bottomed out at window 11 and was still
rising at window 18, a 9% band. What matters is whether the thread-based shape is the same.

### Python processes vs C++ threads, same hour

`nice -n 10 python bench/compare.py --workers 1 2 4 8 --seconds 20 --rounds 2 --tag main`

| workers | Python processes | C++ threads | ratio | C++ efficiency |
|---|---|---|---|---|
| 1 | TODO(measure) | TODO(measure) | | 100% |
| 8 | TODO(measure) | TODO(measure) | | |

Two things this will settle. First, the ratio at 8, which is the only number that decides
whether the C++ env is worth its build step for training: under about 1.3x it is not, and
this README will say so. Second, the efficiency question microduck-rl left open. Its
process-based sweep lost efficiency earlier on the variant with five times the collidable
geoms, which points at L3 or memory bandwidth rather than at core type. Threads in one
address space share the same L3 as processes do, so if the efficiency curve is unchanged
the shared-resource reading gets a second data point; if it improves markedly, something
about process isolation (page tables, scheduler placement) was part of the loss.

A 2-second smoke of the bench, run while writing it on a box that was not idle, read about
5,100 env-steps/s on one thread in `env` mode and about 5,300 in `bare` mode. Those are not
quotable and are not in the tables; they are here so that the first real run has something
to be surprised by.

### PPO on the C++ env

After microduck-rl's step-3 run exists in Python: `scripts/ppo_cpp.py` with the same
configuration and the same seeds. `ppo.py` is executed as `__main__` with `vecenv_cpp`
installed under the module name `vec_env`; nothing in microduck-rl changes. Compare return
curves and final evaluation across seeds, not point by point (different push draws, see
above), and report wall-clock for the 50M budget on both backends. A 1,024-step smoke of the
loop on the C++ backend runs and writes its checkpoint; that is all it has been asked to do.

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

- **Nothing here is faster yet.** Every throughput cell is `TODO(measure)`. The smoke reading
  above was taken on a busy box for two seconds and is disclosed, not claimed.
- **Bit equality holds on one machine, one compiler, one libm.** Python's `pow` and `exp` and
  the C++ ones are the same libm calls on the same host, which is why they agree; a runner
  with a different glibc or a numpy built with a different SIMD `exp` could differ in the last
  bit on the height term. The test tolerances (1e-6 on float32 observations, 1e-9 relative on
  rewards) are set so that a last-bit libm difference passes and a real defect does not; the
  report script counts exact matches separately so the distinction stays visible.
- **The draw is not replicated.** Same seed, different pushes. A training-run replay across
  backends is not available; comparisons are statistical.
- **Sensor noise is not contract-tested**, for the same reason.
- **No policy has trained on this yet.** The 1,024-step smoke proves the plumbing, not that
  a policy trained here matches one trained on the Python env. That comparison is a run-phase
  item with its own row in `docs/ISSUES.md`.
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
