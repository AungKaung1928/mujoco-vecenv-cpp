# Issues to open on GitHub

One block per issue: title, then body. The repo is code-complete and its
tests pass; every number below the equality table in the README is
`TODO(measure)` until the run phase, and these are the open ends.

---

**Measure the sweep: 1/2/4/8 threads, env and bare, on an idle box**

`nice -n 10 ./build/bench_vecenv --threads 1 2 4 8 --seconds 20 --ref-seconds 10 --tag main`
and the same with `--mode bare`. The README's throughput table is empty
until both JSONs exist with `clean: true` and `stable: true`. Record the
host power state in the commit message; WSL cannot read it later.

---

**Measure the sustained run and the plateau band**

`nice -n 10 ./build/bench_vecenv --sustained 8 --windows 18 --seconds 20 --tag sustained`.
microduck-rl's Python run bottomed at window 11 and was still rising at 18;
compare the shape, not just the tail. Budget on the low end of the band.

---

**Python processes vs C++ threads, same hour**

`nice -n 10 python bench/compare.py --workers 1 2 4 8 --seconds 20 --rounds 2 --tag main`.
This is the number the README's headline needs: the ratio at 8 workers, and
whether thread scaling efficiency beats the 55% (walk) / 42% (groundcontact)
the process-based env measured. A ratio under 1.3x means the C++ env is not
worth its build step for training and the README should say so.

---

**Rerun Microduck step 3 on the C++ env and compare curves**

After microduck-rl's Python step-3 run exists: `scripts/ppo_cpp.py` with
the same config and seeds. Same distribution of pushes, different draws, so
compare return curves and final eval statistically (seeds), not point by
point. Report wall-clock for the 50M budget on both.

---

**numpy-equivalent seeding, or decide not to**

Same integer seed, different push schedule than the Python env. Full
equivalence needs SeedSequence, numpy's `choice` without replacement and its
ziggurat normal. The contract test does not need it (it injects the draw).
Decide whether a training-run replay across backends is worth ~400 lines.

---

**Offscreen rendering through the MuJoCo C API**

Image observations in C++ (`mjr_*`, EGL or OSMesa). Deferred: the arm repos
are the ones that consume images, and on the laptop this was written on only
GLFW through WSLg works. Needs a machine with a working EGL first.

---

**Second robot behind the same interface**

`EnvConfig` and `Env` are Microduck-shaped (14 actuators, STAND keyframe,
trunk_base). so-arm100-sim's tasks are the next candidate; the threading
layer is already model-agnostic. Only after the Microduck numbers exist.
