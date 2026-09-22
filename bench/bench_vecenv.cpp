// How fast do N Microducks step on T threads, and does the box hold it?
//
// Same protocol as microduck-rl/bench.py, so the two are comparable:
//   sweep      1, 2, 4, 8 threads, each bracketed by a single-thread reference
//              window; the sweep is UNSTABLE if the reference drifts > 10%, and
//              TRENDING if it moves monotonically across 3+ windows by > 4%
//              (a machine changing state, not noise)
//   sustained  hold T threads flat out for W windows and report the decay,
//              the last-third plateau band, and whether it settled
// Two workloads:
//   env        the full environment: actions in, observation + reward out,
//              autoreset. What a training run does.
//   bare       ctrl = nominal + noise, 10 x mj_step, reset when fallen. What
//              bench.py's worker does. The gap between the two is the cost of
//              everything that is not physics.
// A box check runs first and is recorded in the JSON, because a benchmark
// started next to another job measures contention and says nothing.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vecenv/boxcheck.hpp"
#include "vecenv/env.hpp"
#include "vecenv/model.hpp"
#include "vecenv/rng.hpp"
#include "vecenv/vecenv.hpp"

using namespace vecenv;
using clk = std::chrono::steady_clock;

namespace {

double seconds_since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

struct Args {
    std::vector<int> threads = {1, 2, 4, 8};
    int envs_per_thread = 1;
    double seconds = 12.0;
    double ref_seconds = -1.0;
    std::string variant = "groundcontact";
    std::string reward = "v2";
    std::string mode = "env";
    int sustained = 0;
    int windows = 12;
    double gate = 5000.0;
    bool allow_overcommit = false;
    double cooldown = 0.0;
    std::string tag, out;
    std::string assets;
};

void usage() {
    std::puts("bench_vecenv [--threads 1 2 4 8] [--envs-per-thread 1] [--seconds 12] [--ref-seconds 6]\n"
              "             [--variant groundcontact] [--reward v2] [--mode env|bare]\n"
              "             [--sustained T --windows 12] [--gate 5000] [--allow-overcommit]\n"
              "             [--cooldown SECONDS] [--tag NAME] [--out FILE] [--assets DIR]");
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&](double& v) { if (i + 1 >= argc) throw std::runtime_error(k + " needs a value"); v = std::stod(argv[++i]); };
        auto nexts = [&](std::string& v) { if (i + 1 >= argc) throw std::runtime_error(k + " needs a value"); v = argv[++i]; };
        if (k == "--threads") {
            a.threads.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') a.threads.push_back(std::stoi(argv[++i]));
        } else if (k == "--envs-per-thread") { double v; next(v); a.envs_per_thread = static_cast<int>(v); }
        else if (k == "--seconds") next(a.seconds);
        else if (k == "--ref-seconds") next(a.ref_seconds);
        else if (k == "--variant") nexts(a.variant);
        else if (k == "--reward") nexts(a.reward);
        else if (k == "--mode") nexts(a.mode);
        else if (k == "--sustained") { double v; next(v); a.sustained = static_cast<int>(v); }
        else if (k == "--windows") { double v; next(v); a.windows = static_cast<int>(v); }
        else if (k == "--gate") next(a.gate);
        else if (k == "--allow-overcommit") a.allow_overcommit = true;
        else if (k == "--cooldown") next(a.cooldown);
        else if (k == "--tag") nexts(a.tag);
        else if (k == "--out") nexts(a.out);
        else if (k == "--assets") nexts(a.assets);
        else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown argument " + k);
    }
    if (a.ref_seconds <= 0) a.ref_seconds = a.seconds / 2;
    if (a.mode != "env" && a.mode != "bare") throw std::runtime_error("--mode must be env or bare");
    return a;
}

// ---------------------------------------------------------------- workloads

// One timed window of the full environment on T threads. Returns env-steps/s.
double run_env(const EnvConfig& cfg, int threads, int envs_per_thread, double seconds, bool overcommit) {
    const int n = threads * envs_per_thread;
    VecEnv v(cfg, n, 1234, threads, overcommit);
    v.reset();
    PCG64 r(99);
    std::vector<double> a(static_cast<std::size_t>(n) * kNAct);
    auto fill = [&] { for (auto& x : a) x = r.uniform(-1.0, 1.0); };
    for (int k = 0; k < 200; ++k) { fill(); v.step(a.data()); }   // warm-up outside the window
    long steps = 0;
    const auto t0 = clk::now();
    while (true) {
        fill();
        v.step(a.data());
        ++steps;
        if ((steps & 0xF) == 0 && seconds_since(t0) > seconds) break;
    }
    const double el = seconds_since(t0);
    return static_cast<double>(steps) * n / el;
}

// bench.py's worker: bare mj_step, one model copy per thread, no environment.
double run_bare(const mjModel* proto, const ModelIndex& ix, int threads, double seconds) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<double> rates(static_cast<std::size_t>(threads), 0.0);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            ModelPtr m = copy_model(proto);
            DataPtr d = make_data(m.get());
            mj_resetDataKeyframe(m.get(), d.get(), ix.stand_key);
            mj_forward(m.get(), d.get());
            std::vector<double> nominal(d->ctrl, d->ctrl + kNAct);
            PCG64 r(1234 + static_cast<std::uint64_t>(t));
            auto noisy_ctrl = [&] { for (int i = 0; i < kNAct; ++i) d->ctrl[i] = nominal[static_cast<std::size_t>(i)] + r.normal(0.0, 0.05); };
            for (int k = 0; k < 200; ++k) { noisy_ctrl(); mj_step(m.get(), d.get()); }
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            long steps = 0;
            const auto t0 = clk::now();
            while (true) {
                noisy_ctrl();
                for (int k = 0; k < kSubsteps; ++k) mj_step(m.get(), d.get());
                ++steps;
                if (trunk_height(m.get(), d.get(), ix) < kFallHeight) {
                    mj_resetDataKeyframe(m.get(), d.get(), ix.stand_key);
                    mj_forward(m.get(), d.get());
                }
                if ((steps & 0xFF) == 0 && seconds_since(t0) > seconds) break;
            }
            rates[static_cast<std::size_t>(t)] = static_cast<double>(steps) / seconds_since(t0);
        });
    }
    while (ready.load() < threads) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    for (auto& th : pool) th.join();
    double sum = 0.0;
    for (double r : rates) sum += r;
    return sum;
}

struct Workload {
    Args a;
    EnvConfig cfg;
    ModelPtr proto;
    ModelIndex ix;
    double run(int threads, double seconds) {
        if (a.mode == "bare") return run_bare(proto.get(), ix, threads, seconds);
        return run_env(cfg, threads, a.envs_per_thread, seconds, a.allow_overcommit);
    }
};

// ---------------------------------------------------------------- analysis

// Longest monotonic run in the reference sequence: (length, fractional span, direction).
struct Trend { int run = 1; double span = 0.0; std::string dir; };
Trend reference_trend(const std::vector<double>& refs) {
    Trend best;
    for (int pass = 0; pass < 2; ++pass) {
        const bool rising = pass == 0;
        std::size_t i = 0;
        while (i + 1 < refs.size()) {
            std::size_t j = i;
            while (j + 1 < refs.size() && (rising ? refs[j + 1] > refs[j] : refs[j + 1] < refs[j])) ++j;
            if (j > i) {
                const double span = std::fabs(refs[j] - refs[i]) / std::min(refs[i], refs[j]);
                const int len = static_cast<int>(j - i + 1);
                if (len > best.run || (len == best.run && span > best.span)) best = {len, span, rising ? "rising" : "falling"};
            }
            i = std::max(j, i + 1);
        }
    }
    return best;
}

std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o + "\"";
}
std::string json_list(const std::vector<std::string>& v) {
    std::string o = "[";
    for (std::size_t i = 0; i < v.size(); ++i) o += (i ? ", " : "") + v[i];
    return o + "]";
}
std::string num(double v) { char b[64]; std::snprintf(b, sizeof b, "%.6g", v); return b; }
std::string timestamp() {
    char b[64];
    const std::time_t t = std::time(nullptr);
    std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
    return b;
}

std::string common_json(const Args& a, const BoxState& box) {
    std::vector<std::string> w;
    for (const auto& s : box.warnings) w.push_back(json_str(s));
    return "  \"backend\": \"cpp\",\n  \"mode\": " + json_str(a.mode) + ",\n  \"variant\": " + json_str(a.variant) +
           ",\n  \"reward\": " + json_str(a.reward) + ",\n  \"envs_per_thread\": " + std::to_string(a.envs_per_thread) +
           ",\n  \"cores\": " + std::to_string(std::thread::hardware_concurrency()) +
           ",\n  \"timestamp\": " + json_str(timestamp()) + ",\n  \"startup_warnings\": " + json_list(w) +
           ",\n  \"clean\": " + (box.warnings.empty() ? "true" : "false") + ",\n  \"load1\": " + num(box.load1);
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    try { a = parse(argc, argv); } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); usage(); return 2; }
    for (int t : a.threads) if (t > kMaxWorkers && !a.allow_overcommit) {
        std::fprintf(stderr, "refusing %d threads: the budget on the machine this was written on is %d of 14 cores, "
                             "and every number in the README was measured inside it. --allow-overcommit if you own the box.\n", t, kMaxWorkers);
        return 2;
    }
    if (a.sustained > kMaxWorkers && !a.allow_overcommit) { std::fprintf(stderr, "refusing --sustained %d without --allow-overcommit\n", a.sustained); return 2; }

    check_library_version();
    Workload w;
    w.a = a;
    w.cfg.variant = a.variant;
    w.cfg.reward = parse_reward(a.reward);
    w.cfg.assets_dir = a.assets.empty() ? default_assets_dir() : a.assets;
    w.proto = load_variant(w.cfg.assets_dir, a.variant);
    w.ix = index_model(w.proto.get());

    if (a.cooldown > 0) {
        const auto t0 = clk::now();
        while (seconds_since(t0) < a.cooldown) {
            std::ifstream la("/proc/loadavg"); double l = 0; la >> l;
            if (l <= 1.5) break;
            std::printf("  cooling: load average %.2f, %.0f s of patience left\n", l, a.cooldown - seconds_since(t0));
            std::this_thread::sleep_for(std::chrono::seconds(15));
        }
    }
    const BoxState box = check_box();
    for (const auto& s : box.warnings) std::printf("  WARNING  %s -- numbers below are contention, not capability\n", s.c_str());
    if (!box.warnings.empty()) std::printf("  Let the box idle until the load average is under 1.5 and re-run. --cooldown does the waiting.\n");

    std::string out = a.out;
    if (out.empty()) out = "runs/bench_cpp_" + a.mode + (a.tag.empty() ? "" : "_" + a.tag) + ".json";
    std::printf("\nvariant %s: nq %d nv %d nu %d, %.0f Hz physics, %d substeps per %d Hz control step\n",
                a.variant.c_str(), static_cast<int>(w.proto->nq), static_cast<int>(w.proto->nv), static_cast<int>(w.proto->nu), 1.0 / w.proto->opt.timestep, kSubsteps, kControlHz);
    std::printf("workload: %s%s\n", a.mode.c_str(), a.mode == "env" ? (", " + std::to_string(a.envs_per_thread) + " env(s) per thread, reward " + a.reward).c_str() : "");

    if (a.sustained > 0) {
        std::printf("\nsustained load: %d threads, %d x %.0f s with no pause (%.1f min total)\n",
                    a.sustained, a.windows, a.seconds, a.windows * a.seconds / 60.0);
        std::vector<double> rates;
        double peak = 0.0;
        std::printf("\n  %7s %9s %13s %9s\n", "window", "elapsed", "env-steps/s", "vs peak");
        for (int i = 0; i < a.windows; ++i) {
            const double r = w.run(a.sustained, a.seconds);
            rates.push_back(r);
            peak = std::max(peak, r);
            std::printf("  %7d %8.0fs %13.0f %+8.1f%%\n", i, (i + 1) * a.seconds, r, 100 * (r / peak - 1));
            std::fflush(stdout);
        }
        const int n = static_cast<int>(rates.size());
        double tail = 0.0, tail2 = 0.0;
        for (int i = std::max(0, n - 3); i < n; ++i) tail += rates[static_cast<std::size_t>(i)];
        tail /= std::min(3, n);
        for (int i = std::max(0, n - 2); i < n; ++i) tail2 += rates[static_cast<std::size_t>(i)];
        tail2 /= std::min(2, n);
        const double decay = tail / peak - 1;
        const double last_step = n > 1 ? rates[static_cast<std::size_t>(n - 1)] / rates[static_cast<std::size_t>(n - 2)] - 1 : 0.0;
        const bool settled = std::fabs(last_step) < 0.03;
        const int tail_n = std::max(2, n / 3);
        double band_lo = rates[static_cast<std::size_t>(n - tail_n)], band_hi = band_lo;
        for (int i = n - tail_n; i < n; ++i) { band_lo = std::min(band_lo, rates[static_cast<std::size_t>(i)]); band_hi = std::max(band_hi, rates[static_cast<std::size_t>(i)]); }
        std::printf("\n  peak %.0f   sustained (last 3 windows) %.0f   %+.1f%%\n", peak, tail, 100 * decay);
        std::printf("  last 2 windows %.0f   final window vs previous %+.1f%%\n", tail2, 100 * last_step);
        std::printf("  plateau over the last %d windows: %.0f to %.0f, a %.0f%% band\n", tail_n, band_lo, band_hi, 100 * (band_hi - band_lo) / band_lo);
        if (!settled) std::printf("  Still %s at the last window, so this is not a steady state. Re-run with --windows %d.\n  Budget on %.0f, the low end of the band.\n",
                                  last_step < 0 ? "falling" : "rising", a.windows + 6, band_lo);
        std::vector<std::string> rs;
        for (double r : rates) rs.push_back(num(r));
        std::ofstream f(out);
        f << "{\n" << common_json(a, box) << ",\n  \"sustained_threads\": " << a.sustained << ",\n  \"window_s\": " << num(a.seconds)
          << ",\n  \"rates\": " << json_list(rs) << ",\n  \"peak\": " << num(peak) << ",\n  \"sustained\": " << num(tail)
          << ",\n  \"sustained_last2\": " << num(tail2) << ",\n  \"settled\": " << (settled ? "true" : "false")
          << ",\n  \"band_windows\": " << tail_n << ",\n  \"band_low\": " << num(band_lo) << ",\n  \"band_high\": " << num(band_hi)
          << ",\n  \"decay\": " << num(decay) << "\n}\n";
        std::printf("\n  wrote %s\n", out.c_str());
        return 0;
    }

    std::printf("%.0f s per configuration, %.0f s reference windows, %u cores present, %d permitted\n",
                a.seconds, a.ref_seconds, std::thread::hardware_concurrency(), kMaxWorkers);
    const double ref0 = w.run(1, a.ref_seconds);
    std::printf("\nreference (1 thread, before sweep): %.0f env-steps/s\n\n", ref0);
    std::printf("  %7s %13s %10s %9s %6s %11s %8s\n", "threads", "env-steps/s", "per thr", "speedup", "eff", "ref after", "drift");
    std::vector<std::string> rows;
    std::vector<double> refs = {ref0};
    double baseline = -1.0, best = 0.0, worst = 0.0;
    for (int t : a.threads) {
        const double r = w.run(t, a.seconds);
        const double ref = w.run(1, a.ref_seconds);
        if (baseline < 0) baseline = (t == 1) ? r : ref0;
        const double speedup = r / baseline, eff = speedup / t, drift = ref / ref0 - 1;
        worst = std::max(worst, std::fabs(drift));
        best = std::max(best, r);
        refs.push_back(ref);
        std::printf("  %7d %13.0f %10.0f %8.2fx %5.0f%% %11.0f %+7.1f%%%s\n", t, r, r / t, speedup, 100 * eff, ref, 100 * drift,
                    std::fabs(drift) > 0.10 ? "  <-- reference moved" : "");
        std::fflush(stdout);
        rows.push_back("{\"threads\": " + std::to_string(t) + ", \"envs\": " + std::to_string(t * a.envs_per_thread) +
                       ", \"env_steps_per_s\": " + num(r) + ", \"per_thread\": " + num(r / t) + ", \"speedup\": " + num(speedup) +
                       ", \"efficiency\": " + num(eff) + ", \"ref_after\": " + num(ref) + ", \"ref_drift\": " + num(drift) + "}");
    }
    const Trend tr = reference_trend(refs);
    const bool trending = tr.run >= 3 && tr.span > 0.04;
    std::printf("\n");
    if (worst > 0.10) std::printf("  UNSTABLE: the 1-thread reference moved by up to %.0f%% across the sweep. Do not quote these numbers.\n", 100 * worst);
    else if (trending) std::printf("  TRENDING: the reference moved %s across %d consecutive windows, spanning %.1f%%. The rows are not comparable. Let it idle and re-run.\n", tr.dir.c_str(), tr.run, 100 * tr.span);
    else std::printf("  Stable: the 1-thread reference held to within %.0f%% across the whole sweep, with no trend.\n", 100 * worst);
    if (!box.warnings.empty()) std::printf("  Started dirty. Recorded in the JSON as clean=false.\n");
    std::printf("\n  gate: %.0f env-steps/s   measured: %.0f   %s\n", a.gate, best, best >= a.gate ? "PASS" : "FAIL");
    std::printf("\n  wall-clock at %.0f env-steps/s:\n", best);
    const struct { const char* label; double n; } budgets[] = {
        {"10M  (smoke / hyperparameter probe)", 10e6}, {"50M  (stand + push recovery, expected)", 50e6},
        {"100M (stand, generous)", 100e6}, {"400M (walking gait, upstream-scale)", 400e6}};
    for (const auto& b : budgets) {
        const double h = b.n / best / 3600;
        std::printf("    %-40s %6.2f h%s\n", b.label, h, h <= 2 ? "" : ("  -> " + std::to_string(static_cast<int>(std::ceil(h / 2))) + " chunks of <=2 h").c_str());
    }
    const bool stable = worst <= 0.10 && !trending && box.warnings.empty();
    std::ofstream f(out);
    f << "{\n" << common_json(a, box) << ",\n  \"seconds\": " << num(a.seconds) << ",\n  \"ref_seconds\": " << num(a.ref_seconds)
      << ",\n  \"ref_before\": " << num(ref0) << ",\n  \"baseline\": " << num(baseline) << ",\n  \"rows\": " << json_list(rows)
      << ",\n  \"worst_ref_drift\": " << num(worst) << ",\n  \"ref_trend\": {\"run\": " << tr.run << ", \"span\": " << num(tr.span)
      << ", \"direction\": " << json_str(tr.dir) << "},\n  \"trending\": " << (trending ? "true" : "false")
      << ",\n  \"stable\": " << (stable ? "true" : "false") << ",\n  \"gate\": " << num(a.gate)
      << ",\n  \"pass\": " << (best >= a.gate ? "true" : "false") << "\n}\n";
    std::printf("\n  wrote %s\n", out.c_str());
    return 0;
}
