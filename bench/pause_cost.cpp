// How long a worker spins before it blocks: kSpinIterations pause instructions,
// timed 50 times on one thread. One pause costs ~10 cycles on older Intel cores and
// ~100+ on recent ones, so the spin window depends on the CPU, not only the count.
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "vecenv/vecenv.hpp"

int main() {
    double lo = 1e18, hi = 0.0;
    for (int r = 0; r < 50; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < vecenv::kSpinIterations; ++k) _mm_pause();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        lo = std::min(lo, us);
        hi = std::max(hi, us);
    }
    std::printf("%d pause: min %.1f us, max %.1f us (50 runs)\n", vecenv::kSpinIterations, lo, hi);
}
