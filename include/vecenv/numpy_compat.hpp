// Arithmetic that has to round exactly the way the Python reference rounds.
//
// The contract test compares this environment against the Python one to the
// last bit. That only works if the handful of reductions the reward uses are
// evaluated in the same order numpy evaluates them. numpy does not sum left to
// right: for 8 <= n <= 128 it keeps eight partial sums and combines them as a
// tree, and np.mean is that sum divided by n. Fourteen joints is inside that
// regime, so a naive loop here would differ in the last bit about a third of
// the time (measured: sequential summation matched np.mean on 15,831 of 20,000
// random 14-vectors; this function matched on 20,000 of 20,000).
//
// Likewise Python's `x ** 2` on a float calls libm pow(x, 2.0), which is not
// the same as x * x (measured: 1,621 mismatches in 2,000,000). The compiler
// folds pow(x, 2.0) into x * x unless stopped, so py_pow goes through a
// volatile to force the library call.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace vecenv {
namespace np {

// numpy/core/src/umath/loops_utils.h.src :: pairwise_sum, stride 1.
inline double pairwise_sum(const double* a, std::size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// np.mean of a 1-D float64 array: add.reduce starts from the identity 0 and
// pairwise-sums all n elements (measured, see the header comment), then
// divides by n as a double.
inline double mean(const double* a, std::size_t n) {
    return pairwise_sum(a, n) / static_cast<double>(n);
}

// Python float.__pow__ -> libm pow. The volatile stops the pow(x, 2) -> x*x fold.
inline double py_pow(double x, double y) {
    volatile double yy = y;
    return std::pow(x, yy);
}

// np.clip(x, lo, hi) == minimum(maximum(x, lo), hi).
inline double clip(double x, double lo, double hi) {
    return std::min(std::max(x, lo), hi);
}

}  // namespace np
}  // namespace vecenv
