#include <cmath>
#include <vector>

#include "doctest.h"
#include "vecenv/numpy_compat.hpp"

using vecenv::np::clip;
using vecenv::np::mean;
using vecenv::np::pairwise_sum;
using vecenv::np::py_pow;

TEST_CASE("pairwise_sum below eight elements is a plain left-to-right sum") {
    const double a[5] = {1e16, 1.0, -1e16, 1.0, 1.0};
    double s = 0.0;
    for (double v : a) s += v;
    CHECK(pairwise_sum(a, 5) == s);
}

TEST_CASE("pairwise_sum at fourteen elements is the eight-accumulator tree numpy uses") {
    // 1e16 swallows a lone +1 (the spacing there is 2), so the two orders
    // disagree by construction: left to right loses all seven ones and ends
    // at 0; the tree pairs them first and keeps six of them.
    const double a[14] = {1e16, 1, 1, 1, 1, 1, 1, 1, -1e16, 0, 0, 0, 0, 0};
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (int i = 8; i < 14; ++i) res += a[i];
    CHECK(pairwise_sum(a, 14) == res);
    CHECK(res == 6.0);
    CHECK(mean(a, 14) == res / 14.0);
    double seq = 0.0;
    for (double v : a) seq += v;
    CHECK(seq == 0.0);
    CHECK(seq != res);
}

TEST_CASE("pairwise_sum above 128 recurses and stays close to an extended-precision sum") {
    std::vector<double> a(1000);
    long double ref = 0.0L;
    for (int i = 0; i < 1000; ++i) { a[static_cast<std::size_t>(i)] = std::sin(i) * 1e3; ref += a[static_cast<std::size_t>(i)]; }
    CHECK(std::fabs(pairwise_sum(a.data(), a.size()) - static_cast<double>(ref)) < 1e-9);
}

TEST_CASE("py_pow goes through libm and squares exactly on simple inputs") {
    CHECK(py_pow(3.0, 2.0) == 9.0);
    CHECK(py_pow(0.5, 2.0) == 0.25);
    CHECK(py_pow(2.0, 10.0) == 1024.0);
}

TEST_CASE("clip is min(max(x, lo), hi)") {
    CHECK(clip(5.0, -1.0, 1.0) == 1.0);
    CHECK(clip(-5.0, -1.0, 1.0) == -1.0);
    CHECK(clip(0.3, -1.0, 1.0) == 0.3);
}
