#include <cmath>
#include <set>

#include "doctest.h"
#include "vecenv/rng.hpp"

using vecenv::PCG64;

TEST_CASE("same seed, same stream; different seed, different stream") {
    PCG64 a(7), b(7), c(8);
    bool same = true, diff = false;
    for (int i = 0; i < 100; ++i) {
        const auto x = a.next_u64(), y = b.next_u64(), z = c.next_u64();
        same &= (x == y);
        diff |= (x != z);
    }
    CHECK(same);
    CHECK(diff);
}

TEST_CASE("next_double is in [0, 1) and uniform respects its bounds") {
    PCG64 r(3);
    for (int i = 0; i < 100000; ++i) {
        const double d = r.next_double();
        CHECK(d >= 0.0);
        CHECK(d < 1.0);
        const double u = r.uniform(-0.3, 0.7);
        CHECK(u >= -0.3);
        CHECK(u <= 0.7);
    }
}

TEST_CASE("integers covers the inclusive range and nothing outside it") {
    PCG64 r(11);
    std::set<std::int64_t> seen;
    for (int i = 0; i < 5000; ++i) {
        const auto v = r.integers(25, 29);
        CHECK(v >= 25);
        CHECK(v <= 29);
        seen.insert(v);
    }
    CHECK(seen.size() == 5);
    CHECK(r.integers(4, 4) == 4);
}

TEST_CASE("normal has the requested moments to within sampling error") {
    PCG64 r(5);
    const int n = 200000;
    double s = 0.0, s2 = 0.0;
    for (int i = 0; i < n; ++i) { const double x = r.normal(0.5, 2.0); s += x; s2 += x * x; }
    const double m = s / n, var = s2 / n - m * m;
    CHECK(std::fabs(m - 0.5) < 0.02);
    CHECK(std::fabs(var - 4.0) < 0.08);
}

TEST_CASE("from_words reproduces a state exactly") {
    PCG64 a(99);
    a.next_u64();
    const auto st = a.state(), inc = a.inc();
    PCG64 b = PCG64::from_words(static_cast<std::uint64_t>(st >> 64), static_cast<std::uint64_t>(st),
                                static_cast<std::uint64_t>(inc >> 64), static_cast<std::uint64_t>(inc));
    for (int i = 0; i < 50; ++i) CHECK(a.next_u64() == b.next_u64());
}
