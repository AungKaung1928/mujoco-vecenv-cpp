#include "vecenv/rng.hpp"

#include <cmath>
#include <stdexcept>

namespace vecenv {

namespace {
std::uint64_t splitmix64(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
std::uint64_t rotr64(std::uint64_t v, unsigned rot) {
    return (v >> rot) | (v << ((-rot) & 63));
}
}  // namespace

PCG64::PCG64(std::uint64_t seed) : state_(0), inc_(0) {
    // Expand the 64-bit seed into the 256 bits PCG's setseq seeding takes,
    // then seed the way pcg_setseq_128_srandom_r does. Not numpy's
    // SeedSequence: same integer, different stream. See the header.
    std::uint64_t x = seed;
    const std::uint64_t w0 = splitmix64(x), w1 = splitmix64(x), w2 = splitmix64(x), w3 = splitmix64(x);
    const u128 initstate = (static_cast<u128>(w0) << 64) | w1;
    const u128 initseq = (static_cast<u128>(w2) << 64) | w3;
    state_ = 0;
    inc_ = (initseq << 1) | 1u;
    state_ = state_ * kMult + inc_;
    state_ += initstate;
    state_ = state_ * kMult + inc_;
}

PCG64 PCG64::from_words(std::uint64_t state_hi, std::uint64_t state_lo,
                        std::uint64_t inc_hi, std::uint64_t inc_lo) {
    return PCG64((static_cast<u128>(state_hi) << 64) | state_lo,
                 (static_cast<u128>(inc_hi) << 64) | inc_lo);
}

std::uint64_t PCG64::next_u64() {
    // pcg_setseq_128_xsl_rr_64_random_r: step, then output from the NEW state.
    state_ = state_ * kMult + inc_;
    const std::uint64_t hi = static_cast<std::uint64_t>(state_ >> 64);
    const std::uint64_t lo = static_cast<std::uint64_t>(state_);
    const unsigned rot = static_cast<unsigned>(state_ >> 122);
    return rotr64(hi ^ lo, rot);
}

double PCG64::normal(double mean, double sd) {
    if (has_spare_) {
        has_spare_ = false;
        return mean + sd * spare_;
    }
    double u, v, s;
    do {
        u = uniform(-1.0, 1.0);
        v = uniform(-1.0, 1.0);
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    const double f = std::sqrt(-2.0 * std::log(s) / s);
    spare_ = v * f;
    has_spare_ = true;
    return mean + sd * (u * f);
}

std::int64_t PCG64::integers(std::int64_t lo, std::int64_t hi) {
    if (hi < lo) throw std::invalid_argument("integers: hi < lo");
    const std::uint64_t range = static_cast<std::uint64_t>(hi - lo) + 1u;
    if (range == 1u) return lo;
    std::uint64_t mask = range - 1u;
    mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4;
    mask |= mask >> 8; mask |= mask >> 16; mask |= mask >> 32;
    std::uint64_t v;
    do { v = next_u64() & mask; } while (v >= range);
    return lo + static_cast<std::int64_t>(v);
}

}  // namespace vecenv
