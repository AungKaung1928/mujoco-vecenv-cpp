// PCG64 (XSL-RR, 128-bit state), the generator numpy's default_rng uses.
//
// The core stream is identical to numpy's given the same (state, inc), and
// so is next_double / uniform, which numpy computes as
// (next_u64 >> 11) * 2^-53 and lo + (hi - lo) * next_double. A test hands
// numpy's internal state to this class and checks 1,000 outputs of each.
//
// What is NOT numpy-equivalent, on purpose: seeding from a 64-bit integer
// (numpy runs SeedSequence, which is not worth reimplementing), normal()
// (numpy uses a ziggurat, this is Marsaglia polar) and integers()/choice.
// So an Env seeded with the same integer as the Python env does NOT draw the
// same pushes. The contract test therefore injects the Python env's drawn
// values instead of trying to reproduce the draw; see EpisodeParams.
#pragma once

#include <cstdint>

namespace vecenv {

class PCG64 {
public:
    using u128 = unsigned __int128;

    explicit PCG64(std::uint64_t seed);
    PCG64(u128 state, u128 inc) : state_(state), inc_(inc) {}
    static PCG64 from_words(std::uint64_t state_hi, std::uint64_t state_lo,
                            std::uint64_t inc_hi, std::uint64_t inc_lo);

    std::uint64_t next_u64();
    // numpy: random_standard_uniform = (next64 >> 11) * (1 / 2^53)
    double next_double() { return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0); }
    // numpy: Generator.uniform(lo, hi) = lo + (hi - lo) * next_double
    double uniform(double lo, double hi) { return lo + (hi - lo) * next_double(); }
    // Marsaglia polar. Not numpy's ziggurat.
    double normal(double mean, double sd);
    // Inclusive [lo, hi]. Rejection on a bit mask. Not numpy's algorithm.
    std::int64_t integers(std::int64_t lo, std::int64_t hi);

    u128 state() const { return state_; }
    u128 inc() const { return inc_; }

private:
    static constexpr u128 kMult = (static_cast<u128>(2549297995355413924ULL) << 64)
                                  | 4865540595714422341ULL;
    u128 state_;
    u128 inc_;
    bool has_spare_ = false;
    double spare_ = 0.0;
};

}  // namespace vecenv
