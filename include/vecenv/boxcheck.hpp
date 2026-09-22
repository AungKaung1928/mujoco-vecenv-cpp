// Is anything else using this machine right now? Same two checks bench.py in
// microduck-rl makes: the 1-minute load average, and a sampled (not lifetime)
// CPU rate for every other process, from two reads of /proc/<pid>/stat a
// fraction of a second apart. Percentages are of one core.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace vecenv {

struct BoxState {
    double load1 = 0.0;
    std::vector<std::pair<double, std::string>> busy;   // (% of a core, comm), > 20%
    std::vector<std::string> warnings;                  // empty means clean
};

BoxState check_box(double sample_seconds = 0.4);

}  // namespace vecenv
