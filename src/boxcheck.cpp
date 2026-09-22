#include "vecenv/boxcheck.hpp"

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <thread>

namespace vecenv {

namespace {
// utime + stime in clock ticks and the command name, for every readable pid.
std::map<int, std::pair<long, std::string>> cpu_ticks() {
    std::map<int, std::pair<long, std::string>> out;
    std::error_code ec;
    for (const auto& ent : std::filesystem::directory_iterator("/proc", ec)) {
        const std::string name = ent.path().filename().string();
        if (name.empty() || !std::all_of(name.begin(), name.end(), ::isdigit)) continue;
        std::ifstream st(ent.path() / "stat");
        std::string line;
        if (!std::getline(st, line)) continue;
        // comm may contain spaces and parentheses: split on the LAST ") ".
        const auto pos = line.rfind(") ");
        if (pos == std::string::npos) continue;
        std::istringstream rest(line.substr(pos + 2));
        std::string f;
        long utime = -1, stime = -1;
        for (int k = 0; rest >> f; ++k) {
            if (k == 11) utime = std::stol(f);
            if (k == 12) { stime = std::stol(f); break; }
        }
        if (utime < 0 || stime < 0) continue;
        std::ifstream cm(ent.path() / "comm");
        std::string comm;
        std::getline(cm, comm);
        out[std::stoi(name)] = {utime + stime, comm};
    }
    return out;
}
}  // namespace

BoxState check_box(double sample_seconds) {
    BoxState s;
    {
        std::ifstream la("/proc/loadavg");
        la >> s.load1;
    }
    const long hz = sysconf(_SC_CLK_TCK);
    const auto before = cpu_ticks();
    std::this_thread::sleep_for(std::chrono::duration<double>(sample_seconds));
    const auto after = cpu_ticks();
    const int me = static_cast<int>(getpid());
    for (const auto& [pid, ts] : after) {
        if (pid == me) continue;
        const auto it = before.find(pid);
        if (it == before.end()) continue;
        const double pct = 100.0 * static_cast<double>(ts.first - it->second.first)
                           / static_cast<double>(hz) / sample_seconds;
        if (pct > 20.0) s.busy.emplace_back(pct, ts.second);
    }
    std::sort(s.busy.begin(), s.busy.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (!s.busy.empty()) {
        double share = 0.0;
        std::string who;
        for (std::size_t i = 0; i < s.busy.size(); ++i) {
            share += s.busy[i].first;
            if (i < 4) {
                char buf[128];
                std::snprintf(buf, sizeof buf, "%s%s(%.0f%% of a core)", i ? ", " : "",
                              s.busy[i].second.c_str(), s.busy[i].first);
                who += buf;
            }
        }
        const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        char buf[64];
        std::snprintf(buf, sizeof buf, " -- %.0f%% of the machine", share / static_cast<double>(ncpu > 0 ? ncpu : 1));
        s.warnings.push_back("CPU in use by " + who + buf);
    }
    if (s.load1 > 1.5) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "1-min load average %.2f", s.load1);
        s.warnings.emplace_back(buf);
    }
    return s;
}

}  // namespace vecenv
