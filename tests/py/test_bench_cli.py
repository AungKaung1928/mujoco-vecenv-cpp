"""bench_vecenv refuses arguments its statistics cannot handle (exit 2), instead of writing them.

--windows 1 used to read rates[-1] and write band_low 1.6e-322 with exit 0; --windows 0 segfaulted.
"""
import os
import subprocess

import pytest

BENCH = os.path.join(os.path.dirname(__file__), "..", "..", "build", "bench_vecenv")


@pytest.mark.parametrize("windows", ["0", "1"])
def test_too_few_windows_is_refused(windows, tmp_path):
    if not os.path.exists(BENCH):
        pytest.skip("build/bench_vecenv not built")
    out = tmp_path / "b.json"
    r = subprocess.run([BENCH, "--mode", "bare", "--sustained", "1", "--windows", windows, "--seconds", "1",
                        "--out", str(out)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2, (r.returncode, r.stderr[-300:])
    assert "--windows must be >= 2" in r.stderr
    assert not out.exists()
