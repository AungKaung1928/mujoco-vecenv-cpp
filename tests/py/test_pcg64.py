"""The C++ PCG64 core is numpy's PCG64 core: hand it numpy's internal state
and the next thousand raw outputs, doubles and uniforms are identical."""
import numpy as np
import pytest

MASK = (1 << 64) - 1


def _cpp_from(bg, cpp):
    st = bg.state["state"]
    s, inc = int(st["state"]), int(st["inc"])
    return cpp.PCG64.from_words(s >> 64, s & MASK, inc >> 64, inc & MASK)


@pytest.mark.parametrize("seed", [0, 123, 2**40 + 7])
def test_raw_stream_matches_numpy(seed, cpp):
    bg = np.random.PCG64(seed)
    c = _cpp_from(bg, cpp)
    assert np.array_equal(c.random_raw(1000), bg.random_raw(1000))


@pytest.mark.parametrize("seed", [1, 4242])
def test_random_and_uniform_match_numpy(seed, cpp):
    bg = np.random.PCG64(seed)
    c = _cpp_from(bg, cpp)
    g = np.random.Generator(bg)
    assert np.array_equal(c.random(1000), g.random(1000))
    bg2 = np.random.PCG64(seed)
    c2 = _cpp_from(bg2, cpp)
    g2 = np.random.Generator(bg2)
    assert np.array_equal(c2.uniform(-0.3, 0.7, 1000), g2.uniform(-0.3, 0.7, 1000))


def test_integer_seeding_is_not_numpy_seeding(cpp):
    """Documented, not a bug: same integer, different stream."""
    c = cpp.PCG64(7)
    g = np.random.default_rng(7)
    assert not np.array_equal(c.random(8), g.random(8))
