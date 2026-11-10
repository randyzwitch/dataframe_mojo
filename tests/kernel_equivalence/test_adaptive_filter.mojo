"""Adaptive filter trials preserve a valid permutation over long scans."""
# contract-group: expr.vectors
from std.testing import assert_equal, assert_true

from dataframe.vector.adaptive_filter import AdaptiveFilter


def test_adaptive_filter_long_scan_keeps_swap_indices_in_bounds() raises:
    for conjuncts in [2, 3, 8]:
        var adaptive = AdaptiveFilter(conjuncts, False)
        var tried_swap = False
        # Small fixtures never reached the random draw that selected a
        # nonexistent adjacent pair during a 50-million-row filtered scan.
        for _ in range(10000):
            adaptive._adapt(1.0)
            assert_true(0 <= adaptive.swap_idx < conjuncts - 1)
            tried_swap = tried_swap or adaptive.observe
        assert_true(tried_swap)
        var seen = List[Bool]()
        for _ in range(conjuncts):
            seen.append(False)
        for index in adaptive.permutation:
            assert_true(0 <= index < conjuncts)
            assert_true(not seen[index])
            seen[index] = True
        assert_equal(len(adaptive.permutation), conjuncts)


def test_adaptive_filter_random_stream_is_owned_and_copyable() raises:
    var first = AdaptiveFilter(8, False)
    var second = AdaptiveFilter(8, False)
    var unrelated = AdaptiveFilter(3, False)
    var low_seen = False
    var high_seen = False
    for _ in range(4096):
        _ = unrelated._random(-100, 100)
        var value = first._random(-3, 3)
        assert_equal(value, second._random(-3, 3))
        assert_true(-3 <= value <= 3)
        low_seen = low_seen or value == -3
        high_seen = high_seen or value == 3
    assert_true(low_seen and high_seen)
    var copied = first.copy()
    for _ in range(100):
        assert_equal(first._random(0, 699), copied._random(0, 699))
    assert_equal(first._random(7, 7), 7)
