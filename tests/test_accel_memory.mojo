"""Pure allocation accounting and rejection gates, requiring no GPU SDK."""
from std.testing import TestSuite, assert_equal, assert_raises
from dataframe.accel_memory import estimate_memory


def test_sliced_bitmap_and_workspace_estimate() raises:
    var memory = estimate_memory(257, 4, True, 3, 2)
    assert_equal(memory.input_bytes, 1061)
    assert_equal(memory.blocks, 2)
    assert_equal(memory.workspace_bytes, 32)
    assert_equal(memory.output_bytes, 16)
    assert_equal(memory.peak_bytes, 1109)
    assert_equal(memory.download_bytes, 32)
    assert_equal(memory.launches, 4)
    memory.require_budget(1109)
    with assert_raises(contains="exceeds budget 1108"):
        memory.require_budget(1108)


def test_empty_and_maximum_grid_accounting() raises:
    var empty = estimate_memory(0, 8, True, 7, 3)
    assert_equal(empty.input_bytes, 0)
    assert_equal(empty.peak_bytes, 32)
    assert_equal(empty.blocks, 1)
    assert_equal(empty.download_bytes, 48)
    var large = estimate_memory(1000000, 8, False, 0, 1)
    assert_equal(large.blocks, 1024)
    assert_equal(large.workspace_bytes, 16384)
    assert_equal(large.peak_bytes, 8016400)


def test_accounting_overflow_and_invalid_shapes() raises:
    with assert_raises(contains="overflows Int"):
        _ = estimate_memory(Int.MAX, 8, True, 7, 1)
    with assert_raises(contains="overflows Int"):
        _ = estimate_memory(1, 4, False, 0, Int.MAX)
    with assert_raises(contains="Invalid accelerator memory shape"):
        _ = estimate_memory(-1, 4, False, 0, 1)
    with assert_raises(contains="Invalid accelerator memory shape"):
        _ = estimate_memory(0, 2, False, 0, 1)
    with assert_raises(contains="Invalid accelerator memory shape"):
        _ = estimate_memory(0, 4, True, 8, 1)
    with assert_raises(contains="Invalid accelerator memory shape"):
        _ = estimate_memory(0, 4, False, 0, 0)
    with assert_raises(contains="must be nonnegative"):
        estimate_memory(0, 4, False, 0, 1).require_budget(-1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
