"""Hash-aggregate sampling must not alias repeated input periods."""
from std.testing import TestSuite, assert_false, assert_true
from dataframe import Column, Series
from dataframe.hash_agg import _moderate_groups


def keys(rows: Int, groups: Int) raises -> Series:
    var values = List[Int64](capacity=rows)
    for i in range(rows):
        values.append(Int64((i * 37) % groups - groups // 2))
    return Series("k", Column[Int64](values^))


def test_periodic_half_frames_do_not_look_moderate() raises:
    # With evenly spaced samples, positions k and k + sample/2 hold the
    # same key: the observed cardinality is exactly half the sample. This
    # wrongly accepts hundreds of thousands of distinct keys.
    for rows in [262_144, 1_000_000, 1_048_576]:
        assert_false(_moderate_groups(keys(rows, rows // 2)))


def test_moderate_domains_remain_eligible() raises:
    for groups in [10, 1000, 20_000]:
        assert_true(_moderate_groups(keys(200_003, groups)))


def test_empty_full_samples_and_unique_keys() raises:
    assert_false(
        _moderate_groups(Series("empty", Column[Int64](List[Int64]())))
    )
    assert_true(_moderate_groups(keys(1009, 17)))
    assert_false(_moderate_groups(keys(1009, 1009)))
    assert_false(_moderate_groups(keys(70_001, 70_001)))


def test_exact_cutoff_for_full_samples() raises:
    # Below the budget, strata contain one row each. Early bounds must
    # produce exactly the original strict 85% predicate, including ties.
    assert_true(_moderate_groups(keys(2000, 1699)))
    assert_false(_moderate_groups(keys(2000, 1700)))
    assert_false(_moderate_groups(keys(2000, 1701)))
    assert_true(_moderate_groups(keys(65_536, 55_705)))
    assert_false(_moderate_groups(keys(65_536, 55_706)))


def test_offsets_and_nonmatching_chunks_use_the_same_sample() raises:
    var source = keys(262_158, 131_072)
    var sliced = source.slice(7, 262_144)
    var chunks = Series._from_chunks(
        [
            sliced.slice(0, 9),
            sliced.slice(9, 131_073),
            sliced.slice(131_082, 131_062),
        ]
    )
    assert_false(_moderate_groups(sliced))
    assert_false(_moderate_groups(chunks))
    var moderate = keys(200_017, 17_000).slice(7, 200_003)
    assert_true(_moderate_groups(moderate))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
