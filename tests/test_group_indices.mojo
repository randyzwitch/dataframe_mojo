"""Row-index grouping: the caller decides what to build from the groups."""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, Series, col


def sample() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "region",
                Column[String](["east", "west", "east", "north", "west"]),
            ),
            Series("amount", Column[Int64]([1, 2, 3, 4, 5])),
        ]
    )


def test_groups_are_in_first_occurrence_order() raises:
    var groups = sample().group_indices("region")
    assert_equal(groups.count(), 3)
    assert_equal(groups.height(), 5)
    # east appears first, then west, then north.
    assert_equal(groups.ids(), [0, 1, 0, 2, 1])
    assert_equal(groups.representatives(), [0, 1, 3])
    assert_equal(groups.sizes(), [2, 2, 1])


def test_rows_and_all_rows_agree() raises:
    var groups = sample().group_indices("region")
    assert_equal(groups.rows(0), [0, 2])
    assert_equal(groups.rows(1), [1, 4])
    assert_equal(groups.rows(2), [3])
    var everything = groups.all_rows()
    assert_equal(len(everything), 3)
    for g in range(groups.count()):
        assert_equal(everything[g], groups.rows(g))


def test_caller_builds_its_own_sub_frames() raises:
    # The point of the API: the package groups, the caller materializes.
    var df = sample()
    var groups = df.group_indices("region")
    var panels = List[DataFrame]()
    for g in range(groups.count()):
        panels.append(df.take(groups.rows(g)))
    assert_equal(len(panels), 3)
    assert_equal(panels[0].height(), 2)
    assert_equal(panels[0].column("amount").get(0).int64(), Int64(1))
    assert_equal(panels[0].column("amount").get(1).int64(), Int64(3))
    # The key for a caption comes from the representative row.
    assert_equal(df.item(groups.representative(2), "region").string(), "north")


def test_multiple_keys() raises:
    var df = DataFrame(
        [
            Series("a", Column[String](["x", "x", "y", "x"])),
            Series("b", Column[Int64]([1, 2, 1, 1])),
        ]
    )
    var groups = df.group_indices(["a", "b"])
    assert_equal(groups.count(), 3)
    assert_equal(groups.ids(), [0, 1, 2, 0])
    assert_equal(groups.rows(0), [0, 3])


def test_nulls_form_their_own_group() raises:
    var df = DataFrame(
        [
            Series(
                "k",
                Column[String](
                    ["a", "b", "a", "b"], [True, False, True, False]
                ),
            )
        ]
    )
    var groups = df.group_indices("k")
    assert_equal(groups.count(), 2)
    # Rows 1 and 3 are null and share a group, as in group_by.
    assert_equal(groups.rows(0), [0, 2])
    assert_equal(groups.rows(1), [1, 3])


def test_group_count_matches_group_by() raises:
    var df = sample()
    var indices = df.group_indices("region")
    var aggregated = df.group_by("region").agg(col("amount").sum())
    assert_equal(indices.count(), aggregated.height())


def test_single_group_and_empty_frame() raises:
    var one = DataFrame([Series("k", Column[Int64]([7, 7, 7]))])
    var groups = one.group_indices("k")
    assert_equal(groups.count(), 1)
    assert_equal(groups.rows(0), [0, 1, 2])

    var empty = DataFrame([Series("k", Column[Int64]([]))])
    var none = empty.group_indices("k")
    assert_equal(none.count(), 0)
    assert_equal(none.height(), 0)
    assert_equal(len(none.all_rows()), 0)


def test_bad_arguments_raise() raises:
    var df = sample()
    with assert_raises():
        _ = df.group_indices(List[String]())
    with assert_raises():
        _ = df.group_indices(["region", "region"])
    with assert_raises():
        _ = df.group_indices("missing")
    var groups = df.group_indices("region")
    with assert_raises():
        _ = groups.rows(3)
    with assert_raises():
        _ = groups.representative(-1)
    with assert_raises():
        _ = groups.group_of(99)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
