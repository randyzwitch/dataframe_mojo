"""Count fusion preserves multiplicities, validity, schemas and fallback plans."""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.join_hash import count_inner_join, _checked_join_count_add


def test_duplicate_counts_do_not_require_joined_rows() raises:
    var left_values = List[String]()
    var left_valid = List[Bool]()
    var right_values = List[String]()
    var right_valid = List[Bool]()
    var left_counts = List[Int64](length=4, fill=0)
    var right_counts = List[Int64](length=4, fill=0)
    for i in range(12000):
        left_values.append("long-repeated-key-" + String(i % 4))
        left_valid.append(i % 17 != 0)
        if i % 17 != 0:
            left_counts[i % 4] += 1
    for i in range(20000):
        right_values.append("long-repeated-key-" + String(i % 4))
        right_valid.append(i % 13 != 0)
        if i % 13 != 0:
            right_counts[i % 4] += 1
    var expected = Int64(0)
    for k in range(4):
        expected += left_counts[k] * right_counts[k]
    var left = DataFrame(
        [Series("k", Column[String](left_values^, left_valid^))]
    )
    var right = DataFrame(
        [Series("k", Column[String](right_values^, right_valid^))]
    )
    for reverse in range(2):
        var a = right.copy() if reverse else left.copy()
        var b = left.copy() if reverse else right.copy()
        var result = (
            a.lazy()
            .join(b.lazy(), "k")
            .select_exprs(
                [col("k").len().alias("rows"), col("k").count().alias("valid")]
            )
            .collect()
        )
        assert_equal(result.height(), 1)
        assert_equal(result.item(0, "rows").int64(), expected)
        assert_equal(result.item(0, "valid").int64(), expected)
        assert_equal(result["rows"].dtype(), DataType.INT64)


def test_compound_nan_signed_zero_and_null_counts() raises:
    var nan = Float64(0) / Float64(0)
    var left = DataFrame(
        [
            Series(
                "f",
                Column[Float64](
                    [nan, -0.0, 0.0, 8, 0], [True, True, True, True, False]
                ),
            ),
            Series("s", Column[String](["a", "b", "a", "missing", "a"])),
        ]
    )
    var right = DataFrame(
        [
            Series(
                "f",
                Column[Float64](
                    [0.0, nan, nan, -0.0, 0], [True, True, True, True, False]
                ),
            ),
            Series("s", Column[String](["a", "a", "a", "b", "missing"])),
        ]
    )
    var eager = left.join(right, ["f", "s"])
    assert_equal(eager.height(), 4)
    for streaming in [True, False]:
        var result = (
            left.lazy()
            .join(right.lazy(), ["f", "s"])
            .select(col("f").count())
            .collect(streaming=streaming)
        )
        assert_equal(result.item().int64(), 4)
    var empty = right.clear()
    assert_equal(
        left.lazy()
        .join(empty.lazy(), ["f", "s"])
        .select(col("s").len())
        .collect()
        .item()
        .int64(),
        0,
    )


def test_nullable_payload_counts_and_non_inner_joins_fall_back() raises:
    var left = DataFrame([Series("k", Column[Int64]([1, 3]))])
    var right = DataFrame(
        [
            Series("k", Column[Int64]([1, 1])),
            Series("v", Column[Int64]([0, 7], [False, True])),
        ]
    )
    assert_equal(
        left.lazy()
        .join(right.lazy(), "k")
        .select(col("v").len())
        .collect()
        .item()
        .int64(),
        2,
    )
    assert_equal(
        left.lazy()
        .join(right.lazy(), "k")
        .select(col("v").count())
        .collect()
        .item()
        .int64(),
        1,
    )
    for how in ["left", "full", "semi", "anti"]:
        var expected = left.join(right, "k", how=how).height()
        assert_equal(
            left.lazy()
            .join(right.lazy(), "k", how=how)
            .select(col("k").len())
            .collect()
            .item()
            .int64(),
            Int64(expected),
        )
    var nulls = DataFrame([Series("k", Column[Int64]([0], [False]))])
    assert_equal(
        nulls.lazy()
        .join(nulls.lazy(), "k")
        .select(col("k").len())
        .collect()
        .item()
        .int64(),
        0,
    )
    with assert_raises():
        _ = (
            left.lazy()
            .join(right.lazy(), "k")
            .select(col("missing").len())
            .collect()
        )
    var wrong = DataFrame([Series("k", Column[String](["1"]))])
    with assert_raises():
        _ = left.lazy().join(wrong.lazy(), "k").select(col("k").len()).collect()


def test_count_overflow_is_explicit() raises:
    assert_equal(_checked_join_count_add(Int64.MAX - 1, 1), Int64.MAX)
    with assert_raises():
        _ = _checked_join_count_add(Int64.MAX, 1)


def test_progression_counts_keep_partial_runs_and_extremes() raises:
    var left: List[Series] = [Series("k", Column[Int64]([1, 1, 2, 2, 3]))]
    var right: List[Series] = [
        Series(
            "k", Column[Int64]([3, 1, 2, 9, 1], [True, True, True, True, False])
        )
    ]
    assert_equal(count_inner_join(left, right), 5)
    assert_equal(count_inner_join(right, left), 5)
    var extremes: List[Series] = [
        Series("k", Column[Int64]([Int64.MIN, Int64.MAX]))
    ]
    var probes: List[Series] = [
        Series("k", Column[Int64]([Int64.MAX, 0, Int64.MIN]))
    ]
    assert_equal(count_inner_join(extremes, probes), 2)
    var bool_left: List[Series] = [
        Series(
            "k",
            Column[Bool]([True, True, False, False], [True, True, True, False]),
        )
    ]
    var bool_right: List[Series] = [
        Series(
            "k",
            Column[Bool]([True, False, False, True], [True, True, True, False]),
        )
    ]
    assert_equal(count_inner_join(bool_left, bool_right), 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
