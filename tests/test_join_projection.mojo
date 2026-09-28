"""Lazy joins read only the columns the plan uses; results match eager."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
    assert_raises,
)
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    col,
    lit,
    scan_csv,
    write_csv,
)

comptime LEFT_CSV = "/tmp/dataframe_mojo_join_projection_left.csv"
comptime RIGHT_CSV = "/tmp/dataframe_mojo_join_projection_right.csv"


def lazy_count(
    left: DataFrame, right: DataFrame, on: List[String]
) raises -> Int64:
    return (
        left.lazy()
        .join(right.lazy(), on)
        .select(col(on[0]).len())
        .collect()
        .item()
        .int64()
    )


def test_duplicate_string_key_counts() raises:
    var left_values = List[String]()
    var left_valid = List[Bool]()
    var right_values = List[String]()
    var right_valid = List[Bool]()
    var left_counts = List[Int64](length=4, fill=0)
    var right_counts = List[Int64](length=4, fill=0)
    for i in range(1200):
        left_values.append("long-repeated-key-" + String(i % 4))
        left_valid.append(i % 17 != 0)
        if i % 17 != 0:
            left_counts[i % 4] += 1
    for i in range(2000):
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


def test_nullable_payload_counts_and_other_join_types() raises:
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


def test_counts_match_the_materialized_join() raises:
    var on: List[String] = ["k"]
    var keys: List[List[Int64]] = [
        [1, 2, 3, 4],
        [2, 4, 6],
        [1, 1, 2, 2, 3],
        [3, 1, 2, 9, 1],
        [1, 1, 0, 2],
        [1, 0, 2],
        [1, 2],
        [],
        [Int64.MIN, 0, Int64.MAX],
        [Int64.MAX, Int64.MIN],
    ]
    var valid: List[List[Bool]] = [
        [True, True, True, True],
        [True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, False],
        [True, True, False, True],
        [True, False, True],
        [True, True],
        [],
        [True, True, True],
        [True, True],
    ]
    for pair in range(len(keys) // 2):
        var sides = List[DataFrame]()
        for side in range(2):
            var at = 2 * pair + side
            var n = len(keys[at])
            sides.append(
                DataFrame(
                    [
                        Series(
                            "k",
                            Column[Int64](keys[at].copy(), valid[at].copy()),
                        ),
                        Series(
                            "x",
                            Column[Int64](
                                List[Int64](length=n, fill=Int64(side))
                            ),
                        ),
                    ]
                )
            )
        for reverse in range(2):
            var a = sides[1 - reverse].copy()
            var b = sides[reverse].copy()
            assert_equal(lazy_count(a, b, on), Int64(a.join(b, "k").height()))
    var strings_left = DataFrame(
        [
            Series("s", Column[String](["a", "b", "a", "c"])),
            Series("n", Column[Int64]([1, 2, 1, 3])),
        ]
    )
    var strings_right = DataFrame(
        [
            Series("s", Column[String](["a", "a", "c", "d"])),
            Series("n", Column[Int64]([1, 1, 3, 4])),
        ]
    )
    var compound: List[String] = ["s", "n"]
    assert_equal(
        lazy_count(strings_left, strings_right, compound),
        Int64(strings_left.join(strings_right, compound).height()),
    )
    var bools = DataFrame(
        [
            Series(
                "k",
                Column[Bool](
                    [True, True, False, False], [True, True, True, False]
                ),
            )
        ]
    )
    assert_equal(
        lazy_count(bools, bools, on), Int64(bools.join(bools, "k").height())
    )


def payload_frames() raises -> Tuple[DataFrame, DataFrame]:
    var left = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 2, 3, 5])),
            Series("x", Column[Float64]([1, 2, 3, 4, 5])),
            Series("y", Column[Int64]([5, 6, 7, 8, 9])),
            Series("unused", Column[String](["a", "b", "c", "d", "e"])),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int64]([2, 3, 3, 4])),
            Series("x", Column[Float64]([9, 8, 7, 6])),
            Series("r", Column[Int64]([1, 2, 3, 4])),
            Series("other", Column[String](["p", "q", "r", "s"])),
        ]
    )
    return (left^, right^)


def test_explain_shows_joins_reading_only_used_columns() raises:
    var frames = payload_frames()
    var plan = (
        frames[0].lazy().join(frames[1].lazy(), "k").select(col("k").len())
    ).explain()
    assert_equal(plan.count("[project k]"), 2)
    # A right column behind a suffix keeps its colliding left column.
    var suffixed = (
        frames[0].lazy().join(frames[1].lazy(), "k").select(["x_right", "r"])
    )
    var text = suffixed.explain()
    assert_true("[project k, x]" in text)
    assert_true("[project k, x, r]" in text)
    var eager = frames[0].join(frames[1], "k").select(["x_right", "r"])
    assert_true(suffixed.collect().equals(eager))
    # Semi and anti joins read only the right side's keys.
    var semi = frames[0].lazy().join(frames[1].lazy(), "k", how="semi")
    assert_equal(semi.select(["x"]).explain().count("[project k"), 2)
    var hows: List[String] = ["inner", "left", "right", "full", "semi", "anti"]
    for how in hows:
        var names: List[String] = ["x", "r"]
        if how == "semi" or how == "anti":
            names = ["x"]
        var lazy = (
            frames[0]
            .lazy()
            .join(frames[1].lazy(), "k", how=how)
            .select(names)
            .collect()
        )
        var expected = frames[0].join(frames[1], "k", how=how).select(names)
        assert_true(lazy.equals(expected))


def test_filter_below_a_join_is_narrowed_before_the_join() raises:
    var frames = payload_frames()
    var query = (
        frames[0]
        .lazy()
        .filter(col("y") > lit(Int64(5)))
        .join(frames[1].lazy(), "k")
        .select(["x_right", "k"])
    )
    var plan = query.explain()
    assert_true("SELECT k, x [stream]\n      FILTER" in plan)
    var eager = (
        frames[0]
        .filter(col("y") > lit(Int64(5)))
        .join(frames[1], "k")
        .select(["x_right", "k"])
    )
    assert_true(query.collect().equals(eager))


def test_csv_scans_under_a_join_read_only_used_columns() raises:
    var frames = payload_frames()
    write_csv(frames[0], LEFT_CSV)
    write_csv(frames[1], RIGHT_CSV)
    var query = (
        scan_csv(LEFT_CSV)
        .filter(col("y") > lit(Int64(5)))
        .join(scan_csv(RIGHT_CSV), "k")
        .select(col("k").len())
    )
    var plan = query.explain()
    assert_true("[project k, y]" in plan)
    assert_true("[project k]" in plan)
    assert_false("unused" in plan)
    var eager = frames[0].filter(col("y") > lit(Int64(5))).join(frames[1], "k")
    assert_equal(query.collect().item().int64(), Int64(eager.height()))


def test_colliding_names_still_raise() raises:
    var left = DataFrame(
        [
            Series("k", Column[Int64]([1])),
            Series("v", Column[Int64]([1])),
            Series("v_right", Column[Int64]([1])),
        ]
    )
    var right = DataFrame(
        [Series("k", Column[Int64]([1])), Series("v", Column[Int64]([2]))]
    )
    with assert_raises(contains="collision"):
        _ = left.lazy().join(right.lazy(), "k").select(["k"]).collect()


def test_unknown_join_type_raises_from_a_lazy_plan() raises:
    var frames = payload_frames()
    with assert_raises(contains="Join how must be"):
        _ = (
            frames[0]
            .lazy()
            .join(frames[1].lazy(), "k", how="sideways")
            .select(["x"])
            .collect()
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
