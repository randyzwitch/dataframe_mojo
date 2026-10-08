"""As-of join direction, grouping, sortedness, exact arithmetic and lazy plans."""
from std.collections import Optional
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from dataframe import (
    AnyValue,
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
)
from dataframe.dtype import NUMERIC_DTYPES


def ints(name: String, var values: List[Int64]) -> Series:
    return Series(name, Column[Int64](values^))


def left() raises -> DataFrame:
    return DataFrame(
        [ints("k", [0, 1, 1, 2, 3, 4, 5]), ints("v", [0, 1, 2, 3, 4, 5, 6])]
    )


def right() raises -> DataFrame:
    return DataFrame(
        [ints("k", [1, 1, 3, 3, 5]), ints("v", [10, 11, 30, 31, 50])]
    )


def test_directions_duplicates_ties_and_tolerance() raises:
    var l = left()
    var r = right()
    var backward = l.join_asof(r, on="k")
    assert_true(backward.item(0, "v_right").is_null())
    assert_equal(backward.item(2, "v_right").int64(), 11)
    assert_equal(backward.item(4, "v_right").int64(), 31)
    var forward = l.join_asof(r, on="k", strategy="forward")
    assert_equal(forward.item(0, "v_right").int64(), 10)
    assert_equal(forward.item(4, "v_right").int64(), 30)
    var nearest = l.join_asof(r, on="k", strategy="nearest")
    assert_equal(nearest.item(0, "v_right").int64(), 11)
    assert_equal(nearest.item(3, "v_right").int64(), 31)
    var strict = l.join_asof(
        r, on="k", strategy="nearest", allow_exact_matches=False
    )
    assert_equal(strict.item(1, "v_right").int64(), 31)
    assert_equal(strict.item(6, "v_right").int64(), 31)
    var bounded = l.join_asof(r, on="k", tolerance=0)
    assert_true(bounded.item(3, "v_right").is_null())
    assert_equal(bounded.item(4, "v_right").int64(), 31)
    assert_true(l.join_asof(r, on="k", tolerance=None).equals(backward))
    assert_true(l.join_asof(r, on="k", by=None).equals(backward))
    assert_true(
        l.lazy()
        .join_asof(r.lazy(), on="k", by=None, tolerance=1)
        .collect()
        .equals(backward)
    )
    assert_true(l.join_asof(r, on="k", tolerance=1.0).equals(backward))


def test_numeric_and_temporal_keys() raises:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        var l = DataFrame([left().column("k").cast(DataType.of(D))])
        var r = DataFrame(
            [right().column("k").cast(DataType.of(D)), right().column("v")]
        )
        for strategy in ["backward", "forward", "nearest"]:
            assert_equal(
                l.join_asof(r, on="k", strategy=strategy).item(4, "v").int64(),
                Int64(30 if strategy == "forward" else 31),
            )
    for dtype in [
        DataType.DATE,
        DataType.datetime("ms"),
        DataType.datetime("ns", "UTC"),
        DataType.duration("us"),
    ]:
        var l = DataFrame([left().column("k").with_dtype(dtype)])
        var r = DataFrame(
            [right().column("k").with_dtype(dtype), right().column("v")]
        )
        var tol = AnyValue.temporal(DataType.duration("us"), 1500)
        var result = l.join_asof(r, on="k", tolerance=Optional(tol^))
        assert_equal(result.column("k").dtype(), dtype)
        assert_equal(result.item(4, "v").int64(), 31)
        if dtype.is_date() or dtype.unit() == "us":
            assert_true(
                result.item(3, "v")
                .is_null() if dtype.is_date() else not result.item(3, "v")
                .is_null()
            )


def test_integer_extremes_are_exact() raises:
    var low = Int64.MIN
    var high = Int64.MAX
    var l = DataFrame([ints("k", [low, low + 1, high - 1, high])])
    var r = DataFrame([ints("k", [low, high]), ints("v", [1, 2])])
    var nearest = l.join_asof(r, on="k", strategy="nearest", tolerance=1)
    assert_equal(nearest.item(1, "v").int64(), 1)
    assert_equal(nearest.item(2, "v").int64(), 2)
    var u = UInt64.MAX
    var ul = DataFrame([Series("k", Column[UInt64]([u - 2, u - 1, u]))])
    var ur = DataFrame(
        [Series("k", Column[UInt64]([0, u - 2, u])), ints("v", [0, 1, 2])]
    )
    assert_equal(
        ul.join_asof(ur, on="k", strategy="nearest").item(1, "v").int64(), 2
    )
    var wide = AnyValue(u)
    assert_equal(
        ul.join_asof(ur, on="k", tolerance=Optional(wide^))
        .item(0, "v")
        .int64(),
        1,
    )


def test_groups_interleaved_and_nullable() raises:
    var l = DataFrame(
        [
            ints("k", [1, 0, 3, 2]),
            Series("g", StringColumn(["a", "b", "a", "b"])),
            ints("h", [0, 1, 0, 1]),
        ]
    )
    var r = DataFrame(
        [
            ints("k", [0, 1, 2, 3]),
            Series("g", StringColumn(["a", "b", "a", "b"])),
            ints("h", [0, 1, 0, 1]),
            ints("v", [10, 20, 30, 40]),
        ]
    )
    var result = l.join_asof(r, on="k", by=["g", "h"])
    assert_equal(result.item(0, "v").int64(), 10)
    assert_true(result.item(1, "v").is_null())
    assert_equal(result.item(2, "v").int64(), 30)
    assert_equal(result.item(3, "v").int64(), 20)
    assert_equal(result.width(), 4)
    assert_true(
        l.join_asof(r, on="k", by="g").equals(l.join_asof(r, on="k", by=["g"]))
    )
    assert_true(
        l.lazy()
        .join_asof(r.lazy(), on="k", by="g", tolerance=2)
        .collect()
        .equals(l.join_asof(r, on="k", by=["g"], tolerance=2))
    )
    var missing = Series(
        "g", StringColumn(["a", "b", "a", "b"], [False, True, False, True])
    )
    var nullable = DataFrame([l.column("k"), missing^])
    var joined = nullable.join_asof(r.drop(["h"]), on="k", by=["g"])
    assert_true(joined.item(0, "v").is_null())
    assert_true(joined.item(2, "v").is_null())


def test_sortedness_checks_both_sides_and_groups() raises:
    var bad = DataFrame([ints("time", [2, 1])])
    var good = DataFrame([ints("time", [1, 2])])
    with assert_raises(contains="left column 'time'"):
        _ = bad.join_asof(good, on="time")
    with assert_raises(contains="right column 'time'"):
        _ = good.join_asof(bad, on="time")
    with assert_raises(contains="right column 'time'"):
        _ = good.head(0).join_asof(bad, on="time")
    var plan = bad.lazy().join_asof(good.lazy(), on="time")
    assert_equal(len(plan.collect_schema()), 1)
    for streaming in [False, True]:
        with assert_raises(contains="left column 'time'"):
            _ = plan.collect(streaming=streaming)
    var grouped = DataFrame(
        [ints("time", [2, 0, 1, 3]), ints("g", [0, 1, 0, 1])]
    )
    with assert_raises(contains="left column 'time'"):
        _ = grouped.join_asof(grouped.sort(["g", "time"]), on="time", by=["g"])


def test_nulls_empty_frames_and_chunks() raises:
    var l = DataFrame(
        [Series("k", Column[Int64]([0, 1, 2], [False, True, True]))]
    )
    var r = DataFrame(
        [
            Series("k", Column[Int64]([0, 1, 3], [False, True, True])),
            ints("v", [0, 10, 30]),
        ]
    )
    for strategy in ["backward", "forward", "nearest"]:
        var result = l.join_asof(r, on="k", strategy=strategy)
        assert_true(result.item(0, "v").is_null())
        assert_equal(
            l.head(0).join_asof(r, on="k", strategy=strategy).height(), 0
        )
        assert_equal(
            l.join_asof(r.head(0), on="k", strategy=strategy)
            .column("v")
            .null_count(),
            3,
        )
        var chunks = DataFrame(
            [l.column("k").slice(0, 1).append(l.column("k").slice(1, 2))]
        )
        assert_true(
            chunks.join_asof(r, on="k", strategy=strategy).equals(result)
        )


def test_different_names_suffix_and_invalid_options() raises:
    var l = left()
    var r = DataFrame([right().column("k").renamed("rk"), right().column("v")])
    var result = l.join_asof(r, left_on="k", right_on="rk", suffix="_r")
    assert_equal(result.columns(), ["k", "v", "rk", "v_r"])
    assert_true(result.item(0, "rk").is_null())
    with assert_raises(contains="collision"):
        _ = l.join_asof(right(), on="k", suffix="")
    with assert_raises(contains="strategy"):
        _ = l.join_asof(right(), on="k", strategy="invalid")
    with assert_raises(contains="nonnegative"):
        _ = l.join_asof(right(), on="k", tolerance=-1)
    with assert_raises(contains="NaN"):
        _ = l.join_asof(right(), on="k", tolerance=Float64("nan"))
    with assert_raises(contains="dtypes"):
        _ = l.join_asof(
            DataFrame([right().column("k").cast(DataType.FLOAT64)]), on="k"
        )
    with assert_raises(contains="use on"):
        _ = l.join_asof(right(), on="k", left_on="k", right_on="k")
    with assert_raises(contains="duplicate"):
        _ = l.join_asof(right(), on="k", by=["v", "v"])


def test_lazy_optimization_preserves_asof_boundaries() raises:
    var l = left()
    var r = right()
    for strategy in ["backward", "forward", "nearest"]:
        var eager = (
            l.join_asof(r, on="k", strategy=strategy)
            .filter(col("v_right") > 10)
            .select(["v_right"])
            .head(3)
        )
        var plan = (
            l.lazy()
            .join_asof(r.lazy(), on="k", strategy=strategy)
            .filter(col("v_right") > 10)
            .select(["v_right"])
            .head(3)
        )
        for streaming in [False, True]:
            for optimize in [False, True]:
                assert_true(
                    plan.collect(streaming=streaming, optimize=optimize).equals(
                        eager
                    )
                )
        assert_true("ASOF" in plan.explain())
        assert_true(plan.profile()[0].equals(eager))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
