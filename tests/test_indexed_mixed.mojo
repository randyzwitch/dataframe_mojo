"""Bounded indexed reduction uses the ordinary evaluator as its oracle."""
from std.testing import TestSuite, assert_true, assert_equal, assert_raises
from dataframe import DataType, Column, DataFrame, Series, Expr, col
from dataframe.binding import bind
from dataframe.execution import evaluate
from dataframe.indexed_reduce import (
    reduce_indexed_batches,
    indexed_reductions,
    reduce_indexed,
)
from dataframe.frame import _bind_all


def check(frame: DataFrame, expression: Expr) raises:
    var rows = List[Int]()
    var ids = List[Int]()
    for p in range(frame.height()):
        rows.append((p * 7) % frame.height())
        ids.append(p % 5)
    var gathered = frame.take(rows)
    var expected = evaluate(
        bind(expression, gathered._columns),
        gathered._columns,
        frame.height(),
        grouped=True,
        groups=ids,
        group_count=5,
        batch_size=13,
    )
    var actual = reduce_indexed_batches(
        bind(expression, frame._columns),
        frame._columns,
        rows.unsafe_ptr(),
        ids,
        5,
        13,
    )
    assert_true(actual.equals(expected), expression._name)
    var bound = bind(expression, frame._columns)
    if indexed_reductions([bound.copy()], frame._columns):
        var direct = reduce_indexed(
            bound, frame._columns, rows.unsafe_ptr(), ids, 5
        )
        assert_true(direct.equals(expected), expression._name)


def test_numeric_computed_and_ordered_reducers() raises:
    var xs = List[Int32]()
    var ys = List[Float32]()
    var valid = List[Bool]()
    for i in range(101):
        xs.append(Int32(i % 17 - 8))
        ys.append(Float32(i % 19 - 9))
        valid.append(i % 11 != 0)
    var frame = DataFrame(
        [
            Series("x", Column[Int32](xs^, valid.copy())),
            Series("y", Column[Float32](ys^, valid^)),
        ]
    )
    var expressions: List[Expr] = [
        col("x").sum(),
        col("x").min(),
        col("x").max(),
        col("x").mean(),
        col("x").first(),
        col("x").last(),
        col("x").n_unique(),
        col("x").arg_min(),
        col("x").arg_max(),
        col("x").null_count(),
        col("y").std(),
        col("y").var(),
        (col("x").cast("float32") * col("y") + 2).sum(),
        col("x").sum().cast("float32") / col("y").sum(),
    ]
    for expression in expressions:
        check(frame, expression)


def test_decimal_unsigned_null_nan_and_zero() raises:
    var frame = DataFrame(
        [
            Series(
                "u",
                Column[UInt64](
                    [UInt64.MAX, 0, 7, UInt64.MAX, 2, 0, 1, 3, 4, 6, 9]
                ),
            ),
            Series(
                "f",
                Column[Float64](
                    [
                        0.0,
                        -0.0,
                        Float64(0) / Float64(0),
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                    ],
                    [
                        True,
                        True,
                        True,
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                ),
            ),
            Series(
                "d",
                Column[Int64]([100, -200, 0, 999, 1, 8, 40, -30, 12, 0, 10]),
            ),
        ]
    ).with_columns(col("d").cast(DataType.decimal(20, 2)))
    var expressions: List[Expr] = [
        col("u").min(),
        col("u").max(),
        col("u").mean(),
        col("f").min(),
        col("f").max(),
        col("f").first(),
        col("f").last(),
        col("f").n_unique(),
        col("d").sum(),
        col("d").mean(),
        col("d").min(),
        (col("d") + col("d")).sum(),
    ]
    for expression in expressions:
        check(frame, expression)


def test_mixed_partitioned_matches_whole_and_keeps_order() raises:
    var keys = List[Int64]()
    var xs = List[Int64]()
    var ys = List[Float64]()
    for i in range(30_013):
        keys.append(Int64((i * 37) % 3001))
        xs.append(Int64(i % 31 - 15))
        ys.append(Float64(i % 29))
    var frame = DataFrame(
        [
            Series("key", Column[Int64](keys^)),
            Series("x", Column[Int64](xs^)),
            Series("y", Column[Float64](ys^)),
        ]
    )
    var expressions: List[Expr] = [
        col("x").sum().alias("sum"),
        (col("x") + 1).mean().alias("computed"),
        col("x").first().alias("first"),
        col("y").median().alias("median"),
        col("y").mode().alias("mode"),
        col("y").n_unique().alias("unique"),
    ]
    var grouped = frame.group_by("key", maintain_order=True)
    var bound = _bind_all(expressions, frame._columns)
    var preparation = grouped._partitioned_references(bound)
    # A median takes the indexed route; a mode still needs its source
    # gathered into bucket order.
    assert_equal(len(preparation[2]), 1)
    assert_equal(preparation[2][0].name(), "y")
    assert_equal(
        preparation[0], List[Bool]([True, True, True, True, False, True])
    )
    var expected = grouped._agg_whole(bound, 127)
    # Force the partitioned route so this exercises mixed indexed/gathered
    # reductions independently of the cardinality estimate.
    var actual = grouped._agg_partitioned(
        expressions, bound, 127, 8, whole=False
    )
    assert_true(actual.equals(expected))


def test_indexed_decimal_overflow_is_reported() raises:
    var big = Int128(6) * Int128(10) ** 37
    var source = Series("d", Column[Int128]([big, big])).with_dtype(
        DataType.decimal(38, 0)
    )
    var columns: List[Series] = [source^]
    var expression = col("d").sum()
    var bound = bind(expression, columns)
    var rows: List[Int] = [0, 1]
    var ids: List[Int] = [0, 0]
    with assert_raises(contains="exceeds decimal precision"):
        _ = evaluate(bound, columns, 2, grouped=True, groups=ids, group_count=1)
    with assert_raises(contains="exceeds decimal precision"):
        _ = reduce_indexed_batches(bound, columns, rows.unsafe_ptr(), ids, 1, 1)


def test_categorical_fallback_does_not_gather_numeric_sources() raises:
    var keys = List[Int64]()
    var values = List[Int64]()
    var texts = List[String]()
    var words: List[String] = ["z", "a", "m"]
    for i in range(20_003):
        keys.append(Int64((i * 37) % 2003))
        values.append(Int64(i % 29))
        texts.append(words[i % 3])
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("x", Column[Int64](values^)),
            Series("s", Column[String](texts^)),
        ]
    ).with_columns(col("s").cast("categorical"))
    var expressions: List[Expr] = [
        col("x").sum().alias("sum"),
        col("s").min().alias("min"),
        col("s").max().alias("max"),
        col("s").first().alias("first"),
        col("s").n_unique().alias("unique"),
    ]
    var grouped = frame.group_by("k", maintain_order=True)
    var bound = _bind_all(expressions, frame._columns)
    var preparation = grouped._partitioned_references(bound)
    assert_equal(len(preparation[2]), 1)
    assert_equal(preparation[2][0].name(), "s")
    assert_true(
        grouped._agg_partitioned(
            expressions, bound, 127, 8, whole=False
        ).equals(grouped._agg_whole(bound, 127))
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
