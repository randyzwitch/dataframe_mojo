"""Expression semantics and batch/SIMD equivalence, independent of API syntax."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
    assert_almost_equal,
)
from dataframe import DataFrame, Column, Series, col, lit
from dataframe.binding import bind, ROWS, SCALAR, AGGREGATE
from dataframe.execution import evaluate
from dataframe.reductions import IntSumState, FloatSumState


def fixture() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "x", Column[Float64]([2, 4, 999, 8], [True, True, False, True])
            ),
            Series("n", Column[Int64]([1, 2, 3, 4])),
            Series("key", Column[String](["a", "b", "a", "b"])),
        ]
    )


def test_composition_alias_and_reuse() raises:
    var frame = fixture()
    var expr = (col("x") + lit(Float64(1))) * (col("x") - lit(Float64(1)))
    var result = frame.select(expr.alias("square_minus_one"), batch_size=3)
    var values = result.column("square_minus_one").float64()
    assert_equal(values.value(0), Float64(3))
    assert_equal(values.value(1), Float64(15))
    assert_true(values.is_null(2))
    assert_equal(values.value(3), Float64(63))
    assert_equal(frame.select(expr).schema()[0].name, "x")
    assert_equal(frame.column("x").float64().value(0), Float64(2))


def test_literal_projection_and_shape_metadata() raises:
    var frame = fixture()
    assert_equal(bind(col("x"), frame._columns).shape(), ROWS)
    assert_equal(bind(lit(Int64(1)), frame._columns).shape(), SCALAR)
    assert_equal(bind(col("x").sum(), frame._columns).shape(), AGGREGATE)
    assert_equal(bind(col("x") - col("x").sum(), frame._columns).shape(), ROWS)
    var result = frame.select_exprs(
        [
            lit(Int64(2)).alias("a"),
            lit(String("ok")).alias("b"),
            lit(True).alias("c"),
        ]
    )
    assert_equal(result.height(), 1)
    assert_equal(result.column("a").int64().value(0), Int64(2))
    assert_equal(result.column("b").string().value(0), "ok")
    assert_true(result.column("c").bool().value(0))


def test_global_aggregates_and_broadcast() raises:
    var frame = fixture()
    var result = frame.select_exprs(
        [
            col("x").sum().alias("sum"),
            col("x").count().alias("count"),
            (col("n").sum() + lit(Int64(10))).alias("plus"),
        ],
        batch_size=3,
    )
    assert_equal(result.height(), 1)
    assert_equal(result.column("sum").float64().value(0), Float64(14))
    assert_equal(result.column("count").int64().value(0), Int64(3))
    assert_equal(result.column("plus").int64().value(0), Int64(20))
    var rows = frame.select_exprs(
        [col("x"), col("n").sum().alias("total"), lit(Int64(7)).alias("seven")]
    )
    assert_equal(rows.height(), 4)
    for i in range(4):
        assert_equal(rows.column("total").int64().value(i), Int64(10))
        assert_equal(rows.column("seven").int64().value(i), Int64(7))
    var centered = frame.select(
        (col("x") - col("x").sum()).alias("difference"), batch_size=3
    )
    assert_equal(centered.column("difference").float64().value(0), Float64(-12))
    assert_true(centered.column("difference").float64().is_null(2))


def test_empty_inputs_scalar_and_row_shapes() raises:
    var frame = fixture().take([])
    var sums = frame.select_exprs(
        [
            col("x").sum().alias("zero"),
            col("x").sum(min_count=1).alias("null"),
            col("key").count().alias("count"),
        ]
    )
    assert_equal(sums.height(), 1)
    assert_equal(sums.column("zero").float64().value(0), Float64(0))
    assert_true(sums.column("null").float64().is_null(0))
    assert_equal(sums.column("count").int64().value(0), Int64(0))
    assert_equal(
        frame.select_exprs([col("x"), col("n").sum().alias("total")]).height(),
        0,
    )
    assert_equal(frame.select(lit(Int64(4))).height(), 1)
    assert_equal(
        frame.with_columns(lit(Int64(4)).alias("constant")).height(), 0
    )
    assert_equal(frame.select_exprs([]).height(), 0)
    assert_equal(fixture().select_exprs([]).height(), 4)
    assert_equal(
        DataFrame([], height=7).with_columns(lit(True).alias("flag")).height(),
        7,
    )


def test_null_nan_and_min_count() raises:
    var nan = Float64("nan")
    var frame = DataFrame(
        [
            Series("x", Column[Float64]([nan, 999, 3], [True, False, True])),
            Series(
                "empty", Column[Int64]([999, 999, 999], [False, False, False])
            ),
        ]
    )
    var result = frame.select_exprs(
        [
            col("x").sum().alias("nan"),
            col("empty").sum().alias("zero"),
            col("empty").sum(1).alias("null"),
            col("x").count().alias("count"),
        ]
    )
    var total = result.column("nan").float64().value(0)
    assert_true(total != total)
    assert_equal(result.column("zero").int64().value(0), Int64(0))
    assert_true(result.column("null").int64().is_null(0))
    assert_equal(result.column("count").int64().value(0), Int64(2))
    assert_true(frame.select(col("x").sum(3)).column("x").float64().is_null(0))
    assert_equal(frame.filter(col("x") > lit(Float64(0))).height(), 1)


def test_with_columns_siblings_see_original_input() raises:
    var frame = fixture()
    var result = frame.with_columns(
        [
            (col("n") + lit(Int64(1))).alias("n"),
            (col("n") * lit(Int64(2))).alias("double_original"),
            col("n").sum().alias("total"),
        ],
        batch_size=3,
    )
    assert_equal(result.column("n").int64().value(0), Int64(2))
    assert_equal(result.column("double_original").int64().value(0), Int64(2))
    assert_equal(result.column("total").int64().value(3), Int64(10))
    assert_equal(frame.column("n").int64().value(0), Int64(1))
    with assert_raises():
        _ = frame.with_columns(
            [col("n").alias("new"), col("new").alias("later")]
        )
    assert_equal(frame.with_columns([]).width(), 3)


def test_filter_and_equality() raises:
    var frame = fixture()
    var result = frame.filter(col("key").eq(lit(String("a"))))
    assert_equal(result.height(), 2)
    assert_equal(result.column("n").int64().value(1), Int64(3))
    assert_equal(frame.filter(lit(True)).height(), 4)
    assert_equal(frame.filter(lit(False)).height(), 0)
    assert_equal(frame.filter(col("n").sum() > lit(Int64(5))).height(), 4)
    assert_equal(frame.filter(col("n").eq(lit(Int64(2)))).height(), 1)
    assert_equal(frame.filter(col("x").eq(lit(Float64(4)))).height(), 1)
    assert_equal(
        frame.with_columns(lit(True).alias("b"))
        .filter(col("b").eq(lit(True)))
        .height(),
        4,
    )


def test_schema_errors_before_any_kernel() raises:
    # The first expression would overflow if executed before binding the second.
    var frame = DataFrame(
        [Series("n", Column[Int64]([9223372036854775807, 1]))]
    )
    try:
        _ = frame.select_exprs([col("n").sum(), col("missing")])
        raise Error("Expected schema failure")
    except error:
        assert_true(String(error).find("Unknown expression column") >= 0)
    with assert_raises():
        _ = frame.select(col("n") + lit(Float64(1)))
    with assert_raises():
        _ = frame.filter(col("n"))
    with assert_raises():
        _ = frame.select_exprs([col("n"), col("n")])
    with assert_raises():
        _ = frame.select(col("n").sum().sum())
    with assert_raises():
        _ = frame.select((col("n") + col("n").sum()).sum())
    with assert_raises():
        _ = frame.select(lit(Int64(1)).sum())
    with assert_raises():
        _ = frame.select(col("n").sum(-1))
    with assert_raises():
        _ = fixture().select(col("key").sum())
    with assert_raises():
        _ = fixture().select(col("key") + lit(String("x")))
    with assert_raises():
        _ = frame.select(col("n"), batch_size=0)


def test_checked_integer_arithmetic_boundaries() raises:
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Int64]([-9223372036854775808, 9223372036854775807, 0]),
            )
        ]
    )
    var identity = frame.select((col("x") * lit(Int64(1))).alias("x"))
    assert_equal(
        identity.column("x").int64().value(0), Int64(-9223372036854775808)
    )
    assert_equal(
        identity.column("x").int64().value(1), Int64(9223372036854775807)
    )
    var zero = frame.select(col("x") * lit(Int64(0)))
    assert_equal(zero.column("x").int64().value(0), Int64(0))
    with assert_raises():
        _ = frame.select(col("x") * lit(Int64(-1)))
    with assert_raises():
        _ = frame.select(col("x") + lit(Int64(1)))
    with assert_raises():
        _ = frame.select(col("x") - lit(Int64(1)))
    with assert_raises():
        _ = frame.select(col("x") * lit(Int64(2)))
    var invalid = DataFrame(
        [Series("x", Column[Int64]([-9223372036854775808], [False]))]
    )
    assert_true(
        invalid.select(col("x") * lit(Int64(-1))).column("x").int64().is_null(0)
    )
    var precise = DataFrame([Series("x", Column[Int64]([9007199254740993]))])
    assert_equal(
        precise.select(col("x") + lit(Int64(1))).column("x").int64().value(0),
        Int64(9007199254740994),
    )


def test_integer_multiplication_signs() raises:
    var frame = DataFrame([Series("x", Column[Int64]([-3, -1, 0, 1, 3]))])
    var values = frame.select(col("x") * lit(Int64(-2))).column("x").int64()
    var expected: List[Int64] = [6, 2, 0, -2, -6]
    for i in range(5):
        assert_equal(values.value(i), expected[i])


def test_grouped_expression_aggregates_and_null_keys() raises:
    var frame = DataFrame(
        [
            Series(
                "k",
                Column[String](
                    ["b", "ignored", "a", "b", "ignored", "all_null"],
                    [True, False, True, True, False, True],
                ),
            ),
            Series(
                "x",
                Column[Float64](
                    [2, 4, 6, 8, 10, 999], [True, True, True, True, True, False]
                ),
            ),
        ]
    )
    var result = frame.group_by("k", maintain_order=True).agg(
        [
            (col("x") * lit(Float64(2))).sum().alias("total"),
            col("x").sum(1).alias("nullable"),
            col("x").count().alias("count"),
            (col("x").sum() + lit(Float64(1))).alias("plus_one"),
        ],
        batch_size=3,
    )
    assert_equal(result.height(), 4)
    assert_equal(result.column("k").string().value(0), "b")
    assert_true(result.column("k").string().is_null(1))
    var totals = result.column("total").float64()
    assert_equal(totals.value(0), Float64(20))
    assert_equal(totals.value(1), Float64(28))
    assert_equal(totals.value(2), Float64(12))
    assert_equal(totals.value(3), Float64(0))
    assert_true(result.column("nullable").float64().is_null(3))
    assert_equal(result.column("count").int64().value(0), Int64(2))
    assert_equal(result.column("count").int64().value(3), Int64(0))
    assert_equal(result.column("plus_one").float64().value(3), Float64(1))
    assert_equal(frame.group_by("k").agg([]).height(), 4)
    assert_equal(frame.take([]).group_by("k").agg(col("x").sum()).height(), 0)


def test_group_validation() raises:
    var grouped = fixture().group_by("key")
    with assert_raises():
        _ = grouped.agg(col("x"))
    with assert_raises():
        _ = grouped.agg((col("x") - col("x").sum()).alias("bad"))
    with assert_raises():
        _ = grouped.agg(col("x").sum().alias("key"))
    with assert_raises():
        _ = grouped.agg([col("x").sum(), col("x").count()])
    with assert_raises():
        _ = fixture().group_by("n")
    with assert_raises():
        _ = grouped.agg(col("x").sum(), batch_size=0)


def test_simd_matches_scalar_across_batches_null_masks_and_tails() raises:
    var values = List[Float64]()
    var valid = List[Bool]()
    for i in range(2053):
        values.append(Float64(i - 1000) / 8)
        valid.append(i % 7 != 0)
    var frame = DataFrame([Series("x", Column[Float64](values^, valid))])
    var bound = bind(
        ((col("x") + lit(Float64(3))) * (col("x") - lit(Float64(2)))).alias(
            "y"
        ),
        frame._columns,
    )
    var scalar = evaluate[1](
        bound, frame._columns, frame.height(), batch_size=1
    ).float64()
    var vector = evaluate[4](
        bound, frame._columns, frame.height(), batch_size=1024
    ).float64()
    var wider = evaluate[8](
        bound, frame._columns, frame.height(), batch_size=13
    ).float64()
    for i in range(frame.height()):
        assert_equal(vector.is_null(i), not valid[i])
        assert_equal(wider.is_null(i), not valid[i])
        if valid[i]:
            var x = Float64(i - 1000) / 8
            var expected = (x + 3) * (x - 2)
            assert_equal(scalar.value(i), expected)
            assert_equal(vector.value(i), expected)
            assert_equal(wider.value(i), expected)
    var predicate = bind(col("x") > lit(Float64(0)), frame._columns)
    var booleans = evaluate[4](
        predicate, frame._columns, frame.height(), batch_size=13
    ).bool()
    for i in range(frame.height()):
        assert_equal(booleans.is_null(i), not valid[i])
        if valid[i]:
            assert_equal(booleans.value(i), i > 1000)


def test_many_groups_cross_output_batch_boundaries() raises:
    var keys = List[String]()
    var values = List[Int64]()
    for _ in range(2):
        for i in range(35):
            keys.append(String(i))
            values.append(Int64(i))
    var frame = DataFrame(
        [
            Series("k", Column[String](keys^)),
            Series("v", Column[Int64](values^)),
        ]
    )
    var result = frame.group_by("k", maintain_order=True).agg(
        (col("v").sum() + lit(Int64(1))).alias("v"), batch_size=7
    )
    assert_equal(result.height(), 35)
    var totals = result.column("v").int64()
    for i in range(35):
        assert_equal(totals.value(i), Int64(i * 2 + 1))


def test_expression_sum_checks_final_overflow_and_merges_exactly() raises:
    var frame = DataFrame(
        [Series("x", Column[Int64]([9223372036854775807, 1, -1]))]
    )
    for batch_size in range(1, 5):
        assert_equal(
            frame.select(col("x").sum(), batch_size=batch_size)
            .column("x")
            .int64()
            .value(0),
            Int64(9223372036854775807),
        )
    var a = IntSumState()
    a.add(9223372036854775807)
    a.add(1)
    var b = IntSumState()
    b.add(-1)
    var forward = a.copy()
    forward.merge(b)
    var reverse = b.copy()
    reverse.merge(a)
    assert_equal(forward.value(), Int64(9223372036854775807))
    assert_equal(reverse.value(), forward.value())
    assert_equal(forward.count, Int64(3))
    with assert_raises():
        _ = a.value()
    var negative = IntSumState()
    negative.add(-9223372036854775808)
    negative.add(-1)
    with assert_raises():
        _ = negative.value()
    negative.add(1)
    assert_equal(negative.value(), Int64(-9223372036854775808))
    with assert_raises():
        _ = frame.take([0, 1]).select(col("x").sum())
    var grouped = (
        frame.with_columns(lit(String("a")).alias("k"))
        .group_by("k")
        .agg(col("x").sum(), batch_size=1)
    )
    assert_equal(grouped.column("x").int64().value(0), forward.value())
    var f = FloatSumState()
    f.add(1.5)
    var g = FloatSumState()
    g.add(2.5)
    f.merge(g)
    assert_equal(f.total, Float64(4))
    assert_equal(f.count, Int64(2))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
