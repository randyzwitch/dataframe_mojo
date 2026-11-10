"""Native null/NaN values and total comparisons preserve CPU semantics."""
# contract-group: execution.backends
from std.memory import bitcast
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    lit,
    null,
    when,
)
from dataframe.expr import FILL_NULL, FILL_NAN, EQ_MISSING
from dataframe_metal.accelerator.gpu.null_value import (
    fill_value_bits,
    equal_missing_integer,
    equal_missing_float32,
)
from dataframe_metal.typed_expression_plan import compile_typed_expression
from dataframe_metal.conditional_plan import compile_conditional
from dataframe_metal.predicate_program import compile_predicate
from dataframe_metal.composition_plan import plan_composition


def test_metal_missing_float_equality_and_fill_bits_cover_nan_null_and_zero() raises:
    var values = List[UInt32](
        [
            UInt32(0),
            UInt32(0x80000000),
            UInt32(1),
            UInt32(0x80000001),
            UInt32(0x3F800000),
            UInt32(0xBF800000),
            UInt32(0x7F800000),
            UInt32(0xFF800000),
            UInt32(0x7F800001),
            UInt32(0x7FC12345),
            UInt32(0xFFC54321),
        ]
    )
    for a in values:
        for b in values:
            var x = bitcast[DType.float32](a)
            var y = bitcast[DType.float32](b)
            for av in [False, True]:
                for bv in [False, True]:
                    var expected = (
                        av == bv if not av
                        or not bv else x == y
                        or (isnan(x) and isnan(y))
                    )
                    assert_equal(equal_missing_float32(a, av, b, bv), expected)
                    var filled = fill_value_bits(
                        UInt64(a), av, UInt64(b), bv, FILL_NULL
                    )
                    assert_equal(filled[0], UInt64(a if av else b))
                    assert_equal(filled[1], av or bv)
                    var nan = fill_value_bits(
                        UInt64(a), av, UInt64(b), bv, FILL_NAN
                    )
                    var replace = av and isnan(x)
                    assert_equal(nan[0], UInt64(b if replace else a))
                    assert_equal(nan[1], bv if replace else av)
    for a in [Int64.MIN, Int64(-1), Int64(0), Int64(1), Int64.MAX]:
        for b in [Int64.MIN, Int64(-1), Int64(0), Int64(1), Int64.MAX]:
            for av in [False, True]:
                for bv in [False, True]:
                    assert_equal(
                        equal_missing_integer(a, av, b, bv),
                        av == bv if not av or not bv else a == b,
                    )


def test_metal_fill_plans_read_nan_validity_for_count_and_keep_native_types() raises:
    var frame = DataFrame(
        [
            Series("a", Column[Int32]([1, 2], [True, False])),
            Series("b", Column[Int32]([3, 4], [False, True])),
            Series("f", Column[Float32]([1, 2], [False, True])),
            Series("g", Column[Float32]([3, 4], [True, False])),
        ]
    )
    for value in [
        col("a").fill_null(col("b")),
        col("f").fill_null(0),
        col("f").fill_nan(col("g")),
        col("f").fill_nan(null(DataType.FLOAT32)),
        null(DataType.INT32).fill_null(col("a")),
    ]:
        var tree = compile_typed_expression(value, frame)
        assert_true(Bool(tree))
        var query = frame.lazy().select_exprs(
            [
                value.sum().alias("s"),
                value.count().alias("n"),
                value.mean().alias("avg"),
            ]
        )
        assert_true(Bool(plan_composition(query)))
    var nan = compile_typed_expression(
        col("f").fill_nan(null(DataType.FLOAT32)), frame
    )
    assert_equal(nan.value().nodes[0].literal & 3, UInt64(3))
    var nullable = compile_typed_expression(col("f").fill_null(col("g")), frame)
    assert_equal(nullable.value().nodes[0].literal & 3, UInt64(2))
    var conditional = compile_conditional(
        when(col("a") > 0)
        .then(col("f").fill_nan(null(DataType.FLOAT32)))
        .otherwise(col("g")),
        frame,
    )
    assert_true(Bool(conditional))
    assert_equal(conditional.value().nodes[0].literal, UInt64(1))


def test_metal_missing_predicate_matches_columns_literals_and_negation() raises:
    var frame = DataFrame(
        [
            Series("a", Column[Int32]([0, 1])),
            Series("b", Column[Int32]([1, 0])),
            Series("f", Column[Float32]([0, 1])),
            Series("g", Column[Float32]([1, 0])),
        ]
    )
    var nan = bitcast[DType.float32](UInt32(0x7FC12345))
    var checked = 0
    for predicate in [
        col("a").eq_missing(col("b")),
        col("a").ne_missing(col("b")),
        col("a").eq_missing(0),
        ~(col("a").eq_missing(0)),
        col("a").ne_missing(0),
        col("a").eq_missing(null(DataType.INT32)),
        null(DataType.NULL).ne_missing(col("a")),
        col("f").eq_missing(col("g")),
        col("f").ne_missing(col("g")),
        col("f").eq_missing(0.0),
        col("f").eq_missing(lit(nan)),
        ~(col("f").eq_missing(lit(nan))),
        col("f").ne_missing(null(DataType.FLOAT32)),
    ]:
        assert_true(
            Bool(compile_predicate(predicate, frame)),
            "predicate " + String(checked),
        )
        checked += 1
    from dataframe_metal.predicate_plan import plan_predicates, predicate_key

    var ordinary = plan_predicates(
        frame.lazy().filter(col("a") == 0).select(col("b").sum()),
        allow_simple=True,
    )
    var total = plan_predicates(
        frame.lazy().filter(col("a").eq_missing(0)).select(col("b").sum()),
        allow_simple=True,
    )
    assert_true(Bool(ordinary) and Bool(total))
    assert_true(predicate_key(ordinary.value()) != predicate_key(total.value()))
    assert_false(Bool(compile_predicate(col("a").eq_missing(col("f")), frame)))
    assert_false(
        Bool(
            compile_predicate(
                col("a").eq_missing(null(DataType.FLOAT64)), frame
            )
        )
    )
    assert_false(
        Bool(compile_predicate(col("f").eq_missing(lit(Float64(nan))), frame))
    )


# requires: metal
def test_metal_null_fills_public_count_validity_and_shared_aggregates() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var nan = bitcast[DType.float32](UInt32(0x7FC12345))
    var frame = DataFrame(
        [
            Series(
                "a",
                Column[Int32](
                    [1, 2, 3, 4, 5, 6], [True, False, True, False, True, False]
                ),
            ),
            Series(
                "b",
                Column[Int32](
                    [1, 0, 3, 0, 7, 0], [True, False, False, True, True, False]
                ),
            ),
            Series(
                "f",
                Column[Float32](
                    [nan, 1, 2, 7, nan, 0.5],
                    [True, False, True, True, True, True],
                ),
            ),
            Series(
                "g",
                Column[Float32](
                    [5, 3, 2, 1, 4, 6], [True, True, True, True, False, True]
                ),
            ),
            Series("key", Column[Int64]([2, 1, 2, 1, 2, 1])),
        ]
    )
    var chunked = List[Series]()
    for name in frame.columns():
        var source = frame.column(name)
        chunked.append(
            Series._from_chunks(
                [source.slice(0, 1), source.slice(1, 3), source.slice(4, 2)]
            )
        )
    frame = DataFrame(chunked^)
    var counted = [
        col("f").fill_nan(null(DataType.FLOAT32)),
        col("f").fill_nan(col("g")),
        col("f").fill_null(col("g")),
        when(col("a").eq_missing(col("b")))
        .then(col("f").fill_nan(null(DataType.FLOAT32)))
        .otherwise(col("g").fill_null(1)),
    ]
    var expected_counts: List[Int64] = [3, 4, 6, 4]
    for i in range(len(counted)):
        var query = frame.lazy().select(counted[i].count().alias("n"))
        var cpu = query.collect(engine="cpu")
        assert_equal(cpu.item(0, "n").int64(), expected_counts[i])
        assert_true(execute_timed(query)[0].equals(cpu))
        assert_true(query.collect(accelerator=runtime).equals(cpu))
    for value in [
        col("a").fill_null(col("b")),
        col("f").fill_nan(null(DataType.FLOAT32)).fill_null(2),
    ]:
        var outputs: List[Expr] = [
            value.sum().alias("s"),
            value.count().alias("n"),
            value.mean().alias("avg"),
            value.min().alias("lo"),
            value.max().alias("hi"),
        ]
        for grouped in [False, True]:
            var query = (
                frame.lazy()
                .group_by("key", maintain_order=True)
                .agg(outputs) if grouped else frame.lazy()
                .select_exprs(outputs)
            )
            var cpu = query.collect(engine="cpu")
            assert_true(execute_timed(query)[0].equals(cpu))
            assert_true(query.collect(accelerator=runtime).equals(cpu))


# requires: metal
def test_metal_missing_comparisons_public_total_nan_null_and_boolean_composition() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var first = List[Float32]()
    var second = List[Float32]()
    for raw in [
        UInt32(0x7FC12345),
        UInt32(0x7F800001),
        UInt32(0x80000000),
        UInt32(0),
        UInt32(0x7F800000),
        UInt32(0xFF800000),
        UInt32(0x3F800000),
        UInt32(1),
    ]:
        first.append(bitcast[DType.float32](raw))
    for raw in [
        UInt32(0xFFC54321),
        UInt32(0x3F800000),
        UInt32(0),
        UInt32(0x80000000),
        UInt32(0x7F800000),
        UInt32(0x7F800000),
        UInt32(0x40000000),
        UInt32(1),
    ]:
        second.append(bitcast[DType.float32](raw))
    var frame = DataFrame(
        [
            Series(
                "f",
                Column[Float32](
                    first^, [True, True, True, True, True, True, False, False]
                ),
            ),
            Series(
                "g",
                Column[Float32](
                    second^, [True, True, True, False, True, True, False, True]
                ),
            ),
            Series("v", Column[Int64]([1, 2, 3, 4, 5, 6, 7, 8])),
            Series("key", Column[Int32]([2, 1, 2, 1, 2, 1, 2, 1])),
        ]
    )
    var nan = bitcast[DType.float32](UInt32(0x7FC99999))
    var predicates = [
        col("f").eq_missing(col("g")),
        col("f").ne_missing(col("g")),
        ~(col("f").eq_missing(col("g"))),
        col("f").eq_missing(0.0),
        col("f").ne_missing(0.0),
        col("f").eq_missing(lit(nan)),
        ~(col("f").eq_missing(lit(nan))),
        col("f").eq_missing(null(DataType.FLOAT32)),
        col("f").ne_missing(null(DataType.NULL)),
        col("f").eq_missing(col("g")) | col("f").is_null(),
    ]
    # Hand-enumerated totals independently guard CPU/Metal agreement.
    var expected_sums: List[Int64] = [16, 20, 20, 7, 29, 3, 33, 15, 21, 24]
    var expected_counts: List[Int64] = [4, 4, 4, 2, 6, 2, 6, 2, 6, 5]
    for index in range(len(predicates)):
        var predicate = predicates[index].copy()
        for grouped in [False, True]:
            var filtered = frame.lazy().filter(predicate)
            var outputs: List[Expr] = [
                col("v").sum().alias("s"),
                col("v").count().alias("n"),
            ]
            var query = filtered.group_by("key", maintain_order=True).agg(
                outputs
            ) if grouped else filtered.select_exprs(outputs)
            var cpu = query.collect(engine="cpu")
            if not grouped:
                assert_equal(cpu.item(0, "s").int64(), expected_sums[index])
                assert_equal(cpu.item(0, "n").int64(), expected_counts[index])
            assert_true(execute_timed(query)[0].equals(cpu))
            assert_true(query.collect(accelerator=runtime).equals(cpu))


# requires: metal
def test_metal_null_fills_keep_eager_checked_operands_and_lazy_case_branches() raises:
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series("a", Column[Int64]([-1, -2])),
            Series("b", Column[Int64]([Int64.MAX, Int64.MAX])),
        ]
    )
    same_outcome(frame.lazy().select(col("a").fill_null(col("b") + 1).count()))
    var guarded = (
        when(col("a") > 0)
        .then(col("a").fill_null(col("b") + 1))
        .otherwise(col("a"))
    )
    var query = frame.lazy().select(guarded.sum())
    var cpu = query.collect(engine="cpu")
    assert_true(execute_timed(query)[0].equals(cpu))
    assert_true(query.collect(accelerator=runtime).equals(cpu))
