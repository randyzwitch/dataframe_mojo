"""Scoped grouped List publication and expansion, without intermediate export."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from dataframe_metal.accelerator.grouped_lists import prepare
from dataframe_metal.accelerator.gpu.grouped_lists_producer import produce
from dataframe_metal.accelerator.gpu.residency import Scope
from dataframe_metal.accelerator.gpu.runtime import (
    execute_plan,
    materialize,
    _preflight,
)
from dataframe_metal.accelerator.gpu.explode import execute as explode
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source() raises -> DataFrame:
    var x = Series(
        "x", Column[Int32]([7, 9, 10, 11], [True, False, True, True])
    )
    var values: List[Int128] = [
        (Int128(1) << 80) + 1,
        0,
        (Int128(1) << 80) + 3,
        (Int128(1) << 80) + 4,
    ]
    var wide = Series(
        "w", Column[Int128](values^, List[Bool]([True, False, True, True]))
    ).with_dtype(DataType.decimal(38, 0))
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    x._field_metadata = ArcPointer(metadata^)
    wide._field_metadata = x._field_metadata
    return DataFrame([Series("k", Column[Int64]([2, 1, 2, 3])), x^, wide^])


def _expressions() -> List[Expr]:
    return [
        col("x").implode().alias("xs"),
        col("w").implode().alias("wide"),
        col("x").sum().alias("sum"),
        col("x").count().alias("count"),
    ]


def _literal(output: DataFrame) raises:
    assert_equal(output.height(), 4)
    var keys: List[Int64] = [2, 2, 1, 3]
    var sums: List[Int32] = [17, 17, 0, 11]
    var counts: List[Int64] = [2, 2, 0, 1]
    var values: List[Int32] = [7, 10, 0, 11]
    for row in range(4):
        assert_equal(output.item(row, "k").int64(), keys[row])
        assert_equal(output.item(row, "sum").numeric[DType.int32](), sums[row])
        assert_equal(output.item(row, "count").int64(), counts[row])
        assert_equal(output.item(row, "xs").is_null(), row == 2)
        assert_equal(output.item(row, "wide").is_null(), row == 2)
        if row != 2:
            assert_equal(
                output.item(row, "xs").numeric[DType.int32](), values[row]
            )
            assert_equal(
                output.column("wide")._decimal_raw(row),
                (Int128(1) << 80)
                + Int128(1 if row == 0 else 3 if row == 1 else 4),
            )


def layouts_slices_mixed_siblings_and_resident_consumers() raises:
    var original = _source()
    for variant in range(3):
        var frame = (
            original.copy() if variant
            == 0 else offset_layout(original) if variant
            == 1 else chunk_layout(original)
        )
        var query = (
            frame.lazy().group_by("k", maintain_order=True).agg(_expressions())
        )
        var plan = prepare(query, ordered_source=True)
        var scope = Scope()
        var staged = execute_plan(plan.source, scope, export_values=False)
        var grouped = produce(plan, staged[0], scope)
        assert_equal(grouped.rows, 3)
        assert_equal(len(grouped.fields), 5)
        assert_true(Bool(grouped.fields[1].lists))
        assert_true(Bool(grouped.fields[2].lists))
        assert_equal(grouped.fields[1].lists.value().child.rows, 4)
        assert_true(
            grouped.fields[1]
            .lists.value()
            .descriptors[]
            .scope.same_query(scope)
        )
        var traffic = scope.transfers()
        var expanded = explode(grouped, ["xs", "wide"], scope)
        assert_equal(scope.transfers()[0], traffic[0])
        var output = materialize(expanded)
        _literal(output)
        assert_true(
            output.equals(
                query.collect(
                    engine="cpu", optimize=False, streaming=False
                ).explode(["xs", "wide"])
            )
        )
        assert_true(
            output.column("xs")._field_metadata.value().ptr()
            == frame.column("x")._field_metadata.value().ptr()
        )
        assert_true(
            output.column("wide")._field_metadata.value().ptr()
            == frame.column("w")._field_metadata.value().ptr()
        )
        # An ordinary row interpreter consumes the same expanded matrix.
        var row_plan = _preflight(
            expanded.schema.lazy().select(
                (col("xs") + lit(Int32(1))).alias("plus")
            ),
            source_rows=expanded.rows,
        )
        var computed = execute_plan(
            row_plan, scope, Optional(expanded.copy()), export_values=False
        )
        var actual = materialize(computed[0])
        assert_equal(actual.item(0, "plus").numeric[DType.int32](), Int32(8))
        assert_equal(actual.item(1, "plus").numeric[DType.int32](), Int32(11))
        assert_true(actual.item(2, "plus").is_null())
        assert_equal(actual.item(3, "plus").numeric[DType.int32](), Int32(12))
        _ = computed^
        _ = expanded^
        _ = grouped^
        _ = staged^
    # Trailing slices stay views of descriptor/child and primitive owners.
    var query = (
        original.lazy()
        .group_by("k", maintain_order=True)
        .agg(_expressions())
        .slice(-2, 1)
    )
    var plan = prepare(query, ordered_source=True)
    var scope = Scope()
    var staged = execute_plan(plan.source, scope, export_values=False)
    var grouped = produce(plan, staged[0], scope)
    assert_equal(grouped.rows, 1)
    assert_equal(grouped.fields[1].lists.value().offset, 1)
    assert_equal(grouped.fields[1].lists.value().child.rows, 4)
    var actual = materialize(explode(grouped, ["xs", "wide"], scope))
    assert_equal(actual.height(), 1)
    assert_equal(actual.item(0, "k").int64(), Int64(1))
    assert_true(actual.item(0, "xs").is_null())
    assert_true(
        actual.equals(
            query.collect(
                engine="cpu", optimize=False, streaming=False
            ).explode(["xs", "wide"])
        )
    )
    _ = grouped^
    _ = staged^


def _run(budget: Int) raises -> Tuple[DataFrame, Int]:
    var query = (
        _source().lazy().group_by("k", maintain_order=True).agg(_expressions())
    )
    var plan = prepare(query, ordered_source=True)
    var scope = Scope(budget=budget)
    var staged = execute_plan(plan.source, scope, export_values=False)
    var grouped = produce(plan, staged[0], scope)
    var expanded = explode(grouped, ["xs", "wide"], scope)
    var output = materialize(expanded)
    var peak = scope.info()[1]
    _ = expanded^
    _ = grouped^
    _ = staged^
    return (output^, peak)


def empty_foreign_owner_checked_phase_order_and_exact_budget() raises:
    var source = _source()
    var empty_plan = prepare(
        source.clear()
        .lazy()
        .group_by("k", maintain_order=True)
        .agg(_expressions()),
        ordered_source=True,
    )
    var empty_scope = Scope(999999, 0)
    var staged_empty = execute_plan(
        empty_plan.source, empty_scope, export_values=False
    )
    var empty = produce(empty_plan, staged_empty[0], empty_scope)
    assert_equal(empty.rows, 0)
    assert_equal(explode(empty, ["xs", "wide"], empty_scope).rows, 0)
    assert_equal(empty_scope.info()[1], 0)
    assert_equal(empty_scope.transfers()[2], 0)
    var plan = prepare(
        source.lazy().group_by("k", maintain_order=True).agg(_expressions()),
        ordered_source=True,
    )
    var scope = Scope()
    var staged = execute_plan(plan.source, scope, export_values=False)
    var other = Scope(999999, 0)
    with assert_raises(contains="another query"):
        _ = produce(plan, staged[0], other)
    assert_equal(other.transfers()[2], 0)
    var first = _run(-1)
    assert_true(_run(first[1])[0].equals(first[0]))
    with assert_raises(contains="memory budget"):
        _ = _run(first[1] - 1)
    _literal(_run(-1)[0])
    # Original aggregate phase priority survives even a zero-row consumer.
    var a = Int128(6)
    for _ in range(37):
        a *= 10
    var values = List[Int128](length=4, fill=a)
    var bad = source.with_column(
        Series("x", Column[Int32]([128, 9, 10, 11]))
    ).with_column(
        Series("w", Column[Int128](values^)).with_dtype(DataType.decimal(38, 0))
    )
    for state_first in [False, True]:
        var expressions = List[Expr]()
        var list = col("x").cast(DataType.INT8).implode().alias("xs")
        var sum = col("w").sum().alias("sum")
        if state_first:
            expressions = [sum.copy(), list.copy()]
        else:
            expressions = [list.copy(), sum.copy()]
        var query = (
            bad.lazy()
            .group_by("k", maintain_order=True)
            .agg(expressions)
            .head(0)
        )
        with assert_raises(
            contains="decimal precision" if state_first else "cast from int32 to int8 failed"
        ):
            _ = query.collect(engine="cpu", optimize=False, streaming=False)
        var failed = prepare(query, ordered_source=True)
        var input = execute_plan(failed.source, scope, export_values=False)
        with assert_raises(
            contains="decimal precision" if state_first else "strict cast failed"
        ):
            _ = produce(failed, input[0], scope)
        _ = input^
    _ = staged^
