"""Literal CPU/GPU EXPLODE proof; requires separate logical-count CPU overlay."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, ListColumn, Series, col, lit
from dataframe.accelerator.gpu.residency import Scope, Table
from dataframe.accelerator.gpu.host_list_reference import stage_child, list_ref
from dataframe.accelerator.gpu.device_lists import (
    stage_descriptors,
    empty_grouped,
)
from dataframe.accelerator.gpu.resident_fields import (
    ResidentField,
    ResidentFrame,
)
from dataframe.accelerator.gpu.explode import execute as explode
from dataframe.accelerator.gpu.runtime import (
    _preflight,
    execute_plan,
    materialize,
)
from dataframe.accelerator.joins import lower_join_dag
from dataframe.accelerator.gpu.runtime import capabilities
from dataframe.accelerator.gpu.join_runtime import execute_plan as execute_join
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source() raises -> DataFrame:
    var child = Series(
        "item", Column[Int64]([7, 8, 9, 10], [True, True, False, True])
    )
    var values: List[Int128] = [
        Int128(1) << 80,
        -(Int128(1) << 80),
        0,
        (Int128(1) << 80) + 7,
    ]
    var wide = Series(
        "item", Column[Int128](values^, List[Bool]([True, True, False, True]))
    ).with_dtype(DataType.decimal(38, 0))
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    child._field_metadata = ArcPointer(metadata^)
    wide._field_metadata = child._field_metadata
    var offsets: List[Int64] = [0, 2, 2, 4, 4]
    var bits: List[UInt8] = [6]
    return DataFrame(
        [
            Series("xs", ListColumn(offsets.copy(), child^, bits.copy())),
            Series("wide", ListColumn(offsets^, wide^, bits^)),
            Series("id", Column[Int64]([10, 11, 12, 13])),
        ]
    )


def _stage(
    source: DataFrame,
    scope: Scope,
    child_faults: Bool = False,
    parent_fault: Bool = False,
) raises -> ResidentFrame:
    var fields = List[ResidentField]()
    for input in source._columns:
        var field = input.rechunk()
        if field.dtype().is_list():
            var child = field.list_column().child()
            if not len(field):
                fields.append(
                    ResidentField.list(
                        empty_grouped(child.slice(0, 0), scope, field.name())
                    )
                )
                continue
            var faults = List[UInt32]()
            if child_faults and field.name() == "xs":
                faults = [UInt32(1), UInt32(1), UInt32(32), UInt32(0)]
            var device = stage_child(child, scope, faults)
            var reference = list_ref(field, device, scope)
            var parents = List[UInt32]()
            if parent_fault and field.name() == "xs":
                parents = [UInt32(8), UInt32(0), UInt32(0), UInt32(0)]
            var lists = stage_descriptors(reference, scope, parents)
            fields.append(ResidentField.list(lists))
        else:
            var device = stage_child(field, scope)
            fields.append(ResidentField.primitive(device))
    return ResidentFrame(scope.copy(), source.height(), fields^)


def layouts_and_resident_consumers() raises:
    var original = _source()
    for variant in range(3):
        var source = (
            original.copy() if variant
            == 0 else offset_layout(original) if variant
            == 1 else chunk_layout(original)
        )
        var scope = Scope()
        var input = _stage(source, scope)
        var before = scope.transfers()
        var result = explode(input, ["xs", "wide"], scope)
        var after = scope.transfers()
        assert_equal(after[0], before[0])
        assert_equal(after[1] - before[1], 16 * 3 + 4 * source.height())
        var actual = materialize(result)
        assert_true(actual.equals(source.explode(["xs", "wide"])))
        assert_equal(
            actual.column("id").int64().to_list(),
            List[Int64](
                [Int64(10), Int64(11), Int64(12), Int64(12), Int64(13)]
            ),
        )
        for row in range(5):
            assert_equal(actual.item(row, "xs").is_null(), row != 3)
            assert_equal(actual.item(row, "wide").is_null(), row != 3)
        assert_equal(actual.item(3, "xs").int64(), Int64(10))
        assert_equal(
            actual.column("wide")._decimal_raw(3), (Int128(1) << 80) + 7
        )
        assert_true(
            actual.column("wide")._field_metadata.value().ptr()
            == source.column("wide")
            .list_column()
            .child()
            ._field_metadata.value()
            .ptr()
        )
        # A row region consumes the same resident output; no exploded values
        # are uploaded again. Its terminal reduction then feeds ordinary grouping.
        var query = result.schema.lazy().select_exprs(
            [
                col("id"),
                (col("xs").cast(DataType.INT32) + lit(Int32(1))).alias("plus"),
            ]
        )
        var plan = _preflight(query, source_rows=result.rows)
        var computed = execute_plan(
            plan, scope, Optional(result.copy()), export_values=False
        )
        var host = materialize(computed[0])
        assert_equal(host.item(3, "plus").numeric[DType.int32](), Int32(11))
        var grouped = _preflight(
            computed[0]
            .schema.lazy()
            .group_by("id", maintain_order=True)
            .agg(col("plus").count().alias("count")),
            source_rows=computed[0].rows,
        )
        var counts = execute_plan(
            grouped, scope, Optional(computed[0].copy()), export_values=False
        )
        var counted = materialize(counts[0])
        assert_equal(
            counted.column("count").int64().to_list(),
            List[Int64]([Int64(0), Int64(0), Int64(1), Int64(0)]),
        )
        var right = DataFrame(
            [
                Series("id", Column[Int64]([12, 10])),
                Series("tag", Column[Int32]([120, 100])),
            ]
        )
        var joined_query = result.schema.lazy().join(
            right.clear().lazy(), on=["id"], how="inner"
        )
        var dag = lower_join_dag(joined_query, capabilities())
        var right_values = execute_plan(
            _preflight(right.lazy().select(["id", "tag"])),
            scope,
            export_values=False,
        )
        var joined = execute_join(
            dag.tasks[dag.root],
            result.schema,
            right.clear(),
            scope,
            left_resident=Optional(result.copy()),
            right_resident=Optional(right_values[0].copy()),
            export_values=False,
        )
        var paired = materialize(joined[0])
        assert_equal(
            paired.column("id").int64().to_list(),
            List[Int64]([Int64(10), Int64(12), Int64(12)]),
        )
        assert_equal(
            paired.column("tag").int32().to_list(),
            List[Int32]([Int32(100), Int32(120), Int32(120)]),
        )
        assert_equal(
            paired.column("wide")._decimal_raw(2), (Int128(1) << 80) + 7
        )
        _ = joined^
        _ = right_values^
        _ = counts^
        _ = computed^
        _ = result^
        _ = input^


def errors_mismatch_empty_and_recovery() raises:
    var source = _source()
    var scope = Scope()
    var input = _stage(source, scope, child_faults=True)
    var result = explode(input, ["xs", "wide"], scope)
    # Hidden strict failures under null parent0 never escape; selected null
    # child2 still propagates its negative-power failure to downstream users.
    with assert_raises(contains="nonnegative exponent"):
        result.observe_errors()
    var parent = _stage(source, scope, parent_fault=True)
    with assert_raises(contains="decimal precision"):
        _ = explode(parent, ["xs", "wide"], scope)
    var offsets: List[Int64] = [0, 1, 2, 4, 4]
    var bits: List[UInt8] = [6]
    var bad = source.with_column(
        Series(
            "wide",
            ListColumn(
                offsets^, source.column("wide").list_column().child(), bits^
            ),
        )
    )
    var mismatch = _stage(bad, scope)
    with assert_raises(contains="different element counts at row 1"):
        _ = explode(mismatch, ["xs", "wide"], scope)
    var safe = _stage(source, scope)
    assert_true(
        materialize(explode(safe, ["xs", "wide"], scope)).equals(
            source.explode(["xs", "wide"])
        )
    )
    var foreign = Scope(999999, 0)
    with assert_raises(contains="another query"):
        _ = explode(safe, ["xs", "wide"], foreign)
    assert_equal(foreign.transfers()[2], 0)
    var empty_scope = Scope(999999, 0)
    var empty = _stage(source.clear(), empty_scope)
    var zero = explode(empty, ["xs", "wide"], empty_scope)
    assert_equal(zero.rows, 0)
    assert_equal(empty_scope.info()[1], 0)
    assert_equal(empty_scope.transfers()[2], 0)
    assert_equal(zero.schema.column("wide").dtype(), DataType.decimal(38, 0))


def _run_budget(budget: Int) raises -> Tuple[DataFrame, Int]:
    var scope = Scope(budget=budget)
    var input = _stage(_source(), scope)
    var result = explode(input, ["xs", "wide"], scope)
    var output = materialize(result)
    var peak = scope.info()[1]
    _ = result^
    _ = input^
    return (output^, peak)


def exact_budget_and_mapped_views() raises:
    var initial = _run_budget(-1)
    var exact = _run_budget(initial[1])
    assert_true(exact[0].equals(initial[0]))
    with assert_raises(contains="memory budget"):
        _ = _run_budget(initial[1] - 1)
    # Logical ordinary column1 is a window inside a wider retained matrix.
    var scope = Scope()
    var source = _source()
    var input = _stage(source, scope)
    var padded = DataFrame(
        [
            Series("pad", Column[Int64]([999, 999, 999, 999, 999, 999])),
            Series("id", Column[Int64]([999, 10, 11, 12, 13, 888])),
        ]
    )
    var resident = execute_plan(
        _preflight(padded.lazy().select(["pad", "id"])),
        scope,
        export_values=False,
    )
    var remapped = resident[0].copy()
    remapped.schema = DataFrame([source.column("id").slice(0, 0)])
    remapped.columns = [resident[0].columns[1]]
    remapped.offset = 1
    remapped.rows = 4
    input.fields[2] = ResidentField.primitive(remapped)
    var actual = materialize(explode(input, ["xs", "wide"], scope))
    assert_true(actual.equals(source.explode(["xs", "wide"])))
    # A view keeps the FULL descriptor/child owners; parent offset1 must be
    # applied to offsets, validity and parent errors without shifting child
    # indices twice. Ordinary fields use the matching row window.
    var padded_parents = source.head(1).vstack(source)
    var viewed = _stage(padded_parents, scope)
    viewed.rows = 4
    for index in range(len(viewed.fields)):
        var field = viewed.fields[index].copy()
        if field.lists:
            var value = field.lists.value().copy()
            value.offset = 1
            value.rows = 4
            viewed.fields[index] = ResidentField.list(value)
        else:
            var value = field.table.value().copy()
            value.offset = 1
            value.rows = 4
            viewed.fields[index] = ResidentField.primitive(value)
    var view_result = explode(viewed, ["xs", "wide"], scope)
    assert_true(materialize(view_result).equals(source.explode(["xs", "wide"])))
    _ = view_result^
    _ = viewed^
    _ = remapped^
    _ = resident^
    _ = input^
