"""Native reductions consume retained CPU row mappings without payload gathers."""
# contract-group: execution.backends
from std.memory import bitcast, ArcPointer
from dataframe.execution_observation import DeviceObservation
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, Expr, Series, col
from dataframe.vector.selection_batches import SelectionBatch, SelectionBatches
from dataframe.query_execution.pipeline import plan_selection_batches
from dataframe_metal.selection_plan import plan_selection
from dataframe_metal.selection_reduce import (
    execute_selection_timed,
    selection_execute,
)
from dataframe_metal.accelerator.gpu.selection_inputs import (
    selection_inputs,
    selection_inputs_supported,
)
from dataframe_metal.accelerator.gpu.selection_reduce import SELECTION_WORDS
from tests.support.metal_outcomes import require_metal, same_outcome


def _expressions() -> List[Expr]:
    var expressions = List[Expr]()
    for name in ["i", "l", "f", "constant"]:
        expressions.extend(
            [
                col(name).sum().alias(name + "_sum"),
                col(name).count().alias(name + "_count"),
                col(name).mean().alias(name + "_mean"),
                col(name).min().alias(name + "_min"),
                col(name).max().alias(name + "_max"),
            ]
        )
    expressions.append(col("i").sum().alias("again"))
    return expressions^


def _batches(nulls: Bool) raises -> SelectionBatches:
    var integers = List[Int32]()
    var longs = List[Int64]()
    var floats = List[Float32]()
    var valid = List[Bool]()
    for row in range(137):
        integers.append(Int32(row % 17 - 8))
        longs.append(Int64(row % 19 - 9) * (Int64(1) << 40))
        floats.append(Float32(row % 23 - 11) / 8)
        valid.append(not nulls or row % 11 != 3)
    var source = DataFrame(
        [
            Series("i", Column[Int32](integers^, valid)),
            Series("l", Column[Int64](longs^, valid)),
            Series("f", Column[Float32](floats^, valid)),
            Series("constant", Column[Int64]._repeated(7, 137, not nulls)),
        ]
    )
    var batches = SelectionBatches(source)
    for morsel in range(4):
        var batch = SelectionBatch()
        batch.morsel = morsel
        for column in source._columns:
            batch.columns.append(column.slice(5 + morsel, 120))
        batch.starts = [2, 1, 2, 0]
        batch.lists = [0, 1, 0, 0]
        batch.count = 3017 + morsel
        var left = List[UInt32]()
        var right = List[UInt32]()
        for row in range(batch.count):
            left.append(UInt32((row * 17) % 113))
            right.append(UInt32(112 - (row * 23) % 113))
        batch.rows.append(left^)
        batch.rows.append(right^)
        batches.append(batch^)
    batches.ready = True
    return batches^


def test_metal_selection_input_owners_and_shared_row_lists() raises:
    var batches = _batches(False)
    var region = batches.schema.lazy().select_exprs(_expressions())
    var matched = plan_selection(region._plan)
    assert_true(Bool(matched))
    assert_equal(len(matched.value().fields), 4)
    var inputs = selection_inputs(matched.value(), batches)
    # Three primitive owners are shared by every sliced batch. Constants
    # are represented by one element each, never expanded to source length.
    assert_equal(len(inputs.owners), 7)
    assert_equal(len(inputs.groups), 7)
    assert_equal(inputs.selection_count, 2 * batches.rows)
    var packed = List[UInt32](length=inputs.selection_count, fill=UInt32.MAX)
    inputs.fill_selections(batches, packed.unsafe_ptr())
    for owner in inputs.owners:
        if owner.name() == "constant":
            assert_equal(len(owner), 1)
    assert_true(inputs.partials > 0)
    var batch_offset = 0
    for batch_index in range(len(batches.batches)):
        ref batch = batches.batches[batch_index]
        assert_equal(batch.copied_columns, 0)
        assert_true(batch.columns[3].is_constant())
        for list_index in range(len(batch.rows)):
            for row in range(batch.count):
                var at = batch_offset + list_index * batch.count + row
                assert_equal(packed[at], batch.rows[list_index][row])
        for group_index in range(len(inputs.groups)):
            ref group = inputs.groups[group_index]
            var column = matched.value().fields[group.field].column
            if batch.columns[column].is_constant():
                continue
            var descriptor = (group.first + batch_index) * SELECTION_WORDS
            var expected = batch_offset + batch.lists[column] * batch.count
            assert_equal(inputs.descriptors[descriptor + 4], UInt64(expected))
        batch_offset += len(batch.rows) * batch.count
    for group_index in range(len(inputs.groups)):
        ref group = inputs.groups[group_index]
        if inputs.owners[group.owner].name() == "constant":
            var descriptor = group.first * SELECTION_WORDS
            assert_equal(inputs.descriptors[descriptor + 4], UInt64.MAX - 1)

    var ranges = SelectionBatches(batches.schema)
    var batch = SelectionBatch()
    batch.columns = batches.batches[0].columns.copy()
    batch.starts = [2, 1, 2, 0]
    batch.lists = [-1, -1, -1, -1]
    batch.count = 3
    ranges.append(batch^)
    ranges.ready = True
    var range_inputs = selection_inputs(matched.value(), ranges)
    assert_equal(range_inputs.selection_count, 0)
    var empty = List[UInt32](length=1, fill=UInt32.MAX)
    range_inputs.fill_selections(ranges, empty.unsafe_ptr())
    assert_equal(empty[0], UInt32.MAX)
    for group_index in range(len(range_inputs.groups)):
        ref group = range_inputs.groups[group_index]
        var is_constant = range_inputs.owners[group.owner].name() == "constant"
        var expected = UInt64.MAX - 1 if is_constant else UInt64.MAX
        assert_equal(
            range_inputs.descriptors[group.first * SELECTION_WORDS + 4],
            expected,
        )


def test_metal_selection_entrypoints_reject_invalid_maps_before_submission() raises:
    for malformed_dimensions in [False, True]:
        var batches = _batches(False)
        var region = batches.schema.lazy().select_exprs(_expressions())
        var observation = ArcPointer(DeviceObservation())
        region._plan._device_observation = observation
        var matched = plan_selection(region._plan)
        var expected = "outside its source column"
        if malformed_dimensions:
            batches.batches[0].starts = List[Int]()
            expected = "inconsistent dimensions"
        else:
            batches.batches[0].rows[0][0] = UInt32(120)
        with assert_raises(contains=expected):
            _ = selection_inputs(matched.value(), batches)
        with assert_raises(contains=expected):
            _ = execute_selection_timed(region._plan, batches)
        with assert_raises(contains=expected):
            _ = selection_execute(region._plan, batches)
        assert_equal(observation[].launches.load(), Int64(0))
        assert_equal(observation[].completed_launches.load(), Int64(0))


# requires: metal
def test_metal_selected_reductions_types_slices_nulls_duplicates_and_constants() raises:
    _ = require_metal()
    for nulls in [False, True]:
        var batches = _batches(nulls)
        var expressions = _expressions()
        var region = batches.schema.lazy().select_exprs(expressions)
        var expected = batches.materialize().select_exprs(expressions)
        var measured = execute_selection_timed(region._plan, batches)
        assert_true(measured[0].equals(expected))
        assert_true(measured[1].kernel_ns > 0)
        # A second submission exercises retained sources after the first
        # queue has drained and its device views have been released.
        assert_true(
            execute_selection_timed(region._plan, batches)[0].equals(expected)
        )
        var empty = SelectionBatches(batches.schema)
        empty.ready = True
        assert_true(
            execute_selection_timed(region._plan, empty)[0].equals(
                batches.schema.select_exprs(expressions)
            )
        )

    # A huge logical constant still binds one scalar; the native source
    # address bound must not expand it or reject its selected copies.
    var count = Int(Int32.MAX) + 100
    var source = DataFrame(
        [Series("c", Column[Int64]._repeated(7, count, True))]
    )
    var batch = SelectionBatch()
    batch.columns = source._columns.copy()
    batch.starts = [0]
    batch.lists = [0]
    batch.rows.append([UInt32(0), UInt32(Int32.MAX), UInt32(count - 1)])
    batch.count = 3
    var batches = SelectionBatches(source)
    batches.append(batch^)
    batches.ready = True
    var region = batches.schema.lazy().select(col("c").sum())
    var plan = plan_selection(region._plan)
    assert_true(selection_inputs_supported(plan.value(), batches))
    var actual = execute_selection_timed(region._plan, batches)[0].copy()
    assert_equal(actual.item(0, "c").int64(), Int64(21))
    assert_true(source.column("c").is_constant())


# requires: metal
def test_metal_selected_float_specials_and_integer_faults() raises:
    _ = require_metal()
    var nan = bitcast[DType.float32](UInt32(0x7FC0_1234))
    var negative_zero = bitcast[DType.float32](UInt32(0x8000_0000))
    var cases: List[List[Float32]] = [
        [Float32(1), -Float32(2), Float32(0)],
        [nan, Float32(2), -Float32(3)],
        [negative_zero, negative_zero],
        [Float32(1) / Float32(0), -Float32(1) / Float32(0)],
    ]
    for values in cases:
        var source = DataFrame([Series("f", Column[Float32](values.copy()))])
        var batch = SelectionBatch()
        batch.columns = source._columns.copy()
        batch.starts = [0]
        batch.lists = [0]
        batch.rows.append([UInt32(1), 0, 1])
        batch.count = 3
        var batches = SelectionBatches(source)
        batches.append(batch^)
        batches.ready = True
        var expressions: List[Expr] = [
            col("f").min().alias("min"),
            col("f").max().alias("max"),
        ]
        var region = batches.schema.lazy().select_exprs(expressions)
        var expected = batches.materialize().select_exprs(expressions)
        var actual = execute_selection_timed(region._plan, batches)[0].copy()
        for name in ["min", "max"]:
            assert_equal(
                bitcast[DType.uint32](actual.item(0, name).float32()),
                bitcast[DType.uint32](expected.item(0, name).float32()),
            )
    var source = DataFrame([Series("l", Column[Int64]([Int64.MAX]))])
    var batch = SelectionBatch()
    batch.columns = source._columns.copy()
    batch.starts = [0]
    batch.lists = [0]
    batch.rows.append([UInt32(0), 0])
    batch.count = 2
    var batches = SelectionBatches(source)
    batches.append(batch^)
    batches.ready = True
    var region = batches.schema.lazy().select(col("l").sum())
    with assert_raises(contains="overflow"):
        _ = execute_selection_timed(region._plan, batches)


def _selection_report(report: DataFrame, joined: Bool) raises:
    var found = False
    for row in range(report.height()):
        var algorithm = (
            "handoff integral partials" if joined else "handoff selections"
        )
        if report.item(row, "algorithm").string() == algorithm:
            found = True
            if joined:
                assert_equal(
                    report.item(row, "gpu_completed_launches").int64(), Int64(1)
                )
                assert_equal(
                    report.item(row, "download_bytes").int64(), Int64(3 * 3 * 8)
                )
                assert_true(
                    report.item(row, "upload_bytes").int64() < 70_019 * 4
                )
            else:
                assert_equal(report.item(row, "executions").int64(), Int64(1))
    assert_true(
        found, "expected a native selection handoff in the execution report"
    )


# requires: metal
def test_metal_selected_cpu_string_filter_and_join_use_native_reducers() raises:
    var runtime = require_metal()
    var labels = List[String]()
    var keys = List[Int64]()
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(70_019):
        labels.append("keep" if row % 7 < 3 else "drop")
        keys.append(Int64(row % 13))
        values.append(Int64(row % 101 - 50))
        valid.append(row % 17 != 3)
    var source = DataFrame(
        [
            Series("label", Column[String](labels^)),
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^, valid)),
        ]
    )
    var dimension = DataFrame(
        [
            Series("k", Column[Int64]([1, 1, 3, 8])),
            Series("w", Column[Int32]([10, 20, 30, 80])),
        ]
    )
    for joined in [False, True]:
        var producer = source.lazy().filter(col("label") == "keep")
        var expressions: List[Expr] = [
            col("v").sum().alias("sum"),
            col("v").count().alias("count"),
        ]
        if joined:
            producer = producer.join(dimension.lazy(), on="k")
            expressions.append(col("w").sum().alias("weight"))
        var batches = plan_selection_batches(
            producer._plan, len(producer._plan._nodes) - 1, 4096
        )
        assert_true(not batches[].fallback)
        for batch_index in range(len(batches[].batches)):
            ref batch = batches[].batches[batch_index]
            assert_equal(batch.copied_columns, 0)
        var query = producer.select_exprs(expressions)
        var expected = query.collect(engine="cpu")
        for optimize in [False, True]:
            var actual = query.profile(
                accelerator=runtime, optimize=optimize, batch_size=4096
            )
            assert_true(actual[0].equals(expected))
            _selection_report(actual[1], joined)
        assert_true("HANDOFF metal" in query.explain(accelerator=runtime))
    var overflow = (
        DataFrame(
            [
                Series("label", Column[String](["keep", "keep"])),
                Series("v", Column[Int64]([Int64.MAX, 1])),
            ]
        )
        .lazy()
        .filter(col("label") == "keep")
        .select(col("v").sum())
    )
    same_outcome(overflow)


# requires: metal
def test_metal_join_partial_carry_and_null_counts() raises:
    from dataframe_metal.integral_partials import combine_integrals

    # Two worker rows, each with a wide signed sum and a nullable count.
    # -1 + 2 wraps the low word and cancels the negative high word.
    _ = require_metal()
    var packed: List[UInt64] = [
        UInt64.MAX,
        UInt64.MAX,
        1,
        5,
        0,
        0,
        2,
        0,
        2,
        3,
        0,
        0,
    ]
    var actual = combine_integrals(packed, 2, None)
    var expected: List[UInt64] = [1, 0, 3, 8, 0, 0]
    assert_equal(actual, expected)
