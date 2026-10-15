"""Actual ordered row-program plus checked feed phases; no provider claim."""
from max.gpu.host import DeviceContext
from std.testing import assert_equal, assert_true
from std.memory import ArcPointer
from dataframe import Column, DataFrame, DataType, Expr, Series, col, when
from dataframe.expr import CAST, POW, SUM, COUNT, MIN, ARG_MAX
from dataframe.binding import bind
from dataframe.accelerator import ordered_rows
from dataframe.accelerator.gpu.runtime import _preflight, _stage
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.rows import row_program_ordered
from dataframe.accelerator.gpu.reductions import State, STATE_WORDS
from dataframe.accelerator.gpu import grouped_feed as feed
from dataframe.accelerator.gpu.decimal_descriptors import (
    DECIMAL_SUM,
    DECIMAL_MEAN,
    reduction_parameter,
)
from tests.public_contracts.test_grouped_feed_error_order import (
    decimal_feed_frame,
    integer_feed_frame,
    decimal_feed_operand,
    test_grouped_feed_input_batch_precedes_decimal_update_same_batch_only,
    test_grouped_feed_checked_ast_and_integer_sum_finish_order,
    test_grouped_feed_worker_errors_precede_worker_ordered_merges,
)


def run(
    ctx: DeviceContext,
    frame: DataFrame,
    expression: Expr,
    op: Int64,
    target: Int64,
    parameter: Int64,
    expected_category: UInt32,
    expected_stage: UInt32,
    expected_row: UInt32,
    expected_word: UInt64 = 0,
    groups: Int = 1,
    result_slots: Int = 1,
    target_slot: Int = 0,
    expected_high: UInt64 = 0,
) raises:
    # This probe uses real shared lowering and the exact production interpreter.
    # It deliberately defers whole-output observation to CPU-order feed phases.
    var cloned = ordered_rows.occurrences(bind(expression, frame._columns), 128)
    var plan = _preflight(frame.lazy().select(cloned[0].expr))
    assert_equal(len(plan.steps), 1)
    assert_equal(plan.steps[0].nodes, len(cloned[1]))
    # This isolated kernel proof explicitly supplies occurrence ranks even for
    # a single checked input. It grants no new public ordered-mode admission.
    var ranks = cloned[1].copy()
    while len(ranks) < len(plan.literals):
        ranks.append(UInt32(0))
    assert_equal(len(ranks), len(plan.literals))
    plan.ordered_ranks = ArcPointer(ranks^)
    var program = Program(plan)
    var rows = frame.height()
    var cells = rows * plan.slots
    var words = List[UInt64](length=2 * cells, fill=0)
    var validity = List[UInt8](length=cells, fill=0)
    var tokens = List[UInt32](length=cells, fill=0)
    for source in range(plan.source.width()):
        _stage(plan.source._columns[source], rows, source, 0, words, validity)
    var metadata = List[UInt32](length=2 * rows + 2 * groups, fill=0)
    for row in range(rows):
        metadata[row] = UInt32(row % groups)
        metadata[rows + row] = UInt32(row // groups)
        metadata[2 * rows + row % groups] += 1
    for group in range(groups):
        metadata[2 * rows + groups + group] = UInt32(group)
    # row_to_group[rows],group_ordinal[rows],group_sizes[G],active_order[G].
    var control = List[UInt32](length=feed.CONTROL_WORDS, fill=0)
    var result_words = List[UInt64](length=2 * groups * result_slots, fill=0)
    var result_validity = List[UInt8](length=groups * result_slots, fill=0)
    var result_errors = List[UInt32](length=groups * result_slots, fill=0)
    var dw = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var dv = ctx.enqueue_create_buffer[DType.uint8](len(validity))
    var de = ctx.enqueue_create_buffer[DType.uint32](len(tokens))
    var dm = ctx.enqueue_create_buffer[DType.uint32](len(metadata))
    var dc = ctx.enqueue_create_buffer[DType.int64](len(program.code))
    var dl = ctx.enqueue_create_buffer[DType.uint64](len(program.literals))
    var dt = ctx.enqueue_create_buffer[DType.int64](len(program.node_types))
    var dr = ctx.enqueue_create_buffer[DType.uint32](
        len(program.ordered_ranks[])
    )
    var states = ctx.enqueue_create_buffer[DType.uint64](groups * STATE_WORDS)
    var output_words = ctx.enqueue_create_buffer[DType.uint64](
        2 * groups * result_slots
    )
    var output_validity = ctx.enqueue_create_buffer[DType.uint8](
        groups * result_slots
    )
    var output_errors = ctx.enqueue_create_buffer[DType.uint32](
        groups * result_slots
    )
    var status = ctx.enqueue_create_buffer[DType.uint32](feed.CONTROL_WORDS)
    try:
        ctx.enqueue_copy(dw, words.unsafe_ptr())
        ctx.enqueue_copy(dv, validity.unsafe_ptr())
        ctx.enqueue_copy(de, tokens.unsafe_ptr())
        ctx.enqueue_copy(dm, metadata.unsafe_ptr())
        ctx.enqueue_copy(dc, program.code.unsafe_ptr())
        ctx.enqueue_copy(dl, program.literals.unsafe_ptr())
        ctx.enqueue_copy(dt, program.node_types.unsafe_ptr())
        ctx.enqueue_copy(dr, program.ordered_ranks[].unsafe_ptr())
        ctx.enqueue_copy(output_words, result_words.unsafe_ptr())
        ctx.enqueue_copy(output_validity, result_validity.unsafe_ptr())
        ctx.enqueue_copy(output_errors, result_errors.unsafe_ptr())
        ref step = plan.steps[0]
        ctx.enqueue_function[row_program_ordered](
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            de.unsafe_ptr(),
            dc.unsafe_ptr(),
            dl.unsafe_ptr(),
            dt.unsafe_ptr(),
            dr.unsafe_ptr(),
            UInt32(step.start),
            UInt32(step.nodes),
            UInt32(plan.outputs[0].slot),
            UInt32(rows),
            UInt32(rows),
            UInt32(plan.slots),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[feed.initialize](
            states.unsafe_ptr(),
            output_words.unsafe_ptr(),
            output_validity.unsafe_ptr(),
            output_errors.unsafe_ptr(),
            status.unsafe_ptr(),
            dm.unsafe_ptr().unsafe_offset(2 * rows + groups),
            UInt32(groups),
            UInt32(groups),
            UInt32(result_slots),
            UInt32(target_slot),
            grid_dim=1,
            block_dim=1,
        )
        for begin in range(0, rows, 1024):
            ctx.enqueue_function[feed.feed_batch](
                dw.unsafe_ptr(),
                dv.unsafe_ptr(),
                de.unsafe_ptr(),
                dm.unsafe_ptr(),
                dm.unsafe_ptr().unsafe_offset(rows),
                states.unsafe_ptr(),
                status.unsafe_ptr(),
                UInt32(plan.outputs[0].slot),
                UInt32(begin),
                UInt32(min(rows, begin + 1024)),
                UInt32(rows),
                UInt32(plan.slots),
                UInt32(groups),
                UInt32(begin // 1024),
                op,
                program.node_types[step.start + step.nodes - 1],
                parameter,
                grid_dim=1,
                block_dim=1,
            )
        ctx.enqueue_function[feed.finish](
            states.unsafe_ptr(),
            dm.unsafe_ptr().unsafe_offset(2 * rows),
            dm.unsafe_ptr().unsafe_offset(2 * rows + groups),
            output_words.unsafe_ptr(),
            output_validity.unsafe_ptr(),
            output_errors.unsafe_ptr(),
            status.unsafe_ptr(),
            UInt32(groups),
            UInt32(groups),
            UInt32((rows + 1023) // 1024),
            op,
            program.node_types[step.start + step.nodes - 1],
            target,
            parameter,
            UInt32(result_slots),
            UInt32(target_slot),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_copy(control.unsafe_ptr(), status)
        ctx.enqueue_copy(result_words.unsafe_ptr(), output_words)
        ctx.enqueue_copy(result_validity.unsafe_ptr(), output_validity)
        ctx.enqueue_copy(result_errors.unsafe_ptr(), output_errors)
        ctx.synchronize()
    except error:
        ctx.synchronize()
        _ = words^
        _ = validity^
        _ = tokens^
        _ = metadata^
        _ = program^
        _ = control^
        _ = result_words^
        _ = result_validity^
        _ = result_errors^
        _ = dw^
        _ = dv^
        _ = de^
        _ = dm^
        _ = dc^
        _ = dl^
        _ = dt^
        _ = dr^
        _ = states^
        _ = output_words^
        _ = output_validity^
        _ = output_errors^
        _ = status^
        raise error^
    _ = words^
    _ = validity^
    _ = tokens^
    _ = metadata^
    # Validate rank from the actual descriptor of the first narrowing cast.
    if expected_stage == UInt32(feed.INPUT):
        var expected_rank = UInt32.MAX
        for node in range(len(program.node_types)):
            if (
                expected_category == 1
                and program.code[4 * node] == Int64(CAST)
                and program.node_types[node] == 5
            ) or (
                expected_category == 32 and program.code[4 * node] == Int64(POW)
            ):
                expected_rank = program.ordered_ranks[][node]
                break
        assert_true(expected_rank != UInt32.MAX)
        assert_equal(control[2], expected_rank)
    _ = program^
    _ = dw^
    _ = dv^
    _ = de^
    _ = dm^
    _ = dc^
    _ = dl^
    _ = dt^
    _ = dr^
    _ = states^
    _ = output_words^
    _ = output_validity^
    _ = output_errors^
    _ = status^
    assert_equal(control[0], expected_category)
    assert_equal(control[3], expected_stage)
    if expected_category:
        assert_equal(control[1], expected_row)
    else:
        var base = target_slot * groups
        assert_equal(result_words[base], expected_word)
        assert_equal(result_words[result_slots * groups + base], expected_high)
        assert_equal(result_validity[base], UInt8(1))
        assert_equal(result_errors[base], UInt32(0))


def merge_probe(ctx: DeviceContext, scenario: Int, parameter: Int64) raises:
    var first = State()
    var second = State()
    var controls = List[UInt32](length=2 * feed.CONTROL_WORDS, fill=0)
    # Per-job order[2,1] followed by per-job active counts[2].
    var active: List[UInt32] = [0, 0, 1, 1]
    if scenario == 0:
        first.add(
            UInt64(999),
            True,
            UInt32(0),
            UInt64(0),
            DECIMAL_SUM,
            Int64(12),
            UInt64(0),
            parameter,
        )
        first.add(
            UInt64(1),
            True,
            UInt32(0),
            UInt64(1),
            DECIMAL_SUM,
            Int64(12),
            UInt64(0),
            parameter,
        )
        controls[0] = 8
        controls[1] = 1
        controls[3] = feed.UPDATE
        controls[8] = 1
        controls[9] = 65538
        controls[11] = feed.INPUT
    elif scenario == 1:
        controls[0] = 1
        controls[1] = 2
        controls[3] = feed.INPUT
        second.add(
            UInt64(999),
            True,
            UInt32(0),
            UInt64(65536),
            DECIMAL_SUM,
            Int64(12),
            UInt64(0),
            parameter,
        )
        second.add(
            UInt64(1),
            True,
            UInt32(0),
            UInt64(65537),
            DECIMAL_SUM,
            Int64(12),
            UInt64(0),
            parameter,
        )
        controls[8] = 8
        controls[9] = 65537
        controls[11] = feed.UPDATE
    else:
        first.add(
            UInt64(999),
            True,
            UInt32(0),
            UInt64(0),
            DECIMAL_SUM,
            Int64(12),
            UInt64(0),
            parameter,
        )
        second.add(
            UInt64(1),
            True,
            UInt32(0),
            UInt64(65536),
            DECIMAL_SUM,
            Int64(12),
            UInt64(0),
            parameter,
        )
        if scenario == 3:
            controls[8] = 1
            controls[9] = 66560
            controls[11] = feed.INPUT
    var words = List[UInt64](length=2 * STATE_WORDS, fill=0)
    first.store(words.unsafe_ptr(), 0)
    second.store(words.unsafe_ptr(), STATE_WORDS)
    var result = List[UInt32](length=2 * feed.CONTROL_WORDS, fill=0)
    var ds = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var dc = ctx.enqueue_create_buffer[DType.uint32](len(controls))
    var da = ctx.enqueue_create_buffer[DType.uint32](len(active))
    try:
        ctx.enqueue_copy(ds, words.unsafe_ptr())
        ctx.enqueue_copy(dc, controls.unsafe_ptr())
        ctx.enqueue_copy(da, active.unsafe_ptr())
        ctx.enqueue_function[feed.merge_jobs](
            ds.unsafe_ptr(),
            dc.unsafe_ptr(),
            da.unsafe_ptr(),
            da.unsafe_ptr().unsafe_offset(2),
            UInt32(1),
            UInt32(2),
            DECIMAL_SUM,
            Int64(12),
            parameter,
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_copy(result.unsafe_ptr(), dc)
        ctx.synchronize()
    except error:
        ctx.synchronize()
        _ = words^
        _ = controls^
        _ = active^
        _ = result^
        _ = ds^
        _ = dc^
        _ = da^
        raise error^
    _ = words^
    _ = controls^
    _ = active^
    _ = ds^
    _ = dc^
    _ = da^
    assert_equal(result[0], UInt32(8 if scenario == 0 or scenario == 2 else 1))
    assert_equal(
        result[3],
        feed.UPDATE if scenario
        == 0 else feed.MERGE if scenario
        == 2 else feed.INPUT,
    )
    assert_equal(
        result[1],
        UInt32(
            1 if scenario
            == 0 else 2 if scenario
            == 1 else 66560 if scenario
            == 3 else UInt32.MAX
        ),
    )
    assert_equal(result[6], UInt32(1 if scenario >= 2 else 0))


def main() raises:
    test_grouped_feed_input_batch_precedes_decimal_update_same_batch_only()
    test_grouped_feed_checked_ast_and_integer_sum_finish_order()
    test_grouped_feed_worker_errors_precede_worker_ordered_merges()
    print("PASS CPU feed phase anchors")
    var facts = feed.resources(1, 1, 2)
    assert_equal(facts.state_bytes, 64)
    assert_equal(facts.result_bytes, 21)
    assert_equal(facts.control_bytes, 64)
    assert_equal(facts.peak_bytes, 149)
    assert_equal(facts.kernel_launches, 4)
    assert_equal(facts.download_bytes, 32)
    assert_equal(feed.resources(0, 1, 0).peak_bytes, 0)
    var ctx = DeviceContext(api="metal")
    var dtype = DataType.decimal(3, 0, 128)
    var parameter = reduction_parameter(dtype, 0)
    for scenario in range(4):
        merge_probe(ctx, scenario, parameter)
    for op in [DECIMAL_SUM, DECIMAL_MEAN]:
        for fault_row in [2, 1023, 1024]:
            run(
                ctx,
                decimal_feed_frame(fault_row),
                decimal_feed_operand(),
                op,
                Int64(12 if op == DECIMAL_SUM else 11),
                parameter,
                UInt32(1 if fault_row < 1024 else 8),
                UInt32(feed.INPUT if fault_row < 1024 else feed.UPDATE),
                UInt32(fault_row if fault_row < 1024 else 1),
            )
    var safe = decimal_feed_frame(-1).with_column(
        Series(
            "d", Column[Int128](List[Int128](length=1025, fill=0))
        ).with_dtype(dtype)
    )
    run(
        ctx,
        safe,
        decimal_feed_operand(),
        DECIMAL_SUM,
        Int64(12),
        parameter,
        UInt32(0),
        UInt32(0),
        UInt32.MAX,
    )
    run(
        ctx,
        safe,
        decimal_feed_operand(),
        Int64(COUNT),
        Int64(3),
        Int64(0),
        UInt32(0),
        UInt32(0),
        UInt32.MAX,
        UInt64(1025),
    )
    for strict_first in [False, True]:
        var strict = col("x").cast(DataType.INT8).cast(DataType.INT64)
        var power = col("base").pow(col("exp"))
        var checked = strict + power if strict_first else power + strict
        var expression = when(col("keep")).then(checked).otherwise(col("v"))
        var operations: List[Int64] = [
            Int64(SUM),
            Int64(COUNT),
            Int64(MIN),
            Int64(ARG_MAX),
        ]
        for operation in operations:
            run(
                ctx,
                integer_feed_frame(1024),
                expression,
                operation,
                Int64(3 if operation != Int64(ARG_MAX) else 9),
                Int64(0),
                UInt32(1 if strict_first else 32),
                UInt32(feed.INPUT),
                UInt32(1024),
            )
        run(
            ctx,
            integer_feed_frame(-1),
            expression,
            Int64(SUM),
            Int64(3),
            Int64(0),
            UInt32(2),
            UInt32(feed.FINISH),
            UInt32.MAX,
        )
        run(
            ctx,
            integer_feed_frame(-1),
            expression,
            Int64(COUNT),
            Int64(3),
            Int64(0),
            UInt32(0),
            UInt32(0),
            UInt32.MAX,
            UInt64(1025),
        )
    var wide_values = List[Int128](length=1025, fill=0)
    wide_values[0] = Int128(1) << 80
    var wide_type = DataType.decimal(38, 0, 128)
    var wide_frame = decimal_feed_frame(-1).with_column(
        Series("d", Column[Int128](wide_values^)).with_dtype(wide_type)
    )
    var wide_operand = (
        when(col("keep"))
        .then(col("x").cast(DataType.INT8).cast(wide_type))
        .otherwise(col("d"))
    )
    assert_equal(
        wide_frame.group_by("k")
        .agg(wide_operand.sum().alias("total"))
        .column("total")
        ._decimal_raw(0),
        Int128(1) << 80,
    )
    run(
        ctx,
        wide_frame,
        wide_operand,
        DECIMAL_SUM,
        Int64(12),
        reduction_parameter(wide_type, 0),
        UInt32(0),
        UInt32(0),
        UInt32.MAX,
        result_slots=3,
        target_slot=1,
        expected_high=UInt64(1) << 16,
    )
    for fault_row in [1023, 1024]:
        var values = List[Int128](length=1025, fill=0)
        values[0] = 999
        values[1] = 999
        values[2] = 1
        values[3] = 1
        var keys = List[Int32](length=1025, fill=0)
        for row in range(1025):
            keys[row] = Int32(row % 2)
        var two_groups = (
            decimal_feed_frame(fault_row)
            .with_column(Series("d", Column[Int128](values^)).with_dtype(dtype))
            .with_column(Series("k", Column[Int32](keys^)))
        )
        run(
            ctx,
            two_groups,
            decimal_feed_operand(),
            DECIMAL_SUM,
            Int64(12),
            parameter,
            UInt32(1 if fault_row < 1024 else 8),
            UInt32(feed.INPUT if fault_row < 1024 else feed.UPDATE),
            UInt32(fault_row if fault_row < 1024 else 2),
            groups=2,
        )
    # Group0 first occurs at row0, but group1 overflows at row3 before
    # group0 overflows at row4. This independently anchors source-row update
    # priority for an input with no faulting operation of its own.
    var state_values: List[Int128] = [998, 999, 1, 1, 1]
    var state_dtype = DataType.decimal(3, 0, 128)
    var state_frame = DataFrame(
        [
            Series("d", Column[Int128](state_values^)).with_dtype(state_dtype),
            Series("k", Column[Int32]([0, 1, 0, 1, 0])),
        ]
    )
    run(
        ctx,
        state_frame,
        when(col("d").is_null()).then(col("d")).otherwise(col("d")),
        DECIMAL_SUM,
        Int64(12),
        reduction_parameter(state_dtype, 0),
        UInt32(8),
        feed.UPDATE,
        UInt32(3),
        groups=2,
    )
    print(
        "PASS actual interpreter: batch/input/update/finish ordering, mixed ASTs, multiple groups and SUM/MEAN/COUNT/MIN/ARG_MAX"
    )
