"""Exercise future rolling kernels without enabling them in the main runtime."""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.testing import assert_true, assert_raises
from dataframe import DataFrame, Series, LazyFrame
from dataframe.metal import _capabilities
from dataframe.accelerator.rows import (
    lower_rows,
    ROW_CUMULATIVE_KEY,
    ROW_DURATION_KEY,
)
from dataframe.accelerator.gpu.runtime import _kind, _stage, _output
from dataframe.accelerator.gpu.rows import row_program
from dataframe.accelerator.gpu.ordering import (
    order_init,
    order_merge,
    merge_grid,
)
from dataframe.accelerator.gpu.windows import (
    window_assign,
    window_input_errors,
    window_check_errors,
    duration_assign,
    duration_check_errors,
    duration_headers,
)
from tests.execution_modes.test_backend_metal_moments import near
from tests.execution_modes.test_mojo_gpu_rolling import (
    _rolling_cases,
    _duration_cases,
    _floating_cases,
    _failure_cases,
)


def _gpu(query: LazyFrame) raises -> DataFrame:
    var caps = _capabilities()
    caps.max_nodes = 128
    caps.max_slots = 256
    caps.max_steps = 128
    var plan = lower_rows(query, caps)
    var rows = plan.source.height()
    var count = rows if plan.limit < 0 else min(rows, plan.limit)
    var cells = rows * plan.slots
    var words = List[UInt64](length=cells, fill=0)
    var valid = List[UInt8](length=cells, fill=0)
    var errors = List[UInt32](length=cells, fill=0)
    var code = plan.code.copy()
    var types = List[Int64]()
    var literals = List[UInt64]()
    for dtype in plan.node_dtypes:
        types.append(_kind(dtype))
    for literal in plan.literals:
        literals.append(bitcast[DType.uint64](literal))
    for slot in range(plan.source.width()):
        _stage(plan.source._columns[slot], rows, slot, 0, words, valid)
    if rows:
        var ctx = DeviceContext()
        var dw = ctx.enqueue_create_buffer[DType.uint64](cells)
        var dv = ctx.enqueue_create_buffer[DType.uint8](cells)
        var de = ctx.enqueue_create_buffer[DType.uint32](cells)
        var dc = ctx.enqueue_create_buffer[DType.int64](len(code))
        var dt = ctx.enqueue_create_buffer[DType.int64](len(types))
        var dl = ctx.enqueue_create_buffer[DType.uint64](len(literals))
        ctx.enqueue_copy(dw, words.unsafe_ptr())
        ctx.enqueue_copy(dv, valid.unsafe_ptr())
        ctx.enqueue_copy(de, errors.unsafe_ptr())
        ctx.enqueue_copy(dc, code.unsafe_ptr())
        ctx.enqueue_copy(dt, types.unsafe_ptr())
        ctx.enqueue_copy(dl, literals.unsafe_ptr())
        var blocks = (rows + 255) // 256
        for step in plan.steps:
            assert_true(not step.filter and not step.grouped)
            if not step.order:
                ctx.enqueue_function[row_program](
                    dw.unsafe_ptr(),
                    dv.unsafe_ptr(),
                    de.unsafe_ptr(),
                    dc.unsafe_ptr(),
                    dl.unsafe_ptr(),
                    dt.unsafe_ptr(),
                    UInt32(step.start),
                    UInt32(step.nodes),
                    UInt32(step.slot),
                    UInt32(rows),
                    UInt32(rows),
                    grid_dim=blocks,
                    block_dim=256,
                )
                continue
            var duration = code[4 * step.start] == ROW_DURATION_KEY
            assert_true(duration or code[4 * step.start] == ROW_CUMULATIVE_KEY)
            var headers = Int(
                duration_headers(
                    code.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    UInt32(step.start),
                    UInt32(step.nodes),
                )
            ) if duration else 1
            var failure_count = blocks * (4 if duration else 1)
            var failures = List[UInt32](length=failure_count, fill=0)
            var df = ctx.enqueue_create_buffer[DType.uint32](failure_count)
            var a = ctx.enqueue_create_buffer[DType.uint32](rows)
            var b = ctx.enqueue_create_buffer[DType.uint32](rows)
            ctx.enqueue_function[window_input_errors](
                de.unsafe_ptr(),
                dc.unsafe_ptr(),
                df.unsafe_ptr(),
                UInt32(step.start),
                UInt32(step.nodes),
                UInt32(rows),
                UInt32(rows),
                grid_dim=blocks,
                block_dim=256,
            )
            ctx.enqueue_function[order_init](
                a.unsafe_ptr(), UInt32(rows), grid_dim=blocks, block_dim=256
            )
            var width = 1
            while width < rows:
                ctx.enqueue_function[order_merge](
                    dw.unsafe_ptr(),
                    dv.unsafe_ptr(),
                    a.unsafe_ptr(),
                    b.unsafe_ptr(),
                    dc.unsafe_ptr(),
                    dt.unsafe_ptr(),
                    UInt32(step.start + headers),
                    UInt32(step.nodes - headers),
                    UInt32(rows),
                    UInt32(rows),
                    UInt32(width),
                    UInt32(0),
                    dl.unsafe_ptr(),
                    grid_dim=merge_grid(rows, width),
                    block_dim=256,
                )
                var previous = a^
                a = b^
                b = previous^
                width *= 2
            ctx.enqueue_copy(failures.unsafe_ptr(), df)
            ctx.synchronize()
            for failure in failures:
                if failure:
                    raise Error("window input overflow")
            if duration:
                ctx.enqueue_function[duration_assign](
                    dw.unsafe_ptr(),
                    dv.unsafe_ptr(),
                    de.unsafe_ptr(),
                    a.unsafe_ptr(),
                    dc.unsafe_ptr(),
                    dt.unsafe_ptr(),
                    dl.unsafe_ptr(),
                    UInt32(step.start),
                    UInt32(step.nodes),
                    UInt32(step.slot),
                    UInt32(_kind(plan.slot_dtypes[step.slot])),
                    UInt32(rows),
                    UInt32(rows),
                    grid_dim=blocks,
                    block_dim=256,
                )
                ctx.enqueue_function[duration_check_errors](
                    de.unsafe_ptr(),
                    df.unsafe_ptr(),
                    UInt32(step.slot),
                    UInt32(rows),
                    UInt32(rows),
                    grid_dim=blocks,
                    block_dim=256,
                )
            else:
                ctx.enqueue_function[window_assign](
                    dw.unsafe_ptr(),
                    dv.unsafe_ptr(),
                    de.unsafe_ptr(),
                    a.unsafe_ptr(),
                    dc.unsafe_ptr(),
                    dt.unsafe_ptr(),
                    dl.unsafe_ptr(),
                    UInt32(step.start),
                    UInt32(step.nodes),
                    UInt32(step.slot),
                    UInt32(_kind(plan.slot_dtypes[step.slot])),
                    UInt32(rows),
                    UInt32(rows),
                    grid_dim=blocks,
                    block_dim=256,
                )
                ctx.enqueue_function[window_check_errors](
                    de.unsafe_ptr(),
                    df.unsafe_ptr(),
                    UInt32(step.slot),
                    UInt32(rows),
                    UInt32(rows),
                    grid_dim=blocks,
                    block_dim=256,
                )
            ctx.enqueue_copy(failures.unsafe_ptr(), df)
            ctx.synchronize()
            if duration:
                for block in range(blocks):
                    if failures[4 * block]:
                        raise Error("time index contains null")
                    if failures[4 * block + 1]:
                        raise Error("time index not sorted")
                    if failures[4 * block + 2]:
                        raise Error("time bounds overflow")
                    if failures[4 * block + 3]:
                        raise Error("duration arithmetic overflow")
            else:
                for failure in failures:
                    if failure:
                        raise Error("rolling arithmetic overflow")
        ctx.enqueue_copy(words.unsafe_ptr(), dw)
        ctx.enqueue_copy(valid.unsafe_ptr(), dv)
        ctx.enqueue_copy(errors.unsafe_ptr(), de)
        ctx.synchronize()
    var columns = List[Series]()
    for output in plan.outputs:
        columns.append(
            _output(
                output.name,
                output.dtype,
                output.slot,
                rows,
                count,
                words,
                valid,
            )
        )
    return DataFrame(columns^, height=count)


def main() raises:
    var checked = 0
    for query in _rolling_cases() + _duration_cases() + _floating_cases():
        print("rolling query", checked)
        near(_gpu(query), query.collect(engine="cpu", streaming=False))
        checked += 1
    for pair in _failure_cases():
        with assert_raises(contains=pair[1]):
            _ = _gpu(pair[0])
        checked += 1
    print(
        "PASS Mojo GPU rolling/interpolation/time-bound query probe:", checked
    )
