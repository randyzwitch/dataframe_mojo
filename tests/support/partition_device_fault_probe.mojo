"""Literal run/state/batch/AST/control-order proof, independent of categories."""
from max.gpu.host import DeviceContext
from std.testing import assert_equal
from dataframe.accelerator.gpu.ordered_faults import pack
from dataframe.accelerator.gpu.partition_faults import (
    operand_partials,
    control_partials,
    finish,
)


def _check(kind: Int, slices: Int) raises:
    var rows = 16385
    var stride = rows + 3
    var groups = 4
    var errors = List[UInt32](length=2 * stride, fill=0)
    var row_phase = List[UInt32](length=rows, fill=0)
    var ordinal = List[UInt32](capacity=rows)
    for row in range(rows):
        ordinal.append(UInt32(row))
    var controls = List[UInt32](length=groups * 2, fill=0)
    var group_phase = List[UInt32](length=groups, fill=0)
    var group_rank: List[UInt32] = [3, 0, 2, 1]
    var operand_stage = UInt32(0)
    var control_stage = UInt32(1)
    var expected = List[UInt32](length=8, fill=0)
    if kind == 1:
        errors[stride] = pack(3, 1)
        errors[stride + 7] = pack(1, 32)
        expected = [1, 0, 0, 0, 1, 7, 7, pack(1, 32)]
    elif kind == 2:
        errors[stride] = pack(3, 1)
        errors[stride + 8192] = pack(1, 32)
        expected = [1, 0, 0, 0, 3, 0, 0, pack(3, 1)]
    elif kind == 3:
        # Measured CPU anchor: first-run duplicate edges precede a later
        # run's operand overflow even when the latter has a lower AST rank.
        controls[1] = 1
        group_rank[0] = 0
        row_phase[0] = 1
        errors[stride] = pack(1, 2)
        expected = [2, 0, 1, 0, 0, 0, 0, 1]
    elif kind == 4:
        controls[1] = 1
        group_phase[0] = 1
        errors[stride] = pack(3, 2)
        expected = [1, 0, 0, 0, 3, 0, 0, pack(3, 2)]
    elif kind == 5:
        # In one run, _full materializes the entire operand before any
        # group's controls, including a fault in a later operand batch.
        controls[1] = 1
        errors[stride + 8192] = pack(9, 1)
        expected = [1, 0, 0, 1, 9, 8192, 8192, pack(9, 1)]
    elif kind == 6:
        controls[1] = 2
        controls[3] = 1
        controls[5] = 2
        expected = [2, 0, 1, 0, 0, 0, 1, 1]
    elif kind == 7:
        # The scheduler's gathered run ordinal, not original source row,
        # determines expression batches within a contiguous OVER job.
        ordinal[0] = 9000
        ordinal[9000] = 0
        errors[stride] = pack(1, 1)
        errors[stride + 9000] = pack(3, 32)
        expected = [1, 0, 0, 0, 3, 0, 9000, pack(3, 32)]
    elif kind == 8:
        row_phase[0] = 1
        errors[stride] = pack(1, 1)
        errors[stride + 16000] = pack(3, 32)
        ordinal[16000] = 0
        expected = [1, 0, 0, 0, 3, 0, 16000, pack(3, 32)]
    elif kind == 9:
        group_phase[1] = 1
        controls[1] = 2
        controls[3] = 1
        expected = [2, 0, 1, 0, 3, 0, 0, 2]
    elif kind == 10:
        # A preceding state control phase precedes a later state's operand
        # phase in the same run; the host assigns explicit bound-state stages.
        operand_stage = 2
        controls[1] = 1
        errors[stride] = pack(1, 32)
        expected = [2, 0, 1, 0, 3, 0, 0, 1]
    elif kind == 11:
        errors[stride + 7] = pack(1, 1)
        errors[stride] = pack(1, 32)
        expected = [1, 0, 0, 0, 1, 0, 0, pack(1, 32)]
    var result = List[UInt32](length=8, fill=0xFFFFFFFF)
    var ctx = DeviceContext()
    var d_errors = ctx.enqueue_create_buffer[DType.uint32](len(errors))
    var d_row_phase = ctx.enqueue_create_buffer[DType.uint32](len(row_phase))
    var d_ordinal = ctx.enqueue_create_buffer[DType.uint32](len(ordinal))
    var d_controls = ctx.enqueue_create_buffer[DType.uint32](len(controls))
    var d_group_phase = ctx.enqueue_create_buffer[DType.uint32](
        len(group_phase)
    )
    var d_group_rank = ctx.enqueue_create_buffer[DType.uint32](len(group_rank))
    var d_result = ctx.enqueue_create_buffer[DType.uint32](len(result))
    var d_partials = ctx.enqueue_create_buffer[DType.uint32](16 * slices)
    try:
        ctx.enqueue_copy(d_errors, errors.unsafe_ptr())
        ctx.enqueue_copy(d_row_phase, row_phase.unsafe_ptr())
        ctx.enqueue_copy(d_ordinal, ordinal.unsafe_ptr())
        ctx.enqueue_copy(d_controls, controls.unsafe_ptr())
        ctx.enqueue_copy(d_group_phase, group_phase.unsafe_ptr())
        ctx.enqueue_copy(d_group_rank, group_rank.unsafe_ptr())
        ctx.enqueue_function[operand_partials](
            d_errors.unsafe_ptr(),
            d_row_phase.unsafe_ptr(),
            d_ordinal.unsafe_ptr(),
            d_partials.unsafe_ptr(),
            UInt32(1),
            UInt32(rows),
            UInt32(stride),
            UInt32(slices),
            operand_stage,
            UInt32(8192),
            grid_dim=(slices + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[control_partials](
            d_controls.unsafe_ptr(),
            d_group_phase.unsafe_ptr(),
            d_group_rank.unsafe_ptr(),
            d_partials.unsafe_ptr(),
            UInt32(groups),
            UInt32(slices),
            UInt32(slices),
            control_stage,
            grid_dim=(slices + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[finish](
            d_partials.unsafe_ptr(),
            d_result.unsafe_ptr(),
            UInt32(2 * slices),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_copy(result.unsafe_ptr(), d_result)
        ctx.synchronize()
    except error:
        ctx.synchronize()
        _ = d_errors^
        _ = d_row_phase^
        _ = d_ordinal^
        _ = d_controls^
        _ = d_group_phase^
        _ = d_group_rank^
        _ = d_result^
        _ = d_partials^
        _ = errors^
        _ = row_phase^
        _ = ordinal^
        _ = controls^
        _ = group_phase^
        _ = group_rank^
        _ = result^
        raise error^
    for at in range(8):
        if result[at] != expected[at]:
            print(
                "PARTITION FAULT MISMATCH",
                kind,
                slices,
                at,
                result[at],
                expected[at],
            )
        assert_equal(result[at], expected[at])
    _ = d_errors^
    _ = d_row_phase^
    _ = d_ordinal^
    _ = d_controls^
    _ = d_group_phase^
    _ = d_group_rank^
    _ = d_result^
    _ = d_partials^
    _ = errors^
    _ = row_phase^
    _ = ordinal^
    _ = controls^
    _ = group_phase^
    _ = group_rank^
    _ = result^


def run_probe() raises:
    for slices in [1, 7, 1024]:
        for kind in range(12):
            _check(kind, slices)
    print("PASS36 literal OVER run/state/batch/AST/control fault contracts")
