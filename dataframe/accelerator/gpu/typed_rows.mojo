"""Native-width Float32 row programs with packed validity.

Every Float32 arithmetic node retains the existing software binary64 operation
and binary32 rounding boundary. Native storage does not imply native arithmetic
or permission to contract operations. One invocation owns eight output rows so
packed validity writes never race.
"""

from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives.block import prefix_sum
from std.memory import Pointer, bitcast, unsafe_stack_allocation
from dataframe import DataType
from dataframe.accelerator.rows import RowPlan
from dataframe.expr import COL, LIT_FLOAT, ADD, SUB, MUL, GT, LT, GE, LE, EQ, NE
from . import floating


def supports(plan: RowPlan) -> Bool:
    """Certify the entire program before allocating any native device storage."""
    if (
        plan.source.width() != 1
        or plan.source.height() > Int(Int32.MAX)
        or plan.source._columns[0].dtype() != DataType.FLOAT32
        or plan.slots < 1
        or plan.slots > 128
        or len(plan.outputs) != 1
        or plan.outputs[0].dtype != DataType.FLOAT32
        or plan.reductions
        or plan.ordered_mode
        or plan.decimal_feed_batch_rows
        or len(plan.source_text)
        or len(plan.output_text)
        or len(plan.code) != 4 * len(plan.literals)
        or len(plan.node_dtypes) != len(plan.literals)
        or len(plan.slot_dtypes) != plan.slots
    ):
        return False
    var defined = List[Bool](length=plan.slots, fill=False)
    defined[0] = True
    for step in plan.steps:
        if (
            step.order
            or step.grouped
            or step.nodes < 1
            or step.nodes > 128
            or step.start < 0
            or step.start > len(plan.literals) - step.nodes
            or step.slot < 1
            or step.slot >= plan.slots
            or defined[step.slot]
        ):
            return False
        for node in range(step.nodes):
            var at = step.start + node
            var op = Int(plan.code[4 * at])
            var a = Int(plan.code[4 * at + 1])
            var b = Int(plan.code[4 * at + 2])
            var source = Int(plan.code[4 * at + 3])
            var dtype = plan.node_dtypes[at]
            if op == COL:
                if (
                    source < 0
                    or source >= plan.slots
                    or not defined[source]
                    or dtype != plan.slot_dtypes[source]
                    or dtype != DataType.FLOAT32
                ):
                    return False
            elif op == LIT_FLOAT:
                if dtype != DataType.FLOAT32:
                    return False
            elif (
                op == ADD
                or op == SUB
                or op == MUL
                or op == GT
                or op == LT
                or op == GE
                or op == LE
                or op == EQ
                or op == NE
            ):
                if (
                    a < 0
                    or a >= node
                    or b < 0
                    or b >= node
                    or plan.node_dtypes[step.start + a] != DataType.FLOAT32
                    or plan.node_dtypes[step.start + b] != DataType.FLOAT32
                ):
                    return False
                var arithmetic = op == ADD or op == SUB or op == MUL
                if dtype != (DataType.FLOAT32 if arithmetic else DataType.BOOL):
                    return False
            else:
                return False
        if plan.slot_dtypes[step.slot] != plan.node_dtypes[
            step.start + step.nodes - 1
        ] or (step.filter and plan.slot_dtypes[step.slot] != DataType.BOOL):
            return False
        defined[step.slot] = True
    var output = plan.outputs[0].slot
    return (
        output >= 0
        and output < plan.slots
        and defined[output]
        and plan.slot_dtypes[output] == DataType.FLOAT32
        and plan.outputs[0].reduction < 0
    )


def evaluate[
    nullable: Bool, filtered: Bool
](
    source: Pointer[Float32, ImmutAnyOrigin],
    source_valid: Pointer[UInt8, ImmutAnyOrigin],
    source_bit_offset: UInt32,
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt32, ImmutAnyOrigin],
    steps: Pointer[Int32, ImmutAnyOrigin],
    step_count: UInt32,
    output_slot: UInt32,
    rows: UInt32,
    result: Pointer[Float32, MutAnyOrigin],
    result_valid: Pointer[UInt8, MutAnyOrigin],
    selected: Pointer[UInt8, MutAnyOrigin],
):
    var byte = global_idx.x
    if byte * 8 >= Int(rows):
        return
    var output_bits = UInt8(0)
    var selected_bits = UInt8(0)
    var slots = unsafe_stack_allocation[128, DType.uint32]()
    var slot_valid = unsafe_stack_allocation[128, DType.uint8]()
    var values = unsafe_stack_allocation[128, DType.uint32]()
    var valid = unsafe_stack_allocation[128, DType.uint8]()
    for bit in range(8):
        var row = byte * 8 + bit
        if row >= Int(rows):
            break
        var present = True
        comptime if nullable:
            var at = Int(source_bit_offset) + row
            present = (
                source_valid[unsafe_offset=at // 8] >> UInt8(at % 8)
            ) & 1 != 0
        slots[unsafe_offset=0] = bitcast[DType.uint32](
            source[unsafe_offset=row]
        ) if present else 0
        slot_valid[unsafe_offset=0] = UInt8(present)
        var keep = True
        for step in range(Int(step_count)):
            var start = Int(steps[unsafe_offset=4 * step])
            var nodes = Int(steps[unsafe_offset=4 * step + 1])
            var target = Int(steps[unsafe_offset=4 * step + 2])
            var is_filter = steps[unsafe_offset=4 * step + 3] != 0
            for node in range(nodes):
                var at = start + node
                var op = code[unsafe_offset=4 * at]
                var a = Int(code[unsafe_offset=4 * at + 1])
                var b = Int(code[unsafe_offset=4 * at + 2])
                var value = UInt32(0)
                var available = True
                if op == COL:
                    var column = Int(code[unsafe_offset=4 * at + 3])
                    value = slots[unsafe_offset=column]
                    available = slot_valid[unsafe_offset=column] != 0
                elif op == LIT_FLOAT:
                    value = literals[unsafe_offset=at]
                else:
                    available = (
                        valid[unsafe_offset=a] != 0
                        and valid[unsafe_offset=b] != 0
                    )
                    if available:
                        var x = floating.from_float32(values[unsafe_offset=a])
                        var y = floating.from_float32(values[unsafe_offset=b])
                        if op == ADD:
                            value = floating.to_float32(floating.add(x, y))
                        elif op == SUB:
                            value = floating.to_float32(floating.sub(x, y))
                        elif op == MUL:
                            value = floating.to_float32(floating.mul(x, y))
                        elif op == EQ:
                            value = UInt32(floating.equal(x, y))
                        elif op == NE:
                            value = UInt32(not floating.equal(x, y))
                        elif op == LT:
                            value = UInt32(floating.less(x, y))
                        elif op == GT:
                            value = UInt32(floating.less(y, x))
                        elif op == LE:
                            value = UInt32(
                                floating.less(x, y) or floating.equal(x, y)
                            )
                        elif op == GE:
                            value = UInt32(
                                floating.less(y, x) or floating.equal(x, y)
                            )
                values[unsafe_offset=node] = value
                valid[unsafe_offset=node] = UInt8(available)
            slots[unsafe_offset=target] = values[unsafe_offset=nodes - 1]
            slot_valid[unsafe_offset=target] = valid[unsafe_offset=nodes - 1]
            if is_filter:
                keep = (
                    keep
                    and valid[unsafe_offset=nodes - 1] != 0
                    and values[unsafe_offset=nodes - 1] != 0
                )
        result[unsafe_offset=row] = bitcast[DType.float32](
            slots[unsafe_offset=Int(output_slot)]
        )
        output_bits |= UInt8(
            slot_valid[unsafe_offset=Int(output_slot)] != 0
        ) << UInt8(bit)
        selected_bits |= UInt8(keep) << UInt8(bit)
    comptime if nullable:
        result_valid[unsafe_offset=byte] = output_bits
    comptime if filtered:
        selected[unsafe_offset=byte] = selected_bits


def selection_scan(
    selected: Pointer[UInt8, ImmutAnyOrigin],
    positions: Pointer[UInt32, MutAnyOrigin],
    blocks: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
):
    var row = global_idx.x
    var keep = UInt32(0)
    if row < Int(rows):
        keep = UInt32((selected[unsafe_offset=row // 8] >> UInt8(row % 8)) & 1)
    var prefix = prefix_sum[block_size=256](keep)
    if row < Int(rows):
        positions[unsafe_offset=row] = prefix - keep
    if thread_idx.x == 255:
        blocks[unsafe_offset=block_idx.x] = prefix


def selection_offsets(
    blocks: Pointer[UInt32, MutAnyOrigin],
    total: Pointer[UInt32, MutAnyOrigin],
    count: UInt32,
):
    var offset = UInt32(0)
    for block in range(Int(count)):
        var length = blocks[unsafe_offset=block]
        blocks[unsafe_offset=block] = offset
        offset += length
    total[unsafe_offset=0] = offset


def selection_indices(
    selected: Pointer[UInt8, ImmutAnyOrigin],
    positions: Pointer[UInt32, ImmutAnyOrigin],
    blocks: Pointer[UInt32, ImmutAnyOrigin],
    indices: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
):
    var row = global_idx.x
    if (
        row < Int(rows)
        and (selected[unsafe_offset=row // 8] >> UInt8(row % 8)) & 1
    ):
        var target = (
            positions[unsafe_offset=row] + blocks[unsafe_offset=row // 256]
        )
        indices[unsafe_offset=Int(target)] = UInt32(row)


def gather[
    nullable: Bool
](
    values: Pointer[Float32, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    indices: Pointer[UInt32, ImmutAnyOrigin],
    result: Pointer[Float32, MutAnyOrigin],
    result_valid: Pointer[UInt8, MutAnyOrigin],
    rows: UInt32,
):
    var byte = global_idx.x
    if byte * 8 >= Int(rows):
        return
    var bits = UInt8(0)
    for bit in range(8):
        var target = byte * 8 + bit
        if target >= Int(rows):
            break
        var source = Int(indices[unsafe_offset=target])
        result[unsafe_offset=target] = values[unsafe_offset=source]
        comptime if nullable:
            bits |= (
                (validity[unsafe_offset=source // 8] >> UInt8(source % 8)) & 1
            ) << UInt8(bit)
    comptime if nullable:
        result_valid[unsafe_offset=byte] = bits
