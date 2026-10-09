"""Resident row expressions and stable, byte-race-free compaction primitives."""
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer
from max.gpu.sync import barrier
from std.memory import AddressSpace, stack_allocation
from dataframe.accel_rows import ROW_MAX_NODES, ROW_CODE_WORDS
from dataframe.accel_memory import ACCEL_THREADS
from dataframe.expr import (
    COL,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_NULL,
    ADD,
    SUB,
    MUL,
    NEG,
    GT,
    LT,
    GE,
    LE,
    EQ,
    NE,
    AND,
    OR,
    XOR,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    FILL_NULL,
)
from dataframe.float_ops import arithmetic, compare
from .reductions import reduce_pair

comptime THREADS = ACCEL_THREADS


@always_inline
def evaluate_row[
    D: DType
](
    values: DeviceBuffer[D].device_type,
    valid: DeviceBuffer[DType.uint8].device_type,
    code: DeviceBuffer[DType.int64].device_type,
    literals: DeviceBuffer[DType.float64].device_type,
    capacity: Int,
    row: Int,
    start: Int,
    count: Int,
) -> Tuple[Scalar[D], Bool]:
    var local = stack_allocation[ROW_MAX_NODES, D]()
    var present = stack_allocation[ROW_MAX_NODES, DType.uint8]()
    for n in range(count):
        var at = (start + n) * ROW_CODE_WORDS
        var op = Int(code[unsafe_offset=at])
        var left = Int(code[unsafe_offset=at + 1])
        var right = Int(code[unsafe_offset=at + 2])
        var a = Scalar[D](0)
        var b = Scalar[D](0)
        var av = False
        var bv = False
        if left >= 0:
            a = local[unsafe_offset=left]
            av = present[unsafe_offset=left] != 0
        if right >= 0:
            b = local[unsafe_offset=right]
            bv = present[unsafe_offset=right] != 0
        var value = Scalar[D](0)
        var ok = False
        if op == COL:
            var source = Int(code[unsafe_offset=at + 3]) * capacity + row
            ok = valid[unsafe_offset=source] != 0
            if ok:
                value = values[unsafe_offset=source]
        elif op == LIT_FLOAT or op == LIT_BOOL:
            value = literals[unsafe_offset=start + n].cast[D]()
            ok = True
        elif op == LIT_NULL:
            pass
        elif op == IS_NULL or op == IS_NOT_NULL:
            value = UInt8(av if op == IS_NOT_NULL else not av).cast[D]()
            ok = True
        elif op == FILL_NULL:
            ok = av or bv
            value = a if av else b
        elif op == NOT:
            ok = av
            if ok:
                value = UInt8(not Bool(a)).cast[D]()
        elif op == AND:
            var x = av and Bool(a)
            var y = bv and Bool(b)
            ok = (av and bv) or (av and not x) or (bv and not y)
            value = UInt8(x and y).cast[D]()
        elif op == OR:
            var x = av and Bool(a)
            var y = bv and Bool(b)
            value = UInt8(x or y).cast[D]()
            ok = (av and bv) or x or y
        elif op == XOR:
            ok = av and bv
            if ok:
                value = UInt8(Bool(a) != Bool(b)).cast[D]()
        elif op == NEG:
            ok = av
            if ok:
                value = -a
        elif op == ADD or op == SUB or op == MUL:
            ok = av and bv
            if ok:
                value = arithmetic[D, 1](op, a, b)
        else:
            ok = av and bv
            if ok:
                value = compare[D, 1](op, a, b).cast[D]()
        # Every node is initialized, including null payloads and empty branches.
        local[unsafe_offset=n] = value if ok else Scalar[D](0)
        present[unsafe_offset=n] = UInt8(ok)
    return (
        local[unsafe_offset=count - 1],
        present[unsafe_offset=count - 1] != 0,
    )


def initialize_column[
    D: DType
](
    values: DeviceBuffer[D].device_type,
    valid: DeviceBuffer[DType.uint8].device_type,
    bits: DeviceBuffer[DType.uint8].device_type,
    bool_values: DeviceBuffer[DType.uint8].device_type,
    capacity: Int64,
    rows: Int64,
    slot: Int64,
    bit_offset: Int64,
    has_bits: Int64,
    is_bool: Int64,
    blocks: Int64,
):
    var i = Int(block_idx.x) * THREADS + Int(thread_idx.x)
    while i < Int(rows):
        var bit = Int(bit_offset) + i
        var ok = True
        if has_bits:
            ok = (
                bits[unsafe_offset=bit // 8] & (UInt8(1) << UInt8(bit % 8))
            ) != 0
        var at = Int(slot) * Int(capacity) + i
        valid[unsafe_offset=at] = UInt8(ok)
        if is_bool:
            var value = False
            if ok:
                value = (
                    bool_values[unsafe_offset=bit // 8]
                    & (UInt8(1) << UInt8(bit % 8))
                ) != 0
            values[unsafe_offset=at] = UInt8(value).cast[D]()
        i += Int(blocks) * THREADS


def project[
    D: DType
](
    values: DeviceBuffer[D].device_type,
    valid: DeviceBuffer[DType.uint8].device_type,
    code: DeviceBuffer[DType.int64].device_type,
    literals: DeviceBuffer[DType.float64].device_type,
    rows: DeviceBuffer[DType.int64].device_type,
    capacity: Int64,
    start: Int64,
    count: Int64,
    slot: Int64,
    blocks: Int64,
):
    var i = Int(block_idx.x) * THREADS + Int(thread_idx.x)
    while i < Int(rows[unsafe_offset=0]):
        var result = evaluate_row[D](
            values,
            valid,
            code,
            literals,
            Int(capacity),
            i,
            Int(start),
            Int(count),
        )
        var at = Int(slot) * Int(capacity) + i
        values[unsafe_offset=at] = result[0]
        valid[unsafe_offset=at] = UInt8(result[1])
        i += Int(blocks) * THREADS


def stable_ranks[
    D: DType
](
    values: DeviceBuffer[D].device_type,
    valid: DeviceBuffer[DType.uint8].device_type,
    rows: DeviceBuffer[DType.int64].device_type,
    ranks: DeviceBuffer[DType.int64].device_type,
    counts: DeviceBuffer[DType.int64].device_type,
    capacity: Int64,
    predicate: Int64,
    chunk: Int64,
):
    # Blocks own contiguous input chunks; each scan preserves within-chunk order.
    var scan = stack_allocation[
        THREADS, DType.int64, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var first = Int(block_idx.x) * Int(chunk)
    var running = Int64(0)
    for offset in range(0, Int(chunk), THREADS):
        var i = first + offset + tid
        var selected = False
        if i < Int(rows[unsafe_offset=0]):
            var at = Int(predicate) * Int(capacity) + i
            selected = valid[unsafe_offset=at] != 0 and Bool(
                values[unsafe_offset=at]
            )
        scan[unsafe_offset=tid] = Int64(selected)
        barrier()
        comptime for step in range(8):
            comptime stride = 1 << step
            var add = Int64(0)
            if tid >= stride:
                add = scan[unsafe_offset=tid - stride]
            barrier()
            scan[unsafe_offset=tid] += add
            barrier()
        if i < Int(capacity):
            ranks[unsafe_offset=i] = (
                running + scan[unsafe_offset=tid] - Int64(selected)
            )
        running += scan[unsafe_offset=THREADS - 1]
        barrier()
    if tid == 0:
        counts[unsafe_offset=Int(block_idx.x)] = running


def block_offsets(
    rows: DeviceBuffer[DType.int64].device_type,
    counts: DeviceBuffer[DType.int64].device_type,
    blocks: Int64,
):
    # At most 1024 entries; this bounded serial pass is deterministic.
    var total = Int64(0)
    for i in range(Int(blocks)):
        var count = counts[unsafe_offset=i]
        counts[unsafe_offset=i] = total
        total += count
    rows[unsafe_offset=1] = rows[unsafe_offset=0]
    rows[unsafe_offset=0] = total


def stable_gather[
    D: DType
](
    values: DeviceBuffer[D].device_type,
    valid: DeviceBuffer[DType.uint8].device_type,
    output: DeviceBuffer[D].device_type,
    output_valid: DeviceBuffer[DType.uint8].device_type,
    rows: DeviceBuffer[DType.int64].device_type,
    ranks: DeviceBuffer[DType.int64].device_type,
    counts: DeviceBuffer[DType.int64].device_type,
    columns: DeviceBuffer[DType.int64].device_type,
    capacity: Int64,
    predicate: Int64,
    chunk: Int64,
    column_start: Int64,
    column_count: Int64,
    blocks: Int64,
):
    var i = Int(block_idx.x) * THREADS + Int(thread_idx.x)
    while i < Int(rows[unsafe_offset=1]):
        var pred = Int(predicate) * Int(capacity) + i
        if valid[unsafe_offset=pred] != 0 and Bool(values[unsafe_offset=pred]):
            var destination = Int(
                counts[unsafe_offset=i // Int(chunk)] + ranks[unsafe_offset=i]
            )
            for k in range(Int(column_count)):
                var column = Int(columns[unsafe_offset=Int(column_start) + k])
                var base = column * Int(capacity)
                output[unsafe_offset=base + destination] = values[
                    unsafe_offset=base + i
                ]
                output_valid[unsafe_offset=base + destination] = valid[
                    unsafe_offset=base + i
                ]
        i += Int(blocks) * THREADS


def pack_output[
    D: DType
](
    values: DeviceBuffer[D].device_type,
    valid: DeviceBuffer[DType.uint8].device_type,
    bits: DeviceBuffer[DType.uint8].device_type,
    bool_values: DeviceBuffer[DType.uint8].device_type,
    capacity: Int64,
    rows: Int64,
    slot: Int64,
    is_bool: Int64,
    blocks: Int64,
):
    # One thread owns each byte of both packed outputs: no atomic OR/races.
    var byte = Int(block_idx.x) * THREADS + Int(thread_idx.x)
    while byte < (Int(rows) + 7) // 8:
        var validity = UInt8(0)
        var packed = UInt8(0)
        for bit in range(8):
            var row = byte * 8 + bit
            if row < Int(rows):
                var at = Int(slot) * Int(capacity) + row
                if valid[unsafe_offset=at] != 0:
                    validity |= UInt8(1) << UInt8(bit)
                    if is_bool and Bool(values[unsafe_offset=at]):
                        packed |= UInt8(1) << UInt8(bit)
        bits[unsafe_offset=byte] = validity
        bool_values[unsafe_offset=byte] = packed
        byte += Int(blocks) * THREADS


def partial_reduce[
    D: DType
](
    values: DeviceBuffer[D].device_type,
    valid: DeviceBuffer[DType.uint8].device_type,
    rows: DeviceBuffer[DType.int64].device_type,
    sums: DeviceBuffer[DType.float64].device_type,
    counts: DeviceBuffer[DType.int64].device_type,
    capacity: Int64,
    slot: Int64,
    blocks: Int64,
):
    var total = Float64(0)
    var count = Int64(0)
    var i = Int(block_idx.x) * THREADS + Int(thread_idx.x)
    while i < Int(rows[unsafe_offset=0]):
        var at = Int(slot) * Int(capacity) + i
        if valid[unsafe_offset=at] != 0:
            total += values[unsafe_offset=at].cast[DType.float64]()
            count += 1
        i += Int(blocks) * THREADS
    var reduced = reduce_pair(total, count)
    if thread_idx.x == 0:
        sums[unsafe_offset=Int(block_idx.x)] = reduced[0]
        counts[unsafe_offset=Int(block_idx.x)] = reduced[1]
