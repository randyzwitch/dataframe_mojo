"""Exact, row-bounded discovery of observed pairs of resident text codes.

The host may use the compact output pair list to construct dictionary labels.
This module discovers distinct pairs and maps rows to those pairs entirely on
the device. Table capacity depends on row count, never on the product of the
input dictionary sizes.
"""
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.atomic import Atomic
from std.memory import Pointer
from dataframe.partition import _mix, _combine
from .query_scope import checked_add, checked_mul


comptime PAIR_NULL = UInt32.MAX


def table_capacity(rows: Int) raises -> Int:
    """Power-of-two hash capacity at least twice the input row count."""
    if rows < 0 or rows > Int(UInt32.MAX):
        raise Error("Observed text pair row count exceeds UInt32 capacity")
    if rows > (1 << 30):
        raise Error("Observed text pair table exceeds UInt32 capacity")
    var wanted = max(1, rows * 2)
    var result = 1
    while result < wanted:
        result *= 2
    return result


def estimate_bytes(rows: Int) raises -> Int:
    """Complete device scratch/output footprint, excluding resident inputs."""
    var capacity = table_capacity(rows)
    var row_capacity = max(1, rows)
    # representatives + slot IDs, two compact pair controls, row IDs/errors,
    # and the count/status control. The pair controls retain row capacity
    # because exact output count is data-dependent until discovery completes.
    var total = checked_mul(capacity, 8)
    total = checked_add(total, checked_mul(row_capacity, 16))
    return checked_add(total, 8)


struct ObservedTextPairs(Movable):
    """Own asynchronous discovery scratch through its final device use."""

    var capacity: Int
    var table: DeviceBuffer[DType.uint32]
    var slot_ids: DeviceBuffer[DType.uint32]
    var pair_left: DeviceBuffer[DType.uint32]
    var pair_right: DeviceBuffer[DType.uint32]
    var row_ids: DeviceBuffer[DType.uint32]
    var row_errors: DeviceBuffer[DType.uint32]
    var metadata: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        capacity: Int,
        var table: DeviceBuffer[DType.uint32],
        var slot_ids: DeviceBuffer[DType.uint32],
        var pair_left: DeviceBuffer[DType.uint32],
        var pair_right: DeviceBuffer[DType.uint32],
        var row_ids: DeviceBuffer[DType.uint32],
        var row_errors: DeviceBuffer[DType.uint32],
        var metadata: DeviceBuffer[DType.uint32],
    ):
        self.capacity = capacity
        self.table = table^
        self.slot_ids = slot_ids^
        self.pair_left = pair_left^
        self.pair_right = pair_right^
        self.row_ids = row_ids^
        self.row_errors = row_errors^
        self.metadata = metadata^


def initialize(
    representatives: Pointer[UInt32, MutAnyOrigin],
    slot_ids: Pointer[UInt32, MutAnyOrigin],
    metadata: Pointer[UInt32, MutAnyOrigin],
    capacity: UInt32,
):
    var at = global_idx.x
    if at < Int(capacity):
        representatives[unsafe_offset=at] = PAIR_NULL
        slot_ids[unsafe_offset=at] = PAIR_NULL
    if at == 0:
        metadata[unsafe_offset=0] = 0  # unique pair count
        metadata[unsafe_offset=1] = 0  # protocol/capacity error


@always_inline
def _hash_pair(left: UInt32, right: UInt32) -> UInt32:
    var hash = _combine(UInt64(0), UInt64(left))
    hash = _combine(hash, UInt64(right))
    return UInt32(_mix(hash))


def pack_codes(
    words: Pointer[UInt64, ImmutAnyOrigin],
    packed: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    slot: UInt32,
    stride: UInt32,
):
    """Extract one resident text-code slot from the UInt64 low-word plane.

    UInt64 cells are not bitcast to a contiguous UInt32 array: that would
    interleave low/high words across columns. The reserved UInt32 sentinel
    reports a malformed code with nonzero high bits to `discover`.
    """
    var row = global_idx.x
    if row >= Int(rows):
        return
    var source = Int(slot) * Int(stride) + row
    var value = words[unsafe_offset=source]
    if value >> 32 != 0:
        packed[unsafe_offset=row] = UInt32.MAX
    else:
        packed[unsafe_offset=row] = UInt32(value)


def discover(
    left_codes: Pointer[UInt32, ImmutAnyOrigin],
    left_valid: Pointer[UInt8, ImmutAnyOrigin],
    left_errors: Pointer[UInt32, ImmutAnyOrigin],
    right_codes: Pointer[UInt32, ImmutAnyOrigin],
    right_valid: Pointer[UInt8, ImmutAnyOrigin],
    right_errors: Pointer[UInt32, ImmutAnyOrigin],
    representatives: Pointer[UInt32, MutAnyOrigin],
    metadata: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    capacity: UInt32,
):
    """Insert one immutable representative row per valid, error-free pair.

    Weak CAS failure on an empty bucket retries that same bucket. A real
    collision advances linearly. The table is at least twice the row count,
    so exhausting the full probe sequence indicates a protocol error.
    """
    var row = global_idx.x
    if row >= Int(rows):
        return
    if left_errors[unsafe_offset=row] or right_errors[unsafe_offset=row]:
        return
    if not left_valid[unsafe_offset=row] or not right_valid[unsafe_offset=row]:
        return
    var left = left_codes[unsafe_offset=row]
    var right = right_codes[unsafe_offset=row]
    if left == PAIR_NULL or right == PAIR_NULL:
        Atomic[UInt32].max(metadata.unsafe_offset(1), 1)
        return
    var bucket = _hash_pair(left, right) & (capacity - 1)
    for probe in range(Int(capacity)):
        var expected = PAIR_NULL
        var inserted = Atomic[UInt32].compare_exchange[weak=True](
            representatives.unsafe_offset(Int(bucket)), expected, UInt32(row)
        )
        if not inserted and expected == PAIR_NULL:
            # Metal weak CAS may fail spuriously. Do not split equal keys by
            # advancing to another bucket on this result.
            continue
        if inserted:
            return
        var other = Int(expected)
        if (
            left_codes[unsafe_offset=other] == left
            and right_codes[unsafe_offset=other] == right
            and not left_errors[unsafe_offset=other]
            and not right_errors[unsafe_offset=other]
            and left_valid[unsafe_offset=other]
            and right_valid[unsafe_offset=other]
        ):
            return
        bucket = (bucket + 1) & (capacity - 1)
    Atomic[UInt32].max(metadata.unsafe_offset(1), 1)


def compact(
    representatives: Pointer[UInt32, ImmutAnyOrigin],
    slot_ids: Pointer[UInt32, MutAnyOrigin],
    left_codes: Pointer[UInt32, ImmutAnyOrigin],
    right_codes: Pointer[UInt32, ImmutAnyOrigin],
    pair_left: Pointer[UInt32, MutAnyOrigin],
    pair_right: Pointer[UInt32, MutAnyOrigin],
    metadata: Pointer[UInt32, MutAnyOrigin],
    capacity: UInt32,
):
    """Assign dense pair IDs after discovery has completed."""
    if metadata[unsafe_offset=1]:
        return
    var bucket = global_idx.x
    if bucket >= Int(capacity):
        return
    var representative = representatives[unsafe_offset=bucket]
    if representative == PAIR_NULL:
        return
    var pair_id = Atomic[UInt32].fetch_add(metadata.unsafe_offset(0), 1)
    slot_ids[unsafe_offset=bucket] = pair_id
    pair_left[unsafe_offset=Int(pair_id)] = left_codes[
        unsafe_offset=Int(representative)
    ]
    pair_right[unsafe_offset=Int(pair_id)] = right_codes[
        unsafe_offset=Int(representative)
    ]


def map_rows(
    left_codes: Pointer[UInt32, ImmutAnyOrigin],
    left_valid: Pointer[UInt8, ImmutAnyOrigin],
    left_errors: Pointer[UInt32, ImmutAnyOrigin],
    right_codes: Pointer[UInt32, ImmutAnyOrigin],
    right_valid: Pointer[UInt8, ImmutAnyOrigin],
    right_errors: Pointer[UInt32, ImmutAnyOrigin],
    representatives: Pointer[UInt32, ImmutAnyOrigin],
    slot_ids: Pointer[UInt32, ImmutAnyOrigin],
    pair_ids: Pointer[UInt32, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    metadata: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    capacity: UInt32,
):
    """Map valid pairs to dense IDs; preserve input errors and nulls."""
    var row = global_idx.x
    if row >= Int(rows):
        return
    pair_ids[unsafe_offset=row] = PAIR_NULL
    output_errors[unsafe_offset=row] = (
        left_errors[unsafe_offset=row] | right_errors[unsafe_offset=row]
    )
    if metadata[unsafe_offset=1] or output_errors[unsafe_offset=row]:
        return
    if not left_valid[unsafe_offset=row] or not right_valid[unsafe_offset=row]:
        return
    var left = left_codes[unsafe_offset=row]
    var right = right_codes[unsafe_offset=row]
    if left == PAIR_NULL or right == PAIR_NULL:
        Atomic[UInt32].max(metadata.unsafe_offset(1), 1)
        return
    var bucket = _hash_pair(left, right) & (capacity - 1)
    for probe in range(Int(capacity)):
        var representative = representatives[unsafe_offset=Int(bucket)]
        if representative == PAIR_NULL:
            Atomic[UInt32].max(metadata.unsafe_offset(1), 1)
            return
        var other = Int(representative)
        if (
            left_codes[unsafe_offset=other] == left
            and right_codes[unsafe_offset=other] == right
        ):
            pair_ids[unsafe_offset=row] = slot_ids[unsafe_offset=Int(bucket)]
            return
        bucket = (bucket + 1) & (capacity - 1)
    Atomic[UInt32].max(metadata.unsafe_offset(1), 1)


def write_output(
    row_ids: Pointer[UInt32, ImmutAnyOrigin],
    pair_codes: Pointer[UInt32, ImmutAnyOrigin],
    row_errors: Pointer[UInt32, ImmutAnyOrigin],
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    output_base: Int64,
    output_cells: Int64,
):
    """Publish compact text codes into one resident output column slot."""
    var row = global_idx.x
    if row >= Int(rows):
        return
    var at = Int(output_base) + row
    var pair_id = row_ids[unsafe_offset=row]
    var error = row_errors[unsafe_offset=row]
    errors[unsafe_offset=at] = error
    if pair_id != PAIR_NULL and not error:
        words[unsafe_offset=at] = UInt64(pair_codes[unsafe_offset=Int(pair_id)])
        validity[unsafe_offset=at] = 1
    else:
        words[unsafe_offset=at] = 0
        validity[unsafe_offset=at] = 0
    words[unsafe_offset=Int(output_cells) + at] = 0


def run(
    ctx: DeviceContext,
    left_codes: DeviceBuffer[DType.uint32],
    left_valid: DeviceBuffer[DType.uint8],
    left_errors: DeviceBuffer[DType.uint32],
    right_codes: DeviceBuffer[DType.uint32],
    right_valid: DeviceBuffer[DType.uint8],
    right_errors: DeviceBuffer[DType.uint32],
    rows: Int,
) raises -> ObservedTextPairs:
    """Enqueue discovery, compact controls, and row remapping.

    The caller owns query-scope admission for all returned buffers. Inputs are
    row-contiguous arrays with one validity byte and one error word per row.
    The returned owner retains every scratch buffer referenced by queued
    kernels. Pair controls have row capacity; only metadata[0] entries are
    initialized outputs and may be transferred to construct labels.
    """
    var capacity = table_capacity(rows)
    if (
        rows < 0
        or len(left_codes) < rows
        or len(left_valid) < rows
        or len(left_errors) < rows
        or len(right_codes) < rows
        or len(right_valid) < rows
        or len(right_errors) < rows
    ):
        raise Error("Observed text pair inputs are shorter than row count")
    var table = ctx.enqueue_create_buffer[DType.uint32](capacity)
    var slot_ids = ctx.enqueue_create_buffer[DType.uint32](capacity)
    var metadata = ctx.enqueue_create_buffer[DType.uint32](2)
    var pair_left = ctx.enqueue_create_buffer[DType.uint32](max(1, rows))
    var pair_right = ctx.enqueue_create_buffer[DType.uint32](max(1, rows))
    var row_ids = ctx.enqueue_create_buffer[DType.uint32](max(1, rows))
    var row_errors = ctx.enqueue_create_buffer[DType.uint32](max(1, rows))
    try:
        ctx.enqueue_function[initialize](
            table.unsafe_ptr(),
            slot_ids.unsafe_ptr(),
            metadata.unsafe_ptr(),
            UInt32(capacity),
            grid_dim=(capacity + 255) // 256,
            block_dim=256,
        )
        if rows:
            ctx.enqueue_function[discover](
                left_codes.unsafe_ptr(),
                left_valid.unsafe_ptr(),
                left_errors.unsafe_ptr(),
                right_codes.unsafe_ptr(),
                right_valid.unsafe_ptr(),
                right_errors.unsafe_ptr(),
                table.unsafe_ptr(),
                metadata.unsafe_ptr(),
                UInt32(rows),
                UInt32(capacity),
                grid_dim=(rows + 255) // 256,
                block_dim=256,
            )
        ctx.enqueue_function[compact](
            table.unsafe_ptr(),
            slot_ids.unsafe_ptr(),
            left_codes.unsafe_ptr(),
            right_codes.unsafe_ptr(),
            pair_left.unsafe_ptr(),
            pair_right.unsafe_ptr(),
            metadata.unsafe_ptr(),
            UInt32(capacity),
            grid_dim=(capacity + 255) // 256,
            block_dim=256,
        )
        if rows:
            ctx.enqueue_function[map_rows](
                left_codes.unsafe_ptr(),
                left_valid.unsafe_ptr(),
                left_errors.unsafe_ptr(),
                right_codes.unsafe_ptr(),
                right_valid.unsafe_ptr(),
                right_errors.unsafe_ptr(),
                table.unsafe_ptr(),
                slot_ids.unsafe_ptr(),
                row_ids.unsafe_ptr(),
                row_errors.unsafe_ptr(),
                metadata.unsafe_ptr(),
                UInt32(rows),
                UInt32(capacity),
                grid_dim=(rows + 255) // 256,
                block_dim=256,
            )
    except error:
        # Enqueued commands may still reference locals if a later launch fails.
        ctx.synchronize()
        raise error^
    return ObservedTextPairs(
        capacity,
        table^,
        slot_ids^,
        pair_left^,
        pair_right^,
        row_ids^,
        row_errors^,
        metadata^,
    )
