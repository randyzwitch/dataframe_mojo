"""One-key stable hash joins over typed resident columns."""

from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.host import DeviceBuffer
from max.gpu.primitives.block import prefix_sum
from std.atomic import Atomic
from std.memory import Pointer, bitcast
from std.sys import size_of
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, Series
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.partition import _mix, _combine
from dataframe.accelerator.joins import DagTask, DAG_JOIN
from dataframe.execution_report import ExecutionReport
from .grouping import key_equal
from . import hash_grouping
from .runtime import _kind, _checked_add, _checked_mul, _width
from .typed_storage import (
    NativeColumnAny,
    NativeColumn,
    _NativeReservation,
    upload_native,
)
from .residency import (
    NativeJoinField,
    NativeRowMap,
    Scope,
    Table,
)
from . import joins as j


@fieldwise_init
struct Attempt(Movable):
    var used: Bool
    var table: Optional[Table]
    var report: Optional[DataFrame]


@fieldwise_init
struct PackedKey(Movable):
    var words: DeviceBuffer[DType.uint64]
    var valid: DeviceBuffer[DType.uint8]
    var source_validity: DeviceBuffer[DType.uint8]


@always_inline
def _hash_key(word: UInt64, valid: Bool, dtype: Int64) -> UInt64:
    var payload = UInt64(0)
    if valid:
        payload = hash_grouping._canonical(word, dtype)
    return _mix(_combine(UInt64(valid), payload))


@always_inline
def _keys_equal(
    a: UInt64, a_valid: Bool, b: UInt64, b_valid: Bool, dtype: Int64
) -> Bool:
    if a_valid != b_valid:
        return False
    return not a_valid or key_equal(a, b, dtype)


def pack_key[
    dtype: DType
](
    values: Pointer[Scalar[dtype], ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    words: Pointer[UInt64, MutAnyOrigin],
    valid: Pointer[UInt8, MutAnyOrigin],
    value_offset: UInt32,
    validity_offset: UInt32,
    rows: UInt32,
    has_validity: UInt8,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var present = has_validity == 0 or (
        (
            validity[unsafe_offset=(Int(validity_offset) + row) // 8]
            & (UInt8(1) << UInt8((Int(validity_offset) + row) % 8))
        )
        != 0
    )
    var value = values[unsafe_offset=Int(value_offset) + row]
    var bits = UInt32(0)
    comptime if dtype == DType.float32 or dtype == DType.int32:
        bits = bitcast[DType.uint32](value)
    else:
        bits = UInt32(value)
    words[unsafe_offset=row] = UInt64(bits)
    valid[unsafe_offset=row] = UInt8(present)


def initialize_hash(
    representatives: Pointer[UInt32, MutAnyOrigin],
    counts: Pointer[UInt32, MutAnyOrigin],
    status: Pointer[UInt32, MutAnyOrigin],
    capacity: UInt32,
):
    var at = global_idx.x
    if at < Int(capacity):
        representatives[unsafe_offset=at] = UInt32.MAX
        counts[unsafe_offset=at] = 0
    if at < 1:
        status[unsafe_offset=0] = 0


def build_hash(
    words: Pointer[UInt64, ImmutAnyOrigin],
    valid: Pointer[UInt8, ImmutAnyOrigin],
    representatives: Pointer[UInt32, MutAnyOrigin],
    counts: Pointer[UInt32, MutAnyOrigin],
    row_slots: Pointer[UInt32, MutAnyOrigin],
    status: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    capacity: UInt32,
    dtype: Int64,
    nulls_equal: UInt8,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var present = valid[unsafe_offset=row] != 0
    if not present and nulls_equal == 0:
        row_slots[unsafe_offset=row] = UInt32.MAX
        return
    var key = words[unsafe_offset=row]
    var bucket = UInt32(_hash_key(key, present, dtype)) & (capacity - 1)
    for _ in range(128):
        var expected = UInt32.MAX
        var inserted = Atomic[UInt32].compare_exchange[weak=True](
            representatives.unsafe_offset(Int(bucket)), expected, UInt32(row)
        )
        if not inserted and expected == UInt32.MAX:
            continue
        if inserted or _keys_equal(
            key,
            present,
            words[unsafe_offset=Int(expected)],
            valid[unsafe_offset=Int(expected)] != 0,
            dtype,
        ):
            row_slots[unsafe_offset=row] = bucket
            _ = Atomic[UInt32].fetch_add(counts.unsafe_offset(Int(bucket)), 1)
            return
        bucket = (bucket + 1) & (capacity - 1)
    row_slots[unsafe_offset=row] = UInt32.MAX
    Atomic[UInt32].max(status.unsafe_offset(0), UInt32(1))


def initialize_members(
    row_slots: Pointer[UInt32, ImmutAnyOrigin],
    members: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
):
    var row = global_idx.x
    if row < Int(rows):
        members[unsafe_offset=row] = UInt32(row)


@always_inline
def _member_less(
    a: UInt32, b: UInt32, row_slots: Pointer[UInt32, ImmutAnyOrigin]
) -> Bool:
    var a_slot = row_slots[unsafe_offset=Int(a)]
    var b_slot = row_slots[unsafe_offset=Int(b)]
    return a_slot < b_slot or (a_slot == b_slot and a < b)


def merge_members(
    source: Pointer[UInt32, ImmutAnyOrigin],
    target: Pointer[UInt32, MutAnyOrigin],
    row_slots: Pointer[UInt32, ImmutAnyOrigin],
    rows: UInt32,
    width: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var base = row // (2 * Int(width)) * (2 * Int(width))
    var middle = min(base + Int(width), Int(rows))
    var end = min(base + 2 * Int(width), Int(rows))
    var own = source[unsafe_offset=row]
    if row < middle:
        var lo = middle
        var hi = end
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            if _member_less(source[unsafe_offset=mid], own, row_slots):
                lo = mid + 1
            else:
                hi = mid
        target[unsafe_offset=base + row - base + lo - middle] = own
    else:
        var lo = base
        var hi = middle
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            if not _member_less(own, source[unsafe_offset=mid], row_slots):
                lo = mid + 1
            else:
                hi = mid
        target[unsafe_offset=base + lo - base + row - middle] = own


def scan_slot_counts(
    counts: Pointer[UInt32, ImmutAnyOrigin],
    offsets: Pointer[UInt64, MutAnyOrigin],
    blocks: Pointer[UInt64, MutAnyOrigin],
    slots: UInt32,
):
    var row = global_idx.x
    var value = UInt64(0)
    if row < Int(slots):
        value = UInt64(counts[unsafe_offset=row])
    var total = prefix_sum[block_size=256](value)
    if row < Int(slots):
        offsets[unsafe_offset=row] = total - value
    if thread_idx.x == 255:
        blocks[unsafe_offset=block_idx.x] = total


def finish_slot_blocks(blocks: Pointer[UInt64, MutAnyOrigin], count: UInt32):
    var total = UInt64(0)
    for block in range(Int(count)):
        var value = blocks[unsafe_offset=block]
        blocks[unsafe_offset=block] = total
        total += value


def apply_slot_blocks(
    offsets: Pointer[UInt64, MutAnyOrigin],
    blocks: Pointer[UInt64, ImmutAnyOrigin],
    slots: UInt32,
):
    var row = global_idx.x
    if row < Int(slots):
        offsets[unsafe_offset=row] += blocks[unsafe_offset=row // 256]


def probe_hash(
    build_words: Pointer[UInt64, ImmutAnyOrigin],
    build_valid: Pointer[UInt8, ImmutAnyOrigin],
    probe_words: Pointer[UInt64, ImmutAnyOrigin],
    probe_valid: Pointer[UInt8, ImmutAnyOrigin],
    representatives: Pointer[UInt32, ImmutAnyOrigin],
    bucket_counts: Pointer[UInt32, ImmutAnyOrigin],
    bucket_offsets: Pointer[UInt64, ImmutAnyOrigin],
    members: Pointer[UInt32, ImmutAnyOrigin],
    probe_ranges: Pointer[UInt32, MutAnyOrigin],
    probe_matches: Pointer[UInt32, MutAnyOrigin],
    output_counts: Pointer[UInt64, MutAnyOrigin],
    matched_build: Pointer[UInt32, MutAnyOrigin],
    status: Pointer[UInt32, MutAnyOrigin],
    probe_rows: UInt32,
    build_rows: UInt32,
    capacity: UInt32,
    dtype: Int64,
    nulls_equal: UInt8,
    how: UInt32,
):
    var row = global_idx.x
    if row >= Int(probe_rows):
        return
    var present = probe_valid[unsafe_offset=row] != 0
    var found = UInt32.MAX
    var stopped_at_empty = False
    if present or nulls_equal != 0 or how == 7:
        var key = probe_words[unsafe_offset=row]
        var bucket = UInt32(_hash_key(key, present, dtype)) & (capacity - 1)
        for _ in range(128):
            var representative = representatives[unsafe_offset=Int(bucket)]
            if representative == UInt32.MAX:
                stopped_at_empty = True
                break
            if _keys_equal(
                key,
                present,
                build_words[unsafe_offset=Int(representative)],
                build_valid[unsafe_offset=Int(representative)] != 0,
                dtype,
            ):
                found = bucket
                stopped_at_empty = True
                break
            bucket = (bucket + 1) & (capacity - 1)
        if found == UInt32.MAX and not stopped_at_empty and build_rows > 0:
            Atomic[UInt32].max(status.unsafe_offset(0), UInt32(1))
    var matches = bucket_counts[
        unsafe_offset=Int(found)
    ] if found != UInt32.MAX else UInt32(0)
    probe_ranges[unsafe_offset=row] = found
    probe_matches[unsafe_offset=row] = matches
    var count = UInt64(matches)
    if how == 4:
        count = UInt64(matches != 0)
    elif how == 5:
        count = UInt64(matches == 0)
    elif how == 1 or how == 2 or how == 3 or how == 7:
        count = max(count, UInt64(1))
    output_counts[unsafe_offset=row] = count
    if how == 3 and found != UInt32.MAX:
        var begin = bucket_offsets[unsafe_offset=Int(found)]
        for at in range(Int(matches)):
            var build_row = members[unsafe_offset=Int(begin) + at]
            _ = Atomic[UInt32].max(
                matched_build.unsafe_offset(Int(build_row)), UInt32(1)
            )


def unmatched_flags(
    matched: Pointer[UInt32, ImmutAnyOrigin],
    flags: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
):
    var row = global_idx.x
    if row < Int(rows):
        flags[unsafe_offset=row] = UInt32(matched[unsafe_offset=row] == 0)


def emit_pair_maps(
    probe_ranges: Pointer[UInt32, ImmutAnyOrigin],
    probe_matches: Pointer[UInt32, ImmutAnyOrigin],
    probe_prefix: Pointer[UInt64, ImmutAnyOrigin],
    probe_blocks: Pointer[UInt64, ImmutAnyOrigin],
    members: Pointer[UInt32, ImmutAnyOrigin],
    bucket_offsets: Pointer[UInt64, ImmutAnyOrigin],
    unmatched: Pointer[UInt32, ImmutAnyOrigin],
    metadata: Pointer[UInt64, ImmutAnyOrigin],
    left_map: Pointer[UInt32, MutAnyOrigin],
    right_map: Pointer[UInt32, MutAnyOrigin],
    probe_rows: UInt32,
    probe_side: UInt32,
    how: UInt32,
):
    var row = global_idx.x
    if UInt64(row) >= metadata[unsafe_offset=0]:
        return
    var probe_total = metadata[unsafe_offset=1]
    var left = UInt32.MAX
    var right = UInt32.MAX
    if UInt64(row) >= probe_total:
        left = UInt32.MAX
        right = unmatched[unsafe_offset=Int(UInt64(row) - probe_total)]
    else:
        var lo = UInt32(0)
        var hi = probe_rows
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            var start = (
                probe_prefix[unsafe_offset=Int(mid)]
                + probe_blocks[unsafe_offset=Int(mid) // 256]
            )
            if start <= UInt64(row):
                lo = mid + 1
            else:
                hi = mid
        var probe = lo - 1
        var start = (
            probe_prefix[unsafe_offset=Int(probe)]
            + probe_blocks[unsafe_offset=Int(probe) // 256]
        )
        var within = UInt32(UInt64(row) - start)
        var bucket = probe_ranges[unsafe_offset=Int(probe)]
        var build = UInt32.MAX
        if (
            how != 4
            and how != 5
            and probe_matches[unsafe_offset=Int(probe)] > 0
        ):
            build = members[
                unsafe_offset=Int(bucket_offsets[unsafe_offset=Int(bucket)])
                + Int(within)
            ]
        if how == 4 or how == 5:
            build = UInt32.MAX
        if probe_side != 0:
            left = build
            right = probe
        else:
            left = probe
            right = build
    left_map[unsafe_offset=row] = left
    right_map[unsafe_offset=row] = right


def _supports_frame(frame: DataFrame) -> Bool:
    if frame.width() == 0:
        return False
    for column in frame._columns:
        var dtype = column.dtype()
        if (
            dtype == DataType.BOOL
            or dtype.is_decimal()
            or dtype.is_categorical()
            or dtype.is_nested()
            or not (dtype.is_numeric() or dtype.is_temporal())
        ):
            return False
        var supported = False
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            supported |= dtype.physical() == DataType.of(D)
        if not supported:
            return False
    return True


def _upload_column(scope: Scope, series: Series) raises -> NativeColumnAny:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series.dtype().physical() == DataType.of(D):
            return NativeColumnAny(upload_native[D](scope, series))
    raise Error("Mojo GPU native hash join upload dtype is unsupported")


def _upload_table(scope: Scope, frame: DataFrame) raises -> Table:
    var columns = List[NativeColumnAny](capacity=frame.width())
    for series in frame._columns:
        columns.append(_upload_column(scope, series))
    return Table.native(frame.clear(), columns^)


def _table_supported(table: Table, frame: DataFrame, scope: Scope) -> Bool:
    if (
        not len(table.native_columns)
        or not table.native_columns[0].scope().same_query(scope)
        or table.schema.width() != frame.width()
        or len(table.columns) != frame.width()
        or table.offset < 0
        or table.rows < 0
    ):
        return False
    for i in range(frame.width()):
        if (
            table.schema._columns[i].name() != frame._columns[i].name()
            or table.schema._columns[i].dtype() != frame._columns[i].dtype()
            or table.text[i]
            or table.categories[i]
            or table.grouped_keys[i]
        ):
            return False
    return _supports_frame(frame)


def _key_column(frame: DataFrame, index: Int) -> Bool:
    if index < 0 or index >= frame.width():
        return False
    var dtype = frame._columns[index].dtype()
    return (
        dtype == DataType.INT32
        or dtype == DataType.UINT32
        or dtype == DataType.FLOAT32
    )


def _pack_typed_key[
    dtype: DType
](scope: Scope, column: NativeColumn[dtype], rows: Int,) raises -> PackedKey:
    var ctx = scope.context()
    var words = ctx.enqueue_create_buffer[DType.uint64](max(1, rows))
    var valid = ctx.enqueue_create_buffer[DType.uint8](max(1, rows))
    if rows == 0:
        var bitmap_buffer = ctx.enqueue_create_buffer[DType.uint8](1)
        return PackedKey(words^, valid^, bitmap_buffer^)
    var values = column.values()
    var bitmap = column.validity()
    var bitmap_buffer = (
        bitmap.value().copy() if bitmap else ctx.enqueue_create_buffer[
            DType.uint8
        ](1)
    )
    if rows:
        try:
            ctx.enqueue_function[pack_key[dtype]](
                values.unsafe_ptr(),
                bitmap_buffer.unsafe_ptr(),
                words.unsafe_ptr(),
                valid.unsafe_ptr(),
                UInt32(column.value_offset),
                UInt32(column.validity_bit_offset),
                UInt32(rows),
                UInt8(column.has_validity()),
                grid_dim=(rows + 255) // 256,
                block_dim=256,
            )
        except error:
            scope.synchronize()
            raise error^
    return PackedKey(words^, valid^, bitmap_buffer^)


def _raw_upload_bytes(frame: DataFrame) raises -> Int:
    var bytes = 0
    for series in frame._columns:
        var width = 0
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if series.dtype().physical() == DataType.of(D):
                width = size_of[Scalar[D]]()
        if width == 0:
            return -1
        bytes = _checked_add(bytes, _checked_mul(len(series), width))
        if series.null_count() > 0:
            # A sliced one-chunk bitmap may begin at any bit. Two packed
            # bitmaps plus one boundary byte safely cover device and staging
            # storage for both one- and multi-chunk uploads.
            var bitmap_bound = _checked_add(
                _checked_add(len(series), 7) // 8, 1
            )
            bytes = _checked_add(bytes, _checked_mul(bitmap_bound, 2))
    return bytes


def _scratch_bytes(
    probe: Int, build: Int, capacity: Int, full: Bool
) raises -> Int:
    var pb = max(1, (probe + 255) // 256)
    var bb = max(1, (build + 255) // 256)
    var cb = max(1, (capacity + 255) // 256)
    var bytes = 0
    bytes = _checked_add(bytes, _checked_mul(max(1, build), 8 + 1 + 4 + 8))
    bytes = _checked_add(bytes, _checked_mul(max(1, probe), 8 + 1 + 4 + 4 + 8))
    bytes = _checked_add(bytes, _checked_mul(max(1, capacity), 4 + 4 + 8))
    bytes = _checked_add(bytes, _checked_mul(max(1, build), 8))
    bytes = _checked_add(bytes, _checked_mul(cb, 8))
    bytes = _checked_add(bytes, _checked_mul(pb, 8))
    bytes = _checked_add(bytes, _checked_mul(max(1, build), 4 + 8 + 4 + 4))
    bytes = _checked_add(bytes, _checked_mul(bb, 8))
    return _checked_add(bytes, 2 * (j.METADATA_WORDS * 8 + 4) + 4 + 2)


def _upper_rows(task: DagTask, probe: Int, build: Int) raises -> Int:
    var upper = 0
    if task.how == 4 or task.how == 5:
        upper = probe
    else:
        upper = _checked_mul(probe, build)
        if task.how == 1 or task.how == 2 or task.how == 3 or task.how == 7:
            upper = max(upper, probe)
        if task.how == 3:
            upper = _checked_add(upper, build)
    if task.limit >= 0:
        upper = min(upper, task.limit)
    return min(upper, Int(Int32.MAX))


def _output_bytes_upper(task: DagTask, rows: Int) raises -> Int:
    var bytes = 0
    for field in task.fields:
        var width = _width(field.dtype)
        if width <= 0:
            raise Error("Mojo GPU native hash join output width is unsupported")
        bytes = _checked_add(bytes, _checked_mul(rows, width))
        # Upper-bound sidecars before pair generation reveals whether either
        # side can be absent or contains nullable/error-bearing values.
        bytes = _checked_add(bytes, _checked_add(rows, 7) // 8)
        bytes = _checked_add(bytes, _checked_mul(rows, size_of[UInt32]()))
    return bytes


def _native_column_index(
    table: Table, frame: DataFrame, field: Int
) raises -> Int:
    if field < 0 or field >= frame.width():
        raise Error("Mojo GPU native hash join key is outside its schema")
    for i in range(table.schema.width()):
        if (
            table.schema._columns[i].name() == frame._columns[field].name()
            and table.schema._columns[i].dtype()
            == frame._columns[field].dtype()
        ):
            return Int(table.columns[i])
    raise Error("Mojo GPU native hash join key owner differs from its schema")


def _pack_key_column(
    scope: Scope,
    column: NativeColumnAny,
    table_offset: Int,
    rows: Int,
) raises -> PackedKey:
    var view = column.view(table_offset, rows)
    comptime for k in range(3):
        comptime D = (DType.int32, DType.uint32, DType.float32)[k]
        if (
            view.logical.physical() == DataType.of(D)
            and view._storage.isa[NativeColumn[D]]()
        ):
            var typed = view._storage[NativeColumn[D]].copy()
            return _pack_typed_key[D](scope, typed, rows)
    raise Error("Mojo GPU native hash join key owner differs from its schema")


def _launch_count(rows: Int) -> Int:
    return (rows + 255) // 256


def try_execute(
    task: DagTask,
    left: DataFrame,
    right: DataFrame,
    scope: Scope,
    profiling: Bool,
    left_resident: Optional[Table],
    right_resident: Optional[Table],
    has_text: Bool,
    probe_rows: Int,
    build_rows: Int,
    side: Int,
) raises -> Attempt:
    if not eligible(
        task, left, right, scope, left_resident, right_resident, has_text
    ):
        return Attempt(False, None, None)

    var raw_bytes = 0
    if not left_resident:
        raw_bytes = _checked_add(raw_bytes, _raw_upload_bytes(left))
    if not right_resident:
        raw_bytes = _checked_add(raw_bytes, _raw_upload_bytes(right))
    var capacity = hash_grouping.capacity(build_rows)
    var full = task.how == 3
    var scratch_bytes = _scratch_bytes(probe_rows, build_rows, capacity, full)
    var upper_rows = _upper_rows(task, probe_rows, build_rows)
    var upper_maps = _checked_mul(max(1, upper_rows), 2 * size_of[UInt32]())
    var upper_output = _output_bytes_upper(task, upper_rows)
    var current = scope.info()[0]
    var budget = scope.info()[2]
    if budget >= 0:
        var conservative = _checked_add(
            current,
            _checked_add(
                raw_bytes,
                _checked_add(
                    scratch_bytes,
                    _checked_add(upper_maps, upper_output),
                ),
            ),
        )
        if conservative > budget:
            return Attempt(False, None, None)
        _ = scope.check(conservative - current)

    var started = Int(perf_counter_ns())
    var before = scope.transfers()
    var left_table = (
        left_resident.value().copy() if left_resident else _upload_table(
            scope, left
        )
    )
    var right_table = (
        right_resident.value().copy() if right_resident else _upload_table(
            scope, right
        )
    )
    left_table.observe_errors()
    right_table.observe_errors()
    var scratch_owner = _NativeReservation(scope, scratch_bytes)

    var left_key_index = Int(task.keys[0])
    var right_key_index = Int(task.keys[1]) - left.width()
    var left_column = _native_column_index(left_table, left, left_key_index)
    var right_column = _native_column_index(right_table, right, right_key_index)
    var rows = build_rows
    var p_rows = probe_rows
    var slots = capacity
    var blocks = _launch_count(slots)
    var probe_blocks = max(1, _launch_count(p_rows))
    var build_blocks = max(1, _launch_count(rows))
    var ctx = scope.context()
    var representatives = ctx.enqueue_create_buffer[DType.uint32](slots)
    var bucket_counts = ctx.enqueue_create_buffer[DType.uint32](slots)
    var bucket_offsets = ctx.enqueue_create_buffer[DType.uint64](slots)
    var row_slots = ctx.enqueue_create_buffer[DType.uint32](max(1, rows))
    var members_a = ctx.enqueue_create_buffer[DType.uint32](max(1, rows))
    var members_b = ctx.enqueue_create_buffer[DType.uint32](max(1, rows))
    var slot_blocks = ctx.enqueue_create_buffer[DType.uint64](blocks)
    var probe_ranges = ctx.enqueue_create_buffer[DType.uint32](max(1, p_rows))
    var probe_matches = ctx.enqueue_create_buffer[DType.uint32](max(1, p_rows))
    var output_counts = ctx.enqueue_create_buffer[DType.uint64](max(1, p_rows))
    var probe_blocks_buffer = ctx.enqueue_create_buffer[DType.uint64](
        probe_blocks
    )
    var matched_build = ctx.enqueue_create_buffer[DType.uint32](max(1, rows))
    var build_prefix = ctx.enqueue_create_buffer[DType.uint64](max(1, rows))
    var build_blocks_buffer = ctx.enqueue_create_buffer[DType.uint64](
        build_blocks
    )
    var build_flags = ctx.enqueue_create_buffer[DType.uint32](max(1, rows))
    var unmatched = ctx.enqueue_create_buffer[DType.uint32](max(1, rows))
    var metadata = ctx.enqueue_create_buffer[DType.uint64](j.METADATA_WORDS)
    var status = ctx.enqueue_create_buffer[DType.uint32](1)
    var differences = ctx.enqueue_create_buffer[DType.int32](1)
    # Allocate hash scratch before key packing queues raw-pointer consumers.
    # That leaves no device allocation between packing and the guarded launch
    # sequence below.
    var left_key = _pack_key_column(
        scope,
        left_table.native_columns[left_column],
        left_table.offset,
        left_table.rows,
    )
    var right_key_optional = Optional[PackedKey]()
    try:
        right_key_optional = Optional[PackedKey](
            _pack_key_column(
                scope,
                right_table.native_columns[right_column],
                right_table.offset,
                right_table.rows,
            )
        )
    except error:
        scope.synchronize()
        _ = left_key^
        raise error^
    var right_key = right_key_optional.take()
    var build_words = (
        left_key.words.copy() if side != 0 else right_key.words.copy()
    )
    var build_valid = (
        left_key.valid.copy() if side != 0 else right_key.valid.copy()
    )
    var probe_words = (
        right_key.words.copy() if side != 0 else left_key.words.copy()
    )
    var probe_valid = (
        right_key.valid.copy() if side != 0 else left_key.valid.copy()
    )
    # Keep pair-map buffers and reservations in the function scope so the
    # exception path can drain queued map emission before releasing them.
    var left_map_keepalive = Optional[NativeRowMap]()
    var right_map_keepalive = Optional[NativeRowMap]()
    # Copy destinations remain live until all asynchronous copies that hold
    # their raw pointers have drained, including the exception path below.
    var host_metadata = List[UInt64](length=j.METADATA_WORDS, fill=0)
    var host_status = List[UInt32](length=1, fill=0)
    var launches = 0
    try:
        ctx.enqueue_memset(metadata, UInt64(0))
        ctx.enqueue_memset(probe_blocks_buffer, UInt64(0))
        if full:
            ctx.enqueue_memset(matched_build, UInt32(0))
        ctx.enqueue_function[initialize_hash](
            representatives.unsafe_ptr(),
            bucket_counts.unsafe_ptr(),
            status.unsafe_ptr(),
            UInt32(slots),
            grid_dim=blocks,
            block_dim=256,
        )
        launches += 1
        if rows:
            ctx.enqueue_function[build_hash](
                build_words.unsafe_ptr(),
                build_valid.unsafe_ptr(),
                representatives.unsafe_ptr(),
                bucket_counts.unsafe_ptr(),
                row_slots.unsafe_ptr(),
                status.unsafe_ptr(),
                UInt32(rows),
                UInt32(slots),
                _kind(left._columns[left_key_index].dtype()),
                UInt8(task.nulls_equal or task.how == 7),
                grid_dim=build_blocks,
                block_dim=256,
            )
            launches += 1
            ctx.enqueue_function[initialize_members](
                row_slots.unsafe_ptr(),
                members_a.unsafe_ptr(),
                UInt32(rows),
                grid_dim=build_blocks,
                block_dim=256,
            )
            launches += 1
            var width = 1
            var source_is_a = True
            while width < rows:
                if source_is_a:
                    ctx.enqueue_function[merge_members](
                        members_a.unsafe_ptr(),
                        members_b.unsafe_ptr(),
                        row_slots.unsafe_ptr(),
                        UInt32(rows),
                        UInt32(width),
                        grid_dim=build_blocks,
                        block_dim=256,
                    )
                else:
                    ctx.enqueue_function[merge_members](
                        members_b.unsafe_ptr(),
                        members_a.unsafe_ptr(),
                        row_slots.unsafe_ptr(),
                        UInt32(rows),
                        UInt32(width),
                        grid_dim=build_blocks,
                        block_dim=256,
                    )
                source_is_a = not source_is_a
                width *= 2
                launches += 1
            if not source_is_a:
                var swap = members_a^
                members_a = members_b^
                members_b = swap^
        ctx.enqueue_function[scan_slot_counts](
            bucket_counts.unsafe_ptr(),
            bucket_offsets.unsafe_ptr(),
            slot_blocks.unsafe_ptr(),
            UInt32(slots),
            grid_dim=blocks,
            block_dim=256,
        )
        ctx.enqueue_function[finish_slot_blocks](
            slot_blocks.unsafe_ptr(), UInt32(blocks), grid_dim=1, block_dim=1
        )
        ctx.enqueue_function[apply_slot_blocks](
            bucket_offsets.unsafe_ptr(),
            slot_blocks.unsafe_ptr(),
            UInt32(slots),
            grid_dim=blocks,
            block_dim=256,
        )
        launches += 3
        if p_rows:
            ctx.enqueue_function[probe_hash](
                build_words.unsafe_ptr(),
                build_valid.unsafe_ptr(),
                probe_words.unsafe_ptr(),
                probe_valid.unsafe_ptr(),
                representatives.unsafe_ptr(),
                bucket_counts.unsafe_ptr(),
                bucket_offsets.unsafe_ptr(),
                members_a.unsafe_ptr(),
                probe_ranges.unsafe_ptr(),
                probe_matches.unsafe_ptr(),
                output_counts.unsafe_ptr(),
                matched_build.unsafe_ptr(),
                status.unsafe_ptr(),
                UInt32(p_rows),
                UInt32(rows),
                UInt32(slots),
                _kind(left._columns[left_key_index].dtype()),
                UInt8(task.nulls_equal),
                UInt32(task.how),
                grid_dim=probe_blocks,
                block_dim=256,
            )
            launches += 1
        if p_rows:
            ctx.enqueue_function[j.join_scan](
                output_counts.unsafe_ptr(),
                probe_blocks_buffer.unsafe_ptr(),
                differences.unsafe_ptr(),
                build_flags.unsafe_ptr(),
                UInt32(p_rows),
                UInt32(0),
                grid_dim=probe_blocks,
                block_dim=256,
            )
        var limit = UInt64(task.limit) if task.limit >= 0 else UInt64.MAX
        ctx.enqueue_function[j.join_offsets](
            probe_blocks_buffer.unsafe_ptr(),
            metadata.unsafe_ptr(),
            UInt32(probe_blocks),
            UInt32(0),
            limit,
            grid_dim=1,
            block_dim=1,
        )
        launches += 1 + Int(p_rows > 0)
        if full:
            if rows:
                ctx.enqueue_function[unmatched_flags](
                    matched_build.unsafe_ptr(),
                    build_flags.unsafe_ptr(),
                    UInt32(rows),
                    grid_dim=build_blocks,
                    block_dim=256,
                )
            ctx.enqueue_function[j.join_scan](
                build_prefix.unsafe_ptr(),
                build_blocks_buffer.unsafe_ptr(),
                differences.unsafe_ptr(),
                build_flags.unsafe_ptr(),
                UInt32(rows),
                UInt32(2),
                grid_dim=build_blocks,
                block_dim=256,
            )
            ctx.enqueue_function[j.join_offsets](
                build_blocks_buffer.unsafe_ptr(),
                metadata.unsafe_ptr(),
                UInt32(build_blocks),
                UInt32(2),
                limit,
                grid_dim=1,
                block_dim=1,
            )
            if rows:
                ctx.enqueue_function[j.join_unmatched](
                    build_prefix.unsafe_ptr(),
                    build_blocks_buffer.unsafe_ptr(),
                    build_flags.unsafe_ptr(),
                    unmatched.unsafe_ptr(),
                    UInt32(rows),
                    grid_dim=build_blocks,
                    block_dim=256,
                )
            launches += 2 + 2 * Int(rows > 0)
        ctx.enqueue_copy(host_metadata.unsafe_ptr(), metadata)
        ctx.enqueue_copy(host_status.unsafe_ptr(), status)
        scope.record_download(j.METADATA_WORDS * 8 + 4)
        scope.synchronize()
        if host_status[0] != 0:
            # Probe exhaustion is the only runtime condition that may switch
            # to the established sorted join. All hash commands have drained.
            scope.record_launches(
                launches + Int(left_table.rows > 0) + Int(right_table.rows > 0)
            )
            _ = host_metadata^
            _ = host_status^
            return Attempt(False, None, None)
        var output_rows = Int(host_metadata[0])
        if output_rows > Int(Int32.MAX):
            raise Error(
                "Mojo GPU join output exceeds supported device index range"
            )
        var map_capacity = max(1, output_rows)
        var map_bytes = _checked_mul(map_capacity, size_of[UInt32]())
        _ = scope.check(_checked_mul(map_bytes, 2))
        var left_map_reservation = _NativeReservation(scope, map_bytes)
        var right_map_reservation = _NativeReservation(scope, map_bytes)
        var left_map_buffer = ctx.enqueue_create_buffer[DType.uint32](
            map_capacity
        )
        var right_map_buffer = ctx.enqueue_create_buffer[DType.uint32](
            map_capacity
        )
        var left_map = NativeRowMap(
            scope,
            left_map_buffer^,
            output_rows,
            left_table.rows,
            task.how == 2 or task.how == 3,
            Optional[_NativeReservation](left_map_reservation^),
        )
        var right_map = NativeRowMap(
            scope,
            right_map_buffer^,
            output_rows,
            right_table.rows,
            task.how == 1
            or task.how == 3
            or task.how == 4
            or task.how == 5
            or task.how == 7,
            Optional[_NativeReservation](right_map_reservation^),
        )
        left_map_keepalive = Optional[NativeRowMap](left_map.copy())
        right_map_keepalive = Optional[NativeRowMap](right_map.copy())
        if output_rows:
            ctx.enqueue_function[emit_pair_maps](
                probe_ranges.unsafe_ptr(),
                probe_matches.unsafe_ptr(),
                output_counts.unsafe_ptr(),
                probe_blocks_buffer.unsafe_ptr(),
                members_a.unsafe_ptr(),
                bucket_offsets.unsafe_ptr(),
                unmatched.unsafe_ptr(),
                metadata.unsafe_ptr(),
                left_map.values.unsafe_ptr(),
                right_map.values.unsafe_ptr(),
                UInt32(p_rows),
                UInt32(side),
                UInt32(task.how),
                grid_dim=_launch_count(output_rows),
                block_dim=256,
            )
            launches += 1
        var output_fields = List[NativeJoinField](capacity=len(task.fields))
        for field in task.fields:
            output_fields.append(
                NativeJoinField(
                    field.left,
                    field.right - left.width() if field.right >= 0 else -1,
                    field.prefer_right,
                )
            )
        var result = left_table.gather_join(
            scope,
            right_table,
            task.schema.copy(),
            output_fields,
            left_map,
            right_map,
        )
        # `gather_join` has drained all source-map reads on success.
        _ = left_map_keepalive^
        _ = right_map_keepalive^
        # The key canonicalization kernels were submitted while packing the
        # two keys. Their copies are ordered before the hash-build work on the
        # same context, so they need no intermediate drain, but still belong
        # in the execution report's launch count.
        launches += Int(left_table.rows > 0) + Int(right_table.rows > 0)
        scope.record_launches(launches)
        var traffic = scope.transfers()
        var operator = ExecutionReport()
        operator.record(
            task.root,
            "JOIN",
            "metal",
            probe_rows,
            output_rows,
            build_rows=build_rows,
            builds=1,
            build_side="left" if side else "right",
            algorithm="mojo_gpu_native_hash_join",
            wall_ns=Int(perf_counter_ns()) - started,
        )
        var report = operator.frame()
        var extra = List[Tuple[String, Int]](
            [
                ("kernel_launches", traffic[2] - before[2]),
                ("native_hash_phase_launches", launches),
                ("synchronizations", traffic[3] - before[3]),
                ("peak_requested_bytes", scope.info()[1]),
                ("upload_bytes", traffic[0] - before[0]),
                ("download_bytes", traffic[1] - before[1]),
                ("host_result_bytes", 0),
                (
                    "resident_input_bytes",
                    left_table.retained_bytes() + right_table.retained_bytes(),
                ),
                ("resident_output_bytes", result.retained_bytes()),
                ("native_input_conversion_bytes", 0),
            ]
        )
        for item in extra:
            report = report.with_column(
                Series(item[0], Column[Int64]([Int64(item[1])]))
            )
        report = report.with_column(
            Series("runtime_implementation", Column[String](["mojo"]))
        )
        report = report.with_column(
            Series(
                "query_wall_ms",
                Column[Float64](
                    [Float64(Int(perf_counter_ns()) - started) / 1e6]
                ),
            )
        )
        _ = scratch_owner^
        _ = left_key^
        _ = right_key^
        _ = host_metadata^
        _ = host_status^
        return Attempt(
            True, Optional[Table](result^), Optional[DataFrame](report^)
        )
    except error:
        scope.synchronize()
        _ = host_metadata^
        _ = host_status^
        _ = left_map_keepalive^
        _ = right_map_keepalive^
        _ = left_table^
        _ = right_table^
        _ = scratch_owner^
        _ = left_key^
        _ = right_key^
        raise error^


def eligible(
    task: DagTask,
    left: DataFrame,
    right: DataFrame,
    scope: Scope,
    left_resident: Optional[Table],
    right_resident: Optional[Table],
    has_text: Bool,
) -> Bool:
    if (
        task.kind != DAG_JOIN
        or task.how == 6
        or (task.how < 0 or task.how > 5 and task.how != 7)
        or len(task.keys) != 2
        or has_text
        or not _supports_frame(left)
        or not _supports_frame(right)
        or not _key_column(left, Int(task.keys[0]))
        or not _key_column(right, Int(task.keys[1]) - left.width())
    ):
        return False
    if (
        left._columns[Int(task.keys[0])].dtype()
        != right._columns[Int(task.keys[1]) - left.width()].dtype()
    ):
        return False
    if left_resident and not _table_supported(
        left_resident.value(), left, scope
    ):
        return False
    if right_resident and not _table_supported(
        right_resident.value(), right, scope
    ):
        return False
    for field in task.fields:
        if (
            field.left < 0
            and field.right < 0
            or field.dtype == DataType.BOOL
            or field.dtype.is_decimal()
            or field.dtype.is_categorical()
            or field.dtype.is_nested()
            or not (field.dtype.is_numeric() or field.dtype.is_temporal())
        ):
            return False
    return True
