"""Resident exact CPU OVER scheduling owner, with explicit key provenance.

This private helper does not evaluate operands or admit public expressions.
The caller supplies immutable source/key-code certificates captured before
text/opaque normalization; original categorical codes must remain unrecoded.
"""
from max.gpu import global_idx, block_idx, thread_idx
from max.gpu.primitives.block import prefix_sum
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer, Pointer
from dataframe.dtype import DataType
from dataframe.accelerator.rows import ROW_GROUP_HEADER_WINDOW, ROW_GROUP_KEY
from dataframe.accelerator.window_keys import WindowKey
from .residency import Scope, Table, checked_add, checked_mul
from .program import kind
from . import ordering, grouping, partition_schedule as schedule


struct KeyInput(Copyable):
    var column: String
    var original: WindowKey
    var original_codes: Bool
    var category_high: Bool

    def __init__(
        out self,
        column: String,
        original: WindowKey,
        *,
        original_codes: Bool = False,
        category_high: Bool = False,
    ) raises:
        if original_codes and (
            not original.dtype.is_categorical() or original.string_hash
        ):
            raise Error("OVER original-code certificate differs from its key")
        if (
            original.dtype.is_categorical()
            and not original.string_hash
            and not original_codes
        ):
            raise Error(
                "OVER categorical keys require original unrecoded codes"
            )
        self.column = column
        self.original = original.copy()
        if category_high and not original_codes:
            raise Error(
                "OVER categorical HIGH certificate requires original codes"
            )
        self.original_codes = original_codes
        self.category_high = category_high


def _lookup_bounds(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    limits: Pointer[UInt64, ImmutAnyOrigin],
    blocks: Pointer[UInt32, MutAnyOrigin],
    keys: UInt32,
    rows: UInt32,
    stride: UInt32,
    cells: UInt64,
):
    var row = global_idx.x
    var invalid = UInt32(0)
    if row < Int(rows):
        for key in range(Int(keys)):
            var node = key + 1
            if not (literals[unsafe_offset=node] & (UInt64(1) << UInt64(63))):
                continue
            var at = Int(code[unsafe_offset=node * 4 + 1]) * Int(stride) + row
            if validity[unsafe_offset=at]:
                var value = words[unsafe_offset=at]
                if literals[unsafe_offset=node] & (UInt64(1) << 61):
                    var high = words[unsafe_offset=Int(cells) + at]
                    if high == 0:
                        invalid = 1
                    value = high - 1
                if value >= limits[unsafe_offset=key]:
                    invalid = 1
    var failures = prefix_sum[block_size=256](invalid)
    if thread_idx.x == 255:
        var count = (Int(rows) + 255) // 256
        blocks[unsafe_offset=count + block_idx.x] |= failures


struct PartitionOwner(Movable):
    var scope: Scope
    var source: Table
    var rows: Int
    var groups: Int
    var facts: List[UInt32]
    var retained: Int
    var index: DeviceBuffer[DType.uint32]
    var starts: DeviceBuffer[DType.uint32]
    var group_by_row: DeviceBuffer[DType.uint32]
    var phase_by_group: DeviceBuffer[DType.uint32]
    var rank_by_group: DeviceBuffer[DType.uint32]
    var offset_by_group: DeviceBuffer[DType.uint32]
    var phase_by_row: DeviceBuffer[DType.uint32]
    var ordinal_by_row: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        scope: Scope,
        source: Table,
        groups: Int,
        var facts: List[UInt32],
        var index: DeviceBuffer[DType.uint32],
        var starts: DeviceBuffer[DType.uint32],
        var group_by_row: DeviceBuffer[DType.uint32],
        var phase_by_group: DeviceBuffer[DType.uint32],
        var rank_by_group: DeviceBuffer[DType.uint32],
        var offset_by_group: DeviceBuffer[DType.uint32],
        var phase_by_row: DeviceBuffer[DType.uint32],
        var ordinal_by_row: DeviceBuffer[DType.uint32],
    ) raises:
        var rows = source.rows
        if (
            rows <= 0
            or rows > Int(Int32.MAX)
            or groups <= 0
            or groups > rows
            or len(facts) != 4
            or facts[0] > 1
            or facts[1] > 1
            or facts[3] > 1
            or (facts[3] == 0 and facts[2] != 1)
            or (facts[3] == 1 and facts[2] < 2)
            or facts[2] == 0
            or facts[2] > UInt32(groups)
            or not source.storage
            or not source.matrix()[].scope.same_query(scope)
            or len(index) != rows
            or len(starts) != rows + 1
            or len(group_by_row) != rows
            or len(phase_by_group) != rows
            or len(rank_by_group) != rows
            or len(offset_by_group) != rows
            or len(phase_by_row) != rows
            or len(ordinal_by_row) != rows
        ):
            raise Error("OVER resident partition geometry is invalid")
        var bytes = checked_add(checked_mul(rows, 32), 20)
        scope._retain(bytes)
        self.scope = scope.copy()
        self.source = source.copy()
        self.rows = rows
        self.groups = groups
        self.facts = facts^
        self.retained = bytes
        self.index = index^
        self.starts = starts^
        self.group_by_row = group_by_row^
        self.phase_by_group = phase_by_group^
        self.rank_by_group = rank_by_group^
        self.offset_by_group = offset_by_group^
        self.phase_by_row = phase_by_row^
        self.ordinal_by_row = ordinal_by_row^

    def __deinit__(deinit self):
        self.scope._release(self.retained)


def estimate_bytes(
    rows: Int,
    key_count: Int,
    lookup_words: Int,
    source_columns: Int,
    configured: Int,
) raises -> Int:
    """Peak additional explicit buffer/control payload, excluding input owners.

    Captured WindowKey lookup owners are caller input metadata; both freshly
    flattened host and device copies are included here. Count empty dummy
    buffers too. Source error-readback and construction are separate phases.
    """
    if (
        rows <= 0
        or rows > Int(Int32.MAX)
        or key_count <= 0
        or key_count > Int(UInt32.MAX)
        or lookup_words < 0
        or source_columns < 0
        or configured < 1
        or configured > Int(Int32.MAX)
    ):
        raise Error("OVER partition resource geometry is invalid")
    var blocks = (rows + 255) // 256
    var sample_bytes = 74752 if min(configured, rows // 65536) > 1 else 13
    var program_bytes = checked_mul(
        checked_add(
            checked_mul(key_count, 7), checked_add(6, max(1, lookup_words))
        ),
        8,
    )
    var construction = checked_add(
        checked_mul(rows, 56),
        checked_add(
            36 + sample_bytes,
            checked_add(checked_mul(blocks, 8), checked_mul(program_bytes, 2)),
        ),
    )
    return max(construction, checked_mul(checked_mul(rows, source_columns), 4))


def build(
    source: Table,
    keys: List[KeyInput],
    scope: Scope,
    configured: Int,
    *,
    allow_contiguous: Bool = True,
    cached_layout: Bool = False,
) raises -> ArcPointer[PartitionOwner]:
    # CPU frame.select/with_columns share WindowLayouts. A fallback OVER
    # uses cached layout ids, even when this occurrence forbids contiguous
    # evaluation because its child contains another OVER. Standalone probes
    # have no cache and retain the original _partition_ids fallback.
    if configured < 1 or configured > Int(Int32.MAX):
        raise Error("OVER configured worker count is invalid")
    if source.rows <= 0 or source.rows > Int(Int32.MAX):
        raise Error("Empty OVER input uses its virtual-partition control path")
    if not len(keys) or not source.storage:
        raise Error("OVER scheduling requires resident source keys")
    var matrix = source.matrix()
    if not matrix[].scope.same_query(scope):
        raise Error("OVER source belongs to a different query scope")
    if (
        len(source.columns) != source.schema.width()
        or len(source.text) != source.schema.width()
        or source.offset < 0
        or source.offset > matrix[].capacity
        or source.rows > matrix[].capacity - source.offset
        or matrix[].capacity > Int(UInt32.MAX)
        or matrix[].slots > Int(UInt32.MAX)
    ):
        raise Error("OVER source matrix geometry is invalid")
    var source_cells = checked_mul(matrix[].capacity, matrix[].slots)
    if (
        len(matrix[].words) != checked_mul(source_cells, 2)
        or len(matrix[].validity) != source_cells
        or len(matrix[].errors) != source_cells
    ):
        raise Error("OVER source matrix storage geometry is invalid")
    for slot in source.columns:
        if slot < 0 or slot >= Int64(matrix[].slots):
            raise Error("OVER source column lies outside its matrix")
    var rows = source.rows
    var key_count = len(keys)
    var lookup_words = 0
    for key in keys:
        var field = source.schema._index(key.column)
        var physical = kind(source.schema._columns[field].dtype())
        var original = key.original.dtype
        if key.original_codes and (
            not original.is_categorical()
            or key.original.string_hash
            or (Bool(source.text[field]) and not key.category_high)
        ):
            raise Error(
                "OVER original-code key is not an unrecoded private carrier"
            )
        if (
            original.is_categorical()
            and not key.original.string_hash
            and not key.original_codes
        ):
            raise Error(
                "OVER categorical keys require original unrecoded codes"
            )
        if key.category_high:
            if (
                not source.categories[field]
                or source.categories[field].value().string_codes
                or source.categories[field].value().owner[].dtype != original
            ):
                raise Error("OVER categorical HIGH owner certificate changed")
        var lookup = (
            original == DataType.STRING
            or original.is_categorical()
            or original._is_bytes()
        )
        if lookup:
            if physical != 9 or len(key.original.values[]) > Int(UInt32.MAX):
                raise Error(
                    "OVER key lookup requires bounded UInt32 carrier codes"
                )
            if key.original_codes:
                if len(key.original.values[]) != len(original.dictionary()[]):
                    raise Error("OVER original-code lookup cardinality differs")
                for code in range(len(key.original.values[])):
                    if key.original.values[][code] != UInt64(code):
                        raise Error(
                            "OVER original-code lookup is not an identity"
                        )
            lookup_words = checked_add(lookup_words, len(key.original.values[]))
        elif physical != kind(original) or len(key.original.values[]):
            raise Error("OVER fixed-width key provenance differs")
    lookup_words = max(1, lookup_words)
    var block_count = (rows + 255) // 256
    var sampled = min(configured, rows // 65536) > 1
    var program_bytes = checked_mul(
        checked_add(checked_mul(key_count, 7), checked_add(6, lookup_words)), 8
    )
    _ = scope.check(
        estimate_bytes(
            rows, key_count, lookup_words, source.schema.width(), configured
        )
    )
    # This source is an already materialized CPU-eager prefix. Newly computed
    # CUT/QCUT operands must use partition_faults instead of this observation.
    source.observe_errors()
    var code = List[Int64](capacity=4 * (key_count + 1))
    var header: List[Int64] = [ROW_GROUP_HEADER_WINDOW, Int64(key_count), 0, 0]
    code.extend(header^)
    var types = List[Int64](capacity=key_count + 1)
    types.append(0)
    var literals = List[UInt64](capacity=key_count + 1)
    literals.append(0)
    var limits = List[UInt64](capacity=key_count)
    var values = List[UInt64](capacity=lookup_words)
    for key in keys:
        var field = source.schema._index(key.column)
        var descriptor: List[Int64] = [
            ROW_GROUP_KEY,
            source.columns[field],
            0,
            0,
        ]
        code.extend(descriptor^)
        types.append(kind(source.schema._columns[field].dtype()))
        var original = key.original.dtype
        var lookup = (
            original == DataType.STRING
            or original.is_categorical()
            or original._is_bytes()
        )
        var literal = UInt64(0)
        limits.append(UInt64(len(key.original.values[])))
        if lookup:
            literal = UInt64(len(values)) | (UInt64(1) << UInt64(63))
            if key.original_codes:
                literal |= UInt64(1) << UInt64(62)
            if key.category_high:
                literal |= UInt64(1) << UInt64(61)
                code[len(code) - 1] = 2
            values.extend(key.original.values[].copy())
        literals.append(literal)
    if not len(values):
        values.append(0)
    var facts = List[UInt32](length=4, fill=0)
    var ctx = scope.context()
    var d_index = ctx.enqueue_create_buffer[DType.uint32](rows)
    var d_spare = ctx.enqueue_create_buffer[DType.uint32](rows)
    var d_positions = ctx.enqueue_create_buffer[DType.uint32](rows)
    var d_starts = ctx.enqueue_create_buffer[DType.uint32](rows + 1)
    var d_blocks = ctx.enqueue_create_buffer[DType.uint32](2 * block_count)
    var d_group_rows = ctx.enqueue_create_buffer[DType.uint32](rows)
    var d_phase_group = ctx.enqueue_create_buffer[DType.uint32](rows)
    var d_rank_group = ctx.enqueue_create_buffer[DType.uint32](rows)
    var d_offset_group = ctx.enqueue_create_buffer[DType.uint32](rows)
    var d_phase_rows = ctx.enqueue_create_buffer[DType.uint32](rows)
    var d_ordinal_rows = ctx.enqueue_create_buffer[DType.uint32](rows)
    var d_order_a = ctx.enqueue_create_buffer[DType.uint32](rows)
    var d_order_b = ctx.enqueue_create_buffer[DType.uint32](rows)
    var d_hashes = ctx.enqueue_create_buffer[DType.uint64](rows)
    var d_control = ctx.enqueue_create_buffer[DType.uint32](4)
    var d_sample_hashes = ctx.enqueue_create_buffer[DType.uint64](
        8192 if sampled else 1
    )
    var d_sample_seen = ctx.enqueue_create_buffer[DType.uint8](
        8192 if sampled else 1
    )
    var d_sample_counts = ctx.enqueue_create_buffer[DType.uint32](
        256 if sampled else 1
    )
    var d_code = ctx.enqueue_create_buffer[DType.int64](len(code))
    var d_types = ctx.enqueue_create_buffer[DType.int64](len(types))
    var d_literals = ctx.enqueue_create_buffer[DType.uint64](len(literals))
    var d_limits = ctx.enqueue_create_buffer[DType.uint64](len(limits))
    var d_values = ctx.enqueue_create_buffer[DType.uint64](len(values))
    var held_index = d_index.copy()
    var held_spare = d_spare.copy()
    var held_positions = d_positions.copy()
    var held_starts = d_starts.copy()
    var held_blocks = d_blocks.copy()
    var held_group_rows = d_group_rows.copy()
    var held_phase_group = d_phase_group.copy()
    var held_rank_group = d_rank_group.copy()
    var held_offset_group = d_offset_group.copy()
    var held_phase_rows = d_phase_rows.copy()
    var held_ordinal_rows = d_ordinal_rows.copy()
    var held_order_a = d_order_a.copy()
    var held_order_b = d_order_b.copy()
    var held_hashes = d_hashes.copy()
    var held_control = d_control.copy()
    var held_sample_hashes = d_sample_hashes.copy()
    var held_sample_seen = d_sample_seen.copy()
    var held_sample_counts = d_sample_counts.copy()
    var held_code = d_code.copy()
    var held_types = d_types.copy()
    var held_literals = d_literals.copy()
    var held_limits = d_limits.copy()
    var held_values = d_values.copy()
    var held_source = source.copy()
    try:
        ctx.enqueue_copy(d_code, code.unsafe_ptr())
        ctx.enqueue_copy(d_types, types.unsafe_ptr())
        ctx.enqueue_copy(d_literals, literals.unsafe_ptr())
        ctx.enqueue_copy(d_limits, limits.unsafe_ptr())
        ctx.enqueue_copy(d_values, values.unsafe_ptr())
        scope.record_upload(program_bytes)
        ctx.enqueue_function[ordering.order_init](
            d_index.unsafe_ptr(),
            UInt32(rows),
            grid_dim=block_count,
            block_dim=256,
        )
        scope.record_launches()
        var width = 1
        while width < rows:
            ctx.enqueue_function[ordering.order_merge](
                matrix[].words.unsafe_ptr().unsafe_offset(source.offset),
                matrix[].validity.unsafe_ptr().unsafe_offset(source.offset),
                d_index.unsafe_ptr(),
                d_spare.unsafe_ptr(),
                d_code.unsafe_ptr(),
                d_types.unsafe_ptr(),
                UInt32(1),
                UInt32(key_count),
                UInt32(rows),
                UInt32(matrix[].capacity),
                UInt32(width),
                UInt32(matrix[].slots),
                grid_dim=block_count,
                block_dim=256,
            )
            scope.record_launches()
            var prior = d_index^
            d_index = d_spare^
            d_spare = prior^
            width *= 2
        ctx.enqueue_function[grouping.group_mark](
            matrix[].words.unsafe_ptr().unsafe_offset(source.offset),
            matrix[].validity.unsafe_ptr().unsafe_offset(source.offset),
            matrix[].errors.unsafe_ptr().unsafe_offset(source.offset),
            d_index.unsafe_ptr(),
            d_positions.unsafe_ptr(),
            d_blocks.unsafe_ptr(),
            d_code.unsafe_ptr(),
            d_types.unsafe_ptr(),
            UInt32(0),
            UInt32(key_count),
            UInt32(rows),
            UInt32(matrix[].capacity),
            UInt32(matrix[].slots),
            grid_dim=block_count,
            block_dim=256,
        )
        ctx.enqueue_function[_lookup_bounds](
            matrix[].words.unsafe_ptr().unsafe_offset(source.offset),
            matrix[].validity.unsafe_ptr().unsafe_offset(source.offset),
            d_code.unsafe_ptr(),
            d_literals.unsafe_ptr(),
            d_limits.unsafe_ptr(),
            d_blocks.unsafe_ptr(),
            UInt32(key_count),
            UInt32(rows),
            UInt32(matrix[].capacity),
            UInt64(source_cells),
            grid_dim=block_count,
            block_dim=256,
        )
        ctx.enqueue_function[grouping.group_offsets](
            d_blocks.unsafe_ptr(),
            d_control.unsafe_ptr(),
            UInt32(block_count),
            grid_dim=1,
            block_dim=1,
        )
        scope.record_launches(3)
        scope.read_control_into(d_control, facts)
        scope.record_download(16)
        if facts[1]:
            raise Error("OVER key lookup or source value is invalid")
        var groups = Int(facts[0])
        if groups <= 0 or groups > rows:
            raise Error("OVER resident group count is invalid")
        var group_blocks = (groups + 255) // 256
        ctx.enqueue_function[grouping.group_starts](
            matrix[].words.unsafe_ptr().unsafe_offset(source.offset),
            matrix[].validity.unsafe_ptr().unsafe_offset(source.offset),
            d_index.unsafe_ptr(),
            d_positions.unsafe_ptr(),
            d_blocks.unsafe_ptr(),
            d_starts.unsafe_ptr(),
            d_code.unsafe_ptr(),
            d_types.unsafe_ptr(),
            UInt32(0),
            UInt32(key_count),
            UInt32(rows),
            UInt32(matrix[].capacity),
            UInt32(matrix[].slots),
            grid_dim=block_count,
            block_dim=256,
        )
        ctx.enqueue_function[schedule.end_sentinel](
            d_starts.unsafe_ptr(),
            UInt32(groups),
            UInt32(rows),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[schedule.dense_kind](
            matrix[].words.unsafe_ptr().unsafe_offset(source.offset),
            matrix[].validity.unsafe_ptr().unsafe_offset(source.offset),
            d_index.unsafe_ptr(),
            d_starts.unsafe_ptr(),
            d_code.unsafe_ptr(),
            d_literals.unsafe_ptr(),
            d_types.unsafe_ptr(),
            d_values.unsafe_ptr(),
            d_control.unsafe_ptr(),
            UInt32(0),
            UInt32(groups),
            UInt32(rows),
            UInt32(matrix[].capacity),
            UInt32(matrix[].slots),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[schedule.group_hashes](
            matrix[].words.unsafe_ptr().unsafe_offset(source.offset),
            matrix[].validity.unsafe_ptr().unsafe_offset(source.offset),
            d_index.unsafe_ptr(),
            d_starts.unsafe_ptr(),
            d_code.unsafe_ptr(),
            d_literals.unsafe_ptr(),
            d_types.unsafe_ptr(),
            d_values.unsafe_ptr(),
            d_hashes.unsafe_ptr(),
            d_spare.unsafe_ptr(),
            d_positions.unsafe_ptr(),
            UInt32(0),
            UInt32(groups),
            UInt32(matrix[].capacity),
            UInt32(matrix[].slots),
            grid_dim=group_blocks,
            block_dim=256,
        )
        ctx.enqueue_function[schedule.row_groups](
            d_index.unsafe_ptr(),
            d_starts.unsafe_ptr(),
            d_group_rows.unsafe_ptr(),
            UInt32(groups),
            UInt32(rows),
            grid_dim=block_count,
            block_dim=256,
        )
        ctx.enqueue_function[schedule.fallback_sample](
            d_hashes.unsafe_ptr(),
            d_group_rows.unsafe_ptr(),
            d_sample_hashes.unsafe_ptr(),
            d_sample_seen.unsafe_ptr(),
            d_sample_counts.unsafe_ptr(),
            d_control.unsafe_ptr(),
            UInt32(rows),
            UInt32(configured),
            grid_dim=1,
            block_dim=1,
        )
        scope.record_launches(6)
        width = 1
        while width < groups:
            ctx.enqueue_function[schedule.merge_groups](
                d_hashes.unsafe_ptr(),
                d_index.unsafe_ptr(),
                d_starts.unsafe_ptr(),
                d_control.unsafe_ptr(),
                d_spare.unsafe_ptr(),
                d_order_a.unsafe_ptr(),
                UInt32(groups),
                UInt32(rows),
                UInt32(configured),
                UInt32(0),
                UInt32(width),
                grid_dim=group_blocks,
                block_dim=256,
            )
            ctx.enqueue_function[schedule.merge_groups](
                d_hashes.unsafe_ptr(),
                d_index.unsafe_ptr(),
                d_starts.unsafe_ptr(),
                d_control.unsafe_ptr(),
                d_positions.unsafe_ptr(),
                d_order_b.unsafe_ptr(),
                UInt32(groups),
                UInt32(rows),
                UInt32(configured),
                UInt32(1),
                UInt32(width),
                grid_dim=group_blocks,
                block_dim=256,
            )
            var prior_a = d_spare^
            d_spare = d_order_a^
            d_order_a = prior_a^
            var prior_b = d_positions^
            d_positions = d_order_b^
            d_order_b = prior_b^
            width *= 2
            scope.record_launches(2)
        ctx.enqueue_function[schedule.choose_runs](
            d_starts.unsafe_ptr(),
            d_spare.unsafe_ptr(),
            d_positions.unsafe_ptr(),
            d_control.unsafe_ptr(),
            d_phase_group.unsafe_ptr(),
            d_rank_group.unsafe_ptr(),
            d_offset_group.unsafe_ptr(),
            UInt32(groups),
            UInt32(rows),
            UInt32(configured),
            UInt32(allow_contiguous),
            UInt32(cached_layout),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[schedule.row_ordinals](
            d_index.unsafe_ptr(),
            d_starts.unsafe_ptr(),
            d_group_rows.unsafe_ptr(),
            d_phase_group.unsafe_ptr(),
            d_offset_group.unsafe_ptr(),
            d_control.unsafe_ptr(),
            d_phase_rows.unsafe_ptr(),
            d_ordinal_rows.unsafe_ptr(),
            UInt32(rows),
            grid_dim=block_count,
            block_dim=256,
        )
        scope.record_launches(2)
        scope.read_control_into(d_control, facts)
        scope.record_download(16)
        var result = ArcPointer(
            PartitionOwner(
                scope,
                source,
                groups,
                facts^,
                d_index^,
                d_starts^,
                d_group_rows^,
                d_phase_group^,
                d_rank_group^,
                d_offset_group^,
                d_phase_rows^,
                d_ordinal_rows^,
            )
        )
        _ = held_index^
        _ = held_spare^
        _ = held_positions^
        _ = held_starts^
        _ = held_blocks^
        _ = held_group_rows^
        _ = held_phase_group^
        _ = held_rank_group^
        _ = held_offset_group^
        _ = held_phase_rows^
        _ = held_ordinal_rows^
        _ = held_order_a^
        _ = held_order_b^
        _ = held_hashes^
        _ = held_control^
        _ = held_sample_hashes^
        _ = held_sample_seen^
        _ = held_sample_counts^
        _ = held_code^
        _ = held_types^
        _ = held_literals^
        _ = held_limits^
        _ = held_values^
        _ = held_source^
        _ = matrix^
        _ = code^
        _ = types^
        _ = literals^
        _ = limits^
        _ = values^
        return result^
    except error:
        scope.synchronize()
        _ = held_index^
        _ = held_spare^
        _ = held_positions^
        _ = held_starts^
        _ = held_blocks^
        _ = held_group_rows^
        _ = held_phase_group^
        _ = held_rank_group^
        _ = held_offset_group^
        _ = held_phase_rows^
        _ = held_ordinal_rows^
        _ = held_order_a^
        _ = held_order_b^
        _ = held_hashes^
        _ = held_control^
        _ = held_sample_hashes^
        _ = held_sample_seen^
        _ = held_sample_counts^
        _ = held_code^
        _ = held_types^
        _ = held_literals^
        _ = held_limits^
        _ = held_values^
        _ = held_source^
        _ = matrix^
        _ = code^
        _ = types^
        _ = literals^
        _ = limits^
        _ = values^
        raise error^
