"""Actual CPU whole/bucket/heavy scheduling metadata, produced on the GPU.

Hashes use the original CPU typed key domain via grouping._partition_key.
Only bounded phase controls are read on the host. Source values and hashes
never leave the device. This is the primitive List producer schedule; checked
reducer feed/state phases require their separate CPU association proof.
"""
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer, Pointer
from dataframe import Expr, DataFrame, DataType
from dataframe.partition import _mix, _combine
from dataframe.accelerator.rows import ROW_GROUP_KEY
from dataframe.accelerator.window_keys import WindowKey
from .residency import (
    Scope,
    Matrix,
    Table,
    checked_add,
    checked_mul,
    CELL_BYTES,
)
from .runtime import _preflight
from .program import Program
from . import grouping, ordering


def _validate_lookup_codes(
    words: Pointer[UInt64, ImmutAnyOrigin],
    valid: Pointer[UInt8, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    lengths: Pointer[UInt64, ImmutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    if global_idx.x != 0:
        return
    for at in range(5):
        control[unsafe_offset=at] = 0
    for row in range(Int(rows)):
        for key in range(Int(keys)):
            var limit = lengths[unsafe_offset=key]
            if limit == UInt64.MAX:
                continue
            var node = Int(start) + 1 + key
            var cell = Int(code[unsafe_offset=4 * node + 1]) * Int(stride) + row
            if valid[unsafe_offset=cell] and words[unsafe_offset=cell] >= limit:
                control[unsafe_offset=0] = 1
                control[unsafe_offset=1] = UInt32(row)
                control[unsafe_offset=2] = UInt32(key)
                return


def _hash_rows(
    words: Pointer[UInt64, ImmutAnyOrigin],
    valid: Pointer[UInt8, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    lookup: Pointer[UInt64, ImmutAnyOrigin],
    output: Pointer[UInt64, MutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var hash = UInt64(0)
    for key in range(Int(keys)):
        var node = Int(start) + 1 + key
        var cell = Int(code[unsafe_offset=4 * node + 1]) * Int(stride) + row
        var raw = UInt64(0x9E3779B97F4A7C15)
        if valid[unsafe_offset=cell]:
            raw = grouping._partition_key(
                words,
                lookup,
                cell,
                types[unsafe_offset=node],
                literals[unsafe_offset=node],
                Int(stride) * Int(slots),
            )
        hash = _mix(raw) if key == 0 else _combine(hash, raw)
    output[unsafe_offset=row] = hash


def _sample(
    hashes: Pointer[UInt64, ImmutAnyOrigin],
    distinct_hashes: Pointer[UInt64, MutAnyOrigin],
    distinct_seen: Pointer[UInt8, MutAnyOrigin],
    counts: Pointer[UInt32, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    workers: UInt32,
    buckets: UInt32,
):
    if global_idx.x != 0:
        return
    for slot in range(8192):
        distinct_seen[unsafe_offset=slot] = 0
    for slot in range(256):
        counts[unsafe_offset=slot] = 0
    var sample = min(Int(rows), 4096)
    var stride = max(1, Int(rows) // sample)
    var row = 0
    var taken = 0
    var occupied = 0
    var largest = 0
    var distinct = 0
    while row < Int(rows) and taken < sample:
        var hash = hashes[unsafe_offset=row]
        var slot = Int(hash >> UInt64(56))
        if not counts[unsafe_offset=slot]:
            occupied += 1
        counts[unsafe_offset=slot] += 1
        largest = max(largest, Int(counts[unsafe_offset=slot]))
        var at = Int(hash & UInt64(8191))
        while (
            distinct_seen[unsafe_offset=at]
            and distinct_hashes[unsafe_offset=at] != hash
        ):
            at = (at + 1) & 8191
        if not distinct_seen[unsafe_offset=at]:
            distinct += 1
            distinct_seen[unsafe_offset=at] = 1
            distinct_hashes[unsafe_offset=at] = hash
        taken += 1
        row = taken * stride + (taken * 7919) % stride
    var whole = workers <= 1 or 2 * occupied < min(taken, 256)
    if not whole and taken >= 128 and 4 * largest >= taken:
        whole = 3 * distinct <= taken
    control[unsafe_offset=0] = UInt32(whole)
    control[unsafe_offset=1] = workers
    control[unsafe_offset=2] = buckets
    control[unsafe_offset=3] = 0
    control[unsafe_offset=4] = 0


def _bucket_merge(
    hashes: Pointer[UInt64, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    target: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    shift: UInt32,
    run_width: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var width = Int(run_width)
    var base = row // (2 * width) * (2 * width)
    var middle = min(base + width, Int(rows))
    var end = min(base + 2 * width, Int(rows))
    var left = row < middle
    var first = middle if left else base
    var lo = first
    var hi = end if left else middle
    var own = index[unsafe_offset=row]
    var key = hashes[unsafe_offset=Int(own)] >> UInt64(shift)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        var other = index[unsafe_offset=mid]
        var other_key = hashes[unsafe_offset=Int(other)] >> UInt64(shift)
        if other_key < key or (other_key == key and other < own):
            lo = mid + 1
        else:
            hi = mid
    target[
        unsafe_offset=base + row - (base if left else middle) + lo - first
    ] = own


def _starts(
    hashes: Pointer[UInt64, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    buckets: UInt32,
    shift: UInt32,
):
    var bucket = global_idx.x
    if bucket > Int(buckets):
        return
    var lo = 0
    var hi = Int(rows)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        var key = hashes[unsafe_offset=Int(index[unsafe_offset=mid])] >> UInt64(
            shift
        )
        if key < UInt64(bucket):
            lo = mid + 1
        else:
            hi = mid
    starts[unsafe_offset=bucket] = UInt32(lo)


def _phase_order(
    starts: Pointer[UInt32, ImmutAnyOrigin],
    order: Pointer[UInt32, MutAnyOrigin],
    ranks: Pointer[UInt32, MutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    buckets: UInt32,
):
    if global_idx.x != 0:
        return
    for bucket in range(Int(buckets)):
        order[unsafe_offset=bucket] = UInt32.MAX
        ranks[unsafe_offset=bucket] = UInt32.MAX
    if control[unsafe_offset=0]:
        control[unsafe_offset=3] = 0
        control[unsafe_offset=4] = 1
        return
    var threshold = max(Int(rows) // 8, 4 * Int(rows) // Int(buckets))
    var phases = UInt32(0)
    var heavy = UInt32(0)
    # CPU runs all heavy buckets synchronously before submitting light jobs.
    for phase_kind in range(2):
        for bucket in range(Int(buckets)):
            var count = Int(
                starts[unsafe_offset=bucket + 1] - starts[unsafe_offset=bucket]
            )
            if count == 0 or (count > threshold) != (phase_kind == 0):
                continue
            order[unsafe_offset=Int(phases)] = UInt32(bucket)
            ranks[unsafe_offset=bucket] = phases
            phases += 1
            heavy += UInt32(phase_kind == 0)
    control[unsafe_offset=3] = heavy
    control[unsafe_offset=4] = phases


def _ordinals(
    hashes: Pointer[UInt64, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    starts: Pointer[UInt32, ImmutAnyOrigin],
    ranks: Pointer[UInt32, ImmutAnyOrigin],
    control: Pointer[UInt32, ImmutAnyOrigin],
    phase: Pointer[UInt32, MutAnyOrigin],
    ordinal: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    shift: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    if control[unsafe_offset=0]:
        phase[unsafe_offset=row] = 0
        ordinal[unsafe_offset=row] = UInt32(row)
        return
    var source = Int(index[unsafe_offset=row])
    var bucket = Int(hashes[unsafe_offset=source] >> UInt64(shift))
    phase[unsafe_offset=source] = ranks[unsafe_offset=bucket]
    ordinal[unsafe_offset=source] = UInt32(row) - starts[unsafe_offset=bucket]


struct ScheduleOwner(Movable):
    var scope: Scope
    var rows: Int
    var buckets: Int
    var facts: List[UInt32]
    var retained: Int
    var hashes: DeviceBuffer[DType.uint64]
    var index: DeviceBuffer[DType.uint32]
    var starts: DeviceBuffer[DType.uint32]
    var phase_order: DeviceBuffer[DType.uint32]
    var phase_rank: DeviceBuffer[DType.uint32]
    var phase_by_row: DeviceBuffer[DType.uint32]
    var ordinal: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        scope: Scope,
        rows: Int,
        buckets: Int,
        var facts: List[UInt32],
        var hashes: DeviceBuffer[DType.uint64],
        var index: DeviceBuffer[DType.uint32],
        var starts: DeviceBuffer[DType.uint32],
        var order: DeviceBuffer[DType.uint32],
        var ranks: DeviceBuffer[DType.uint32],
        var phases: DeviceBuffer[DType.uint32],
        var ordinal: DeviceBuffer[DType.uint32],
    ) raises:
        if (
            rows <= 0
            or rows > Int(Int32.MAX)
            or buckets < 1
            or buckets > 256
            or len(facts) != 5
            or facts[2] != UInt32(buckets)
            or facts[4] < 1
            or facts[4] > UInt32(buckets)
            or len(hashes) != rows
            or len(index) != rows
            or len(starts) != buckets + 1
            or len(order) != buckets
            or len(ranks) != buckets
            or len(phases) != rows
            or len(ordinal) != rows
        ):
            raise Error("Grouped CPU schedule geometry is invalid")
        var bytes = checked_add(
            checked_mul(rows, 20), checked_add(checked_mul(buckets, 12), 24)
        )
        scope._retain(bytes)
        self.scope = scope.copy()
        self.rows = rows
        self.buckets = buckets
        self.facts = facts^
        self.retained = bytes
        self.hashes = hashes^
        self.index = index^
        self.starts = starts^
        self.phase_order = order^
        self.phase_rank = ranks^
        self.phase_by_row = phases^
        self.ordinal = ordinal^

    def __deinit__(deinit self):
        self.scope._release(self.retained)


def build(
    source: Table,
    keys: List[String],
    scope: Scope,
    configured: Int,
    *,
    provenance: List[WindowKey] = List[WindowKey](),
) raises -> ArcPointer[ScheduleOwner]:
    if configured < 1 or configured > Int(Int32.MAX):
        raise Error("Grouped CPU schedule worker count is invalid")
    if not source.rows:
        raise Error("Empty input uses grouped empty fast path")
    if not source.storage or not source.matrix()[].scope.same_query(scope):
        raise Error(
            "Grouped CPU schedule source belongs to another query context"
        )
    if not len(keys):
        raise Error("Grouped CPU schedule keys are missing")
    for name in keys:
        var dtype = source.schema.column(name).dtype()
        if (
            dtype.is_nested()
            or dtype == DataType.STRING
            or dtype._is_bytes()
            or dtype.is_categorical()
            or (
                dtype.physical() == DataType.of(DType.int128)
                and not dtype.is_decimal()
            )
        ):
            raise Error(
                "Grouped CPU schedule requires original fixed-width key provenance"
            )
    var source_matrix = source.matrix()
    if (
        len(source.columns) != source.schema.width()
        or source.offset < 0
        or source.offset > source_matrix[].capacity
        or source.rows > source_matrix[].capacity - source.offset
    ):
        raise Error("Grouped CPU schedule source geometry is invalid")
    for column in source.columns:
        if column < 0 or column >= Int64(source_matrix[].slots):
            raise Error(
                "Grouped CPU schedule source column lies outside its matrix"
            )
    var key_query = (
        source.schema.lazy()
        .group_by(keys, maintain_order=True)
        .agg(List[Expr]())
    )
    key_query._nodes[len(key_query._nodes) - 1].window_keys = provenance.copy()
    var has_lookup = False
    for at in range(len(provenance)):
        var found = False
        for name in keys:
            found |= name == provenance[at].name
        if not found:
            raise Error("Grouped original key provenance name is missing")
        for previous in range(at):
            if provenance[previous].name == provenance[at].name:
                raise Error("Grouped original key provenance is duplicated")
        var dtype = provenance[at].dtype
        var lookup = (
            dtype == DataType.STRING
            or dtype._is_bytes()
            or dtype.is_categorical()
        )
        has_lookup |= lookup
        if (
            lookup
            and source.schema.column(provenance[at].name).dtype()
            != DataType.UINT32
        ):
            raise Error(
                "Grouped original key lookup requires its UInt32 transport carrier"
            )
        if dtype.is_categorical() and (
            not dtype.has_dictionary()
            or len(provenance[at].values[]) != len(dtype.dictionary()[])
        ):
            raise Error(
                "Grouped original categorical dictionary lookup differs"
            )
    var key_lengths = List[UInt64](
        length=len(keys) if has_lookup else 0, fill=UInt64.MAX
    )
    if has_lookup:
        for key in range(len(keys)):
            for original in provenance:
                if original.name != keys[key]:
                    continue
                if (
                    original.dtype == DataType.STRING
                    or original.dtype._is_bytes()
                    or original.dtype.is_categorical()
                ):
                    key_lengths[key] = UInt64(len(original.values[]))
    var plan = _preflight(key_query, source_rows=source.rows)
    var start = -1
    for step in plan.steps:
        if step.grouped:
            if start >= 0:
                raise Error("Grouped CPU schedule needs one grouping header")
            start = step.start
    if start < 0 or not len(keys):
        raise Error("Grouped CPU schedule keys are missing")
    var program = Program(plan)
    for node in range(len(program.node_types)):
        if program.code[4 * node] != ROW_GROUP_KEY:
            continue
        var logical = Int(program.code[4 * node + 1])
        var name = plan.source._columns[logical].name()
        var found = -1
        for column in range(source.schema.width()):
            if source.schema._columns[column].name() == name:
                found = column
                break
        if found < 0:
            raise Error("Grouped CPU schedule key source is missing")
        program.code[4 * node + 1] = source.columns[found]
    var rows = source.rows
    var workers = max(
        1, max(min(configured, rows // 65536), min(configured, rows // 16384))
    )
    var buckets = 1
    var shift = 64
    while buckets < 2 * workers and buckets < 256:
        buckets *= 2
        shift -= 1
    var program_bytes = checked_mul(
        checked_add(
            len(program.code),
            checked_add(len(program.literals), len(program.node_types)),
        ),
        8,
    )
    var bytes = checked_add(
        checked_mul(rows, 24), checked_add(checked_mul(buckets, 12), 44 + 74752)
    )
    bytes = checked_add(bytes, checked_mul(program_bytes, 2))
    # Captured WindowKey owners belong to caller staging. This phase allocates
    # the flattened host lookup +device copy and host/device key length guards.
    bytes = checked_add(
        bytes,
        checked_mul(checked_add(len(plan.key_values), len(key_lengths)), 16),
    )
    _ = scope.check(
        max(bytes, checked_mul(checked_mul(rows, source.schema.width()), 4))
    )
    source.observe_errors()
    var matrix = source.matrix()
    var ctx = scope.context()
    var hashes = ctx.enqueue_create_buffer[DType.uint64](rows)
    var index = ctx.enqueue_create_buffer[DType.uint32](rows)
    var spare = ctx.enqueue_create_buffer[DType.uint32](rows)
    var starts = ctx.enqueue_create_buffer[DType.uint32](buckets + 1)
    var order = ctx.enqueue_create_buffer[DType.uint32](buckets)
    var ranks = ctx.enqueue_create_buffer[DType.uint32](buckets)
    var phases = ctx.enqueue_create_buffer[DType.uint32](rows)
    var ordinal = ctx.enqueue_create_buffer[DType.uint32](rows)
    var control = ctx.enqueue_create_buffer[DType.uint32](5)
    var sample_hashes = ctx.enqueue_create_buffer[DType.uint64](8192)
    var sample_seen = ctx.enqueue_create_buffer[DType.uint8](8192)
    var sample_counts = ctx.enqueue_create_buffer[DType.uint32](256)
    var code = ctx.enqueue_create_buffer[DType.int64](len(program.code))
    var literals = ctx.enqueue_create_buffer[DType.uint64](
        len(program.literals)
    )
    var types = ctx.enqueue_create_buffer[DType.int64](len(program.node_types))
    var lookup = Optional[DeviceBuffer[DType.uint64]]()
    var lengths = Optional[DeviceBuffer[DType.uint64]]()
    if len(plan.key_values):
        lookup = ctx.enqueue_create_buffer[DType.uint64](len(plan.key_values))
    if has_lookup:
        lengths = ctx.enqueue_create_buffer[DType.uint64](len(key_lengths))
    var held_lookup = lookup.copy()
    var held_lengths = lengths.copy()
    var held_hashes = hashes.copy()
    var held_index = index.copy()
    var held_spare = spare.copy()
    var held_starts = starts.copy()
    var held_order = order.copy()
    var held_ranks = ranks.copy()
    var held_phases = phases.copy()
    var held_ordinal = ordinal.copy()
    var held_control = control.copy()
    var held_sample_hashes = sample_hashes.copy()
    var held_sample_seen = sample_seen.copy()
    var held_sample_counts = sample_counts.copy()
    var held_code = code.copy()
    var held_literals = literals.copy()
    var held_types = types.copy()
    try:
        ctx.enqueue_copy(code, program.code.unsafe_ptr())
        ctx.enqueue_copy(literals, program.literals.unsafe_ptr())
        ctx.enqueue_copy(types, program.node_types.unsafe_ptr())
        if lookup:
            ctx.enqueue_copy(lookup.value(), plan.key_values.unsafe_ptr())
        if lengths:
            ctx.enqueue_copy(lengths.value(), key_lengths.unsafe_ptr())
        scope.record_upload(
            checked_add(
                program_bytes,
                checked_mul(
                    checked_add(len(plan.key_values), len(key_lengths)), 8
                ),
            )
        )
        if has_lookup:
            ctx.enqueue_function[_validate_lookup_codes](
                matrix[].words.unsafe_ptr().unsafe_offset(source.offset),
                matrix[].validity.unsafe_ptr().unsafe_offset(source.offset),
                code.unsafe_ptr(),
                lengths.value().unsafe_ptr(),
                control.unsafe_ptr(),
                UInt32(start),
                UInt32(len(keys)),
                UInt32(rows),
                UInt32(matrix[].capacity),
                grid_dim=1,
                block_dim=1,
            )
            scope.record_launches()
            var proof = scope.read_control(control)
            scope.record_download(20)
            if proof[0]:
                raise Error("Mojo GPU original key code is outside its domain")
            _ = proof^
        var raw_lookup: Pointer[UInt64, ImmutAnyOrigin] = (
            literals.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
        )
        if lookup:
            raw_lookup = (
                lookup.value()
                .unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin]()
            )
        var blocks = (rows + 255) // 256
        ctx.enqueue_function[_hash_rows](
            matrix[].words.unsafe_ptr().unsafe_offset(source.offset),
            matrix[].validity.unsafe_ptr().unsafe_offset(source.offset),
            code.unsafe_ptr(),
            literals.unsafe_ptr().unsafe_mut_cast[False](),
            types.unsafe_ptr().unsafe_mut_cast[False](),
            raw_lookup,
            hashes.unsafe_ptr(),
            UInt32(start),
            UInt32(len(keys)),
            UInt32(rows),
            UInt32(matrix[].capacity),
            UInt32(matrix[].slots),
            grid_dim=blocks,
            block_dim=256,
        )
        ctx.enqueue_function[_sample](
            hashes.unsafe_ptr(),
            sample_hashes.unsafe_ptr(),
            sample_seen.unsafe_ptr(),
            sample_counts.unsafe_ptr(),
            control.unsafe_ptr(),
            UInt32(rows),
            UInt32(workers),
            UInt32(buckets),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[ordering.order_init](
            index.unsafe_ptr(), UInt32(rows), grid_dim=blocks, block_dim=256
        )
        scope.record_launches(3)
        var width = 1
        while width < rows:
            ctx.enqueue_function[_bucket_merge](
                hashes.unsafe_ptr(),
                index.unsafe_ptr(),
                spare.unsafe_ptr(),
                UInt32(rows),
                UInt32(shift),
                UInt32(width),
                grid_dim=blocks,
                block_dim=256,
            )
            var old = index^
            index = spare^
            spare = old^
            width *= 2
            scope.record_launches()
        ctx.enqueue_function[_starts](
            hashes.unsafe_ptr(),
            index.unsafe_ptr(),
            starts.unsafe_ptr(),
            UInt32(rows),
            UInt32(buckets),
            UInt32(shift),
            grid_dim=(buckets + 1 + 127) // 128,
            block_dim=128,
        )
        ctx.enqueue_function[_phase_order](
            starts.unsafe_ptr(),
            order.unsafe_ptr(),
            ranks.unsafe_ptr(),
            control.unsafe_ptr(),
            UInt32(rows),
            UInt32(buckets),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[_ordinals](
            hashes.unsafe_ptr(),
            index.unsafe_ptr(),
            starts.unsafe_ptr(),
            ranks.unsafe_ptr(),
            control.unsafe_ptr(),
            phases.unsafe_ptr(),
            ordinal.unsafe_ptr(),
            UInt32(rows),
            UInt32(shift),
            grid_dim=blocks,
            block_dim=256,
        )
        scope.record_launches(3)
        var facts = scope.read_control(control)
        scope.record_download(20)
        var result = ArcPointer(
            ScheduleOwner(
                scope,
                rows,
                buckets,
                facts^,
                hashes^,
                index^,
                starts^,
                order^,
                ranks^,
                phases^,
                ordinal^,
            )
        )
        _ = plan^
        _ = key_lengths^
        _ = held_lookup^
        _ = held_lengths^
        _ = program^
        _ = matrix^
        _ = held_hashes^
        _ = held_index^
        _ = held_spare^
        _ = held_starts^
        _ = held_order^
        _ = held_ranks^
        _ = held_phases^
        _ = held_ordinal^
        _ = held_control^
        _ = held_sample_hashes^
        _ = held_sample_seen^
        _ = held_sample_counts^
        _ = held_code^
        _ = held_literals^
        _ = held_types^
        return result^
    except error:
        scope.synchronize()
        _ = plan^
        _ = key_lengths^
        _ = held_lookup^
        _ = held_lengths^
        _ = program^
        _ = matrix^
        _ = held_hashes^
        _ = held_index^
        _ = held_spare^
        _ = held_starts^
        _ = held_order^
        _ = held_ranks^
        _ = held_phases^
        _ = held_ordinal^
        _ = held_control^
        _ = held_sample_hashes^
        _ = held_sample_seen^
        _ = held_sample_counts^
        _ = held_code^
        _ = held_literals^
        _ = held_types^
        raise error^


struct HostPhases(Movable):
    var scope: Scope
    var retained: Int
    var starts: List[UInt32]
    var order: List[UInt32]

    def __init__(
        out self,
        scope: Scope,
        var starts: List[UInt32],
        var order: List[UInt32],
    ) raises:
        var bytes = checked_mul(checked_add(len(starts), len(order)), 4)
        scope._retain(bytes)
        self.scope = scope.copy()
        self.retained = bytes
        self.starts = starts^
        self.order = order^

    def __deinit__(deinit self):
        self.scope._release(self.retained)


def read_phases(
    value: ArcPointer[ScheduleOwner], scope: Scope
) raises -> HostPhases:
    if not value[].scope.same_query(scope):
        raise Error("Grouped CPU schedule belongs to another query context")
    var bytes = checked_add(checked_mul(value[].buckets, 8), 4)
    _ = scope.check(bytes)
    var starts = scope.read_control(value[].starts)
    var order = scope.read_control(value[].phase_order)
    scope.record_download(bytes)
    return HostPhases(scope, starts^, order^)


def _gather(
    words: Pointer[UInt64, ImmutAnyOrigin],
    valid: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    mapping: Pointer[Int64, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_valid: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    begin: UInt32,
    rows: UInt32,
    capacity: UInt32,
    slots: UInt32,
    columns: UInt32,
    source_offset: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var original = Int(index[unsafe_offset=Int(begin) + row]) + Int(
        source_offset
    )
    for column in range(Int(columns)):
        var cell = Int(mapping[unsafe_offset=column]) * Int(capacity) + original
        var target = column * Int(rows) + row
        output_words[unsafe_offset=target] = words[unsafe_offset=cell]
        output_words[unsafe_offset=Int(rows) * Int(columns) + target] = words[
            unsafe_offset=Int(capacity) * Int(slots) + cell
        ]
        output_valid[unsafe_offset=target] = valid[unsafe_offset=cell]
        output_errors[unsafe_offset=target] = errors[unsafe_offset=cell]


def gather_phase(
    source: Table,
    value: ArcPointer[ScheduleOwner],
    phases: HostPhases,
    phase: Int,
    scope: Scope,
) raises -> Table:
    if not value[].scope.same_query(scope) or not phases.scope.same_query(
        scope
    ):
        raise Error("Grouped CPU schedule belongs to another query context")
    if (
        source.rows != value[].rows
        or phase < 0
        or phase >= Int(value[].facts[4])
    ):
        raise Error("Grouped CPU phase ordinal is invalid")
    var matrix = source.matrix()
    if not matrix[].scope.same_query(scope):
        raise Error("Grouped CPU phase source belongs to another query context")
    if (
        len(phases.starts) != value[].buckets + 1
        or len(phases.order) != value[].buckets
    ):
        raise Error("Grouped CPU phase controls have invalid geometry")
    var columns = source.schema.width()
    if (
        len(source.columns) != columns
        or source.offset < 0
        or source.offset > matrix[].capacity
        or source.rows > matrix[].capacity - source.offset
    ):
        raise Error("Grouped CPU phase source geometry is invalid")
    for column in source.columns:
        if column < 0 or column >= Int64(matrix[].slots):
            raise Error("Grouped CPU phase column lies outside its matrix")
    if value[].facts[0]:
        return source.copy()
    var bucket = Int(phases.order[phase])
    if bucket < 0 or bucket >= value[].buckets:
        raise Error("Grouped CPU phase bucket is invalid")
    var begin = Int(phases.starts[bucket])
    var rows = Int(phases.starts[bucket + 1]) - begin
    if (
        begin < 0
        or rows <= 0
        or begin > source.rows
        or rows > source.rows - begin
    ):
        raise Error("Grouped CPU phase range is invalid")
    var cells = checked_mul(rows, columns)
    _ = scope.check(
        checked_add(checked_mul(cells, CELL_BYTES), checked_mul(columns, 16))
    )
    var mapping = source.columns.copy()
    var ctx = scope.context()
    var dm = ctx.enqueue_create_buffer[DType.int64](columns)
    var dw = ctx.enqueue_create_buffer[DType.uint64](2 * cells)
    var dv = ctx.enqueue_create_buffer[DType.uint8](cells)
    var de = ctx.enqueue_create_buffer[DType.uint32](cells)
    var held_mapping = dm.copy()
    var held_words = dw.copy()
    var held_valid = dv.copy()
    var held_errors = de.copy()
    var held_schedule = value.copy()
    var held_source = source.copy()
    try:
        ctx.enqueue_copy(dm, mapping.unsafe_ptr())
        scope.record_upload(columns * 8)
        ctx.enqueue_function[_gather](
            matrix[].words.unsafe_ptr(),
            matrix[].validity.unsafe_ptr(),
            matrix[].errors.unsafe_ptr(),
            value[].index.unsafe_ptr(),
            dm.unsafe_ptr(),
            dw.unsafe_ptr(),
            dv.unsafe_ptr(),
            de.unsafe_ptr(),
            UInt32(begin),
            UInt32(rows),
            UInt32(matrix[].capacity),
            UInt32(matrix[].slots),
            UInt32(columns),
            UInt32(source.offset),
            grid_dim=(rows + 255) // 256,
            block_dim=256,
        )
        scope.record_launches()
        scope.synchronize()
        var output_columns = List[Int64]()
        for column in range(columns):
            output_columns.append(Int64(column))
        var result = Table(
            Matrix(scope, rows, columns, dw^, dv^, de^),
            source.schema.copy(),
            output_columns^,
            rows,
        )
        _ = mapping^
        _ = matrix^
        _ = held_mapping^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        _ = held_schedule^
        _ = held_source^
        return result^
    except error:
        scope.synchronize()
        _ = mapping^
        _ = matrix^
        _ = held_mapping^
        _ = held_words^
        _ = held_valid^
        _ = held_errors^
        _ = held_schedule^
        _ = held_source^
        raise error^
