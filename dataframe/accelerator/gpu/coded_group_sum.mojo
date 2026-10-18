"""Direct-address Int64 sums over small retained String dictionaries.

Only a terminal, single-key/single-SUM frame scan is admitted. Original codes
are immutable input, never recomputed from strings. Block-private UInt32
atomics accumulate exact signed 128-bit sums; a second kernel merges them.
No row sort, generic word matrix, or native Metal helper participates.
"""
from max.gpu import block_idx, thread_idx, barrier
from std.atomic import Atomic
from std.collections import Dict
from std.memory import Pointer, unsafe_stack_allocation
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, Series
from dataframe.binding import bind
from dataframe.expr import COL, SUM
from dataframe.lazy import LazyFrame, SCAN_FRAME, AGG
from dataframe.string_column import StringColumn
from dataframe.accelerator.metadata import schema_frame
from dataframe.accelerator.faults import data_fault
from dataframe.execution_report import ExecutionReport
from dataframe.trace import trace_path
from .decimal import Wide, wide_add
from .residency import Scope, checked_add, checked_mul

comptime MAX_GROUPS = 128  # Includes one bin for the null key.
comptime STATE_WORDS = 5  # Four sum limbs and the first source row.


def eligible(query: LazyFrame) raises -> Bool:
    """Inspect logical shape and retained-code metadata, never source rows."""
    if (
        len(query._nodes) != 2
        or query._nodes[0].kind != SCAN_FRAME
        or query._nodes[0].source_failure
        or query._nodes[1].kind != AGG
        or query._nodes[1].left != 0
        or len(query._nodes[1].names) != 1
        or len(query._nodes[1].exprs) != 1
    ):
        return False
    ref node = query._nodes[1]
    ref frame = query._frames[query._nodes[0].offset]
    var bound = bind(node.exprs[0], frame._columns)
    if (
        len(bound.expr._nodes) != 2
        or bound.expr._nodes[0].op != COL
        or bound.expr._nodes[1].op != SUM
        or bound.expr._nodes[1].min_count != 0
        or bound.expr._nodes[1].left != 0
        or bound.dtypes[0] != DataType.INT64
        or frame.height() > Int(UInt32.MAX)
    ):
        return False
    var key = frame.column(node.names[0])
    if key.dtype() != DataType.STRING:
        return False
    var dtype = Optional[DataType]()
    for part in key.chunks():
        ref strings = part._data[StringColumn]
        if not strings._codes or strings._codes.value()[].rows:
            return False
        var current = strings._codes.value()[].dtype
        if len(current.dictionary()[]) >= MAX_GROUPS:
            return False
        if dtype and dtype.value() != current:
            return False
        dtype = current
    return Bool(dtype)


@always_inline
def _valid(bits: Pointer[UInt8, ImmutAnyOrigin], at: Int) -> Bool:
    return (bits[unsafe_offset=at // 8] >> UInt8(at % 8)) & 1 != 0


def _partials(
    codes: Pointer[UInt32, ImmutAnyOrigin],
    code_map: Pointer[UInt32, ImmutAnyOrigin],
    values: Pointer[Int64, ImmutAnyOrigin],
    key_bits: Pointer[UInt8, ImmutAnyOrigin],
    value_bits: Pointer[UInt8, ImmutAnyOrigin],
    partials: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    groups: UInt32,
    blocks: UInt32,
    key_offset: UInt32,
    value_offset: UInt32,
    key_nulls: UInt32,
    value_nulls: UInt32,
):
    var state = unsafe_stack_allocation[
        MAX_GROUPS * STATE_WORDS,
        DType.uint32,
        address_space=AddressSpace.SHARED,
    ]()
    var lane = thread_idx.x
    for at in range(lane, Int(groups) * STATE_WORDS, 256):
        state[unsafe_offset=at] = UInt32.MAX if at % STATE_WORDS == 4 else 0
    barrier()
    for row in range(block_idx.x * 256 + lane, Int(rows), Int(blocks) * 256):
        var group = Int(groups) - 1
        if not key_nulls or _valid(key_bits, row + Int(key_offset)):
            group = Int(code_map[unsafe_offset=Int(codes[unsafe_offset=row])])
        # Retained dictionary codes are a source storage invariant.
        var base = group * STATE_WORDS
        Atomic[UInt32].min(state.unsafe_offset(base + 4), UInt32(row))
        if value_nulls and not _valid(value_bits, row + Int(value_offset)):
            continue
        var value = values[unsafe_offset=row].cast[DType.uint64]()
        var negative = (value >> 63) != 0
        var carry = UInt32(0)
        # Each atomic's wrap contributes a carry to the next limb. Other
        # lanes may interleave, but no limb is read until after the barrier.
        comptime for word in range(4):
            var addend = (
                UInt32(value) if word
                == 0 else UInt32(value >> 32) if word
                == 1 else UInt32.MAX if negative else UInt32(0)
            )
            var next_carry = UInt32(0)
            if addend:
                var old = Atomic[UInt32].fetch_add(
                    state.unsafe_offset(base + word), addend
                )
                next_carry = UInt32(old + addend < old)
            if carry:
                var old = Atomic[UInt32].fetch_add(
                    state.unsafe_offset(base + word), carry
                )
                next_carry += UInt32(old + carry < old)
            carry = next_carry
    barrier()
    for at in range(lane, Int(groups) * STATE_WORDS, 256):
        partials[
            unsafe_offset=block_idx.x * Int(groups) * STATE_WORDS + at
        ] = state[unsafe_offset=at]


def _finish(
    partials: Pointer[UInt32, ImmutAnyOrigin],
    result: Pointer[UInt32, MutAnyOrigin],
    groups: UInt32,
    blocks: UInt32,
):
    var group = block_idx.x * 128 + thread_idx.x
    if group >= Int(groups):
        return
    var total = Wide(0)
    var first = UInt32.MAX
    for block in range(Int(blocks)):
        var at = (block * Int(groups) + group) * STATE_WORDS
        total = wide_add(
            total,
            Wide(
                UInt64(partials[unsafe_offset=at])
                | (UInt64(partials[unsafe_offset=at + 1]) << 32),
                UInt64(partials[unsafe_offset=at + 2])
                | (UInt64(partials[unsafe_offset=at + 3]) << 32),
            ),
        )
        first = min(first, partials[unsafe_offset=at + 4])
    var at = group * STATE_WORDS
    result[unsafe_offset=at] = UInt32(total.low)
    result[unsafe_offset=at + 1] = UInt32(total.low >> 32)
    result[unsafe_offset=at + 2] = UInt32(total.high)
    result[unsafe_offset=at + 3] = UInt32(total.high >> 32)
    result[unsafe_offset=at + 4] = first


def execute(
    query: LazyFrame,
    device: Int,
    budget: Int,
    *,
    observation_only: Bool = False,
) raises -> Tuple[DataFrame, DataFrame]:
    var started = Int(perf_counter_ns())
    # Validate output names/types before allocation, including a SUM alias
    # colliding with the key. This is a schema operation, not a CPU aggregate.
    var schema = schema_frame(query, 1).clear()
    trace_path("metal.coded_int64_sum")
    ref node = query._nodes[1]
    ref frame = query._frames[query._nodes[0].offset]
    var rows = frame.height()
    var scope = Scope(device, budget)
    var result = schema.copy()
    var host_bytes = 0
    var prepare_ns = 0
    if rows:
        var key = frame.column(node.names[0])
        var dictionary = (
            key.chunks()[0]._data[StringColumn]._codes.value()[].dtype
        )
        var groups = len(dictionary.dictionary()[]) + 1
        var blocks = min(1024, (rows + 4095) // 4096)
        var partial_words = checked_mul(
            checked_mul(groups, blocks), STATE_WORDS
        )
        var output_words = checked_mul(groups, STATE_WORDS)
        # Rechunking may own one host copy of each selected column. Bound
        # those copies, device input, validity, partials, readback and output
        # before gathering any chunk or creating a device context.
        var input_bytes = checked_add(
            checked_mul(rows, 12), 2 * ((rows + 14) // 8)
        )
        var reserved = checked_add(
            checked_mul(input_bytes, 2),
            checked_add(
                checked_mul(partial_words, 4),
                checked_add(
                    checked_mul(groups, 256),
                    checked_mul(len(dictionary.dictionary()[].bytes), 3),
                ),
            ),
        )
        _ = scope.check(reserved)
        # String equality is label equality, even when two retained codes
        # name the same label. Canonicalize only dictionary entries; row
        # codes remain immutable and are remapped on the device.
        var lookup = Dict[String, UInt32]()
        var code_map = List[UInt32](capacity=max(1, groups - 1))
        for code in range(groups - 1):
            var label = String(dictionary.dictionary()[].get(code))
            if label not in lookup:
                lookup[label] = UInt32(code)
            code_map.append(lookup[label])
        if not len(code_map):
            code_map.append(0)
        var retained = key._dictionary_codes()
        var coded = retained.take().rechunk()
        var values = frame.column(node.exprs[0]._nodes[0].text).rechunk()
        ref kc = coded._data[Column[UInt32]]
        ref vc = values._data[Column[Int64]]
        var key_nulls = kc.null_count() > 0
        var value_nulls = vc.null_count() > 0
        var kb = (rows + kc._offset % 8 + 7) // 8 if key_nulls else 1
        var vb = (rows + vc._offset % 8 + 7) // 8 if value_nulls else 1
        var host = List[UInt32](length=output_words, fill=0)
        prepare_ns = Int(perf_counter_ns()) - started
        var ctx = scope.context()
        var keys_d = ctx.enqueue_create_buffer[DType.uint32](rows)
        var map_d = ctx.enqueue_create_buffer[DType.uint32](len(code_map))
        var values_d = ctx.enqueue_create_buffer[DType.int64](rows)
        var keys_b = ctx.enqueue_create_buffer[DType.uint8](kb)
        var values_b = ctx.enqueue_create_buffer[DType.uint8](vb)
        var partials = ctx.enqueue_create_buffer[DType.uint32](partial_words)
        var output = ctx.enqueue_create_buffer[DType.uint32](output_words)
        try:
            ctx.enqueue_copy(keys_d, kc._ptr())
            ctx.enqueue_copy(map_d, code_map.unsafe_ptr())
            ctx.enqueue_copy(values_d, vc._ptr())
            if key_nulls:
                ctx.enqueue_copy(
                    keys_b,
                    kc._bits[].unsafe_ptr().unsafe_offset(kc._offset // 8),
                )
            if value_nulls:
                ctx.enqueue_copy(
                    values_b,
                    vc._bits[].unsafe_ptr().unsafe_offset(vc._offset // 8),
                )
            scope.record_upload(
                rows * 12
                + len(code_map) * 4
                + (kb if key_nulls else 0)
                + (vb if value_nulls else 0)
            )
            ctx.enqueue_function[_partials](
                keys_d.unsafe_ptr(),
                map_d.unsafe_ptr(),
                values_d.unsafe_ptr(),
                keys_b.unsafe_ptr(),
                values_b.unsafe_ptr(),
                partials.unsafe_ptr(),
                UInt32(rows),
                UInt32(groups),
                UInt32(blocks),
                UInt32(kc._offset % 8),
                UInt32(vc._offset % 8),
                UInt32(key_nulls),
                UInt32(value_nulls),
                grid_dim=blocks,
                block_dim=256,
            )
            ctx.enqueue_function[_finish](
                partials.unsafe_ptr(),
                output.unsafe_ptr(),
                UInt32(groups),
                UInt32(blocks),
                grid_dim=(groups + 127) // 128,
                block_dim=128,
            )
            scope.record_launches(2)
            scope.read_control_into(output, host)
            scope.record_download(output_words * 4)
        except error:
            scope.synchronize()
            raise error^
        var order = List[Int]()
        for group in range(groups):
            var at = group * STATE_WORDS
            if host[at + 4] != UInt32.MAX:
                var low = UInt64(host[at]) | (UInt64(host[at + 1]) << 32)
                var high = UInt64(host[at + 2]) | (UInt64(host[at + 3]) << 32)
                var extension = UInt64.MAX if low >> 63 else UInt64(0)
                if high != extension:
                    raise data_fault("Int64 expression sum overflow")
                order.append(group)
        # Only O(dictionary cardinality) controls are ordered on the host.
        # Group membership and all sum arithmetic were computed on device.
        def earlier(a: Int, b: Int) {imm host} -> Bool:
            return host[a * STATE_WORDS + 4] < host[b * STATE_WORDS + 4]

        sort(order, earlier)
        var labels = List[String](capacity=len(order))
        var valid = List[Bool](capacity=len(order))
        var sums = List[Int64](capacity=len(order))
        for group in order:
            var at = group * STATE_WORDS
            var present = group < groups - 1
            labels.append(
                String(dictionary.dictionary()[].get(group)) if present else ""
            )
            valid.append(present)
            sums.append(
                (UInt64(host[at]) | (UInt64(host[at + 1]) << 32)).cast[
                    DType.int64
                ]()
            )
        # Publish only the dictionary entries that occurred. Taking source
        # rows would retain the original frame's large String/code buffers.
        var keys = Series(key.name(), StringColumn(labels^, valid^))
        keys._field_metadata = key._field_metadata
        var totals = Series(node.exprs[0]._name, Column[Int64](sums^))
        totals._field_metadata = schema.column(
            node.exprs[0]._name
        )._field_metadata
        host_bytes = (
            len(keys._data[StringColumn]._bytes[])
            + 8 * (len(order) + 1)
            + len(order) * 8
            + 2 * ((len(order) + 7) // 8)
        )
        result = DataFrame([keys^, totals^])
    var output_rows = result.height()
    if observation_only:
        result = schema.copy()
    var report = ExecutionReport()
    report.record(
        1,
        "GROUP_BY",
        "metal",
        rows,
        output_rows,
        algorithm="mojo_gpu_coded_int64_sum",
        wall_ns=Int(perf_counter_ns()) - started,
    )
    var fields = report.frame()._columns.copy()
    var traffic = scope.transfers()
    for item in List[Tuple[String, Int]](
        [
            ("kernel_launches", traffic[2]),
            ("synchronizations", traffic[3]),
            ("upload_bytes", traffic[0]),
            ("download_bytes", traffic[1]),
            ("peak_requested_bytes", scope.info()[1]),
            ("host_result_bytes", host_bytes),
            ("query_kernel_launches", traffic[2]),
            ("query_synchronizations", traffic[3]),
            ("query_upload_bytes", traffic[0]),
            ("query_download_bytes", traffic[1]),
            ("query_peak_requested_bytes", scope.info()[1]),
            ("query_host_result_bytes", host_bytes),
        ]
    ):
        fields.append(Series(item[0], Column[Int64]([Int64(item[1])])))
    fields.append(Series("runtime_implementation", Column[String](["mojo"])))
    fields.append(
        Series("prepare_ms", Column[Float64]([Float64(prepare_ns) / 1e6]))
    )
    fields.append(
        Series(
            "query_wall_ms",
            Column[Float64]([Float64(Int(perf_counter_ns()) - started) / 1e6]),
        )
    )
    return (result^, DataFrame(fields^))
