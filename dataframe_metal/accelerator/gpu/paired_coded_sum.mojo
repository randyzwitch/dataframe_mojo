"""Exact direct-address sums over two bounded retained String dimensions.

Region/status-style dimensions can have a small Cartesian domain even for
millions of rows. Bounded block-private global histograms avoid row sorting
without requiring their entire state to fit in threadgroup memory.
"""
from max.gpu import block_idx, thread_idx
from std.atomic import Atomic
from std.collections import Dict
from std.memory import Pointer
from std.time import perf_counter_ns
from dataframe import Column, DataFrame, DataType, Series
from dataframe.binding import bind
from dataframe.expr import COL, SUM
from dataframe.lazy import LazyFrame, SCAN_FRAME, AGG
from dataframe.string_column import StringColumn
from dataframe_metal.accelerator.metadata import schema_frame
from dataframe_metal.accelerator.faults import data_fault
from dataframe.execution_report import ExecutionReport
from dataframe.trace import trace_path
from .coded_group_sum import _valid, _finish
from .residency import Scope, checked_add, checked_mul

comptime STATE_WORDS = 5
comptime MAX_GROUPS = 16384


def _key_dtype(key: Series) raises -> Optional[DataType]:
    if key.dtype() != DataType.STRING:
        return None
    var dtype = Optional[DataType]()
    for part in key.chunks():
        ref strings = part._data[StringColumn]
        if not strings._codes or strings._codes.value()[].rows:
            return None
        var current = strings._codes.value()[].dtype
        if len(current.dictionary()[]) >= MAX_GROUPS:
            return None
        if dtype and dtype.value() != current:
            return None
        dtype = current
    return dtype^


def eligible(query: LazyFrame) raises -> Bool:
    if (
        len(query._nodes) != 2
        or query._nodes[0].kind != SCAN_FRAME
        or query._nodes[0].source_failure
        or query._nodes[1].kind != AGG
        or query._nodes[1].left != 0
        or len(query._nodes[1].names) != 2
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
    var left = _key_dtype(frame.column(node.names[0]))
    var right = _key_dtype(frame.column(node.names[1]))
    if not left or not right:
        return False
    return (len(left.value().dictionary()[]) + 1) * (
        len(right.value().dictionary()[]) + 1
    ) <= MAX_GROUPS


def _code_map(dtype: DataType) raises -> List[UInt32]:
    var lookup = Dict[String, UInt32]()
    var result = List[UInt32](capacity=max(1, len(dtype.dictionary()[])))
    for code in range(len(dtype.dictionary()[])):
        var label = String(dtype.dictionary()[].get(code))
        if label not in lookup:
            lookup[label] = UInt32(code)
        result.append(lookup[label])
    if not len(result):
        result.append(0)
    return result^


def _initialize(state: Pointer[UInt32, MutAnyOrigin], words: UInt32):
    var at = block_idx.x * 256 + thread_idx.x
    if at < Int(words):
        state[unsafe_offset=at] = UInt32.MAX if at % STATE_WORDS == 4 else 0


def _partials(
    codes: Pointer[UInt32, ImmutAnyOrigin],
    code_map: Pointer[UInt32, ImmutAnyOrigin],
    codes2: Pointer[UInt32, ImmutAnyOrigin],
    code_map2: Pointer[UInt32, ImmutAnyOrigin],
    values: Pointer[Int64, ImmutAnyOrigin],
    key_bits: Pointer[UInt8, ImmutAnyOrigin],
    key_bits2: Pointer[UInt8, ImmutAnyOrigin],
    value_bits: Pointer[UInt8, ImmutAnyOrigin],
    partials: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    left_groups: UInt32,
    right_groups: UInt32,
    blocks: UInt32,
    key_offset: UInt32,
    key_offset2: UInt32,
    value_offset: UInt32,
    key_nulls: UInt32,
    key_nulls2: UInt32,
    value_nulls: UInt32,
):
    var groups = Int(left_groups) * Int(right_groups)
    for row in range(
        block_idx.x * 256 + thread_idx.x, Int(rows), Int(blocks) * 256
    ):
        var left = Int(left_groups) - 1
        if not key_nulls or _valid(key_bits, row + Int(key_offset)):
            left = Int(code_map[unsafe_offset=Int(codes[unsafe_offset=row])])
        var right = Int(right_groups) - 1
        if not key_nulls2 or _valid(key_bits2, row + Int(key_offset2)):
            right = Int(code_map2[unsafe_offset=Int(codes2[unsafe_offset=row])])
        var group = left * Int(right_groups) + right
        var base = (block_idx.x * groups + group) * STATE_WORDS
        Atomic[UInt32].min(partials.unsafe_offset(base + 4), UInt32(row))
        if value_nulls and not _valid(value_bits, row + Int(value_offset)):
            continue
        var value = values[unsafe_offset=row].cast[DType.uint64]()
        var negative = value >> 63 != 0
        var carry = UInt32(0)
        comptime for word in range(4):
            var addend = (
                UInt32(value) if word
                == 0 else UInt32(value >> 32) if word
                == 1 else UInt32.MAX if negative else UInt32(0)
            )
            var next_carry = UInt32(0)
            if addend:
                var old = Atomic[UInt32].fetch_add(
                    partials.unsafe_offset(base + word), addend
                )
                next_carry = UInt32(old + addend < old)
            if carry:
                var old = Atomic[UInt32].fetch_add(
                    partials.unsafe_offset(base + word), carry
                )
                next_carry += UInt32(old + carry < old)
            carry = next_carry


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
    trace_path("metal.paired_coded_int64_sum")
    ref node = query._nodes[1]
    ref frame = query._frames[query._nodes[0].offset]
    var rows = frame.height()
    var scope = Scope(device, budget)
    var result = schema.copy()
    var host_bytes = 0
    var prepare_ns = 0
    if rows:
        var key = frame.column(node.names[0])
        var key2 = frame.column(node.names[1])
        var dictionary = (
            key.chunks()[0]._data[StringColumn]._codes.value()[].dtype
        )
        var dictionary2 = (
            key2.chunks()[0]._data[StringColumn]._codes.value()[].dtype
        )
        var left_groups = len(dictionary.dictionary()[]) + 1
        var right_groups = len(dictionary2.dictionary()[]) + 1
        var groups = left_groups * right_groups
        var blocks = min(64, (rows + 4095) // 4096)
        var partial_words = checked_mul(
            checked_mul(groups, blocks), STATE_WORDS
        )
        var output_words = checked_mul(groups, STATE_WORDS)
        # Rechunking may own one host copy of each selected column. Bound
        # those copies, device input, validity, partials, readback and output
        # before gathering any chunk or creating a device context.
        var input_bytes = checked_add(
            checked_mul(rows, 16), 3 * ((rows + 14) // 8)
        )
        var reserved = checked_add(
            checked_mul(input_bytes, 2),
            checked_add(
                checked_mul(partial_words, 4),
                checked_add(
                    checked_mul(groups, 256),
                    checked_mul(
                        checked_add(
                            checked_mul(
                                len(dictionary.dictionary()[].bytes),
                                right_groups,
                            ),
                            checked_mul(
                                len(dictionary2.dictionary()[].bytes),
                                left_groups,
                            ),
                        ),
                        3,
                    ),
                ),
            ),
        )
        _ = scope.check(reserved)
        # Canonicalize dictionary labels only; row codes stay immutable.
        var code_map = _code_map(dictionary)
        var code_map2 = _code_map(dictionary2)
        var retained = key._dictionary_codes()
        var coded = retained.take().rechunk()
        var retained2 = key2._dictionary_codes()
        var coded2 = retained2.take().rechunk()
        var values = frame.column(node.exprs[0]._nodes[0].text).rechunk()
        ref kc = coded._data[Column[UInt32]]
        ref kc2 = coded2._data[Column[UInt32]]
        ref vc = values._data[Column[Int64]]
        var key_nulls = kc.null_count() > 0
        var key_nulls2 = kc2.null_count() > 0
        var value_nulls = vc.null_count() > 0
        var kb = (rows + kc._offset % 8 + 7) // 8 if key_nulls else 1
        var kb2 = (rows + kc2._offset % 8 + 7) // 8 if key_nulls2 else 1
        var vb = (rows + vc._offset % 8 + 7) // 8 if value_nulls else 1
        var host = List[UInt32](length=output_words, fill=0)
        prepare_ns = Int(perf_counter_ns()) - started
        var ctx = scope.context()
        var keys_d = ctx.enqueue_create_buffer[DType.uint32](rows)
        var map_d = ctx.enqueue_create_buffer[DType.uint32](len(code_map))
        var keys_d2 = ctx.enqueue_create_buffer[DType.uint32](rows)
        var map_d2 = ctx.enqueue_create_buffer[DType.uint32](len(code_map2))
        var values_d = ctx.enqueue_create_buffer[DType.int64](rows)
        var keys_b = ctx.enqueue_create_buffer[DType.uint8](kb)
        var keys_b2 = ctx.enqueue_create_buffer[DType.uint8](kb2)
        var values_b = ctx.enqueue_create_buffer[DType.uint8](vb)
        var partials = ctx.enqueue_create_buffer[DType.uint32](partial_words)
        var output = ctx.enqueue_create_buffer[DType.uint32](output_words)
        try:
            ctx.enqueue_copy(keys_d, kc._ptr())
            ctx.enqueue_copy(map_d, code_map.unsafe_ptr())
            ctx.enqueue_copy(keys_d2, kc2._ptr())
            ctx.enqueue_copy(map_d2, code_map2.unsafe_ptr())
            ctx.enqueue_copy(values_d, vc._ptr())
            if key_nulls:
                ctx.enqueue_copy(
                    keys_b,
                    kc._bits[].unsafe_ptr().unsafe_offset(kc._offset // 8),
                )
            if key_nulls2:
                ctx.enqueue_copy(
                    keys_b2,
                    kc2._bits[].unsafe_ptr().unsafe_offset(kc2._offset // 8),
                )
            if value_nulls:
                ctx.enqueue_copy(
                    values_b,
                    vc._bits[].unsafe_ptr().unsafe_offset(vc._offset // 8),
                )
            scope.record_upload(
                rows * 16
                + (len(code_map) + len(code_map2)) * 4
                + (kb if key_nulls else 0)
                + (kb2 if key_nulls2 else 0)
                + (vb if value_nulls else 0)
            )
            ctx.enqueue_function[_initialize](
                partials.unsafe_ptr(),
                UInt32(partial_words),
                grid_dim=(partial_words + 255) // 256,
                block_dim=256,
            )
            ctx.enqueue_function[_partials](
                keys_d.unsafe_ptr(),
                map_d.unsafe_ptr(),
                keys_d2.unsafe_ptr(),
                map_d2.unsafe_ptr(),
                values_d.unsafe_ptr(),
                keys_b.unsafe_ptr(),
                keys_b2.unsafe_ptr(),
                values_b.unsafe_ptr(),
                partials.unsafe_ptr(),
                UInt32(rows),
                UInt32(left_groups),
                UInt32(right_groups),
                UInt32(blocks),
                UInt32(kc._offset % 8),
                UInt32(kc2._offset % 8),
                UInt32(vc._offset % 8),
                UInt32(key_nulls),
                UInt32(key_nulls2),
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
            scope.record_launches(3)
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
        var labels2 = List[String](capacity=len(order))
        var valid = List[Bool](capacity=len(order))
        var valid2 = List[Bool](capacity=len(order))
        var sums = List[Int64](capacity=len(order))
        for group in order:
            var at = group * STATE_WORDS
            var left = group // right_groups
            var right = group % right_groups
            var present = left < left_groups - 1
            var present2 = right < right_groups - 1
            labels.append(
                String(dictionary.dictionary()[].get(left)) if present else ""
            )
            labels2.append(
                String(
                    dictionary2.dictionary()[].get(right)
                ) if present2 else ""
            )
            valid.append(present)
            valid2.append(present2)
            sums.append(
                (UInt64(host[at]) | (UInt64(host[at + 1]) << 32)).cast[
                    DType.int64
                ]()
            )
        var keys = Series(key.name(), StringColumn(labels, valid))
        var keys2 = Series(key2.name(), StringColumn(labels2, valid2))
        keys._field_metadata = key._field_metadata
        keys2._field_metadata = key2._field_metadata
        var totals = Series(node.exprs[0]._name, Column[Int64](sums^))
        totals._field_metadata = schema.column(
            node.exprs[0]._name
        )._field_metadata
        host_bytes = (
            len(keys._data[StringColumn]._bytes[])
            + len(keys2._data[StringColumn]._bytes[])
            + 16 * (len(order) + 1)
            + len(order) * 8
            + 3 * ((len(order) + 7) // 8)
        )
        result = DataFrame([keys^, keys2^, totals^])
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
        algorithm="mojo_gpu_paired_coded_int64_sum",
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
