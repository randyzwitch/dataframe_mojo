"""Raw OVER scheduling metadata versus the actual pinned CPU dispatcher.

The uploaded equality layout is test-only. No public adapter, input-value
readback or CPU expression fallback is introduced by this diagnostic.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe import Column, DataType, Series, StringColumn
from dataframe.execution import _dense_partitions, _partition_ids
from dataframe.hashing import encode_rows
from dataframe.partition import _hash_column, _hash_bytes, low_cardinality, _mix
from dataframe.dtype import CategoricalDictionary
from dataframe.window_layout import (
    PartitionLayout,
    partition_layout,
    keyed_layout,
)
from dataframe.accelerator.rows import ROW_GROUP_HEADER_WINDOW, ROW_GROUP_KEY
from dataframe.accelerator.gpu.partition_schedule import (
    dense_kind,
    group_hashes,
    row_groups,
    fallback_sample,
    merge_groups,
    choose_runs,
    row_ordinals,
)
from tests.support.lifecycle_jobs import WorkerSetting


struct Expected(Movable):
    var phase: List[UInt32]
    var rank: List[UInt32]
    var offset: List[UInt32]
    var row_phase: List[UInt32]
    var ordinal: List[UInt32]
    var facts: List[UInt32]

    def __init__(out self, rows: Int, groups: Int):
        self.phase = List[UInt32](length=groups, fill=0)
        self.rank = List[UInt32](length=groups, fill=0)
        self.offset = List[UInt32](length=groups, fill=0)
        self.row_phase = List[UInt32](length=rows, fill=0)
        self.ordinal = List[UInt32](length=rows, fill=0)
        self.facts = List[UInt32](length=4, fill=0)


def _cpu_layout(keys: List[Series], lanes: Int) raises -> PartitionLayout:
    var dense = _dense_partitions(keys)
    if dense:
        var groups = dense.take()
        return partition_layout(groups.ids.copy(), groups.count(), lanes)
    return keyed_layout(keys, lanes)


def _expected(
    keys: List[Series],
    ids: List[Int],
    representatives: List[Int],
    configured: Int,
    allow_contiguous: Bool,
) raises -> Expected:
    var n = len(ids)
    var groups = len(representatives)
    var result = Expected(n, groups)
    var dense = _dense_partitions(keys)
    result.facts[0] = UInt32(Bool(dense))
    var workers = max(1, min(configured, n // 65536))
    result.facts[1] = UInt32(
        True if dense or workers <= 1 else low_cardinality(keys)
    )
    var lanes = min(configured, n // 2048)
    if lanes > 1 and allow_contiguous:
        var layout = _cpu_layout(keys, lanes)
        # This is the production _over_contiguous boundary loop, deliberately
        # independent from the GPU's accumulate-until-target implementation.
        var target = max(2048, (n + 4 * lanes - 1) // (4 * lanes))
        var cuts: List[Int] = [0]
        var since = 0
        for group in range(layout.count):
            if layout.starts[group] - since >= target:
                cuts.append(group)
                since = layout.starts[group]
        cuts.append(layout.count)
        if len(cuts) > 2:
            result.facts[2] = UInt32(len(cuts) - 1)
            result.facts[3] = 1
            for run in range(len(cuts) - 1):
                var lo = layout.starts[cuts[run]]
                for group in range(cuts[run], cuts[run + 1]):
                    var first = layout.starts[group]
                    var original = ids[layout.order[first]]
                    result.phase[original] = UInt32(run)
                    result.rank[original] = UInt32(group - cuts[run])
                    result.offset[original] = UInt32(first - lo)
                    for at in range(first, layout.starts[group + 1]):
                        var row = layout.order[at]
                        result.row_phase[row] = UInt32(run)
                        result.ordinal[row] = UInt32(at - lo)
            return result^
    # Use the actual configured CPU fallback, including its dense, whole-frame
    # and partitioned dispatch. Neither its hash order nor its sample decision
    # is reconstructed from the GPU code in this oracle.
    var fallback = _partition_ids(keys, n)
    result.facts[2] = 1
    for group in range(groups):
        result.rank[group] = UInt32(fallback.ids[representatives[group]])
    for row in range(n):
        result.ordinal[row] = UInt32(row)
    return result^


def _check(kind: Int, n: Int, configured: Int, allow_contiguous: Bool) raises:
    var setting = WorkerSetting(configured)
    var first = List[UInt64](capacity=n)
    var second = List[UInt64](capacity=n)
    var masks = List[Bool](capacity=n)
    var float_patterns: List[UInt64] = [
        0,
        0x8000000000000000,
        0x7FF8000000000001,
        0xFFF8000000000042,
        0x3FF0000000000000,
        0xBFF0000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
    ]
    var float32_patterns: List[UInt64] = [
        0,
        0x80000000,
        0x7FC00001,
        0xFFC00042,
        0x3F800000,
        0xBF800000,
        0x7F800000,
        0xFF800000,
    ]
    var dtype = Int64(3)
    for row in range(n):
        var value = UInt64(row % 2)
        if kind == 1:
            value = UInt64(row >= 192)
        elif kind == 2:
            value = float_patterns[row % len(float_patterns)]
            dtype = 11
        elif kind == 3:
            value = UInt64.MAX - UInt64(row % 31) * UInt64(1000000007)
            dtype = 10
        elif kind == 4:
            value = UInt64(row) * UInt64(1000000007)
            dtype = 10
        elif kind == 5:
            value = float32_patterns[row % len(float32_patterns)]
            dtype = 1
        elif kind == 6:
            value = UInt64(row % 4)
            dtype = 9
        elif kind == 7:
            value = UInt64(row % 2)
            dtype = 9
        elif kind == 8:
            value = UInt64(row % 5)
            dtype = 12
        first.append(value)
        second.append(Int64(row % 3 - 1).cast[DType.uint64]())
        masks.append(kind < 2 or row % 17 != 0)
    var keys = List[Series]()
    if dtype == 3:
        var values = List[Int64](capacity=n)
        for word in first:
            values.append(word.cast[DType.int64]())
        keys.append(Series("k", Column[Int64](values^, masks.copy())))
    elif dtype == 10:
        keys.append(Series("k", Column[UInt64](first.copy(), masks.copy())))
    elif dtype == 11:
        var values = List[Float64](capacity=n)
        for word in first:
            values.append(bitcast[DType.float64](word))
        keys.append(Series("k", Column[Float64](values^, masks.copy())))
    elif dtype == 9:
        if kind == 6:
            var labels: List[String] = ["a", "z", "é🙂", ""]
            var values = List[String](capacity=n)
            for row in range(n):
                values.append(labels[Int(first[row])])
            keys.append(Series("k", StringColumn(values^, masks.copy())))
        else:
            var dictionary = CategoricalDictionary()
            dictionary.append("same")
            dictionary.append("same")
            var values = List[UInt32](capacity=n)
            for word in first:
                values.append(UInt32(word))
            keys.append(
                Series("k", Column[UInt32](values^, masks.copy())).with_dtype(
                    DataType.categorical(dictionary^)
                )
            )
    elif dtype == 12:
        var values = List[Int128](capacity=n)
        for row in range(n):
            var high = UInt64((1 << 36) + row % 3)
            values.append(
                bitcast[DType.int128](
                    (UInt128(high) << UInt128(64)) | UInt128(first[row])
                )
            )
        keys.append(Series("k", Column[Int128](values^, masks.copy())))
    else:
        var values = List[Float32](capacity=n)
        for word in first:
            values.append(bitcast[DType.float32](UInt32(word)))
        keys.append(Series("k", Column[Float32](values^, masks.copy())))
    if kind == 3:
        var values = List[Int64](capacity=n)
        for word in second:
            values.append(word.cast[DType.int64]())
        keys.append(Series("k2", Column[Int64](values^)))
    var slots = len(keys)
    var words = List[UInt64](length=2 * slots * n, fill=0)
    var validity = List[UInt8](length=slots * n, fill=1)
    for row in range(n):
        words[row] = first[row]
        validity[row] = UInt8(masks[row])
        if dtype == 12:
            words[slots * n + row] = UInt64((1 << 36) + row % 3)
        if slots == 2:
            words[n + row] = second[row]
    var equality = encode_rows(keys, nulls_equal=True)
    var groups = equality.count()
    var layout = partition_layout(equality.ids.copy(), groups, 1)
    var index = List[UInt32](capacity=n)
    var starts = List[UInt32](capacity=groups + 1)
    for row in layout.order:
        index.append(UInt32(row))
    for at in layout.starts:
        starts.append(UInt32(at))
    var expected = _expected(
        keys,
        equality.ids,
        equality.representatives,
        configured,
        allow_contiguous,
    )
    # Independent literal geometry: alternating equal-sized dense groups
    # produce two whole-run jobs; the 192-row group plus one 8000-row group
    # cannot produce two jobs and must revert to original-row evaluation.
    if kind == 0:
        assert_equal(
            expected.facts[3], UInt32(configured > 1 and allow_contiguous)
        )
        if expected.facts[3]:
            assert_equal(expected.facts[2], UInt32(2))
            assert_equal(expected.phase[0], UInt32(0))
            assert_equal(expected.phase[1], UInt32(1))
            assert_equal(expected.ordinal[2], UInt32(1))
    if kind == 1:
        assert_equal(expected.facts[3], UInt32(0))
        assert_equal(expected.ordinal[n - 1], UInt32(n - 1))
    var code: List[Int64] = [ROW_GROUP_HEADER_WINDOW, Int64(slots), 0, 0]
    var types: List[Int64] = [0]
    var literals = List[UInt64](length=slots + 1, fill=0)
    for slot in range(slots):
        var descriptor: List[Int64] = [ROW_GROUP_KEY, Int64(slot), 0, 0]
        code.extend(descriptor^)
        types.append(dtype if slot == 0 else Int64(3))
    var key_values: List[UInt64] = [0]
    if kind == 6:
        key_values.clear()
        var labels: List[String] = ["a", "z", "é🙂", ""]
        for label in labels:
            var raw = label.as_bytes()
            var bytes = Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=raw.unsafe_ptr()
                .unsafe_mut_cast[False]()
                .unsafe_origin_cast[ImmutAnyOrigin](),
                length=len(raw),
            )
            key_values.append(_hash_bytes(bytes))
        literals[1] = UInt64(1) << UInt64(63)
    elif kind == 7:
        key_values = [0, 1]
        literals[1] = (UInt64(1) << UInt64(63)) | (UInt64(1) << UInt64(62))
    var group_by_row = List[UInt32](length=n, fill=0xFFFFFFFF)
    var phase = List[UInt32](length=groups, fill=0xFFFFFFFF)
    var rank = List[UInt32](length=groups, fill=0xFFFFFFFF)
    var offset = List[UInt32](length=groups, fill=0xFFFFFFFF)
    var row_phase = List[UInt32](length=n, fill=0xFFFFFFFF)
    var ordinal = List[UInt32](length=n, fill=0xFFFFFFFF)
    var facts = List[UInt32](length=4, fill=0xFFFFFFFF)
    var hashes = List[UInt64](length=groups, fill=0)
    var cpu_hashes = List[UInt64](length=n, fill=0)
    for slot in range(slots):
        _hash_column(keys[slot], 0, n, Int(cpu_hashes.unsafe_ptr()), slot == 0)
    var ctx = DeviceContext()
    var d_words = ctx.enqueue_create_buffer[DType.uint64](len(words))
    var d_validity = ctx.enqueue_create_buffer[DType.uint8](len(validity))
    var d_index = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_starts = ctx.enqueue_create_buffer[DType.uint32](groups + 1)
    var d_code = ctx.enqueue_create_buffer[DType.int64](len(code))
    var d_types = ctx.enqueue_create_buffer[DType.int64](len(types))
    var d_literals = ctx.enqueue_create_buffer[DType.uint64](len(literals))
    var d_key_values = ctx.enqueue_create_buffer[DType.uint64](len(key_values))
    var d_group_by_row = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_phase = ctx.enqueue_create_buffer[DType.uint32](groups)
    var d_rank = ctx.enqueue_create_buffer[DType.uint32](groups)
    var d_offset = ctx.enqueue_create_buffer[DType.uint32](groups)
    var d_row_phase = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_ordinal = ctx.enqueue_create_buffer[DType.uint32](n)
    var d_facts = ctx.enqueue_create_buffer[DType.uint32](4)
    var d_hashes = ctx.enqueue_create_buffer[DType.uint64](groups)
    var d_contiguous = ctx.enqueue_create_buffer[DType.uint32](groups)
    var d_fallback = ctx.enqueue_create_buffer[DType.uint32](groups)
    var d_contiguous_next = ctx.enqueue_create_buffer[DType.uint32](groups)
    var d_fallback_next = ctx.enqueue_create_buffer[DType.uint32](groups)
    var d_sample_hashes = ctx.enqueue_create_buffer[DType.uint64](8192)
    var d_sample_seen = ctx.enqueue_create_buffer[DType.uint8](8192)
    var d_sample_counts = ctx.enqueue_create_buffer[DType.uint32](256)
    var blocks = max(1, (n + 255) // 256)
    var group_blocks = max(1, (groups + 255) // 256)
    try:
        ctx.enqueue_copy(d_words, words.unsafe_ptr())
        ctx.enqueue_copy(d_validity, validity.unsafe_ptr())
        ctx.enqueue_copy(d_index, index.unsafe_ptr())
        ctx.enqueue_copy(d_starts, starts.unsafe_ptr())
        ctx.enqueue_copy(d_code, code.unsafe_ptr())
        ctx.enqueue_copy(d_types, types.unsafe_ptr())
        ctx.enqueue_copy(d_literals, literals.unsafe_ptr())
        ctx.enqueue_copy(d_key_values, key_values.unsafe_ptr())
        ctx.enqueue_function[dense_kind](
            d_words.unsafe_ptr(),
            d_validity.unsafe_ptr(),
            d_index.unsafe_ptr(),
            d_starts.unsafe_ptr(),
            d_code.unsafe_ptr(),
            d_literals.unsafe_ptr(),
            d_types.unsafe_ptr(),
            d_key_values.unsafe_ptr(),
            d_facts.unsafe_ptr(),
            UInt32(0),
            UInt32(groups),
            UInt32(n),
            UInt32(n),
            UInt32(slots),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[group_hashes](
            d_words.unsafe_ptr(),
            d_validity.unsafe_ptr(),
            d_index.unsafe_ptr(),
            d_starts.unsafe_ptr(),
            d_code.unsafe_ptr(),
            d_literals.unsafe_ptr(),
            d_types.unsafe_ptr(),
            d_key_values.unsafe_ptr(),
            d_hashes.unsafe_ptr(),
            d_contiguous.unsafe_ptr(),
            d_fallback.unsafe_ptr(),
            UInt32(0),
            UInt32(groups),
            UInt32(n),
            UInt32(slots),
            grid_dim=group_blocks,
            block_dim=256,
        )
        ctx.enqueue_function[row_groups](
            d_index.unsafe_ptr(),
            d_starts.unsafe_ptr(),
            d_group_by_row.unsafe_ptr(),
            UInt32(groups),
            UInt32(n),
            grid_dim=blocks,
            block_dim=256,
        )
        ctx.enqueue_function[fallback_sample](
            d_hashes.unsafe_ptr(),
            d_group_by_row.unsafe_ptr(),
            d_sample_hashes.unsafe_ptr(),
            d_sample_seen.unsafe_ptr(),
            d_sample_counts.unsafe_ptr(),
            d_facts.unsafe_ptr(),
            UInt32(n),
            UInt32(configured),
            grid_dim=1,
            block_dim=1,
        )
        var width = 1
        while width < groups:
            ctx.enqueue_function[merge_groups](
                d_hashes.unsafe_ptr(),
                d_index.unsafe_ptr(),
                d_starts.unsafe_ptr(),
                d_facts.unsafe_ptr(),
                d_contiguous.unsafe_ptr(),
                d_contiguous_next.unsafe_ptr(),
                UInt32(groups),
                UInt32(n),
                UInt32(configured),
                UInt32(0),
                UInt32(width),
                grid_dim=group_blocks,
                block_dim=256,
            )
            ctx.enqueue_function[merge_groups](
                d_hashes.unsafe_ptr(),
                d_index.unsafe_ptr(),
                d_starts.unsafe_ptr(),
                d_facts.unsafe_ptr(),
                d_fallback.unsafe_ptr(),
                d_fallback_next.unsafe_ptr(),
                UInt32(groups),
                UInt32(n),
                UInt32(configured),
                UInt32(1),
                UInt32(width),
                grid_dim=group_blocks,
                block_dim=256,
            )
            var previous = d_contiguous^
            d_contiguous = d_contiguous_next^
            d_contiguous_next = previous^
            var previous_fallback = d_fallback^
            d_fallback = d_fallback_next^
            d_fallback_next = previous_fallback^
            width *= 2
        ctx.enqueue_function[choose_runs](
            d_starts.unsafe_ptr(),
            d_contiguous.unsafe_ptr(),
            d_fallback.unsafe_ptr(),
            d_facts.unsafe_ptr(),
            d_phase.unsafe_ptr(),
            d_rank.unsafe_ptr(),
            d_offset.unsafe_ptr(),
            UInt32(groups),
            UInt32(n),
            UInt32(configured),
            UInt32(allow_contiguous),
            UInt32(0),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[row_ordinals](
            d_index.unsafe_ptr(),
            d_starts.unsafe_ptr(),
            d_group_by_row.unsafe_ptr(),
            d_phase.unsafe_ptr(),
            d_offset.unsafe_ptr(),
            d_facts.unsafe_ptr(),
            d_row_phase.unsafe_ptr(),
            d_ordinal.unsafe_ptr(),
            UInt32(n),
            grid_dim=blocks,
            block_dim=256,
        )
        ctx.enqueue_copy(group_by_row.unsafe_ptr(), d_group_by_row)
        ctx.enqueue_copy(phase.unsafe_ptr(), d_phase)
        ctx.enqueue_copy(rank.unsafe_ptr(), d_rank)
        ctx.enqueue_copy(offset.unsafe_ptr(), d_offset)
        ctx.enqueue_copy(row_phase.unsafe_ptr(), d_row_phase)
        ctx.enqueue_copy(ordinal.unsafe_ptr(), d_ordinal)
        ctx.enqueue_copy(facts.unsafe_ptr(), d_facts)
        ctx.enqueue_copy(hashes.unsafe_ptr(), d_hashes)
        ctx.synchronize()
    except error:
        ctx.synchronize()
        _ = d_words^
        _ = d_validity^
        _ = d_index^
        _ = d_starts^
        _ = d_code^
        _ = d_types^
        _ = d_literals^
        _ = d_key_values^
        _ = d_group_by_row^
        _ = d_phase^
        _ = d_rank^
        _ = d_offset^
        _ = d_row_phase^
        _ = d_ordinal^
        _ = d_facts^
        _ = d_hashes^
        _ = d_contiguous^
        _ = d_fallback^
        _ = d_contiguous_next^
        _ = d_fallback_next^
        _ = d_sample_hashes^
        _ = d_sample_seen^
        _ = d_sample_counts^
        _ = words^
        _ = validity^
        _ = index^
        _ = starts^
        _ = code^
        _ = types^
        _ = literals^
        _ = key_values^
        _ = group_by_row^
        _ = phase^
        _ = rank^
        _ = offset^
        _ = row_phase^
        _ = ordinal^
        _ = facts^
        _ = hashes^
        raise error^
    for at in range(4):
        if facts[at] != expected.facts[at]:
            print(
                "SCHEDULE FACT MISMATCH",
                kind,
                n,
                configured,
                allow_contiguous,
                at,
                facts[at],
                expected.facts[at],
            )
        assert_equal(facts[at], expected.facts[at])
    for group in range(groups):
        assert_equal(phase[group], expected.phase[group])
        assert_equal(rank[group], expected.rank[group])
        assert_equal(offset[group], expected.offset[group])
        assert_equal(hashes[group], cpu_hashes[equality.representatives[group]])
    for row in range(n):
        assert_equal(group_by_row[row], UInt32(equality.ids[row]))
        assert_equal(row_phase[row], expected.row_phase[row])
        assert_equal(ordinal[row], expected.ordinal[row])
    _ = d_words^
    _ = d_validity^
    _ = d_index^
    _ = d_starts^
    _ = d_code^
    _ = d_types^
    _ = d_literals^
    _ = d_key_values^
    _ = d_group_by_row^
    _ = d_phase^
    _ = d_rank^
    _ = d_offset^
    _ = d_row_phase^
    _ = d_ordinal^
    _ = d_facts^
    _ = d_hashes^
    _ = d_contiguous^
    _ = d_fallback^
    _ = d_contiguous_next^
    _ = d_fallback_next^
    _ = d_sample_hashes^
    _ = d_sample_seen^
    _ = d_sample_counts^
    _ = words^
    _ = validity^
    _ = index^
    _ = starts^
    _ = code^
    _ = types^
    _ = literals^
    _ = key_values^
    _ = group_by_row^
    _ = phase^
    _ = rank^
    _ = offset^
    _ = row_phase^
    _ = ordinal^
    _ = facts^
    _ = hashes^
    _ = setting^
    print("PASS OVER schedule", kind, n, configured, allow_contiguous, groups)


def run_probe() raises:
    assert_equal(_mix(UInt64(0)), UInt64(0))
    assert_equal(_mix(UInt64(4)), UInt64(0xB7A4712C74562914))
    for configured in [1, 2, 4, 8]:
        for allow_contiguous in [False, True]:
            for kind in range(9):
                _check(
                    kind,
                    131073 if kind == 4 else 8192,
                    configured,
                    allow_contiguous,
                )
    print("PASS72 exact CPU OVER scheduling cases")
