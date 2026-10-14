"""Exact device OVER schedules against CPU dense/hash partition layouts."""

from max.gpu.host import DeviceContext
from std.memory import bitcast
from std.testing import assert_equal
from dataframe import Column, Series, StringColumn
from dataframe.hashing import _encode_dense, RowKeys
from dataframe.partition import _hash_bytes
from dataframe.window_layout import (
    PartitionLayout,
    partition_layout,
    keyed_layout,
)
from dataframe.accelerator.rows import ROW_GROUP_HEADER_WINDOW, ROW_GROUP_KEY
from dataframe.accelerator.gpu.grouping import (
    group_window_prepare,
    group_window_merge,
    group_window_schedule,
)
from tests.execution_modes.test_mojo_gpu_partition_schedule import _skew_keys


def _check(
    keys: List[Series],
    raw_words: List[UInt64],
    raw_masks: List[UInt8],
    types: List[Int64],
    descriptors: List[UInt64],
    var key_values: List[UInt64],
    configured: Int,
) raises:
    var rows = len(keys[0])
    var count = len(keys)
    var slots = count + 2
    var cells = rows * slots
    var lanes = min(configured, rows // 2048)
    var dense = Optional[RowKeys]()
    if count == 1:
        dense = _encode_dense(keys[0], True)
    var layout: PartitionLayout
    if dense:
        var encoded = dense.take()
        layout = partition_layout(encoded.ids.copy(), encoded.count(), lanes)
    else:
        layout = keyed_layout(keys, lanes)
    var groups = layout.count
    var words = List[UInt64](length=2 * cells, fill=0)
    var masks = List[UInt8](length=cells, fill=0)
    for key in range(count):
        for row in range(rows):
            words[key * rows + row] = raw_words[key * rows + row]
            words[cells + key * rows + row] = raw_words[
                rows * count + key * rows + row
            ]
            masks[key * rows + row] = raw_masks[key * rows + row]
    # Reverse the CPU groups deliberately: the kernel must reconstruct CPU
    # ordering from original row representatives, not input group IDs.
    var index = List[UInt32](capacity=rows)
    var starts = List[UInt32](capacity=groups)
    for group in range(groups):
        starts.append(UInt32(len(index)))
        var cpu_group = groups - group - 1
        for at in range(layout.starts[cpu_group], layout.starts[cpu_group + 1]):
            index.append(UInt32(layout.order[at]))
    var code: List[Int64] = [
        ROW_GROUP_HEADER_WINDOW,
        Int64(count),
        Int64(count + 1),
        0,
    ]
    var literals: List[UInt64] = [UInt64(configured) | (UInt64(1) << 32)]
    var node_types: List[Int64] = [3]
    for key in range(count):
        code.extend([ROW_GROUP_KEY, Int64(key), 0, 0])
        literals.append(descriptors[key])
        node_types.append(types[key])
    var expected = List[UInt64](length=2 * groups, fill=0)
    var runs = 0
    if lanes > 1 and groups > 1:
        var target = max(2048, (rows + 4 * lanes - 1) // (4 * lanes))
        var cuts: List[Int] = [0]
        var since = 0
        for group in range(groups):
            if layout.starts[group] - since >= target:
                cuts.append(group)
                since = layout.starts[group]
        cuts.append(groups)
        runs = len(cuts) - 1
        for run in range(runs):
            var first = cuts[run]
            var last = cuts[run + 1]
            var size = layout.starts[last] - layout.starts[first]
            var workers = max(
                1,
                min(
                    configured,
                    min(size // 65536, size // max(1, 4 * (last - first))),
                ),
            )
            var step = ((size + workers - 1) // workers + 8191) // 8192 * 8192
            for cpu_group in range(first, last):
                var group = groups - cpu_group - 1
                expected[group] = UInt64(workers) | (UInt64(step) << 32)
                expected[groups + group] = UInt64(
                    layout.starts[cpu_group] - layout.starts[first]
                )
    var metadata: List[UInt32] = [UInt32(groups), 0, 0, 0]
    var result = List[UInt64](length=2 * groups, fill=0)
    var ctx = DeviceContext()
    var dw = ctx.enqueue_create_buffer[DType.uint64](2 * cells)
    var dv = ctx.enqueue_create_buffer[DType.uint8](cells)
    var di = ctx.enqueue_create_buffer[DType.uint32](rows)
    var ds = ctx.enqueue_create_buffer[DType.uint32](groups)
    var da = ctx.enqueue_create_buffer[DType.uint32](groups)
    var db = ctx.enqueue_create_buffer[DType.uint32](groups)
    var dc = ctx.enqueue_create_buffer[DType.int64](len(code))
    var dl = ctx.enqueue_create_buffer[DType.uint64](len(literals))
    var dt = ctx.enqueue_create_buffer[DType.int64](len(node_types))
    var dk = ctx.enqueue_create_buffer[DType.uint64](len(key_values))
    var dm = ctx.enqueue_create_buffer[DType.uint32](4)
    ctx.enqueue_copy(dw, words.unsafe_ptr())
    ctx.enqueue_copy(dv, masks.unsafe_ptr())
    ctx.enqueue_copy(di, index.unsafe_ptr())
    ctx.enqueue_copy(ds, starts.unsafe_ptr())
    ctx.enqueue_copy(dc, code.unsafe_ptr())
    ctx.enqueue_copy(dl, literals.unsafe_ptr())
    ctx.enqueue_copy(dt, node_types.unsafe_ptr())
    ctx.enqueue_copy(dk, key_values.unsafe_ptr())
    ctx.enqueue_copy(dm, metadata.unsafe_ptr())
    ctx.enqueue_function[group_window_prepare](
        dw.unsafe_ptr(),
        dv.unsafe_ptr(),
        di.unsafe_ptr(),
        ds.unsafe_ptr(),
        da.unsafe_ptr(),
        dc.unsafe_ptr(),
        dl.unsafe_ptr(),
        dt.unsafe_ptr(),
        dk.unsafe_ptr(),
        UInt32(0),
        UInt32(groups),
        UInt32(rows),
        UInt32(rows),
        UInt32(slots),
        grid_dim=1,
        block_dim=1,
    )
    var width = 1
    while width < groups:
        ctx.enqueue_function[group_window_merge](
            dw.unsafe_ptr(),
            di.unsafe_ptr(),
            ds.unsafe_ptr(),
            da.unsafe_ptr(),
            db.unsafe_ptr(),
            UInt32(count + 1),
            UInt32(groups),
            UInt32(rows),
            UInt32(width),
            grid_dim=(groups + 255) // 256,
            block_dim=256,
        )
        var previous = da^
        da = db^
        db = previous^
        width *= 2
    ctx.enqueue_function[group_window_schedule](
        dw.unsafe_ptr(),
        ds.unsafe_ptr(),
        da.unsafe_ptr(),
        dm.unsafe_ptr(),
        UInt32(count + 1),
        UInt32(groups),
        UInt32(rows),
        UInt32(rows),
        UInt32(configured),
        grid_dim=1,
        block_dim=1,
    )
    var packed = dw.create_sub_buffer[DType.uint64](count * rows, groups)
    var offsets = dw.create_sub_buffer[DType.uint64]((count + 1) * rows, groups)
    ctx.enqueue_copy(result.unsafe_ptr(), packed)
    ctx.enqueue_copy(result.unsafe_ptr().unsafe_offset(groups), offsets)
    ctx.enqueue_copy(metadata.unsafe_ptr(), dm)
    ctx.synchronize()
    assert_equal(metadata[2], UInt32(runs > 1))
    assert_equal(metadata[3], UInt32(runs))
    for group in range(groups):
        assert_equal(result[group], expected[group])
        assert_equal(result[groups + group], expected[groups + group])
    # Explicit last uses preserve host staging owners through asynchronous
    # uploads; the SDK's raw-pointer enqueue_copy erases their origins.
    _ = words^
    _ = masks^
    _ = index^
    _ = starts^
    _ = code^
    _ = literals^
    _ = node_types^
    _ = key_values^
    print("PASS schedule", rows, groups, configured, runs)


def _integers() raises:
    for hashed in [False, True]:
        var values = _skew_keys(hashed)
        var words = List[UInt64](length=len(values) * 2, fill=0)
        for row in range(len(values)):
            words[row] = values[row]
        _check(
            [Series("k", Column[UInt64](values^))],
            words^,
            List[UInt8](length=262147, fill=1),
            [10],
            [0],
            [0],
            4,
        )
    for groups in [1, 3, 257, 8193]:
        var values = List[Int16](capacity=131073)
        var words = List[UInt64](length=262146, fill=0)
        var validity = List[Bool](capacity=131073)
        var masks = List[UInt8](capacity=131073)
        for row in range(131073):
            var value = Int16(row % groups - groups // 2)
            values.append(value)
            words[row] = bitcast[DType.uint64](Int64(value))
            var valid = row % 13 != 0
            validity.append(valid)
            masks.append(UInt8(valid))
        _check(
            [Series("k", Column[Int16](values^, validity^))],
            words^,
            masks^,
            [6],
            [0],
            [0],
            8,
        )


def _mixed() raises:
    var rows = 131073
    var xs = List[Float64](capacity=rows)
    var wide = List[Int128](capacity=rows)
    var flags = List[Bool](capacity=rows)
    var valid = List[Bool](capacity=rows)
    var words = List[UInt64](length=6 * rows, fill=0)
    var masks = List[UInt8](length=3 * rows, fill=0)
    var encodings: List[UInt64] = [
        0,
        0x8000000000000000,
        0x7FF8000000000001,
        0xFFF8000000000002,
        0x3FF0000000000000,
    ]
    for row in range(rows):
        var x = encodings[row % 5]
        var w = (Int128(row % 17 - 8) << 80) + Int128(row % 3)
        var b = row % 2 != 0
        var ok = row % 11 != 0
        xs.append(bitcast[DType.float64](x))
        wide.append(w)
        flags.append(b)
        valid.append(ok)
        words[row] = x
        words[rows + row] = bitcast[DType.uint128](w).cast[DType.uint64]()
        words[4 * rows + row] = UInt64(bitcast[DType.uint128](w) >> 64)
        words[2 * rows + row] = UInt64(b)
        masks[row] = UInt8(ok)
        masks[rows + row] = 1
        masks[2 * rows + row] = UInt8(ok)
    _check(
        [
            Series("x", Column[Float64](xs^, valid.copy())),
            Series("w", Column[Int128](wide^)),
            Series("b", Column[Bool](flags^, valid^)),
        ],
        words^,
        masks^,
        [11, 12, 4],
        [0, 0, 0],
        [0],
        8,
    )


def _dictionary() raises:
    # Both table modes use deliberately permuted staged codes. Original
    # categorical physical codes must preserve dense first-occurrence order;
    # original strings must instead preserve their logical hash buckets.
    var names: List[String] = ["beta", "a", "a very long key", "β"]
    var original: List[UInt64] = [3, 0, 2, 1]
    var hashes = List[UInt64](capacity=4)
    for name in names:
        var bytes = name.as_bytes()
        hashes.append(
            _hash_bytes(
                Span[UInt8, ImmutAnyOrigin](
                    unsafe_ptr=bytes.unsafe_ptr()
                    .unsafe_mut_cast[False]()
                    .unsafe_origin_cast[ImmutAnyOrigin](),
                    length=len(bytes),
                )
            )
        )
    for categorical in [False, True]:
        var rows = 131073
        var strings = List[String](capacity=rows)
        var codes = List[UInt32](capacity=rows)
        var words = List[UInt64](length=2 * rows, fill=0)
        var masks = List[UInt8](length=rows, fill=1)
        for row in range(rows):
            var code = row % 4
            strings.append(names[code])
            codes.append(UInt32(original[code]))
            words[row] = UInt64(code)
        var keys: List[Series] = [
            Series("k", Column[UInt32](codes^)) if categorical else (
                Series("k", StringColumn(strings^))
            )
        ]
        _check(
            keys^,
            words^,
            masks^,
            [9],
            [
                (UInt64(1) << 63)
                | ((UInt64(1) << 62) if categorical else UInt64(0))
            ],
            original.copy() if categorical else hashes.copy(),
            8,
        )


def main() raises:
    _integers()
    _mixed()
    _dictionary()
    print("PASS exact CPU partition schedule probe")
