"""Stable GPU row ordering over shared fixed-width dataframe words.

Sort descriptors are the shared RowPlan code and aligned physical type array.
Each merge pass ranks every row against the opposite sorted run. Source row
indices break ties, preserving NaN payloads, signed zeros and nullable values.
"""
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives.block import prefix_sum
from std.memory import Pointer
from . import floating, decimal
from .rows import unsigned_type


@always_inline
def _word_less(a: UInt64, b: UInt64, dtype: Int64) -> Bool:
    if dtype == 11:
        return floating.less(a, b)
    if dtype == 1:
        # Compare encodings without native floating arithmetic: Float32
        # subnormals must not be flushed to zero by the device.
        var x = UInt32(a)
        var y = UInt32(b)
        var ax = x & UInt32(0x7FFFFFFF)
        var ay = y & UInt32(0x7FFFFFFF)
        if ax > UInt32(0x7F800000) or ay > UInt32(0x7F800000):
            return False
        if ax == 0 and ay == 0:
            return False
        var sx = x >> 31 != 0
        var sy = y >> 31 != 0
        return sx if sx != sy else (x > y if sx else x < y)
    if unsigned_type(dtype) or dtype == 4:
        return a < b
    return a.cast[DType.int64]() < b.cast[DType.int64]()


@always_inline
def _nan(word: UInt64, dtype: Int64) -> Bool:
    if dtype == 11:
        return floating.is_nan(word)
    if dtype == 1:
        return (UInt32(word) & UInt32(0x7FFFFFFF)) > UInt32(0x7F800000)
    return False


@always_inline
def order_less(
    a: UInt32,
    b: UInt32,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
) -> Bool:
    for key in range(Int(keys)):
        var node = Int(start) + key
        var base = Int(code[unsafe_offset=4 * node + 1]) * Int(stride)
        var ai = base + Int(a)
        var bi = base + Int(b)
        var av = validity[unsafe_offset=ai] != 0
        var bv = validity[unsafe_offset=bi] != 0
        var nulls_last = code[unsafe_offset=4 * node + 3] != 0
        if av != bv:
            return av if nulls_last else not av
        if not av:
            continue
        var aw = words[unsafe_offset=ai]
        var bw = words[unsafe_offset=bi]
        var dtype = types[unsafe_offset=node]
        if dtype == 12:
            var high_base = Int(stride) * Int(slots)
            var ah = words[unsafe_offset=high_base + ai]
            var bh = words[unsafe_offset=high_base + bi]
            if ah != bh or aw != bw:
                var less = decimal.signed_less(
                    decimal.Wide(aw, ah), decimal.Wide(bw, bh)
                )
                return (
                    not less if code[unsafe_offset=4 * node + 2] != 0 else less
                )
            continue
        var an = _nan(aw, dtype)
        var bn = _nan(bw, dtype)
        # Both directions place valid NaNs after non-NaNs. NaN payload and
        # sign do not participate in ordering; later keys can still break ties.
        if an != bn:
            return not an
        var descending = code[unsafe_offset=4 * node + 2] != 0
        var less = _word_less(bw, aw, dtype) if descending else _word_less(
            aw, bw, dtype
        )
        if less:
            return True
        var greater = _word_less(aw, bw, dtype) if descending else _word_less(
            bw, aw, dtype
        )
        if greater:
            return False
    return a < b


def order_init(index: Pointer[UInt32, MutAnyOrigin], rows: UInt32):
    var row = global_idx.x
    if row < Int(rows):
        index[unsafe_offset=row] = UInt32(row)


def order_merge(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    target: Pointer[UInt32, MutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    rows: UInt32,
    stride: UInt32,
    run_width: UInt32,
    slots: UInt32 = 0,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    # Int arithmetic avoids overflow in the final pass at Int32.MAX rows.
    var width = Int(run_width)
    var base = (row // (2 * width)) * (2 * width)
    var middle = min(base + width, Int(rows))
    var end = min(base + 2 * width, Int(rows))
    var left = row < middle
    var first = middle if left else base
    var lo = first
    var hi = end if left else middle
    var own = index[unsafe_offset=row]
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if order_less(
            index[unsafe_offset=mid],
            own,
            words,
            validity,
            code,
            types,
            start,
            keys,
            stride,
            slots,
        ):
            lo = mid + 1
        else:
            hi = mid
    var own_offset = row - (base if left else middle)
    target[unsafe_offset=base + own_offset + lo - first] = own


def order_key_errors(
    errors: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    failures: Pointer[UInt32, MutAnyOrigin],
    start: UInt32,
    keys: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    var row = global_idx.x
    var failed = UInt32(0)
    if row < Int(rows):
        for key in range(Int(keys)):
            var node = Int(start) + key
            var base = Int(code[unsafe_offset=4 * node + 1]) * Int(stride)
            failed |= errors[unsafe_offset=base + row]
    var count = prefix_sum[block_size=256](UInt32(failed != 0))
    if thread_idx.x == 255:
        failures[unsafe_offset=block_idx.x] = count


def order_gather(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_validity: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    gathers: Pointer[Int64, ImmutAnyOrigin],
    count: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var source = Int(index[unsafe_offset=row])
    for column in range(Int(count)):
        var base = Int(gathers[unsafe_offset=column]) * Int(stride)
        output_words[unsafe_offset=base + row] = words[
            unsafe_offset=base + source
        ]
        if slots:
            var high_base = Int(stride) * Int(slots)
            output_words[unsafe_offset=high_base + base + row] = words[
                unsafe_offset=high_base + base + source
            ]
        output_validity[unsafe_offset=base + row] = validity[
            unsafe_offset=base + source
        ]
        output_errors[unsafe_offset=base + row] = errors[
            unsafe_offset=base + source
        ]
