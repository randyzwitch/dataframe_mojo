"""Stable partitioned ranks, assigned directly back to source row positions."""
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives.block import prefix_sum
from std.memory import Pointer
from .ordering import _word_less, _nan
from . import floating


@always_inline
def key_equal(
    a: UInt32,
    b: UInt32,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    node: UInt32,
    stride: UInt32,
) -> Bool:
    var base = Int(code[unsafe_offset=4 * Int(node) + 1]) * Int(stride)
    var av = validity[unsafe_offset=base + Int(a)] != 0
    var bv = validity[unsafe_offset=base + Int(b)] != 0
    if av != bv:
        return False
    if not av:
        return True
    var aw = words[unsafe_offset=base + Int(a)]
    var bw = words[unsafe_offset=base + Int(b)]
    var dtype = types[unsafe_offset=Int(node)]
    if dtype == 1 or dtype == 11:
        if _nan(aw, dtype) or _nan(bw, dtype):
            return _nan(aw, dtype) and _nan(bw, dtype)
        var mask = UInt64(0x7FFFFFFF) if dtype == 1 else UInt64(
            0x7FFFFFFFFFFFFFFF
        )
        if aw & mask == 0 and bw & mask == 0:
            return True
    return aw == bw


@always_inline
def partition_equal(
    a: UInt32,
    b: UInt32,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    stride: UInt32,
) -> Bool:
    for key in range(1, Int(nodes)):
        if not key_equal(
            a, b, words, validity, code, types, start + UInt32(key), stride
        ):
            return False
    return True


@always_inline
def _rank_equal(
    a: UInt32,
    b: UInt32,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    stride: UInt32,
) -> Bool:
    return partition_equal(
        a, b, words, validity, code, types, start, nodes, stride
    ) and key_equal(a, b, words, validity, code, types, start, stride)


@always_inline
def _rank_less(
    a: UInt32,
    b: UInt32,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    stride: UInt32,
) -> Bool:
    # Rank metadata stores the input first; partitions must sort before it.
    for key in range(Int(nodes)):
        var last = key + 1 == Int(nodes)
        var node = Int(start) if last else Int(start) + key + 1
        var base = Int(code[unsafe_offset=4 * node + 1]) * Int(stride)
        var av = validity[unsafe_offset=base + Int(a)] != 0
        var bv = validity[unsafe_offset=base + Int(b)] != 0
        if av != bv:
            return av
        if not av:
            continue
        var aw = words[unsafe_offset=base + Int(a)]
        var bw = words[unsafe_offset=base + Int(b)]
        var dtype = types[unsafe_offset=node]
        var an = _nan(aw, dtype)
        var bn = _nan(bw, dtype)
        var descending = last and code[unsafe_offset=4 * node + 2] != 0
        if an != bn:
            # Rank reverses the NaN category along with numbers; ordinary
            # dataframe sorting deliberately keeps NaNs last both ways.
            return an if descending else not an
        if _word_less(bw, aw, dtype) if descending else _word_less(
            aw, bw, dtype
        ):
            return True
        if _word_less(aw, bw, dtype) if descending else _word_less(
            bw, aw, dtype
        ):
            return False
    return a < b


def rank_order_merge(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    target: Pointer[UInt32, MutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    rows: UInt32,
    stride: UInt32,
    run_width: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
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
        if _rank_less(
            index[unsafe_offset=mid],
            own,
            words,
            validity,
            code,
            types,
            start,
            nodes,
            stride,
        ):
            lo = mid + 1
        else:
            hi = mid
    var own_offset = row - (base if left else middle)
    target[unsafe_offset=base + own_offset + lo - first] = own


def rank_prepare(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    prefix: Pointer[UInt32, MutAnyOrigin],
    blocks: Pointer[UInt32, MutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    var position = global_idx.x
    var head = UInt32(0)
    var input = Int(code[unsafe_offset=4 * Int(start) + 1]) * Int(stride)
    if position < Int(rows):
        var own = index[unsafe_offset=position]
        if validity[unsafe_offset=input + Int(own)] != 0:
            head = UInt32(
                position == 0
                or not _rank_equal(
                    index[unsafe_offset=position - 1],
                    own,
                    words,
                    validity,
                    code,
                    types,
                    start,
                    nodes,
                    stride,
                )
            )
    var count = prefix_sum[block_size=256](head)
    if position < Int(rows):
        prefix[unsafe_offset=position] = count
    if thread_idx.x == 255:
        blocks[unsafe_offset=block_idx.x] = count


def rank_offsets(blocks: Pointer[UInt32, MutAnyOrigin], count: UInt32):
    var total = UInt32(0)
    for block in range(Int(count)):
        var value = blocks[unsafe_offset=block]
        blocks[unsafe_offset=block] = total
        total += value


def rank_assign(
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    prefix: Pointer[UInt32, ImmutAnyOrigin],
    blocks: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    output: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    var position = global_idx.x
    if position >= Int(rows):
        return
    var own = index[unsafe_offset=position]
    var target = Int(output) * Int(stride) + Int(own)
    var source = Int(code[unsafe_offset=4 * Int(start) + 1]) * Int(stride)
    source += Int(own)
    words[unsafe_offset=target] = 0
    validity[unsafe_offset=target] = validity[unsafe_offset=source]
    errors[unsafe_offset=target] = errors[unsafe_offset=source]
    if validity[unsafe_offset=source] == 0:
        return
    var lo = 0
    var hi = position
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if partition_equal(
            index[unsafe_offset=mid],
            own,
            words,
            validity,
            code,
            types,
            start,
            nodes,
            stride,
        ):
            hi = mid
        else:
            lo = mid + 1
    var base = lo
    var method = code[unsafe_offset=4 * Int(start) + 3]
    if method == 4:
        words[unsafe_offset=target] = UInt64(position - base + 1)
        return
    if method == 3:
        var count = (
            prefix[unsafe_offset=position]
            + blocks[unsafe_offset=position // 256]
        )
        var before = UInt32(0)
        if base:
            before = (
                prefix[unsafe_offset=base - 1]
                + blocks[unsafe_offset=(base - 1) // 256]
            )
        words[unsafe_offset=target] = UInt64(count - before)
        return
    lo = base
    hi = position
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if _rank_equal(
            index[unsafe_offset=mid],
            own,
            words,
            validity,
            code,
            types,
            start,
            nodes,
            stride,
        ):
            hi = mid
        else:
            lo = mid + 1
    var first = lo
    lo = position + 1
    hi = Int(rows)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if _rank_equal(
            index[unsafe_offset=mid],
            own,
            words,
            validity,
            code,
            types,
            start,
            nodes,
            stride,
        ):
            lo = mid + 1
        else:
            hi = mid
    var first_rank = UInt64(first - base + 1)
    var last_rank = UInt64(lo - base)
    words[unsafe_offset=target] = (
        first_rank if method
        == 1 else last_rank if method
        == 2 else floating.div(
            floating.from_unsigned(first_rank + last_rank),
            0x4000000000000000,
        )
    )
