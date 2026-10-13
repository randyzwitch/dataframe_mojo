"""Source-order cumulative, shift and fill windows over stable partitions."""
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives.block import prefix_sum
from std.memory import Pointer
from dataframe.expr import ADD
from .ranking import partition_equal
from .ordering import _nan, _word_less
from .rows import integer_arithmetic
from . import floating


@always_inline
def _extrema_better(a: UInt64, b: UInt64, dtype: Int64, maximum: Bool) -> Bool:
    var an = _nan(a, dtype)
    var bn = _nan(b, dtype)
    if an or bn:
        return an != bn and (an if maximum else bn)
    return _word_less(b, a, dtype) if maximum else _word_less(a, b, dtype)


def window_assign(
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    output: UInt32,
    output_type: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    var position = global_idx.x
    if position >= Int(rows):
        return
    var first = index[unsafe_offset=position]
    var input_base = Int(code[unsafe_offset=4 * Int(start) + 1]) * Int(stride)
    var output_base = Int(output) * Int(stride)
    var method = code[unsafe_offset=4 * Int(start) + 3]
    var parameter = literals[unsafe_offset=Int(start)].cast[DType.int64]()
    if method == 4:
        # A shift is a parallel gather within its partition, including very
        # large shifts and Int64.MIN without negating a signed minimum.
        var lo = 0
        var hi = position
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            if partition_equal(
                index[unsafe_offset=mid],
                first,
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
        var begin = lo
        lo = position + 1
        hi = Int(rows)
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            if partition_equal(
                index[unsafe_offset=mid],
                first,
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
        var end = lo
        var from_position = position
        var present = False
        if parameter >= 0:
            var distance = parameter.cast[DType.uint64]()
            if distance <= UInt64(position - begin):
                from_position = position - Int(distance)
                present = True
        else:
            var distance = UInt64(0) - parameter.cast[DType.uint64]()
            if distance < UInt64(end - position):
                from_position = position + Int(distance)
                present = True
        var target = output_base + Int(first)
        words[unsafe_offset=target] = 0
        validity[unsafe_offset=target] = 0
        errors[unsafe_offset=target] = 0
        if present:
            var source = input_base + Int(index[unsafe_offset=from_position])
            words[unsafe_offset=target] = words[unsafe_offset=source]
            validity[unsafe_offset=target] = validity[unsafe_offset=source]
            errors[unsafe_offset=target] = errors[unsafe_offset=source]
        return
    # One owner walks each partition. This preserves floating association and
    # validates every integer prefix before later filters or terminal limits.
    if position and partition_equal(
        index[unsafe_offset=position - 1],
        first,
        words,
        validity,
        code,
        types,
        start,
        nodes,
        stride,
    ):
        return
    var lo = position + 1
    var hi = Int(rows)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if partition_equal(
            index[unsafe_offset=mid],
            first,
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
    var end = lo
    var reverse = (
        method
        == 6 if method
        >= 5 else code[unsafe_offset=4 * Int(start) + 2]
        != 0
    )
    var dtype = types[unsafe_offset=Int(start)]
    var state = UInt64(0)
    var seen = False
    var last = UInt32(0)
    var distance = UInt64(0)
    var failure = UInt32(0)
    for offset in range(end - position):
        var at_position = end - 1 - offset if reverse else position + offset
        var row = index[unsafe_offset=at_position]
        var source = input_base + Int(row)
        var target = output_base + Int(row)
        var valid = validity[unsafe_offset=source] != 0
        var word = words[unsafe_offset=source]
        words[unsafe_offset=target] = 0
        validity[unsafe_offset=target] = UInt8(valid)
        errors[unsafe_offset=target] = errors[unsafe_offset=source] | failure
        if method >= 5:
            if valid:
                last = row
                seen = True
                distance = 0
            else:
                distance += 1
            if valid or (
                seen and (parameter < 0 or distance <= UInt64(parameter))
            ):
                var anchor = input_base + Int(last)
                words[unsafe_offset=target] = words[unsafe_offset=anchor]
                validity[unsafe_offset=target] = validity[unsafe_offset=anchor]
                errors[unsafe_offset=target] = errors[unsafe_offset=anchor]
            continue
        if method == 3:
            state += UInt64(valid)
            words[unsafe_offset=target] = state
            validity[unsafe_offset=target] = 1
            continue
        if not valid:
            continue
        if method == 0:
            if dtype == 1 or dtype == 11:
                state = floating.add(state, floating.word(word, dtype))
            else:
                # CPU cumulative sums use a checked Int64 accumulator even
                # for UInt64. Narrow outputs retain their declared bounds.
                var kind = Int64(output_type)
                if kind == 10:
                    kind = 3
                var addition = integer_arithmetic(ADD, state, word, kind)
                state = addition[0]
                failure |= addition[1] * 2
                if dtype == 10 and word > UInt64(Int64.MAX):
                    failure |= 2
                errors[unsafe_offset=target] |= failure
        else:
            if not seen or _extrema_better(word, state, dtype, method == 2):
                state = word
            seen = True
        words[unsafe_offset=target] = (
            UInt64(floating.to_float32(state)) if method == 0
            and dtype == 1 else state
        )


def window_check_errors(
    errors: Pointer[UInt32, ImmutAnyOrigin],
    failures: Pointer[UInt32, MutAnyOrigin],
    output: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    var row = global_idx.x
    var failed = UInt32(0)
    if row < Int(rows):
        failed = UInt32(
            errors[unsafe_offset=Int(output) * Int(stride) + row] != 0
        )
    var count = prefix_sum[block_size=256](failed)
    if thread_idx.x == 255:
        failures[unsafe_offset=block_idx.x] = count
