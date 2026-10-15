"""Source-order cumulative, shift and fill windows over stable partitions."""
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives.block import prefix_sum
from std.memory import Pointer
from dataframe.expr import ADD
from dataframe.accelerator.rows import (
    ROW_DURATION_KEY,
    ROW_DURATION_CALENDAR,
    ROW_DURATION_UTC,
    ROW_CALENDAR_OFFSET,
)
from .ranking import key_equal
from .ordering import _nan, _word_less
from .rows import integer_arithmetic, integer_fits, unsigned_type
from .moments import RollingVarianceState
from . import floating, temporal, decimal


@always_inline
def duration_headers(
    code: Pointer[Int64, ImmutAnyOrigin], start: UInt32, nodes: UInt32
) -> UInt32:
    var count = UInt32(2)
    if (
        count < nodes
        and code[unsafe_offset=4 * Int(start + count)] == ROW_DURATION_CALENDAR
    ):
        count += 1
    if (
        count < nodes
        and code[unsafe_offset=4 * Int(start + count)] == ROW_DURATION_UTC
    ):
        count += 1
    return count


@always_inline
def _partition_equal(
    a: UInt32,
    b: UInt32,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
) -> Bool:
    var headers = (
        Int(duration_headers(code, start, nodes)) if code[
            unsafe_offset=4 * Int(start)
        ]
        == ROW_DURATION_KEY else 1
    )
    for key in range(headers, Int(nodes)):
        if not key_equal(
            a,
            b,
            words,
            validity,
            code,
            types,
            start + UInt32(key),
            stride,
            slots,
        ):
            return False
    return True


def _partition_begin(
    position: Int,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
) -> Int:
    var own = index[unsafe_offset=position]
    var lo = 0
    var hi = position
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if _partition_equal(
            index[unsafe_offset=mid],
            own,
            words,
            validity,
            code,
            types,
            start,
            nodes,
            stride,
            slots,
        ):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _partition_end(
    position: Int,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    types: Pointer[Int64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    rows: UInt32,
    stride: UInt32,
    slots: UInt32 = 0,
) -> Int:
    var own = index[unsafe_offset=position]
    var lo = position + 1
    var hi = Int(rows)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if _partition_equal(
            index[unsafe_offset=mid],
            own,
            words,
            validity,
            code,
            types,
            start,
            nodes,
            stride,
            slots,
        ):
            lo = mid + 1
        else:
            hi = mid
    return lo


@always_inline
def _extrema_better(a: UInt64, b: UInt64, dtype: Int64, maximum: Bool) -> Bool:
    var an = _nan(a, dtype)
    var bn = _nan(b, dtype)
    if an or bn:
        return an != bn and (an if maximum else bn)
    return _word_less(b, a, dtype) if maximum else _word_less(a, b, dtype)


def _aggregate_window(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    input_base: Int,
    begin: Int,
    end: Int,
    dtype: Int64,
    output_type: Int64,
    method: Int64,
    needed: UInt64,
    duration: Bool,
    high_base: Int = 0,
    precision: UInt32 = 38,
) -> Tuple[UInt64, UInt8, UInt32, UInt64]:
    var state = UInt64(0)
    var high = UInt64(0)
    var count = UInt64(0)
    var failure = UInt32(0)
    var integer_sum = method == 0 and dtype != 1 and dtype != 11
    for position in range(begin, end):
        var at = input_base + Int(index[unsafe_offset=position])
        if validity[unsafe_offset=at] == 0:
            continue
        var word = words[unsafe_offset=at]
        if (
            not duration
            and method <= 1
            and dtype == 10
            and word > UInt64(Int64.MAX)
        ):
            failure |= 2
        if integer_sum:
            var upper = words[
                unsafe_offset=high_base + at
            ] if dtype == 12 else UInt64(0)
            if dtype != 12 and not unsigned_type(dtype) and word >> 63:
                upper = UInt64.MAX
            # Duration CPU sums validate every observed decimal operand
            # before addition; a preceding opposite value cannot mask an
            # out-of-precision physical payload through cancellation.
            if (
                duration
                and dtype == 12
                and not decimal.cast(
                    decimal.Wide(word, upper),
                    0,
                    0,
                    precision,
                    128,
                    False,
                    False,
                ).valid
            ):
                failure |= 2
            var next = state + word
            high += upper + UInt64(next < state)
            state = next
            if (
                dtype == 12
                and not decimal.cast(
                    decimal.Wide(state, high),
                    0,
                    0,
                    precision,
                    128,
                    False,
                    False,
                ).valid
            ):
                failure |= 2
        elif method <= 1:
            state = floating.add(state, floating.word(word, dtype))
        else:
            var better = _extrema_better(word, state, dtype, method == 3)
            var upper = words[
                unsafe_offset=high_base + at
            ] if dtype == 12 else UInt64(0)
            if dtype == 12:
                better = decimal.signed_less(
                    decimal.Wide(state, high), decimal.Wide(word, upper)
                ) if method == 3 else decimal.signed_less(
                    decimal.Wide(word, upper), decimal.Wide(state, high)
                )
            if duration and (dtype == 1 or dtype == 11):
                # Duration extrema propagate the last source NaN for both
                # min and max, as window.rolling_by_op does.
                better = _nan(word, dtype) or (
                    not _nan(state, dtype) and better
                )
            if count == 0 or better:
                state = word
                high = upper
        count += 1
    if count < needed or (count == 0 and (not duration or method != 0)):
        # Duration windows count eligible values before checked arithmetic.
        # Input and axis faults are observed independently by the caller.
        return (0, 0, UInt32(0) if duration else failure, 0)
    if integer_sum and dtype != 12:
        var expected_high = UInt64(0)
        if not duration or not unsigned_type(output_type):
            expected_high = UInt64.MAX if state >> 63 else UInt64(0)
        if high != expected_high or not integer_fits(state, output_type):
            failure |= 2
        if not duration and output_type == 10 and state >> 63:
            failure |= 2
    if method == 1:
        state = floating.div(state, floating.from_unsigned(count))
    elif method == 0 and dtype == 1:
        state = UInt64(floating.to_float32(state))
    return (state, 1, failure, high if dtype == 12 else UInt64(0))


def _first_finite(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    input_base: Int,
    begin: Int,
    end: Int,
    previous: UInt64,
) -> UInt64:
    for position in range(begin, end):
        var at = input_base + Int(index[unsafe_offset=position])
        if validity[unsafe_offset=at] != 0:
            var word = words[unsafe_offset=at]
            if floating.absolute(word) < floating.INF:
                return word
    return previous


def _update_moments(
    mut state: RollingVarianceState,
    mut origin: UInt64,
    mut lower: Int,
    mut upper: Int,
    begin: Int,
    end: Int,
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    input_base: Int,
):
    var rebuild = begin < lower or end < upper or begin >= upper
    if rebuild:
        lower = begin
        upper = begin
        state = RollingVarianceState()
    while lower < begin:
        var at = input_base + Int(index[unsafe_offset=lower])
        lower += 1
        if validity[unsafe_offset=at] != 0:
            var reset = state.remove(words[unsafe_offset=at], origin)
            rebuild = rebuild or reset
    if rebuild:
        lower = begin
        upper = begin
        state = RollingVarianceState()
        origin = _first_finite(
            words, validity, index, input_base, begin, end, origin
        )
    while upper < end:
        var at = input_base + Int(index[unsafe_offset=upper])
        upper += 1
        if validity[unsafe_offset=at] != 0:
            state.add(words[unsafe_offset=at], origin)


def _interpolate_partition(
    words: Pointer[UInt64, MutAnyOrigin],
    validity: Pointer[UInt8, MutAnyOrigin],
    errors: Pointer[UInt32, MutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    input_base: Int,
    axis_base: Int,
    output_base: Int,
    begin: Int,
    end: Int,
    method: Int64,
    slots: UInt32 = 0,
    stride: UInt32 = 0,
):
    for position in range(begin, end):
        var row = Int(index[unsafe_offset=position])
        var at = input_base + row
        var target = output_base + row
        validity[unsafe_offset=target] = validity[unsafe_offset=at]
        if slots:
            words[unsafe_offset=Int(stride) * Int(slots) + target] = words[
                unsafe_offset=Int(stride) * Int(slots) + at
            ] if validity[unsafe_offset=at] != 0 else UInt64(0)
        words[unsafe_offset=target] = words[unsafe_offset=at] if validity[
            unsafe_offset=at
        ] != 0 else UInt64(0)
        errors[unsafe_offset=target] = errors[unsafe_offset=at]
    var previous = begin
    var seen = False
    var position = begin
    while position < end:
        var row = Int(index[unsafe_offset=position])
        if validity[unsafe_offset=input_base + row] != 0:
            if method != 15 or validity[unsafe_offset=axis_base + row] != 0:
                previous = position
                seen = True
            position += 1
            continue
        if not seen:
            position += 1
            continue
        var following = position + 1
        while following < end:
            var next_row = Int(index[unsafe_offset=following])
            if validity[unsafe_offset=input_base + next_row] != 0 and (
                method != 15
                or validity[unsafe_offset=axis_base + next_row] != 0
            ):
                break
            following += 1
        if following == end:
            break
        var before = Int(index[unsafe_offset=previous])
        var after = Int(index[unsafe_offset=following])
        for fill in range(position, following):
            var at_row = Int(index[unsafe_offset=fill])
            var target = output_base + at_row
            if method == 14:
                var anchor = (
                    before if fill - previous < following - fill else after
                )
                words[unsafe_offset=target] = words[
                    unsafe_offset=input_base + anchor
                ]
                if slots:
                    words[
                        unsafe_offset=Int(stride) * Int(slots) + target
                    ] = words[
                        unsafe_offset=Int(stride) * Int(slots)
                        + input_base
                        + anchor
                    ]
            else:
                var fraction = UInt64(0)
                if method == 15:
                    if (
                        validity[unsafe_offset=input_base + at_row] != 0
                        or validity[unsafe_offset=axis_base + at_row] == 0
                    ):
                        continue
                    var span = floating.sub(
                        words[unsafe_offset=axis_base + after],
                        words[unsafe_offset=axis_base + before],
                    )
                    if floating.absolute(span) == 0:
                        continue
                    fraction = floating.div(
                        floating.sub(
                            words[unsafe_offset=axis_base + at_row],
                            words[unsafe_offset=axis_base + before],
                        ),
                        span,
                    )
                else:
                    fraction = floating.div(
                        floating.from_unsigned(UInt64(fill - previous)),
                        floating.from_unsigned(UInt64(following - previous)),
                    )
                words[unsafe_offset=target] = floating.fma(
                    floating.sub(
                        words[unsafe_offset=input_base + after],
                        words[unsafe_offset=input_base + before],
                    ),
                    fraction,
                    words[unsafe_offset=input_base + before],
                )
            validity[unsafe_offset=target] = 1
        position = following


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
    slots: UInt32 = 0,
):
    var position = global_idx.x
    if position >= Int(rows):
        return
    var first = index[unsafe_offset=position]
    var input_base = Int(code[unsafe_offset=4 * Int(start) + 1]) * Int(stride)
    var output_base = Int(output) * Int(stride)
    var method = code[unsafe_offset=4 * Int(start) + 3]
    var parameter = literals[unsafe_offset=Int(start)].cast[DType.int64]()
    if method >= 7 and method <= 10:
        var begin = _partition_begin(
            position,
            words,
            validity,
            index,
            code,
            types,
            start,
            nodes,
            stride,
            slots,
        )
        var window = literals[unsafe_offset=Int(start)]
        var lower = (
            position - Int(window) + 1 if UInt64(position - begin + 1)
            > window else begin
        )
        var result = _aggregate_window(
            words,
            validity,
            index,
            input_base,
            lower,
            position + 1,
            types[unsafe_offset=Int(start)],
            Int64(output_type),
            method - 7,
            UInt64(code[unsafe_offset=4 * Int(start) + 2]),
            False,
            Int(stride) * Int(slots),
        )
        var target = output_base + Int(first)
        words[unsafe_offset=target] = result[0]
        if slots:
            words[unsafe_offset=Int(stride) * Int(slots) + target] = result[3]
        validity[unsafe_offset=target] = result[1]
        errors[unsafe_offset=target] = result[2]
        return
    if method == 4:
        # A shift is a parallel gather within its partition, including very
        # large shifts and Int64.MIN without negating a signed minimum.
        var lo = 0
        var hi = position
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            if _partition_equal(
                index[unsafe_offset=mid],
                first,
                words,
                validity,
                code,
                types,
                start,
                nodes,
                stride,
                slots,
            ):
                hi = mid
            else:
                lo = mid + 1
        var begin = lo
        lo = position + 1
        hi = Int(rows)
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            if _partition_equal(
                index[unsafe_offset=mid],
                first,
                words,
                validity,
                code,
                types,
                start,
                nodes,
                stride,
                slots,
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
            if slots:
                words[unsafe_offset=Int(stride) * Int(slots) + target] = words[
                    unsafe_offset=Int(stride) * Int(slots) + source
                ]
            validity[unsafe_offset=target] = validity[unsafe_offset=source]
            errors[unsafe_offset=target] = errors[unsafe_offset=source]
        return
    # One owner walks each partition. This preserves floating association and
    # validates every integer prefix before later filters or terminal limits.
    if position and _partition_equal(
        index[unsafe_offset=position - 1],
        first,
        words,
        validity,
        code,
        types,
        start,
        nodes,
        stride,
        slots,
    ):
        return
    var lo = position + 1
    var hi = Int(rows)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if _partition_equal(
            index[unsafe_offset=mid],
            first,
            words,
            validity,
            code,
            types,
            start,
            nodes,
            stride,
            slots,
        ):
            lo = mid + 1
        else:
            hi = mid
    var end = lo
    if method >= 13:
        var axis_base = (
            Int(code[unsafe_offset=4 * Int(start) + 2]) * Int(stride) if method
            == 15 else 0
        )
        _interpolate_partition(
            words,
            validity,
            errors,
            index,
            input_base,
            axis_base,
            output_base,
            position,
            end,
            method,
            slots,
            stride,
        )
        return
    if method == 11 or method == 12:
        var origin = _first_finite(
            words, validity, index, input_base, position, end, 0
        )
        var bounds = literals[unsafe_offset=Int(start)]
        var window = UInt64(UInt32(bounds))
        var ddof = bounds >> 32
        var needed = UInt64(code[unsafe_offset=4 * Int(start) + 2])
        var lower = position
        var upper = position
        var state = RollingVarianceState()
        for at_position in range(position, end):
            var begin = (
                at_position
                - Int(window)
                + 1 if UInt64(at_position - position + 1)
                > window else position
            )
            _update_moments(
                state,
                origin,
                lower,
                upper,
                begin,
                at_position + 1,
                words,
                validity,
                index,
                input_base,
            )
            var target = output_base + Int(index[unsafe_offset=at_position])
            words[unsafe_offset=target] = 0
            validity[unsafe_offset=target] = 0
            errors[unsafe_offset=target] = 0
            if state.valid(needed, ddof):
                words[unsafe_offset=target] = state.finish(ddof, method == 12)
                validity[unsafe_offset=target] = 1
        return
    var reverse = (
        method
        == 6 if method
        >= 5 else code[unsafe_offset=4 * Int(start) + 2]
        != 0
    )
    var dtype = types[unsafe_offset=Int(start)]
    var state = UInt64(0)
    var state_high = UInt64(0)
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
                if slots:
                    words[
                        unsafe_offset=Int(stride) * Int(slots) + target
                    ] = words[unsafe_offset=Int(stride) * Int(slots) + anchor]
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
            if output_type == 12:
                var upper = words[
                    unsafe_offset=Int(stride) * Int(slots) + source
                ] if dtype == 12 else (UInt64.MAX if word >> 63 else UInt64(0))
                var sum = decimal.wide_add(
                    decimal.Wide(state, state_high),
                    decimal.Wide(word, upper),
                )
                state = sum.low
                state_high = sum.high
                if not decimal.cast(
                    sum, 0, 0, UInt32(parameter), 128, False, False
                ).valid:
                    failure |= 2
                errors[unsafe_offset=target] |= failure
            elif dtype == 1 or dtype == 11:
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
            var better = _extrema_better(word, state, dtype, method == 2)
            var high = words[
                unsafe_offset=Int(stride) * Int(slots) + source
            ] if dtype == 12 else UInt64(0)
            if dtype == 12:
                better = decimal.signed_less(
                    decimal.Wide(state, state_high), decimal.Wide(word, high)
                ) if method == 2 else decimal.signed_less(
                    decimal.Wide(word, high), decimal.Wide(state, state_high)
                )
            if not seen or better:
                state = word
                state_high = high
            seen = True
        if slots:
            words[
                unsafe_offset=Int(stride) * Int(slots) + target
            ] = state_high if output_type == 12 else UInt64(0)
        words[unsafe_offset=target] = (
            UInt64(floating.to_float32(state)) if method == 0
            and dtype == 1 else state
        )


def window_input_errors(
    errors: Pointer[UInt32, ImmutAnyOrigin],
    code: Pointer[Int64, ImmutAnyOrigin],
    failures: Pointer[UInt32, MutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    var row = global_idx.x
    var failed = UInt32(0)
    var header = 4 * Int(start)
    var duration = code[unsafe_offset=header] == ROW_DURATION_KEY
    var headers = Int(duration_headers(code, start, nodes)) if duration else 1
    if row < Int(rows):
        var input = Int(code[unsafe_offset=header + 1]) * Int(stride) + row
        failed |= errors[unsafe_offset=input]
        if duration or code[unsafe_offset=header + 3] == 15:
            var axis = Int(code[unsafe_offset=header + 2]) * Int(stride) + row
            failed |= errors[unsafe_offset=axis]
        for key in range(headers, Int(nodes)):
            var base = Int(
                code[unsafe_offset=4 * (Int(start) + key) + 1]
            ) * Int(stride)
            failed |= errors[unsafe_offset=base + row]
    var count = prefix_sum[block_size=256](UInt32(failed != 0))
    if thread_idx.x == 255:
        var at = block_idx.x * (4 if duration else 1)
        failures[unsafe_offset=at] = count
        if duration:
            for fault in range(1, 4):
                failures[unsafe_offset=at + fault] = 0


def _relocalize(
    value: Int64, unit: Int64, offset: Int64
) -> Tuple[Int64, UInt32]:
    var per = temporal.per_second(unit)
    var seconds = value // per
    var fraction = value % per
    if fraction < 0:
        seconds -= 1
        fraction += per
    var product = temporal.checked_mul(seconds - offset, per)
    var joined = temporal.checked_add(product[0], fraction)
    return (joined[0], product[1] | joined[1])


def _duration_lower(
    time: Int64,
    code: Pointer[Int64, ImmutAnyOrigin],
    literals: Pointer[UInt64, ImmutAnyOrigin],
    start: UInt32,
    nodes: UInt32,
) -> Tuple[Int64, UInt32]:
    var period = literals[unsafe_offset=Int(start)].cast[DType.int64]()
    var headers = duration_headers(code, start, nodes)
    var calendar = (
        headers > 2
        and code[unsafe_offset=4 * Int(start + 2)] == ROW_DURATION_CALENDAR
    )
    var zone_node = start + headers - 1
    var zoned = (
        headers > 2
        and code[unsafe_offset=4 * Int(zone_node)] == ROW_DURATION_UTC
    )
    var months = code[
        unsafe_offset=4 * Int(start + 2) + 1
    ] if calendar else Int64(0)
    var unit = code[
        unsafe_offset=4 * Int(start + 2) + 2
    ] if calendar else Int64(0)
    if not zoned:
        if calendar:
            var bound = temporal.evaluate(
                time.cast[DType.uint64](),
                ROW_CALENDAR_OFFSET,
                -months,
                unit,
                (Int64(0) - period).cast[DType.uint64](),
            )
            return (bound[0].cast[DType.int64](), bound[1])
        return temporal.checked_add(time, -period)
    var zone_at = 4 * Int(zone_node)
    var days = code[unsafe_offset=zone_at + 1]
    unit = code[unsafe_offset=zone_at + 2]
    var offset = code[unsafe_offset=zone_at + 3]
    var per = temporal.per_second(unit)
    var offset_ticks = offset * per
    var current = time
    var failure = UInt32(0)
    if calendar:
        var local = (
            time.cast[DType.uint64]() + offset_ticks.cast[DType.uint64]()
        )
        var bound = temporal.evaluate(local, ROW_CALENDAR_OFFSET, -months, unit)
        var instant = _relocalize(bound[0].cast[DType.int64](), unit, offset)
        current = instant[0]
        failure |= bound[1] | instant[1]
    if days:
        var local = (
            current.cast[DType.uint64]() + offset_ticks.cast[DType.uint64]()
        ).cast[DType.int64]()
        var delta = temporal.checked_mul(-days, per * 86400)
        var bound = temporal.checked_add(local, delta[0])
        var instant = _relocalize(bound[0], unit, offset)
        current = instant[0]
        failure |= delta[1] | bound[1] | instant[1]
    var bound = temporal.checked_add(current, -period)
    return (bound[0], failure | bound[1])


def duration_assign(
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
    slots: UInt32 = 0,
):
    var position = global_idx.x
    if position >= Int(rows):
        return
    var own = index[unsafe_offset=position]
    if position and _partition_equal(
        index[unsafe_offset=position - 1],
        own,
        words,
        validity,
        code,
        types,
        start,
        nodes,
        stride,
        slots,
    ):
        return
    var end = _partition_end(
        position,
        words,
        validity,
        index,
        code,
        types,
        start,
        nodes,
        rows,
        stride,
        slots,
    )
    var header = 4 * Int(start)
    var input_base = Int(code[unsafe_offset=header + 1]) * Int(stride)
    var axis_base = Int(code[unsafe_offset=header + 2]) * Int(stride)
    var output_base = Int(output) * Int(stride)
    var owner_target = output_base + Int(own)
    for at_position in range(position, end):
        var target = output_base + Int(index[unsafe_offset=at_position])
        words[unsafe_offset=target] = 0
        validity[unsafe_offset=target] = 0
        errors[unsafe_offset=target] = 0
    # Validate the whole partition before producing a window, including rows
    # later hidden by a conditional, filter or terminal limit.
    for at_position in range(position, end):
        var at = axis_base + Int(index[unsafe_offset=at_position])
        if validity[unsafe_offset=at] == 0:
            errors[unsafe_offset=owner_target] = 4
            return
        if at_position > position:
            var previous = axis_base + Int(index[unsafe_offset=at_position - 1])
            if (
                words[unsafe_offset=at].cast[DType.int64]()
                < words[unsafe_offset=previous].cast[DType.int64]()
            ):
                errors[unsafe_offset=owner_target] = 8
                return
    var method = code[unsafe_offset=header + 3]
    var needed = UInt64(code[unsafe_offset=header + 5])
    var closed = code[unsafe_offset=header + 6]
    var ddof = UInt64(code[unsafe_offset=header + 7])
    var origin = _first_finite(
        words, validity, index, input_base, position, end, 0
    ) if method >= 4 else UInt64(0)
    var lower = position
    var upper = position
    var state = RollingVarianceState()
    for at_position in range(position, end):
        var row = Int(index[unsafe_offset=at_position])
        var target = output_base + row
        var time = words[unsafe_offset=axis_base + row].cast[DType.int64]()
        var bound = _duration_lower(time, code, literals, start, nodes)
        if bound[1]:
            errors[unsafe_offset=target] = 16
            return
        var lo = position
        var hi = end
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            var tick = words[
                unsafe_offset=axis_base + Int(index[unsafe_offset=mid])
            ].cast[DType.int64]()
            if tick < bound[0] or (closed & 1 == 0 and tick == bound[0]):
                lo = mid + 1
            else:
                hi = mid
        var begin = lo
        lo = position
        hi = end
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            var tick = words[
                unsafe_offset=axis_base + Int(index[unsafe_offset=mid])
            ].cast[DType.int64]()
            if tick < time or (closed & 2 != 0 and tick == time):
                lo = mid + 1
            else:
                hi = mid
        var stop = max(begin, lo)
        if method >= 4:
            _update_moments(
                state,
                origin,
                lower,
                upper,
                begin,
                stop,
                words,
                validity,
                index,
                input_base,
            )
            if state.valid(needed, ddof):
                words[unsafe_offset=target] = state.finish(ddof, method == 5)
                validity[unsafe_offset=target] = 1
        else:
            var result = _aggregate_window(
                words,
                validity,
                index,
                input_base,
                begin,
                stop,
                types[unsafe_offset=Int(start)],
                Int64(output_type),
                method,
                needed,
                True,
                Int(stride) * Int(slots),
                UInt32(code[unsafe_offset=header + 7]) if method
                == 0 else UInt32(38),
            )
            words[unsafe_offset=target] = result[0]
            if slots:
                words[unsafe_offset=Int(stride) * Int(slots) + target] = result[
                    3
                ]
            validity[unsafe_offset=target] = result[1]
            errors[unsafe_offset=target] = result[2]


def duration_check_errors(
    errors: Pointer[UInt32, ImmutAnyOrigin],
    failures: Pointer[UInt32, MutAnyOrigin],
    output: UInt32,
    rows: UInt32,
    stride: UInt32,
):
    var row = global_idx.x
    var fault = UInt32(0)
    if row < Int(rows):
        fault = errors[unsafe_offset=Int(output) * Int(stride) + row]
    var nulls = prefix_sum[block_size=256](UInt32(fault & 4 != 0))
    var unsorted = prefix_sum[block_size=256](UInt32(fault & 8 != 0))
    var bounds = prefix_sum[block_size=256](UInt32(fault & 16 != 0))
    var overflow = prefix_sum[block_size=256](UInt32(fault & 2 != 0))
    if thread_idx.x == 255:
        var base = 4 * block_idx.x
        failures[unsafe_offset=base] = nulls
        failures[unsafe_offset=base + 1] = unsorted
        failures[unsafe_offset=base + 2] = bounds
        failures[unsafe_offset=base + 3] = overflow


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
