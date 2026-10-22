"""Ordered fixed-width as-of matching with the CPU's stateful scan policy.

Inputs use joins' combined columns and paired high-word storage. Stable-sort
each side by BY keys only with join_merge (left side=1, right side=0); ordered
keys must remain in source order within groups. Validate both complete sides,
reduce/read the control, and reject faults before matching or output limiting.

Nearest retains null right rows and adjacent duplicate-run boundaries. It
uses ordinary IEEE comparisons for its cursors, unlike the total ordering
used by sortedness and backward/forward. One thread scans each BY group.
Groups with null BY members are validated but never matched. A host binder
must validate identical logical key types and normalize dictionary identity.
"""
from max.gpu import global_idx
from std.memory import Pointer
from .joins import COLUMN_WORDS, OUTPUT_WORDS, MISSING, tuple_compare
from .ordering import _word_less, _nan
from .grouping import key_equal
from .rows import unsigned_type
from . import floating

comptime CONTROL_WORDS = 3


def workspace_bytes(left_rows: Int, right_rows: Int) -> Int:
    """Device permutation/check/end/pair/control storage, excluding inputs.

    Host control readback adds CONTROL_WORDS * 4 bytes. Allocate one element
    even for empty arrays, and charge input/output descriptors separately.
    """
    return 24 * max(1, left_rows) + 20 * max(1, right_rows) + CONTROL_WORDS * 4


@always_inline
def _at(
    columns: Pointer[Int64, ImmutAnyOrigin], column: Int64, row: UInt32
) -> Int:
    return Int(columns[unsafe_offset=Int(column) * COLUMN_WORDS]) + Int(row)


@always_inline
def _total_less(a: UInt64, b: UInt64, dtype: Int64) -> Bool:
    var an = _nan(a, dtype)
    var bn = _nan(b, dtype)
    return not an if an != bn else _word_less(a, b, dtype)


@always_inline
def _ordinary_equal(a: UInt64, b: UInt64, dtype: Int64) -> Bool:
    return not _nan(a, dtype) and not _nan(b, dtype) and key_equal(a, b, dtype)


@always_inline
def _ordinary_ge(a: UInt64, b: UInt64, dtype: Int64) -> Bool:
    return (
        not _nan(a, dtype)
        and not _nan(b, dtype)
        and not _word_less(a, b, dtype)
    )


@always_inline
def _integer_distance(a: UInt64, b: UInt64, dtype: Int64) -> UInt64:
    # Bias signed storage to unsigned order; the largest exact distance is
    # UInt64.MAX, including Int64.MIN -> Int64.MAX.
    var left = a
    var right = b
    if not unsigned_type(dtype):
        left ^= UInt64(1) << 63
        right ^= UInt64(1) << 63
    return left - right if left >= right else right - left


@always_inline
def _float_distance(a: UInt64, b: UInt64, dtype: Int64) -> UInt64:
    return floating.absolute(
        floating.sub(floating.word(a, dtype), floating.word(b, dtype))
    )


@always_inline
def _float_le(a: UInt64, b: UInt64) -> Bool:
    return floating.less(a, b) or floating.equal(a, b)


def asof_validate(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    columns: Pointer[Int64, ImmutAnyOrigin],
    keys: Pointer[Int64, ImmutAnyOrigin],
    checks: Pointer[UInt64, MutAnyOrigin],
    ordered_column: Int64,
    key_count: UInt32,
    side: UInt32,
    rows: UInt32,
    cells: UInt64 = 0,
):
    """Write each row's key errors and first-order violation source position.

    checks are in permutation order, one word per row: low32 error bits,
    high32 unsorted source row +1 (zero means no order fault).
    """
    var begin = global_idx.x
    if begin >= Int(rows):
        return
    var first = index[unsafe_offset=begin]
    if (
        begin
        and tuple_compare(
            index[unsafe_offset=begin - 1],
            first,
            False,
            words,
            validity,
            columns,
            keys,
            key_count,
            side,
            cells,
        )
        == 0
    ):
        return
    var dtype = columns[unsafe_offset=Int(ordered_column) * COLUMN_WORDS + 2]
    var previous = MISSING
    var position = begin
    while position < Int(rows):
        var row = index[unsafe_offset=position]
        if (
            position != begin
            and tuple_compare(
                first,
                row,
                False,
                words,
                validity,
                columns,
                keys,
                key_count,
                side,
                cells,
            )
            != 0
        ):
            break
        var at = _at(columns, ordered_column, row)
        var observed = errors[unsafe_offset=at]
        for key in range(Int(key_count)):
            var column = keys[unsafe_offset=2 * key + (0 if side else 1)]
            observed |= errors[unsafe_offset=_at(columns, column, row)]
        var fault = UInt64(0)
        if validity[unsafe_offset=at]:
            if previous != MISSING and _total_less(
                words[unsafe_offset=at],
                words[unsafe_offset=_at(columns, ordered_column, previous)],
                dtype,
            ):
                fault = UInt64(row + 1) << 32
            previous = row
        checks[unsafe_offset=position] = fault | UInt64(observed)
        position += 1


def asof_control(
    left: Pointer[UInt64, ImmutAnyOrigin],
    right: Pointer[UInt64, ImmutAnyOrigin],
    control: Pointer[UInt32, MutAnyOrigin],
    left_rows: UInt32,
    right_rows: UInt32,
):
    """Read back three UInt32s: first bad left/right row, and observed errors."""
    if global_idx.x:
        return
    var first_left = MISSING
    var first_right = MISSING
    var observed = UInt32(0)
    for row in range(Int(left_rows)):
        var value = left[unsafe_offset=row]
        observed |= UInt32(value)
        if value >> 32:
            first_left = min(first_left, UInt32(value >> 32) - 1)
    for row in range(Int(right_rows)):
        var value = right[unsafe_offset=row]
        observed |= UInt32(value)
        if value >> 32:
            first_right = min(first_right, UInt32(value >> 32) - 1)
    control[unsafe_offset=0] = first_left
    control[unsafe_offset=1] = first_right
    control[unsafe_offset=2] = observed


def asof_right_ends(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    index: Pointer[UInt32, ImmutAnyOrigin],
    columns: Pointer[Int64, ImmutAnyOrigin],
    keys: Pointer[Int64, ImmutAnyOrigin],
    ends: Pointer[UInt32, MutAnyOrigin],
    ordered_column: Int64,
    key_count: UInt32,
    rows: UInt32,
    cells: UInt64 = 0,
):
    var begin = global_idx.x
    if begin >= Int(rows):
        return
    var first = index[unsafe_offset=begin]
    if (
        begin
        and tuple_compare(
            index[unsafe_offset=begin - 1],
            first,
            False,
            words,
            validity,
            columns,
            keys,
            key_count,
            0,
            cells,
        )
        == 0
    ):
        return
    var end = begin + 1
    while (
        end < Int(rows)
        and tuple_compare(
            first,
            index[unsafe_offset=end],
            False,
            words,
            validity,
            columns,
            keys,
            key_count,
            0,
            cells,
        )
        == 0
    ):
        end += 1
    var dtype = columns[unsafe_offset=Int(ordered_column) * COLUMN_WORDS + 2]
    for position in range(end - 1, begin - 1, -1):
        var at = _at(columns, ordered_column, index[unsafe_offset=position])
        var last = UInt32(position)
        if position + 1 < end:
            var next = _at(
                columns, ordered_column, index[unsafe_offset=position + 1]
            )
            if (
                validity[unsafe_offset=at]
                and validity[unsafe_offset=next]
                and _ordinary_equal(
                    words[unsafe_offset=at], words[unsafe_offset=next], dtype
                )
            ):
                last = ends[unsafe_offset=position + 1]
        ends[unsafe_offset=position] = last


def asof_match(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    left_index: Pointer[UInt32, ImmutAnyOrigin],
    right_index: Pointer[UInt32, ImmutAnyOrigin],
    right_ends: Pointer[UInt32, ImmutAnyOrigin],
    columns: Pointer[Int64, ImmutAnyOrigin],
    keys: Pointer[Int64, ImmutAnyOrigin],
    pairs: Pointer[UInt64, MutAnyOrigin],
    ordered_left: Int64,
    ordered_right: Int64,
    key_count: UInt32,
    left_rows: UInt32,
    right_rows: UInt32,
    strategy: UInt32,
    exact: UInt32,
    has_tolerance: UInt32,
    tolerance_integer: UInt64,
    tolerance_float: UInt64,
    cells: UInt64 = 0,
):
    """Scatter source pairs (left low32, right high32; MISSING means null).

    strategy is backward0/forward1/nearest2. Float tolerance is binary64 bits.
    Host-validated integer/Duration tolerances saturate at UInt64.MAX. Control
    validation must succeed before this kernel is submitted.
    """
    var begin = global_idx.x
    if begin >= Int(left_rows):
        return
    var first = left_index[unsafe_offset=begin]
    if (
        begin
        and tuple_compare(
            left_index[unsafe_offset=begin - 1],
            first,
            False,
            words,
            validity,
            columns,
            keys,
            key_count,
            1,
            cells,
        )
        == 0
    ):
        return
    var matchable = True
    for key in range(Int(key_count)):
        var column = keys[unsafe_offset=2 * key]
        matchable &= validity[unsafe_offset=_at(columns, column, first)] != 0
    var low = 0
    var high = Int(right_rows)
    while low < high:
        var mid = low + (high - low) // 2
        if (
            tuple_compare(
                right_index[unsafe_offset=mid],
                first,
                True,
                words,
                validity,
                columns,
                keys,
                key_count,
                0,
                cells,
            )
            < 0
        ):
            low = mid + 1
        else:
            high = mid
    var right_begin = low
    high = Int(right_rows)
    while low < high:
        var mid = low + (high - low) // 2
        if (
            tuple_compare(
                right_index[unsafe_offset=mid],
                first,
                True,
                words,
                validity,
                columns,
                keys,
                key_count,
                0,
                cells,
            )
            <= 0
        ):
            low = mid + 1
        else:
            high = mid
    var right_end = low
    var lower = right_begin
    var upper = right_begin
    var smaller = MISSING
    var dtype = columns[unsafe_offset=Int(ordered_left) * COLUMN_WORDS + 2]
    var position = begin
    while position < Int(left_rows):
        var row = left_index[unsafe_offset=position]
        if (
            position != begin
            and tuple_compare(
                first,
                row,
                False,
                words,
                validity,
                columns,
                keys,
                key_count,
                1,
                cells,
            )
            != 0
        ):
            break
        var chosen = MISSING
        var at = _at(columns, ordered_left, row)
        if matchable and validity[unsafe_offset=at]:
            var value = words[unsafe_offset=at]
            if strategy == 0:
                while upper < right_end:
                    var candidate = right_index[unsafe_offset=upper]
                    var key_at = _at(columns, ordered_right, candidate)
                    if not validity[unsafe_offset=key_at]:
                        upper += 1
                        continue
                    var word = words[unsafe_offset=key_at]
                    var equal = not _total_less(
                        value, word, dtype
                    ) and not _total_less(word, value, dtype)
                    if _total_less(word, value, dtype) or (exact and equal):
                        smaller = candidate
                        upper += 1
                    else:
                        break
                chosen = smaller
            elif strategy == 1:
                while lower < right_end:
                    var candidate = right_index[unsafe_offset=lower]
                    var key_at = _at(columns, ordered_right, candidate)
                    if validity[unsafe_offset=key_at]:
                        var word = words[unsafe_offset=key_at]
                        var equal = not _total_less(
                            value, word, dtype
                        ) and not _total_less(word, value, dtype)
                        if _total_less(value, word, dtype) or (exact and equal):
                            break
                    lower += 1
                if lower < right_end:
                    chosen = right_index[unsafe_offset=lower]
            else:
                var hi = upper
                while hi < right_end:
                    var key_at = _at(
                        columns, ordered_right, right_index[unsafe_offset=hi]
                    )
                    if validity[unsafe_offset=key_at] and _word_less(
                        value, words[unsafe_offset=key_at], dtype
                    ):
                        break
                    hi += 1
                var previous_at = (
                    _at(
                        columns,
                        ordered_right,
                        right_index[unsafe_offset=hi - 1],
                    ) if hi
                    > right_begin else 0
                )
                if (
                    exact
                    and hi > right_begin
                    and validity[unsafe_offset=previous_at]
                    and _ordinary_equal(
                        words[unsafe_offset=previous_at], value, dtype
                    )
                ):
                    chosen = right_index[unsafe_offset=hi - 1]
                else:
                    if hi < right_end:
                        hi = Int(right_ends[unsafe_offset=hi])
                    var lo = lower
                    while lo < hi:
                        var candidate = right_index[unsafe_offset=lo]
                        var key_at = _at(columns, ordered_right, candidate)
                        if validity[unsafe_offset=key_at]:
                            if _ordinary_ge(
                                words[unsafe_offset=key_at], value, dtype
                            ):
                                break
                            smaller = candidate
                        lo += 1
                    lower = lo
                    chosen = smaller
                    if hi < right_end:
                        var candidate = right_index[unsafe_offset=hi]
                        if chosen == MISSING:
                            chosen = candidate
                        else:
                            var newer = words[
                                unsafe_offset=_at(
                                    columns, ordered_right, candidate
                                )
                            ]
                            var older = words[
                                unsafe_offset=_at(
                                    columns, ordered_right, chosen
                                )
                            ]
                            var closer = _float_le(
                                _float_distance(newer, value, dtype),
                                _float_distance(value, older, dtype),
                            ) if dtype == 1 or dtype == 11 else _integer_distance(
                                newer, value, dtype
                            ) <= _integer_distance(
                                value, older, dtype
                            )
                            if closer:
                                chosen = candidate
                upper = hi
            if chosen != MISSING and has_tolerance:
                var candidate = words[
                    unsafe_offset=_at(columns, ordered_right, chosen)
                ]
                var within = (
                    _float_le(
                        _float_distance(value, candidate, dtype),
                        tolerance_float,
                    ) if dtype
                    == 1
                    or dtype
                    == 11 else _integer_distance(value, candidate, dtype)
                    <= tolerance_integer
                )
                if not within:
                    chosen = MISSING
        pairs[unsafe_offset=Int(row)] = UInt64(row) | (UInt64(chosen) << 32)
        position += 1


def asof_gather(
    words: Pointer[UInt64, ImmutAnyOrigin],
    validity: Pointer[UInt8, ImmutAnyOrigin],
    errors: Pointer[UInt32, ImmutAnyOrigin],
    columns: Pointer[Int64, ImmutAnyOrigin],
    outputs: Pointer[Int64, ImmutAnyOrigin],
    pairs: Pointer[UInt64, ImmutAnyOrigin],
    output_words: Pointer[UInt64, MutAnyOrigin],
    output_validity: Pointer[UInt8, MutAnyOrigin],
    output_errors: Pointer[UInt32, MutAnyOrigin],
    rows: UInt32,
    output_count: UInt32,
    cells: UInt64 = 0,
):
    """Gather the requested left prefix after full-input validation/matching."""
    var row = global_idx.x
    if row >= Int(rows):
        return
    var pair = pairs[unsafe_offset=row]
    var left_row = UInt32(pair)
    var right_row = UInt32(pair >> 32)
    var capacity = Int(max(rows, UInt32(1)))
    for column in range(Int(output_count)):
        var at = column * OUTPUT_WORDS
        var left = outputs[unsafe_offset=at]
        var right = outputs[unsafe_offset=at + 1]
        var prefer_right = outputs[unsafe_offset=at + 2] != 0
        var source = left
        var selected = left_row
        if prefer_right and right_row != MISSING:
            source = right
            selected = right_row
        elif left < 0:
            source = right
            selected = right_row
        var word = UInt64(0)
        var high = UInt64(0)
        var valid = UInt8(0)
        var error = UInt32(0)
        if source >= 0 and selected != MISSING:
            var original = _at(columns, source, selected)
            valid = validity[unsafe_offset=original]
            error = errors[unsafe_offset=original]
            if valid:
                word = words[unsafe_offset=original]
                if cells:
                    high = words[unsafe_offset=Int(cells) + original]
        var target = Int(outputs[unsafe_offset=at + 3]) * capacity + row
        output_words[unsafe_offset=target] = word
        if cells:
            output_words[
                unsafe_offset=Int(output_count) * capacity + target
            ] = high
        output_validity[unsafe_offset=target] = valid
        output_errors[unsafe_offset=target] = error
