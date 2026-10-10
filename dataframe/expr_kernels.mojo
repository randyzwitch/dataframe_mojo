"""Batch kernels: operation/dtype dispatch occurs outside element loops."""
from std.math import sqrt, exp, log, floor, ceil, pow, isinf, isnan
from std.sys import size_of
from .float_ops import arithmetic, compare
from .nested_column import ListColumn, StructColumn
from .bool_column import BoolColumn
from .column import Column, _bit, _copy_validity, _pack_bits
from .dtype import DataType, NUMERIC_DTYPES
from .decimal import (
    common_decimal,
    check_limit,
    check_precision,
    precision_limit,
    divide_half_even,
    pow10,
    round_half_even,
)
from .string_column import StringColumn, StringBuilder
from .series import Series
from .string_predicates import compare_with_literal
from .expr import (
    ADD,
    SUB,
    MUL,
    GT,
    EQ,
    LT,
    GE,
    LE,
    NE,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    CLIP_LOW,
    CLIP_HIGH,
    NEG,
    ABS,
    SQRT,
    EXP,
    LOG,
    FLOOR,
    CEIL,
    ROUND,
    AND,
    OR,
    XOR,
    FILL_NULL,
    FILL_NAN,
    KEEP_NULLS,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    is_comparison,
    is_logical,
)

comptime INT64_MIN = Int64(-9223372036854775807) - 1
comptime INT64_MAX = Int64(9223372036854775807)


def checked_add(a: Int64, b: Int64) raises -> Int64:
    """Raise before signed 64-bit overflow; never silently promote to float."""
    if (b > 0 and a > INT64_MAX - b) or (b < 0 and a < INT64_MIN - b):
        raise Error("Int64 sum overflow")
    return a + b


def _checked_sub(a: Int64, b: Int64) raises -> Int64:
    if (b > 0 and a < INT64_MIN + b) or (b < 0 and a > INT64_MAX + b):
        raise Error("Int64 subtraction overflow")
    return a - b


def _checked_mul(a: Int64, b: Int64) raises -> Int64:
    var negative = (a < 0) != (b < 0)
    var ua = UInt64(-(a + 1)) + 1 if a < 0 else UInt64(a)
    var ub = UInt64(-(b + 1)) + 1 if b < 0 else UInt64(b)
    var limit = UInt64(9223372036854775807) + UInt64(negative)
    if ub != 0 and ua > limit // ub:
        raise Error("Int64 multiplication overflow")
    var magnitude = ua * ub
    if negative and magnitude > 0:
        return -Int64(magnitude - 1) - 1
    return Int64(magnitude)


def _checked_pow(base: Int64, exponent: Int64) raises -> Int64:
    if exponent < 0:
        raise Error("Int64 pow requires a nonnegative exponent")
    var result = Int64(1)
    var factor = base
    var remaining = exponent
    while remaining > 0:
        if remaining & 1 == 1:
            result = _checked_mul(result, factor)
        remaining >>= 1
        if remaining > 0:
            factor = _checked_mul(factor, factor)
    return result


def _round_half_away(value: Float64, decimals: Int) -> Float64:
    # Values at or beyond 2**52 are already integral; scaling could overflow.
    if isnan(value) or isinf(value) or abs(value) >= 4503599627370496.0:
        return value
    var scale = pow(Float64(10), Float64(abs(decimals)))
    var scaled = value * scale if decimals >= 0 else value / scale
    var rounded = floor(abs(scaled) + 0.5)
    if scaled < 0:
        rounded = -rounded
    return rounded / scale if decimals >= 0 else rounded * scale


def _length(left: Int, right: Int) raises -> Int:
    if left != right and left != 1 and right != 1:
        raise Error("Incompatible expression lengths")
    if left == 0 or right == 0:
        return 0
    return max(left, right)


def fit_mask(mask: List[Bool], n: Int) -> List[Bool]:
    """Resize an evaluation mask to a kernel's output length.

    Empty means every row is active. A scalar result is active when any row
    that could observe it is active.
    """
    if len(mask) == 0 or len(mask) == n:
        return mask.copy()
    if n == 1:
        var any_active = False
        for active in mask:
            any_active = any_active or active
        return [any_active]
    return List[Bool](length=n, fill=mask[0])


def _float_scalar[op: Int, D: DType](x: Scalar[D], y: Scalar[D]) -> Scalar[D]:
    comptime if op == FLOORDIV:
        return floor(x / y)
    elif op == MOD:
        if y == 0 or isnan(x) or isnan(y) or isinf(x):
            return Scalar[D](0) / Scalar[D](0)
        return x % y
    elif op == POW:
        return pow(x, y)
    elif op == CLIP_LOW:
        if x != x or y != y:
            return x
        return y if x < y else x
    else:
        if x != x or y != y:
            return x
        return y if x > y else x


def _lanes[
    D: DType, width: Int
](column: Column[Scalar[D]], start: Int) -> SIMD[D, width]:
    """`width` payloads from row `start` read straight from the buffer, or a
    splat of the single value of a broadcast (one-row) operand."""
    if len(column) == 1:
        return SIMD[D, width](column._get(0))
    return column._ptr().unsafe_load[width=width](start)


def _mask[width: Int](valid: List[Bool], start: Int) -> SIMD[DType.bool, width]:
    """Validity flags for rows [start, start + width) as a vector mask."""
    return (
        valid.unsafe_ptr()
        .unsafe_offset(start)
        .unsafe_bitcast[Scalar[DType.bool]]()
        .unsafe_load[width=width]()
    )


def _numeric_float[
    op: Int, width: Int, D: DType
](left: Column[Scalar[D]], right: Column[Scalar[D]]) raises -> Series:
    """Float binary kernel with contiguous SIMD loads from the source
    buffers (no gather or copy). Null lanes are computed on their payload
    slots (harmless for floats) and zeroed in the output; the final
    `n % width` rows run one lane at a time, so no load passes the end."""
    var n = _length(len(left), len(right))
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        valid[i] = left._valid(0 if len(left) == 1 else i) and right._valid(
            0 if len(right) == 1 else i
        )
    comptime predicate = is_comparison(op)
    var values = List[Scalar[D]](length=0 if predicate else n, fill=0)
    var predicates = List[Bool](length=n if predicate else 0, fill=False)
    var main = n - n % width
    for start in range(0, main, width):
        _float_block[op, width, D](
            left, right, start, valid, values, predicates
        )
    for start in range(main, n):
        _float_block[op, 1, D](left, right, start, valid, values, predicates)
    comptime if predicate:
        return Series("", BoolColumn(predicates^, valid))
    else:
        return Series("", Column[Scalar[D]](values^, valid))


def _float_block[
    op: Int, width: Int, D: DType
](
    left: Column[Scalar[D]],
    right: Column[Scalar[D]],
    start: Int,
    valid: List[Bool],
    mut values: List[Scalar[D]],
    mut predicates: List[Bool],
):
    var x = _lanes[D, width](left, start)
    var y = _lanes[D, width](right, start)
    comptime if is_comparison(op):
        var result = compare[D, width](op, x, y)
        var mask = _mask[width](valid, start)
        predicates.unsafe_ptr().unsafe_offset(start).unsafe_bitcast[
            Scalar[DType.bool]
        ]().unsafe_store(result & mask)
    else:
        var result: SIMD[D, width]
        comptime if op == ADD or op == SUB or op == MUL or op == DIV:
            result = arithmetic[D, width](op, x, y)
        else:
            result = SIMD[D, width](0)
            comptime for lane in range(width):
                result[lane] = _float_scalar[op, D](x[lane], y[lane])
        values.unsafe_ptr().unsafe_offset(start).unsafe_store(
            _mask[width](valid, start).select(result, SIMD[D, width](0))
        )


# Integer widths other than Int64 compute in 128 bits (signed or unsigned to
# match the operand) and range-check the result, which covers MIN // -1,
# unsigned underflow, and every product of two 64-bit magnitudes.


def _wide_type(D: DType) -> DType:
    return DType.int128 if D.is_signed() else DType.uint128


def _narrow[
    D: DType, W: DType
](value: Scalar[W], what: String) raises -> Scalar[D]:
    if value > Scalar[D].MAX.cast[W]() or value < Scalar[D].MIN.cast[W]():
        raise Error(String(D) + " " + what + " overflow")
    return value.cast[D]()


def _int_binary[
    op: Int, D: DType
](x: Scalar[D], y: Scalar[D]) raises -> Scalar[D]:
    """Checked + - * ** // % for one integer width; // and % assume y != 0."""
    comptime if D == DType.int64:
        var a = rebind[Int64](x)
        var b = rebind[Int64](y)
        var r: Int64
        comptime if op == ADD:
            r = checked_add(a, b)
        elif op == SUB:
            r = _checked_sub(a, b)
        elif op == MUL:
            r = _checked_mul(a, b)
        elif op == POW:
            r = _checked_pow(a, b)
        elif op == FLOORDIV:
            if b == -1:
                if a == INT64_MIN:
                    raise Error("Int64 floor division overflow")
                r = -a
            else:
                r = a // b
        else:
            r = 0 if b == -1 else a % b
        return rebind[Scalar[D]](r)
    else:
        comptime W = _wide_type(D)
        var a = x.cast[W]()
        var b = y.cast[W]()
        comptime if op == ADD:
            return _narrow[D, W](a + b, "addition")
        elif op == SUB:
            comptime if not D.is_signed():
                if b > a:
                    raise Error(String(D) + " subtraction overflow")
            return _narrow[D, W](a - b, "subtraction")
        elif op == MUL:
            return _narrow[D, W](a * b, "multiplication")
        elif op == FLOORDIV:
            return _narrow[D, W](a // b, "floor division")
        elif op == MOD:
            return (a % b).cast[D]()
        else:
            comptime if D.is_signed():
                if b < 0:
                    raise Error(
                        String(D) + " pow requires a nonnegative exponent"
                    )
            var result = Scalar[W](1)
            var factor = a
            var remaining = b
            while remaining > 0:
                if remaining & 1 == 1:
                    result = _narrow[D, W](result * factor, "pow").cast[W]()
                remaining >>= 1
                if remaining > 0:
                    factor = _narrow[D, W](factor * factor, "pow").cast[W]()
            return result.cast[D]()


@always_inline
def _lane_op[
    op: Int, L: DType, lanes: Int
](x: SIMD[L, lanes], y: SIMD[L, lanes]) -> SIMD[L, lanes]:
    comptime if op == ADD:
        return x + y
    elif op == SUB:
        return x - y
    else:
        return x * y


def _vector_int[
    op: Int, D: DType
](left: Column[Scalar[D]], right: Column[Scalar[D]], n: Int) raises -> Series:
    """`+`, `-` or `*` of signed 8-, 16- or 32-bit columns without nulls,
    16 rows at a time in Int64 lanes, with each block's range checked once
    (#384). An overflowing block is redone row by row, so the error is the
    checked operation's own."""
    comptime lanes = 16
    # 8- and 16-bit values add, subtract and multiply within Int32, so
    # their lanes are Int32: half the registers of Int64 lanes (ClickBench
    # q29 adds a constant to a 16-bit column 90 times).
    comptime L = DType.int32 if size_of[Scalar[D]]() <= 2 else DType.int64
    var values = List[Scalar[D]](unsafe_uninit_length=n)
    var out = values.unsafe_ptr()
    var a = left._ptr()
    var b = right._ptr()
    var a_scalar = len(left) == 1
    var b_scalar = len(right) == 1
    var low = SIMD[L, lanes](Scalar[L](Scalar[D].MIN))
    var high = SIMD[L, lanes](Scalar[L](Scalar[D].MAX))
    var i = 0
    while i + lanes <= n:
        var x = SIMD[L, lanes](Scalar[L](a[])) if a_scalar else a.unsafe_load[
            width=lanes
        ](i).cast[L]()
        var y = SIMD[L, lanes](Scalar[L](b[])) if b_scalar else b.unsafe_load[
            width=lanes
        ](i).cast[L]()
        var r = _lane_op[op, L, lanes](x, y)
        if (r.lt(low) | r.gt(high)).reduce_or():
            for k in range(lanes):
                _ = _int_binary[op, D](
                    left._get(0 if a_scalar else i + k),
                    right._get(0 if b_scalar else i + k),
                )
        out.unsafe_store(i, r.cast[D]())
        i += lanes
    while i < n:
        out.unsafe_offset(i)[] = _int_binary[op, D](
            left._get(0 if a_scalar else i), right._get(0 if b_scalar else i)
        )
        i += 1
    return Series("", Column[Scalar[D]](values^))


def _numeric_int[
    op: Int, D: DType
](
    left: Column[Scalar[D]], right: Column[Scalar[D]], mask: List[Bool]
) raises -> Series:
    var n = _length(len(left), len(right))
    comptime if (op == ADD or op == SUB or op == MUL) and (
        D == DType.int8 or D == DType.int16 or D == DType.int32
    ):
        if (
            len(mask) == 0
            and n > 0
            and left.null_count() == 0
            and right.null_count() == 0
        ):
            return _vector_int[op, D](left, right, n)
    var active = fit_mask(mask, n)
    var valid = List[Bool](length=n, fill=False)
    comptime predicate = is_comparison(op)
    comptime floating = op == DIV
    var values = List[Scalar[D]](
        length=0 if predicate or floating else n, fill=0
    )
    var floats = List[Float64](length=n if floating else 0, fill=0)
    var predicates = List[Bool](length=n if predicate else 0, fill=False)
    # Checked integer arithmetic stays scalar until a vector overflow path
    # has equivalent semantics. No null payload enters arithmetic.
    for i in range(n):
        var a = 0 if len(left) == 1 else i
        var b = 0 if len(right) == 1 else i
        valid[i] = left._valid(a) and right._valid(b)
        # Rows outside the mask are never selected, so they must not raise.
        if len(active) > 0 and not active[i]:
            valid[i] = False
        if not valid[i]:
            continue
        var x = left._get(a)
        var y = right._get(b)
        comptime if op == ADD or op == SUB or op == MUL or op == POW:
            values[i] = _int_binary[op, D](x, y)
        elif op == DIV:
            floats[i] = x.cast[DType.float64]() / y.cast[DType.float64]()
        elif op == FLOORDIV or op == MOD:
            if y == 0:
                valid[i] = False
            else:
                values[i] = _int_binary[op, D](x, y)
        elif op == CLIP_LOW:
            values[i] = max(x, y)
        elif op == CLIP_HIGH:
            values[i] = min(x, y)
        elif op == GT:
            predicates[i] = x > y
        elif op == LT:
            predicates[i] = x < y
        elif op == GE:
            predicates[i] = x >= y
        elif op == LE:
            predicates[i] = x <= y
        elif op == EQ:
            predicates[i] = x == y
        else:
            predicates[i] = x != y
    comptime if predicate:
        return Series("", BoolColumn(predicates^, valid))
    elif floating:
        return Series("", Column[Float64](floats^, valid))
    else:
        return Series("", Column[Scalar[D]](values^, valid))


def _compare[
    op: Int, T: Copyable & Deinitable & Comparable
](left: Column[T], right: Column[T]) raises -> Series:
    var n = _length(len(left), len(right))
    var values = List[Bool](length=n, fill=False)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        var a = 0 if len(left) == 1 else i
        var b = 0 if len(right) == 1 else i
        valid[i] = left._valid(a) and right._valid(b)
        if valid[i]:
            ref x = left._get(a)
            ref y = right._get(b)
            comptime if op == GT:
                values[i] = x > y
            elif op == LT:
                values[i] = x < y
            elif op == GE:
                values[i] = x >= y
            elif op == LE:
                values[i] = x <= y
            elif op == EQ:
                values[i] = x == y
            else:
                values[i] = x != y
    return Series("", BoolColumn(values^, valid))


def _compare_bools[
    op: Int
](left: BoolColumn, right: BoolColumn) raises -> Series:
    var n = _length(len(left), len(right))
    var values = List[Bool](length=n, fill=False)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        var a = 0 if len(left) == 1 else i
        var b = 0 if len(right) == 1 else i
        valid[i] = left._valid(a) and right._valid(b)
        if valid[i]:
            var x = Int(left._get(a))
            var y = Int(right._get(b))
            comptime if op == GT:
                values[i] = x > y
            elif op == LT:
                values[i] = x < y
            elif op == GE:
                values[i] = x >= y
            elif op == LE:
                values[i] = x <= y
            elif op == EQ:
                values[i] = x == y
            else:
                values[i] = x != y
    return Series("", BoolColumn(values^, valid))


def _fill_null_bools(left: BoolColumn, right: BoolColumn) raises -> BoolColumn:
    """Left where it is valid, else right, on bitmaps a byte (eight rows) at
    a time. Numeric `is_in` runs this once per listed value."""
    var n = _length(len(left), len(right))
    var count = (n + 7) // 8
    var values = _window_bytes(left._data[], left._offset, len(left), n, 0)
    var valid = _window_bytes(left._bits[], left._offset, len(left), n, 255)
    var other = _window_bytes(right._data[], right._offset, len(right), n, 0)
    var other_valid = _window_bytes(
        right._bits[], right._offset, len(right), n, 255
    )
    var v = values.unsafe_ptr()
    var m = valid.unsafe_ptr()
    var w = other.unsafe_ptr()
    var p = other_valid.unsafe_ptr()
    for k in range(count):
        var mine = m.unsafe_offset(k)[]
        var either = mine | p.unsafe_offset(k)[]
        v.unsafe_offset(k)[] = (
            (v.unsafe_offset(k)[] & mine) | (w.unsafe_offset(k)[] & ~mine)
        ) & either
        m.unsafe_offset(k)[] = either
    return BoolColumn(values=values^, bits=valid^, length=n)


def _validity_bytes(series: Series, n: Int) -> List[UInt8]:
    """A series' validity as an n-row bitmap, read from its column."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            ref column = series._data[Column[Scalar[D]]]
            return _window_bytes(
                column._bits[], column._offset, len(column), n, 255
            )
    if series._data.isa[BoolColumn]():
        ref bools = series._data[BoolColumn]
        return _window_bytes(bools._bits[], bools._offset, len(bools), n, 255)
    return _pack_bits(validity(series))


def _keep_nulls_bits(left: Series, right: BoolColumn) raises -> BoolColumn:
    """Right's values, null where left is null: bitmaps, a byte at a time."""
    var n = _length(len(left), len(right))
    var count = (n + 7) // 8
    var mask = _validity_bytes(left, n)
    var values = _window_bytes(right._data[], right._offset, len(right), n, 0)
    var valid = _window_bytes(right._bits[], right._offset, len(right), n, 255)
    var v = values.unsafe_ptr()
    var m = valid.unsafe_ptr()
    var l = mask.unsafe_ptr()
    for k in range(count):
        var keep = m.unsafe_offset(k)[] & l.unsafe_offset(k)[]
        m.unsafe_offset(k)[] = keep
        v.unsafe_offset(k)[] &= keep
    return BoolColumn(values=values^, bits=valid^, length=n)


def _keep_nulls_bools(mask: List[Bool], right: BoolColumn) raises -> BoolColumn:
    var n = _length(len(mask), len(right))
    var values = List[Bool](capacity=n)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        var a = 0 if len(mask) == 1 else i
        var b = 0 if len(right) == 1 else i
        values.append(right._get(b))
        valid.append(mask[a] and right._valid(b))
    return BoolColumn(values^, valid)


def _choose_bools(
    selected: List[Bool], then: BoolColumn, other: BoolColumn
) raises -> BoolColumn:
    var n = len(selected)
    var values = List[Bool](capacity=n)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        if selected[i]:
            var a = 0 if len(then) == 1 else i
            values.append(then._get(a))
            valid.append(then._valid(a))
        else:
            var b = 0 if len(other) == 1 else i
            values.append(other._get(b))
            valid.append(other._valid(b))
    return BoolColumn(values^, valid)


def _compare_strings[
    op: Int
](left: StringColumn, right: StringColumn) raises -> Series:
    """Byte-wise comparison of borrowed UTF-8 rows (code point order)."""
    # A column against a literal runs on every worker over raw bytes (#373).
    if len(right) == 1 and len(left) != 1 and right._valid(0):
        return compare_with_literal(left, op, right._get(0), False)
    if len(left) == 1 and len(right) != 1 and left._valid(0):
        return compare_with_literal(right, op, left._get(0), True)
    var n = _length(len(left), len(right))
    var values = List[Bool](length=n, fill=False)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        var a = 0 if len(left) == 1 else i
        var b = 0 if len(right) == 1 else i
        valid[i] = left._valid(a) and right._valid(b)
        if valid[i]:
            var x = left._get(a)
            var y = right._get(b)
            # StringSlice lacks a slice-to-slice >=, so derive it from <.
            comptime if op == GT:
                values[i] = y < x
            elif op == LT:
                values[i] = x < y
            elif op == GE:
                values[i] = not (x < y)
            elif op == LE:
                values[i] = not (y < x)
            elif op == EQ:
                values[i] = x == y
            else:
                values[i] = x != y
    return Series("", BoolColumn(values^, valid))


def _decimal_result(
    op: Int, left: DataType, right: DataType
) raises -> DataType:
    if is_comparison(op):
        return DataType.BOOL
    if op == ADD or op == SUB or op == MUL or op == DIV:
        return DataType.decimal(38, max(left.scale(), right.scale()))
    raise Error("decimal operation is not supported")


def _decimal_mul(a: Int128, b: Int128) raises -> Int128:
    # Factors that fit 64 bits cannot overflow 128: skip the division (#385).
    comptime limit = Int128(Int64.MAX)
    if a <= limit and a >= -limit and b <= limit and b >= -limit:
        return a * b
    if a != 0 and (b > Int128.MAX / abs(a) or b < Int128.MIN / abs(a)):
        raise Error("decimal multiplication overflow")
    return a * b


def _decimal_div(
    a: Int128, b: Int128, shift: Int, dtype: DataType
) raises -> Int128:
    var negative = (a < 0) != (b < 0)
    var numerator = -a if a < 0 else a
    var denominator = -b if b < 0 else b
    var result = check_precision(numerator / denominator, dtype)
    var remainder = numerator % denominator
    for _ in range(shift):
        result = check_precision(result * 10, dtype)
        remainder *= 10
        result += remainder / denominator
        remainder %= denominator
    # The digits past the result scale round half to even, as Polars does
    # (#342): 0.25 / 2 is 0.12 at scale 2, 0.75 / 2 is 0.38.
    result = check_precision(
        round_half_even(result, remainder, denominator), dtype
    )
    return -result if negative else result


@always_inline
def _round_div[divisor: Int](numerator: Int128) -> Int128:
    """`divide_half_even` by a power of ten known when compiling, so a
    quotient that fits 64 bits is a multiply and shift, not a division."""
    var negative = numerator < 0
    var magnitude = -numerator if negative else numerator
    if magnitude > Int128(Int64.MAX):
        return divide_half_even(numerator, Int128(divisor))
    comptime d = Int64(divisor)
    var m = Int64(magnitude)
    var q = m // d
    var r = m - q * d
    var rest = d - r
    if r > rest or (r == rest and q % 2 == 1):
        q += 1
    return Int128(-q) if negative else Int128(q)


def _decimal_dense[
    op: Int, divisor: Int, A: DType, B: DType
](
    a: Column[Scalar[A]],
    b: Column[Scalar[B]],
    n: Int,
    left_factor: Int128,
    right_factor: Int128,
    rescale: Int128,
    limit: Int128,
    dtype: DataType,
) raises -> List[Int128]:
    """Decimal ADD, SUB or MUL with no nulls on either side: values through
    pointers, no validity to build. `divisor` is the multiplication's
    rescaling power of ten when it is one the compiler can specialize (0
    otherwise, which divides by `rescale` at run time)."""
    var values = List[Int128](unsafe_uninit_length=n)
    var out = values.unsafe_ptr()
    var xs = a._ptr()
    var ys = b._ptr()
    var left_one = len(a) == 1
    var right_one = len(b) == 1
    for i in range(n):
        # decimal64 inputs are read at their width and widened per value.
        var x = Int128(xs[unsafe_offset=0 if left_one else i])
        var y = Int128(ys[unsafe_offset=0 if right_one else i])
        var value: Int128
        comptime if op == MUL:
            value = _decimal_mul(x, y)
            comptime if divisor > 1:
                value = _round_div[divisor](value)
            elif divisor == 0:
                value = divide_half_even(value, rescale)
        else:
            if left_factor != 1:
                x = _decimal_mul(x, left_factor)
            if right_factor != 1:
                y = _decimal_mul(y, right_factor)
            value = x + y if op == ADD else x - y
        out[unsafe_offset=i] = check_limit(value, limit, dtype)
    return values^


def _dense_decimal[
    op: Int, A: DType, B: DType
](
    a: Column[Scalar[A]],
    b: Column[Scalar[B]],
    left_dtype: DataType,
    right_dtype: DataType,
    result_dtype: DataType,
) raises -> Series:
    """Decimal ADD, SUB or MUL of columns without nulls, at their storage
    widths (Int64 or Int128); the result is decimal128."""
    var n = _length(len(a), len(b))
    var common = max(left_dtype.scale(), right_dtype.scale())
    var lf = pow10(common - left_dtype.scale())
    var rf = pow10(common - right_dtype.scale())
    var lim = precision_limit(result_dtype.precision())
    var shift = left_dtype.scale() + right_dtype.scale() - result_dtype.scale()
    var scale = pow10(max(shift, 0))
    var dense: List[Int128]
    comptime if op == MUL:
        if shift <= 0:
            dense = _decimal_dense[op, 1, A, B](
                a, b, n, lf, rf, scale, lim, result_dtype
            )
        elif shift == 1:
            dense = _decimal_dense[op, 10, A, B](
                a, b, n, lf, rf, scale, lim, result_dtype
            )
        elif shift == 2:
            dense = _decimal_dense[op, 100, A, B](
                a, b, n, lf, rf, scale, lim, result_dtype
            )
        elif shift == 3:
            dense = _decimal_dense[op, 1000, A, B](
                a, b, n, lf, rf, scale, lim, result_dtype
            )
        elif shift == 4:
            dense = _decimal_dense[op, 10000, A, B](
                a, b, n, lf, rf, scale, lim, result_dtype
            )
        else:
            dense = _decimal_dense[op, 0, A, B](
                a, b, n, lf, rf, scale, lim, result_dtype
            )
    else:
        dense = _decimal_dense[op, 1, A, B](
            a, b, n, lf, rf, scale, lim, result_dtype
        )
    return Series("", Column[Int128](dense^)).with_dtype(result_dtype)


def _decimal_side(
    series: Series, common: Int
) raises -> Optional[Column[Int64]]:
    """One side of a decimal comparison as Int64 at scale `common`, without
    widening: a decimal64 column already at that scale, or a one-row value
    rescaled exactly. None when that is not possible."""
    var scale = series.dtype().scale()
    if series._data.isa[Column[Int64]]():
        if scale != common:
            return None
        return series._data[Column[Int64]].copy()
    if len(series) != 1 or not series._data.isa[Column[Int128]]():
        return None
    ref column = series._data[Column[Int128]]
    if not column._valid(0):
        return Column[Int64]([Int64(0)], [False])
    var value = column._get(0) * pow10(common - scale)
    if value > Int128(Int64.MAX) or value < Int128(Int64.MIN):
        return None
    return Column[Int64]([Int64(value)])


def _decimal_binary[
    op: Int
](left: Series, right: Series, mask: List[Bool]) raises -> Series:
    comptime if is_comparison(op):
        # A comparison cannot fail on an inactive row, and its results
        # there go unread: compare every row on the unmasked paths. The
        # masked per-row path made PDS-H decimal q6 4.5 times slower once
        # filter parts ran under an AND's mask.
        if len(mask) > 0:
            return _decimal_binary[op](left, right, List[Bool]())
    var result_dtype = _decimal_result(op, left.dtype(), right.dtype())
    var narrow = (
        left._data.isa[Column[Int64]]() or right._data.isa[Column[Int64]]()
    )
    var no_nulls = left.null_count() == 0 and right.null_count() == 0
    comptime if op == ADD or op == SUB or op == MUL:
        if len(mask) == 0 and no_nulls:
            var ld = left.dtype()
            var rd = right.dtype()
            if left._data.isa[Column[Int64]]():
                if right._data.isa[Column[Int64]]():
                    return _dense_decimal[op, DType.int64, DType.int64](
                        left._data[Column[Int64]],
                        right._data[Column[Int64]],
                        ld,
                        rd,
                        result_dtype,
                    )
                return _dense_decimal[op, DType.int64, DType.int128](
                    left._data[Column[Int64]],
                    right._data[Column[Int128]],
                    ld,
                    rd,
                    result_dtype,
                )
            if right._data.isa[Column[Int64]]():
                return _dense_decimal[op, DType.int128, DType.int64](
                    left._data[Column[Int128]],
                    right._data[Column[Int64]],
                    ld,
                    rd,
                    result_dtype,
                )
            return _dense_decimal[op, DType.int128, DType.int128](
                left._data[Column[Int128]],
                right._data[Column[Int128]],
                ld,
                rd,
                result_dtype,
            )
    if narrow:
        comptime if is_comparison(op):
            # decimal64 values compare at their width when both sides are
            # at one scale (a literal rescaled exactly to the column's).
            if len(mask) == 0:
                var common = max(left.dtype().scale(), right.dtype().scale())
                var x = _decimal_side(left, common)
                var y = _decimal_side(right, common)
                if Bool(x) and Bool(y):
                    return _compare_bits[op, DType.int64](x.value(), y.value())
        return _decimal_binary[op](
            left._decimal128(), right._decimal128(), mask
        )
    ref a = left._data[Column[Int128]]
    ref b = right._data[Column[Int128]]
    var n = _length(len(a), len(b))
    var valid = List[Bool](length=n, fill=False)
    comptime predicate = is_comparison(op)
    var values = List[Int128](length=0 if predicate else n, fill=0)
    var predicates = List[Bool](length=n if predicate else 0, fill=False)
    var common_scale = max(left.dtype().scale(), right.dtype().scale())
    # Per-column constants, read once instead of per row (#385).
    var left_factor = pow10(common_scale - left.dtype().scale())
    var right_factor = pow10(common_scale - right.dtype().scale())
    var limit = precision_limit(result_dtype.precision())
    var source_scale = left.dtype().scale() + right.dtype().scale()
    var rescale = pow10(max(source_scale - result_dtype.scale(), 0))
    for i in range(n):
        var ai = 0 if len(a) == 1 else i
        var bi = 0 if len(b) == 1 else i
        valid[i] = a._valid(ai) and b._valid(bi)
        if len(mask) == n and not mask[i]:
            valid[i] = False
        if not valid[i]:
            continue
        var x = a._get(ai)
        if left_factor != 1:
            x = _decimal_mul(x, left_factor)
        var y = b._get(bi)
        if right_factor != 1:
            y = _decimal_mul(y, right_factor)
        comptime if is_comparison(op):
            if op == GT:
                predicates[i] = x > y
            elif op == LT:
                predicates[i] = x < y
            elif op == GE:
                predicates[i] = x >= y
            elif op == LE:
                predicates[i] = x <= y
            elif op == EQ:
                predicates[i] = x == y
            else:
                predicates[i] = x != y
        elif op == ADD or op == SUB:
            values[i] = check_limit(
                x + y if op == ADD else x - y, limit, result_dtype
            )
        elif op == MUL:
            var raw = _decimal_mul(a._get(ai), b._get(bi))
            if rescale != 1:
                raw = divide_half_even(raw, rescale)
            values[i] = check_limit(raw, limit, result_dtype)
        elif op == DIV:
            if b._get(bi) == 0:
                valid[i] = False
            else:
                var shift = (
                    result_dtype.scale()
                    + right.dtype().scale()
                    - left.dtype().scale()
                )
                values[i] = _decimal_div(
                    a._get(ai), b._get(bi), shift, result_dtype
                )
    comptime if predicate:
        return Series("", BoolColumn(predicates^, valid))
    else:
        return Series("", Column[Int128](values^, valid)).with_dtype(
            result_dtype
        )


def _arithmetic[
    op: Int, width: Int
](left: Series, right: Series, mask: List[Bool]) raises -> Series:
    if left.dtype().is_decimal():
        return _decimal_binary[op](left, right, mask)
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if left._data.isa[Column[Scalar[D]]]():
            comptime if is_comparison(op):
                # A mask comes only from guarded when/then branches; those
                # keep the per-row path, which clears masked rows.
                if len(mask) == 0:
                    return _compare_bits[op, D](
                        left._data[Column[Scalar[D]]],
                        right._data[Column[Scalar[D]]],
                    )
            comptime if D.is_floating_point():
                return _numeric_float[op, width, D](
                    left._data[Column[Scalar[D]]],
                    right._data[Column[Scalar[D]]],
                )
            else:
                return _numeric_int[op, D](
                    left._data[Column[Scalar[D]]],
                    right._data[Column[Scalar[D]]],
                    mask,
                )
    comptime if is_comparison(op):
        if left._data.isa[BoolColumn]():
            return _compare_bools[op](
                left._data[BoolColumn], right._data[BoolColumn]
            )
        if left._data.isa[StringColumn]():
            return _compare_strings[op](
                left._data[StringColumn], right._data[StringColumn]
            )
    raise Error("Unsupported binary kernel")


def _unary_float[
    op: Int, width: Int, D: DType
](
    input: Column[Scalar[D]], decimals: Int
) raises -> Series where D.is_floating_point():
    """Float unary kernel with contiguous SIMD loads (see _numeric_float)."""
    var n = len(input)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        valid[i] = input._valid(i)
    var values = List[Scalar[D]](length=n, fill=0)
    var main = n - n % width
    for start in range(0, main, width):
        _unary_block[op, width, D](input, start, decimals, valid, values)
    for start in range(main, n):
        _unary_block[op, 1, D](input, start, decimals, valid, values)
    return Series("", Column[Scalar[D]](values^, valid))


def _unary_block[
    op: Int, width: Int, D: DType
](
    input: Column[Scalar[D]],
    start: Int,
    decimals: Int,
    valid: List[Bool],
    mut values: List[Scalar[D]],
) where D.is_floating_point():
    var x = input._ptr().unsafe_load[width=width](start)
    var result: SIMD[D, width]
    comptime if op == NEG:
        result = -x
    elif op == ABS:
        result = abs(x)
    elif op == SQRT:
        result = sqrt(x)
    elif op == EXP:
        result = exp(x)
    elif op == LOG:
        result = log(x)
    elif op == FLOOR:
        result = floor(x)
    elif op == CEIL:
        result = ceil(x)
    else:
        result = SIMD[D, width](0)
        comptime for lane in range(width):
            result[lane] = _round_half_away(
                x[lane].cast[DType.float64](), decimals
            ).cast[D]()
    values.unsafe_ptr().unsafe_offset(start).unsafe_store(
        _mask[width](valid, start).select(result, SIMD[D, width](0))
    )


def _unary_int[
    op: Int, D: DType
](input: Column[Scalar[D]], mask: List[Bool]) raises -> Series:
    var n = len(input)
    var active = fit_mask(mask, n)
    var valid = List[Bool](length=n, fill=False)
    comptime floating = op == SQRT or op == EXP or op == LOG
    var values = List[Scalar[D]](length=0 if floating else n, fill=0)
    var floats = List[Float64](length=n if floating else 0, fill=0)
    for i in range(n):
        valid[i] = input._valid(i) and (len(active) == 0 or active[i])
        if not valid[i]:
            continue
        var x = input._get(i)
        comptime if op == NEG or op == ABS:
            comptime if D.is_signed():
                if op == ABS and x >= 0:
                    values[i] = x
                else:
                    if x == Scalar[D].MIN:
                        raise Error(
                            ("Int64" if D == DType.int64 else String(D))
                            + " "
                            + ("abs" if op == ABS else "negation")
                            + " overflow"
                        )
                    values[i] = -x
            else:
                if op == NEG and x != 0:
                    raise Error(String(D) + " negation overflow")
                values[i] = x
        elif op == SQRT:
            floats[i] = sqrt(x.cast[DType.float64]())
        elif op == EXP:
            floats[i] = exp(x.cast[DType.float64]())
        elif op == LOG:
            floats[i] = log(x.cast[DType.float64]())
        else:
            values[i] = x
    comptime if floating:
        return Series("", Column[Float64](floats^, valid))
    else:
        return Series("", Column[Scalar[D]](values^, valid))


def _math[
    op: Int, width: Int
](input: Series, integer: Int64, mask: List[Bool]) raises -> Series:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if input._data.isa[Column[Scalar[D]]]():
            comptime if D.is_floating_point():
                return _unary_float[op, width, D](
                    input._data[Column[Scalar[D]]], Int(integer)
                )
            else:
                return _unary_int[op, D](input._data[Column[Scalar[D]]], mask)
    raise Error("Unsupported unary kernel")


# --- packed-bit comparisons and Kleene logic (#327) -------------------------
#
# Comparisons and Boolean logic produce Arrow-style bitmaps directly: eight
# rows per step, one output byte of values and one of validity. Validity is
# the byte-wise AND of the operands' bitmaps, read at each window's bit
# offset, so no per-row validity branch runs and nothing is packed twice.


def _window_byte(bits: List[UInt8], bit: Int, empty: UInt8) -> UInt8:
    """Eight bits of a bitmap starting at bit `bit`, LSB first; `empty`
    when the bitmap is empty (no nulls). Bits past the end read as zero."""
    if len(bits) == 0:
        return empty
    var j = bit >> 3
    var shift = UInt8(bit & 7)
    var low = bits[j] >> shift if j < len(bits) else UInt8(0)
    if shift == 0:
        return low
    var high = bits[j + 1] << (8 - shift) if j + 1 < len(bits) else UInt8(0)
    return low | high


def _window_bytes(
    bits: List[UInt8], offset: Int, length: Int, n: Int, empty: UInt8
) -> List[UInt8]:
    """The bitmap of an n-row result aligned to row 0: a window of `length`
    rows starting at bit `offset`, or one row broadcast when length is 1."""
    var count = (n + 7) // 8
    if len(bits) == 0:
        return List[UInt8](length=count, fill=empty)
    if length == 1 and n != 1:
        var one = _bit(bits, offset)
        return List[UInt8](length=count, fill=UInt8(255) if one else UInt8(0))
    var out = List[UInt8](length=count, fill=0)
    var first = offset >> 3
    var shift = UInt8(offset & 7)
    # Bytes readable without running off the source bitmap.
    var available = len(bits) - first
    var src = bits.unsafe_ptr().unsafe_offset(first)
    var dst = out.unsafe_ptr()
    if shift == 0:
        var copied = min(count, available)
        for k in range(copied):
            dst.unsafe_offset(k)[] = src.unsafe_offset(k)[]
        return out^
    var inner = max(min(count, available - 1), 0)
    for k in range(inner):
        dst.unsafe_offset(k)[] = (src.unsafe_offset(k)[] >> shift) | (
            src.unsafe_offset(k + 1)[] << (8 - shift)
        )
    for k in range(inner, count):
        dst.unsafe_offset(k)[] = _window_byte(bits, offset + 8 * k, empty)
    return out^


@always_inline
def _compare_lanes[
    op: Int, D: DType, width: Int
](x: SIMD[D, width], y: SIMD[D, width]) -> SIMD[DType.bool, width]:
    comptime if op == GT:
        return x.gt(y)
    elif op == LT:
        return x.lt(y)
    elif op == GE:
        return x.ge(y)
    elif op == LE:
        return x.le(y)
    elif op == EQ:
        return x.eq(y)
    else:
        # SIMD ne is an ordered comparison; IEEE requires NaN != NaN.
        return ~x.eq(y)


def _compare_bits[
    op: Int, D: DType
](left: Column[Scalar[D]], right: Column[Scalar[D]]) raises -> Series:
    """A numeric comparison written straight into value and validity
    bitmaps. Null rows have a zero value bit, as before."""
    var n = _length(len(left), len(right))
    var count = (n + 7) // 8
    # Neither side has a validity bitmap: the result is all valid, kept as
    # an empty bitmap (as the logical kernels expect) rather than a full one
    # every later AND would read again. A filter's comparisons spent as long
    # filling and applying all-ones bytes as comparing.
    var all_valid = len(left._bits[]) == 0 and len(right._bits[]) == 0
    var valid = List[UInt8]() if all_valid else _window_bytes(
        left._bits[], left._offset, len(left), n, UInt8(255)
    )
    if not all_valid and len(right._bits[]) > 0:
        var other = _window_bytes(
            right._bits[], right._offset, len(right), n, UInt8(255)
        )
        for k in range(count):
            valid[k] &= other[k]
    var values = List[UInt8](length=count, fill=0)
    if n == 0:
        return Series("", BoolColumn(values=values^, bits=valid^, length=0))
    var weights = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
    var left_one = len(left) == 1
    var right_one = len(right) == 1
    var x_splat = SIMD[D, 8](left._get(0))
    var y_splat = SIMD[D, 8](right._get(0))
    var xs = left._ptr()
    var ys = right._ptr()
    var full = n // 8
    var out = values.unsafe_ptr()
    if all_valid:
        for k in range(full):
            var x = x_splat if left_one else xs.unsafe_load[width=8](8 * k)
            var y = y_splat if right_one else ys.unsafe_load[width=8](8 * k)
            var hits = _compare_lanes[op, D, 8](x, y).cast[DType.uint8]()
            out[unsafe_offset=k] = (hits * weights).reduce_add()
    else:
        for k in range(full):
            var x = x_splat if left_one else xs.unsafe_load[width=8](8 * k)
            var y = y_splat if right_one else ys.unsafe_load[width=8](8 * k)
            var hits = _compare_lanes[op, D, 8](x, y).cast[DType.uint8]()
            out[unsafe_offset=k] = (hits * weights).reduce_add() & valid[k]
    if full < count:
        var byte = UInt8(0)
        for i in range(8 * full, n):
            var x = left._get(0 if left_one else i)
            var y = right._get(0 if right_one else i)
            if _compare_lanes[op, D, 1](x, y)[0]:
                byte |= UInt8(1) << UInt8(i - 8 * full)
        values[full] = byte if all_valid else byte & valid[full]
    return Series("", BoolColumn(values=values^, bits=valid^, length=n))


def _logical_bits[
    op: Int
](left: BoolColumn, right: BoolColumn) raises -> Series:
    """Kleene AND, OR and XOR on bitmaps, a byte (eight rows) at a time.

    With values v and validity m: AND is valid where both are, or where
    either is a valid false; OR where both are, or where either is a valid
    true; XOR only where both are. Value bits are cleared where the result
    is null. Without nulls on either side this is plain bitwise logic.
    """
    var n = _length(len(left), len(right))
    var count = (n + 7) // 8
    var va = _window_bytes(left._data[], left._offset, len(left), n, 0)
    var vb = _window_bytes(right._data[], right._offset, len(right), n, 0)
    var a = va.unsafe_ptr()
    var b = vb.unsafe_ptr()
    if len(left._bits[]) == 0 and len(right._bits[]) == 0:
        for k in range(count):
            comptime if op == AND:
                a.unsafe_offset(k)[] &= b.unsafe_offset(k)[]
            elif op == OR:
                a.unsafe_offset(k)[] |= b.unsafe_offset(k)[]
            else:
                a.unsafe_offset(k)[] ^= b.unsafe_offset(k)[]
        return Series(
            "",
            BoolColumn(
                values=va^, bits=List[UInt8](length=count, fill=255), length=n
            ),
        )
    var ma = _window_bytes(left._bits[], left._offset, len(left), n, 255)
    var mb = _window_bytes(right._bits[], right._offset, len(right), n, 255)
    var p = ma.unsafe_ptr()
    var q = mb.unsafe_ptr()
    for k in range(count):
        var x = a.unsafe_offset(k)[] & p.unsafe_offset(k)[]
        var y = b.unsafe_offset(k)[] & q.unsafe_offset(k)[]
        var m: UInt8
        comptime if op == AND:
            m = (
                (p.unsafe_offset(k)[] & q.unsafe_offset(k)[])
                | (p.unsafe_offset(k)[] & ~x)
                | (q.unsafe_offset(k)[] & ~y)
            )
            a.unsafe_offset(k)[] = x & y & m
        elif op == OR:
            m = (p.unsafe_offset(k)[] & q.unsafe_offset(k)[]) | x | y
            a.unsafe_offset(k)[] = (x | y) & m
        else:
            m = p.unsafe_offset(k)[] & q.unsafe_offset(k)[]
            a.unsafe_offset(k)[] = (x ^ y) & m
        p.unsafe_offset(k)[] = m
    return Series("", BoolColumn(values=va^, bits=ma^, length=n))


def _not_bits(column: BoolColumn) -> Series:
    """Kleene NOT on bitmaps: nulls stay null, value bits cleared there."""
    var n = len(column)
    var count = (n + 7) // 8
    var values = _window_bytes(column._data[], column._offset, n, n, 0)
    var valid = _window_bytes(column._bits[], column._offset, n, n, 255)
    for k in range(count):
        values[k] = ~values[k] & valid[k]
    if count > 0 and n % 8 != 0:
        values[count - 1] &= UInt8((1 << (n % 8)) - 1)
    return Series("", BoolColumn(values=values^, bits=valid^, length=n))


def _logical[op: Int](left: BoolColumn, right: BoolColumn) raises -> Series:
    """Kleene logic: a dominant operand decides even when the other is null.
    The scalar reference that _logical_bits replaced; kept for tests."""
    var n = _length(len(left), len(right))
    var values = List[Bool](length=n, fill=False)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        var a = 0 if len(left) == 1 else i
        var b = 0 if len(right) == 1 else i
        var p = left._valid(a)
        var q = right._valid(b)
        var x = p and left._get(a)
        var y = q and right._get(b)
        comptime if op == AND:
            if (p and not x) or (q and not y):
                valid[i] = True
            elif p and q:
                valid[i] = True
                values[i] = True
        elif op == OR:
            if x or y:
                valid[i] = True
                values[i] = True
            elif p and q:
                valid[i] = True
        else:
            valid[i] = p and q
            values[i] = x != y
    return Series("", BoolColumn(values^, valid))


def _fill_null[
    T: Copyable & Deinitable
](left: Column[T], right: Column[T]) raises -> Column[T]:
    var n = _length(len(left), len(right))
    var values = List[T](capacity=n)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        var a = 0 if len(left) == 1 else i
        var b = 0 if len(right) == 1 else i
        if left._valid(a):
            values.append(left._get(a).copy())
            valid.append(True)
        else:
            values.append(right._get(b).copy())
            valid.append(right._valid(b))
    return Column[T](values^, valid)


def _fill_nan[
    D: DType
](left: Column[Scalar[D]], right: Column[Scalar[D]]) raises -> Series:
    var n = _length(len(left), len(right))
    var values = List[Scalar[D]](length=n, fill=0)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        var a = 0 if len(left) == 1 else i
        var b = 0 if len(right) == 1 else i
        if left._valid(a) and isnan(left._get(a)):
            values[i] = right._get(b)
            valid[i] = right._valid(b)
        else:
            values[i] = left._get(a)
            valid[i] = left._valid(a)
    return Series("", Column[Scalar[D]](values^, valid))


def validity(series: Series) -> List[Bool]:
    if series.is_chunked():
        var result = List[Bool](capacity=len(series))
        for part in series.chunks():
            result.extend(Span(validity(part)))
        return result^
    var n = len(series)
    var valid = List[Bool](capacity=n)
    if series.dtype().is_decimal():
        for i in range(n):
            valid.append(series._decimal_valid(i))
        return valid^
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            ref column = series._data[Column[Scalar[D]]]
            for i in range(n):
                valid.append(column._valid(i))
            return valid^
    if series._data.isa[BoolColumn]():
        for i in range(n):
            valid.append(series._data[BoolColumn]._valid(i))
    elif series._data.isa[ListColumn]():
        return series._data[ListColumn].validity()
    elif series._data.isa[StructColumn]():
        return series._data[StructColumn].validity()
    else:
        for i in range(n):
            valid.append(series._data[StringColumn]._valid(i))
    return valid^


def _keep_nulls[
    T: Copyable & Deinitable
](mask: List[Bool], right: Column[T]) raises -> Column[T]:
    var n = _length(len(mask), len(right))
    var values = List[T](capacity=n)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        var a = 0 if len(mask) == 1 else i
        var b = 0 if len(right) == 1 else i
        values.append(right._get(b).copy())
        valid.append(mask[a] and right._valid(b))
    return Column[T](values^, valid)


def binary[
    op: Int, width: Int = 4
](
    left: Series, right: Series, mask: List[Bool] = List[Bool]()
) raises -> Series:
    if left.is_chunked() or right.is_chunked():
        return binary[op, width](left.rechunk(), right.rechunk(), mask)
    comptime if op == FILL_NULL:
        if left.dtype().is_decimal() and right.dtype().is_decimal():
            # Left where valid, else right, at the operands' common type
            # and storage width.
            var pair = _decimal_pair(left, right, "fill_null")
            ref a = pair[0]
            ref b = pair[1]
            if a._data.isa[Column[Int128]]():
                return Series(
                    "",
                    _fill_null(
                        a._data[Column[Int128]], b._data[Column[Int128]]
                    ),
                ).with_dtype(a.dtype())
            if a._data.isa[Column[Int64]]():
                return Series(
                    "",
                    _fill_null(a._data[Column[Int64]], b._data[Column[Int64]]),
                ).with_dtype(a.dtype())
            return Series(
                "", _fill_null(a._data[Column[Int32]], b._data[Column[Int32]])
            ).with_dtype(a.dtype())
    # decimal64 operands go to the decimal kernels at their width, which
    # widen what they do not cover; decimal32 operands are widened to 64
    # bits here, where the typed compares and dense arithmetic apply.
    if (left.dtype().is_decimal() and left.dtype().decimal_width() == 32) or (
        right.dtype().is_decimal() and right.dtype().decimal_width() == 32
    ):
        return binary[op, width](left._decimal64(), right._decimal64(), mask)
    comptime if is_logical(op):
        return _logical_bits[op](
            left._data[BoolColumn], right._data[BoolColumn]
        )
    elif op == FILL_NAN:
        if left._data.isa[Column[Float32]]():
            return _fill_nan(
                left._data[Column[Float32]], right._data[Column[Float32]]
            )
        return _fill_nan(
            left._data[Column[Float64]], right._data[Column[Float64]]
        )
    elif op == FILL_NULL:
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if left._data.isa[Column[Scalar[D]]]():
                return Series(
                    "",
                    _fill_null(
                        left._data[Column[Scalar[D]]],
                        right._data[Column[Scalar[D]]],
                    ),
                )
        if left._data.isa[BoolColumn]():
            return Series(
                "",
                _fill_null_bools(
                    left._data[BoolColumn], right._data[BoolColumn]
                ),
            )
        ref a = left._data[StringColumn]
        ref b = right._data[StringColumn]
        var n = _length(len(a), len(b))
        var out = StringBuilder(n)
        for i in range(n):
            var x = 0 if len(a) == 1 else i
            if a._valid(x):
                out._append_row(a, x)
            else:
                out._append_row(b, 0 if len(b) == 1 else i)
        return Series("", out^.finish())
    elif op == KEEP_NULLS:
        if right._data.isa[BoolColumn]() and (
            len(left) == 1 or len(right) == 1 or len(left) == len(right)
        ):
            return Series("", _keep_nulls_bits(left, right._data[BoolColumn]))
        var mask = validity(left)
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if right._data.isa[Column[Scalar[D]]]():
                return Series(
                    "", _keep_nulls(mask, right._data[Column[Scalar[D]]])
                )
        if right._data.isa[BoolColumn]():
            return Series("", _keep_nulls_bools(mask, right._data[BoolColumn]))
        ref column = right._data[StringColumn]
        var n = _length(len(mask), len(column))
        var out = StringBuilder(n)
        for i in range(n):
            var b = 0 if len(column) == 1 else i
            if mask[0 if len(mask) == 1 else i]:
                out._append_row(column, b)
            else:
                out.append_null()
        return Series("", out^.finish())
    else:
        return _arithmetic[op, width](left, right, mask)


def _float_predicate[
    op: Int, D: DType
](input: Column[Scalar[D]]) raises -> Series:
    var n = len(input)
    var values = List[Bool](length=n, fill=False)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        valid[i] = input._valid(i)
        if valid[i]:
            var x = input._get(i)
            comptime if op == IS_NAN:
                values[i] = isnan(x)
            elif op == IS_NOT_NAN:
                values[i] = not isnan(x)
            elif op == IS_FINITE:
                values[i] = not isnan(x) and not isinf(x)
            else:
                values[i] = isinf(x)
    return Series("", BoolColumn(values^, valid))


def unary[
    op: Int, width: Int = 4
](
    input: Series, integer: Int64, mask: List[Bool] = List[Bool]()
) raises -> Series:
    if input.is_chunked():
        return unary[op, width](input.rechunk(), integer, mask)
    if input.dtype().is_decimal() and input.dtype().decimal_width() != 128:
        return unary[op, width](input._decimal128(), integer, mask)
    comptime if op == IS_NULL or op == IS_NOT_NULL:
        var valid = validity(input)
        var values = List[Bool](capacity=len(valid))
        for v in valid:
            values.append(v if op == IS_NOT_NULL else not v)
        return Series("", BoolColumn(values^))
    elif op == NOT:
        return _not_bits(input._data[BoolColumn])
    elif op >= IS_NAN and op <= IS_INFINITE:
        if input._data.isa[Column[Float32]]():
            return _float_predicate[op](input._data[Column[Float32]])
        return _float_predicate[op](input._data[Column[Float64]])
    else:
        return _math[op, width](input, integer, mask)


def _choose[
    T: Copyable & Deinitable
](selected: List[Bool], then: Column[T], other: Column[T]) raises -> Column[T]:
    var n = len(selected)
    var values = List[T](capacity=n)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        if selected[i]:
            var a = 0 if len(then) == 1 else i
            values.append(then._get(a).copy())
            valid.append(then._valid(a))
        else:
            var b = 0 if len(other) == 1 else i
            values.append(other._get(b).copy())
            valid.append(other._valid(b))
    return Column[T](values^, valid)


def _choose_numeric[
    D: DType
](
    selected: List[Bool], then: Column[Scalar[D]], other: Column[Scalar[D]]
) raises -> Column[Scalar[D]]:
    """`_choose` for a numeric type: a SIMD select over the selection
    bytes, either branch broadcast when it holds one value, and a validity
    bitmap only when a branch has nulls (TPC-DS q9 selects 44M values
    this way; the element-wise version cost 4 ns each)."""
    comptime width = 16
    var n = len(selected)
    var values = List[Scalar[D]](unsafe_uninit_length=n)
    var out = values.unsafe_ptr()
    var flags = selected.unsafe_ptr().unsafe_bitcast[UInt8]()
    var then_values = then._ptr()
    var other_values = other._ptr()
    var then_scalar = len(then) == 1
    var other_scalar = len(other) == 1
    var then_fill = SIMD[D, width](then_values[unsafe_offset=0]) if len(
        then
    ) > 0 else SIMD[D, width](0)
    var other_fill = SIMD[D, width](other_values[unsafe_offset=0]) if len(
        other
    ) > 0 else SIMD[D, width](0)
    var i = 0
    while i + width <= n:
        var take = (
            flags.unsafe_offset(i)
            .unsafe_load[width=width]()
            .ne(SIMD[DType.uint8, width](0))
        )
        var a = then_fill if then_scalar else then_values.unsafe_offset(
            i
        ).unsafe_load[width=width]()
        var b = other_fill if other_scalar else other_values.unsafe_offset(
            i
        ).unsafe_load[width=width]()
        out.unsafe_offset(i).unsafe_store[width=width](take.select(a, b))
        i += width
    while i < n:
        if flags[unsafe_offset=i] != 0:
            out[unsafe_offset=i] = then_values[
                unsafe_offset=0 if then_scalar else i
            ]
        else:
            out[unsafe_offset=i] = other_values[
                unsafe_offset=0 if other_scalar else i
            ]
        i += 1
    var then_nulls = then.null_count() > 0
    var other_nulls = other.null_count() > 0
    if not then_nulls and not other_nulls:
        return Column[Scalar[D]](values^)
    var bits = List[UInt8](length=(n + 7) // 8, fill=0)
    var packed = bits.unsafe_ptr()
    for row in range(n):
        var valid: Bool
        if flags[unsafe_offset=row] != 0:
            valid = not then_nulls or then._valid(0 if then_scalar else row)
        else:
            valid = not other_nulls or other._valid(0 if other_scalar else row)
        if valid:
            packed[unsafe_offset=row >> 3] |= UInt8(1) << UInt8(row & 7)
    return Column[Scalar[D]](values=values^, bits=bits^)


def _decimal_pair(
    left: Series, right: Series, what: String
) raises -> Tuple[Series, Series]:
    """Two decimal inputs at their common type (`common_decimal`), so one
    typed loop can take a value from either."""
    var target = common_decimal(left.dtype(), right.dtype(), what)
    if left.dtype() == right.dtype():
        return (left.copy(), right.copy())
    return (
        left._decimal128().with_dtype(target),
        right._decimal128().with_dtype(target),
    )


def choose(selected: List[Bool], then: Series, other: Series) raises -> Series:
    """Row-wise pick between branch results, broadcasting scalar branches."""
    if then.is_chunked() or other.is_chunked():
        return choose(selected, then.rechunk(), other.rechunk())
    if then.dtype().is_decimal() and other.dtype().is_decimal():
        # Decimals are stored as Int32, Int64 or Int128 by declared width.
        # Pick at the branches' common type and keep its tag.
        var pair = _decimal_pair(then, other, "when/then/otherwise")
        ref a = pair[0]
        ref b = pair[1]
        if a._data.isa[Column[Int128]]():
            return Series(
                "",
                _choose(
                    selected, a._data[Column[Int128]], b._data[Column[Int128]]
                ),
            ).with_dtype(a.dtype())
        if a._data.isa[Column[Int64]]():
            return Series(
                "",
                _choose(
                    selected, a._data[Column[Int64]], b._data[Column[Int64]]
                ),
            ).with_dtype(a.dtype())
        return Series(
            "",
            _choose(selected, a._data[Column[Int32]], b._data[Column[Int32]]),
        ).with_dtype(a.dtype())
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if then._data.isa[Column[Scalar[D]]]():
            return Series(
                "",
                _choose_numeric[D](
                    selected,
                    then._data[Column[Scalar[D]]],
                    other._data[Column[Scalar[D]]],
                ),
            )
    if then._data.isa[BoolColumn]():
        return Series(
            "",
            _choose_bools(
                selected, then._data[BoolColumn], other._data[BoolColumn]
            ),
        )
    ref a = then._data[StringColumn]
    ref b = other._data[StringColumn]
    var out = StringBuilder(len(selected))
    for i in range(len(selected)):
        if selected[i]:
            out._append_row(a, 0 if len(a) == 1 else i)
        else:
            out._append_row(b, 0 if len(b) == 1 else i)
    return Series("", out^.finish())


def integer_is_in(input: Series, text: String) raises -> Series:
    """Whether each row of an integer column equals one of the values
    `text` lists (decimal, comma separated); null rows stay null. One pass
    over the rows, the values compared in registers."""
    var column_input = input.rechunk() if input.is_chunked() else input.copy()
    var parts = text.split(",")
    var wanted = List[Int64](capacity=len(parts))
    for part in parts:
        wanted.append(Int64(atol(part)))
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        comptime if D.is_integral():
            if column_input._data.isa[Column[Scalar[D]]]():
                ref column = column_input._data[Column[Scalar[D]]]
                var n = len(column)
                var targets = List[Scalar[D]](capacity=len(wanted))
                for w in wanted:
                    targets.append(Scalar[D](w))
                var count = len(targets)
                var table = targets.unsafe_ptr()
                var packed = List[UInt8](length=(n + 7) // 8, fill=0)
                var out = packed.unsafe_ptr()
                var xs = column._ptr()
                var weights = SIMD[DType.uint8, 8](1, 2, 4, 8, 16, 32, 64, 128)
                var full = n // 8
                for k in range(full):
                    var x = xs.unsafe_load[width=8](8 * k)
                    var hits = SIMD[DType.bool, 8](fill=False)
                    for t in range(count):
                        hits = hits | x.eq(SIMD[D, 8](table.unsafe_offset(t)[]))
                    out[unsafe_offset=k] = (
                        hits.cast[DType.uint8]() * weights
                    ).reduce_add()
                if full * 8 < n:
                    var byte: UInt8 = 0
                    for i in range(full * 8, n):
                        var x = xs.unsafe_offset(i)[]
                        for t in range(count):
                            if x == table.unsafe_offset(t)[]:
                                byte |= UInt8(1) << UInt8(i - full * 8)
                                break
                    out[unsafe_offset=full] = byte
                var bits = List[UInt8]()
                if column.null_count() > 0:
                    bits = _copy_validity(column._bits[], column._offset, n)
                return Series(
                    input.name(),
                    BoolColumn(values=packed^, bits=bits^, length=n),
                )
    raise Error("is_in over integers requires an integer column")
