"""Batch kernels: operation/dtype dispatch occurs outside element loops."""
from std.math import sqrt, exp, log, floor, ceil, pow, isinf, isnan
from .nested_column import ListColumn, StructColumn
from .bool_column import BoolColumn
from .column import Column
from .dtype import NUMERIC_DTYPES
from .string_column import StringColumn, StringBuilder
from .series import Series
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
        var result: SIMD[DType.bool, width]
        comptime if op == GT:
            result = x.gt(y)
        elif op == LT:
            result = x.lt(y)
        elif op == GE:
            result = x.ge(y)
        elif op == LE:
            result = x.le(y)
        elif op == EQ:
            result = x.eq(y)
        else:
            # SIMD ne is an ordered comparison; IEEE requires NaN != NaN.
            result = ~x.eq(y)
        var mask = _mask[width](valid, start)
        predicates.unsafe_ptr().unsafe_offset(start).unsafe_bitcast[
            Scalar[DType.bool]
        ]().unsafe_store(result & mask)
    else:
        var result: SIMD[D, width]
        comptime if op == ADD:
            result = x + y
        elif op == SUB:
            result = x - y
        elif op == MUL:
            result = x * y
        elif op == DIV:
            result = x / y
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


def _numeric_int[
    op: Int, D: DType
](
    left: Column[Scalar[D]], right: Column[Scalar[D]], mask: List[Bool]
) raises -> Series:
    var n = _length(len(left), len(right))
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
    var n = _length(len(left), len(right))
    var values = List[Bool](capacity=n)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        var a = 0 if len(left) == 1 else i
        var b = 0 if len(right) == 1 else i
        if left._valid(a):
            values.append(left._get(a))
            valid.append(True)
        else:
            values.append(right._get(b))
            valid.append(right._valid(b))
    return BoolColumn(values^, valid)


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


def _arithmetic[
    op: Int, width: Int
](left: Series, right: Series, mask: List[Bool]) raises -> Series:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if left._data.isa[Column[Scalar[D]]]():
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


def _logical[op: Int](left: BoolColumn, right: BoolColumn) raises -> Series:
    """Kleene logic: a dominant operand decides even when the other is null."""
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
    comptime if is_logical(op):
        return _logical[op](left._data[BoolColumn], right._data[BoolColumn])
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
    comptime if op == IS_NULL or op == IS_NOT_NULL:
        var valid = validity(input)
        var values = List[Bool](capacity=len(valid))
        for v in valid:
            values.append(v if op == IS_NOT_NULL else not v)
        return Series("", BoolColumn(values^))
    elif op == NOT:
        ref column = input._data[BoolColumn]
        var values = List[Bool](capacity=len(column))
        var valid = List[Bool](capacity=len(column))
        for i in range(len(column)):
            valid.append(column._valid(i))
            values.append(column._valid(i) and not column._get(i))
        return Series("", BoolColumn(values^, valid))
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


def choose(selected: List[Bool], then: Series, other: Series) raises -> Series:
    """Row-wise pick between branch results, broadcasting scalar branches."""
    if then.is_chunked() or other.is_chunked():
        return choose(selected, then.rechunk(), other.rechunk())
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if then._data.isa[Column[Scalar[D]]]():
            return Series(
                "",
                _choose(
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
