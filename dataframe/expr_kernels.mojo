"""Batch kernels: operation/dtype dispatch occurs outside element loops."""
from std.math import sqrt, exp, log, floor, ceil, pow, isinf, isnan
from .column import Column
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


def _float_scalar[op: Int](x: Float64, y: Float64) -> Float64:
    comptime if op == FLOORDIV:
        return floor(x / y)
    elif op == MOD:
        if y == 0 or isnan(x) or isnan(y) or isinf(x):
            return Float64(0) / Float64(0)
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


def _numeric_float[
    op: Int, width: Int
](left: Column[Float64], right: Column[Float64]) raises -> Series:
    var n = _length(len(left), len(right))
    var valid = List[Bool](length=n, fill=False)
    comptime predicate = is_comparison(op)
    var values = List[Float64](length=0 if predicate else n, fill=0)
    var predicates = List[Bool](length=n if predicate else 0, fill=False)
    # Gather/load valid lanes into vectors. A future buffer-view layer can
    # replace these lane loads without altering the IR or public API.
    for start in range(0, n, width):
        var x = SIMD[DType.float64, width](0)
        var y = SIMD[DType.float64, width](0)
        comptime for lane in range(width):
            var i = start + lane
            if i < n:
                var a = 0 if len(left) == 1 else i
                var b = 0 if len(right) == 1 else i
                valid[i] = left._valid(a) and right._valid(b)
                if valid[i]:
                    x[lane] = left._get(a)
                    y[lane] = right._get(b)
        comptime if predicate:
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
            comptime for lane in range(width):
                if start + lane < n:
                    predicates[start + lane] = result[lane]
        else:
            var result: SIMD[DType.float64, width]
            comptime if op == ADD:
                result = x + y
            elif op == SUB:
                result = x - y
            elif op == MUL:
                result = x * y
            elif op == DIV:
                result = x / y
            else:
                result = SIMD[DType.float64, width](0)
                comptime for lane in range(width):
                    result[lane] = _float_scalar[op](x[lane], y[lane])
            comptime for lane in range(width):
                if start + lane < n:
                    values[start + lane] = result[lane]
    comptime if predicate:
        return Series("", Column[Bool](predicates^, valid))
    else:
        return Series("", Column[Float64](values^, valid))


def _numeric_int[
    op: Int
](left: Column[Int64], right: Column[Int64], mask: List[Bool]) raises -> Series:
    var n = _length(len(left), len(right))
    var active = fit_mask(mask, n)
    var valid = List[Bool](length=n, fill=False)
    comptime predicate = is_comparison(op)
    comptime floating = op == DIV
    var values = List[Int64](length=0 if predicate or floating else n, fill=0)
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
        comptime if op == ADD:
            values[i] = checked_add(x, y)
        elif op == SUB:
            values[i] = _checked_sub(x, y)
        elif op == MUL:
            values[i] = _checked_mul(x, y)
        elif op == DIV:
            floats[i] = Float64(x) / Float64(y)
        elif op == FLOORDIV:
            if y == 0:
                valid[i] = False
            elif y == -1:
                if x == INT64_MIN:
                    raise Error("Int64 floor division overflow")
                values[i] = -x
            else:
                values[i] = x // y
        elif op == MOD:
            if y == 0:
                valid[i] = False
            elif y == -1:
                values[i] = 0
            else:
                values[i] = x % y
        elif op == POW:
            values[i] = _checked_pow(x, y)
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
        return Series("", Column[Bool](predicates^, valid))
    elif floating:
        return Series("", Column[Float64](floats^, valid))
    else:
        return Series("", Column[Int64](values^, valid))


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
    return Series("", Column[Bool](values^, valid))


def _arithmetic[
    op: Int, width: Int
](left: Series, right: Series, mask: List[Bool]) raises -> Series:
    if left._data.isa[Column[Float64]]():
        return _numeric_float[op, width](
            left._data[Column[Float64]], right._data[Column[Float64]]
        )
    if left._data.isa[Column[Int64]]():
        return _numeric_int[op](
            left._data[Column[Int64]], right._data[Column[Int64]], mask
        )
    comptime if is_comparison(op):
        if left._data.isa[Column[Bool]]():
            return _compare[op](
                left._data[Column[Bool]], right._data[Column[Bool]]
            )
        if left._data.isa[Column[String]]():
            return _compare[op](
                left._data[Column[String]], right._data[Column[String]]
            )
    raise Error("Unsupported binary kernel")


def _unary_float[
    op: Int, width: Int
](input: Column[Float64], decimals: Int) raises -> Series:
    var n = len(input)
    var valid = List[Bool](length=n, fill=False)
    var values = List[Float64](length=n, fill=0)
    for start in range(0, n, width):
        var x = SIMD[DType.float64, width](0)
        comptime for lane in range(width):
            var i = start + lane
            if i < n:
                valid[i] = input._valid(i)
                if valid[i]:
                    x[lane] = input._get(i)
        var result: SIMD[DType.float64, width]
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
            result = SIMD[DType.float64, width](0)
            comptime for lane in range(width):
                result[lane] = _round_half_away(x[lane], decimals)
        comptime for lane in range(width):
            if start + lane < n:
                values[start + lane] = result[lane]
    return Series("", Column[Float64](values^, valid))


def _unary_int[
    op: Int
](input: Column[Int64], mask: List[Bool]) raises -> Series:
    var n = len(input)
    var active = fit_mask(mask, n)
    var valid = List[Bool](length=n, fill=False)
    comptime floating = op == SQRT or op == EXP or op == LOG
    var values = List[Int64](length=0 if floating else n, fill=0)
    var floats = List[Float64](length=n if floating else 0, fill=0)
    for i in range(n):
        valid[i] = input._valid(i) and (len(active) == 0 or active[i])
        if not valid[i]:
            continue
        var x = input._get(i)
        comptime if op == NEG or op == ABS:
            if op == ABS and x >= 0:
                values[i] = x
            else:
                if x == INT64_MIN:
                    raise Error(
                        "Int64 "
                        + ("abs" if op == ABS else "negation")
                        + " overflow"
                    )
                values[i] = -x
        elif op == SQRT:
            floats[i] = sqrt(Float64(x))
        elif op == EXP:
            floats[i] = exp(Float64(x))
        elif op == LOG:
            floats[i] = log(Float64(x))
        else:
            values[i] = x
    comptime if floating:
        return Series("", Column[Float64](floats^, valid))
    else:
        return Series("", Column[Int64](values^, valid))


def _math[
    op: Int, width: Int
](input: Series, integer: Int64, mask: List[Bool]) raises -> Series:
    if input._data.isa[Column[Float64]]():
        return _unary_float[op, width](
            input._data[Column[Float64]], Int(integer)
        )
    if input._data.isa[Column[Int64]]():
        return _unary_int[op](input._data[Column[Int64]], mask)
    raise Error("Unsupported unary kernel")


def _logical[op: Int](left: Column[Bool], right: Column[Bool]) raises -> Series:
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
    return Series("", Column[Bool](values^, valid))


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


def _fill_nan(left: Column[Float64], right: Column[Float64]) raises -> Series:
    var n = _length(len(left), len(right))
    var values = List[Float64](length=n, fill=0)
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
    return Series("", Column[Float64](values^, valid))


def validity(series: Series) -> List[Bool]:
    var n = len(series)
    var valid = List[Bool](capacity=n)
    if series._data.isa[Column[Int64]]():
        for i in range(n):
            valid.append(series._data[Column[Int64]]._valid(i))
    elif series._data.isa[Column[Float64]]():
        for i in range(n):
            valid.append(series._data[Column[Float64]]._valid(i))
    elif series._data.isa[Column[Bool]]():
        for i in range(n):
            valid.append(series._data[Column[Bool]]._valid(i))
    else:
        for i in range(n):
            valid.append(series._data[Column[String]]._valid(i))
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
    comptime if is_logical(op):
        return _logical[op](left._data[Column[Bool]], right._data[Column[Bool]])
    elif op == FILL_NAN:
        return _fill_nan(
            left._data[Column[Float64]], right._data[Column[Float64]]
        )
    elif op == FILL_NULL:
        if left._data.isa[Column[Int64]]():
            return Series(
                "",
                _fill_null(
                    left._data[Column[Int64]], right._data[Column[Int64]]
                ),
            )
        if left._data.isa[Column[Float64]]():
            return Series(
                "",
                _fill_null(
                    left._data[Column[Float64]], right._data[Column[Float64]]
                ),
            )
        if left._data.isa[Column[Bool]]():
            return Series(
                "",
                _fill_null(left._data[Column[Bool]], right._data[Column[Bool]]),
            )
        return Series(
            "",
            _fill_null(left._data[Column[String]], right._data[Column[String]]),
        )
    elif op == KEEP_NULLS:
        var mask = validity(left)
        if right._data.isa[Column[Int64]]():
            return Series("", _keep_nulls(mask, right._data[Column[Int64]]))
        if right._data.isa[Column[Float64]]():
            return Series("", _keep_nulls(mask, right._data[Column[Float64]]))
        if right._data.isa[Column[Bool]]():
            return Series("", _keep_nulls(mask, right._data[Column[Bool]]))
        return Series("", _keep_nulls(mask, right._data[Column[String]]))
    else:
        return _arithmetic[op, width](left, right, mask)


def _float_predicate[op: Int](input: Column[Float64]) raises -> Series:
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
    return Series("", Column[Bool](values^, valid))


def unary[
    op: Int, width: Int = 4
](
    input: Series, integer: Int64, mask: List[Bool] = List[Bool]()
) raises -> Series:
    comptime if op == IS_NULL or op == IS_NOT_NULL:
        var valid = validity(input)
        var values = List[Bool](capacity=len(valid))
        for v in valid:
            values.append(v if op == IS_NOT_NULL else not v)
        return Series("", Column[Bool](values^))
    elif op == NOT:
        ref column = input._data[Column[Bool]]
        var values = List[Bool](capacity=len(column))
        var valid = List[Bool](capacity=len(column))
        for i in range(len(column)):
            valid.append(column._valid(i))
            values.append(column._valid(i) and not column._get(i))
        return Series("", Column[Bool](values^, valid))
    elif op >= IS_NAN and op <= IS_INFINITE:
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
    if then._data.isa[Column[Int64]]():
        return Series(
            "",
            _choose(
                selected, then._data[Column[Int64]], other._data[Column[Int64]]
            ),
        )
    if then._data.isa[Column[Float64]]():
        return Series(
            "",
            _choose(
                selected,
                then._data[Column[Float64]],
                other._data[Column[Float64]],
            ),
        )
    if then._data.isa[Column[Bool]]():
        return Series(
            "",
            _choose(
                selected, then._data[Column[Bool]], other._data[Column[Bool]]
            ),
        )
    return Series(
        "",
        _choose(
            selected, then._data[Column[String]], other._data[Column[String]]
        ),
    )
