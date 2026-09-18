"""Batch kernels: operation/dtype dispatch occurs outside element loops."""
from .column import Column
from .series import Series
from .expr import ADD, SUB, MUL, GT, EQ
from .kernels import checked_add


def _checked_sub(a: Int64, b: Int64) raises -> Int64:
    comptime high = Int64(9223372036854775807)
    comptime low = Int64(-9223372036854775807) - 1
    if (b > 0 and a < low + b) or (b < 0 and a > high + b):
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


def _length(left: Int, right: Int) raises -> Int:
    if left != right and left != 1 and right != 1:
        raise Error("Incompatible expression lengths")
    if left == 0 or right == 0:
        return 0
    return max(left, right)


def _numeric_float[
    op: Int, width: Int
](left: Column[Float64], right: Column[Float64]) raises -> Series:
    var n = _length(len(left), len(right))
    var valid = List[Bool](length=n, fill=False)
    var values = List[Float64](length=n, fill=0)
    var predicates = List[Bool](length=n, fill=False)
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
                    x[lane] = left._values[a]
                    y[lane] = right._values[b]
        comptime if op == GT or op == EQ:
            var result = x.gt(y) if op == GT else x.eq(y)
            comptime for lane in range(width):
                if start + lane < n:
                    predicates[start + lane] = result[lane]
        else:
            var result = x + y
            comptime if op == SUB:
                result = x - y
            elif op == MUL:
                result = x * y
            comptime for lane in range(width):
                if start + lane < n:
                    values[start + lane] = result[lane]
    comptime if op == GT or op == EQ:
        return Series("", Column[Bool](predicates^, valid))
    else:
        return Series("", Column[Float64](values^, valid))


def _numeric_int[
    op: Int
](left: Column[Int64], right: Column[Int64]) raises -> Series:
    var n = _length(len(left), len(right))
    var valid = List[Bool](length=n, fill=False)
    var values = List[Int64](length=n, fill=0)
    var predicates = List[Bool](length=n, fill=False)
    # Checked integer arithmetic stays scalar until a vector overflow path
    # has equivalent semantics. No null payload enters arithmetic.
    for i in range(n):
        var a = 0 if len(left) == 1 else i
        var b = 0 if len(right) == 1 else i
        valid[i] = left._valid(a) and right._valid(b)
        if valid[i]:
            var x = left._values[a]
            var y = right._values[b]
            comptime if op == ADD:
                values[i] = checked_add(x, y)
            elif op == SUB:
                values[i] = _checked_sub(x, y)
            elif op == MUL:
                values[i] = _checked_mul(x, y)
            elif op == GT:
                predicates[i] = x > y
            else:
                predicates[i] = x == y
    comptime if op == GT or op == EQ:
        return Series("", Column[Bool](predicates^, valid))
    else:
        return Series("", Column[Int64](values^, valid))


def _equal[
    T: Copyable & Deinitable & Equatable
](left: Column[T], right: Column[T]) raises -> Series:
    var n = _length(len(left), len(right))
    var values = List[Bool](length=n, fill=False)
    var valid = List[Bool](length=n, fill=False)
    for i in range(n):
        var a = 0 if len(left) == 1 else i
        var b = 0 if len(right) == 1 else i
        valid[i] = left._valid(a) and right._valid(b)
        if valid[i]:
            values[i] = left._values[a] == right._values[b]
    return Series("", Column[Bool](values^, valid))


def binary[
    op: Int, width: Int = 4
](left: Series, right: Series) raises -> Series:
    if left._data.isa[Column[Float64]]():
        return _numeric_float[op, width](
            left._data[Column[Float64]], right._data[Column[Float64]]
        )
    if left._data.isa[Column[Int64]]():
        return _numeric_int[op](
            left._data[Column[Int64]], right._data[Column[Int64]]
        )
    comptime if op == EQ:
        if left._data.isa[Column[Bool]]():
            return _equal(left._data[Column[Bool]], right._data[Column[Bool]])
        if left._data.isa[Column[String]]():
            return _equal(
                left._data[Column[String]], right._data[Column[String]]
            )
    raise Error("Unsupported binary kernel")
