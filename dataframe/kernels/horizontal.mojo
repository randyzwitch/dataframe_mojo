"""Row-wise reductions across columns (#221): types and kernels.

A horizontal reduction casts inputs to a common type and reduces them
pairwise: sums fill nulls with zero and
add (`binary[ADD]`, so integers keep their overflow checks), min and max keep
the other side's value where one side is null, and any/all are Kleene `|`
and `&`. Float sums and means, and numeric min and max, accumulate into one
output buffer a SIMD vector at a time instead of allocating a column per
pair. Dtype dispatch happens once per input, never per row.
"""
from std.math import isnan
from std.memory import pack_bits
from std.sys import size_of
from dataframe.storage.boolean import BoolColumn
from dataframe.kernels.cast import cast_series
from dataframe.storage.column import Column
from dataframe.decimal import (
    common_decimal,
    decimal_add_type,
    decimal_width_for,
)
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.expr import (
    ADD,
    AND,
    FILL_NULL,
    GT,
    LT,
    OR,
    H_ALL,
    H_ANY,
    H_MAX,
    H_MEAN,
    H_MIN,
    H_SUM,
    horizontal_name,
)
from dataframe.kernels.binary import (
    binary,
    choose,
    fit_mask,
    validity,
    _packed_mask,
)
from dataframe.series import Series


def _integer_digits(dtype: DataType) -> Int:
    """The decimal digits needed to represent an integer type combined
    with a decimal."""
    var bits = dtype.bit_width()
    if bits == 8:
        return 3
    if bits == 16:
        return 5
    if bits == 32:
        return 10
    return 20 if dtype.is_unsigned() else 19


def _signed(bits: Int) -> DataType:
    if bits == 16:
        return DataType.INT16
    if bits == 32:
        return DataType.INT32
    return DataType.INT64


def _primitive(dtype: DataType) -> Bool:
    return dtype.is_integer() or dtype.is_float() or dtype == DataType.BOOL


def horizontal_supertype(
    kind: Int, left: DataType, right: DataType
) raises -> DataType:
    """The common input type: promote integers, floats, and Bool to their
    numeric supertype. For decimals, retain decimal scale and widen an
    integer to enough digits to hold its values."""
    if left == right:
        return left
    if left.is_decimal() or right.is_decimal():
        if left.is_decimal() and right.is_decimal():
            return common_decimal(left, right)
        var decimal = left if left.is_decimal() else right
        var other = right if left.is_decimal() else left
        if other.is_float():
            return DataType.FLOAT64
        if other.is_integer():
            var scale = decimal.scale()
            var digits = _integer_digits(other)
            if digits <= decimal.precision() - scale:
                return decimal
            var precision = min(38, digits + scale)
            return DataType.decimal(
                precision, scale, decimal_width_for(precision)
            )
    elif _primitive(left) and _primitive(right):
        if left == DataType.BOOL:
            return right
        if right == DataType.BOOL:
            return left
        if left.is_float() or right.is_float():
            if left == DataType.FLOAT64 or right == DataType.FLOAT64:
                return DataType.FLOAT64
            var integer = right if left.is_float() else left
            return (
                DataType.FLOAT32 if integer.bit_width()
                <= 16 else DataType.FLOAT64
            )
        if left.is_unsigned() == right.is_unsigned():
            return left if left.bit_width() > right.bit_width() else right
        var signed = right if left.is_unsigned() else left
        var unsigned = left if left.is_unsigned() else right
        if signed.bit_width() > unsigned.bit_width():
            return signed
        if unsigned.bit_width() == 64:
            return DataType.FLOAT64
        return _signed(2 * unsigned.bit_width())
    raise Error(
        horizontal_name(kind)
        + " requires inputs with a common type, found "
        + left.name()
        + " and "
        + right.name()
    )


def horizontal_input(kind: Int, dtype: DataType) raises:
    """Check one input of any_horizontal/all_horizontal: Bool or a number
    that can be converted to Bool (nonzero is true)."""
    if (kind == H_ANY or kind == H_ALL) and not _primitive(dtype):
        raise Error(
            horizontal_name(kind)
            + " requires Bool or numeric inputs, found "
            + dtype.name()
        )


def horizontal_result(
    kind: Int, common: DataType, inputs: Int
) raises -> DataType:
    """The result type of a reduction of `inputs` values with common type
    `common`; decimal sums use the type produced by repeated decimal
    addition."""
    if kind == H_ANY or kind == H_ALL:
        return DataType.BOOL
    if kind == H_MIN or kind == H_MAX:
        if common.is_nested() or common.is_geometry():
            raise Error(
                horizontal_name(kind)
                + " requires comparable inputs, found "
                + common.name()
            )
        return common
    if kind == H_MEAN:
        if not (common.is_numeric() or common == DataType.BOOL):
            raise Error(
                "mean_horizontal requires numeric inputs, found "
                + common.name()
            )
        return DataType.FLOAT32 if common == DataType.FLOAT32 else (
            DataType.FLOAT64
        )
    if common == DataType.BOOL:
        return DataType.UINT32
    if not (common.is_numeric() or common.is_duration()):
        raise Error(
            "sum_horizontal requires numeric, Bool or duration inputs, found "
            + common.name()
        )
    var result = common
    if common.is_decimal():
        for _ in range(inputs - 1):
            result = decimal_add_type(result, common)
    return result


def _store_flags[
    W: Int
](mut flags: List[UInt8], start: Int, value: SIMD[DType.bool, W]):
    comptime if W == 8:
        flags[start // 8] = pack_bits[DType.uint8](value)
    else:
        comptime assert W == 1
        var shift = UInt8(start & 7)
        var bit = UInt8(1) << shift
        flags[start // 8] = (flags[start // 8] & ~bit) | (
            UInt8(value[0]) << shift
        )


struct _Input[D: DType](Movable):
    """One input's values and validity, read W rows at a time; a one-row
    input broadcasts."""

    var column: Column[Scalar[Self.D]]
    var dense: Bool
    var scalar: Bool

    def __init__(out self, series: Series) raises:
        var flat = series.rechunk() if series.is_chunked() else series.copy()
        self.column = flat._data[Column[Scalar[Self.D]]].copy()
        self.dense = len(self.column._bits[]) == 0
        self.scalar = len(self.column) == 1

    def values[W: Int](self, start: Int) -> SIMD[Self.D, W]:
        if self.scalar:
            return SIMD[Self.D, W](self.column._get(0))
        return self.column._ptr().unsafe_load[width=W](start)

    def valid[W: Int](self, start: Int) -> SIMD[DType.bool, W]:
        if self.dense:
            return SIMD[DType.bool, W](fill=True)
        if self.scalar:
            return SIMD[DType.bool, W](fill=self.column._valid(0))
        return _packed_mask[W](
            self.column._bits[].span(), self.column._offset + start
        )


@always_inline
def _takes_right[
    D: DType, W: Int, is_max: Bool
](x: SIMD[D, W], y: SIMD[D, W]) -> SIMD[DType.bool, W]:
    """Where y replaces x as the running min/max. Skip NaN; a NaN x always
    gives way to a non-NaN y."""
    var better = y.gt(x) if is_max else y.lt(x)
    comptime if D.is_floating_point():
        return better | isnan(x)
    else:
        return better


def _extreme_block[
    D: DType, W: Int, is_max: Bool
](
    input: _Input[D],
    start: Int,
    mut values: List[Scalar[D]],
    mut valid: List[UInt8],
):
    var x = values.unsafe_ptr().unsafe_offset(start).unsafe_load[width=W]()
    var x_valid = _packed_mask[W](Span(valid), start)
    var y = input.values[W](start)
    var y_valid = input.valid[W](start)
    # Where only one side is valid that side wins; where both are, the
    # better one; where neither is, the slot stays null.
    var take = y_valid & (~x_valid | _takes_right[D, W, is_max](x, y))
    values.unsafe_ptr().unsafe_offset(start).unsafe_store(take.select(y, x))
    _store_flags[W](valid, start, x_valid | y_valid)


def _extreme[
    D: DType, is_max: Bool
](inputs: List[Series], n: Int) raises -> Series:
    # Polars' min/max horizontal also separates null-free inputs. They need
    # only payload comparisons: no running validity bitmap to unpack/repack.
    var dense = True
    for input in inputs:
        if input.null_count() != 0:
            dense = False
            break
    if dense:
        return _extreme_dense[D, is_max](inputs, n)
    comptime W = 1 if size_of[Scalar[D]]() > 8 else 8
    var values = List[Scalar[D]](length=n, fill=0)
    var valid = List[UInt8](length=(n + 7) // 8, fill=0)
    var main = n - n % W
    for k in range(len(inputs)):
        var input = _Input[D](inputs[k])
        for start in range(0, main, W):
            _extreme_block[D, W, is_max](input, start, values, valid)
        for start in range(main, n):
            _extreme_block[D, 1, is_max](input, start, values, valid)
    return Series("", Column[Scalar[D]](values=values^, bits=valid^))


def _extreme_dense[
    D: DType, is_max: Bool
](inputs: List[Series], n: Int) raises -> Series:
    comptime W = 1 if size_of[Scalar[D]]() > 8 else 8
    var values = List[Scalar[D]](unsafe_uninit_length=n)
    var out = values.unsafe_ptr()
    var main = n - n % W
    var first = _Input[D](inputs[0])
    for start in range(0, main, W):
        out.unsafe_offset(start).unsafe_store(first.values[W](start))
    for start in range(main, n):
        values[start] = first.values[1](start)[0]
    for k in range(1, len(inputs)):
        var input = _Input[D](inputs[k])
        for start in range(0, main, W):
            var x = out.unsafe_offset(start).unsafe_load[width=W]()
            var y = input.values[W](start)
            out.unsafe_offset(start).unsafe_store(
                _takes_right[D, W, is_max](x, y).select(y, x)
            )
        for start in range(main, n):
            var x = SIMD[D, 1](values[start])
            var y = input.values[1](start)
            values[start] = _takes_right[D, 1, is_max](x, y).select(y, x)[0]
    return Series("", Column[Scalar[D]](values^))


def _extreme_rows[is_max: Bool](inputs: List[Series], n: Int) raises -> Series:
    """Min or max of Bool or String inputs, which lack primitive numeric
    storage. Each step compares values, keeps the better non-null value,
    and picks with `choose`."""
    var result = inputs[0].copy()
    for k in range(1, len(inputs)):
        ref right = inputs[k]
        var better = binary[GT](result, right) if is_max else binary[LT](
            result, right
        )
        ref compared = better._data[BoolColumn]
        var left_valid = validity(result)
        var right_valid = validity(right)
        var keep = List[Bool](length=n, fill=False)
        for i in range(n):
            var a = 0 if len(left_valid) == 1 else i
            var b = 0 if len(right_valid) == 1 else i
            var c = 0 if len(compared) == 1 else i
            keep[i] = left_valid[a] and (
                not right_valid[b] or compared._valid(c) and compared._get(c)
            )
        result = choose(keep, result, right)
    return result^


def _float_fold[
    D: DType, mean: Bool
](inputs: List[Series], n: Int, ignore_nulls: Bool) raises -> Series:
    """Sum or mean of float inputs: one pass per input adds valid values
    and counts them, then one pass divides. Null adds nothing; NaN
    propagates."""
    comptime W = 8
    var sums = List[Scalar[D]](length=n, fill=0)
    var counts = List[Scalar[D]](length=n, fill=0)
    var main = n - n % W
    for k in range(len(inputs)):
        var input = _Input[D](inputs[k])
        var s = sums.unsafe_ptr()
        var c = counts.unsafe_ptr()
        for start in range(0, main, W):
            var valid = input.valid[W](start)
            var y = valid.select(input.values[W](start), SIMD[D, W](0))
            s.unsafe_offset(start).unsafe_store(
                s.unsafe_offset(start).unsafe_load[width=W]() + y
            )
            c.unsafe_offset(start).unsafe_store(
                c.unsafe_offset(start).unsafe_load[width=W]()
                + valid.select(SIMD[D, W](1), SIMD[D, W](0))
            )
        for start in range(main, n):
            if input.valid[1](start)[0]:
                sums[start] += input.values[1](start)[0]
                counts[start] += 1
    var total = Scalar[D](len(inputs))
    comptime if not mean:
        if ignore_nulls:
            return Series("", Column[Scalar[D]](sums^))
    var valid = List[UInt8](length=(n + 7) // 8, fill=0)
    for start in range(0, main, W):
        var count = counts.unsafe_ptr().unsafe_load[width=W](start)
        var present = count.gt(SIMD[D, W](0)) if ignore_nulls else count.eq(
            SIMD[D, W](total)
        )
        comptime if mean:
            var ptr = sums.unsafe_ptr().unsafe_offset(start)
            ptr.unsafe_store(
                present.select(
                    ptr.unsafe_load[width=W]() / count, SIMD[D, W](0)
                )
            )
        valid[start // 8] = pack_bits[DType.uint8](present)
    for i in range(main, n):
        var present = counts[i] > 0 if ignore_nulls else counts[i] == total
        comptime if mean:
            sums[i] = sums[i] / counts[i] if present else 0
        valid[i // 8] |= UInt8(present) << UInt8(i % 8)
    return Series("", Column[Scalar[D]](values=sums^, bits=valid^))


def _zero(dtype: DataType) raises -> Series:
    """A one-row zero of `dtype`'s storage, tagged `dtype`."""
    var storage = dtype.physical()
    if dtype.is_decimal():
        var width = dtype.decimal_width()
        if width == 128:
            return Series("", Column[Int128]([Int128(0)])).with_dtype(dtype)
        if width == 64:
            return Series("", Column[Int64]([Int64(0)])).with_dtype(dtype)
        return Series("", Column[Int32]([Int32(0)])).with_dtype(dtype)
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if storage == DataType.of(D):
            return Series("", Column[Scalar[D]]([Scalar[D](0)])).with_dtype(
                dtype
            )
    raise Error("sum_horizontal has no zero for " + dtype.name())


def horizontal(
    kind: Int,
    ignore_nulls: Bool,
    var inputs: List[Series],
    common: DataType,
    result: DataType,
    offset: Int,
    length: Int,
    mask: List[Bool],
) raises -> Series:
    """Reduce evaluated inputs row by row. `common` is their supertype and
    `result` the bound result type; one-row inputs broadcast. The result
    carries its storage type; the caller tags it `result`."""
    var n = 1
    for input in inputs:
        if len(input) != 1:
            if n != 1 and n != len(input):
                raise Error("Incompatible expression lengths")
            n = len(input)
    # Sums, means and any/all read every input at one type; min and max
    # read the supertype.
    var target = common
    if kind == H_ANY or kind == H_ALL:
        target = DataType.BOOL
    elif kind == H_MEAN:
        target = result
    elif kind == H_SUM and common == DataType.BOOL:
        target = DataType.UINT32
    for k in range(len(inputs)):
        if inputs[k].dtype() != target:
            var rows = len(inputs[k])
            inputs[k] = cast_series(
                inputs[k],
                target,
                True,
                offset if rows == length else 0,
                fit_mask(mask, rows),
            )
    if kind == H_ANY or kind == H_ALL:
        var out = inputs[0].copy()
        for k in range(1, len(inputs)):
            out = binary[OR](out, inputs[k], mask) if kind == H_ANY else (
                binary[AND](out, inputs[k], mask)
            )
        return out^
    if kind == H_MEAN or (kind == H_SUM and target.is_float()):
        if target == DataType.FLOAT32:
            if kind == H_MEAN:
                return _float_fold[DType.float32, True](inputs, n, ignore_nulls)
            return _float_fold[DType.float32, False](inputs, n, ignore_nulls)
        if kind == H_MEAN:
            return _float_fold[DType.float64, True](inputs, n, ignore_nulls)
        return _float_fold[DType.float64, False](inputs, n, ignore_nulls)
    if kind == H_SUM:
        # Polars' sum_horizontal: nulls become zero, then pairwise `+`.
        # Durations add as their Int64 counts.
        if target.is_duration():
            for k in range(len(inputs)):
                inputs[k] = inputs[k].with_dtype(DataType.INT64)
            target = DataType.INT64
        if ignore_nulls:
            var zero = _zero(target)
            for k in range(len(inputs)):
                if inputs[k].null_count() > 0:
                    inputs[k] = binary[FILL_NULL](inputs[k], zero)
        var out = inputs[0].copy()
        for k in range(1, len(inputs)):
            out = binary[ADD](out, inputs[k], mask)
        return out^
    # Min and max compare storage: decimals share one type and scale by
    # now, and temporal values order as their Int64 counts.
    if target.is_decimal() and target.decimal_width() == 128:
        if kind == H_MAX:
            return _extreme[DType.int128, True](inputs, n)
        return _extreme[DType.int128, False](inputs, n)
    var storage = target.physical()
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if storage == DataType.of(D):
            if kind == H_MAX:
                return _extreme[D, True](inputs, n)
            return _extreme[D, False](inputs, n)
    if kind == H_MAX:
        return _extreme_rows[True](inputs, n)
    return _extreme_rows[False](inputs, n)
