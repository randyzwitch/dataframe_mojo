"""Explicit dtype conversion. String parsing uses the shared strict cast grammar.

Values pass through an exact intermediate: Int128 for every integer type
(UInt64 included) and Bool, Float64 for both float types. Integer targets
are range-checked; float to integer truncates toward zero and rejects NaN,
infinities, and out-of-range values.
"""
from std.math import isinf, isnan, trunc
from std.sys import size_of
from .bool_column import BoolColumn
from .column import Column, _copy_validity
from .string_column import StringColumn, StringBuilder
from .dtype import DataType, NUMERIC_DTYPES
from .parse import parse_bool, parse_float64, parse_integer
from .decimal import (
    check_precision,
    divide_half_even,
    format_decimal,
    parse_decimal,
    pow10,
)
from .series import Series
from .parallel import Job, configured_workers, run_jobs
from .categorical import decode, encode, recode
from .temporal_kernels import cast_temporal


def _dtype(series: Series) -> DataType:
    return series.dtype()


def _text(series: Series, row: Int) -> String:
    if series.dtype().is_decimal():
        return format_decimal(series._decimal_raw(row), series.dtype().scale())
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            return String(series._data[Column[Scalar[D]]]._get(row))
    if series._data.isa[BoolColumn]():
        return "true" if series._data[BoolColumn]._get(row) else "false"
    return String(series._data[StringColumn]._get(row))


def _valid(series: Series, row: Int) -> Bool:
    if series.dtype().is_decimal():
        return series._decimal_valid(row)
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            return series._data[Column[Scalar[D]]]._valid(row)
    if series._data.isa[BoolColumn]():
        return series._data[BoolColumn]._valid(row)
    return series._data[StringColumn]._valid(row)


def _to_integer[D: DType](value: Int128) raises -> Scalar[D]:
    if (
        value < Scalar[D].MIN.cast[DType.int128]()
        or value > Scalar[D].MAX.cast[DType.int128]()
    ):
        raise Error("out of " + String(D) + " range")
    return value.cast[D]()


def _float_to_integer[D: DType](value: Float64) raises -> Scalar[D]:
    if isnan(value) or isinf(value):
        raise Error("out of " + String(D) + " range")
    # Exact bounds: 2**(bits - 1) for signed types, 2**bits for unsigned.
    var bits = size_of[Scalar[D]]() * 8 - (1 if D.is_signed() else 0)
    var upper = Float64(1)
    for _ in range(bits):
        upper *= 2
    var lower = -upper if D.is_signed() else Float64(0)
    var whole = trunc(value)
    if whole >= upper or whole < lower:
        raise Error("out of " + String(D) + " range")
    return whole.cast[D]()


struct _Values(Movable):
    """The exact intermediate for one cast: `is_float` picks the list."""

    var ints: List[Int128]
    var floats: List[Float64]
    var is_float: Bool

    def __init__(out self, n: Int, is_float: Bool):
        self.is_float = is_float
        self.ints = List[Int128](length=0 if is_float else n, fill=0)
        self.floats = List[Float64](length=n if is_float else 0, fill=0)


def _read_source(
    input: Series,
    source: DataType,
    target: DataType,
    i: Int,
    mut values: _Values,
) raises:
    """Row i of input into the intermediate, parsing strings for target."""
    if source.is_decimal():
        var raw = input._decimal_raw(i)
        if target.is_float():
            values.floats[i] = Float64(raw) / Float64(pow10(source.scale()))
        else:
            # Half to even, as Polars casts: 1.5 to 2, 0.5 to 0 (#342).
            values.ints[i] = divide_half_even(raw, pow10(source.scale()))
        return
    if source == DataType.STRING:
        # Numeric parsers consume borrowed slices; avoid an owned copy per row.
        var text = input._data[StringColumn]._get(i)
        if target.is_float():
            values.floats[i] = parse_float64(text)
        elif target == DataType.BOOL:
            values.ints[i] = Int128(Int(parse_bool(text)))
        elif target == DataType.UINT64:
            values.ints[i] = parse_integer[DType.uint64](text).cast[
                DType.int128
            ]()
        else:
            values.ints[i] = parse_integer[DType.int64](text).cast[
                DType.int128
            ]()
        return
    if source == DataType.BOOL:
        var bit = Int(input._data[BoolColumn]._get(i))
        if values.is_float:
            values.floats[i] = Float64(bit)
        else:
            values.ints[i] = Int128(bit)
        return
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if input._data.isa[Column[Scalar[D]]]():
            var x = input._data[Column[Scalar[D]]]._get(i)
            comptime if D.is_floating_point():
                if values.is_float:
                    values.floats[i] = x.cast[DType.float64]()
                else:
                    values.floats[i] = x.cast[DType.float64]()
            else:
                if values.is_float:
                    values.floats[i] = x.cast[DType.float64]()
                else:
                    values.ints[i] = x.cast[DType.int128]()


def _integer_at(input: Series, i: Int) raises -> Int128:
    """Row i of an integer or Bool column as Int128. The integer-to-decimal
    cast used to read it through a one-row buffer indexed by i, which went
    out of bounds from the second row on."""
    if input._data.isa[BoolColumn]():
        return Int128(1) if input._data[BoolColumn]._get(i) else Int128(0)
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if D.is_integral():
            if input._data.isa[Column[Scalar[D]]]():
                return (
                    input._data[Column[Scalar[D]]]._get(i).cast[DType.int128]()
                )
    raise Error("expected an integer column, found " + input.dtype().name())


def _cast_binary(
    input: Series,
    source: DataType,
    target: DataType,
    strict: Bool,
    offset: Int,
    mask: List[Bool],
) raises -> Series:
    """Binary casts, as in Polars: anything that casts to string casts to
    binary as its UTF-8 bytes; binary casts only to string, and each value
    must be valid UTF-8 (strict raises, otherwise the row is null)."""
    if target.is_binary():
        var text = input.copy() if source == DataType.STRING else cast_series(
            input, DataType.STRING, strict, offset, mask
        )
        return text.with_dtype(DataType.BINARY)
    if target != DataType.STRING:
        raise Error(
            "cannot cast binary to " + target.name() + "; cast to string first"
        )
    ref column = input._data[StringColumn]
    var n = len(column)
    var out = StringBuilder(n)
    for i in range(n):
        if not column._valid(i) or (len(mask) == n and not mask[i]):
            out.append_null()
            continue
        try:
            out.append(String(StringSlice(from_utf8=column._row_bytes(i))))
        except:
            if strict:
                raise Error(
                    "cast from binary to string failed at row "
                    + String(offset + i)
                    + ": the value is not valid UTF-8"
                )
            out.append_null()
    return Series(input.name(), out^.finish())


struct _ParseJob[D: DType](Job):
    """Parse rows [first, last) of a string column into `values` (#149).

    Integers parse as Int64 and are range-checked into D (UInt64 parses
    directly), floats parse as Float64 and narrow to D: exactly the generic
    cast's steps, so accepted text, rounding and messages are unchanged.
    The first failing row is recorded rather than raised, so the caller can
    report the earliest across ranges, as a serial loop would."""

    var column: StringColumn
    var values: Int
    var valid: Int
    var mask: Int
    var first: Int
    var last: Int
    var failed_row: Int
    var message: String

    def __init__(
        out self,
        column: StringColumn,
        values: Int,
        valid: Int,
        mask: Int,
        first: Int,
        last: Int,
    ):
        self.column = column.copy()
        self.values = values
        self.valid = valid
        self.mask = mask
        self.first = first
        self.last = last
        self.failed_row = -1
        self.message = String()

    def run(mut self) raises:
        var out = Pointer[List[Scalar[Self.D]], MutAnyOrigin](
            unsafe_from_address=self.values
        )[].unsafe_ptr()
        var valid = Pointer[List[Bool], MutAnyOrigin](
            unsafe_from_address=self.valid
        )[].unsafe_ptr()
        ref mask = Pointer[List[Bool], MutAnyOrigin](
            unsafe_from_address=self.mask
        )[]
        var masked = len(mask) > 0
        var nulls = self.column.null_count() > 0
        var views = self.column._is_view_storage()
        var offsets = (
            self.column._offsets[]
            .unsafe_ptr()
            .unsafe_offset(
                self.column._offset
            ) if not views else self.column._offsets[]
            .unsafe_ptr()
        )
        var data = self.column._bytes[].unsafe_ptr()
        for i in range(self.first, self.last):
            if (nulls and not self.column._valid(i)) or (
                masked and not mask[i]
            ):
                valid.unsafe_offset(i)[] = False
                continue
            var text: StringSlice[ImmutAnyOrigin]
            if views:
                text = self.column._get(i)
            else:
                var start = Int(offsets.unsafe_offset(i)[])
                text = StringSlice[ImmutAnyOrigin](
                    unsafe_from_utf8=Span[UInt8, ImmutAnyOrigin](
                        unsafe_ptr=data.unsafe_offset(start)
                        .unsafe_mut_cast[False]()
                        .unsafe_origin_cast[ImmutAnyOrigin](),
                        length=Int(offsets.unsafe_offset(i + 1)[]) - start,
                    )
                )
            try:
                comptime if Self.D.is_floating_point():
                    out.unsafe_offset(i)[] = parse_float64(text).cast[Self.D]()
                elif Self.D == DType.uint64:
                    out.unsafe_offset(i)[] = rebind[Scalar[Self.D]](
                        parse_integer[DType.uint64](text)
                    )
                else:
                    out.unsafe_offset(i)[] = _to_integer[Self.D](
                        parse_integer[DType.int64](text).cast[DType.int128]()
                    )
                valid.unsafe_offset(i)[] = True
            except e:
                valid.unsafe_offset(i)[] = False
                if self.failed_row < 0:
                    self.failed_row = i
                    self.message = String(e)


def _parse_strings[
    D: DType
](
    input: Series,
    source: DataType,
    target: DataType,
    strict: Bool,
    offset: Int,
    mask: List[Bool],
) raises -> Series:
    """String to a number type in one pass per row range, on every worker
    (#149). The generic loop re-dispatched on the source and target types
    for every row and built a 128-bit intermediate: about 125 ns per row,
    where the parse itself is a few."""
    ref column = input._data[StringColumn]
    var n = len(column)
    var values = List[Scalar[D]](length=n, fill=0)
    var valid = List[Bool](length=n, fill=False)
    var observed = mask.copy() if len(mask) == n else List[Bool]()
    var workers = configured_workers() if n >= (1 << 16) else 1
    var jobs = List[_ParseJob[D]](capacity=workers)
    for w in range(workers):
        jobs.append(
            _ParseJob[D](
                column,
                Int(Pointer(to=values)),
                Int(Pointer(to=valid)),
                Int(Pointer(to=observed)),
                n * w // workers,
                n * (w + 1) // workers,
            )
        )
    run_jobs(jobs)
    # The jobs read the mask by address; keep it alive past them.
    _ = observed^
    if strict:
        for w in range(len(jobs)):
            if jobs[w].failed_row >= 0:
                var row = jobs[w].failed_row
                raise Error(
                    "cast from "
                    + source.name()
                    + " to "
                    + target.name()
                    + " failed at row "
                    + String(offset + row)
                    + " for value '"
                    + _text(input, row)
                    + "': "
                    + jobs[w].message
                )
    return Series(input.name(), Column[Scalar[D]](values^, valid))


def _decimal_to_float(input: Series) raises -> Series:
    """A decimal column of any width as Float64, through a pointer at its
    width: the same value / 10^scale the row path computes."""
    var n = len(input)
    var divisor = Float64(pow10(input.dtype().scale()))
    var values = List[Float64](unsafe_uninit_length=n)
    var out = values.unsafe_ptr()
    var bits: List[UInt8]
    if input._data.isa[Column[Int64]]():
        ref column = input._data[Column[Int64]]
        var source = column._ptr()
        for i in range(n):
            out[unsafe_offset=i] = Float64(source[unsafe_offset=i]) / divisor
        bits = _copy_validity(column._bits[], column._offset, n)
    elif input._data.isa[Column[Int32]]():
        ref column = input._data[Column[Int32]]
        var source = column._ptr()
        for i in range(n):
            out[unsafe_offset=i] = Float64(source[unsafe_offset=i]) / divisor
        bits = _copy_validity(column._bits[], column._offset, n)
    else:
        ref column = input._data[Column[Int128]]
        var source = column._ptr()
        for i in range(n):
            out[unsafe_offset=i] = Float64(source[unsafe_offset=i]) / divisor
        bits = _copy_validity(column._bits[], column._offset, n)
    return Series(input.name(), Column[Float64](values=values^, bits=bits^))


def _cast_exact(input: Series, target: DataType) raises -> Optional[Series]:
    """Numeric casts in one typed loop. Those that cannot fail: Bool to any
    number, an integer to a type that holds its whole range, any integer to
    a float, and Float32 to Float64. And an integer to a narrower or
    differently signed integer type when every value turns out to fit.

    The general path reads each row into a 128-bit intermediate inside a
    try block, about 30 times the cost per row; it remains for casts from
    floats to integers and for columns with a value that does not fit.
    None means the general path must run. Null rows hold zero, as there."""
    var n = len(input)
    if input._data.isa[BoolColumn]():
        ref bools = input._data[BoolColumn]
        var nulls = bools.null_count() > 0
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime T = NUMERIC_DTYPES[k]
            if target == DataType.of(T):
                var out = List[Scalar[T]](unsafe_uninit_length=n)
                var to = out.unsafe_ptr()
                for i in range(n):
                    to[unsafe_offset=i] = Scalar[T](
                        1 if bools._get(i)
                        and (not nulls or bools._valid(i)) else 0
                    )
                return Series(
                    input.name(),
                    Column[Scalar[T]](
                        values=out^,
                        bits=_copy_validity(bools._bits[], bools._offset, n),
                    ),
                )
        return None
    comptime for j in range(len(NUMERIC_DTYPES)):
        comptime S = NUMERIC_DTYPES[j]
        if input._data.isa[Column[Scalar[S]]]():
            ref column = input._data[Column[Scalar[S]]]
            var nulls = column.null_count() > 0
            comptime for k in range(len(NUMERIC_DTYPES)):
                comptime T = NUMERIC_DTYPES[k]
                comptime wider = size_of[Scalar[T]]() >= size_of[Scalar[S]]()
                comptime if S != T and (
                    (
                        T.is_floating_point()
                        and (not S.is_floating_point() or wider)
                    )
                    or (
                        not T.is_floating_point()
                        and not S.is_floating_point()
                        and (
                            (S.is_signed() == T.is_signed() and wider)
                            or (
                                not S.is_signed()
                                and T.is_signed()
                                and size_of[Scalar[T]]() > size_of[Scalar[S]]()
                            )
                        )
                    )
                ):
                    if target == DataType.of(T):
                        var out = List[Scalar[T]](unsafe_uninit_length=n)
                        var to = out.unsafe_ptr()
                        var source = column._ptr()
                        for i in range(n):
                            to[unsafe_offset=i] = source[unsafe_offset=i].cast[
                                T
                            ]()
                        if nulls:
                            for i in range(n):
                                if not column._valid(i):
                                    to[unsafe_offset=i] = 0
                        return Series(
                            input.name(),
                            Column[Scalar[T]](
                                values=out^,
                                bits=_copy_validity(
                                    column._bits[], column._offset, n
                                ),
                            ),
                        )
            # An integer to a type that may not hold it: the same loop,
            # which also checks every valid value fits. If one does not,
            # the general path below reports it or nulls it.
            comptime for k in range(len(NUMERIC_DTYPES)):
                comptime T = NUMERIC_DTYPES[k]
                comptime if (
                    S != T
                    and not S.is_floating_point()
                    and not T.is_floating_point()
                ):
                    if target == DataType.of(T):
                        var low = Scalar[T].MIN.cast[DType.int128]()
                        var high = Scalar[T].MAX.cast[DType.int128]()
                        var out = List[Scalar[T]](unsafe_uninit_length=n)
                        var to = out.unsafe_ptr()
                        var source = column._ptr()
                        var fits = True
                        for i in range(n):
                            var value = source[unsafe_offset=i]
                            if nulls and not column._valid(i):
                                to[unsafe_offset=i] = 0
                                continue
                            var wide = value.cast[DType.int128]()
                            fits = fits and wide >= low and wide <= high
                            to[unsafe_offset=i] = value.cast[T]()
                        if not fits:
                            return None
                        return Series(
                            input.name(),
                            Column[Scalar[T]](
                                values=out^,
                                bits=_copy_validity(
                                    column._bits[], column._offset, n
                                ),
                            ),
                        )
            return None
    return None


def cast_series(
    input: Series, target: DataType, strict: Bool, offset: Int, mask: List[Bool]
) raises -> Series:
    """Convert every valid, observed row; others become null."""
    if input.is_chunked():
        return cast_series(input.rechunk(), target, strict, offset, mask)
    if input.dtype().is_nested() or target.is_nested():
        raise Error(
            "cannot cast "
            + input.dtype().name()
            + " to "
            + target.name()
            + ": list and struct casts are not supported yet"
        )
    var source = _dtype(input)
    if source.is_decimal() and target == DataType.FLOAT64 and len(mask) == 0:
        return _decimal_to_float(input)
    if source.is_categorical() or target.is_categorical():
        # Categoricals convert through their values (#106): decode, cast
        # the strings, and encode when the target is categorical.
        if source.is_categorical() and target.is_categorical():
            if not target.has_dictionary() or source == target:
                return input.copy()
            return recode(input, target)
        if source.is_categorical():
            return cast_series(decode(input), target, strict, offset, mask)
        var strings = (
            input.copy() if source.physical()
            == DataType.STRING else cast_series(
                input, DataType.STRING, strict, offset, mask
            )
        )
        if target.has_dictionary():
            return recode(strings, target)
        return encode(strings)
    if source == target:
        return input.copy()
    if source.is_binary() or target.is_binary():
        return _cast_binary(input, source, target, strict, offset, mask)
    if source.is_temporal() or target.is_temporal():
        var observed = input.copy()
        return cast_temporal(observed, source, target, strict)
    if source == DataType.STRING and (target.is_integer() or target.is_float()):
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if target == DataType.of(D):
                return _parse_strings[D](
                    input, source, target, strict, offset, mask
                )
    # Plain numeric types only: a logical type (a decimal, a categorical)
    # over the same storage means something else.
    if len(mask) == 0 and source == input.dtype().physical():
        var exact = _cast_exact(input, target)
        if exact:
            return exact.take()
    var n = len(input)
    var valid = List[Bool](length=n, fill=False)
    if target.is_decimal():
        var decimal_values = List[Int128](length=n, fill=0)
        for i in range(n):
            if not _valid(input, i) or (len(mask) == n and not mask[i]):
                continue
            try:
                if source == DataType.STRING:
                    decimal_values[i] = parse_decimal(
                        input._data[StringColumn]._get(i), target
                    )
                elif source.is_decimal():
                    var raw = input._decimal_raw(i)
                    if source.scale() <= target.scale():
                        raw *= pow10(target.scale() - source.scale())
                    else:
                        raw = divide_half_even(
                            raw, pow10(source.scale() - target.scale())
                        )
                    decimal_values[i] = check_precision(raw, target)
                elif source.is_integer() or source == DataType.BOOL:
                    decimal_values[i] = check_precision(
                        _integer_at(input, i) * pow10(target.scale()), target
                    )
                else:
                    decimal_values[i] = parse_decimal(
                        StringSlice(String(_text(input, i))), target
                    )
                valid[i] = True
            except e:
                if strict:
                    raise Error(
                        "cast from "
                        + source.name()
                        + " to "
                        + target.name()
                        + " failed at row "
                        + String(offset + i)
                        + " for value \x27"
                        + _text(input, i)
                        + "\x27: "
                        + String(e)
                    )
        var width = target.decimal_width()
        if width == 64 or width == 32:
            # Precision 18 (or 9) fits the declared width; store it there.
            if width == 64:
                var narrow = List[Int64](capacity=n)
                for v in decimal_values:
                    narrow.append(Int64(v))
                return Series(
                    input.name(), Column[Int64](narrow^, valid)
                ).with_dtype(target)
            var narrow = List[Int32](capacity=n)
            for v in decimal_values:
                narrow.append(Int32(v))
            return Series(
                input.name(), Column[Int32](narrow^, valid)
            ).with_dtype(target)
        return Series(
            input.name(), Column[Int128](decimal_values^, valid)
        ).with_dtype(target)
    if target == DataType.STRING:
        var out = StringBuilder(n)
        for i in range(n):
            if _valid(input, i) and (len(mask) != n or mask[i]):
                out.append(_text(input, i))
            else:
                out.append_null()
        return Series(input.name(), out^.finish())
    # Floats stay floats in the intermediate; everything else is exact Int128.
    var float_source = source.is_float()
    var values = _Values(n, target.is_float() or float_source)
    var targets = List[Bool](length=n, fill=False)
    for i in range(n):
        if not _valid(input, i):
            continue
        if len(mask) == n and not mask[i]:
            continue
        try:
            _read_source(input, source, target, i, values)
            if target == DataType.BOOL:
                if values.is_float:
                    if isnan(values.floats[i]):
                        raise Error("NaN has no Boolean value")
                    targets[i] = values.floats[i] != 0
                else:
                    targets[i] = values.ints[i] != 0
            elif target.is_integer():
                comptime for k in range(len(NUMERIC_DTYPES)):
                    comptime D = NUMERIC_DTYPES[k]
                    comptime if D.is_integral():
                        if target == DataType.of(D):
                            if values.is_float:
                                _ = _float_to_integer[D](values.floats[i])
                            else:
                                _ = _to_integer[D](values.ints[i])
            valid[i] = True
        except e:
            if strict:
                raise Error(
                    "cast from "
                    + source.name()
                    + " to "
                    + target.name()
                    + " failed at row "
                    + String(offset + i)
                    + " for value '"
                    + _text(input, i)
                    + "': "
                    + String(e)
                )
    if target == DataType.BOOL:
        return Series(input.name(), BoolColumn(targets^, valid))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if target == DataType.of(D):
            var out = List[Scalar[D]](length=n, fill=0)
            for i in range(n):
                if not valid[i]:
                    continue
                comptime if D.is_floating_point():
                    out[i] = (
                        values.floats[i]
                        .cast[D]() if values.is_float else values.ints[i]
                        .cast[D]()
                    )
                else:
                    out[i] = _float_to_integer[D](
                        values.floats[i]
                    ) if values.is_float else _to_integer[D](values.ints[i])
            return Series(input.name(), Column[Scalar[D]](out^, valid))
    raise Error("Unsupported cast target " + target.name())
