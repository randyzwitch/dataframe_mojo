"""Explicit dtype conversion. String parsing uses the shared strict cast grammar.

Values pass through an exact intermediate: Int128 for every integer type
(UInt64 included) and Bool, Float64 for both float types. Integer targets
are range-checked; float to integer truncates toward zero and rejects NaN,
infinities, and out-of-range values.
"""
from std.math import isinf, isnan, trunc
from std.sys import size_of
from std.bit import count_trailing_zeros
from std.memory import ArcPointer, Pointer
from dataframe.storage.boolean import BoolColumn, _bits64
from dataframe.storage.column import (
    Column,
    _append_validity,
    _copy_validity,
    _append_validity_bit,
    SCALAR_DTYPES,
)
from dataframe.storage.string import StringColumn, StringBuilder
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.text.parse import parse_bool, parse_float64, parse_integer
from dataframe.text.bytes import _Bytes
from dataframe.decimal import (
    check_limit,
    check_precision,
    divide_half_even,
    format_decimal,
    parse_decimal,
    pow10,
    precision_limit,
)
from dataframe.series import Series, Storage
from dataframe.parallel import Job, configured_workers, run_jobs
from dataframe.categorical import decode, encode, recode
from dataframe.kernels.temporal import cast_temporal
from dataframe.kernels.physical_input import _has_physical_storage


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


def _integers_to_decimal(
    input: Series, target: DataType, strict: Bool, offset: Int
) raises -> Optional[Series]:
    """Convert an integer column to decimal in one typed pass per source
    type: multiply each value by the scale's power of ten, compute the
    precision limit once, then write at the target width with the source's
    validity. A value beyond the precision raises when `strict` and is
    null otherwise. Return None for storage this path does not read."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if D.is_integral():
            if input._data.isa[Column[Scalar[D]]]():
                ref column = input._data[Column[Scalar[D]]]
                var width = target.decimal_width()
                if width == 32:
                    return _scaled_integers[D, DType.int32](
                        column, input, target, strict, offset
                    )
                if width == 64:
                    return _scaled_integers[D, DType.int64](
                        column, input, target, strict, offset
                    )
                return _scaled_integers[D, DType.int128](
                    column, input, target, strict, offset
                )
    return None


def _scaled_integers[
    D: DType, W: DType
](
    column: Column[Scalar[D]],
    input: Series,
    target: DataType,
    strict: Bool,
    offset: Int,
) raises -> Series:
    var n = len(column)
    var factor = pow10(target.scale())
    var limit = precision_limit(target.precision())
    var values = List[Scalar[W]](unsafe_uninit_length=n)
    var out = values.unsafe_ptr()
    var source = column._ptr()
    var nulls = column.null_count() > 0
    # Every value of the source type fits the target (BIGINT to
    # DECIMAL(19,0), INTEGER to DECIMAL(10,0) or wider): no row can
    # overflow, so the scaling is one unchecked pass over every slot,
    # which vectorizes, and the source's validity is copied as it is.
    # DuckDB's binder makes the same choice from the types alone.
    var bound = Scalar[D].MAX.cast[DType.int128]() + 1
    if bound <= (limit - 1) // factor:
        var scaled = Scalar[W](factor.cast[W]())
        for i in range(n):
            out[unsafe_offset=i] = source[unsafe_offset=i].cast[W]() * scaled
        if not nulls:
            return Series(input.name(), Column[Scalar[W]](values^)).with_dtype(
                target
            )
        var valid = _copy_validity(column._bits[], column._offset, n)
        # Null slots hold 0, as on the checked path.
        for k in range(len(valid)):
            var byte = valid[k]
            if byte == 255:
                continue
            for b in range(8):
                var row = 8 * k + b
                if row < n and not (byte >> UInt8(b)) & 1:
                    out[unsafe_offset=row] = 0
        return Series(
            input.name(), Column[Scalar[W]](values=values^, bits=valid^)
        ).with_dtype(target)
    var overflowed = List[Int]()
    for i in range(n):
        if nulls and not column._valid(i):
            out[unsafe_offset=i] = 0
            continue
        var value = source[unsafe_offset=i].cast[DType.int128]() * factor
        if value <= -limit or value >= limit:
            if strict:
                try:
                    _ = check_limit(value, limit, target)
                except e:
                    raise Error(
                        "cast from "
                        + input.dtype().name()
                        + " to "
                        + target.name()
                        + " failed at row "
                        + String(offset + i)
                        + " for value \x27"
                        + _text(input, i)
                        + "\x27: "
                        + String(e)
                    )
            overflowed.append(i)
            out[unsafe_offset=i] = 0
            continue
        out[unsafe_offset=i] = value.cast[W]()
    if not nulls and len(overflowed) == 0:
        return Series(input.name(), Column[Scalar[W]](values^)).with_dtype(
            target
        )
    var bits = _copy_validity(
        column._bits[], column._offset, n
    ) if nulls else List[UInt8](length=(n + 7) // 8, fill=255)
    for i in overflowed:
        bits[i >> 3] &= ~(UInt8(1) << UInt8(i & 7))
    return Series(
        input.name(), Column[Scalar[W]](values=values^, bits=bits^)
    ).with_dtype(target)


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


@no_inline
def _string_cast_host(input: Series) -> Bool:
    """Admit readable unencoded windows; malformed metadata keeps the adapter."""
    ref chunks = input._chunked.value()[]
    if chunks.cache[].state.load() == 2:
        return False
    for array in chunks.arrays:
        if not array.isa[StringColumn]():
            return False
        ref column = array[StringColumn]
        if column._codes:
            return False
        var offset = column._offset
        var count = len(column)
        if offset < 0 or count < 0:
            return False
        if column._is_view_storage():
            var storage = column._view_storage_unchecked()
            if column._bits.ptr() != storage._bits.ptr():
                return False
            if not storage._views.on_host() or not storage._bits.on_host():
                return False
            for buffer in storage._buffers[]:
                if not buffer.on_host():
                    return False
            var rows = len(storage)
            var descriptors = len(storage._views[])
            if offset > rows or count > rows - offset:
                return False
            if offset > descriptors or count > descriptors - offset:
                return False
            var end = offset + count
            if len(storage._bits[]) != 0 and (
                end // 8 + Int(end % 8 != 0) > len(storage._bits[])
            ):
                return False
            for row in range(offset, end):
                var view = storage._view_unchecked(row)
                if not view.is_inline():
                    var index = Int(view.buffer_index)
                    if index >= len(storage._buffers[]):
                        return False
                    var available = len(storage._buffers[][index])
                    var length = Int(view.length)
                    if (
                        length > available
                        or Int(view.offset) > available - length
                    ):
                        return False
        else:
            if (
                not column._bytes.on_host()
                or not column._offsets.on_host()
                or not column._bits.on_host()
            ):
                return False
            var entries = len(column._offsets[])
            if entries == 0 or offset > entries - 1:
                return False
            if count > entries - 1 - offset:
                return False
            var end = offset + count
            if len(column._bits[]) != 0 and (
                end // 8 + Int(end % 8 != 0) > len(column._bits[])
            ):
                return False
            var available = Int64(len(column._bytes[]))
            var previous = column._offsets[][offset]
            if previous < 0 or previous > available:
                return False
            for row in range(offset + 1, end + 1):
                var next = column._offsets[][row]
                if next < previous or next > available:
                    return False
                previous = next
    return True


@no_inline
def _string_format_host[S: DType](input: Series) -> Bool:
    ref chunks = input._chunked.value()[]
    if chunks.cache[].state.load() == 2:
        return False
    for array in chunks.arrays:
        comptime if S == DType.bool:
            if not array.isa[BoolColumn]():
                return False
            ref column = array[BoolColumn]
            if not column._data.on_host() or not column._bits.on_host():
                return False
        else:
            if not array.isa[Column[Scalar[S]]]():
                return False
            ref column = array[Column[Scalar[S]]]
            if not column._data.on_host() or not column._bits.on_host():
                return False
    return True


@no_inline
def _string_cast_spans(input: Series) -> ArcPointer[List[_Bytes]]:
    var spans = List[_Bytes](capacity=len(input._chunked.value()[].arrays))
    for array in input._chunked.value()[].arrays:
        spans.append(_Bytes(array[StringColumn]))
    return ArcPointer(spans^)


@no_inline
def _chunk_format_strings[
    S: DType
](input: Series, mask: List[Bool]) raises -> Series:
    var output = StringBuilder(len(input))
    var masked = len(mask) == len(input)
    var base = 0
    for array in input._chunked.value()[].arrays:
        var count = 0
        comptime if S == DType.bool:
            ref column = array[BoolColumn]
            count = len(column)
            for row in range(len(column)):
                if column._valid(row) and (not masked or mask[base + row]):
                    output.append("true" if column._get(row) else "false")
                else:
                    output.append_null()
        else:
            ref column = array[Column[Scalar[S]]]
            count = len(column)
            var values = column._ptr()
            for row in range(len(column)):
                if column._valid(row) and (not masked or mask[base + row]):
                    output.append(String(values[unsafe_offset=row]))
                else:
                    output.append_null()
        base += count
    return Series(input.name(), output^.finish())


struct _ChunkParseJob[D: DType](Job):
    """Own source spans while writing one unchanged global worker range."""

    var spans: ArcPointer[List[_Bytes]]
    var values: Int
    var valid: Int
    var mask: Int
    var first: Int
    var last: Int
    var failed_row: Int
    var message: String

    def __init__(
        out self,
        spans: ArcPointer[List[_Bytes]],
        values: Int,
        valid: Int,
        mask: Int,
        first: Int,
        last: Int,
    ):
        self.spans = spans
        self.values = values
        self.valid = valid
        self.mask = mask
        self.first = first
        self.last = last
        self.failed_row = -1
        self.message = String()

    def run(mut self) raises:
        if self.first == self.last:
            return
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
        var span = self.spans[].unsafe_ptr()
        var base = 0
        var end = len(span[].column)
        while end <= self.first:
            base = end
            span = span.unsafe_offset(1)
            end += len(span[].column)
        for row in range(self.first, self.last):
            while row >= end:
                base = end
                span = span.unsafe_offset(1)
                end += len(span[].column)
            var local = row - base
            if not span[].column._valid(local) or (masked and not mask[row]):
                valid.unsafe_offset(row)[] = False
                continue
            var text = StringSlice[ImmutAnyOrigin](
                unsafe_from_utf8=span[].get(local)
            )
            try:
                comptime if Self.D.is_floating_point():
                    out.unsafe_offset(row)[] = parse_float64(text).cast[
                        Self.D
                    ]()
                elif Self.D == DType.uint64:
                    out.unsafe_offset(row)[] = rebind[Scalar[Self.D]](
                        parse_integer[DType.uint64](text)
                    )
                else:
                    out.unsafe_offset(row)[] = _to_integer[Self.D](
                        parse_integer[DType.int64](text).cast[DType.int128]()
                    )
                valid.unsafe_offset(row)[] = True
            except error:
                valid.unsafe_offset(row)[] = False
                if self.failed_row < 0:
                    self.failed_row = row
                    self.message = String(error)


@no_inline
def _chunk_string_error(
    target: DataType, row: Int, text: StringSlice, cause: String
) raises:
    raise Error(
        "cast from string to "
        + target.name()
        + " failed at row "
        + String(row)
        + " for value '"
        + String(text)
        + "': "
        + cause
    )


@no_inline
def _chunk_parse_strings[
    D: DType
](
    input: Series, target: DataType, strict: Bool, offset: Int, mask: List[Bool]
) raises -> Series:
    var n = len(input)
    var spans = _string_cast_spans(input)
    var values = List[Scalar[D]](length=n, fill=0)
    var valid = List[Bool](length=n, fill=False)
    var observed = mask.copy() if len(mask) == n else List[Bool]()
    # Preserve the contiguous parser's worker policy and disjoint byte writes.
    var workers = configured_workers() if n >= (1 << 16) else 1
    var jobs = List[_ChunkParseJob[D]](capacity=workers)
    for worker in range(workers):
        jobs.append(
            _ChunkParseJob[D](
                spans,
                Int(Pointer(to=values)),
                Int(Pointer(to=valid)),
                Int(Pointer(to=observed)),
                n * worker // workers,
                n * (worker + 1) // workers,
            )
        )
    run_jobs(jobs)
    _ = observed^
    if strict:
        for worker in range(len(jobs)):
            if jobs[worker].failed_row >= 0:
                var row = jobs[worker].failed_row
                var base = 0
                for span in spans[]:
                    if row < base + len(span.column):
                        var text = StringSlice[ImmutAnyOrigin](
                            unsafe_from_utf8=span.get(row - base)
                        )
                        _chunk_string_error(
                            target, offset + row, text, jobs[worker].message
                        )
                    base += len(span.column)
    return Series(input.name(), Column[Scalar[D]](values^, valid))


@no_inline
def _chunk_parse_bools(
    input: Series, strict: Bool, offset: Int, mask: List[Bool]
) raises -> Series:
    var n = len(input)
    var spans = _string_cast_spans(input)
    var values = List[UInt8](length=(n + 7) // 8, fill=0)
    var bits = List[UInt8](length=(n + 7) // 8, fill=0)
    var out = values.unsafe_ptr()
    var validity = bits.unsafe_ptr()
    var masked = len(mask) == n
    var base = 0
    for span in spans[]:
        for local in range(len(span.column)):
            var row = base + local
            if not span.column._valid(local) or (masked and not mask[row]):
                continue
            var text = StringSlice[ImmutAnyOrigin](
                unsafe_from_utf8=span.get(local)
            )
            try:
                var value = parse_bool(text)
                var flag = UInt8(1) << UInt8(row & 7)
                if value:
                    out[unsafe_offset=row // 8] |= flag
                validity[unsafe_offset=row // 8] |= flag
            except error:
                if strict:
                    _chunk_string_error(
                        DataType.BOOL, offset + row, text, String(error)
                    )
        base += len(span.column)
    return Series(
        input.name(), BoolColumn(values=values^, bits=bits^, length=n)
    )


@no_inline
def _cast_string_chunks(
    input: Series, target: DataType, strict: Bool, offset: Int, mask: List[Bool]
) raises -> Optional[Series]:
    var source = input.dtype()
    if source == DataType.STRING:
        if target == DataType.BOOL:
            if _string_cast_host(input):
                return _chunk_parse_bools(input, strict, offset, mask)
            return None
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if target == DataType.of(D):
                if _string_cast_host(input):
                    return _chunk_parse_strings[D](
                        input, target, strict, offset, mask
                    )
                return None
        return None
    if target == DataType.STRING:
        if source == DataType.BOOL:
            if _string_format_host[DType.bool](input):
                return _chunk_format_strings[DType.bool](input, mask)
            return None
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if source == DataType.of(D):
                if _string_format_host[D](input):
                    return _chunk_format_strings[D](input, mask)
                return None
    return None


def _cast_binary(
    input: Series,
    source: DataType,
    target: DataType,
    strict: Bool,
    offset: Int,
    mask: List[Bool],
) raises -> Series:
    """Convert string-like inputs to their UTF-8 bytes. Binary values cast
    only to string, and each must be valid UTF-8 (strict raises, otherwise
    the row is null)."""
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


@no_inline
def _primitive_cast_host[S: DType](input: Series) -> Bool:
    """Validate every cold physical window before observing any row."""
    ref chunks = input._chunked.value()[]
    if chunks.cache[].state.load() == 2:
        return False
    for storage in chunks.arrays:
        comptime if S == DType.bool:
            if not storage.isa[BoolColumn]():
                return False
            ref column = storage[BoolColumn]
            if not column._data.on_host() or not column._bits.on_host():
                return False
        else:
            if not storage.isa[Column[Scalar[S]]]():
                return False
            ref column = storage[Column[Scalar[S]]]
            if not column._data.on_host() or not column._bits.on_host():
                return False
    return True


@no_inline
def _primitive_cast_failure(
    source: DataType, target: DataType, row: Int, value: String, cause: String
) raises:
    raise Error(
        "cast from "
        + source.name()
        + " to "
        + target.name()
        + " failed at row "
        + String(row)
        + " for value '"
        + value
        + "': "
        + cause
    )


@no_inline
def _chunk_boolean_numeric[
    T: DType
](input: Series, mask: List[Bool]) raises -> Series:
    var n = len(input)
    var values = List[Scalar[T]](length=n, fill=0)
    var bits = List[UInt8](length=(n + 7) // 8, fill=0)
    var to = values.unsafe_ptr()
    var validity = bits.unsafe_ptr()
    var masked = len(mask) == n
    var base = 0
    for storage in input._chunked.value()[].arrays:
        ref column = storage[BoolColumn]
        for row in range(len(column)):
            var i = base + row
            if not column._valid(row) or (masked and not mask[i]):
                continue
            to[unsafe_offset=i] = Scalar[T](1 if column._get(row) else 0)
            validity[unsafe_offset=i // 8] |= UInt8(1) << UInt8(i & 7)
        base += len(column)
    return Series(input.name(), Column[Scalar[T]](values=values^, bits=bits^))


@no_inline
def _chunk_float_numeric[
    S: DType, T: DType
](
    input: Series, target: DataType, strict: Bool, offset: Int, mask: List[Bool]
) raises -> Series:
    var n = len(input)
    var values = List[Scalar[T]](length=n, fill=0)
    var bits = List[UInt8](length=(n + 7) // 8, fill=0)
    var to = values.unsafe_ptr()
    var validity = bits.unsafe_ptr()
    var lower = Float64(0)
    var upper = Float64(0)
    comptime if T.is_integral():
        var precision_bits = size_of[Scalar[T]]() * 8 - (
            1 if T.is_signed() else 0
        )
        upper = Float64(1)
        for _ in range(precision_bits):
            upper *= 2
        lower = -upper if T.is_signed() else Float64(0)
    var masked = len(mask) == n
    var base = 0
    for storage in input._chunked.value()[].arrays:
        ref column = storage[Column[Scalar[S]]]
        var source = column._ptr()
        for row in range(len(column)):
            var i = base + row
            if not column._valid(row) or (masked and not mask[i]):
                continue
            var native = source[unsafe_offset=row]
            var value = native.cast[DType.float64]()
            comptime if T.is_integral():
                var fits = not isnan(value) and not isinf(value)
                var whole = Float64(0)
                if fits:
                    whole = trunc(value)
                    fits = whole >= lower and whole < upper
                if not fits:
                    if strict:
                        _primitive_cast_failure(
                            input.dtype(),
                            target,
                            offset + i,
                            String(native),
                            "out of " + String(T) + " range",
                        )
                    continue
                to[unsafe_offset=i] = whole.cast[T]()
            else:
                to[unsafe_offset=i] = value.cast[T]()
            validity[unsafe_offset=i // 8] |= UInt8(1) << UInt8(i & 7)
        base += len(column)
    return Series(input.name(), Column[Scalar[T]](values=values^, bits=bits^))


@no_inline
def _chunk_numeric_boolean[
    S: DType
](input: Series, strict: Bool, offset: Int, mask: List[Bool]) raises -> Series:
    var n = len(input)
    var values = List[UInt8](length=(n + 7) // 8, fill=0)
    var bits = List[UInt8](length=(n + 7) // 8, fill=0)
    var to = values.unsafe_ptr()
    var validity = bits.unsafe_ptr()
    var masked = len(mask) == n
    var base = 0
    for storage in input._chunked.value()[].arrays:
        ref column = storage[Column[Scalar[S]]]
        var source = column._ptr()
        for row in range(len(column)):
            var i = base + row
            if not column._valid(row) or (masked and not mask[i]):
                continue
            var native = source[unsafe_offset=row]
            var truth: Bool
            comptime if S.is_floating_point():
                var value = native.cast[DType.float64]()
                if isnan(value):
                    if strict:
                        _primitive_cast_failure(
                            input.dtype(),
                            DataType.BOOL,
                            offset + i,
                            String(native),
                            "NaN has no Boolean value",
                        )
                    continue
                truth = value != 0
            else:
                truth = native != 0
            var flag = UInt8(1) << UInt8(i & 7)
            if truth:
                to[unsafe_offset=i // 8] |= flag
            validity[unsafe_offset=i // 8] |= flag
        base += len(column)
    return Series(
        input.name(), BoolColumn(values=values^, bits=bits^, length=n)
    )


@no_inline
def _cast_primitive_chunks(
    input: Series, target: DataType, strict: Bool, offset: Int, mask: List[Bool]
) raises -> Optional[Series]:
    var source = input.dtype()
    if source == DataType.BOOL:
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime T = NUMERIC_DTYPES[k]
            if target == DataType.of(T):
                if _primitive_cast_host[DType.bool](input):
                    return _chunk_boolean_numeric[T](input, mask)
                return None
        return None
    if target == DataType.BOOL:
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime S = NUMERIC_DTYPES[k]
            if source == DataType.of(S):
                if _primitive_cast_host[S](input):
                    return _chunk_numeric_boolean[S](
                        input, strict, offset, mask
                    )
                return None
        return None
    # Keep the existing unmasked infallible widening before all window scans.
    if (
        source == DataType.FLOAT32
        and target == DataType.FLOAT64
        and len(mask) == 0
    ):
        return None
    comptime for j in range(len(NUMERIC_DTYPES)):
        comptime S = NUMERIC_DTYPES[j]
        comptime if S.is_floating_point():
            if source == DataType.of(S):
                comptime for k in range(len(NUMERIC_DTYPES)):
                    comptime T = NUMERIC_DTYPES[k]
                    comptime if S != T:
                        if target == DataType.of(T):
                            if _primitive_cast_host[S](input):
                                return _chunk_float_numeric[S, T](
                                    input, target, strict, offset, mask
                                )
                            return None
                return None
    return None


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
                            # Null slots to zero, 64 rows a word: a word of
                            # valid rows is skipped whole rather than read
                            # bit by bit.
                            var i = 0
                            while i < n:
                                var word = _bits64(
                                    column._bits[],
                                    column._offset + i,
                                    column._offset + n,
                                )
                                var rows = min(64, n - i)
                                var full = (
                                    UInt64(0xFFFFFFFFFFFFFFFF) if (
                                        rows == 64
                                    ) else (UInt64(1) << UInt64(rows))
                                    - 1
                                )
                                var missing = ~word & full
                                while missing != 0:
                                    var j = Int(count_trailing_zeros(missing))
                                    to[unsafe_offset=i + j] = 0
                                    missing &= missing - 1
                                i += 64
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


def _cast_chunk_values[S: DType, T: DType](input: Series) raises -> Series:
    """Convert non-failing numeric spans into one output without merging input."""
    var values = List[Scalar[T]](unsafe_uninit_length=len(input))
    var bits = List[UInt8]()
    var to = values.unsafe_ptr()
    var base = 0
    ref chunks = input._chunked.value()[]
    for chunk in range(len(chunks.arrays)):
        ref column = chunks.arrays[chunk][Column[Scalar[S]]]
        var n = len(column)
        var source = column._ptr()
        for row in range(n):
            to[unsafe_offset=base + row] = source[unsafe_offset=row].cast[T]()
        if len(column._bits[]) != 0:
            # Match the contiguous path's null payloads while copying only
            # the sliced validity window, including unaligned boundaries.
            var row = 0
            while row < n:
                var word = _bits64(
                    column._bits[], column._offset + row, column._offset + n
                )
                var rows = min(64, n - row)
                var full = (
                    UInt64(0xFFFFFFFFFFFFFFFF) if rows
                    == 64 else (UInt64(1) << UInt64(rows)) - 1
                )
                var missing = ~word & full
                while missing != 0:
                    var bit = Int(count_trailing_zeros(missing))
                    to[unsafe_offset=base + row + bit] = 0
                    missing &= missing - 1
                row += 64
        _append_validity(bits, base, column._bits[], column._offset, n)
        base += n
    return Series(input.name(), Column[Scalar[T]](values=values^, bits=bits^))


def _cast_chunk_exact(
    input: Series, target: DataType
) raises -> Optional[Series]:
    # Logical types sharing numeric storage (categoricals, decimals and
    # temporal values) retain their existing converters and metadata rules.
    if input.dtype() != input.dtype().physical():
        return None
    comptime for j in range(len(NUMERIC_DTYPES)):
        comptime S = NUMERIC_DTYPES[j]
        if input._data.isa[Column[Scalar[S]]]():
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
                        return _cast_chunk_values[S, T](input)
            return None
    return None


@no_inline
def _cast_checked_chunk_values[
    S: DType, T: DType
](
    input: Series, target: DataType, strict: Bool, offset: Int, mask: List[Bool]
) raises -> Series:
    """Read observed integer windows once into an owned numeric result."""
    var n = len(input)
    var values = List[Scalar[T]](length=n, fill=0)
    var bits = List[UInt8](length=(n + 7) // 8, fill=0)
    var to = values.unsafe_ptr()
    var validity = bits.unsafe_ptr()
    var masked = len(mask) == n
    var base = 0
    for storage in input._chunked.value()[].arrays:
        ref column = storage[Column[Scalar[S]]]
        var source = column._ptr()
        for row in range(len(column)):
            var i = base + row
            if not column._valid(row) or (masked and not mask[i]):
                continue
            var value = source[unsafe_offset=row]
            comptime if T.is_integral():
                var wide = value.cast[DType.int128]()
                if (
                    wide < Scalar[T].MIN.cast[DType.int128]()
                    or wide > Scalar[T].MAX.cast[DType.int128]()
                ):
                    if strict:
                        raise Error(
                            "cast from "
                            + input.dtype().name()
                            + " to "
                            + target.name()
                            + " failed at row "
                            + String(offset + i)
                            + " for value '"
                            + String(value)
                            + "': out of "
                            + String(T)
                            + " range"
                        )
                    continue
                to[unsafe_offset=i] = value.cast[T]()
            else:
                # Every nonempty mask uses the existing Float64 intermediate,
                # even when its length differs and observation ignores it.
                to[unsafe_offset=i] = (
                    value.cast[T]() if len(mask)
                    == 0 else value.cast[DType.float64]().cast[T]()
                )
            validity[unsafe_offset=i // 8] |= UInt8(1) << UInt8(i & 7)
        base += len(column)
    return Series(input.name(), Column[Scalar[T]](values=values^, bits=bits^))


@no_inline
def _cast_checked_chunks(
    input: Series, target: DataType, strict: Bool, offset: Int, mask: List[Bool]
) raises -> Optional[Series]:
    """Admit only cold, host-readable plain integer windows."""
    ref chunks = input._chunked.value()[]
    if chunks.cache[].state.load() == 2:
        return None
    comptime for j in range(len(NUMERIC_DTYPES)):
        comptime S = NUMERIC_DTYPES[j]
        comptime if S.is_integral():
            if input.dtype() == DataType.of(S):
                if not _has_physical_storage[Column[Scalar[S]]](input):
                    return None
                for storage in chunks.arrays:
                    ref column = storage[Column[Scalar[S]]]
                    if not column._data.on_host() or not column._bits.on_host():
                        return None
                comptime for k in range(len(NUMERIC_DTYPES)):
                    comptime T = NUMERIC_DTYPES[k]
                    if target == DataType.of(T):
                        return _cast_checked_chunk_values[S, T](
                            input, target, strict, offset, mask
                        )
                return None
    return None


def _decimal_cast_canonical(dtype: DataType) raises -> Bool:
    if not dtype.is_decimal():
        return False
    var width = dtype.decimal_width()
    var precision = dtype.precision()
    var scale = dtype.scale()
    var most = 38 if width == 128 else 18 if width == 64 else 9
    if precision < 1 or precision > most or scale < 0 or scale > precision:
        return False
    return dtype == DataType.decimal(precision, scale, width)


def _decimal_cast_plain_numeric(dtype: DataType) -> Bool:
    comptime for k in range(len(NUMERIC_DTYPES)):
        if dtype == DataType.of(NUMERIC_DTYPES[k]):
            return True
    return False


@no_inline
def _decimal_cast_string_host(column: StringColumn) -> Bool:
    # Merging coded Strings also validates/combines dictionary state before
    # observing masks. Leave those preparation effects to the old adapter.
    if column._codes:
        return False
    var offset = column._offset
    var count = len(column)
    if offset < 0 or count <= 0:
        return False
    if column._is_view_storage():
        var storage = column._view_storage_unchecked()
        if column._bits.ptr() != storage._bits.ptr():
            return False
        if not storage._views.on_host() or not storage._bits.on_host():
            return False
        for buffer in storage._buffers[]:
            if not buffer.on_host():
                return False
        var rows = len(storage)
        var descriptors = len(storage._views[])
        if offset > rows or count > rows - offset:
            return False
        if offset > descriptors or count > descriptors - offset:
            return False
        var end = offset + count
        if len(storage._bits[]) != 0 and (
            end // 8 + Int(end % 8 != 0) > len(storage._bits[])
        ):
            return False
        for row in range(offset, end):
            var view = storage._view_unchecked(row)
            if not view.is_inline():
                var index = Int(view.buffer_index)
                if index >= len(storage._buffers[]):
                    return False
                var available = len(storage._buffers[][index])
                var length = Int(view.length)
                if length > available or Int(view.offset) > available - length:
                    return False
        return True
    if (
        not column._bytes.on_host()
        or not column._offsets.on_host()
        or not column._bits.on_host()
    ):
        return False
    var entries = len(column._offsets[])
    if entries == 0 or offset > entries - 1 or count > entries - 1 - offset:
        return False
    var end = offset + count
    if len(column._bits[]) != 0 and (
        end // 8 + Int(end % 8 != 0) > len(column._bits[])
    ):
        return False
    var available = Int64(len(column._bytes[]))
    var previous = column._offsets[][offset]
    if previous < 0 or previous > available:
        return False
    for row in range(offset + 1, end + 1):
        var next = column._offsets[][row]
        if next < previous or next > available:
            return False
        previous = next
    return True


@no_inline
def _decimal_cast_storage_count(storage: Storage, source: DataType) -> Int:
    if source == DataType.STRING:
        if not storage.isa[StringColumn]():
            return -1
        ref column = storage[StringColumn]
        return len(column) if _decimal_cast_string_host(column) else -1
    if source == DataType.BOOL:
        if not storage.isa[BoolColumn]():
            return -1
        ref column = storage[BoolColumn]
        if not column._data.on_host() or not column._bits.on_host():
            return -1
        var count = len(column)
        if count <= 0 or column._offset < 0 or count > Int.MAX - column._offset:
            return -1
        var end = column._offset + count
        var bytes = end // 8 + Int(end % 8 != 0)
        if bytes > len(column._data) or (
            len(column._bits) != 0 and bytes > len(column._bits)
        ):
            return -1
        return count
    comptime for k in range(len(SCALAR_DTYPES)):
        comptime D = SCALAR_DTYPES[k]
        if source.storage() and source.storage().value() == D:
            if not storage.isa[Column[Scalar[D]]]():
                return -1
            ref column = storage[Column[Scalar[D]]]
            if not column._data.on_host() or not column._bits.on_host():
                return -1
            var count = len(column)
            if (
                count <= 0
                or column._offset < 0
                or column._offset > len(column._data)
            ):
                return -1
            if count > len(column._data) - column._offset:
                return -1
            var end = column._offset + count
            if len(column._bits) != 0 and (
                end // 8 + Int(end % 8 != 0) > len(column._bits)
            ):
                return -1
            return count
    return -1


@no_inline
def _decimal_cast_chunks_host(input: Series, source: DataType) -> Bool:
    ref chunks = input._chunked.value()[]
    if chunks.cache[].state.load() == 2:
        return False
    if len(chunks.arrays) == 0 or len(chunks.arrays) != len(chunks.ends):
        return False
    var representative = False
    if source == DataType.STRING:
        representative = input._data.isa[StringColumn]()
    elif source == DataType.BOOL:
        representative = input._data.isa[BoolColumn]()
    else:
        var physical = source.storage()
        comptime for k in range(len(SCALAR_DTYPES)):
            comptime D = SCALAR_DTYPES[k]
            if physical and physical.value() == D:
                representative = input._data.isa[Column[Scalar[D]]]()
    if not representative:
        return False
    var total = 0
    for k in range(len(chunks.arrays)):
        var count = _decimal_cast_storage_count(chunks.arrays[k], source)
        if count <= 0 or count > Int.MAX - total:
            return False
        total += count
        if chunks.ends[k] != total:
            return False
    return True


# Decimal conversions share one runtime source window and specialize only
# the final physical destination, rather than every source/destination pair.
comptime _DecimalIntegerReader = def(Int, Int) thin -> Int128
comptime _DecimalTextReader = def(Int, Int) thin -> String


def _decimal_read_integer[D: DType](address: Int, row: Int) -> Int128:
    return Pointer[Scalar[D], ImmutAnyOrigin](unsafe_from_address=address)[
        unsafe_offset=row
    ].cast[DType.int128]()


def _decimal_read_text[D: DType](address: Int, row: Int) -> String:
    return String(
        Pointer[Scalar[D], ImmutAnyOrigin](unsafe_from_address=address)[
            unsafe_offset=row
        ]
    )


def _decimal_read_bool(address: Int, row: Int) -> Int128:
    var bytes = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=address)
    return Int128((bytes[unsafe_offset=row // 8] >> UInt8(row % 8)) & 1)


def _decimal_bool_text(address: Int, row: Int) -> String:
    return "true" if _decimal_read_bool(address, row) != 0 else "false"


def _decimal_unused_integer(address: Int, row: Int) -> Int128:
    return 0


def _decimal_unused_text(address: Int, row: Int) -> String:
    return String()


struct _DecimalCastWindow(Movable):
    """A physical owner and prepared readers; no per-row Series/owner rebuild."""

    var storage: Storage
    var source: DataType
    var address: Int
    var bits: Int
    var offset: Int
    var row_offset: Int
    var length: Int
    var integer_reader: _DecimalIntegerReader
    var text_reader: _DecimalTextReader

    def __init__(out self, storage: Storage, source: DataType):
        self.storage = storage.copy()
        self.source = source.copy()
        self.address = 0
        self.bits = 0
        self.offset = 0
        self.row_offset = 0
        self.length = 0
        self.integer_reader = _decimal_unused_integer
        self.text_reader = _decimal_unused_text
        if source == DataType.STRING:
            ref column = self.storage[StringColumn]
            self.offset = column._offset
            self.length = len(column)
            if len(column._bits) != 0:
                self.bits = Int(column._bits[].unsafe_ptr())
            return
        if source == DataType.BOOL:
            ref column = self.storage[BoolColumn]
            self.offset = column._offset
            self.row_offset = column._offset
            self.length = len(column)
            self.address = Int(column._data[].unsafe_ptr())
            if len(column._bits) != 0:
                self.bits = Int(column._bits[].unsafe_ptr())
            self.integer_reader = _decimal_read_bool
            self.text_reader = _decimal_bool_text
            return
        comptime for k in range(len(SCALAR_DTYPES)):
            comptime D = SCALAR_DTYPES[k]
            if self.storage.isa[Column[Scalar[D]]]():
                ref column = self.storage[Column[Scalar[D]]]
                self.offset = column._offset
                self.length = len(column)
                self.address = Int(column._ptr())
                if len(column._bits) != 0:
                    self.bits = Int(column._bits[].unsafe_ptr())
                comptime if D.is_integral():
                    self.integer_reader = _decimal_read_integer[D]
                self.text_reader = _decimal_read_text[D]
                return

    @always_inline
    def valid(self, row: Int) -> Bool:
        if self.bits == 0:
            return True
        var bit = self.offset + row
        var bytes = Pointer[UInt8, ImmutAnyOrigin](
            unsafe_from_address=self.bits
        )
        return (bytes[unsafe_offset=bit // 8] >> UInt8(bit % 8)) & 1 != 0

    @always_inline
    def integer(self, row: Int) -> Int128:
        return self.integer_reader(self.address, self.row_offset + row)

    def text(self, row: Int) -> String:
        if self.source.is_decimal():
            return format_decimal(self.integer(row), self.source.scale())
        if self.source == DataType.STRING:
            return String(self.storage[StringColumn]._get(row))
        return self.text_reader(self.address, self.row_offset + row)


struct _DecimalCastPlan(Copyable):
    var source: DataType
    var target: DataType
    var factor: Int128
    var limit: Int128
    var grow: Bool

    def __init__(out self, source: DataType, target: DataType):
        self.source = source.copy()
        self.target = target.copy()
        self.factor = 1
        self.limit = 1
        self.grow = True
        if target.is_decimal():
            self.limit = precision_limit(target.precision())
            if source.is_decimal():
                self.grow = source.scale() <= target.scale()
                self.factor = pow10(
                    target.scale()
                    - source.scale() if self.grow else source.scale()
                    - target.scale()
                )
            else:
                self.factor = pow10(target.scale())
        elif source.is_decimal():
            self.factor = pow10(source.scale())

    @no_inline
    def integer(self, window: _DecimalCastWindow, row: Int) raises -> Int128:
        if not self.target.is_decimal():
            return divide_half_even(window.integer(row), self.factor)
        if self.source == DataType.STRING:
            return parse_decimal(
                window.storage[StringColumn]._get(row), self.target
            )
        if self.source.is_decimal():
            var raw = window.integer(row)
            if self.grow:
                raw *= self.factor
            else:
                raw = divide_half_even(raw, self.factor)
            return check_limit(raw, self.limit, self.target)
        if self.source.is_integer() or self.source == DataType.BOOL:
            return check_limit(
                window.integer(row) * self.factor, self.limit, self.target
            )
        # Preserve original-width scalar formatting and the exact parser,
        # including exponent/excess-digit errors and Int128 accumulation wrap.
        var rendered = window.text(row)
        var value = parse_decimal(StringSlice(rendered), self.target)
        _ = rendered^
        return value

    def float32(self, window: _DecimalCastWindow, row: Int) -> Float32:
        return (Float64(window.integer(row)) / Float64(self.factor)).cast[
            DType.float32
        ]()


@no_inline
def _decimal_cast_failure(
    plan: _DecimalCastPlan,
    window: _DecimalCastWindow,
    row: Int,
    global_row: Int,
    offset: Int,
    message: String,
) raises:
    raise Error(
        "cast from "
        + plan.source.name()
        + " to "
        + plan.target.name()
        + " failed at row "
        + String(offset + global_row)
        + " for value '"
        + window.text(row)
        + "': "
        + message
    )


@always_inline
def _decimal_cast_validity(
    mut bits: List[UInt8], row: Int, valid: Bool, n: Int, lazy: Bool
):
    if lazy:
        _append_validity_bit(bits, row, valid, n)
    elif valid:
        bits[row // 8] |= UInt8(1) << UInt8(row % 8)


@no_inline
def _decimal_cast_numeric[
    D: DType
](
    input: Series,
    plan: _DecimalCastPlan,
    strict: Bool,
    offset: Int,
    mask: List[Bool],
) raises -> Series:
    var n = len(input)
    var values = List[Scalar[D]](length=n, fill=0)
    # The existing unmasked integer-to-decimal route represents an all-valid
    # output with no bitmap. Generic routes always produce packed validity.
    var lazy = (
        plan.target.is_decimal() and plan.source.is_integer() and len(mask) == 0
    )
    var bits = List[UInt8]()
    if not lazy:
        bits = List[UInt8](length=(n + 7) // 8, fill=0)
    var out = values.unsafe_ptr()
    var masked = len(mask) == n
    var base = 0
    for storage in input._chunked.value()[].arrays:
        var window = _DecimalCastWindow(storage, plan.source)
        for row in range(window.length):
            var global_row = base + row
            if not window.valid(row) or (masked and not mask[global_row]):
                _decimal_cast_validity(bits, global_row, False, n, lazy)
                continue
            var valid = False
            try:
                comptime if D.is_floating_point():
                    out[unsafe_offset=global_row] = plan.float32(
                        window, row
                    ).cast[D]()
                else:
                    var value = plan.integer(window, row)
                    if plan.target.is_decimal():
                        out[unsafe_offset=global_row] = value.cast[D]()
                    else:
                        out[unsafe_offset=global_row] = _to_integer[D](value)
                valid = True
            except e:
                if strict:
                    _decimal_cast_failure(
                        plan, window, row, global_row, offset, String(e)
                    )
            _decimal_cast_validity(bits, global_row, valid, n, lazy)
        base += window.length
    var result = Series(
        input.name(), Column[Scalar[D]](values=values^, bits=bits^)
    )
    if plan.target.is_decimal():
        return result.with_dtype(plan.target)
    return result^


@no_inline
def _decimal_cast_bool(
    input: Series,
    plan: _DecimalCastPlan,
    strict: Bool,
    offset: Int,
    mask: List[Bool],
) raises -> Series:
    var n = len(input)
    var values = List[UInt8](length=(n + 7) // 8, fill=0)
    var bits = List[UInt8](length=(n + 7) // 8, fill=0)
    var masked = len(mask) == n
    var base = 0
    for storage in input._chunked.value()[].arrays:
        var window = _DecimalCastWindow(storage, plan.source)
        for row in range(window.length):
            var global_row = base + row
            if not window.valid(row) or (masked and not mask[global_row]):
                continue
            try:
                if plan.integer(window, row) != 0:
                    values[global_row // 8] |= UInt8(1) << UInt8(global_row % 8)
                _decimal_cast_validity(bits, global_row, True, n, False)
            except e:
                if strict:
                    _decimal_cast_failure(
                        plan, window, row, global_row, offset, String(e)
                    )
        base += window.length
    return Series(
        input.name(), BoolColumn(values=values^, bits=bits^, length=n)
    )


@no_inline
def _decimal_cast_string(
    input: Series,
    plan: _DecimalCastPlan,
    mask: List[Bool],
) raises -> Series:
    var n = len(input)
    var out = StringBuilder(n)
    var masked = len(mask) == n
    var base = 0
    for storage in input._chunked.value()[].arrays:
        var window = _DecimalCastWindow(storage, plan.source)
        for row in range(window.length):
            if window.valid(row) and (not masked or mask[base + row]):
                out.append(window.text(row))
            else:
                out.append_null()
        base += window.length
    return Series(input.name(), out^.finish())


@no_inline
def _cast_decimal_chunks(
    input: Series,
    target: DataType,
    strict: Bool,
    offset: Int,
    mask: List[Bool],
) raises -> Optional[Series]:
    var source = input.dtype()
    # Identity and specialized Float64 semantics remain at their existing
    # earlier dispatch/adapter. No source or target outside this family scans.
    if source == target or (
        not source.is_decimal() and not target.is_decimal()
    ):
        return None
    if source.is_decimal() and target == DataType.FLOAT64:
        return None
    if input._chunked.value()[].cache[].state.load() == 2:
        return None
    if source.is_decimal():
        if not _decimal_cast_canonical(source):
            return None
    elif (
        source != DataType.BOOL
        and source != DataType.STRING
        and not _decimal_cast_plain_numeric(source)
    ):
        return None
    if target.is_decimal():
        if not _decimal_cast_canonical(target):
            return None
    elif (
        target != DataType.BOOL
        and target != DataType.STRING
        and target != DataType.FLOAT32
    ):
        if not _decimal_cast_plain_numeric(target) or not target.is_integer():
            return None
    if not _decimal_cast_chunks_host(input, source):
        return None
    var plan = _DecimalCastPlan(source, target)
    if target == DataType.STRING:
        return _decimal_cast_string(input, plan, mask)
    if target == DataType.BOOL:
        return _decimal_cast_bool(input, plan, strict, offset, mask)
    if target.is_decimal():
        if target.decimal_width() == 32:
            return _decimal_cast_numeric[DType.int32](
                input, plan, strict, offset, mask
            )
        if target.decimal_width() == 64:
            return _decimal_cast_numeric[DType.int64](
                input, plan, strict, offset, mask
            )
        return _decimal_cast_numeric[DType.int128](
            input, plan, strict, offset, mask
        )
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if target == DataType.of(D):
            comptime if D != DType.float64:
                return _decimal_cast_numeric[D](
                    input, plan, strict, offset, mask
                )
    return None


def cast_series(
    input: Series, target: DataType, strict: Bool, offset: Int, mask: List[Bool]
) raises -> Series:
    """Convert every valid, observed row; others become null."""
    if input.dtype().is_nested() or target.is_nested():
        raise Error(
            "cannot cast "
            + input.dtype().name()
            + " to "
            + target.name()
            + ": list and struct casts are not supported yet"
        )
    var source = _dtype(input)
    if source.is_geometry() or target.is_geometry():
        if source == target:
            return input.copy()
        raise Error("Geometry casts require explicit WKB import/export")
    # Accepted identities and byte retags need only the Series metadata.
    # Keep their array windows and owners, including with an inactive mask.
    if source.is_categorical() and target.is_categorical():
        if not target.has_dictionary() or source == target:
            return input.copy()
        # A different dictionary still validates and recodes all source labels.
        # Keep its existing contiguous adapter, without comparing twice.
        var whole = input.rechunk() if input.is_chunked() else input.copy()
        return recode(whole, target)
    if source == target:
        return input.copy()
    if source == DataType.STRING and target == DataType.BINARY:
        return input.with_dtype(DataType.BINARY)
    if input.is_chunked():
        var text_result = _cast_string_chunks(
            input, target, strict, offset, mask
        )
        if text_result:
            return text_result.take()
    if input.is_chunked():
        var decimal = _cast_decimal_chunks(input, target, strict, offset, mask)
        if decimal:
            return decimal.take()
        var primitive = _cast_primitive_chunks(
            input, target, strict, offset, mask
        )
        if primitive:
            return primitive.take()
        if len(mask) == 0:
            var exact = _cast_chunk_exact(input, target)
            if exact:
                return exact.take()
        var checked = _cast_checked_chunks(input, target, strict, offset, mask)
        if checked:
            return checked.take()
        return cast_series(input.rechunk(), target, strict, offset, mask)
    if source.is_decimal() and target == DataType.FLOAT64:
        # Under a mask too: the conversion cannot fail on an inactive row,
        # and its result there goes unread, so every row converts through
        # the pointer. The row path under a `when` mask cost TPC-DS
        # decimal q9 (fifteen conditional means of cast columns) 51 ms
        # against 3 for the DOUBLE variant.
        return _decimal_to_float(input)
    if source.is_categorical() or target.is_categorical():
        # Categoricals convert through their values (#106): decode, cast
        # the strings, and encode when the target is categorical.
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
    if source.is_binary() or target.is_binary():
        return _cast_binary(input, source, target, strict, offset, mask)
    if source.is_temporal() or target.is_temporal():
        var observed = input.copy()
        return cast_temporal(observed, source, target, strict, mask)
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
    if target.is_decimal() and source.is_integer() and len(mask) == 0:
        var scaled = _integers_to_decimal(input, target, strict, offset)
        if scaled:
            return scaled.take()
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
        return Series(input.name(), BoolColumn(targets, valid))
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
