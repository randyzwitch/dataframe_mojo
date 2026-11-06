"""Normalized sort keys for every column type (DuckDB 2.0's create_sort_key).

A row's key is one byte string whose unsigned byte order is the row order:
for each key column a byte saying null or valid (1 and 2, swapped for
nulls last; never flipped), then the value's bytes, flipped for a
descending column (create_sort_key.cpp, `SortKeyVectorData`,
`TemplatedConstructSortKeyInternal`). A null writes its byte alone.

- Integers are big-endian with the sign bit toggled (Radix::EncodeData);
  128-bit decimals are their upper word that way, then the lower word.
- Floats are Radix::EncodeDouble/EncodeFloat: -0.0 equals 0.0, NaN is the
  greatest value, infinities next to it.
- Booleans are one byte.
- Strings (`SortKeyVarcharOperator`) are every byte plus one, then a zero
  delimiter, so a prefix comes first; binary values
  (`SortKeyBlobOperator`) escape bytes 0 and 1 with a 1 instead.
- Categorical keys are first replaced by their value's rank in string
  order (`sort_ranks`), so they sort as their strings.

The key goes into a fixed record when its widest form fits, as
TupleDataLayout::Initialize(orders) decides: with the 8-byte payload, at
most 16, 24 or 32 bytes take `PAYLOAD_FIXED_16/24/32`, anything wider or of
unknown length `PAYLOAD_VARIABLE_32` (8 inlined bytes, the size, and the
key's bytes elsewhere). A string column's widest form is 1 + its longest
value + 1 bytes, the bound DuckDB reads from the column's statistics; here
the column is in memory and measured. The payload is the row's index into
the sorted input.

`maintain_order` appends the row index as a last key: equal keys then
keep input order. Otherwise their order is unspecified, as Polars leaves
it.
"""
from std.bit import byte_swap
from std.memory import Pointer, bitcast
from std.sys import bit_width_of

from dataframe.categorical import codes_to_ranks, dictionary_ranks
from dataframe.dtype import DataType
from dataframe.series import Series
from dataframe.sorting.algorithms import SkaInfo
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.storage.string import StringColumn
from dataframe.text.bytes import _Bytes
from dataframe.vector.types import (
    PHYSICAL_BOOL,
    PHYSICAL_INT128,
    PHYSICAL_UINT32,
    PHYSICAL_VARCHAR,
    StringT,
    UnifiedVectorFormat,
    physical_of,
    string_data,
)

# STANDARD_VECTOR_SIZE and DEFAULT_ROW_GROUP_SIZE (vector_size.hpp,
# storage_info.hpp): the vector and the row group, which is both a scan
# morsel and a merge partition (sort.cpp, Sort::Finalize).
comptime STANDARD_VECTOR_SIZE = 2048
comptime DEFAULT_ROW_GROUP_SIZE = 122880

comptime _NULL_FIRST_BYTE = UInt8(1)
comptime _NULL_LAST_BYTE = UInt8(2)

comptime _ENC_SIGNED = 0
comptime _ENC_UNSIGNED = 1
comptime _ENC_INT128 = 2
comptime _ENC_FLOAT = 3
comptime _ENC_BOOL = 4
comptime _ENC_VARCHAR = 5
comptime _ENC_BLOB = 6
# The row index, for `maintain_order`: 8 bytes, never null.
comptime _ENC_ROW = 7

comptime _LSB = UInt64(0x0101010101010101)
comptime _LOW7 = UInt64(0x7F7F7F7F7F7F7F7F)
comptime _MSB = UInt64(0x8080808080808080)


@fieldwise_init
struct SortKeyColumn(Copyable, Movable):
    """One key column and how it encodes (SortKeyVectorData)."""

    var series: Series
    var encoding: Int
    # Value bytes of a fixed-width encoding; 0 for strings.
    var width: Int
    var descending: Bool
    var null_byte: UInt8
    var valid_byte: UInt8
    var nulls: Bool
    # Strings: the longest value's bytes.
    var max_length: Int
    # A categorical key: each code's rank, by which its rows sort.
    var ranks: List[UInt32]
    # The chunk vector the key is bound to (`rebind_vectors`), read as
    # its UnifiedVectorFormat.
    var format: UnifiedVectorFormat

    def constant_size(self) -> Bool:
        return self.encoding != _ENC_VARCHAR and self.encoding != _ENC_BLOB


def _encoding_of(series: Series) raises -> Tuple[Int, Int]:
    """(encoding, value bytes) of a column's storage (its first chunk's: a
    chunked series keeps its arrays apart)."""
    var column = series.chunks()[0].copy() if series.is_chunked() else (
        series.copy()
    )
    if column.dtype().is_nested():
        raise Error(
            "cannot sort by a "
            + column.dtype().name()
            + " column: "
            + column.name()
        )
    if column._data.isa[StringColumn]():
        if column.dtype().physical() == DataType.STRING and not (
            column.dtype()._is_bytes()
        ):
            return (_ENC_VARCHAR, 0)
        return (_ENC_BLOB, 0)
    if column._data.isa[BoolColumn]():
        return (_ENC_BOOL, 1)
    if column._data.isa[Column[Int128]]():
        return (_ENC_INT128, 16)
    if column._data.isa[Column[Float64]]():
        return (_ENC_FLOAT, 8)
    if column._data.isa[Column[Float32]]():
        return (_ENC_FLOAT, 4)
    if column._data.isa[Column[Int64]]():
        return (_ENC_SIGNED, 8)
    if column._data.isa[Column[Int32]]():
        return (_ENC_SIGNED, 4)
    if column._data.isa[Column[Int16]]():
        return (_ENC_SIGNED, 2)
    if column._data.isa[Column[Int8]]():
        return (_ENC_SIGNED, 1)
    if column._data.isa[Column[UInt64]]():
        return (_ENC_UNSIGNED, 8)
    if column._data.isa[Column[UInt32]]():
        return (_ENC_UNSIGNED, 4)
    if column._data.isa[Column[UInt16]]():
        return (_ENC_UNSIGNED, 2)
    if column._data.isa[Column[UInt8]]():
        return (_ENC_UNSIGNED, 1)
    raise Error("cannot sort by column " + column.name())


def _longest(column: Series) -> Int:
    var longest = 0
    for part in column.chunks():
        var rows = _Bytes(part._data[StringColumn])
        for i in range(len(part)):
            longest = max(longest, len(rows.get(i)))
    return longest


struct SortKeyLayout(Copyable, Movable):
    """The key of a sort (Sort's create_sort_key binding plus the key
    TupleDataLayout): its columns, its record form, and the radix sort's
    settings."""

    var columns: List[SortKeyColumn]
    var rows: Int
    # Whether the columns are bound to a chunk's vectors (`rebind_vectors`)
    # rather than to at-rest columns (`rebind`).
    var vectors: Bool
    # Key words of a fixed record (1-3); 0 when variable.
    var words: Int
    var variable: Bool
    var sort_width: Int
    var info: SkaInfo

    def __init__(
        out self,
        keys: List[Series],
        descending: List[Bool],
        nulls_last: List[Bool],
        maintain_order: Bool,
        *,
        statistics: Bool = True,
    ) raises:
        """`statistics=False` lays the key out from the column types alone,
        for keys whose rows arrive later in chunks (`rebind`): a string
        key's length is unknown, so the key is variable, and every column
        may hold nulls."""
        if len(keys) == 0:
            raise Error("sort requires at least one column")
        if len(descending) != len(keys) or len(nulls_last) != len(keys):
            raise Error(
                "descending and nulls_last must have one entry per sort column"
            )
        var rows = len(keys[0])
        var columns = List[SortKeyColumn](capacity=len(keys) + 1)
        for k in range(len(keys)):
            var column = keys[k].copy()
            if len(column) != rows:
                raise Error("sort key columns must have the same length")
            var ranks = List[UInt32]()
            if column.dtype().is_categorical():
                ranks = dictionary_ranks(column.dtype())
                column = codes_to_ranks(column, ranks)
            var encoding = _encoding_of(column)
            var null_byte = _NULL_FIRST_BYTE
            var valid_byte = _NULL_LAST_BYTE
            if nulls_last[k]:
                null_byte = _NULL_LAST_BYTE
                valid_byte = _NULL_FIRST_BYTE
            var nulls = not statistics or column.null_count() > 0
            var longest = (
                _longest(column) if (
                    encoding[0] == _ENC_VARCHAR or encoding[0] == _ENC_BLOB
                )
                and statistics else 0
            )
            columns.append(
                SortKeyColumn(
                    column^,
                    encoding[0],
                    encoding[1],
                    descending[k],
                    null_byte,
                    valid_byte,
                    nulls,
                    longest,
                    ranks^,
                    UnifiedVectorFormat(),
                )
            )
        if maintain_order:
            columns.append(
                SortKeyColumn(
                    keys[0].copy(),
                    _ENC_ROW,
                    8,
                    False,
                    _NULL_FIRST_BYTE,
                    _NULL_LAST_BYTE,
                    False,
                    0,
                    List[UInt32](),
                    UnifiedVectorFormat(),
                )
            )
        # TupleDataLayout::Initialize(orders): the widest key, and the null
        # bytes every key shares while the key is still all fixed-width
        # and valid.
        var all_constant = True
        var constant_size = 0
        var sort_width = 0
        var skippable = 0
        var all_valid_and_truly_constant = True
        for c in columns:
            if c.encoding == _ENC_ROW:
                constant_size += 8
                if sort_width >= 0:
                    sort_width += 8
                all_valid_and_truly_constant = False
                continue
            if (
                all_valid_and_truly_constant
                and not c.nulls
                and c.constant_size()
                and sort_width < 7
            ):
                skippable |= 1 << sort_width
            else:
                all_valid_and_truly_constant = False
            if c.constant_size():
                constant_size += 1 + c.width
                if sort_width >= 0:
                    sort_width += 1 + c.width
            else:
                all_constant = False
                if (
                    c.encoding == _ENC_VARCHAR
                    and sort_width >= 0
                    and statistics
                ):
                    sort_width += 1 + c.max_length + 1
                else:
                    sort_width = -1
        # create_sort_key returns a BIGINT when every key is fixed and the
        # whole key fits in 8 bytes; the record is then 16 bytes.
        var row_width = (
            16 if (all_constant and constant_size <= 8) else sort_width + 8
        )
        var variable = False
        var words = 0
        if sort_width < 0 or row_width > 32:
            variable = True
        elif row_width <= 16:
            words = 1
        elif row_width <= 24:
            words = 2
        else:
            words = 3
        var ska_sort_width = 8 if sort_width < 0 else min(sort_width, 8)
        self.columns = columns^
        self.rows = rows
        self.vectors = False
        self.words = words
        self.variable = variable
        self.sort_width = sort_width
        self.info = SkaInfo(variable or words != 1, ska_sort_width, skippable)

    def record_words(self) -> Int:
        return 4 if self.variable else self.words + 1

    def rebind(mut self, keys: List[Series]) raises:
        """The same key over other rows: `keys` are the key columns of a
        new chunk, of the types the layout was made for."""
        self.vectors = False
        if len(keys) == 0:
            self.rows = 0
            return
        self.rows = len(keys[0])
        for k in range(len(keys)):
            var column = keys[k].copy()
            if column.dtype().is_categorical():
                # The chunk's codes through the ranks found once.
                column = codes_to_ranks(column, self.columns[k].ranks)
            self.columns[k].series = column^

    def rebind_vectors(
        mut self, formats: List[UnifiedVectorFormat], count: Int
    ):
        """The same key over a chunk's `count` rows, read where they are:
        `formats` are the key columns' vectors as UnifiedVectorFormat
        (create_sort_key.cpp builds its SortKeyVectorData from
        ToUnifiedFormat), valid while the chunk is. A categorical key's
        codes are read through its ranks."""
        self.vectors = True
        self.rows = count
        for k in range(len(formats)):
            self.columns[k].format = formats[k]


@always_inline
def _store_be[D: DType](address: Int, value: Scalar[D]):
    comptime if bit_width_of[D]() == 8:
        Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=address)[] = value
    else:
        Pointer[Scalar[D], MutAnyOrigin](
            unsafe_from_address=address
        ).unsafe_store[alignment=1](byte_swap(value))


@always_inline
def _encode_double(x: Float64) -> UInt64:
    """Radix::EncodeDouble."""
    if x == 0:
        return UInt64(1) << 63
    if x != x:
        return UInt64.MAX
    if x > Float64.MAX_FINITE:
        return UInt64.MAX - 1
    if x < -Float64.MAX_FINITE:
        return 0
    var buff = bitcast[DType.uint64](x)
    if buff < (UInt64(1) << 63):
        return buff + (UInt64(1) << 63)
    return ~buff


@always_inline
def _encode_float(x: Float32) -> UInt32:
    """Radix::EncodeFloat."""
    if x == 0:
        return UInt32(1) << 31
    if x != x:
        return UInt32.MAX
    if x > Float32.MAX_FINITE:
        return UInt32.MAX - 1
    if x < -Float32.MAX_FINITE:
        return 0
    var buff = bitcast[DType.uint32](x)
    if (buff & (UInt32(1) << 31)) == 0:
        return buff | (UInt32(1) << 31)
    return ~buff


@always_inline
def _blob_length(bytes: Span[UInt8, ImmutAnyOrigin]) -> Int:
    """SortKeyBlobOperator::GetEncodeLength."""
    var escaped = 0
    for b in range(len(bytes)):
        if bytes.unsafe_ptr()[unsafe_offset=b] <= 1:
            escaped += 1
    return len(bytes) + escaped + 1


@always_inline
def _encode_varchar(
    address: Int, bytes: Span[UInt8, ImmutAnyOrigin], flip: Bool
) -> Int:
    """SortKeyVarcharOperator::Encode: every byte plus one, word at a
    time, then the delimiter; returns the bytes written."""
    var data = Int(bytes.unsafe_ptr())
    var n = len(bytes)
    var mask = UInt64.MAX if flip else UInt64(0)
    var pos = 0
    while pos + 8 <= n:
        var word = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=data + pos
        ).unsafe_load[alignment=1]()
        var encoded = ((word & _LOW7) + _LSB) ^ (word & _MSB)
        Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=address + pos
        ).unsafe_store[alignment=1](encoded ^ mask)
        pos += 8
    var out = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)
    var flip_byte = UInt8(0xFF) if flip else UInt8(0)
    while pos < n:
        out[unsafe_offset=pos] = (
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=data)[
                unsafe_offset=pos
            ]
            + 1
        ) ^ flip_byte
        pos += 1
    out[unsafe_offset=n] = flip_byte
    return n + 1


@always_inline
def _encode_blob(
    address: Int, bytes: Span[UInt8, ImmutAnyOrigin], flip: Bool
) -> Int:
    """SortKeyBlobOperator::Encode: bytes 0 and 1 escaped with a 1."""
    var out = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)
    var flip_byte = UInt8(0xFF) if flip else UInt8(0)
    var at = 0
    for b in range(len(bytes)):
        var byte = bytes.unsafe_ptr()[unsafe_offset=b]
        if byte <= 1:
            out[unsafe_offset=at] = UInt8(1) ^ flip_byte
            at += 1
        out[unsafe_offset=at] = byte ^ flip_byte
        at += 1
    out[unsafe_offset=at] = flip_byte
    return at + 1


@always_inline
def _order_bits[D: DType, U: DType](value: Scalar[D]) -> Scalar[U]:
    """Radix::EncodeData's value, before the byte swap."""
    comptime if D == DType.float64:
        return rebind[Scalar[U]](_encode_double(rebind[Float64](value)))
    elif D == DType.float32:
        return rebind[Scalar[U]](_encode_float(rebind[Float32](value)))
    elif D.is_signed():
        return bitcast[U](value) ^ (
            Scalar[U](1) << Scalar[U](bit_width_of[D]() - 1)
        )
    else:
        return bitcast[U](value)


def _encode_numeric[
    D: DType
](
    column: Column[Scalar[D]],
    c: SortKeyColumn,
    first: Int,
    count: Int,
    cursors: Pointer[Int, MutAnyOrigin],
):
    """Rows [first, first + count) of a fixed-width column, each at its
    cursor, which moves past it."""
    var values = column.unsafe_values()
    var nulls = c.nulls and column.null_count() > 0
    comptime B = bit_width_of[D]()
    comptime U = DType.uint64 if B == 64 else (
        DType.uint32 if B == 32 else (DType.uint16 if B == 16 else DType.uint8)
    )
    var flip = Scalar[U].MAX if c.descending else Scalar[U](0)
    for i in range(count):
        var at = cursors[unsafe_offset=i]
        var row = first + i
        if nulls and not column._valid(row):
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)[] = c.null_byte
            cursors[unsafe_offset=i] = at + 1
            continue
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)[] = c.valid_byte
        var bits = _order_bits[D, U](values[unsafe_offset=row])
        _store_be[U](at + 1, bits ^ flip)
        cursors[unsafe_offset=i] = at + 1 + B // 8


def _encode_int128(
    column: Column[Int128],
    c: SortKeyColumn,
    first: Int,
    count: Int,
    cursors: Pointer[Int, MutAnyOrigin],
):
    var values = column.unsafe_values()
    var nulls = c.nulls and column.null_count() > 0
    var flip = UInt64.MAX if c.descending else UInt64(0)
    for i in range(count):
        var at = cursors[unsafe_offset=i]
        var row = first + i
        if nulls and not column._valid(row):
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)[] = c.null_byte
            cursors[unsafe_offset=i] = at + 1
            continue
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)[] = c.valid_byte
        var value = values[unsafe_offset=row]
        var upper = (value >> 64).cast[DType.uint64]() ^ (UInt64(1) << 63)
        var lower = value.cast[DType.uint64]()
        _store_be[DType.uint64](at + 1, upper ^ flip)
        _store_be[DType.uint64](at + 9, lower ^ flip)
        cursors[unsafe_offset=i] = at + 17


def _encode_bool(
    column: BoolColumn,
    c: SortKeyColumn,
    first: Int,
    count: Int,
    cursors: Pointer[Int, MutAnyOrigin],
):
    var nulls = c.nulls and column.null_count() > 0
    var flip = UInt8(0xFF) if c.descending else UInt8(0)
    for i in range(count):
        var at = cursors[unsafe_offset=i]
        var row = first + i
        var out = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)
        if nulls and not column._valid(row):
            out[] = c.null_byte
            cursors[unsafe_offset=i] = at + 1
            continue
        out[] = c.valid_byte
        out[unsafe_offset=1] = UInt8(Int(column._get(row))) ^ flip
        cursors[unsafe_offset=i] = at + 2


def _encode_strings(
    column: StringColumn,
    c: SortKeyColumn,
    first: Int,
    count: Int,
    cursors: Pointer[Int, MutAnyOrigin],
):
    var rows = _Bytes(column)
    var nulls = c.nulls and column.null_count() > 0
    var blob = c.encoding == _ENC_BLOB
    for i in range(count):
        var at = cursors[unsafe_offset=i]
        var row = first + i
        if nulls and not column._valid(row):
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)[] = c.null_byte
            cursors[unsafe_offset=i] = at + 1
            continue
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)[] = c.valid_byte
        var written = _encode_blob(
            at + 1, rows.get(row), c.descending
        ) if blob else _encode_varchar(at + 1, rows.get(row), c.descending)
        cursors[unsafe_offset=i] = at + 1 + written


def _encode_part(
    part: Series,
    c: SortKeyColumn,
    first: Int,
    count: Int,
    cursors: Pointer[Int, MutAnyOrigin],
) raises:
    if part._data.isa[StringColumn]():
        _encode_strings(part._data[StringColumn], c, first, count, cursors)
    elif part._data.isa[BoolColumn]():
        _encode_bool(part._data[BoolColumn], c, first, count, cursors)
    elif part._data.isa[Column[Int128]]():
        _encode_int128(part._data[Column[Int128]], c, first, count, cursors)
    elif part._data.isa[Column[Int64]]():
        _encode_numeric(part._data[Column[Int64]], c, first, count, cursors)
    elif part._data.isa[Column[Int32]]():
        _encode_numeric(part._data[Column[Int32]], c, first, count, cursors)
    elif part._data.isa[Column[Float64]]():
        _encode_numeric(part._data[Column[Float64]], c, first, count, cursors)
    elif part._data.isa[Column[Float32]]():
        _encode_numeric(part._data[Column[Float32]], c, first, count, cursors)
    elif part._data.isa[Column[Int16]]():
        _encode_numeric(part._data[Column[Int16]], c, first, count, cursors)
    elif part._data.isa[Column[Int8]]():
        _encode_numeric(part._data[Column[Int8]], c, first, count, cursors)
    elif part._data.isa[Column[UInt64]]():
        _encode_numeric(part._data[Column[UInt64]], c, first, count, cursors)
    elif part._data.isa[Column[UInt32]]():
        _encode_numeric(part._data[Column[UInt32]], c, first, count, cursors)
    elif part._data.isa[Column[UInt16]]():
        _encode_numeric(part._data[Column[UInt16]], c, first, count, cursors)
    elif part._data.isa[Column[UInt8]]():
        _encode_numeric(part._data[Column[UInt8]], c, first, count, cursors)
    else:
        raise Error("cannot sort by column " + part.name())


def _string_lengths(
    part: Series,
    c: SortKeyColumn,
    first: Int,
    count: Int,
    lengths: Pointer[Int, MutAnyOrigin],
):
    ref column = part._data[StringColumn]
    var rows = _Bytes(column)
    var nulls = c.nulls and column.null_count() > 0
    var blob = c.encoding == _ENC_BLOB
    for i in range(count):
        var row = first + i
        if nulls and not column._valid(row):
            lengths[unsafe_offset=i] += 1
            continue
        var bytes = rows.get(row)
        lengths[unsafe_offset=i] += 1 + (
            _blob_length(bytes) if blob else len(bytes) + 1
        )


def _fixed_lengths[
    D: DType
](
    column: Column[Scalar[D]],
    c: SortKeyColumn,
    first: Int,
    count: Int,
    lengths: Pointer[Int, MutAnyOrigin],
):
    var nulls = c.nulls and column.null_count() > 0
    for i in range(count):
        if nulls and not column._valid(first + i):
            lengths[unsafe_offset=i] += 1
        else:
            lengths[unsafe_offset=i] += 1 + c.width


def _part_lengths(
    part: Series,
    c: SortKeyColumn,
    first: Int,
    count: Int,
    lengths: Pointer[Int, MutAnyOrigin],
) raises:
    """GetSortKeyLength: the bytes each row of this column adds."""
    if not c.nulls:
        if c.constant_size():
            for i in range(count):
                lengths[unsafe_offset=i] += 1 + c.width
            return
    if part._data.isa[StringColumn]():
        _string_lengths(part, c, first, count, lengths)
        return
    if part._data.isa[BoolColumn]():
        ref column = part._data[BoolColumn]
        for i in range(count):
            lengths[unsafe_offset=i] += 1 + (
                c.width if column._valid(first + i) else 0
            )
        return
    if part._data.isa[Column[Int128]]():
        _fixed_lengths(part._data[Column[Int128]], c, first, count, lengths)
    elif part._data.isa[Column[Int64]]():
        _fixed_lengths(part._data[Column[Int64]], c, first, count, lengths)
    elif part._data.isa[Column[Int32]]():
        _fixed_lengths(part._data[Column[Int32]], c, first, count, lengths)
    elif part._data.isa[Column[Float64]]():
        _fixed_lengths(part._data[Column[Float64]], c, first, count, lengths)
    elif part._data.isa[Column[Float32]]():
        _fixed_lengths(part._data[Column[Float32]], c, first, count, lengths)
    elif part._data.isa[Column[Int16]]():
        _fixed_lengths(part._data[Column[Int16]], c, first, count, lengths)
    elif part._data.isa[Column[Int8]]():
        _fixed_lengths(part._data[Column[Int8]], c, first, count, lengths)
    elif part._data.isa[Column[UInt64]]():
        _fixed_lengths(part._data[Column[UInt64]], c, first, count, lengths)
    elif part._data.isa[Column[UInt32]]():
        _fixed_lengths(part._data[Column[UInt32]], c, first, count, lengths)
    elif part._data.isa[Column[UInt16]]():
        _fixed_lengths(part._data[Column[UInt16]], c, first, count, lengths)
    elif part._data.isa[Column[UInt8]]():
        _fixed_lengths(part._data[Column[UInt8]], c, first, count, lengths)


def _for_parts(
    column: Series, lo: Int, hi: Int
) -> List[Tuple[Series, Int, Int, Int]]:
    """The chunks of `column` covering rows [lo, hi): (chunk, first row in
    the chunk, rows, position of that row in [lo, hi))."""
    var result = List[Tuple[Series, Int, Int, Int]]()
    var start = 0
    for part in column.chunks():
        var n = len(part)
        var a = max(lo, start)
        var b = min(hi, start + n)
        if a < b:
            result.append((part.copy(), a - start, b - a, a - lo))
        start += n
        if start >= hi:
            break
    return result^


comptime _VECTOR_NUMERIC = [
    DType.int64,
    DType.float64,
    DType.int32,
    DType.float32,
    DType.int16,
    DType.int8,
    DType.uint64,
    DType.uint32,
    DType.uint16,
    DType.uint8,
]

# Keys bound to a chunk's vectors (`SortKeyLayout.rebind_vectors`): row
# `first + i` of the chunk, or `rows[i]` when `rows` (an address of Ints)
# is set, read through its UnifiedVectorFormat's selection and validity,
# as create_sort_key.cpp's SortKeyVectorData reads any vector.


@always_inline
def _vector_row(first: Int, rows: Int, i: Int) -> Int:
    if rows == 0:
        return first + i
    return Pointer[Int, MutAnyOrigin](unsafe_from_address=rows)[unsafe_offset=i]


@always_inline
def _vector_string(
    format: UnifiedVectorFormat, index: Int
) -> Span[UInt8, ImmutAnyOrigin]:
    """The bytes of the string_t at `index`."""
    var entry = format.data + 16 * index
    var s = Pointer[StringT, MutAnyOrigin](unsafe_from_address=entry)[]
    return Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
            unsafe_from_address=string_data(entry)
        ),
        length=Int(s.length),
    )


def _encode_vector_numeric[
    D: DType
](
    c: SortKeyColumn,
    first: Int,
    count: Int,
    rows: Int,
    cursors: Pointer[Int, MutAnyOrigin],
):
    """`_encode_numeric` over a vector; a categorical key's codes are
    encoded as their ranks."""
    ref format = c.format
    var values = format.values[D]()
    var nulls = c.nulls and not format.validity.all_valid()
    var ranked = len(c.ranks) > 0
    var ranks = c.ranks.unsafe_ptr()
    comptime B = bit_width_of[D]()
    comptime U = DType.uint64 if B == 64 else (
        DType.uint32 if B == 32 else (DType.uint16 if B == 16 else DType.uint8)
    )
    var flip = Scalar[U].MAX if c.descending else Scalar[U](0)
    for i in range(count):
        var at = cursors[unsafe_offset=i]
        var index = format.sel.get_index(_vector_row(first, rows, i))
        if nulls and not format.validity.row_is_valid_unsafe(index):
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)[] = c.null_byte
            cursors[unsafe_offset=i] = at + 1
            continue
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)[] = c.valid_byte
        var value = values[unsafe_offset=index]
        comptime if D == DType.uint32:
            if ranked:
                value = rebind[Scalar[D]](ranks[unsafe_offset=Int(value)])
        var bits = _order_bits[D, U](value)
        _store_be[U](at + 1, bits ^ flip)
        cursors[unsafe_offset=i] = at + 1 + B // 8


def _encode_vector(
    c: SortKeyColumn,
    first: Int,
    count: Int,
    rows: Int,
    cursors: Pointer[Int, MutAnyOrigin],
) raises:
    """`_encode_part` over the vector the column is bound to."""
    ref format = c.format
    var nulls = c.nulls and not format.validity.all_valid()
    if format.physical == PHYSICAL_VARCHAR:
        var blob = c.encoding == _ENC_BLOB
        for i in range(count):
            var at = cursors[unsafe_offset=i]
            var index = format.sel.get_index(_vector_row(first, rows, i))
            if nulls and not format.validity.row_is_valid_unsafe(index):
                Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=at
                )[] = c.null_byte
                cursors[unsafe_offset=i] = at + 1
                continue
            Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=at
            )[] = c.valid_byte
            var bytes = _vector_string(format, index)
            var written = _encode_blob(
                at + 1, bytes, c.descending
            ) if blob else _encode_varchar(at + 1, bytes, c.descending)
            cursors[unsafe_offset=i] = at + 1 + written
        return
    if format.physical == PHYSICAL_BOOL:
        var data = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=format.data)
        var flip = UInt8(0xFF) if c.descending else UInt8(0)
        for i in range(count):
            var at = cursors[unsafe_offset=i]
            var index = format.sel.get_index(_vector_row(first, rows, i))
            var out = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)
            if nulls and not format.validity.row_is_valid_unsafe(index):
                out[] = c.null_byte
                cursors[unsafe_offset=i] = at + 1
                continue
            out[] = c.valid_byte
            out[unsafe_offset=1] = (
                UInt8(1) if data[unsafe_offset=index] != 0 else UInt8(0)
            ) ^ flip
            cursors[unsafe_offset=i] = at + 2
        return
    if format.physical == PHYSICAL_INT128:
        var values = format.values[DType.int128]()
        var flip = UInt64.MAX if c.descending else UInt64(0)
        for i in range(count):
            var at = cursors[unsafe_offset=i]
            var index = format.sel.get_index(_vector_row(first, rows, i))
            if nulls and not format.validity.row_is_valid_unsafe(index):
                Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=at
                )[] = c.null_byte
                cursors[unsafe_offset=i] = at + 1
                continue
            Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=at
            )[] = c.valid_byte
            var value = values[unsafe_offset=index]
            var upper = (value >> 64).cast[DType.uint64]() ^ (UInt64(1) << 63)
            var lower = value.cast[DType.uint64]()
            _store_be[DType.uint64](at + 1, upper ^ flip)
            _store_be[DType.uint64](at + 9, lower ^ flip)
            cursors[unsafe_offset=i] = at + 17
        return
    comptime for k in range(len(_VECTOR_NUMERIC)):
        comptime D = _VECTOR_NUMERIC[k]
        if format.physical == physical_of[D]():
            _encode_vector_numeric[D](c, first, count, rows, cursors)
            return
    raise Error("cannot sort by a vector of this type")


def _vector_lengths(
    c: SortKeyColumn,
    first: Int,
    count: Int,
    rows: Int,
    lengths: Pointer[Int, MutAnyOrigin],
):
    """`_part_lengths` over the vector the column is bound to."""
    ref format = c.format
    var nulls = c.nulls and not format.validity.all_valid()
    if format.physical != PHYSICAL_VARCHAR:
        for i in range(count):
            if nulls and not format.validity.row_is_valid_unsafe(
                format.sel.get_index(_vector_row(first, rows, i))
            ):
                lengths[unsafe_offset=i] += 1
            else:
                lengths[unsafe_offset=i] += 1 + c.width
        return
    var blob = c.encoding == _ENC_BLOB
    for i in range(count):
        var index = format.sel.get_index(_vector_row(first, rows, i))
        if nulls and not format.validity.row_is_valid_unsafe(index):
            lengths[unsafe_offset=i] += 1
            continue
        var bytes = _vector_string(format, index)
        lengths[unsafe_offset=i] += 1 + (
            _blob_length(bytes) if blob else len(bytes) + 1
        )


def _vector_string_prefixes(
    c: SortKeyColumn,
    first: Int,
    count: Int,
    cursors: Pointer[Int, MutAnyOrigin],
    limits: Pointer[Int, MutAnyOrigin],
):
    """`_string_prefixes` over the vector the column is bound to: a
    string_t holds the length and first four bytes inline."""
    ref format = c.format
    var nulls = c.nulls and not format.validity.all_valid()
    var flip = UInt8(0xFF) if c.descending else UInt8(0)
    for i in range(count):
        var at = cursors[unsafe_offset=i]
        var limit = limits[unsafe_offset=i]
        if at >= limit:
            continue
        var index = format.sel.get_index(first + i)
        var out_bytes = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)
        if nulls and not format.validity.row_is_valid_unsafe(index):
            out_bytes[] = c.null_byte
            cursors[unsafe_offset=i] = at + 1
            continue
        out_bytes[] = c.valid_byte
        var s = Pointer[StringT, MutAnyOrigin](
            unsafe_from_address=format.data + 16 * index
        )[]
        var n = Int(s.length)
        var room = limit - at - 1
        var shown = min(n, 4)
        for j in range(min(shown, room)):
            out_bytes[unsafe_offset=1 + j] = (
                UInt8((s.prefix >> UInt32(8 * j)) & 0xFF) + 1
            ) ^ flip
        if n < 4:
            if n < room:
                out_bytes[unsafe_offset=1 + n] = flip
            cursors[unsafe_offset=i] = min(at + n + 2, limit)
        else:
            cursors[unsafe_offset=i] = limit


struct KeyBlock(Movable):
    """Encoded key records of rows [lo, hi) of the sort input (the sink's
    key TupleDataCollection for one run, or one chunk of a top-N): `words`
    per record, and for a variable layout the keys' bytes in `heap`."""

    var records: List[UInt64]
    var heap: List[UInt8]
    var count: Int

    def __init__(out self):
        self.records = List[UInt64]()
        self.heap = List[UInt8]()
        self.count = 0

    def address(self) -> Int:
        return Int(self.records.unsafe_ptr())


def encode_keys(layout: SortKeyLayout, lo: Int, hi: Int) raises -> KeyBlock:
    """Records for rows [lo, hi): each row's key (create_sort_key, then
    SortKey::Construct) and its row index as the payload."""
    return _encode_rows(layout, lo, hi - lo, 0)


def _encode_rows(
    layout: SortKeyLayout, lo: Int, n: Int, rows: Int
) raises -> KeyBlock:
    """`encode_keys` of rows [lo, lo + n), or of the `n` rows at `rows` (an
    address of Ints; a layout bound to vectors only)."""
    var block = KeyBlock()
    block.count = n
    var r = layout.record_words()
    block.records = List[UInt64](length=max(1, n * r), fill=0)
    if n == 0:
        return block^
    var base = block.address()
    var cursors = List[Int](unsafe_uninit_length=n)
    var cursor = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=Int(cursors.unsafe_ptr())
    )
    var lengths = List[Int]()
    if layout.variable:
        lengths = List[Int](length=n, fill=0)
        var length = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=Int(lengths.unsafe_ptr())
        )
        for c in layout.columns:
            if c.encoding == _ENC_ROW:
                for i in range(n):
                    length[unsafe_offset=i] += 8
                continue
            if layout.vectors:
                _vector_lengths(c, lo, n, rows, length)
                continue
            for piece in _for_parts(c.series, lo, lo + n):
                _part_lengths(
                    piece[0],
                    c,
                    piece[1],
                    piece[2],
                    length.unsafe_offset(piece[3]),
                )
        var total = 0
        for i in range(n):
            total += lengths[i]
        block.heap = List[UInt8](length=max(1, total), fill=0)
        var at = Int(block.heap.unsafe_ptr())
        for i in range(n):
            cursor[unsafe_offset=i] = at
            at += lengths[i]
    else:
        for i in range(n):
            cursor[unsafe_offset=i] = base + i * r * 8
    for c in layout.columns:
        if c.encoding == _ENC_ROW:
            for i in range(n):
                var at = cursor[unsafe_offset=i]
                _store_be[DType.uint64](at, UInt64(_vector_row(lo, rows, i)))
                cursor[unsafe_offset=i] = at + 8
            continue
        if layout.vectors:
            _encode_vector(c, lo, n, rows, cursor)
            continue
        for piece in _for_parts(c.series, lo, lo + n):
            _encode_part(
                piece[0], c, piece[1], piece[2], cursor.unsafe_offset(piece[3])
            )
    var records = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=base)
    if layout.variable:
        var heap = Int(block.heap.unsafe_ptr())
        var at = heap
        for i in range(n):
            var size = lengths[i]
            var part0 = UInt64(0)
            var bytes = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)
            for b in range(min(size, 8)):
                part0 |= UInt64(bytes[unsafe_offset=b]) << UInt64(56 - 8 * b)
            var rec = records.unsafe_offset(i * 4)
            rec[] = part0
            rec[unsafe_offset=1] = UInt64(size)
            rec[unsafe_offset=2] = UInt64(at)
            rec[unsafe_offset=3] = UInt64(_vector_row(lo, rows, i))
            at += size
    else:
        var w = layout.words
        for i in range(n):
            var rec = records.unsafe_offset(i * r)
            for k in range(w):
                rec[unsafe_offset=k] = byte_swap(rec[unsafe_offset=k])
            rec[unsafe_offset=w] = UInt64(_vector_row(lo, rows, i))
    _ = cursors^
    _ = lengths^
    return block^


def _string_prefixes(
    column: StringColumn,
    c: SortKeyColumn,
    first: Int,
    count: Int,
    cursors: Pointer[Int, MutAnyOrigin],
    limits: Pointer[Int, MutAnyOrigin],
):
    """String rows encoded as `_encode_strings` does, from each value's
    length and first four bytes alone (an Arrow view holds them inline, as
    DuckDB's string_t its prefix): a longer value's remaining bytes are
    left zero, a lower bound, and its row stops there."""
    var rows = _Bytes(column)
    var nulls = c.nulls and column.null_count() > 0
    var flip = UInt8(0xFF) if c.descending else UInt8(0)
    for i in range(count):
        var at = cursors[unsafe_offset=i]
        var limit = limits[unsafe_offset=i]
        if at >= limit:
            continue
        var row = first + i
        var out_bytes = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at)
        if nulls and not column._valid(row):
            out_bytes[] = c.null_byte
            cursors[unsafe_offset=i] = at + 1
            continue
        out_bytes[] = c.valid_byte
        var head = rows.head(row)
        var n = Int(head & 0xFFFFFFFF)
        var room = limit - at - 1
        var shown = min(n, 4)
        for j in range(min(shown, room)):
            out_bytes[unsafe_offset=1 + j] = (
                UInt8((head >> UInt64(32 + 8 * j)) & 0xFF) + 1
            ) ^ flip
        if n < 4:
            if n < room:
                out_bytes[unsafe_offset=1 + n] = flip
            cursors[unsafe_offset=i] = min(at + n + 2, limit)
        else:
            cursors[unsafe_offset=i] = limit


def key_prefixes(
    layout: SortKeyLayout, lo: Int, hi: Int, mut prefixes: List[UInt64]
) raises -> Bool:
    """For rows [lo, hi), a lower bound of each key's first word (the
    record's `part0`), as unsigned integers: the key columns encoded up to
    the word's eighth byte, zeros past a binary column. A row whose bound
    exceeds a boundary key's first word is worse than that key:
    CheckBoundaryValues, over the leading sort columns."""
    var n = hi - lo
    if n <= 0:
        return False
    if layout.columns[0].encoding == _ENC_BLOB:
        return False
    prefixes = List[UInt64](length=n, fill=0)
    # A row's slot: the word, and room for a fixed-width value (at most
    # 17 bytes) started before the word's end.
    comptime SLOT = 32
    var scratch = List[UInt64](length=n * SLOT // 8, fill=0)
    var base = Int(scratch.unsafe_ptr())
    var cursors = List[Int](unsafe_uninit_length=n)
    var limits = List[Int](unsafe_uninit_length=n)
    for i in range(n):
        cursors[i] = base + SLOT * i
        limits[i] = base + SLOT * i + 8
    var cursor = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=Int(cursors.unsafe_ptr())
    )
    var limit = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=Int(limits.unsafe_ptr())
    )
    for c in layout.columns:
        if c.encoding == _ENC_BLOB:
            break
        var open = False
        for i in range(n):
            if cursors[i] < limits[i]:
                open = True
                break
        if not open:
            break
        if c.encoding == _ENC_ROW:
            for i in range(n):
                var at = cursors[i]
                if at < limits[i]:
                    _store_be[DType.uint64](at, UInt64(lo + i))
                    cursors[i] = at + 8
        elif c.encoding == _ENC_VARCHAR and layout.vectors:
            _vector_string_prefixes(c, lo, n, cursor, limit)
        elif layout.vectors:
            _encode_vector(c, lo, n, 0, cursor)
        elif c.encoding == _ENC_VARCHAR:
            for piece in _for_parts(c.series, lo, hi):
                _string_prefixes(
                    piece[0]._data[StringColumn],
                    c,
                    piece[1],
                    piece[2],
                    cursor.unsafe_offset(piece[3]),
                    limit.unsafe_offset(piece[3]),
                )
        else:
            # Rows already past the word write into their slot's spare
            # room, which is never read.
            for piece in _for_parts(c.series, lo, hi):
                _encode_part(
                    piece[0],
                    c,
                    piece[1],
                    piece[2],
                    cursor.unsafe_offset(piece[3]),
                )
        for i in range(n):
            cursors[i] = min(cursors[i], limits[i])
    var words = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=base)
    for i in range(n):
        prefixes[i] = byte_swap(words[unsafe_offset=SLOT // 8 * i])
    _ = cursors^
    _ = limits^
    _ = scratch^
    return True


def has_row_key(layout: SortKeyLayout) -> Bool:
    for c in layout.columns:
        if c.encoding == _ENC_ROW:
            return True
    return False


def encode_selected(
    layout: SortKeyLayout, lo: Int, selection: List[Int]
) raises -> KeyBlock:
    """Records for the rows `lo + selection[i]` alone, as `encode_keys`
    makes them (the rows CheckBoundaryValues keeps; a layout with the row
    index as a key encodes whole ranges instead)."""
    var rows = List[Int](capacity=len(selection))
    for s in selection:
        rows.append(lo + s)
    if layout.vectors:
        # The vectors are read at the selected rows where they are.
        var picked = _encode_rows(layout, 0, len(rows), Int(rows.unsafe_ptr()))
        _ = rows^
        return picked^
    var sub = layout.copy()
    for k in range(len(sub.columns)):
        sub.columns[k].series = layout.columns[k].series.take(rows)
    sub.rows = len(rows)
    var block = encode_keys(sub, 0, len(rows))
    var r = layout.record_words()
    var records = Pointer[UInt64, MutAnyOrigin](
        unsafe_from_address=block.address()
    )
    for i in range(len(rows)):
        records[unsafe_offset=i * r + r - 1] = UInt64(rows[i])
    return block^
