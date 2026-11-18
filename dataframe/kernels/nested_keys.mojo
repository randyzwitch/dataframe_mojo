"""List and struct values as key bytes, and key bytes back as values.

A key compares by value: two lists are equal when they have the same
length and equal elements in order, two structs when their fields are
equal in order, and a null equals only a null. Each nested value is
written as one byte string such that two values are equal exactly when
their bytes are, so a nested key groups, hashes and compares as a string
key does, its bytes held on the heap that string keys use.

Encoding of a valid value:
- a fixed-width value is its stored bytes; a float's -0.0 is written as
  0.0 and every NaN as one NaN, so they compare as the flat keys do;
- a Boolean is one byte, 0 or 1;
- a string is its length (four bytes) and then its bytes;
- a list is its length (four bytes) and then each element's slot;
- a struct is each field's slot, in order.
A slot is one byte, 0 for a null and 1 for a value, then the value when
there is one. So a null struct (a null slot) differs from a struct whose
fields are null (a valid slot of null slots), and an empty list (a valid
slot of length 0) from a null list. Every part has a known length, so no
encoding is a prefix of another, and equal bytes decode to equal values.

A top-level null is a null string, not a slot: it takes the key's null
bit, as a flat key's null does.
"""
from std.memory import Pointer, bitcast, unsafe_memcpy
from std.sys import size_of

from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.series import Series
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column, _pack_bits
from dataframe.storage.nested import ListColumn, StructColumn
from dataframe.storage.string import StringBuilder, StringColumn

comptime _FIXED = 0
comptime _BOOL = 1
comptime _STRING = 2
comptime _LIST = 3
comptime _STRUCT = 4


@fieldwise_init
struct _Reader(Copyable, Movable):
    """One level of a nested column, read through raw addresses: its kind,
    its validity bitmap (0 for none) and the bit of its row 0, and its
    values: a fixed value's row 0 and width (a float's width negated), a
    Boolean's bitmap and bit of row 0, a list's offsets from row 0, or a
    string column's place in `_Readers.strings`. A list's child and a
    struct's fields are other readers."""

    var kind: Int
    var bits: Int
    var bit: Int
    var values: Int
    var width: Int
    var value_bit: Int
    var children: List[Int]


struct _Readers(Movable):
    """The readers of a nested column's levels, by index (the column's own
    at 0), and the columns they read, kept alive while they are read."""

    var nodes: List[_Reader]
    var strings: List[StringColumn]
    var owners: List[Series]

    def __init__(out self, series: Series) raises:
        self.nodes = List[_Reader]()
        self.strings = List[StringColumn]()
        self.owners = List[Series]()
        _ = self._add(series)

    def _add(mut self, given: Series) raises -> Int:
        var series = given.rechunk() if given.is_chunked() else given.copy()
        var at = len(self.nodes)
        self.nodes.append(_Reader(0, 0, 0, 0, 0, 0, List[Int]()))
        var dtype = series.dtype()
        var node = _Reader(0, 0, 0, 0, 0, 0, List[Int]())
        if dtype.is_list():
            var lists = series.list_column()
            node.kind = _LIST
            if len(lists._bits[]) > 0:
                node.bits = Int(lists._bits.unsafe_ptr())
                node.bit = lists._offset
            node.values = Int(lists._offsets.unsafe_ptr()) + 8 * (lists._offset)
            # The offsets index the whole child.
            node.children.append(self._add(lists.child()))
        elif dtype.is_struct():
            var structs = series.struct_column()
            node.kind = _STRUCT
            if len(structs._bits[]) > 0:
                node.bits = Int(structs._bits.unsafe_ptr())
                node.bit = structs._offset
            for f in range(structs.field_count()):
                node.children.append(self._add(structs.field(f)))
        elif series._data.isa[StringColumn]():
            ref strings = series._data[StringColumn]
            node.kind = _STRING
            if len(strings._bits[]) > 0:
                node.bits = Int(strings.unsafe_validity())
                node.bit = strings.validity_offset()
            node.values = len(self.strings)
            self.strings.append(strings.copy())
        elif series._data.isa[BoolColumn]():
            ref flags = series._data[BoolColumn]
            node.kind = _BOOL
            if len(flags._bits[]) > 0:
                node.bits = Int(flags.unsafe_validity())
                node.bit = flags.validity_offset()
            node.values = Int(flags.unsafe_values())
            node.value_bit = flags._offset
        elif series._data.isa[Column[Int128]]():
            ref column = series._data[Column[Int128]]
            node.kind = _FIXED
            if len(column._bits[]) > 0:
                node.bits = Int(column.unsafe_validity())
                node.bit = column.validity_offset()
            node.values = Int(column.unsafe_values())
            node.width = 16
        else:
            var found = False
            comptime for k in range(len(NUMERIC_DTYPES)):
                comptime D = NUMERIC_DTYPES[k]
                if not found and series._data.isa[Column[Scalar[D]]]():
                    ref column = series._data[Column[Scalar[D]]]
                    node.kind = _FIXED
                    if len(column._bits[]) > 0:
                        node.bits = Int(column.unsafe_validity())
                        node.bit = column.validity_offset()
                    node.values = Int(column.unsafe_values())
                    node.width = size_of[Scalar[D]]()
                    comptime if D.is_floating_point():
                        node.width = -node.width
                    found = True
            if not found:
                raise Error(
                    "list and struct keys cannot hold "
                    + dtype.name()
                    + " values"
                )
        self.owners.append(series^)
        self.nodes[at] = node^
        return at

    @always_inline
    def valid(self, n: Int, row: Int) -> Bool:
        ref node = self.nodes[n]
        if node.bits == 0:
            return True
        var at = node.bit + row
        var byte = Pointer[UInt8, ImmutAnyOrigin](
            unsafe_from_address=node.bits
        )[unsafe_offset=at >> 3]
        return (byte >> UInt8(at & 7)) & 1 != 0

    def slot(self, n: Int, row: Int, mut out: List[UInt8]):
        """Row `row` of level `n` as a slot: its null byte, then its value."""
        if not self.valid(n, row):
            out.append(0)
            return
        out.append(1)
        self.value(n, row, out)

    def value(self, n: Int, row: Int, mut out: List[UInt8]):
        """Row `row` of level `n`, a valid value, as its bytes."""
        ref node = self.nodes[n]
        if node.kind == _FIXED:
            var width = node.width
            if width == -8:
                var x = Pointer[Float64, ImmutAnyOrigin](
                    unsafe_from_address=node.values
                )[unsafe_offset=row]
                _append_word(out, bitcast[DType.uint64](_canonical(x)), 8)
            elif width == -4:
                var x = Pointer[Float32, ImmutAnyOrigin](
                    unsafe_from_address=node.values
                )[unsafe_offset=row]
                _append_word(
                    out, UInt64(bitcast[DType.uint32](_canonical(x))), 4
                )
            else:
                var at = node.values + width * row
                var start = len(out)
                out.resize(start + width, 0)
                unsafe_memcpy(
                    dest=out.unsafe_ptr().unsafe_offset(start),
                    src=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at),
                    count=width,
                )
        elif node.kind == _BOOL:
            var at = node.value_bit + row
            var byte = Pointer[UInt8, ImmutAnyOrigin](
                unsafe_from_address=node.values
            )[unsafe_offset=at >> 3]
            out.append((byte >> UInt8(at & 7)) & 1)
        elif node.kind == _STRING:
            var bytes = self.strings[node.values]._row_bytes(row)
            _append_word(out, UInt64(len(bytes)), 4)
            out.extend(bytes)
        elif node.kind == _LIST:
            var offsets = Pointer[Int64, ImmutAnyOrigin](
                unsafe_from_address=node.values
            )
            var start = Int(offsets[unsafe_offset=row])
            var end = Int(offsets[unsafe_offset=row + 1])
            _append_word(out, UInt64(end - start), 4)
            var child = node.children[0]
            for e in range(start, end):
                self.slot(child, e, out)
        else:
            for c in node.children:
                self.slot(c, row, out)


@always_inline
def _append_word(mut out: List[UInt8], word: UInt64, bytes: Int):
    """The low `bytes` bytes of `word`, least significant first."""
    for b in range(bytes):
        out.append(UInt8((word >> UInt64(8 * b)) & 0xFF))


@always_inline
def _canonical[D: DType](x: Scalar[D]) -> Scalar[D]:
    """One float for each class of equal keys: 0.0 for -0.0, one NaN for
    every NaN."""
    if x != x:
        var zero = Scalar[D](0)
        return zero / zero
    if x == 0:
        return 0
    return x


def nested_key_supported(dtype: DataType) -> Bool:
    """Whether a list or struct type is a key here: every value it holds,
    however deep, is a number, a Boolean, a string or binary, or one of
    the types stored as those (dates, times, decimals, categoricals)."""
    try:
        if dtype.is_list():
            return _leaf_supported(dtype.inner())
        if dtype.is_struct():
            for field in dtype.field_dtypes():
                if not _leaf_supported(field):
                    return False
            return dtype.field_count() > 0
    except:
        return False
    return False


def _leaf_supported(dtype: DataType) raises -> Bool:
    if dtype.is_nested():
        return nested_key_supported(dtype)
    if dtype.is_geometry() or dtype.is_untyped():
        return False
    try:
        var empty = Series.full_null("", dtype, 0)
        var readers = _Readers(empty)
        _ = readers^
        return True
    except:
        return False


def encode_nested_keys(series: Series) raises -> Series:
    """A list or struct column as one byte string a row (the module's
    encoding), a null row a null string; named as `series` is."""
    var readers = _Readers(series)
    var n = len(series)
    var bytes = List[UInt8](capacity=16 * n)
    var offsets = List[Int64](capacity=n + 1)
    offsets.append(0)
    var valid = List[Bool](capacity=n)
    var any_null = False
    for row in range(n):
        var ok = readers.valid(0, row)
        valid.append(ok)
        if ok:
            readers.value(0, row, bytes)
        else:
            any_null = True
        offsets.append(Int64(len(bytes)))
    var bits = _pack_bits(valid) if any_null else List[UInt8]()
    _ = readers^
    return Series(
        series.name(),
        StringColumn(bytes=bytes^, offsets=offsets^, bits=bits^, length=n),
    )


struct _Builder(Movable):
    """One level of a column being decoded: its type and kind, each row's
    validity, and its values (fixed bytes, Boolean values, string rows, or
    a list's offsets); a list's child and a struct's fields are other
    builders."""

    var dtype: DataType
    var kind: Int
    var width: Int
    var valid: List[Bool]
    var fixed: List[UInt8]
    var flags: List[Bool]
    var strings: StringBuilder
    var offsets: List[Int64]
    var children: List[Int]

    def __init__(out self, dtype: DataType, kind: Int, width: Int):
        self.dtype = dtype
        self.kind = kind
        self.width = width
        self.valid = List[Bool]()
        self.fixed = List[UInt8]()
        self.flags = List[Bool]()
        self.strings = StringBuilder()
        self.offsets = List[Int64]()
        self.offsets.append(0)
        self.children = List[Int]()


struct _Builders(Movable):
    """The builders of a nested type's levels, by index (its own at 0)."""

    var nodes: List[_Builder]

    def __init__(out self, dtype: DataType) raises:
        self.nodes = List[_Builder]()
        _ = self._add(dtype)

    def _add(mut self, dtype: DataType) raises -> Int:
        var at = len(self.nodes)
        if dtype.is_list():
            self.nodes.append(_Builder(dtype, _LIST, 0))
            var child = self._add(dtype.inner())
            self.nodes[at].children.append(child)
            return at
        if dtype.is_struct():
            self.nodes.append(_Builder(dtype, _STRUCT, 0))
            for field in dtype.field_dtypes():
                var child = self._add(field)
                self.nodes[at].children.append(child)
            return at
        # A leaf's storage, as an empty column of its type reads it.
        var readers = _Readers(Series.full_null("", dtype, 0))
        ref leaf = readers.nodes[0]
        var kind = leaf.kind
        var width = leaf.width
        _ = readers^
        self.nodes.append(_Builder(dtype, kind, abs(width)))
        return at

    def null(mut self, n: Int):
        """A null row of level `n` (a struct's fields null too, so they
        keep its length)."""
        self.nodes[n].valid.append(False)
        var kind = self.nodes[n].kind
        if kind == _FIXED:
            var start = len(self.nodes[n].fixed)
            self.nodes[n].fixed.resize(start + self.nodes[n].width, 0)
        elif kind == _BOOL:
            self.nodes[n].flags.append(False)
        elif kind == _STRING:
            self.nodes[n].strings.append_null()
        elif kind == _LIST:
            var last = self.nodes[n].offsets[len(self.nodes[n].offsets) - 1]
            self.nodes[n].offsets.append(last)
        else:
            for c in self.nodes[n].children.copy():
                self.null(c)

    def slot(mut self, n: Int, data: Int, at: Int) raises -> Int:
        """A slot read at `data + at` into level `n`; where it ends."""
        var flag = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=data)[
            unsafe_offset=at
        ]
        if flag == 0:
            self.null(n)
            return at + 1
        return self.value(n, data, at + 1)

    def value(mut self, n: Int, data: Int, at: Int) raises -> Int:
        """A valid value read at `data + at` into level `n`; where it
        ends."""
        var bytes = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=data)
        self.nodes[n].valid.append(True)
        var kind = self.nodes[n].kind
        if kind == _FIXED:
            var width = self.nodes[n].width
            for b in range(width):
                self.nodes[n].fixed.append(bytes[unsafe_offset=at + b])
            return at + width
        if kind == _BOOL:
            self.nodes[n].flags.append(bytes[unsafe_offset=at] != 0)
            return at + 1
        var length = Int(_read_word(data + at))
        if kind == _STRING:
            # The bytes as they were: a binary value need not be UTF-8.
            ref strings = self.nodes[n].strings
            strings._bytes.extend(
                Span[UInt8, ImmutAnyOrigin](
                    unsafe_ptr=bytes.unsafe_offset(at + 4), length=length
                )
            )
            strings._offsets.append(Int64(len(strings._bytes)))
            strings._push_bit(True)
            return at + 4 + length
        if kind == _LIST:
            var next = at + 4
            var child = self.nodes[n].children[0]
            for _ in range(length):
                next = self.slot(child, data, next)
            var last = self.nodes[n].offsets[len(self.nodes[n].offsets) - 1]
            self.nodes[n].offsets.append(last + Int64(length))
            return next
        var next = at
        for c in self.nodes[n].children.copy():
            next = self.slot(c, data, next)
        return next

    def series(mut self, n: Int, var name: String) raises -> Series:
        """Level `n`'s rows as a column named `name`, typed as it was."""
        var rows = len(self.nodes[n].valid)
        var any_null = False
        for ok in self.nodes[n].valid:
            any_null = any_null or not ok
        var bits = _pack_bits(self.nodes[n].valid) if any_null else List[
            UInt8
        ]()
        var kind = self.nodes[n].kind
        var dtype = self.nodes[n].dtype
        var result: Series
        if kind == _LIST:
            var child = self.series(self.nodes[n].children[0], "item")
            result = Series(
                name^,
                ListColumn(self.nodes[n].offsets.copy(), child^, bits^),
            )
        elif kind == _STRUCT:
            var names = dtype.field_names()
            var fields = List[Series]()
            var children = self.nodes[n].children.copy()
            for c in range(len(children)):
                fields.append(self.series(children[c], names[c]))
            result = Series(name^, StructColumn(fields^, bits^, length=rows))
        elif kind == _STRING:
            var builder = StringBuilder()
            swap(builder, self.nodes[n].strings)
            result = Series(name^, builder^.finish())
        elif kind == _BOOL:
            if any_null:
                result = Series(
                    name^,
                    BoolColumn(self.nodes[n].flags, self.nodes[n].valid),
                )
            else:
                result = Series(name^, BoolColumn(self.nodes[n].flags))
        else:
            result = self._fixed(n, name^, any_null)
        if result.dtype() != dtype:
            return result.with_dtype(dtype)
        return result^

    def _fixed(
        mut self, n: Int, var name: String, any_null: Bool
    ) raises -> Series:
        ref node = self.nodes[n]
        var rows = len(node.valid)
        var storage = node.dtype.storage()
        if node.width == 16:
            return _fixed_column[DType.int128](
                node.fixed, rows, node.valid, any_null, name^
            )
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if storage and storage.value() == D:
                return _fixed_column[D](
                    node.fixed, rows, node.valid, any_null, name^
                )
        raise Error("nested keys: no column for " + node.dtype.name())


def _fixed_column[
    D: DType
](
    fixed: List[UInt8],
    rows: Int,
    valid: List[Bool],
    any_null: Bool,
    var name: String,
) raises -> Series:
    var values = List[Scalar[D]](unsafe_uninit_length=rows)
    if rows > 0:
        unsafe_memcpy(
            dest=values.unsafe_ptr().unsafe_bitcast[UInt8](),
            src=fixed.unsafe_ptr(),
            count=rows * size_of[Scalar[D]](),
        )
    if any_null:
        return Series(name^, Column[Scalar[D]](values^, valid))
    return Series(name^, Column[Scalar[D]](values^))


@always_inline
def _read_word(address: Int) -> UInt32:
    var bytes = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=address)
    var word = UInt32(0)
    for b in range(4):
        word |= UInt32(bytes[unsafe_offset=b]) << UInt32(8 * b)
    return word


def decode_nested_keys(keys: Series, dtype: DataType) raises -> Series:
    """Byte strings `encode_nested_keys` wrote back as the list or struct
    column of type `dtype` they came from, a null string a null row."""
    var column = keys.rechunk() if keys.is_chunked() else keys.copy()
    ref strings = column._data[StringColumn]
    var builders = _Builders(dtype)
    for row in range(len(strings)):
        if not strings._valid(row):
            builders.null(0)
            continue
        var bytes = strings._row_bytes(row)
        _ = builders.value(0, Int(bytes.unsafe_ptr()), 0)
    # The bytes were read through their address: the column lives to here.
    _ = column^
    return builders.series(0, keys.name())
