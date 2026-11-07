"""DuckDB 2.0's vector format: vector types, selection vectors, validity
masks, 16-byte strings and the unified format kernels read through.

Ported from duckdb-2.0 `common/vector_size.hpp`, `common/enums/
vector_type.hpp`, `common/types/selection_vector.hpp`, `common/types/
validity_mask.hpp`, `common/types/string_type.hpp`, `common/types/
vector.hpp` and `common/vector/unified_vector_format.hpp`.

A `Vector` here is a plain value: addresses and counts, no reference
counts. DuckDB's vectors hold their buffers through shared pointers; ours
borrow them from three owners that outlive every vector of a pipeline
run: the at-rest columns (`ColumnArena`), the `DataChunk` whose caches
and selection buffers a step wrote, and the constants of a bound
expression. Copying a vector is copying a few words.
"""
from std.memory import Pointer, unsafe_memcpy
from std.ffi import external_call

# DuckDB's STANDARD_VECTOR_SIZE (common/vector_size.hpp): the rows of one
# data chunk.
comptime STANDARD_VECTOR_SIZE = 2048

# VectorType (common/enums/vector_type.hpp), same numbering. FSST and
# shredded vectors have no counterpart here; at-rest Arrow strings and
# packed booleans play FSST's role (decoded when a kernel reads them, see
# `Vector.encoded`).
comptime FLAT_VECTOR = 0
comptime FSST_VECTOR = 1
comptime CONSTANT_VECTOR = 2
comptime DICTIONARY_VECTOR = 3
comptime SEQUENCE_VECTOR = 4

# PhysicalType: how a vector's values are laid out in memory
# (common/types.hpp PhysicalType). OPAQUE stands for the nested and other
# types no vector kernel reads yet: such a vector is only passed along and
# read through its at-rest column.
comptime PHYSICAL_BOOL = 1
comptime PHYSICAL_UINT8 = 2
comptime PHYSICAL_INT8 = 3
comptime PHYSICAL_UINT16 = 4
comptime PHYSICAL_INT16 = 5
comptime PHYSICAL_UINT32 = 6
comptime PHYSICAL_INT32 = 7
comptime PHYSICAL_UINT64 = 8
comptime PHYSICAL_INT64 = 9
comptime PHYSICAL_FLOAT = 11
comptime PHYSICAL_DOUBLE = 12
# Nested types (STRUCT = 100, LIST = 101 in DuckDB). A vector of either
# is still OPAQUE in the chunk: it is read through its at-rest column. The
# row layout (`tuple_data_layout.mojo`) uses these to describe columns.
comptime PHYSICAL_STRUCT = 100
comptime PHYSICAL_LIST = 101
comptime PHYSICAL_VARCHAR = 200
comptime PHYSICAL_INT128 = 204
comptime PHYSICAL_OPAQUE = 254


def physical_size(physical: Int) -> Int:
    """Bytes a value takes in a flat vector (GetTypeIdSize)."""
    if (
        physical == PHYSICAL_BOOL
        or physical == PHYSICAL_UINT8
        or physical == PHYSICAL_INT8
    ):
        return 1
    if physical == PHYSICAL_UINT16 or physical == PHYSICAL_INT16:
        return 2
    if (
        physical == PHYSICAL_UINT32
        or physical == PHYSICAL_INT32
        or physical == PHYSICAL_FLOAT
    ):
        return 4
    if (
        physical == PHYSICAL_UINT64
        or physical == PHYSICAL_INT64
        or physical == PHYSICAL_DOUBLE
    ):
        return 8
    if physical == PHYSICAL_VARCHAR or physical == PHYSICAL_INT128:
        return 16
    return 0


def physical_of[D: DType]() -> Int:
    """The physical type of a `Scalar[D]` value."""
    comptime if D == DType.bool:
        return PHYSICAL_BOOL
    elif D == DType.uint8:
        return PHYSICAL_UINT8
    elif D == DType.int8:
        return PHYSICAL_INT8
    elif D == DType.uint16:
        return PHYSICAL_UINT16
    elif D == DType.int16:
        return PHYSICAL_INT16
    elif D == DType.uint32:
        return PHYSICAL_UINT32
    elif D == DType.int32:
        return PHYSICAL_INT32
    elif D == DType.uint64:
        return PHYSICAL_UINT64
    elif D == DType.int64:
        return PHYSICAL_INT64
    elif D == DType.float32:
        return PHYSICAL_FLOAT
    elif D == DType.float64:
        return PHYSICAL_DOUBLE
    elif D == DType.int128:
        return PHYSICAL_INT128
    else:
        return PHYSICAL_OPAQUE


@always_inline
def _u32_at(address: Int) -> Pointer[UInt32, MutAnyOrigin]:
    return Pointer[UInt32, MutAnyOrigin](unsafe_from_address=address)


@always_inline
def _u8_at(address: Int) -> Pointer[UInt8, MutAnyOrigin]:
    return Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address)


struct SelectionVector(Copyable, ImplicitlyCopyable, Movable):
    """DuckDB's `SelectionVector`: `sel_t` (uint32) row indices, or none
    (address 0), which selects rows 0, 1, 2, ... (`get_index(i) == i`).

    The entries belong to whoever made them (a `DataChunk`'s selection
    buffers, a filter's state, a join's output); this value only points
    at them, as a SelectionVector initialized from a raw `sel_t *` does.
    """

    var data: Int

    @always_inline
    def __init__(out self):
        self.data = 0

    @always_inline
    def __init__(out self, data: Int):
        self.data = data

    @always_inline
    def is_set(self) -> Bool:
        return self.data != 0

    @always_inline
    def get_index(self, i: Int) -> Int:
        if self.data == 0:
            return i
        return Int(_u32_at(self.data)[unsafe_offset=i])

    @always_inline
    def get_index_unsafe(self, i: Int) -> Int:
        return Int(_u32_at(self.data)[unsafe_offset=i])

    @always_inline
    def set_index(self, i: Int, location: Int):
        _u32_at(self.data)[unsafe_offset=i] = UInt32(location)


struct ValidityMask(Copyable, ImplicitlyCopyable, Movable):
    """DuckDB's `ValidityMask`: a bit a row, 1 = valid, or no bitmap
    (address 0) when every row is valid.

    One difference: DuckDB's bitmap starts at row 0. Ours may start at bit
    `offset` of an Arrow bitmap, because a dictionary's child or a flat
    vector over storage is a window onto an at-rest column at any row.
    Bitmaps a kernel writes start at bit 0.
    """

    var bits: Int
    var offset: Int

    @always_inline
    def __init__(out self):
        self.bits = 0
        self.offset = 0

    @always_inline
    def __init__(out self, bits: Int, offset: Int):
        self.bits = bits
        self.offset = offset

    @always_inline
    def all_valid(self) -> Bool:
        return self.bits == 0

    @always_inline
    def row_is_valid(self, row: Int) -> Bool:
        if self.bits == 0:
            return True
        var bit = self.offset + row
        return (
            _u8_at(self.bits)[unsafe_offset=bit >> 3] >> UInt8(bit & 7)
        ) & 1 == 1

    @always_inline
    def row_is_valid_unsafe(self, row: Int) -> Bool:
        var bit = self.offset + row
        return (
            _u8_at(self.bits)[unsafe_offset=bit >> 3] >> UInt8(bit & 7)
        ) & 1 == 1


@always_inline
def set_invalid(bits: Int, row: Int):
    """Clear row `row`'s bit of a bitmap a kernel writes (offset 0)."""
    var at = _u8_at(bits)
    at[unsafe_offset=row >> 3] &= ~(UInt8(1) << UInt8(row & 7))


@always_inline
def set_all_valid(bits: Int, count: Int):
    """Set the first `count` bits of a bitmap a kernel writes."""
    var at = _u8_at(bits)
    for b in range((count + 7) // 8):
        at[unsafe_offset=b] = 255


# string_t (common/types/string_type.hpp): 16 bytes, the length then
# either the bytes themselves (at most 12, zero padded) or their first four
# bytes and a pointer to all of them.
comptime STRING_INLINE_BYTES = 12
comptime STRING_PREFIX_BYTES = 4


@fieldwise_init
struct StringT(Copyable, ImplicitlyCopyable, Movable):
    """DuckDB's `string_t`. `prefix` holds the first four bytes in either
    form; `rest` holds bytes 4 to 11 of an inlined string, or the address
    of a longer string's bytes. Arrow's inline view is the same 16 bytes,
    so an inline Utf8View descriptor is a string_t as it stands."""

    var length: UInt32
    var prefix: UInt32
    var rest: UInt64

    @always_inline
    def is_inlined(self) -> Bool:
        return self.length <= UInt32(STRING_INLINE_BYTES)

    @staticmethod
    @always_inline
    def make(data: Int, length: Int) -> StringT:
        """The string_t of `length` bytes at `data` (string_t(const char*,
        uint32_t)): copied inline when short, else pointed at."""
        var result = StringT(UInt32(length), 0, 0)
        if length <= STRING_INLINE_BYTES:
            if length > 0:
                unsafe_memcpy(
                    dest=Pointer(to=result)
                    .unsafe_bitcast[UInt8]()
                    .unsafe_offset(4),
                    src=_u8_at(data),
                    count=length,
                )
            return result
        result.prefix = _load_u32(data)
        result.rest = UInt64(data)
        return result


@always_inline
def _load_u32(address: Int) -> UInt32:
    var p = _u8_at(address)
    return (
        UInt32(p[unsafe_offset=0])
        | (UInt32(p[unsafe_offset=1]) << 8)
        | (UInt32(p[unsafe_offset=2]) << 16)
        | (UInt32(p[unsafe_offset=3]) << 24)
    )


@always_inline
def string_data(entry: Int) -> Int:
    """The address of the bytes of the string_t stored at `entry`
    (string_t::GetData)."""
    var s = Pointer[StringT, MutAnyOrigin](unsafe_from_address=entry)[]
    if s.is_inlined():
        return entry + 4
    return Int(s.rest)


@always_inline
def string_equals(a: Int, b: Int) -> Bool:
    """Whether the string_t values at addresses `a` and `b` are equal
    (StringComparisonOperators::Equals): length and prefix in one 8-byte
    compare, then the inlined bytes or the pointed-at ones."""
    var left = Pointer[StringT, MutAnyOrigin](unsafe_from_address=a)[]
    var right = Pointer[StringT, MutAnyOrigin](unsafe_from_address=b)[]
    if left.length != right.length or left.prefix != right.prefix:
        return False
    if left.is_inlined():
        return left.rest == right.rest
    var length = Int(left.length)
    return (
        external_call["memcmp", Int32](
            Int(left.rest) + STRING_PREFIX_BYTES,
            Int(right.rest) + STRING_PREFIX_BYTES,
            length - STRING_PREFIX_BYTES,
        )
        == 0
    )


@always_inline
def string_compare(a: Int, b: Int) -> Int:
    """Negative, zero or positive as the string_t at `a` sorts before, with
    or after the one at `b`, by bytes (StringComparisonOperators::
    GreaterThan's memcmp order): the prefixes first, byte-swapped so a
    word compare is a byte compare."""
    var left = Pointer[StringT, MutAnyOrigin](unsafe_from_address=a)[]
    var right = Pointer[StringT, MutAnyOrigin](unsafe_from_address=b)[]
    if left.prefix != right.prefix:
        var l = _bswap32(left.prefix)
        var r = _bswap32(right.prefix)
        return -1 if l < r else 1
    var left_length = Int(left.length)
    var right_length = Int(right.length)
    var shared = min(left_length, right_length)
    if shared > STRING_PREFIX_BYTES:
        var compared = external_call["memcmp", Int32](
            string_data(a) + STRING_PREFIX_BYTES,
            string_data(b) + STRING_PREFIX_BYTES,
            shared - STRING_PREFIX_BYTES,
        )
        if compared != 0:
            return Int(compared)
    return left_length - right_length


@always_inline
def _bswap32(value: UInt32) -> UInt32:
    return (
        (value >> 24)
        | ((value >> 8) & 0xFF00)
        | ((value << 8) & 0xFF0000)
        | (value << 24)
    )


struct UnifiedVectorFormat(Copyable, ImplicitlyCopyable, Movable):
    """DuckDB's `UnifiedVectorFormat`: row i of any flat, constant or
    dictionary vector is `data[sel.get_index(i)]`, valid when
    `validity.row_is_valid(sel.get_index(i))`. Strings are string_t and
    booleans one byte, whatever the storage they came from."""

    var sel: SelectionVector
    var data: Int
    var validity: ValidityMask
    var physical: Int

    @always_inline
    def __init__(out self):
        """No vector: no rows to read."""
        self.sel = SelectionVector()
        self.data = 0
        self.validity = ValidityMask()
        self.physical = PHYSICAL_OPAQUE

    @always_inline
    def __init__(
        out self,
        sel: SelectionVector,
        data: Int,
        validity: ValidityMask,
        physical: Int,
    ):
        self.sel = sel
        self.data = data
        self.validity = validity
        self.physical = physical

    @always_inline
    def values[D: DType](self) -> Pointer[Scalar[D], MutAnyOrigin]:
        return Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=self.data)


struct Vector(Copyable, ImplicitlyCopyable, Movable):
    """A column of a data chunk (DuckDB `Vector`), as a plain value.

    - FLAT: `data` is row 0 of the values.
    - CONSTANT: `data` holds the one value; every row reads it.
    - DICTIONARY: row i is child row `sel.get_index(i)`; `data` and
      `validity` are the child's (`dictionary_size` rows). The child is
      always flat (Vector::Slice merges a dictionary of a dictionary).
    - SEQUENCE: row i is `sequence_start + i * sequence_increment`.

    `source` names the at-rest column (`ColumnArena`) the values were
    wrapped from, and `source_offset` the row of it that is row 0 of the
    flat values (or of the dictionary's child), -1 when the values live in
    a chunk's cache. A vector over an Arrow string or packed boolean column
    is `encoded`: `data` is 0 and the values are read through the at-rest
    column until `DataChunk.unified` decodes them, as DuckDB decodes an
    FSST vector on `ToUnifiedFormat`.
    """

    var vector_type: Int
    var physical: Int
    var data: Int
    var validity: ValidityMask
    var sel: SelectionVector
    var dictionary_size: Int
    var sequence_start: Int64
    var sequence_increment: Int64
    var source: Int
    var source_offset: Int

    @always_inline
    def __init__(out self):
        self.vector_type = FLAT_VECTOR
        self.physical = PHYSICAL_OPAQUE
        self.data = 0
        self.validity = ValidityMask()
        self.sel = SelectionVector()
        self.dictionary_size = 0
        self.sequence_start = 0
        self.sequence_increment = 0
        self.source = -1
        self.source_offset = 0

    @staticmethod
    def flat(physical: Int, data: Int, validity: ValidityMask) -> Vector:
        var result = Vector()
        result.physical = physical
        result.data = data
        result.validity = validity
        return result

    @staticmethod
    def constant(physical: Int, data: Int, valid: Bool) -> Vector:
        """A constant vector of the value at `data` (null when not
        `valid`, ConstantVector::SetNull)."""
        var result = Vector()
        result.vector_type = CONSTANT_VECTOR
        result.physical = physical
        result.data = data
        if not valid:
            result.validity = ValidityMask(Int(_NULL_BYTE_ADDRESS()), 0)
        return result

    @staticmethod
    def sequence(start: Int64, increment: Int64) -> Vector:
        """Vector::Sequence: row i is start + i * increment (BIGINT)."""
        var result = Vector()
        result.vector_type = SEQUENCE_VECTOR
        result.physical = PHYSICAL_INT64
        result.sequence_start = start
        result.sequence_increment = increment
        return result

    @always_inline
    def encoded(self) -> Bool:
        """Whether the values are still in their at-rest layout (an Arrow
        string or packed boolean column) and must be decoded to read."""
        return (
            self.data == 0
            and self.source >= 0
            and (
                self.vector_type == FLAT_VECTOR
                or self.vector_type == DICTIONARY_VECTOR
                or self.vector_type == CONSTANT_VECTOR
            )
        )

    @always_inline
    def child_row(self, row: Int) -> Int:
        """The row of the values (the child's, for a dictionary) that row
        `row` reads."""
        if self.vector_type == DICTIONARY_VECTOR:
            return self.sel.get_index(row)
        if self.vector_type == CONSTANT_VECTOR:
            return 0
        return row


# A process-wide byte with every bit clear: the validity of a null
# constant (ConstantVector::SetNull), shared by every such vector.
def _null_byte_init() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    var address = external_call["calloc", Int](1, 8)
    return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=address)


def _null_byte_keep(address: Optional[Pointer[NoneType, MutUntrackedOrigin]]):
    pass


def _NULL_BYTE_ADDRESS() -> Int:
    from std.ffi import _get_global

    var address = _get_global[
        "DATAFRAME_VECTOR_NULL_BYTE", _null_byte_init, _null_byte_keep
    ]()
    return Int(address.value())


# ConstantVector::ZeroSelectionVector: every row reads entry 0.
def _zero_sel_init() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    var address = external_call["calloc", Int](STANDARD_VECTOR_SIZE, 4)
    return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=address)


def zero_selection() -> SelectionVector:
    """A selection of STANDARD_VECTOR_SIZE zeros, shared process-wide."""
    from std.ffi import _get_global

    var address = _get_global[
        "DATAFRAME_VECTOR_ZERO_SEL", _zero_sel_init, _null_byte_keep
    ]()
    return SelectionVector(Int(address.value()))
