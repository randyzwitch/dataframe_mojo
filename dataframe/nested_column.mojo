"""List and struct columns over shared child series.

`ListColumn` is Arrow `large_list`: Int64 offsets into one child series plus
a validity bitmap; row i holds child rows `[offsets[i], offsets[i + 1])`.
`StructColumn` is Arrow `struct`: named child series of equal length plus a
validity bitmap for the struct itself. Both are windows (offset, length)
onto shared buffers, like `Column` and `StringColumn`, so copies and slices
are O(1) and buffers are never mutated while shared.

A child is itself a `Series`, and a `Series` can hold one of these columns,
so the type would refer to itself. Mojo rejects that cycle in an imported
module however it is routed (`List`, `Variant`, `ArcPointer`), so children
live on the heap behind `_Held`: a shared cell holding only an address and
which kind of value to free. Method signatures may still name `Series`;
fields may not. `_Box` is the typed form for holders outside the cycle.
"""
from std.memory import Allocation, ArcPointer, Layout, Pointer, alloc, dealloc

from .column import (
    _bit,
    _count_valid,
    _copy_validity,
    _pack_bits,
)
from .dtype import DataType
from .series import Series


def _leak[T: Movable](var value: T) -> Int:
    var pointer = alloc(Layout[T](count=1)).unsafe_leak()
    pointer.unsafe_write(value^)
    return Int(pointer)


def _reclaim[T: Deinitable & Movable](address: Int) -> T:
    var pointer = Pointer[T, MutUntrackedOrigin](unsafe_from_address=address)
    var value = pointer.unsafe_take_pointee()
    dealloc(Allocation[T](unsafe_owned_ptr=pointer, layout=Layout[T](count=1)))
    return value^


struct _Box[T: Deinitable & Movable](Deinitable, Movable):
    """Owns one heap value by address, so a type can contain its own kind."""

    var address: Int

    def __init__(out self, var value: Self.T):
        self.address = _leak[Self.T](value^)

    def __deinit__(deinit self):
        _ = _reclaim[Self.T](self.address)

    def get(self) -> ref[MutAnyOrigin] Self.T:
        return Pointer[Self.T, MutAnyOrigin](unsafe_from_address=self.address)[]


struct _Held(Deinitable, Movable):
    """Owns a heap `Series` (or `List[Series]`) by address, type-erased so a
    column's fields never name the series type that contains the column."""

    var address: Int
    var many: Bool

    def __init__(out self, address: Int, many: Bool):
        self.address = address
        self.many = many

    def __deinit__(deinit self):
        if self.many:
            _ = _reclaim[List[Series]](self.address)
        else:
            _ = _reclaim[Series](self.address)


def _hold(var series: Series) -> ArcPointer[_Held]:
    return ArcPointer(_Held(_leak[Series](series^), False))


def _hold_many(var fields: List[Series]) -> ArcPointer[_Held]:
    return ArcPointer(_Held(_leak[List[Series]](fields^), True))


def _series_at(held: ArcPointer[_Held]) -> ref[MutAnyOrigin] Series:
    return Pointer[Series, MutAnyOrigin](unsafe_from_address=held[].address)[]


def _fields_at(held: ArcPointer[_Held]) -> ref[MutAnyOrigin] List[Series]:
    return Pointer[List[Series], MutAnyOrigin](
        unsafe_from_address=held[].address
    )[]


def _check_bits(bits: List[UInt8], length: Int) raises:
    if len(bits) != 0 and len(bits) < (length + 7) // 8:
        raise Error("Validity bitmap is shorter than the column")


struct ListColumn(Copyable, Deinitable, Movable, Sized):
    """A window onto shared list offsets, validity, and one child series."""

    var _offsets: ArcPointer[List[Int64]]
    var _bits: ArcPointer[List[UInt8]]
    var _offset: Int
    var _length: Int
    var _child: ArcPointer[_Held]

    def __init__(
        out self,
        var offsets: List[Int64],
        var child: Series,
        var bits: List[UInt8] = List[UInt8](),
    ) raises:
        """Adopt offsets (one more than the rows), a child, and an optional
        prepacked validity bitmap. Offsets must not decrease and must stay
        within the child."""
        if len(offsets) == 0:
            raise Error("List offsets need at least one entry")
        var previous = offsets[0]
        if previous < 0:
            raise Error("List offsets must not be negative")
        for i in range(1, len(offsets)):
            if offsets[i] < previous:
                raise Error("List offsets must not decrease")
            previous = offsets[i]
        if Int(previous) > len(child):
            raise Error("List offsets run past the child series")
        var length = len(offsets) - 1
        _check_bits(bits, length)
        self._offsets = ArcPointer(offsets^)
        self._bits = ArcPointer(bits^)
        self._offset = 0
        self._length = length
        self._child = _hold(child^)

    @staticmethod
    def from_lists(
        rows: List[Series], inner: DataType, valid: List[Bool] = List[Bool]()
    ) raises -> Self:
        """Build from one series per row (an empty series for an empty list;
        `valid[i] == False` makes row i null)."""
        var offsets = List[Int64](capacity=len(rows) + 1)
        offsets.append(0)
        var parts = List[Series]()
        var total = 0
        for row in rows:
            if row.dtype() != inner:
                raise Error(
                    "List rows must share the element dtype "
                    + inner.name()
                    + ", found "
                    + row.dtype().name()
                )
            total += len(row)
            offsets.append(Int64(total))
            if len(row) > 0:
                parts.append(row.renamed("item"))
        var child: Series
        if len(parts) == 0:
            child = Series.full_null("item", inner, 0)
        else:
            child = Series._from_chunks(parts).rechunk()
        var bits = List[UInt8]()
        if len(valid) > 0:
            if len(valid) != len(rows):
                raise Error("List validity must have one entry per row")
            bits = _pack_bits(valid)
        return Self(offsets^, child^, bits^)

    @staticmethod
    def _nulls(length: Int, inner: DataType) raises -> Self:
        var offsets = List[Int64](length=length + 1, fill=0)
        var valid = List[Bool](length=length, fill=False)
        return Self(
            offsets^, Series.full_null("item", inner, 0), _pack_bits(valid)
        )

    def __len__(self) -> Int:
        return self._length

    def dtype(self) -> DataType:
        return DataType.list(self.child().dtype())

    def child(self) -> Series:
        """The whole child series (shared, O(1))."""
        return _series_at(self._child).copy()

    def _valid(self, i: Int) -> Bool:
        return len(self._bits[]) == 0 or _bit(self._bits[], self._offset + i)

    def is_null(self, i: Int) raises -> Bool:
        if i < 0 or i >= self._length:
            raise Error("Column index out of bounds")
        return not self._valid(i)

    def null_count(self) -> Int:
        if len(self._bits[]) == 0:
            return 0
        return self._length - _count_valid(
            self._bits[], self._offset, self._length
        )

    def _start(self, i: Int) -> Int:
        return Int(self._offsets[][self._offset + i])

    def _end(self, i: Int) -> Int:
        return Int(self._offsets[][self._offset + i + 1])

    def element_count(self, i: Int) -> Int:
        """Elements in row i (0 for null rows)."""
        return self._end(i) - self._start(i)

    def row(self, i: Int) raises -> Series:
        """Row i's elements as a series (empty for an empty or null row)."""
        if i < 0 or i >= self._length:
            raise Error("Column index out of bounds")
        return self.child().slice(self._start(i), self.element_count(i))

    def slice(self, offset: Int, length: Int) raises -> Self:
        if offset < 0 or length < 0 or offset + length > self._length:
            raise Error("Invalid column slice")
        var result = self.copy()
        result._offset = self._offset + offset
        result._length = length
        return result^

    def take(self, indices: List[Int]) raises -> Self:
        """Rows by index; gathers the child rows each list refers to."""
        var offsets = List[Int64](capacity=len(indices) + 1)
        offsets.append(0)
        var child_rows = List[Int]()
        var valid = List[Bool](capacity=len(indices))
        var total = 0
        for i in indices:
            if i < 0 or i >= self._length:
                raise Error("Row index out of bounds")
            valid.append(self._valid(i))
            for r in range(self._start(i), self._end(i)):
                child_rows.append(r)
            total += self.element_count(i)
            offsets.append(Int64(total))
        var bits = List[UInt8]()
        if len(self._bits[]) != 0:
            bits = _pack_bits(valid)
        return Self(offsets^, self.child().take(child_rows), bits^)

    def take_or_null(self, indices: List[Int]) raises -> Self:
        """Like take, but a negative index yields a null row."""
        var offsets = List[Int64](capacity=len(indices) + 1)
        offsets.append(0)
        var child_rows = List[Int]()
        var valid = List[Bool](capacity=len(indices))
        var total = 0
        for i in indices:
            if i >= self._length:
                raise Error("Row index out of bounds")
            if i < 0:
                valid.append(False)
                offsets.append(Int64(total))
                continue
            valid.append(self._valid(i))
            for r in range(self._start(i), self._end(i)):
                child_rows.append(r)
            total += self.element_count(i)
            offsets.append(Int64(total))
        return Self(offsets^, self.child().take(child_rows), _pack_bits(valid))

    def _broadcast(self, length: Int) raises -> Self:
        if self._length != 1:
            raise Error("Only a one-row column can be broadcast")
        var indices = List[Int](length=length, fill=0)
        return self.take(indices)

    def _append_column(self, other: Self) raises -> Self:
        """A new column with other's rows after this one's."""
        var offsets = List[Int64](capacity=self._length + other._length + 1)
        var valid = List[Bool](capacity=self._length + other._length)
        offsets.append(0)
        var total = 0
        for i in range(self._length):
            total += self.element_count(i)
            offsets.append(Int64(total))
            valid.append(self._valid(i))
        for i in range(other._length):
            total += other.element_count(i)
            offsets.append(Int64(total))
            valid.append(other._valid(i))
        var mine = self.child().slice(
            self._start(0) if self._length > 0 else 0,
            (self._end(self._length - 1) - self._start(0)) if self._length
            > 0 else 0,
        )
        var theirs = other.child().slice(
            other._start(0) if other._length > 0 else 0,
            (other._end(other._length - 1) - other._start(0)) if other._length
            > 0 else 0,
        )
        var child = Series._from_chunks([mine^, theirs^]).rechunk()
        var bits = List[UInt8]()
        if len(self._bits[]) != 0 or len(other._bits[]) != 0:
            bits = _pack_bits(valid)
        return Self(offsets^, child^, bits^)

    def equals(self, other: Self) -> Bool:
        if self._length != other._length:
            return False
        for i in range(self._length):
            if self._valid(i) != other._valid(i):
                return False
            if not self._valid(i):
                continue
            try:
                if not self.row(i).equals(other.row(i)):
                    return False
            except:
                return False
        return True

    def lengths(self) -> List[Int64]:
        """Elements per row; null rows count 0 (pair with the validity)."""
        var out = List[Int64](capacity=self._length)
        for i in range(self._length):
            out.append(Int64(self.element_count(i)))
        return out^

    def validity(self) -> List[Bool]:
        var out = List[Bool](capacity=self._length)
        for i in range(self._length):
            out.append(self._valid(i))
        return out^

    def unsafe_validity(self) -> Int:
        """Address of the validity bitmap, 0 when every row is valid."""
        if len(self._bits[]) == 0:
            return 0
        return Int(self._bits[].unsafe_ptr())


struct StructColumn(Copyable, Deinitable, Movable, Sized):
    """A window onto named child series of equal length and a validity."""

    var _fields: ArcPointer[_Held]
    var _bits: ArcPointer[List[UInt8]]
    var _offset: Int
    var _length: Int

    def __init__(
        out self,
        var fields: List[Series],
        var bits: List[UInt8] = List[UInt8](),
    ) raises:
        """Adopt named children (their names are the field names) and an
        optional prepacked validity bitmap for the struct rows."""
        if len(fields) == 0:
            raise Error("A struct needs at least one field")
        var length = len(fields[0])
        for i in range(len(fields)):
            if len(fields[i]) != length:
                raise Error("Struct fields must have equal lengths")
            for j in range(i):
                if fields[i].name() == fields[j].name():
                    raise Error(
                        "Struct field names must be unique: " + fields[i].name()
                    )
        _check_bits(bits, length)
        self._fields = _hold_many(fields^)
        self._bits = ArcPointer(bits^)
        self._offset = 0
        self._length = length

    @staticmethod
    def _nulls(length: Int, dtype: DataType) raises -> Self:
        var fields = List[Series]()
        var names = dtype.field_names()
        var dtypes = dtype.field_dtypes()
        for i in range(len(names)):
            fields.append(Series.full_null(names[i], dtypes[i], length))
        var valid = List[Bool](length=length, fill=False)
        return Self(fields^, _pack_bits(valid))

    def __len__(self) -> Int:
        return self._length

    def dtype(self) -> DataType:
        var names = List[String]()
        var dtypes = List[DataType]()
        for field in _fields_at(self._fields):
            names.append(field.name())
            dtypes.append(field.dtype())
        try:
            return DataType.struct(names^, dtypes^)
        except:
            return DataType.STRING

    def field_count(self) -> Int:
        return len(_fields_at(self._fields))

    def field_names(self) -> List[String]:
        var names = List[String]()
        for field in _fields_at(self._fields):
            names.append(field.name())
        return names^

    def field(self, index: Int) raises -> Series:
        """Field `index` over this window (shared, O(1))."""
        if index < 0 or index >= self.field_count():
            raise Error("Struct field index out of range")
        return _fields_at(self._fields)[index].slice(self._offset, self._length)

    def field(self, name: String) raises -> Series:
        ref fields = _fields_at(self._fields)
        for i in range(len(fields)):
            if fields[i].name() == name:
                return self.field(i)
        raise Error("Struct has no field named " + name)

    def _valid(self, i: Int) -> Bool:
        return len(self._bits[]) == 0 or _bit(self._bits[], self._offset + i)

    def is_null(self, i: Int) raises -> Bool:
        if i < 0 or i >= self._length:
            raise Error("Column index out of bounds")
        return not self._valid(i)

    def null_count(self) -> Int:
        if len(self._bits[]) == 0:
            return 0
        return self._length - _count_valid(
            self._bits[], self._offset, self._length
        )

    def slice(self, offset: Int, length: Int) raises -> Self:
        if offset < 0 or length < 0 or offset + length > self._length:
            raise Error("Invalid column slice")
        var result = self.copy()
        result._offset = self._offset + offset
        result._length = length
        return result^

    def _windowed_fields(self) raises -> List[Series]:
        var out = List[Series]()
        for i in range(self.field_count()):
            out.append(self.field(i))
        return out^

    def take(self, indices: List[Int]) raises -> Self:
        var fields = List[Series]()
        for field in self._windowed_fields():
            fields.append(field.take(indices))
        var bits = List[UInt8]()
        if len(self._bits[]) != 0:
            var valid = List[Bool](capacity=len(indices))
            for i in indices:
                if i < 0 or i >= self._length:
                    raise Error("Row index out of bounds")
                valid.append(self._valid(i))
            bits = _pack_bits(valid)
        return Self(fields^, bits^)

    def take_or_null(self, indices: List[Int]) raises -> Self:
        var fields = List[Series]()
        for field in self._windowed_fields():
            fields.append(field.take_or_null(indices))
        var valid = List[Bool](capacity=len(indices))
        for i in indices:
            if i >= self._length:
                raise Error("Row index out of bounds")
            valid.append(i >= 0 and self._valid(i))
        return Self(fields^, _pack_bits(valid))

    def _broadcast(self, length: Int) raises -> Self:
        if self._length != 1:
            raise Error("Only a one-row column can be broadcast")
        var indices = List[Int](length=length, fill=0)
        return self.take(indices)

    def _append_column(self, other: Self) raises -> Self:
        if self.field_count() != other.field_count():
            raise Error("Cannot append structs with different fields")
        var fields = List[Series]()
        for i in range(self.field_count()):
            var mine = self.field(i)
            var theirs = other.field(i)
            if mine.name() != theirs.name():
                raise Error("Cannot append structs with different fields")
            fields.append(Series._from_chunks([mine^, theirs^]).rechunk())
        var bits = List[UInt8]()
        if len(self._bits[]) != 0 or len(other._bits[]) != 0:
            var valid = List[Bool](capacity=self._length + other._length)
            for i in range(self._length):
                valid.append(self._valid(i))
            for i in range(other._length):
                valid.append(other._valid(i))
            bits = _pack_bits(valid)
        return Self(fields^, bits^)

    def equals(self, other: Self) -> Bool:
        if (
            self._length != other._length
            or self.field_count() != other.field_count()
        ):
            return False
        for i in range(self._length):
            if self._valid(i) != other._valid(i):
                return False
        try:
            for k in range(self.field_count()):
                var a = self.field(k)
                var b = other.field(k)
                if a.name() != b.name() or not a.equals(b):
                    return False
        except:
            return False
        return True

    def validity(self) -> List[Bool]:
        var out = List[Bool](capacity=self._length)
        for i in range(self._length):
            out.append(self._valid(i))
        return out^

    def unsafe_validity(self) -> Int:
        if len(self._bits[]) == 0:
            return 0
        return Int(self._bits[].unsafe_ptr())
