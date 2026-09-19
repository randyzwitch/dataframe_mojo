"""Arrow C Data Interface export and import.

https://arrow.apache.org/docs/format/CDataInterface.html

The interface is two C structs, `ArrowSchema` and `ArrowArray`, allocated by
the consumer and filled by the producer. No Arrow library is involved:
Polars, DuckDB, pyarrow, nanoarrow, or marrow exchange data with us through
these structs directly.

Export shares buffers wherever the layouts already agree, and the exported
array keeps them alive (via reference counts) until the consumer calls
`release`:

| dtype        | format   | export          |
|--------------|----------|-----------------|
| Int64        | `l`      | zero-copy       |
| Float64      | `g`      | zero-copy       |
| String       | `U`      | zero-copy       |
| Datetime     | `ts?:`   | zero-copy       |
| Duration     | `tD?`    | zero-copy       |
| Time         | `ttn`    | zero-copy       |
| Bool         | `b`      | packs values to bits (n / 8 bytes) |
| Date         | `tdD`    | narrows days to Int32 |

Validity bitmaps are always shared; a column window's offset becomes the
ArrowArray `offset`. Import copies into our own buffers (so foreign memory
is never retained), then calls the producer's `release` exactly once.

Pointers in the C structs are held as `Int` addresses: C pointer fields are
nullable and Mojo `Pointer`s are not.
"""
from std.memory import Allocation, ArcPointer, Layout, Pointer, alloc, dealloc
from .bool_column import BoolColumn
from .column import Column, _copy_bits
from .dtype import DataType, NUMERIC_DTYPES
from .frame import DataFrame
from .series import Series
from .string_column import StringColumn


struct ArrowSchema(Movable):
    """`struct ArrowSchema` from the C Data Interface (72 bytes)."""

    var format: Int
    var name: Int
    var metadata: Int
    var flags: Int64
    var n_children: Int64
    var children: Int
    var dictionary: Int
    var release: Int
    var private_data: Int

    def __init__(out self):
        """A released (empty) struct, ready to be filled by a producer."""
        self.format = 0
        self.name = 0
        self.metadata = 0
        self.flags = 0
        self.n_children = 0
        self.children = 0
        self.dictionary = 0
        self.release = 0
        self.private_data = 0


struct ArrowArray(Movable):
    """`struct ArrowArray` from the C Data Interface (80 bytes)."""

    var length: Int64
    var null_count: Int64
    var offset: Int64
    var n_buffers: Int64
    var n_children: Int64
    var buffers: Int
    var children: Int
    var dictionary: Int
    var release: Int
    var private_data: Int

    def __init__(out self):
        """A released (empty) struct, ready to be filled by a producer."""
        self.length = 0
        self.null_count = 0
        self.offset = 0
        self.n_buffers = 0
        self.n_children = 0
        self.buffers = 0
        self.children = 0
        self.dictionary = 0
        self.release = 0
        self.private_data = 0


comptime _SchemaRelease = def(Pointer[ArrowSchema, MutAnyOrigin]) thin abi(
    "C"
) -> None
comptime _ArrayRelease = def(Pointer[ArrowArray, MutAnyOrigin]) thin abi(
    "C"
) -> None

# ArrowSchema.flags
comptime ARROW_FLAG_NULLABLE: Int64 = 2


# ---------------------------------------------------------------------------
# Raw memory helpers


def _at[T: AnyType](address: Int) -> Pointer[T, MutAnyOrigin]:
    return Pointer[T, MutAnyOrigin](unsafe_from_address=address)


def _leak[T: Movable](var value: T) -> Int:
    """Move value to the heap and return its address (see _reclaim)."""
    var pointer = alloc(Layout[T](count=1)).unsafe_leak()
    pointer.unsafe_write(value^)
    return Int(pointer)


def _reclaim[T: Movable](address: Int) -> T:
    """Take back a value leaked with _leak and free its memory."""
    var pointer = Pointer[T, MutUntrackedOrigin](unsafe_from_address=address)
    var value = pointer.unsafe_take_pointee()
    dealloc(Allocation[T](unsafe_owned_ptr=pointer, layout=Layout[T](count=1)))
    return value^


def _c_string(text: String) -> List[UInt8]:
    var bytes = List[UInt8](capacity=text.byte_length() + 1)
    bytes.extend(text.as_bytes())
    bytes.append(0)
    return bytes^


def _read_c_string(address: Int) -> String:
    if address == 0:
        return ""
    var length = 0
    while _at[UInt8](address + length)[] != 0:
        length += 1
    var bytes = List[UInt8](capacity=length)
    for i in range(length):
        bytes.append(_at[UInt8](address + i)[])
    return String(unsafe_from_utf8=bytes^)


# ---------------------------------------------------------------------------
# Export


struct _SchemaState(Movable):
    """Private data of an exported ArrowSchema: its strings and children."""

    var format: List[UInt8]
    var name: List[UInt8]
    var children: List[Int]  # ArrowSchema* (heap), owned

    def __init__(out self, format: String, name: String):
        self.format = _c_string(format)
        self.name = _c_string(name)
        self.children = List[Int]()


struct _ArrayState(Movable):
    """Private data of an exported ArrowArray.

    `keep` holds a Series sharing the exported buffers, so they outlive the
    source frame; `owned` holds buffers converted for export (Bool bits,
    Date32 values). `children` are heap ArrowArrays released with the parent.
    """

    var keep: List[Series]
    var owned: List[List[UInt8]]
    var buffers: List[Int]  # the `const void**` array
    var children: List[Int]  # ArrowArray* (heap), owned
    var releases: Int  # optional Int* counter, for tests

    def __init__(out self):
        self.keep = List[Series]()
        self.owned = List[List[UInt8]]()
        self.buffers = List[Int]()
        self.children = List[Int]()
        self.releases = 0


def _release_schema(schema: Pointer[ArrowSchema, MutAnyOrigin]) abi("C"):
    var state = _reclaim[_SchemaState](schema[].private_data)
    for child in state.children:
        var pointer = _at[ArrowSchema](child)
        if pointer[].release != 0:
            _call_schema_release(pointer)
        _ = _reclaim[ArrowSchema](child)
    schema[].release = 0


def _release_array(array: Pointer[ArrowArray, MutAnyOrigin]) abi("C"):
    var state = _reclaim[_ArrayState](array[].private_data)
    for child in state.children:
        var pointer = _at[ArrowArray](child)
        if pointer[].release != 0:
            _call_array_release(pointer)
        _ = _reclaim[ArrowArray](child)
    if state.releases != 0:
        _at[Int](state.releases)[] += 1
    array[].release = 0


# Function pointers and addresses convert through memory: C fields hold
# addresses (nullable), and Mojo cannot rebind an Int to a function type.


def _schema_release_address() -> Int:
    var callback: _SchemaRelease = _release_schema
    return Pointer(to=callback).unsafe_bitcast[Int]()[]


def _array_release_address() -> Int:
    var callback: _ArrayRelease = _release_array
    return Pointer(to=callback).unsafe_bitcast[Int]()[]


def _call_schema_release(schema: Pointer[ArrowSchema, MutAnyOrigin]):
    var address = schema[].release
    Pointer(to=address).unsafe_bitcast[_SchemaRelease]()[](schema)


def _call_array_release(array: Pointer[ArrowArray, MutAnyOrigin]):
    var address = array[].release
    Pointer(to=address).unsafe_bitcast[_ArrayRelease]()[](array)


def _numeric_format(dtype: DType) -> String:
    """Arrow format characters for the numeric storage types."""
    if dtype == DType.int8:
        return "c"
    if dtype == DType.uint8:
        return "C"
    if dtype == DType.int16:
        return "s"
    if dtype == DType.uint16:
        return "S"
    if dtype == DType.int32:
        return "i"
    if dtype == DType.uint32:
        return "I"
    if dtype == DType.int64:
        return "l"
    if dtype == DType.uint64:
        return "L"
    if dtype == DType.float32:
        return "f"
    return "g"


def _format(dtype: DataType) raises -> String:
    if dtype.is_numeric():
        return _numeric_format(dtype.storage().value())
    if dtype == DataType.BOOL:
        return "b"
    if dtype == DataType.STRING:
        return "U"
    if dtype == DataType.DATE:
        return "tdD"
    if dtype == DataType.TIME:
        return "ttn"
    var unit = String(dtype.unit()[byte=0])
    if dtype.is_datetime():
        return "ts" + unit + ":"
    if dtype.is_duration():
        return "tD" + unit
    raise Error("Cannot export dtype " + dtype.name() + " to Arrow")


def _fill_schema(mut schema: ArrowSchema, series: Series) raises:
    var state = _SchemaState(_format(series.dtype()), series.name())
    schema.format = Int(state.format.unsafe_ptr())
    schema.name = Int(state.name.unsafe_ptr())
    schema.metadata = 0
    schema.flags = ARROW_FLAG_NULLABLE
    schema.n_children = 0
    schema.children = 0
    schema.dictionary = 0
    schema.private_data = _leak(state^)
    schema.release = _schema_release_address()


def _fill_array(
    mut array: ArrowArray, series: Series, releases: Int = 0
) raises:
    var state = _ArrayState()
    state.releases = releases
    state.keep.append(series.copy())
    ref kept = state.keep[0]
    var length = len(series)
    array.length = Int64(length)
    array.null_count = Int64(series.null_count())
    array.n_children = 0
    array.children = 0
    array.dictionary = 0
    var dtype = series.dtype()
    if kept._data.isa[StringColumn]():
        ref column = kept._data[StringColumn]
        array.offset = Int64(column._offset)
        state.buffers.append(Int(column._bits[].unsafe_ptr()))
        state.buffers.append(Int(column._offsets[].unsafe_ptr()))
        state.buffers.append(Int(column._bytes[].unsafe_ptr()))
    elif kept._data.isa[BoolColumn]():
        # Values are already an Arrow bitmap: zero-copy, like every other
        # fixed-width type.
        ref column = kept._data[BoolColumn]
        array.offset = Int64(column._offset)
        state.buffers.append(Int(column._bits[].unsafe_ptr()))
        state.buffers.append(Int(column._data[].unsafe_ptr()))
    elif dtype == DataType.DATE:
        # Arrow date32 holds Int32 days; narrow (range-checked) at export.
        ref column = kept._data[Column[Int64]]
        var days = List[UInt8](length=4 * length, fill=0)
        var out = days.unsafe_ptr().unsafe_bitcast[Int32]()
        for i in range(length):
            var day = column._get(i)
            if day < Int64(Int32.MIN) or day > Int64(Int32.MAX):
                raise Error("date outside the Arrow date32 range")
            out.unsafe_offset(i).unsafe_store(Int32(day))
        state.owned.append(_copy_bits(column._bits[], column._offset, length))
        state.owned.append(days^)
        array.offset = 0
        state.buffers.append(Int(state.owned[0].unsafe_ptr()))
        state.buffers.append(Int(state.owned[1].unsafe_ptr()))
    else:
        # Every numeric (and remaining temporal) type is zero-copy.
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if kept._data.isa[Column[Scalar[D]]]():
                ref column = kept._data[Column[Scalar[D]]]
                array.offset = Int64(column._offset)
                state.buffers.append(Int(column._bits[].unsafe_ptr()))
                state.buffers.append(Int(column._data[].unsafe_ptr()))
    array.n_buffers = Int64(len(state.buffers))
    array.buffers = Int(state.buffers.unsafe_ptr())
    array.private_data = _leak(state^)
    array.release = _array_release_address()


def export_arrow_series(
    series: Series, mut array: ArrowArray, mut schema: ArrowSchema
) raises:
    """Fill consumer-allocated ArrowArray and ArrowSchema structs.

    The consumer owns both structs afterwards and must call each `release`
    once; the exported buffers stay alive until then.
    """
    _export_series(series, array, schema, 0)


def _export_series(
    series: Series,
    mut array: ArrowArray,
    mut schema: ArrowSchema,
    releases: Int,
) raises:
    _fill_schema(schema, series)
    _fill_array(array, series, releases)


def export_arrow_series(
    series: Series, array_address: Int, schema_address: Int
) raises:
    """Export into structs at raw addresses (for C or Python consumers)."""
    export_arrow_series(
        series,
        _at[ArrowArray](array_address)[],
        _at[ArrowSchema](schema_address)[],
    )


def export_arrow(
    frame: DataFrame, mut array: ArrowArray, mut schema: ArrowSchema
) raises:
    """Export a frame as an Arrow struct array (format `+s`), the shape
    pyarrow and Polars import as a RecordBatch."""
    _export_frame(frame, array, schema, 0)


def export_arrow(
    frame: DataFrame, array_address: Int, schema_address: Int
) raises:
    """Export into structs at raw addresses (for C or Python consumers)."""
    export_arrow(
        frame,
        _at[ArrowArray](array_address)[],
        _at[ArrowSchema](schema_address)[],
    )


def _set_struct_schema(mut schema: ArrowSchema, var state: _SchemaState):
    """Fill a "+s" schema; `state` moves to the heap as private data."""
    schema.format = Int(state.format.unsafe_ptr())
    schema.name = Int(state.name.unsafe_ptr())
    schema.metadata = 0
    schema.flags = 0
    schema.n_children = Int64(len(state.children))
    schema.children = Int(state.children.unsafe_ptr())
    schema.dictionary = 0
    schema.release = _schema_release_address()
    schema.private_data = _leak(state^)


def _set_struct_array(
    mut array: ArrowArray, length: Int, var state: _ArrayState
):
    """Fill a struct array; `state` moves to the heap as private data."""
    array.length = Int64(length)
    array.null_count = 0
    array.offset = 0
    array.n_buffers = 1
    array.buffers = Int(state.buffers.unsafe_ptr())
    array.n_children = Int64(len(state.children))
    array.children = Int(state.children.unsafe_ptr())
    array.dictionary = 0
    array.release = _array_release_address()
    array.private_data = _leak(state^)


def _export_frame(
    frame: DataFrame,
    mut array: ArrowArray,
    mut schema: ArrowSchema,
    releases: Int,
) raises:
    var schema_state = _SchemaState("+s", "")
    var array_state = _ArrayState()
    array_state.releases = releases
    array_state.buffers.append(0)  # a struct array has no validity here
    for column in frame._columns:
        var child_schema = _leak(ArrowSchema())
        var child_array = _leak(ArrowArray())
        _fill_schema(_at[ArrowSchema](child_schema)[], column)
        _fill_array(_at[ArrowArray](child_array)[], column, releases)
        schema_state.children.append(child_schema)
        array_state.children.append(child_array)
    # Write through `mut` arguments: after a value is moved, any MutAnyOrigin
    # dereference in the same function counts as a later use of it.
    _set_struct_schema(schema, schema_state^)
    _set_struct_array(array, frame.height(), array_state^)


# ---------------------------------------------------------------------------
# Import


def _buffer(array: ArrowArray, index: Int) -> Int:
    return _at[Int](array.buffers + 8 * index)[]


def _import_bits(address: Int, offset: Int, length: Int) -> List[UInt8]:
    """Validity bits [offset, offset + length), rebased; all valid if NULL."""
    if address == 0:
        return List[UInt8](length=(length + 7) // 8, fill=255)
    var first = offset // 8
    var bytes = (offset + length + 7) // 8 - first
    var source = List[UInt8](capacity=bytes)
    source.extend(_span[UInt8](address, first, bytes))
    return _copy_bits(source, offset % 8, length)


def _read[T: TrivialRegisterPassable](address: Int, index: Int) -> T:
    return _at[T](address).unsafe_offset(index)[]


def _span[
    T: TrivialRegisterPassable
](address: Int, start: Int, length: Int) -> Span[T, MutAnyOrigin]:
    return Span[T, MutAnyOrigin](
        unsafe_ptr=_at[T](address).unsafe_offset(start), length=length
    )


def _import_fixed[
    T: TrivialRegisterPassable
](array: ArrowArray, length: Int, offset: Int) -> List[T]:
    """Bulk-copy `length` values starting at `offset` from buffer 1."""
    var values = List[T](capacity=length)
    values.extend(_span[T](_buffer(array, 1), offset, length))
    return values^


def _as_int64[
    T: TrivialRegisterPassable & Intable
](array: ArrowArray, length: Int, offset: Int, scale: Int64 = 1) -> List[Int64]:
    var values = List[Int64](capacity=length)
    var data = _buffer(array, 1)
    for i in range(length):
        values.append(Int64(Int(_read[T](data, offset + i))) * scale)
    return values^


def _int64_column(
    var values: List[Int64], var bits: List[UInt8]
) -> Column[Int64]:
    var column = Column[Int64](values^)
    column._bits = ArcPointer(bits^)
    return column^


def _import_child(array: ArrowArray, schema: ArrowSchema) raises -> Series:
    var format = _read_c_string(schema.format)
    var name = _read_c_string(schema.name)
    var length = Int(array.length)
    var offset = Int(array.offset)
    if array.dictionary != 0 or schema.dictionary != 0:
        raise Error("Arrow dictionary arrays are not supported")
    var bits = _import_bits(_buffer(array, 0), offset, length)
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if format == _numeric_format(D):
            var column = Column[Scalar[D]](
                _import_fixed[Scalar[D]](array, length, offset)
            )
            column._bits = ArcPointer(bits^)
            return Series(name, column^)
    if format == "b":
        return Series(
            name,
            BoolColumn(
                values=_import_bits(_buffer(array, 1), offset, length),
                bits=bits^,
                length=length,
            ),
        )
    if format == "U" or format == "u":
        var large = format == "U"
        var offsets_address = _buffer(array, 1)
        var data = _buffer(array, 2)

        def text_offset(i: Int) {imm large, imm offsets_address} -> Int:
            if large:
                return Int(_read[Int64](offsets_address, i))
            return Int(_read[Int32](offsets_address, i))

        var first = text_offset(offset)
        var total = text_offset(offset + length) - first
        var text = List[UInt8](capacity=total)
        text.extend(_span[UInt8](data, first, total))
        var offsets = List[Int64](capacity=length + 1)
        for i in range(length + 1):
            offsets.append(Int64(text_offset(offset + i) - first))
        return Series(
            name,
            StringColumn(
                bytes=text^, offsets=offsets^, bits=bits^, length=length
            ),
        )
    if format == "tdD":
        return Series(
            name, _int64_column(_as_int64[Int32](array, length, offset), bits^)
        ).with_dtype(DataType.DATE)
    if format == "tdm":
        var ms = _import_fixed[Int64](array, length, offset)
        for i in range(length):
            ms[i] = ms[i] // 86_400_000
        return Series(name, _int64_column(ms^, bits^)).with_dtype(DataType.DATE)
    if format == "ttn":
        return Series(
            name,
            _int64_column(_import_fixed[Int64](array, length, offset), bits^),
        ).with_dtype(DataType.TIME)
    if format == "ttu":
        return Series(
            name,
            _int64_column(_as_int64[Int64](array, length, offset, 1000), bits^),
        ).with_dtype(DataType.TIME)
    if format == "ttm":
        return Series(
            name,
            _int64_column(
                _as_int64[Int32](array, length, offset, 1_000_000), bits^
            ),
        ).with_dtype(DataType.TIME)
    if format == "tts":
        return Series(
            name,
            _int64_column(
                _as_int64[Int32](array, length, offset, 1_000_000_000), bits^
            ),
        ).with_dtype(DataType.TIME)
    if format.startswith("ts") and format.byte_length() >= 4:
        if format.byte_length() > 4:
            raise Error(
                "Arrow timestamps with a time zone are not supported: " + format
            )
        var dtype = DataType.datetime(_unit(format[byte=2]))
        return Series(
            name,
            _int64_column(_import_fixed[Int64](array, length, offset), bits^),
        ).with_dtype(dtype)
    if format.startswith("tD") and format.byte_length() == 3:
        var dtype = DataType.duration(_unit(format[byte=2]))
        return Series(
            name,
            _int64_column(_import_fixed[Int64](array, length, offset), bits^),
        ).with_dtype(dtype)
    raise Error("Unsupported Arrow format '" + format + "' for column " + name)


def _unit(code: StringSlice) raises -> String:
    if code == "n":
        return "ns"
    if code == "u":
        return "us"
    if code == "m":
        return "ms"
    raise Error("Unsupported Arrow time unit '" + String(code) + "'")


def _release_imported(mut array: ArrowArray, mut schema: ArrowSchema):
    if array.release != 0:
        _call_array_release(
            Pointer(to=array).unsafe_origin_cast[MutAnyOrigin]()
        )
    if schema.release != 0:
        _call_schema_release(
            Pointer(to=schema).unsafe_origin_cast[MutAnyOrigin]()
        )


def import_arrow_series(
    mut array: ArrowArray, mut schema: ArrowSchema
) raises -> Series:
    """Copy an exported Arrow array into a Series, then release it.

    The input structs are consumed (released) even when import fails.
    """
    try:
        if array.release == 0 or schema.release == 0:
            raise Error("Arrow array or schema was already released")
        var result = _import_child(array, schema)
        _release_imported(array, schema)
        return result^
    except e:
        _release_imported(array, schema)
        raise e^


def import_arrow_series(
    array_address: Int, schema_address: Int
) raises -> Series:
    """Import from structs at raw addresses (from C or Python producers)."""
    return import_arrow_series(
        _at[ArrowArray](array_address)[], _at[ArrowSchema](schema_address)[]
    )


def import_arrow(
    mut array: ArrowArray, mut schema: ArrowSchema
) raises -> DataFrame:
    """Copy an exported Arrow struct array (a record batch) into a frame,
    then release it. The input structs are consumed even when import fails.
    """
    try:
        if array.release == 0 or schema.release == 0:
            raise Error("Arrow array or schema was already released")
        var format = _read_c_string(schema.format)
        if format != "+s":
            raise Error(
                "Expected an Arrow struct array ('+s'), got '" + format + "'"
            )
        if array.n_children != schema.n_children:
            raise Error("Arrow array and schema child counts differ")
        if _buffer(array, 0) != 0 and array.null_count != 0:
            raise Error("Arrow struct arrays with null rows are not supported")
        var columns = List[Series]()
        for k in range(Int(array.n_children)):
            var child = _at[ArrowArray](_at[Int](array.children + 8 * k)[])
            var child_schema = _at[ArrowSchema](
                _at[Int](schema.children + 8 * k)[]
            )
            var series = _import_child(child[], child_schema[])
            # A parent offset shifts every child.
            columns.append(series.slice(Int(array.offset), Int(array.length)))
        var result = DataFrame(columns^, height=Int(array.length))
        _release_imported(array, schema)
        return result^
    except e:
        _release_imported(array, schema)
        raise e^


def import_arrow(array_address: Int, schema_address: Int) raises -> DataFrame:
    """Import from structs at raw addresses (from C or Python producers)."""
    return import_arrow(
        _at[ArrowArray](array_address)[], _at[ArrowSchema](schema_address)[]
    )
