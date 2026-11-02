"""DuckDB 2.0's `DataChunk` and `VectorCache` (common/types/data_chunk.hpp,
common/types/data_chunk.cpp, common/types/vector_cache.cpp), and the
bridge between vectors and our at-rest columns.

A chunk holds at most STANDARD_VECTOR_SIZE rows. Its buffers (value
caches, selection buffers, decode scratch) are allocated the first time a
step writes them and reused after every `reset`, so a pipeline allocates
per worker, not per chunk, as DuckDB's operators keep one output chunk
each and `Reset` it.

At-rest columns stay the storage format, as DuckDB keeps its column
segments apart from its vectors. A `ColumnArena` holds the columns a
worker's vectors read (the morsel's source columns, join builds, columns
gathered for an adapter), and `flat_over` wraps a window of one as a flat
vector without copying. `to_series` turns a vector back into a column for
operators not yet on vectors.
"""
from std.memory import Pointer, unsafe_memcpy

from dataframe.dtype import DataType
from dataframe.series import Series
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.storage.string import StringBuilder, StringColumn
from dataframe.vector.types import (
    CONSTANT_VECTOR,
    DICTIONARY_VECTOR,
    FLAT_VECTOR,
    PHYSICAL_BOOL,
    PHYSICAL_INT128,
    PHYSICAL_INT64,
    PHYSICAL_OPAQUE,
    PHYSICAL_VARCHAR,
    SEQUENCE_VECTOR,
    STANDARD_VECTOR_SIZE,
    SelectionVector,
    StringT,
    UnifiedVectorFormat,
    ValidityMask,
    Vector,
    physical_of,
    physical_size,
    set_invalid,
    string_data,
    zero_selection,
)

comptime _NUMERIC_STORAGE = [
    DType.int64,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float32,
    DType.int128,
]


struct ColumnArena(Movable):
    """The at-rest columns a worker's vectors read, by handle. A vector
    names its column by handle (`Vector.source`), so it holds no reference
    count; the arena keeps the column alive. `mark` and `truncate` drop
    the columns added for one morsel."""

    var columns: List[Series]

    def __init__(out self):
        self.columns = List[Series]()

    def add(mut self, series: Series) raises -> Int:
        """Hold `series` as one contiguous array; its handle."""
        if series.is_chunked():
            if series.n_chunks() == 1:
                self.columns.append(series.chunks()[0].copy())
            else:
                self.columns.append(series.rechunk())
        else:
            self.columns.append(series.copy())
        return len(self.columns) - 1

    def mark(self) -> Int:
        return len(self.columns)

    def truncate(mut self, mark: Int):
        while len(self.columns) > mark:
            _ = self.columns.pop()


def flat_over(arena: ColumnArena, handle: Int, offset: Int) -> Vector:
    """A flat vector over the arena column `handle` from row `offset` (no
    copy; Vector::Slice(other, offset, end) of a column segment). Fixed
    width values are read where they are; strings and booleans are left
    encoded until a kernel reads them."""
    ref series = arena.columns[handle]
    var result = Vector()
    result.source = handle
    result.source_offset = offset
    comptime for k in range(len(_NUMERIC_STORAGE)):
        comptime D = _NUMERIC_STORAGE[k]
        if series._data.isa[Column[Scalar[D]]]():
            ref column = series._data[Column[Scalar[D]]]
            result.physical = physical_of[D]()
            result.data = Int(column._ptr()) + offset * physical_size(
                result.physical
            )
            if len(column._bits[]) > 0:
                result.validity = ValidityMask(
                    Int(column._bits[].unsafe_ptr()), column._offset + offset
                )
            return result
    if series._data.isa[BoolColumn]():
        ref column = series._data[BoolColumn]
        result.physical = PHYSICAL_BOOL
        if len(column._bits[]) > 0:
            result.validity = ValidityMask(
                Int(column._bits[].unsafe_ptr()), column._offset + offset
            )
        return result
    if series._data.isa[StringColumn]():
        ref column = series._data[StringColumn]
        result.physical = PHYSICAL_VARCHAR
        if len(column._bits[]) > 0:
            result.validity = ValidityMask(
                Int(column._bits[].unsafe_ptr()), column._offset + offset
            )
        return result
    result.physical = PHYSICAL_OPAQUE
    return result


struct DataChunk(Movable):
    """A set of vectors of equal length (DuckDB `DataChunk`), at most
    STANDARD_VECTOR_SIZE rows.

    `caches` are the per-column value buffers (`VectorCache`) a step writes
    its results into; `selections` the selection buffers this chunk's
    dictionary vectors point at (DuckDB's SelectionData, owned here rather
    than shared); `scratch` the buffers `unified` decodes encoded vectors
    into. All three are kept across `reset`.
    """

    var data: List[Vector]
    var count: Int
    var caches: List[List[UInt8]]
    var validity_caches: List[List[UInt8]]
    var selections: List[List[UInt32]]
    var selections_used: Int
    var scratch: List[List[UInt8]]
    var scratch_used: Int

    def __init__(out self, columns: Int = 0):
        self.data = List[Vector](length=columns, fill=Vector())
        self.count = 0
        self.caches = List[List[UInt8]]()
        self.validity_caches = List[List[UInt8]]()
        self.selections = List[List[UInt32]]()
        self.selections_used = 0
        self.scratch = List[List[UInt8]]()
        self.scratch_used = 0

    def size(self) -> Int:
        return self.count

    def column_count(self) -> Int:
        return len(self.data)

    def reset(mut self):
        """DataChunk::Reset: no rows, every buffer kept for reuse."""
        self.count = 0
        self.selections_used = 0
        self.scratch_used = 0
        for i in range(len(self.data)):
            self.data[i] = Vector()

    def set_columns(mut self, columns: Int):
        """Resize to `columns` vectors (InitializeEmpty)."""
        while len(self.data) > columns:
            _ = self.data.pop()
        while len(self.data) < columns:
            self.data.append(Vector())

    def new_selection(mut self) -> SelectionVector:
        """A selection buffer of STANDARD_VECTOR_SIZE entries owned by this
        chunk until its next reset."""
        if self.selections_used == len(self.selections):
            self.selections.append(
                List[UInt32](unsafe_uninit_length=STANDARD_VECTOR_SIZE)
            )
        var at = Int(self.selections[self.selections_used].unsafe_ptr())
        self.selections_used += 1
        return SelectionVector(at)

    def _new_scratch(mut self, bytes: Int) -> Int:
        if self.scratch_used == len(self.scratch):
            self.scratch.append(List[UInt8]())
        ref buffer = self.scratch[self.scratch_used]
        if len(buffer) < bytes:
            buffer = List[UInt8](unsafe_uninit_length=bytes)
        self.scratch_used += 1
        return Int(buffer.unsafe_ptr())

    def writable(mut self, column: Int, physical: Int) -> Vector:
        """Column `column` as a flat vector over this chunk's cache for it,
        sized for STANDARD_VECTOR_SIZE values of `physical`
        (Vector::ResetFromCache), every row valid. A kernel writes the
        values at `data` and clears validity bits through `validity_bits`
        after `ensure_validity`."""
        while len(self.caches) <= column:
            self.caches.append(List[UInt8]())
            self.validity_caches.append(List[UInt8]())
        var bytes = STANDARD_VECTOR_SIZE * physical_size(physical)
        if len(self.caches[column]) < bytes:
            self.caches[column] = List[UInt8](unsafe_uninit_length=bytes)
        var result = Vector()
        result.physical = physical
        result.data = Int(self.caches[column].unsafe_ptr())
        self.data[column] = result
        return result

    def validity_bits(mut self, column: Int) -> Int:
        """The validity bitmap of column `column`'s cache, every bit set,
        now the column's validity (FlatVector::Validity(...).Initialize)."""
        var bytes = STANDARD_VECTOR_SIZE // 8
        if len(self.validity_caches[column]) < bytes:
            self.validity_caches[column] = List[UInt8](length=bytes, fill=255)
        else:
            var at = self.validity_caches[column].unsafe_ptr()
            for b in range(bytes):
                at[b] = 255
        var address = Int(self.validity_caches[column].unsafe_ptr())
        self.data[column].validity = ValidityMask(address, 0)
        return address

    def reference(mut self, column: Int, vector: Vector):
        """Column `column` reads `vector` (Vector::Reference)."""
        self.data[column] = vector

    def slice(
        mut self, input: DataChunk, sel: SelectionVector, count: Int
    ) raises:
        """DataChunk::Slice(other, sel, count): every vector of `input` as
        a dictionary vector through `sel`, which must live as long as this
        chunk. A dictionary is not stacked on a dictionary: its selection
        is merged with `sel` (SelectionVector::Slice), once per distinct
        selection (the merge cache)."""
        self.set_columns(len(input.data))
        var merged_from = List[Int]()
        var merged_to = List[Int]()
        for c in range(len(input.data)):
            var v = input.data[c]
            if v.vector_type == CONSTANT_VECTOR:
                self.data[c] = v
            elif v.vector_type == FLAT_VECTOR:
                v.vector_type = DICTIONARY_VECTOR
                v.sel = sel
                v.dictionary_size = input.count
                self.data[c] = v
            elif v.vector_type == DICTIONARY_VECTOR:
                var found = -1
                for m in range(len(merged_from)):
                    if merged_from[m] == v.sel.data:
                        found = merged_to[m]
                if found < 0:
                    var target = self.new_selection()
                    for i in range(count):
                        target.set_index(i, v.sel.get_index(sel.get_index(i)))
                    merged_from.append(v.sel.data)
                    merged_to.append(target.data)
                    found = target.data
                v.sel = SelectionVector(found)
                self.data[c] = v
            else:
                # A sequence is flattened at the selected rows.
                var out = self.writable(c, PHYSICAL_INT64)
                var values = Pointer[Int64, MutAnyOrigin](
                    unsafe_from_address=out.data
                )
                for i in range(count):
                    values[unsafe_offset=i] = v.sequence_start + Int64(
                        sel.get_index(i)
                    ) * v.sequence_increment
        self.count = count

    def unified(
        mut self, column: Int, arena: ColumnArena
    ) raises -> UnifiedVectorFormat:
        """Vector::ToUnifiedFormat of column `column`. Flat, constant and
        dictionary vectors convert for free; a sequence is flattened, and
        an encoded vector (Arrow strings, packed booleans) is decoded here
        into this chunk's scratch, at the rows it reads only."""
        var v = self.data[column]
        if v.vector_type == SEQUENCE_VECTOR:
            var data = self._new_scratch(8 * STANDARD_VECTOR_SIZE)
            var values = Pointer[Int64, MutAnyOrigin](unsafe_from_address=data)
            for i in range(self.count):
                values[unsafe_offset=i] = (
                    v.sequence_start + Int64(i) * v.sequence_increment
                )
            return UnifiedVectorFormat(
                SelectionVector(), data, ValidityMask(), PHYSICAL_INT64
            )
        if v.encoded():
            return self._decoded(v, arena)
        if v.vector_type == CONSTANT_VECTOR:
            return UnifiedVectorFormat(
                zero_selection(), v.data, v.validity, v.physical
            )
        if v.vector_type == DICTIONARY_VECTOR:
            return UnifiedVectorFormat(v.sel, v.data, v.validity, v.physical)
        return UnifiedVectorFormat(
            SelectionVector(), v.data, v.validity, v.physical
        )

    def _decoded(
        mut self, v: Vector, arena: ColumnArena
    ) raises -> UnifiedVectorFormat:
        """The rows an encoded vector reads, decoded: string_t for strings,
        a byte a value for booleans. A constant decodes its one row; a flat
        vector keeps its rows (and validity); a dictionary is flattened at
        its selected rows, validity with them."""
        ref series = arena.columns[v.source]
        var rows = 1 if v.vector_type == CONSTANT_VECTOR else self.count
        var flattened = v.vector_type == DICTIONARY_VECTOR
        var validity = v.validity
        var valid_bits = 0
        if flattened and not v.validity.all_valid():
            valid_bits = self._new_scratch(STANDARD_VECTOR_SIZE // 8)
            var at = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=valid_bits)
            for b in range(STANDARD_VECTOR_SIZE // 8):
                at[unsafe_offset=b] = 255
            for i in range(rows):
                if not v.validity.row_is_valid_unsafe(v.sel.get_index(i)):
                    set_invalid(valid_bits, i)
            validity = ValidityMask(valid_bits, 0)
        if v.physical == PHYSICAL_VARCHAR:
            ref strings = series._data[StringColumn]
            var data = self._new_scratch(16 * STANDARD_VECTOR_SIZE)
            var out = Pointer[StringT, MutAnyOrigin](unsafe_from_address=data)
            for i in range(rows):
                var row = v.source_offset + (
                    v.sel.get_index(i) if flattened else i
                )
                if strings._valid(row):
                    var text = strings._get(row)
                    out[unsafe_offset=i] = StringT.make(
                        Int(text.unsafe_ptr()), text.byte_length()
                    )
                else:
                    out[unsafe_offset=i] = StringT(0, 0, 0)
            var sel = SelectionVector()
            if v.vector_type == CONSTANT_VECTOR:
                sel = zero_selection()
            return UnifiedVectorFormat(sel, data, validity, PHYSICAL_VARCHAR)
        if v.physical == PHYSICAL_BOOL:
            ref bools = series._data[BoolColumn]
            var data = self._new_scratch(STANDARD_VECTOR_SIZE)
            var out = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=data)
            for i in range(rows):
                var row = v.source_offset + (
                    v.sel.get_index(i) if flattened else i
                )
                out[unsafe_offset=i] = UInt8(1) if bools._get(row) else UInt8(0)
            var sel = SelectionVector()
            if v.vector_type == CONSTANT_VECTOR:
                sel = zero_selection()
            return UnifiedVectorFormat(sel, data, validity, PHYSICAL_BOOL)
        raise Error("vector: no unified format for this column type")


def _source_rows(v: Vector, count: Int) -> List[Int]:
    """The at-rest rows a vector over a source reads, in order."""
    var rows = List[Int](unsafe_uninit_length=count)
    var out = rows.unsafe_ptr()
    if v.vector_type == DICTIONARY_VECTOR:
        for i in range(count):
            out[unsafe_offset=i] = v.source_offset + v.sel.get_index(i)
    elif v.vector_type == CONSTANT_VECTOR:
        for i in range(count):
            out[unsafe_offset=i] = v.source_offset
    else:
        for i in range(count):
            out[unsafe_offset=i] = v.source_offset + i
    return rows^


def to_series(
    mut chunk: DataChunk,
    column: Int,
    arena: ColumnArena,
    var name: String,
    dtype: DataType,
) raises -> Series:
    """Column `column` of `chunk` as an at-rest column of `chunk.count`
    rows typed `dtype`: a flat window of a source column is that column's
    slice (no copy), a dictionary over one its rows gathered, and values a
    kernel wrote are copied out of the chunk's cache. For operators that
    do not read vectors yet."""
    var v = chunk.data[column]
    var count = chunk.count
    if v.source >= 0 and v.vector_type != SEQUENCE_VECTOR:
        ref series = arena.columns[v.source]
        var result: Series
        if v.vector_type == FLAT_VECTOR:
            result = series.slice(v.source_offset, count)
        else:
            result = series.take(_source_rows(v, count))
        return _typed(result.renamed(name^), dtype)
    var format = chunk.unified(column, arena)
    var result = _from_unified(format, count, name^)
    return _typed(result^, dtype)


def _typed(var series: Series, dtype: DataType) raises -> Series:
    if series.dtype() == dtype:
        return series^
    return series.with_dtype(dtype)


def _from_unified(
    format: UnifiedVectorFormat, count: Int, var name: String
) raises -> Series:
    """A new column of the `count` rows `format` reads."""
    var any_null = not format.validity.all_valid()
    var valid = List[Bool]()
    if any_null:
        valid = List[Bool](capacity=count)
        for i in range(count):
            valid.append(
                format.validity.row_is_valid_unsafe(format.sel.get_index(i))
            )
    if format.physical == PHYSICAL_VARCHAR:
        var builder = StringBuilder(count)
        for i in range(count):
            var at = format.sel.get_index(i)
            if any_null and not valid[i]:
                builder.append_null()
                continue
            var entry = format.data + 16 * at
            var s = Pointer[StringT, MutAnyOrigin](unsafe_from_address=entry)[]
            builder.append(
                StringSlice[ImmutAnyOrigin](
                    unsafe_from_utf8=Span[UInt8, ImmutAnyOrigin](
                        unsafe_ptr=Pointer[UInt8, ImmutAnyOrigin](
                            unsafe_from_address=string_data(entry)
                        ),
                        length=Int(s.length),
                    )
                )
            )
        return Series(name^, builder^.finish())
    if format.physical == PHYSICAL_BOOL:
        var values = List[Bool](capacity=count)
        var data = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=format.data)
        for i in range(count):
            values.append(data[unsafe_offset=format.sel.get_index(i)] != 0)
        if any_null:
            return Series(name^, BoolColumn(values, valid))
        return Series(name^, BoolColumn(values))
    comptime for k in range(len(_NUMERIC_STORAGE)):
        comptime D = _NUMERIC_STORAGE[k]
        if format.physical == physical_of[D]():
            var values = List[Scalar[D]](unsafe_uninit_length=count)
            var out = values.unsafe_ptr()
            var data = format.values[D]()
            if format.sel.is_set():
                for i in range(count):
                    out[unsafe_offset=i] = data[
                        unsafe_offset=format.sel.get_index_unsafe(i)
                    ]
            else:
                unsafe_memcpy(dest=out, src=data, count=count)
            if any_null:
                return Series(name^, Column[Scalar[D]](values^, valid))
            return Series(name^, Column[Scalar[D]](values^))
    raise Error("vector: no column for this physical type")


def from_frame_window(
    mut chunk: DataChunk,
    mut arena: ColumnArena,
    handles: List[Int],
    offset: Int,
    count: Int,
):
    """Fill `chunk` with flat vectors over rows [offset, offset + count) of
    the arena columns `handles` (a table scan's output chunk)."""
    chunk.set_columns(len(handles))
    for c in range(len(handles)):
        chunk.data[c] = flat_over(arena, handles[c], offset)
    chunk.count = count
