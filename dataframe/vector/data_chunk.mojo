"""Bridge execution vectors and retained column storage.

A chunk owns reusable value, selection and decoding buffers. Allocate them
per worker, then reset the chunk between calls instead of allocating for
every batch. A ColumnArena retains the columns backing borrowed vectors,
including source columns, join payloads and expression results.
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
    # Per column, its values decoded once when a kernel first reads them:
    # string_t for an Arrow string column, a byte a value for a packed
    # Boolean one; empty until decoding is needed.
    var decoded: List[List[UInt8]]
    # Per column, the address of values already decoded by their owner
    # (`add_decoded`), 0 for none.
    var shared: List[Int]

    def __init__(out self):
        self.columns = List[Series]()
        self.decoded = List[List[UInt8]]()
        self.shared = List[Int]()

    def add(mut self, series: Series) raises -> Int:
        """Hold `series` as one contiguous array; its handle."""
        if series.is_chunked():
            if series.n_chunks() == 1:
                self.columns.append(series.chunks()[0].copy())
            else:
                self.columns.append(series.rechunk())
        else:
            self.columns.append(series.copy())
        self.decoded.append(List[UInt8]())
        self.shared.append(0)
        return len(self.columns) - 1

    def add_decoded(mut self, series: Series, values: Int) raises -> Int:
        """Hold `series`, whose values decoded to their vector layout are
        at `values` (kept alive by their owner while the arena reads
        them; 0 for none): a join table's dictionary columns, decoded once
        for every worker reading them."""
        var handle = self.add(series)
        self.shared[handle] = values
        return handle

    def mark(self) -> Int:
        return len(self.columns)

    def truncate(mut self, mark: Int):
        while len(self.columns) > mark:
            _ = self.columns.pop()
            _ = self.decoded.pop()
            _ = self.shared.pop()

    def values(mut self, handle: Int) raises -> Int:
        """The address of column `handle`'s values decoded to their vector
        layout (string_t, or a byte a Boolean), decoded on first use."""
        if self.shared[handle] != 0:
            return self.shared[handle]
        if len(self.decoded[handle]) == 0 and len(self.columns[handle]) > 0:
            ref series = self.columns[handle]
            if series._data.isa[StringColumn]():
                self.decoded[handle] = _decode_strings(
                    series._data[StringColumn]
                )
            elif series._data.isa[BoolColumn]():
                ref bools = series._data[BoolColumn]
                var out = List[UInt8](unsafe_uninit_length=len(bools))
                for i in range(len(bools)):
                    out[i] = UInt8(1) if bools._get(i) else UInt8(0)
                self.decoded[handle] = out^
            else:
                raise Error("vector: no unified format for this column type")
        return Int(self.decoded[handle].unsafe_ptr())


def storage_physical(series: Series) -> Int:
    """The physical type a vector over `series` reads (its first chunk's
    storage), as `flat_over` wraps it."""
    var column = series.chunks()[0].copy() if series.is_chunked() else (
        series.copy()
    )
    comptime for k in range(len(_NUMERIC_STORAGE)):
        comptime D = _NUMERIC_STORAGE[k]
        if column._data.isa[Column[Scalar[D]]]():
            return physical_of[D]()
    if column._data.isa[BoolColumn]():
        return PHYSICAL_BOOL
    if column._data.isa[StringColumn]():
        return PHYSICAL_VARCHAR
    return PHYSICAL_OPAQUE


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


struct DataChunk(Copyable, Movable):
    """A set of vectors of equal length, at most
    STANDARD_VECTOR_SIZE rows.

    `caches` are the per-column value buffers (`VectorCache`) a step writes
    its results into; `selections` the selection buffers this chunk's
    dictionary vectors point at; `scratch` the buffers `unified` decodes encoded vectors
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
                at[unsafe_offset=b] = 255
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
                    values[unsafe_offset=i] = (
                        v.sequence_start
                        + Int64(sel.get_index(i)) * v.sequence_increment
                    )
        self.count = count

    def unified(
        mut self, column: Int, mut arena: ColumnArena
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
        mut self, v: Vector, mut arena: ColumnArena
    ) raises -> UnifiedVectorFormat:
        """An encoded vector read through its column's values decoded once
        (`ColumnArena.values`): its selection and validity as they are."""
        var size = 16 if v.physical == PHYSICAL_VARCHAR else 1
        var data = arena.values(v.source) + size * v.source_offset
        var sel = SelectionVector()
        if v.vector_type == CONSTANT_VECTOR:
            sel = zero_selection()
        elif v.vector_type == DICTIONARY_VECTOR:
            sel = v.sel
        return UnifiedVectorFormat(sel, data, v.validity, v.physical)


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
    mut arena: ColumnArena,
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


def gather_series(
    mut chunk: DataChunk,
    column: Int,
    mut arena: ColumnArena,
    rows: SelectionVector,
    count: Int,
    var name: String,
    dtype: DataType,
) raises -> Series:
    """The `count` rows of column `column` that `rows` selects (every row
    when unset), in order, as an at-rest column typed `dtype`."""
    if not rows.is_set() and count == chunk.count:
        return to_series(chunk, column, arena, name^, dtype)
    var v = chunk.data[column]
    if v.source >= 0 and v.vector_type != SEQUENCE_VECTOR:
        var picked = List[Int](unsafe_uninit_length=count)
        var out = picked.unsafe_ptr()
        for k in range(count):
            out[unsafe_offset=k] = v.source_offset + v.child_row(
                rows.get_index(k)
            )
        return _typed(
            arena.columns[v.source].take(picked).renamed(name^), dtype
        )
    var format = chunk.unified(column, arena)
    var composed = List[UInt32](unsafe_uninit_length=max(count, 1))
    var at = composed.unsafe_ptr()
    for k in range(count):
        at[unsafe_offset=k] = UInt32(format.sel.get_index(rows.get_index(k)))
    var through = UnifiedVectorFormat(
        SelectionVector(Int(at)), format.data, format.validity, format.physical
    )
    var result = _from_unified(through, count, name^)
    _ = composed^
    return _typed(result^, dtype)


def decode_values(series: Series) -> List[UInt8]:
    """A string or Boolean column's values in their vector layout (string_t,
    or a byte a value), as `ColumnArena.values` decodes them; empty for
    other columns."""
    if series._data.isa[StringColumn]():
        return _decode_strings(series._data[StringColumn])
    if series._data.isa[BoolColumn]():
        ref bools = series._data[BoolColumn]
        var out = List[UInt8](unsafe_uninit_length=len(bools))
        for i in range(len(bools)):
            out[i] = UInt8(1) if bools._get(i) else UInt8(0)
        return out^
    return List[UInt8]()


def _decode_strings(column: StringColumn) -> List[UInt8]:
    """Every row of an Arrow string column as a 16-byte string_t pointing
    into the column's own bytes (inline views are string_t already)."""
    var n = len(column)
    var out = List[UInt8](unsafe_uninit_length=16 * max(n, 1))
    var target = Pointer[StringT, MutAnyOrigin](
        unsafe_from_address=Int(out.unsafe_ptr())
    )
    # Validity read through the bitmap's address: a method call on the
    # column would copy the whole column struct for every row.
    var validity = ValidityMask()
    if column.null_count() > 0:
        validity = ValidityMask(
            Int(column._bits[].unsafe_ptr()), column._offset
        )
    if (
        column._is_view_storage()
        and not column._view_storage.value()._selects()
    ):
        ref storage = column._view_storage.value()
        var views = storage._views[].unsafe_ptr().unsafe_offset(column._offset)
        ref buffers = storage._buffers[]
        var bases = List[Int](capacity=len(buffers))
        for b in range(len(buffers)):
            bases.append(Int(buffers[b][].unsafe_ptr()))
        var base_at = bases.unsafe_ptr()
        for i in range(n):
            if not validity.row_is_valid(i):
                target[unsafe_offset=i] = StringT(0, 0, 0)
                continue
            var view = views[unsafe_offset=i].copy()
            if view.is_inline():
                target[unsafe_offset=i] = StringT(
                    view.length,
                    view.prefix,
                    UInt64(view.buffer_index) | (UInt64(view.offset) << 32),
                )
            else:
                var address = base_at[
                    unsafe_offset=Int(view.buffer_index)
                ] + Int(view.offset)
                target[unsafe_offset=i] = StringT(
                    view.length, view.prefix, UInt64(address)
                )
        _ = bases^
        return out^
    if not column._is_view_storage():
        var offsets = (
            column._offsets[].unsafe_ptr().unsafe_offset(column._offset)
        )
        var base = Int(column._base())
        for i in range(n):
            if not validity.row_is_valid(i):
                target[unsafe_offset=i] = StringT(0, 0, 0)
                continue
            var start = Int(offsets[unsafe_offset=i])
            var length = Int(offsets[unsafe_offset=i + 1]) - start
            target[unsafe_offset=i] = StringT.make(base + start, length)
        return out^
    for i in range(n):
        if not validity.row_is_valid(i):
            target[unsafe_offset=i] = StringT(0, 0, 0)
            continue
        var text = column._get(i)
        target[unsafe_offset=i] = StringT.make(
            Int(text.unsafe_ptr()), text.byte_length()
        )
    return out^
