"""DuckDB 2.0's `TupleDataCollection` (common/types/row/
tuple_data_collection.cpp): rows of a fixed layout, appended from data
chunks and gathered back into them.

    var layout = TupleDataLayout(types)
    var rows = TupleDataCollection(layout)
    var append = TupleDataAppendState()
    rows.initialize_append(append)
    rows.append(append, chunk, arena)         # per chunk
    var scan = TupleDataScanState()
    rows.initialize_scan(scan)
    while rows.scan(scan, out_chunk, arena):  # per chunk of rows
        ...

Appending converts each column to its unified format (`DataChunk.
unified`; a nested column through its at-rest column, see
`_storage_format`), computes each row's heap size, makes room for the
rows (`TupleDataSegment.build`) and scatters the values. Gathering reads
rows by pointer, so the hash join and the aggregate table can gather any
rows they hold pointers to, not only whole chunks.

A collection is used by one thread at a time; parallel builds give each
thread its own and `combine` them, as DuckDB's sinks do.
"""
from dataframe.dtype import DataType
from dataframe.series import Series
from dataframe.storage.nested import ListColumn, StructColumn
from dataframe.vector.data_chunk import ColumnArena, DataChunk, flat_over
from dataframe.vector.tuple_data_allocator import TupleDataSegment
from dataframe.vector.tuple_data_layout import (
    TUPLE_DATA_BLOCK_SIZE,
    TupleDataLayout,
    physical_of_dtype,
)
from dataframe.vector.tuple_data_scatter_gather import (
    compute_heap_sizes,
    gather_column,
    scatter,
)
from dataframe.vector.tuple_data_states import (
    TupleDataAppendState,
    TupleDataChunkState,
    TupleDataScanState,
    TupleDataVectorFormat,
)
from dataframe.vector.types import (
    CONSTANT_VECTOR,
    DICTIONARY_VECTOR,
    FLAT_VECTOR,
    PHYSICAL_LIST,
    PHYSICAL_STRUCT,
    SelectionVector,
    UnifiedVectorFormat,
    ValidityMask,
)


struct TupleDataCollection(Movable):
    """TupleDataCollection: segments of rows of one layout."""

    var layout: TupleDataLayout
    var segments: List[TupleDataSegment]
    var _count: Int
    var _data_size: Int

    def __init__(out self, layout: TupleDataLayout) raises:
        if TUPLE_DATA_BLOCK_SIZE // layout.row_width() == 0:
            raise Error(
                "tuple data: row width exceeds the block size of "
                + String(TUPLE_DATA_BLOCK_SIZE)
            )
        self.layout = layout.copy()
        self.segments = List[TupleDataSegment]()
        self.segments.append(TupleDataSegment(self.layout))
        self._count = 0
        self._data_size = 0

    def count(self) -> Int:
        return self._count

    def chunk_count(self) -> Int:
        var total = 0
        for i in range(len(self.segments)):
            total += self.segments[i].chunk_count()
        return total

    def size_in_bytes(self) -> Int:
        return self._data_size

    def tuples_per_block(self) -> Int:
        return TUPLE_DATA_BLOCK_SIZE // self.layout.row_width()

    def column_types(self) -> List[DataType]:
        return self.layout.types()

    # -- Append ------------------------------------------------------------

    def initialize_append(
        self,
        mut state: TupleDataAppendState,
        column_ids: List[Int] = List[Int](),
    ) raises:
        """InitializeAppend: append `column_ids` (every column when empty);
        the others are left unwritten."""
        self.initialize_chunk_state(state.chunk_state, column_ids)

    def initialize_chunk_state(
        self, mut state: TupleDataChunkState, column_ids: List[Int]
    ) raises:
        """InitializeChunkState: a format per column (and per nested level)
        and the columns to read or write."""
        var types = self.layout.types()
        state.vector_data = List[TupleDataVectorFormat]()
        for t in types:
            state.vector_data.append(
                TupleDataVectorFormat(physical_of_dtype(t))
            )
        for c in range(len(types)):
            _initialize_format(state.vector_data, c, types[c])
        state.column_ids = List[Int]()
        if len(column_ids) == 0:
            for c in range(len(types)):
                state.column_ids.append(c)
        else:
            state.column_ids = column_ids.copy()

    def append(
        mut self,
        mut state: TupleDataAppendState,
        mut chunk: DataChunk,
        mut arena: ColumnArena,
        append_sel: SelectionVector = SelectionVector(),
        append_count: Int = -1,
    ) raises:
        """Append: rows `append_sel` picks of `chunk` (its first
        `append_count`, every row by default). `arena` holds the columns
        the chunk's vectors read."""
        var mark = arena.mark()
        self.to_unified_format(state.chunk_state, chunk, arena)
        self.append_unified(state.chunk_state, chunk, append_sel, append_count)
        # The nested levels added for this chunk are copied into the rows.
        arena.truncate(mark)

    def append_unified(
        mut self,
        mut state: TupleDataChunkState,
        chunk: DataChunk,
        append_sel: SelectionVector = SelectionVector(),
        append_count: Int = -1,
    ):
        """AppendUnified: append after `to_unified_format`."""
        var count = chunk.count if append_count < 0 else append_count
        if count == 0:
            return
        if not self.layout.all_constant():
            compute_heap_sizes(state, append_sel, count)
        self.build(state, 0, count)
        scatter(self.layout, state, append_sel, count)

    def to_unified_format(
        self,
        mut state: TupleDataChunkState,
        mut chunk: DataChunk,
        mut arena: ColumnArena,
    ) raises:
        """ToUnifiedFormat of each appended column, its nested levels
        too."""
        for k in range(len(state.column_ids)):
            var col = state.column_ids[k]
            var physical = state.vector_data[col].physical
            if physical == PHYSICAL_LIST or physical == PHYSICAL_STRUCT:
                var v = chunk.data[col]
                if v.source < 0:
                    raise Error("tuple data: a nested vector needs its column")
                var series = arena.columns[v.source].copy()
                var window: Series
                if v.vector_type == FLAT_VECTOR:
                    window = series.slice(v.source_offset, chunk.count)
                else:
                    # A dictionary or constant nested vector: its rows in
                    # order, so every level reads without a selection.
                    var picked = List[Int](capacity=chunk.count)
                    for i in range(chunk.count):
                        picked.append(v.source_offset + v.child_row(i))
                    window = series.take(picked)
                _storage_format(state, col, window, arena)
                continue
            var format = chunk.unified(col, arena)
            if format.physical != physical:
                raise Error(
                    "tuple data: column "
                    + String(col)
                    + " does not match the layout's type"
                )
            state.vector_data[col].unified = format
            state.vector_data[col].original_sel = format.sel

    def build(
        mut self,
        mut state: TupleDataChunkState,
        append_offset: Int,
        append_count: Int,
    ):
        """Build: room for the rows in the last segment, their pointers in
        `state`."""
        ref segment = self.segments[len(self.segments) - 1]
        var before = segment.size_in_bytes()
        segment.build(state, append_offset, append_count)
        self._data_size += segment.size_in_bytes() - before
        self._count += append_count

    # -- Combine -----------------------------------------------------------

    def combine(mut self, mut other: TupleDataCollection) raises:
        """Combine: take over `other`'s segments, which keep their blocks."""
        if other._count == 0:
            return
        var mine = self.layout.types()
        var theirs = other.layout.types()
        var same = len(mine) == len(theirs)
        for i in range(len(mine)):
            same = same and mine[i] == theirs[i]
        if not same:
            raise Error("tuple data: combining collections of other types")
        var taken = List[TupleDataSegment]()
        while len(other.segments) > 0:
            taken.append(other.segments.pop())
        while len(taken) > 0:
            var segment = taken.pop()
            self._count += segment.count
            self._data_size += segment.data_size
            self.segments.append(segment^)
        other.reset()

    def reset(mut self):
        """Reset: no rows, a fresh segment."""
        self._count = 0
        self._data_size = 0
        self.segments = List[TupleDataSegment]()
        self.segments.append(TupleDataSegment(self.layout))

    # -- Scan and gather ---------------------------------------------------

    def initialize_scan(
        self,
        mut state: TupleDataScanState,
        column_ids: List[Int] = List[Int](),
    ) raises:
        self.initialize_chunk_state(state.chunk_state, column_ids)
        state.segment_index = 0
        state.chunk_index = 0

    def scan(
        self,
        mut state: TupleDataScanState,
        mut result: DataChunk,
        mut arena: ColumnArena,
    ) raises -> Bool:
        """Scan: the next chunk's rows gathered into `result`; False when
        none remain. Nested columns land in `arena`."""
        while state.segment_index < len(self.segments):
            if (
                state.chunk_index
                < self.segments[state.segment_index].chunk_count()
            ):
                break
            state.segment_index += 1
            state.chunk_index = 0
        if state.segment_index >= len(self.segments):
            result.reset()
            return False
        var count = self.segments[
            state.segment_index
        ].initialize_chunk_state(state.chunk_state, state.chunk_index, False)
        state.chunk_index += 1
        result.reset()
        self.gather(
            state.chunk_state.rows(),
            SelectionVector(),
            count,
            state.chunk_state.column_ids,
            result,
            SelectionVector(),
            arena,
        )
        result.count = count
        return True

    def fetch_chunk(
        self, mut state: TupleDataChunkState, chunk_index: Int, init_heap: Bool
    ) -> Int:
        """FetchChunk: the row pointers of chunk `chunk_index` of the whole
        collection into `state`; its row count."""
        var at = chunk_index
        for s in range(len(self.segments)):
            ref segment = self.segments[s]
            if at < segment.chunk_count():
                return segment.initialize_chunk_state(state, at, init_heap)
            at -= segment.chunk_count()
        return 0

    def gather(
        self,
        rows: Int,
        scan_sel: SelectionVector,
        scan_count: Int,
        column_ids: List[Int],
        mut result: DataChunk,
        target_sel: SelectionVector,
        mut arena: ColumnArena,
    ) raises:
        """Gather: columns `column_ids` of the rows at `rows` (an array of
        row pointers) that `scan_sel` picks, into `result`'s vectors 0, 1,
        ... at the positions `target_sel` gives."""
        if result.column_count() < len(column_ids):
            result.set_columns(len(column_ids))
        for k in range(len(column_ids)):
            gather_column(
                self.layout,
                rows,
                scan_sel,
                scan_count,
                column_ids[k],
                result,
                k,
                target_sel,
                arena,
            )


def _initialize_format(
    mut formats: List[TupleDataVectorFormat], index: Int, dtype: DataType
) raises:
    """InitializeVectorFormat: formats for a struct's fields or a list's
    child, after the ones already made."""
    if dtype.is_struct():
        var fields = dtype.field_dtypes()
        var first = len(formats)
        var children = List[Int]()
        for i in range(len(fields)):
            formats.append(TupleDataVectorFormat(physical_of_dtype(fields[i])))
            children.append(first + i)
        formats[index].children = children.copy()
        for i in range(len(fields)):
            _initialize_format(formats, first + i, fields[i])
    elif dtype.is_list():
        var inner = dtype.inner()
        var child = len(formats)
        formats.append(TupleDataVectorFormat(physical_of_dtype(inner)))
        formats[index].children = [child]
        _initialize_format(formats, child, inner)


def _storage_format(
    mut state: TupleDataChunkState,
    f: Int,
    series: Series,
    mut arena: ColumnArena,
) raises:
    """The unified format of every level of a nested at-rest column (Vector
    ::ToUnifiedFormat of a list or struct vector and its children): a
    list's list_entry_t from its offsets and its child from the elements
    these rows use; a struct's fields; each leaf read as `flat_over` and
    `ColumnArena.values` read a column. `arena` keeps every level alive
    until the rows are written."""
    var column = series.rechunk() if series.is_chunked() else series.copy()
    var physical = state.vector_data[f].physical
    if physical == PHYSICAL_LIST:
        if not column._data.isa[ListColumn]():
            raise Error("tuple data: a list column was expected")
        _ = arena.add(column)
        ref list = column._data[ListColumn]
        var n = len(list)
        var entries = List[UInt64](length=2 * max(n, 1), fill=0)
        var low = list._start(0) if n > 0 else 0
        var high = list._end(n - 1) if n > 0 else 0
        for i in range(n):
            var start = list._start(i)
            entries[2 * i] = UInt64(start - low)
            entries[2 * i + 1] = UInt64(list._end(i) - start)
        var validity = ValidityMask()
        if len(list._bits[]) > 0:
            validity = ValidityMask(Int(list._bits[].unsafe_ptr()), list._offset)
        var child = list.child().slice(low, high - low)
        ref format = state.vector_data[f]
        format.entries = entries^
        format.unified = UnifiedVectorFormat(
            SelectionVector(),
            Int(format.entries.unsafe_ptr()),
            validity,
            PHYSICAL_LIST,
        )
        format.original_sel = SelectionVector()
        format.list_size = high - low
        var child_format = format.children[0]
        _storage_format(state, child_format, child, arena)
        return
    if physical == PHYSICAL_STRUCT:
        if not column._data.isa[StructColumn]():
            raise Error("tuple data: a struct column was expected")
        _ = arena.add(column)
        ref fields_of = column._data[StructColumn]
        var validity = ValidityMask()
        if len(fields_of._bits[]) > 0:
            validity = ValidityMask(
                Int(fields_of._bits[].unsafe_ptr()), fields_of._offset
            )
        var fields = fields_of._windowed_fields()
        state.vector_data[f].unified = UnifiedVectorFormat(
            SelectionVector(), 0, validity, PHYSICAL_STRUCT
        )
        state.vector_data[f].original_sel = SelectionVector()
        var children = state.vector_data[f].children.copy()
        for c in range(len(children)):
            _storage_format(state, children[c], fields[c], arena)
        return
    var handle = arena.add(column)
    var v = flat_over(arena, handle, 0)
    if v.physical != physical:
        raise Error("tuple data: a nested field does not match its type")
    var data = arena.values(handle) if v.encoded() else v.data
    state.vector_data[f].unified = UnifiedVectorFormat(
        SelectionVector(), data, v.validity, physical
    )
    state.vector_data[f].original_sel = SelectionVector()
