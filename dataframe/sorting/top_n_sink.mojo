"""PhysicalTopN's sink over a stream of chunks (physical_top_n.cpp).

Each thread's `TopNSink` encodes its chunks' key columns a vector at a
time and adds the rows that can enter to its `TopNHeap`, copying those
rows into its payload (`heap_data`, AddLargeHeap). Once the payload holds
more than ReduceThreshold rows it is cut down to the live entries
(`Reduce`). The threads share a `TopNBoundary`. `combine` merges each
thread's heap and payload into one global heap (`TopNHeap::Combine`); the
finish sorts that heap (`Finalize`) and gathers the payload in order.
"""
from dataframe.frame import DataFrame
from dataframe.dtype import DataType
from dataframe.series import Series
from dataframe.storage.string import StringBuilder, StringColumn
from dataframe.sorting.sort_key import (
    STANDARD_VECTOR_SIZE,
    KeyBlock,
    SortKeyLayout,
    encode_keys,
    key_prefixes,
    encode_selected,
)
from dataframe.sorting.algorithms import SortKeys
from dataframe.sorting.top_n import TopNBoundary, TopNHeap
from dataframe.table_filter import DynamicFilterData, set_dynamic_value
from dataframe.vector.sink import Sink
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    flat_over,
    gather_series,
    storage_physical,
    to_series,
)
from dataframe.vector.types import (
    SEQUENCE_VECTOR,
    SelectionVector,
    UnifiedVectorFormat,
)
from std.collections import Optional
from std.memory import ArcPointer


def _copied_strings(
    column: Series, rows: List[Int], name: String
) raises -> Series:
    """Rows `rows` of a string column as a fresh offsets column."""
    ref source = column._data[StringColumn]
    var built = StringBuilder(len(rows))
    for r in rows:
        if source._valid(r):
            built.append(source._get(r))
        else:
            built.append_null()
    return Series(name, built^.finish())


def _appendable(column: Series) raises -> Series:
    """A payload column in a layout that grows in place: strings as offsets
    and bytes, which append amortized, rather than views, whose append
    builds a new combined storage each time."""
    if column._data.isa[StringColumn]():
        ref strings = column._data[StringColumn]
        if strings._is_view_storage():
            return Series(column.name(), strings.to_large_utf8()).with_dtype(
                column.dtype()
            )
    return column.copy()


def _sink_rows[
    W: Int, VAR: Bool
](mut heap: TopNHeap, block: KeyBlock, base: Int, mut matches: List[Int]):
    """AddLargeHeap: rows that enter get the next payload positions, and
    their rows are recorded to be copied into the payload."""
    comptime K = SortKeys[W, VAR]
    var k = K(block.address())
    for i in range(block.count):
        var rec = k.get(i)
        if heap.should_add[W, VAR](rec):
            matches.append(Int(rec[K.R - 1]))
            rec[K.R - 1] = UInt64(base + len(matches) - 1)
            heap.add[W, VAR](rec)


struct TopNSink(Sink):
    """PhysicalTopN's sink over a stream of chunks, for one thread: the key
    is laid out from the schema alone (no statistics in a stream), its
    columns named `names`; `limit` rows are kept."""

    var layout: SortKeyLayout
    var names: List[String]
    var heap: TopNHeap
    var schema: DataFrame
    # heap_data: the payload rows copied so far, one growing column per
    # schema column (DataChunk::Append with ALLOW_RESIZE).
    var payload: List[Series]
    var payload_rows: Int
    var boundary: TopNBoundary
    # The scan's dynamic filter on the first sort column, set from the
    # boundary when this sink's key becomes it, and its comparison.
    var dynamic: Optional[ArcPointer[DynamicFilterData]]
    var dynamic_op: Int
    # Each key's column in the schema, and the physical type its vectors
    # are read as.
    var key_columns: List[Int]
    var key_physical: List[Int]

    def __init__(
        out self,
        schema: DataFrame,
        names: List[String],
        descending: List[Bool],
        nulls_last: List[Bool],
        limit: Int,
        boundary: TopNBoundary,
        dynamic: Optional[ArcPointer[DynamicFilterData]] = None,
        dynamic_op: Int = 0,
    ) raises:
        var keys = List[Series](capacity=len(names))
        for name in names:
            keys.append(schema.column(name))
        self.layout = SortKeyLayout(
            keys, descending, nulls_last, False, statistics=False
        )
        self.names = names.copy()
        self.heap = TopNHeap(limit, self.layout.record_words())
        self.schema = schema.clear()
        self.payload = List[Series](capacity=self.schema.width())
        for column in self.schema._columns:
            self.payload.append(_appendable(column))
        self.payload_rows = 0
        self.boundary = boundary
        self.dynamic = dynamic
        self.dynamic_op = dynamic_op
        self.key_columns = List[Int](capacity=len(names))
        self.key_physical = List[Int](capacity=len(names))
        for name in names:
            var c = self.schema._index(name)
            self.key_columns.append(c)
            self.key_physical.append(storage_physical(self.schema._columns[c]))

    def begin(mut self, morsel: Int):
        pass

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        """Sink (`sink_chunk`)."""
        self.sink_chunk(chunk, arena)

    def release(mut self):
        pass

    def combine(var self, mut combined: List[TopNSink]) raises:
        """Combine (physical_top_n.cpp:532): this thread's heap into the
        global heap, which the first thread's becomes."""
        if len(combined) == 0:
            combined.append(self^)
            return
        combined[0]._absorb(self^)

    def _absorb(mut self, var other: TopNSink) raises:
        if self.layout.variable:
            self._combine[1, True](other^)
        elif self.layout.words == 1:
            self._combine[1, False](other^)
        elif self.layout.words == 2:
            self._combine[2, False](other^)
        else:
            self._combine[3, False](other^)

    def _combine[W: Int, VAR: Bool](mut self, var other: TopNSink) raises:
        """TopNHeap::Combine: the other heap sorted (`Finalize`), its
        entries best first until one cannot enter, each entry's payload row
        appended to this payload, then Reduce."""
        comptime K = SortKeys[W, VAR]
        other.heap.finalize[W, VAR]()
        var k = other.heap.keys[W, VAR]()
        var rows = List[Int]()
        for i in range(other.heap.size):
            var rec = k.get(i)
            if not self.heap.entry_should_be_added[W, VAR](rec):
                break
            rows.append(Int(rec[K.R - 1]))
            rec[K.R - 1] = UInt64(self.payload_rows + len(rows) - 1)
            self.heap.add[W, VAR](rec)
        if len(rows) == 0:
            return
        self._append(other._payload().take(rows)._columns)
        self.payload_rows += len(rows)
        if self.payload_rows >= self.heap.reduce_threshold():
            self._reduce[W, VAR]()

    def sink(mut self, frame: DataFrame) raises:
        """Sink: the chunk's rows that enter, a vector at a time."""
        if self.heap.heap_size == 0 or frame.height() == 0:
            return
        var keys = List[Series](capacity=len(self.names))
        for name in self.names:
            keys.append(frame.column(name))
        self.layout.rebind(keys)
        if self.layout.variable:
            self._sink[1, True](frame)
        elif self.layout.words == 1:
            self._sink[1, False](frame)
        elif self.layout.words == 2:
            self._sink[2, False](frame)
        else:
            self._sink[3, False](frame)

    def sink_chunk(
        mut self, mut chunk: DataChunk, mut arena: ColumnArena
    ) raises:
        """PhysicalTopN::Sink over one chunk whose columns are the schema's:
        the key columns are read out of the chunk, and only the rows that
        enter the heap are copied into the payload (AddLargeHeap)."""
        if self.heap.heap_size == 0 or chunk.count == 0:
            return
        # The key columns are read where they are, through their
        # UnifiedVectorFormat (TopNHeap::Sink builds the sort key from the
        # chunk's vectors, create_sort_key.cpp): flat windows of a morsel's
        # columns, a filter's dictionary vectors and computed vectors
        # alike. A vector whose physical type is not the key's (a column
        # stored narrower than the schema says) is cast first.
        var formats = List[UnifiedVectorFormat](capacity=len(self.names))
        for k in range(len(self.names)):
            var c = self.key_columns[k]
            if chunk.data[c].physical != self.key_physical[k]:
                var cast = to_series(
                    chunk,
                    c,
                    arena,
                    self.names[k],
                    self.schema._columns[c].dtype(),
                )
                chunk.data[c] = flat_over(arena, arena.add(cast), 0)
            formats.append(chunk.unified(c, arena))
        self.layout.rebind_vectors(formats, chunk.count)
        var matches: List[Int]
        if self.layout.variable:
            matches = self._enter[1, True](0, chunk.count)
        elif self.layout.words == 1:
            matches = self._enter[1, False](0, chunk.count)
        elif self.layout.words == 2:
            matches = self._enter[2, False](0, chunk.count)
        else:
            matches = self._enter[3, False](0, chunk.count)
        if len(matches) > 0:
            var rows = List[UInt32](unsafe_uninit_length=len(matches))
            for i in range(len(matches)):
                rows[i] = UInt32(matches[i])
            var selection = SelectionVector(Int(rows.unsafe_ptr()))
            var columns = List[Series](capacity=self.schema.width())
            for c in range(self.schema.width()):
                ref v = chunk.data[c]
                if (
                    v.source >= 0
                    and v.vector_type != SEQUENCE_VECTOR
                    and self.schema._columns[c].dtype() == DataType.STRING
                    and arena.columns[v.source]._data.isa[StringColumn]()
                ):
                    # Strings are copied out as bytes into a fresh offsets
                    # column: a take would share the source's view blocks
                    # (reference counts every thread touches) only to be
                    # copied into the payload's layout afterwards.
                    var picked = List[Int](capacity=len(matches))
                    for i in range(len(matches)):
                        picked.append(v.source_offset + v.child_row(matches[i]))
                    columns.append(
                        _copied_strings(
                            arena.columns[v.source],
                            picked,
                            self.schema._columns[c].name(),
                        )
                    )
                    continue
                columns.append(
                    gather_series(
                        chunk,
                        c,
                        arena,
                        selection,
                        len(matches),
                        self.schema._columns[c].name(),
                        self.schema._columns[c].dtype(),
                    )
                )
            _ = rows^
            self._append(columns)
            self.payload_rows += len(matches)
        # Nothing entered: the heap's worst key is unchanged, so the shared
        # boundary has nothing new from this chunk.
        if len(matches) > 0:
            self._settle_any()

    def _settle_any(mut self) raises:
        if self.layout.variable:
            self._settle[1, True]()
        elif self.layout.words == 1:
            self._settle[1, False]()
        elif self.layout.words == 2:
            self._settle[2, False]()
        else:
            self._settle[3, False]()

    def _append(mut self, columns: List[Series]) raises:
        """The entering rows onto the payload (heap_data.Append in
        TopNHeap::AddLargeHeap, physical_top_n.cpp:276)."""
        for c in range(len(self.payload)):
            self.payload[c]._append_series(_appendable(columns[c]))

    def _enter[
        W: Int, VAR: Bool
    ](mut self, start: Int, n: Int) raises -> List[Int]:
        """The bound key's rows [start, start + n) that enter the heap, a
        vector at a time (their payload positions follow the payload so
        far); the rows, as the bound key's."""
        self.heap.view.refresh[W, VAR](self.boundary)
        var matches = List[Int]()
        var at = start
        var stop = start + n
        var prefixes = List[UInt64]()
        while at < stop:
            var end = min(stop, at + STANDARD_VECTOR_SIZE)
            var block: KeyBlock
            var entering = Optional[List[Int]]()
            if key_prefixes(self.layout, at, end, prefixes):
                entering = self.heap.entering[W, VAR](prefixes, end - at)
            if entering and len(entering.value()) == 0:
                at = end
                continue
            if entering and len(entering.value()) < end - at:
                block = encode_selected(self.layout, at, entering.value())
            else:
                block = encode_keys(self.layout, at, end)
            _sink_rows[W, VAR](self.heap, block, self.payload_rows, matches)
            _ = block^
            at = end
        return matches^

    def _settle[W: Int, VAR: Bool](mut self) raises:
        """After rows enter: the shared boundary and the scan's dynamic
        filter from it, and Reduce once the payload is large."""
        if self.heap.publish[W, VAR](self.boundary) and self.dynamic:
            _ = set_dynamic_value(
                self.dynamic.value(),
                self.dynamic_op,
                self._payload_value(self.heap.top_payload[W, VAR]()),
            )
        if self.payload_rows >= self.heap.reduce_threshold():
            self._reduce[W, VAR]()

    def _sink[W: Int, VAR: Bool](mut self, frame: DataFrame) raises:
        var matches = self._enter[W, VAR](0, frame.height())
        if len(matches) > 0:
            self._append(frame.take(matches)._columns)
            self.payload_rows += len(matches)
        # TopNBoundaryValue::UpdateValue: the boundary's first sort value
        # becomes the scan filter's constant.
        self._settle[W, VAR]()

    def _payload_value(self, row: Int) raises -> Series:
        """Payload row `row`'s first sort column, one row."""
        return self.payload[self.schema._index(self.names[0])].slice(row, 1)

    def _payload(self) raises -> DataFrame:
        return DataFrame(self.payload.copy())

    def _reduce[W: Int, VAR: Bool](mut self) raises:
        """Reduce (physical_top_n.cpp:431): once the payload passes
        ReduceThreshold, it is cut to the live entries, renumbered."""
        var live = self._payload().take(self.heap.payloads[W, VAR]())
        self.heap.renumber[W, VAR]()
        self.heap.compact_keys[W, VAR]()
        self.payload = List[Series](capacity=live.width())
        for column in live._columns:
            self.payload.append(_appendable(column))
        self.payload_rows = self.heap.size

    def finish(mut self) raises -> DataFrame:
        """Finalize, then the payload rows in key order."""
        if self.layout.variable:
            return self._finish[1, True]()
        if self.layout.words == 1:
            return self._finish[1, False]()
        if self.layout.words == 2:
            return self._finish[2, False]()
        return self._finish[3, False]()

    def _finish[W: Int, VAR: Bool](mut self) raises -> DataFrame:
        self.heap.finalize[W, VAR]()
        return self._payload().take(self.heap.payloads[W, VAR]())
