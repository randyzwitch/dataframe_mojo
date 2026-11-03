"""PhysicalTopN's sink over a stream of chunks (physical_top_n.cpp).

Each thread's `TopNSink` encodes its chunks' key columns a vector at a
time and adds the rows that can enter to its `TopNHeap`, copying those
rows into its payload (`heap_data`, AddLargeHeap). Once the payload holds
more than ReduceThreshold rows it is cut down to the live entries
(`Reduce`). The threads share a `TopNBoundary`. The finish sorts the heap
(`Finalize`) and gathers the payload in order; the caller combines the
threads' results.
"""
from dataframe.frame import DataFrame, concat
from dataframe.series import Series
from dataframe.sorting.sort_key import (
    STANDARD_VECTOR_SIZE,
    KeyBlock,
    SortKeyLayout,
    encode_keys,
    key_prefixes,
)
from dataframe.sorting.algorithms import SortKeys
from dataframe.sorting.top_n import TopNBoundary, TopNHeap
from dataframe.table_filter import DynamicFilterData, set_dynamic_value
from std.memory import ArcPointer


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


struct TopNSink(Movable):
    """PhysicalTopN's sink over a stream of chunks, for one thread: the key
    is laid out from the schema alone (no statistics in a stream), its
    columns named `names`; `limit` rows are kept."""

    var layout: SortKeyLayout
    var names: List[String]
    var heap: TopNHeap
    var schema: DataFrame
    # heap_data: the payload rows copied so far, in pieces.
    var parts: List[DataFrame]
    var payload_rows: Int
    var boundary: TopNBoundary
    # The scan's dynamic filter on the first sort column, set from the
    # boundary when this sink's key becomes it, and its comparison.
    var dynamic: Optional[ArcPointer[DynamicFilterData]]
    var dynamic_op: Int

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
        self.parts = List[DataFrame]()
        self.payload_rows = 0
        self.boundary = boundary
        self.dynamic = dynamic
        self.dynamic_op = dynamic_op

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

    def _sink[W: Int, VAR: Bool](mut self, frame: DataFrame) raises:
        self.heap.view.refresh[W, VAR](self.boundary)
        var matches = List[Int]()
        var at = 0
        var n = frame.height()
        var prefixes = List[UInt64]()
        while at < n:
            var end = min(n, at + STANDARD_VECTOR_SIZE)
            if key_prefixes(
                self.layout, at, end, prefixes
            ) and not self.heap.can_enter[W, VAR](prefixes, end - at):
                # No row of the vector can be in the result.
                at = end
                continue
            var block = encode_keys(self.layout, at, end)
            _sink_rows[W, VAR](self.heap, block, self.payload_rows, matches)
            _ = block^
            at = end
        if len(matches) > 0:
            self.parts.append(frame.take(matches))
            self.payload_rows += len(matches)
        if self.heap.publish[W, VAR](self.boundary) and self.dynamic:
            # TopNBoundaryValue::UpdateValue: the boundary's first sort
            # value becomes the scan filter's constant.
            _ = set_dynamic_value(
                self.dynamic.value(),
                self.dynamic_op,
                self._payload_value(self.heap.top_payload[W, VAR]()),
            )
        if self.payload_rows >= self.heap.reduce_threshold():
            self._reduce[W, VAR]()

    def _payload_value(self, row: Int) raises -> Series:
        """Payload row `row`'s first sort column, one row."""
        var at = row
        for part in self.parts:
            if at < part.height():
                return part.column(self.names[0]).slice(at, 1)
            at -= part.height()
        raise Error("top-N payload row out of range")

    def _payload(self) raises -> DataFrame:
        if len(self.parts) == 0:
            return self.schema.copy()
        if len(self.parts) == 1:
            return self.parts[0].copy()
        return concat(self.parts)

    def _reduce[W: Int, VAR: Bool](mut self) raises:
        """Reduce: the payload cut to the live entries, renumbered."""
        var live = self._payload().take(self.heap.payloads[W, VAR]())
        self.heap.renumber[W, VAR]()
        self.heap.compact_keys[W, VAR]()
        self.parts = [live^]
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
