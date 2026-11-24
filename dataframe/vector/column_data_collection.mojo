"""Collect output chunks in owned columns without repeated concatenation.

Append fixed-width vectors as raw bytes when their physical and logical
types agree; otherwise retain values. Keep each chunk as a separate piece
and join once when producing a frame, avoiding quadratic copying as the
result grows. Batch collections can transfer independent pieces to output
in parallel.
"""
from std.atomic import Atomic
from std.memory import (
    Allocation,
    Layout,
    Pointer,
    alloc,
    dealloc,
    unsafe_memcpy,
)

from dataframe.frame import DataFrame, concat
from dataframe.parallel import (
    Job,
    Rounds,
    partitions,
    round_width,
    run_rounds,
    run_rounds_serially,
    worker_count,
)
from dataframe.series import ChunkMerge, Series
from dataframe.storage.column import Column
from dataframe.dtype import DataType
from dataframe.vector.data_chunk import (
    ColumnArena,
    DataChunk,
    _from_unified,
    _typed,
    to_series,
)
from dataframe.vector.sink import Sink
from dataframe.vector.types import (
    PHYSICAL_BOOL,
    PHYSICAL_OPAQUE,
    PHYSICAL_VARCHAR,
    CONSTANT_VECTOR,
    FLAT_VECTOR,
    SEQUENCE_VECTOR,
    UnifiedVectorFormat,
    physical_of,
    physical_size,
)

comptime RAW_STORAGE = [
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

comptime _COPY_NONE = 0
comptime _COPY_RAW = 1
comptime _COPY_VALUES = 2
# A window of a column the collection's rows already are, in order: kept
# as a reference to its buffers, not copied.
comptime _COPY_WINDOW = 3
# Rows that are all one stored constant column's value: kept as that
# constant, not written out.
comptime _COPY_CONSTANT = 4
# The fixed widths a selected copy moves as one typed value.
comptime _COPY_WIDTHS = [
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.uint128,
]
comptime _COPY_SIZES = [1, 2, 4, 8, 16]


@always_inline
def _copy_selected[
    D: DType
](address: Int, format: UnifiedVectorFormat, count: Int):
    """`count` values of type `D` read through `format`'s selection,
    written in order from `address`."""
    var src = Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=format.data)
    var dst = Pointer[Scalar[D], MutAnyOrigin](unsafe_from_address=address)
    for i in range(count):
        dst[unsafe_offset=i] = src[unsafe_offset=format.sel.get_index_unsafe(i)]


def append_raw(
    mut bytes: List[UInt8],
    mut bits: List[UInt8],
    height: Int,
    capacity: Int,
    format: UnifiedVectorFormat,
    count: Int,
    size: Int,
):
    """`count` fixed-width values of `size` bytes read through `format` appended
    to `bytes` after `height` rows, their validity into the packed bitmap `bits`
    (left empty while every row is valid). Storage is sized for `capacity` rows
    once, so appends never reallocate."""
    var start = len(bytes)
    if bytes.capacity() < start + size * count:
        bytes.reserve(
            max(
                start + size * count,
                size * capacity,
                2 * bytes.capacity(),
            )
        )
    var data = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=format.data)
    if not format.sel.is_set():
        bytes.extend(
            Span[UInt8, MutAnyOrigin](unsafe_ptr=data, length=size * count)
        )
    else:
        # Each selected value written in place at its width, not a list append
        # per row.
        bytes.resize(start + size * count, 0)
        var dest = Int(bytes.unsafe_ptr()) + start
        var copied = False
        comptime for k in range(len(_COPY_WIDTHS)):
            comptime D = _COPY_WIDTHS[k]
            comptime W = _COPY_SIZES[k]
            if size == W:
                _copy_selected[D](dest, format, count)
                copied = True
        if not copied:
            for i in range(count):
                unsafe_memcpy(
                    dest=Pointer[UInt8, MutAnyOrigin](
                        unsafe_from_address=dest + size * i
                    ),
                    src=data.unsafe_offset(
                        size * format.sel.get_index_unsafe(i)
                    ),
                    count=size,
                )
    if format.validity.all_valid() and len(bits) == 0:
        return
    # The validity bits.
    var needed = (max(capacity, height + count) + 7) // 8
    if len(bits) < (height + count + 7) // 8:
        # The rows before were all valid.
        bits.resize(needed, 255)
    var out = bits.unsafe_ptr()
    for i in range(count):
        var invalid = UInt8(
            not format.validity.row_is_valid(format.sel.get_index(i))
        )
        var r = height + i
        out[unsafe_offset=r >> 3] &= ~(invalid << UInt8(r & 7))


def raw_series(
    bytes: List[UInt8], bits: List[UInt8], physical: Int, template: Series
) raises -> Series:
    """Raw values of vector type `physical` as a column typed and named as
    `template`, with its field metadata."""
    var size = physical_size(physical)
    var n = len(bytes) // size if size > 0 else 0
    var result = template.copy()
    comptime for k in range(len(RAW_STORAGE)):
        comptime D = RAW_STORAGE[k]
        if physical == physical_of[D]():
            var values = List[Scalar[D]](unsafe_uninit_length=n)
            unsafe_memcpy(
                dest=values.unsafe_ptr().unsafe_bitcast[UInt8](),
                src=bytes.unsafe_ptr(),
                count=size * n,
            )
            var validity = bits.copy()
            if len(validity) > 0:
                validity.resize((n + 7) // 8, 255)
            var column = Column[Scalar[D]](values=values^, bits=validity^)
            result = Series(template.name(), column^)
    var dtype = template.dtype()
    if result.dtype() != dtype:
        result = result.with_dtype(dtype)
    # The values are the template's field, Arrow metadata included, so they
    # concatenate with other pieces of it.
    result._field_metadata = template._field_metadata
    return result^


struct ColumnDataCollection(Movable):
    """Chunks of one schema copied into owned columns. A column is copied as raw bytes when
    its vectors are fixed-width and its type is their storage type (or
    the values were computed, typed afterwards), and otherwise appended
    as values; a column switches to values the first time a vector cannot
    be copied raw. Values are kept as one piece per chunk and joined
    once by `frame` (an
    append per chunk copied every row so far again: quadratic)."""

    var schema: DataFrame
    var modes: List[Int]
    var raw: List[List[UInt8]]
    var raw_bits: List[List[UInt8]]
    var raw_physical: List[Int]
    var values: List[List[Series]]
    # Per column in `_COPY_WINDOW`, the column its rows are a window of,
    # from row `starts`.
    var windows: List[Series]
    var starts: List[Int]
    var count: Int
    # The rows the collection is expected to take: raw storage is sized
    # for them once.
    var capacity: Int

    def __init__(out self, schema: DataFrame, capacity: Int = 0):
        self.schema = schema.copy()
        self.capacity = capacity
        var width = schema.width()
        self.modes = List[Int](length=width, fill=_COPY_NONE)
        self.raw = List[List[UInt8]](length=width, fill=List[UInt8]())
        self.raw_bits = List[List[UInt8]](length=width, fill=List[UInt8]())
        self.raw_physical = List[Int](length=width, fill=0)
        self.values = List[List[Series]](capacity=width)
        self.windows = List[Series](capacity=width)
        for c in range(width):
            self.values.append([schema._columns[c].copy()])
            self.windows.append(schema._columns[c].copy())
        self.starts = List[Int](length=width, fill=0)
        self.count = 0

    def append(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        """Copy `chunk`'s rows in, column by column."""
        var n = chunk.count
        if n == 0:
            return
        for c in range(len(self.modes)):
            var v = chunk.data[c]
            var mode = self.modes[c]
            var size = physical_size(v.physical)
            var dtype = self.schema._columns[c].dtype()
            # A flat vector over a stored column is a window of it. While
            # every chunk's vector continues the same window (consecutive
            # rows of one column, as a scan's chunks over a column that no
            # step changed are), the collected rows are that window, so
            # they are referenced rather than copied: columns are
            # immutable, and the reference keeps their buffers alive.
            if v.vector_type == CONSTANT_VECTOR and v.physical != (
                PHYSICAL_OPAQUE
            ):
                # A constant vector: while every chunk's is the same value,
                # the collected rows are that one value over more rows.
                var piece = self._constant_piece(chunk, c, arena, dtype)
                if piece:
                    if mode == _COPY_NONE and self.count == 0:
                        self.modes[c] = _COPY_CONSTANT
                        self.windows[c] = piece.take()
                        continue
                    if mode == _COPY_CONSTANT and piece.value()._same_constant(
                        self.windows[c]
                    ):
                        continue
            if mode == _COPY_CONSTANT:
                # The constant so far as a piece, then this chunk's.
                self.values[c] = [self._constant(c)]
                self.windows[c] = self.schema._columns[c].copy()
                self.modes[c] = _COPY_VALUES
                mode = _COPY_VALUES
            if v.vector_type == FLAT_VECTOR and v.source >= 0:
                ref column = arena.columns[v.source]
                if mode == _COPY_NONE and self.count == 0:
                    # A categorical's codes mean what its own dictionary
                    # says; one recoded by a step is copied as values.
                    if column.dtype() == dtype and not dtype.is_categorical():
                        self.modes[c] = _COPY_WINDOW
                        self.windows[c] = column.copy()
                        self.starts[c] = v.source_offset
                        continue
                elif (
                    mode == _COPY_WINDOW
                    and v.source_offset == self.starts[c] + self.count
                    and column._shares_buffers_with(self.windows[c])
                ):
                    continue
            if mode == _COPY_WINDOW:
                # The window so far as values, then this chunk's.
                self.values[c] = [self._window(c)]
                self.windows[c] = self.schema._columns[c].copy()
                self.modes[c] = _COPY_VALUES
                mode = _COPY_VALUES
            var raw = (
                v.vector_type != SEQUENCE_VECTOR
                and v.physical != PHYSICAL_BOOL
                and v.physical != PHYSICAL_VARCHAR
                and size > 0
                and (v.source < 0 or dtype.is_integer() or dtype.is_float())
            )
            if raw and (
                (mode == _COPY_NONE and self.count == 0)
                or (mode == _COPY_RAW and self.raw_physical[c] == v.physical)
            ):
                if mode == _COPY_NONE:
                    self.modes[c] = _COPY_RAW
                    self.raw_physical[c] = v.physical
                append_raw(
                    self.raw[c],
                    self.raw_bits[c],
                    self.count,
                    self.capacity,
                    chunk.unified(c, arena),
                    n,
                    size,
                )
                continue
            if mode == _COPY_RAW:
                self.values[c] = [self._raw_column(c)]
                self.raw[c] = List[UInt8]()
                self.raw_bits[c] = List[UInt8]()
            self.modes[c] = _COPY_VALUES
            self.values[c].append(
                to_series(
                    chunk, c, arena, self.schema._columns[c].name(), dtype
                )
            )
        self.count += n

    def append_frame(mut self, frame: DataFrame) raises:
        """`frame`'s columns (the schema's, in its order) kept as they are,
        each one piece of its column: rows an operator already holds as
        columns, such as a join's output for a morsel, are not cut into
        chunks and copied in again."""
        var n = frame.height()
        if n == 0:
            return
        for c in range(len(self.modes)):
            var mode = self.modes[c]
            var piece = frame._columns[c].renamed(
                self.schema._columns[c].name()
            )
            if self.count == 0:
                self.modes[c] = _COPY_VALUES
                self.values[c] = [piece^]
                continue
            if mode == _COPY_RAW:
                self.values[c] = [self._raw_column(c)]
                self.raw[c] = List[UInt8]()
                self.raw_bits[c] = List[UInt8]()
            elif mode == _COPY_WINDOW:
                self.values[c] = [self._window(c)]
                self.windows[c] = self.schema._columns[c].copy()
            elif mode == _COPY_CONSTANT:
                self.values[c] = [self._constant(c)]
                self.windows[c] = self.schema._columns[c].copy()
            self.modes[c] = _COPY_VALUES
            self.values[c].append(piece^)
        self.count += n

    def _window(self, c: Int) raises -> Series:
        """Column `c`'s rows so far, a window of the column they are in."""
        return (
            self.windows[c]
            .slice(self.starts[c], self.count)
            .renamed(self.schema._columns[c].name())
        )

    def _constant_piece(
        self,
        mut chunk: DataChunk,
        c: Int,
        mut arena: ColumnArena,
        dtype: DataType,
    ) raises -> Optional[Series]:
        """Column `c` of `chunk`, a constant vector, as a one-row constant
        column of the schema type; None for a type kept otherwise."""
        var v = chunk.data[c]
        if v.source >= 0:
            ref column = arena.columns[v.source]
            if column.is_constant() and column.dtype() == dtype:
                return column._constant_resized(1)
            return None
        if dtype.is_categorical() or dtype.is_nested():
            return None
        # The value alone: one row read through the vector's format.
        var one = _typed(
            _from_unified(
                chunk.unified(c, arena), 1, self.schema._columns[c].name()
            ),
            dtype,
        )._broadcast(1)
        if not one.is_constant() or one.dtype() != dtype:
            return None
        return one^

    def _constant(self, c: Int) -> Series:
        """Column `c`'s rows so far, all its one constant value."""
        return (
            self.windows[c]
            ._constant_resized(self.count)
            .renamed(self.schema._columns[c].name())
        )

    def _raw_column(self, c: Int) raises -> Series:
        return raw_series(
            self.raw[c],
            self.raw_bits[c],
            self.raw_physical[c],
            self.schema._columns[c],
        )

    def frame(mut self, merge: Bool = True) raises -> DataFrame:
        """The rows appended, as a frame of the schema's columns. Without
        `merge`, a column copied out in several pieces stays chunked
        (`chunked_columns` names them), for its merge to run apart."""
        var columns = List[Series](capacity=len(self.modes))
        for c in range(len(self.modes)):
            if self.modes[c] == _COPY_RAW:
                columns.append(self._raw_column(c))
            elif self.modes[c] == _COPY_WINDOW:
                columns.append(self._window(c))
            elif self.modes[c] == _COPY_CONSTANT:
                columns.append(self._constant(c))
            else:
                ref pieces = self.values[c]
                if len(pieces) == 1:
                    columns.append(pieces[0].copy())
                elif self.schema._columns[c].dtype().is_categorical():
                    # Pieces may each carry their own dictionary: they
                    # are moved onto the union of them as they join.
                    var frames = List[DataFrame](capacity=len(pieces))
                    for piece in pieces:
                        frames.append(DataFrame([piece.copy()]))
                    columns.append(concat(frames)._columns[0].rechunk())
                elif merge:
                    columns.append(Series._from_chunks(pieces).rechunk())
                else:
                    columns.append(Series._from_chunks(pieces))
        return DataFrame(columns^, height=self.count)

    def release(mut self):
        """Drop the rows held: the frame made from them holds them now.
        `count` stays."""
        self.raw = List[List[UInt8]]()
        self.raw_bits = List[List[UInt8]]()
        self.values = List[List[Series]]()
        self.windows = List[Series]()

    def chunked_columns(self) -> List[Int]:
        """The columns `frame(merge=False)` leaves chunked."""
        var chunked = List[Int]()
        for c in range(len(self.modes)):
            if (
                self.modes[c] != _COPY_RAW
                and self.modes[c] != _COPY_WINDOW
                and self.modes[c] != _COPY_CONSTANT
                and len(self.values[c]) > 1
                and not self.schema._columns[c].dtype().is_categorical()
            ):
                chunked.append(c)
        return chunked^


struct LimitBound(ImplicitlyCopyable, Movable):
    """The last morsel a limited pipeline still needs, shared by every
    worker by address (0: no limit). Workers take morsels in increasing
    number, so once one worker holds the limit's rows from morsels up to
    `b`, the first rows in source order are all in morsels up to `b`:
    no later morsel can hold one, and no worker needs to read it."""

    var address: Int

    def __init__(out self, address: Int):
        self.address = address

    @staticmethod
    def new() -> LimitBound:
        var pointer = alloc(Layout[Atomic[Int64]](length=1)).unsafe_leak()
        pointer.unsafe_write(Atomic[Int64](Int64.MAX))
        return LimitBound(Int(pointer))

    @staticmethod
    def none() -> LimitBound:
        return LimitBound(0)

    def free(self):
        if self.address == 0:
            return
        dealloc(
            Allocation[Atomic[Int64]](
                unsafe_owned_ptr=Pointer[Atomic[Int64], MutUntrackedOrigin](
                    unsafe_from_address=self.address
                ),
                layout=Layout[Atomic[Int64]](length=1),
            )
        )

    @always_inline
    def _cell(self) -> ref[MutAnyOrigin] Atomic[Int64]:
        return Pointer[Atomic[Int64], MutAnyOrigin](
            unsafe_from_address=self.address
        )[]

    def last(self) -> Int:
        """The last morsel still needed (`Int64.MAX` while every morsel
        may be)."""
        if self.address == 0:
            return Int(Int64.MAX)
        return Int(self._cell().load())

    def lower(self, morsel: Int):
        """No morsel after `morsel` is needed."""
        if self.address == 0:
            return
        var current = self._cell().load()
        while Int64(morsel) < current:
            if self._cell().compare_exchange(current, Int64(morsel)):
                return


struct MaterializeSink(Sink):
    """One worker's materialize sink state: a collection per batch index, the
    morsel the rows come from, so the finish puts every worker's rows back in
    source order.

    With a `limit`, only the first `limit` rows in source order are wanted.
    A worker takes morsels in increasing number, so its own rows are in
    source order too, and once it holds `limit` rows none it would sink
    later can be among the first `limit`: it keeps at most that many,
    says it is `full`, and lowers the shared `bound` to its morsel so no
    worker reads past it."""

    var schema: DataFrame
    var capacity: Int
    var batches: List[Int]
    var collections: List[ColumnDataCollection]
    var batch: Int
    var limit: Int
    var kept: Int
    var bound: LimitBound

    def __init__(
        out self,
        schema: DataFrame,
        capacity: Int = 0,
        limit: Int = -1,
        bound: LimitBound = LimitBound.none(),
    ):
        self.schema = schema.copy()
        self.capacity = capacity
        self.batches = List[Int]()
        self.collections = List[ColumnDataCollection]()
        self.batch = -1
        self.limit = limit
        self.kept = 0
        self.bound = bound

    def begin(mut self, morsel: Int):
        self.batch = morsel

    def full(self) -> Bool:
        """Whether no row sunk from here on can be among the first
        `limit`."""
        return self.limit >= 0 and self.kept >= self.limit

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        if chunk.count == 0 or self.full():
            return
        var count = chunk.count
        if self.limit >= 0:
            # The chunk's first rows, up to the limit.
            chunk.count = min(count, self.limit - self.kept)
            self.kept += chunk.count
        var last = len(self.batches) - 1
        if last < 0 or self.batches[last] != self.batch:
            self.batches.append(self.batch)
            self.collections.append(
                ColumnDataCollection(self.schema, self.capacity)
            )
            last += 1
        self.collections[last].append(chunk, arena)
        chunk.count = count
        if self.full():
            self.bound.lower(self.batch)

    def sink_frame(mut self, frame: DataFrame) raises:
        """A morsel's rows already held as whole columns (`frame`, of the
        schema's columns), kept as they are rather than copied in a chunk
        at a time."""
        if frame.height() == 0 or self.full():
            return
        var rows = frame.height()
        if self.limit >= 0:
            rows = min(rows, self.limit - self.kept)
            self.kept += rows
        var last = len(self.batches) - 1
        if last < 0 or self.batches[last] != self.batch:
            self.batches.append(self.batch)
            self.collections.append(
                ColumnDataCollection(self.schema, self.capacity)
            )
            last += 1
        if rows < frame.height():
            self.collections[last].append_frame(frame.slice(0, rows))
        else:
            self.collections[last].append_frame(frame)
        if self.full():
            self.bound.lower(self.batch)

    def release(mut self):
        pass

    def combine(var self, mut combined: List[MaterializeSink]) raises:
        """The worker's batches into the global state, every worker's. A morsel
        goes to one worker, so no batch is on two workers and merging needs no
        row-level work."""
        combined.append(self^)


struct _FrameJob(Job):
    """One batch's collection turned into its frame, on a worker: each
    batch's columns are copied out of its collection in parallel,
    avoiding a serial pass over every result row."""

    var collection: Int
    var frame: DataFrame
    # The frame's columns still in pieces, merged by the next round.
    var chunked: List[Int]

    def __init__(out self, collection: Int, schema: DataFrame):
        self.collection = collection
        self.frame = schema.copy()
        self.chunked = List[Int]()

    def run(mut self) raises:
        ref collection = Pointer[ColumnDataCollection, MutAnyOrigin](
            unsafe_from_address=self.collection
        )[]
        self.frame = collection.frame(merge=False)
        self.chunked = collection.chunked_columns()
        # Its rows now live in the frame: what the collection held goes
        # here, on this task, rather than serially once every task is
        # done.
        collection.release()


struct MaterializeFinish(Movable, Rounds):
    """Every worker's batches as one frame, in batch index order, one
    chunk each, as a round of tasks (`next_round`, `round_task`): each
    batch's columns copied out of its collection by the task that takes
    it, avoiding a serial pass over every result row. The workers'
    batches are each in the order the worker took them, so a merge by
    batch index over the workers orders all. `frame` is the result once
    `next_round` returns 0. With a `limit`, the batches stop once they
    hold that many rows: later batches hold none of the first rows.

    A column a batch copied out in several pieces is made one chunk by a
    second round: a fixed-width number column by ranges of its rows, each
    a piece of the round, any other column whole. The pieces of every
    batch share the round, so one large batch is merged on every thread
    rather than by the one task that copied it.

    A third round stacks the batches: each task takes a share of the
    columns and makes each one's batches its chunks. A filter that keeps
    few rows leaves many small batches, and stacking them all column after
    column on one thread would be the finish's longest step."""

    var sinks: List[MaterializeSink]
    var schema: DataFrame
    var jobs: List[_FrameJob]
    var width: Int
    var ran: Bool
    var merging: Bool
    # The second round: by merged column, its batch and column and its
    # merge by row ranges (none: rechunked whole); its pieces (merged
    # column, first row, end row; -1 for the whole column).
    var merged: List[Tuple[Int, Int]]
    var merges: List[Optional[ChunkMerge]]
    var pieces: List[Tuple[Int, Int, Int]]
    # The third round: each column's batches stacked, by column.
    var stacking: Bool
    var stacked: List[Series]
    var done: Bool
    var frame: DataFrame
    var limit: Int

    def __init__(
        out self,
        var sinks: List[MaterializeSink],
        schema: DataFrame,
        limit: Int = -1,
    ):
        self.sinks = sinks^
        self.schema = schema.copy()
        self.jobs = List[_FrameJob]()
        self.width = 0
        self.ran = False
        self.merging = False
        self.merged = List[Tuple[Int, Int]]()
        self.merges = List[Optional[ChunkMerge]]()
        self.pieces = List[Tuple[Int, Int, Int]]()
        self.stacking = False
        self.stacked = List[Series]()
        self.done = False
        self.frame = schema.copy()
        self.limit = limit

    def next_round(mut self) raises -> Int:
        if self.done:
            return 0
        if not self.ran:
            self.ran = True
            var next = List[Int](length=len(self.sinks), fill=0)
            var rows = 0
            while self.limit < 0 or rows < self.limit:
                var best = -1
                var best_batch = 0
                for j in range(len(self.sinks)):
                    if next[j] < len(self.sinks[j].batches):
                        var batch = self.sinks[j].batches[next[j]]
                        if best < 0 or batch < best_batch:
                            best = j
                            best_batch = batch
                if best < 0:
                    break
                rows += self.sinks[best].collections[next[best]].count
                self.jobs.append(
                    _FrameJob(
                        Int(
                            Pointer(to=self.sinks[best].collections[next[best]])
                        ),
                        self.schema,
                    )
                )
                next[best] += 1
            self.width = round_width(len(self.jobs))
            if self.width > 0:
                return self.width
        if not self.merging:
            self.merging = True
            for j in range(len(self.jobs)):
                var chunked = self.jobs[j].chunked.copy()
                for c in chunked:
                    self._add_merge(j, c)
            if len(self.pieces) > 0:
                self.width = round_width(len(self.pieces))
                return self.width
        if not self.stacking:
            for k in range(len(self.merged)):
                if self.merges[k]:
                    var at = self.merged[k]
                    self.jobs[at[0]].frame._columns[at[1]] = (
                        self.merges[k].value().merged.copy()
                    )
            self.merges = List[Optional[ChunkMerge]]()
            self.pieces = List[Tuple[Int, Int, Int]]()
            self.stacking = True
            if len(self.jobs) > 1 and not self._categorical():
                # Each column's batches, one under another, as its chunks:
                # a task per share of the columns.
                var width = self.schema.width()
                self.stacked = (
                    List[Series](
                        length=width, fill=self.schema._columns[0].copy()
                    ) if width
                    > 0 else List[Series]()
                )
                self.width = round_width(width)
                if self.width > 0:
                    return self.width
        self.done = True
        if len(self.stacked) > 0:
            var height = 0
            for k in range(len(self.jobs)):
                height += self.jobs[k].frame.height()
            var columns = List[Series]()
            swap(columns, self.stacked)
            self.frame = DataFrame(columns^, height=height)
        elif len(self.jobs) == 1:
            self.frame = self.jobs[0].frame.copy()
        elif len(self.jobs) > 1:
            # Categorical columns may each hold a dictionary of their own:
            # `concat` moves them onto one.
            var parts = List[DataFrame](capacity=len(self.jobs))
            for k in range(len(self.jobs)):
                parts.append(self.jobs[k].frame.copy())
            var whole = concat(parts)
            for c in range(whole.width()):
                whole._columns[c] = whole._columns[c]._joined_windows()
            self.frame = whole^
        if self.limit >= 0 and self.frame.height() > self.limit:
            self.frame = self.frame.slice(0, self.limit)
        self.jobs = List[_FrameJob]()
        return 0

    def _add_merge(mut self, j: Int, c: Int) raises:
        """Column `c` of batch `j`'s frame as pieces of the merge round."""
        var k = len(self.merged)
        self.merged.append((j, c))
        ref column = self.jobs[j].frame._columns[c]
        var merge = ChunkMerge.of(column)
        if merge:
            var n = len(column)
            var bounds = partitions(n, worker_count(n), 64)
            for w in range(len(bounds) - 1):
                if bounds[w] < bounds[w + 1]:
                    self.pieces.append((k, bounds[w], bounds[w + 1]))
        else:
            self.pieces.append((k, 0, -1))
        self.merges.append(merge^)

    def _categorical(self) -> Bool:
        """Whether a column of the schema is categorical."""
        for column in self.schema._columns:
            if column.dtype().is_categorical():
                return True
        return False

    def round_task(mut self, task: Int) raises:
        if self.stacking:
            for c in range(task, len(self.stacked), self.width):
                var pieces = List[Series](capacity=len(self.jobs))
                for k in range(len(self.jobs)):
                    pieces.append(self.jobs[k].frame._columns[c].copy())
                # Batches whose columns are consecutive windows of one
                # stored column (a column no step changed) join back into
                # one window of it.
                self.stacked[c] = Series._from_chunks(pieces)._joined_windows()
            return
        if self.merging:
            for p in range(task, len(self.pieces), self.width):
                var piece = self.pieces[p]
                var k = piece[0]
                if self.merges[k]:
                    self.merges[k].value().copy_rows(piece[1], piece[2])
                else:
                    var at = self.merged[k]
                    ref frame = self.jobs[at[0]].frame
                    frame._columns[at[1]] = frame._columns[at[1]].rechunk()
            return
        var j = task
        while j < len(self.jobs):
            self.jobs[j].run()
            j += self.width


def materialize_finish(
    mut sinks: List[MaterializeSink], schema: DataFrame, serial: Bool = False
) raises -> DataFrame:
    """Every batch in batch index order, one chunk each
    (`MaterializeFinish`), its round run here: on this thread alone when
    `serial` (work that is one task's share of a parallel step)."""
    var owned = List[MaterializeSink]()
    swap(owned, sinks)
    var finish = MaterializeFinish(owned^, schema)
    if serial:
        run_rounds_serially(finish)
    else:
        run_rounds(finish)
    swap(sinks, finish.sinks)
    return finish.frame.copy()
