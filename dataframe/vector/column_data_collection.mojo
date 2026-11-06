"""DuckDB 2.0's `ColumnDataCollection` append and `BatchedDataCollection`
(common/types/column/column_data_collection.cpp,
common/types/batched_data_collection.cpp), and the materialize sink built
on them (PhysicalBatchCollector's sink: one collection per batch index).

A collection copies every chunk it is given into storage it owns
(`ColumnDataCopy`): fixed-width values a vector reads through its
unified format are copied as raw bytes with their validity
(`TemplatedColumnDataCopy`, `ColumnDataCopyValidity`), and other values
are appended to an owned column. Nothing it holds points into the
worker's arena, so the arena's columns can go when the next morsel
starts.
"""
from std.memory import Pointer, unsafe_memcpy

from dataframe.frame import DataFrame, concat
from dataframe.series import Series
from dataframe.storage.column import Column
from dataframe.vector.data_chunk import ColumnArena, DataChunk, to_series
from dataframe.vector.sink import Sink
from dataframe.vector.types import (
    PHYSICAL_BOOL,
    PHYSICAL_VARCHAR,
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


def append_raw(
    mut bytes: List[UInt8],
    mut bits: List[UInt8],
    height: Int,
    capacity: Int,
    format: UnifiedVectorFormat,
    count: Int,
    size: Int,
):
    """`count` fixed-width values of `size` bytes read through `format`
    appended to `bytes` after `height` rows, their validity into the
    packed bitmap `bits` (left empty while every row is valid). Storage is
    sized for `capacity` rows once (`TemplatedColumnDataCopy`)."""
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
        for i in range(count):
            var at = size * format.sel.get_index_unsafe(i)
            bytes.extend(
                Span[UInt8, MutAnyOrigin](
                    unsafe_ptr=data.unsafe_offset(at), length=size
                )
            )
    if format.validity.all_valid() and len(bits) == 0:
        return
    # ColumnDataCopyValidity.
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
    `template`."""
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
    return result^


struct ColumnDataCollection(Movable):
    """Chunks of one schema copied into owned columns (DuckDB's
    `ColumnDataCollection::Append`). A column is copied as raw bytes when
    its vectors are fixed-width and its type is their storage type (or
    the values were computed, typed afterwards), and otherwise appended
    as values; a column switches to values the first time a vector cannot
    be copied raw."""

    var schema: DataFrame
    var modes: List[Int]
    var raw: List[List[UInt8]]
    var raw_bits: List[List[UInt8]]
    var raw_physical: List[Int]
    var values: List[Series]
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
        self.values = List[Series](capacity=width)
        for c in range(width):
            self.values.append(schema._columns[c].copy())
        self.count = 0

    def append(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        """Copy `chunk`'s rows in (`ColumnDataCopy` per column)."""
        var n = chunk.count
        if n == 0:
            return
        for c in range(len(self.modes)):
            var v = chunk.data[c]
            var mode = self.modes[c]
            var size = physical_size(v.physical)
            var dtype = self.schema._columns[c].dtype()
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
                self.values[c] = self._raw_column(c)
                self.raw[c] = List[UInt8]()
                self.raw_bits[c] = List[UInt8]()
            self.modes[c] = _COPY_VALUES
            self.values[c]._append_series(
                to_series(
                    chunk, c, arena, self.schema._columns[c].name(), dtype
                )
            )
        self.count += n

    def _raw_column(self, c: Int) raises -> Series:
        return raw_series(
            self.raw[c],
            self.raw_bits[c],
            self.raw_physical[c],
            self.schema._columns[c],
        )

    def frame(mut self) raises -> DataFrame:
        """The rows appended, as a frame of the schema's columns."""
        var columns = List[Series](capacity=len(self.modes))
        for c in range(len(self.modes)):
            if self.modes[c] == _COPY_RAW:
                columns.append(self._raw_column(c))
            else:
                columns.append(self.values[c].copy())
        return DataFrame(columns^, height=self.count)


struct MaterializeSink(Sink):
    """One worker's materialize sink state: a collection per batch index,
    the morsel the rows come from (BatchedDataCollection::Append), so the
    finish puts every worker's rows back in source order."""

    var schema: DataFrame
    var capacity: Int
    var batches: List[Int]
    var collections: List[ColumnDataCollection]
    var batch: Int

    def __init__(out self, schema: DataFrame, capacity: Int = 0):
        self.schema = schema.copy()
        self.capacity = capacity
        self.batches = List[Int]()
        self.collections = List[ColumnDataCollection]()
        self.batch = -1

    def begin(mut self, morsel: Int):
        self.batch = morsel

    def sink(mut self, mut chunk: DataChunk, mut arena: ColumnArena) raises:
        if chunk.count == 0:
            return
        var last = len(self.batches) - 1
        if last < 0 or self.batches[last] != self.batch:
            self.batches.append(self.batch)
            self.collections.append(
                ColumnDataCollection(self.schema, self.capacity)
            )
            last += 1
        self.collections[last].append(chunk, arena)

    def release(mut self):
        pass

    def combine(var self, mut combined: List[MaterializeSink]) raises:
        """The worker's batches into the global state, every worker's
        (BatchedDataCollection::Merge: no batch is on two workers)."""
        combined.append(self^)


def materialize_finish(
    mut sinks: List[MaterializeSink], schema: DataFrame
) raises -> DataFrame:
    """Every batch in batch index order, one chunk each (the workers'
    batches are each in the order the worker took them, so a merge by
    batch index over the workers orders all)."""
    var parts = List[DataFrame]()
    var next = List[Int](length=len(sinks), fill=0)
    while True:
        var best = -1
        var best_batch = 0
        for j in range(len(sinks)):
            if next[j] < len(sinks[j].batches):
                var batch = sinks[j].batches[next[j]]
                if best < 0 or batch < best_batch:
                    best = j
                    best_batch = batch
        if best < 0:
            break
        parts.append(sinks[best].collections[next[best]].frame())
        next[best] += 1
    if len(parts) == 0:
        return schema.copy()
    if len(parts) == 1:
        return parts[0].copy()
    return concat(parts)
