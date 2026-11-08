"""DuckDB 2.0's `RadixPartitionedTupleData` (common/types/row/
partitioned_tuple_data.cpp, common/radix_partitioning.cpp): rows of one
layout in one TupleDataCollection per radix partition of a hash column.

    var rows = RadixPartitionedTupleData(layout, radix_bits, hash_column)
    var append = PartitionedTupleDataAppendState()
    rows.initialize_append_state(append)
    # per chunk, after `to_unified_format` and the hashes are written:
    rows.append_unified(append, chunk, append_sel, count)
    combined.combine(rows)                    # a thread's rows, at Combine
    var all = combined.get_unpartitioned()    # partition 0's rows first

A row's partition is `radix_bits` bits of its hash, just below the 16
salt bits (RadixPartitioning::ApplyMask). Appending computes every row's
partition, makes one selection that lists each partition's rows together
(BuildPartitionSel, a counting pass), makes room in each partition for
its rows (BuildBufferSpace), and scatters the whole chunk in one call. A
chunk whose rows all fall in one partition goes straight to it.

There is no buffer manager, so blocks never move and no pin states are
kept; `FlushAppendState` has nothing to unpin. Repartitioning to more bits
is not ported: with no memory limit, the join never goes external.
"""
from dataframe.vector.data_chunk import DataChunk
from dataframe.vector.tuple_data_collection import TupleDataCollection
from dataframe.vector.tuple_data_layout import TupleDataLayout
from dataframe.vector.tuple_data_scatter_gather import (
    compute_heap_sizes,
    scatter,
)
from dataframe.vector.tuple_data_states import (
    TupleDataChunkState,
)
from dataframe.vector.types import (
    STANDARD_VECTOR_SIZE,
    SelectionVector,
)

# RadixPartitioning::MAX_RADIX_BITS.
comptime MAX_RADIX_BITS = 12


@always_inline
def number_of_partitions(radix_bits: Int) -> Int:
    """RadixPartitioning::NumberOfPartitions."""
    return 1 << radix_bits


@always_inline
def radix_partition(hash: UInt64, radix_bits: Int) -> Int:
    """RadixPartitioning::ApplyMask: the `radix_bits` bits of `hash` below
    its 16 salt bits (Shift = 48 - radix_bits)."""
    var shift = UInt64(48 - radix_bits)
    var mask = (UInt64(1) << UInt64(radix_bits)) - 1
    return Int((hash >> shift) & mask)


struct PartitionedTupleDataAppendState(Movable):
    """PartitionedTupleDataAppendState: the chunk state shared by every
    partition, each row's partition, the selection grouping the rows by
    partition, and each partition's count and first place in it."""

    var chunk_state: TupleDataChunkState
    var partition_indices: List[Int32]
    var partition_sel: List[UInt32]
    var counts: List[Int]
    var offsets: List[Int]

    def __init__(out self):
        self.chunk_state = TupleDataChunkState()
        self.partition_indices = List[Int32](
            unsafe_uninit_length=STANDARD_VECTOR_SIZE
        )
        self.partition_sel = List[UInt32](
            unsafe_uninit_length=STANDARD_VECTOR_SIZE
        )
        self.counts = List[Int]()
        self.offsets = List[Int]()


struct RadixPartitionedTupleData(Movable):
    """RadixPartitionedTupleData: a TupleDataCollection per partition."""

    var layout: TupleDataLayout
    var radix_bits: Int
    var hash_col_idx: Int
    var partitions: List[TupleDataCollection]
    var _count: Int
    var _data_size: Int

    def __init__(
        out self, layout: TupleDataLayout, radix_bits: Int, hash_col_idx: Int
    ) raises:
        if radix_bits > MAX_RADIX_BITS:
            raise Error("radix partitioning: too many radix bits")
        self.layout = layout.copy()
        self.radix_bits = radix_bits
        self.hash_col_idx = hash_col_idx
        self.partitions = List[TupleDataCollection](
            capacity=number_of_partitions(radix_bits)
        )
        for _ in range(number_of_partitions(radix_bits)):
            self.partitions.append(TupleDataCollection(self.layout))
        self._count = 0
        self._data_size = 0

    def count(self) -> Int:
        return self._count

    def size_in_bytes(self) -> Int:
        return self._data_size

    def partition_count(self) -> Int:
        return len(self.partitions)

    def initialize_append_state(
        self, mut state: PartitionedTupleDataAppendState
    ) raises:
        """InitializeAppendStateInternal: one chunk state over every
        column, and a count and offset per partition."""
        self.partitions[0].initialize_chunk_state(
            state.chunk_state, List[Int]()
        )
        var n = len(self.partitions)
        state.counts = List[Int](length=n, fill=0)
        state.offsets = List[Int](length=n, fill=0)

    def append_unified(
        mut self,
        mut state: PartitionedTupleDataAppendState,
        chunk: DataChunk,
        append_sel: SelectionVector,
        append_count: Int,
    ):
        """AppendUnified: rows `append_sel` picks of `chunk` (already in
        `state.chunk_state`'s unified formats), each into its partition."""
        if append_count == 0:
            return
        # ComputePartitionIndices: from the hash column, through the
        # append selection.
        var indices = state.partition_indices.unsafe_ptr()
        var hashes = state.chunk_state.vector_data[self.hash_col_idx].unified
        var hash_data = hashes.values[DType.uint64]()
        for i in range(append_count):
            var at = hashes.sel.get_index(append_sel.get_index(i))
            indices[unsafe_offset=i] = Int32(
                radix_partition(hash_data[unsafe_offset=at], self.radix_bits)
            )
        # BuildPartitionSel: each partition's count, its first place, and
        # one selection listing its rows together, in their order.
        var parts = len(self.partitions)
        var counts = state.counts.unsafe_ptr()
        var offsets = state.offsets.unsafe_ptr()
        for p in range(parts):
            counts[unsafe_offset=p] = 0
        for i in range(append_count):
            counts[unsafe_offset=Int(indices[unsafe_offset=i])] += 1
        var single = Int(indices[unsafe_offset=0])
        if counts[unsafe_offset=single] == append_count:
            # Every row in one partition: appended to it as it stands.
            ref partition = self.partitions[single]
            var before = partition.size_in_bytes()
            partition.append_unified(
                state.chunk_state, chunk, append_sel, append_count
            )
            self._data_size += partition.size_in_bytes() - before
            self._count += append_count
            return
        var offset = 0
        for p in range(parts):
            offsets[unsafe_offset=p] = offset
            offset += counts[unsafe_offset=p]
        var sel = state.partition_sel.unsafe_ptr()
        for i in range(append_count):
            var p = Int(indices[unsafe_offset=i])
            var at = offsets[unsafe_offset=p]
            sel[unsafe_offset=at] = UInt32(append_sel.get_index(i))
            offsets[unsafe_offset=p] = at + 1
        var partition_sel = SelectionVector(Int(sel))
        # The heap sizes of the whole chunk, in partition order.
        if not self.layout.all_constant():
            compute_heap_sizes(state.chunk_state, partition_sel, append_count)
        # BuildBufferSpace: room for each partition's rows, their pointers
        # at its place in the chunk state.
        var start = 0
        for p in range(parts):
            var length = counts[unsafe_offset=p]
            if length == 0:
                continue
            ref partition = self.partitions[p]
            var before = partition.size_in_bytes()
            partition.build(state.chunk_state, start, length)
            self._data_size += partition.size_in_bytes() - before
            start += length
        # Everything scattered in one go.
        scatter(self.layout, state.chunk_state, partition_sel, append_count)
        self._count += append_count

    def flush_append_state(self, mut state: PartitionedTupleDataAppendState):
        """FlushAppendState: no pins to release here."""
        pass

    def combine(mut self, mut other: RadixPartitionedTupleData) raises:
        """Combine: `other`'s rows, partition by partition, into these."""
        if other._count == 0:
            return
        if len(other.partitions) != len(self.partitions):
            raise Error("radix partitioning: combining other partition counts")
        for p in range(len(self.partitions)):
            self.partitions[p].combine(other.partitions[p])
        self._count += other._count
        self._data_size += other._data_size
        other._count = 0
        other._data_size = 0

    def get_sizes_and_counts(self, mut sizes: List[Int], mut counts: List[Int]):
        """GetSizesAndCounts: each partition's bytes and rows, added."""
        for p in range(len(self.partitions)):
            sizes[p] += self.partitions[p].size_in_bytes()
            counts[p] += self.partitions[p].count()

    def get_unpartitioned(mut self) raises -> TupleDataCollection:
        """GetUnpartitioned: every partition's rows in one collection, in
        partition order; the partitions are left empty."""
        var result = TupleDataCollection(self.layout)
        for p in range(len(self.partitions)):
            result.combine(self.partitions[p])
        self._count = 0
        self._data_size = 0
        return result^

    def reset(mut self):
        for p in range(len(self.partitions)):
            self.partitions[p].reset()
        self._count = 0
        self._data_size = 0
