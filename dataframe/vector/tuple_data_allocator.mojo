"""DuckDB 2.0's `TupleDataAllocator` and `TupleDataSegment`
(common/types/row/tuple_data_allocator.cpp, tuple_data_segment.cpp): the
row and heap blocks of a collection, and the chunks and chunk parts that
say where each appended row went.

- A row block holds whole rows, TUPLE_DATA_BLOCK_SIZE bytes. A heap block
  holds the variable-sized data of rows, at least that large (larger when
  one row's heap is).
- A chunk is at most STANDARD_VECTOR_SIZE rows. It is made of parts; a
  part is a run of rows in one row block whose heap data is one run in
  one heap block.

DuckDB allocates blocks through its buffer manager, which can evict and
reload them; a reloaded heap block may sit at a new address, so rows'
heap pointers are recomputed on pin. We have no buffer manager: blocks
are plain allocations that live as long as the collection and never
move, so a part's heap pointer is fixed when it is built.

Each segment owns its allocator. DuckDB shares one allocator among the
segments of a collection, but a collection has at most one segment of its
own (made by its first build) until it combines another's, which keeps
its own allocator, so ownership is the same. Blocks are made when rows
first land, so an empty partition allocates nothing.
"""
from dataframe.vector.tuple_data_layout import (
    TUPLE_DATA_BLOCK_SIZE,
    TupleDataLayout,
)
from dataframe.vector.tuple_data_states import TupleDataChunkState
from dataframe.vector.types import STANDARD_VECTOR_SIZE

comptime _INVALID_INDEX = -1


struct TupleDataBlock(Copyable, Movable):
    """TupleDataBlock: an allocation, its capacity and the bytes used."""

    var data: List[UInt8]
    var capacity: Int
    var size: Int

    def __init__(out self, capacity: Int):
        self.data = List[UInt8](unsafe_uninit_length=capacity)
        self.capacity = capacity
        self.size = 0

    @always_inline
    def address(self) -> Int:
        return Int(self.data.unsafe_ptr())

    @always_inline
    def remaining_capacity(self) -> Int:
        return self.capacity - self.size

    @always_inline
    def remaining_rows(self, row_width: Int) -> Int:
        return self.remaining_capacity() // row_width


@fieldwise_init
struct TupleDataChunkPart(Copyable, ImplicitlyCopyable, Movable):
    """TupleDataChunkPart: `count` rows at `row_block_offset` of row block
    `row_block_index`, their heap data `total_heap_size` bytes at
    `heap_block_offset` of heap block `heap_block_index`, whose address is
    `base_heap_ptr`."""

    var row_block_index: Int
    var row_block_offset: Int
    var heap_block_index: Int
    var heap_block_offset: Int
    var base_heap_ptr: Int
    var total_heap_size: Int
    var count: Int

    @staticmethod
    def empty() -> TupleDataChunkPart:
        return TupleDataChunkPart(0, 0, _INVALID_INDEX, _INVALID_INDEX, 0, 0, 0)

    def set_heap_empty(mut self):
        self.heap_block_index = _INVALID_INDEX
        self.heap_block_offset = _INVALID_INDEX
        self.total_heap_size = 0
        self.base_heap_ptr = 0


struct ContinuousIdSet(Copyable, ImplicitlyCopyable, Movable):
    """ContinuousIdSet: the range [min, max] of the ids inserted."""

    var min_id: Int
    var max_id: Int

    def __init__(out self):
        self.min_id = _INVALID_INDEX
        self.max_id = _INVALID_INDEX

    def insert(mut self, id: Int):
        if self.empty():
            self.min_id = id
            self.max_id = id
        else:
            self.min_id = min(self.min_id, id)
            self.max_id = max(self.max_id, id)

    def contains(self, id: Int) -> Bool:
        if self.empty():
            return False
        return id >= self.min_id and id <= self.max_id

    @always_inline
    def empty(self) -> Bool:
        return self.min_id == _INVALID_INDEX

    @always_inline
    def start(self) -> Int:
        return self.min_id

    @always_inline
    def end(self) -> Int:
        return self.max_id + 1

    @always_inline
    def size(self) -> Int:
        return self.end() - self.start()

    def decrement_max(mut self):
        self.max_id -= 1


struct TupleDataChunk(Copyable, ImplicitlyCopyable, Movable):
    """TupleDataChunk: its parts and the blocks they use."""

    var part_ids: ContinuousIdSet
    var row_block_ids: ContinuousIdSet
    var heap_block_ids: ContinuousIdSet
    var count: Int

    def __init__(out self):
        self.part_ids = ContinuousIdSet()
        self.row_block_ids = ContinuousIdSet()
        self.heap_block_ids = ContinuousIdSet()
        self.count = 0


struct TupleDataAllocator(Copyable, Movable):
    """TupleDataAllocator: the layout and the row and heap blocks."""

    var layout: TupleDataLayout
    var row_blocks: List[TupleDataBlock]
    var heap_blocks: List[TupleDataBlock]
    var block_allocation_size: Int

    def __init__(out self, layout: TupleDataLayout):
        self.layout = layout.copy()
        self.row_blocks = List[TupleDataBlock]()
        self.heap_blocks = List[TupleDataBlock]()
        self.block_allocation_size = 0

    def create_row_block(mut self):
        self.row_blocks.append(TupleDataBlock(TUPLE_DATA_BLOCK_SIZE))
        self.block_allocation_size += TUPLE_DATA_BLOCK_SIZE

    def create_heap_block(mut self, size: Int):
        self.heap_blocks.append(TupleDataBlock(size))
        self.block_allocation_size += size

    @always_inline
    def row_pointer(self, part: TupleDataChunkPart) -> Int:
        """GetRowPointer: the first row of `part`."""
        return (
            self.row_blocks[part.row_block_index].address()
            + part.row_block_offset
        )


struct TupleDataSegment(Copyable, Movable):
    """TupleDataSegment: an allocator and the chunks built with it."""

    var allocator: TupleDataAllocator
    var chunks: List[TupleDataChunk]
    var chunk_parts: List[TupleDataChunkPart]
    var count: Int
    var data_size: Int

    def __init__(out self, layout: TupleDataLayout):
        self.allocator = TupleDataAllocator(layout)
        self.chunks = List[TupleDataChunk]()
        self.chunk_parts = List[TupleDataChunkPart]()
        self.count = 0
        self.data_size = 0

    def chunk_count(self) -> Int:
        return len(self.chunks)

    def size_in_bytes(self) -> Int:
        return self.data_size

    def _add_part(mut self, chunk_index: Int, part: TupleDataChunkPart):
        """TupleDataChunk::AddPart."""
        ref chunk = self.chunks[chunk_index]
        chunk.count += part.count
        chunk.row_block_ids.insert(part.row_block_index)
        if (
            not self.allocator.layout.all_constant()
            and part.total_heap_size > 0
        ):
            chunk.heap_block_ids.insert(part.heap_block_index)
        chunk.part_ids.insert(len(self.chunk_parts))
        self.chunk_parts.append(part)

    def _merge_last_chunk_part(mut self, chunk_index: Int):
        """TupleDataChunk::MergeLastChunkPart: the last two parts of a chunk
        become one when their rows (and heaps) are adjacent. (DuckDB
        compares the heap offset with `heap_block_index + total_heap_size`;
        the offset is meant, as written here.)"""
        ref chunk = self.chunks[chunk_index]
        if chunk.part_ids.size() < 2:
            return
        var last_id = chunk.part_ids.end() - 1
        var second = self.chunk_parts[last_id - 1]
        var last = self.chunk_parts[last_id]
        var row_width = self.allocator.layout.row_width()
        var rows_align = (
            last.row_block_index == second.row_block_index
            and last.row_block_offset
            == second.row_block_offset + second.count * row_width
        )
        if not rows_align:
            return
        if not self.allocator.layout.all_constant() and not (
            last.heap_block_index == second.heap_block_index
            and last.heap_block_offset
            == second.heap_block_offset + second.total_heap_size
            and last.base_heap_ptr == second.base_heap_ptr
        ):
            return
        self.chunk_parts[last_id - 1].count += last.count
        self.chunk_parts[last_id - 1].total_heap_size += last.total_heap_size
        if len(self.chunk_parts) == chunk.part_ids.end():
            _ = self.chunk_parts.pop()
        chunk.part_ids.decrement_max()

    def build(
        mut self,
        mut state: TupleDataChunkState,
        append_offset: Int,
        append_count: Int,
    ):
        """TupleDataAllocator::Build: room for `append_count` rows (and the
        heap sizes in `state`), their row and heap pointers in `state` from
        `append_offset`."""
        if self._build_fast_path(state, append_offset, append_count):
            return
        state.chunk_parts.clear()
        state.chunk_part_chunks.clear()
        var offset = 0
        var row_width = self.allocator.layout.row_width()
        var all_constant = self.allocator.layout.all_constant()
        while offset != append_count:
            if (
                len(self.chunks) == 0
                or self.chunks[len(self.chunks) - 1].count
                == STANDARD_VECTOR_SIZE
            ):
                self.chunks.append(TupleDataChunk())
            var chunk_index = len(self.chunks) - 1
            var next = min(
                append_count - offset,
                STANDARD_VECTOR_SIZE - self.chunks[chunk_index].count,
            )
            var part = self._build_chunk_part(
                state, append_offset + offset, next
            )
            self._add_part(chunk_index, part)
            next = part.count
            self.count += next
            self.data_size += next * row_width
            if not all_constant:
                self.data_size += part.total_heap_size
            offset += next
            state.chunk_part_chunks.append(chunk_index)
            state.chunk_parts.append(
                self.chunks[chunk_index].part_ids.end() - 1
            )
        self._initialize_chunk_state(state, append_offset, True, False)
        # Only the last part of the first chunk can merge, given how parts
        # are built.
        self._merge_last_chunk_part(state.chunk_part_chunks[0])

    def _build_fast_path(
        mut self,
        mut state: TupleDataChunkState,
        append_offset: Int,
        append_count: Int,
    ) -> Bool:
        """TupleDataAllocator::BuildFastPath: rows without a heap that fit
        the last chunk and its row block extend its last part."""
        if not self.allocator.layout.all_constant():
            return False
        if len(self.chunks) == 0:
            return False
        var chunk_index = len(self.chunks) - 1
        if self.chunks[chunk_index].count + append_count > STANDARD_VECTOR_SIZE:
            return False
        var part_id = self.chunks[chunk_index].part_ids.end() - 1
        var part = self.chunk_parts[part_id]
        var row_width = self.allocator.layout.row_width()
        var added = append_count * row_width
        ref block = self.allocator.row_blocks[part.row_block_index]
        if block.size + added > block.capacity:
            return False
        var base = self.allocator.row_pointer(part) + part.count * row_width
        var rows = state.row_locations.unsafe_ptr()
        for i in range(append_count):
            rows[unsafe_offset=append_offset + i] = base + i * row_width
        self.chunks[chunk_index].count += append_count
        self.chunk_parts[part_id].count += append_count
        self.count += append_count
        self.allocator.row_blocks[part.row_block_index].size += added
        self.data_size += added
        return True

    def _build_chunk_part(
        mut self,
        state: TupleDataChunkState,
        append_offset: Int,
        append_count: Int,
    ) -> TupleDataChunkPart:
        """TupleDataAllocator::BuildChunkPart: as many of the rows as fit
        the current row block and one heap block."""
        var result = TupleDataChunkPart.empty()
        var row_width = self.allocator.layout.row_width()
        var row_blocks = len(self.allocator.row_blocks)
        if (
            row_blocks == 0
            or self.allocator.row_blocks[row_blocks - 1].remaining_capacity()
            < row_width
        ):
            self.allocator.create_row_block()
        result.row_block_index = len(self.allocator.row_blocks) - 1
        result.row_block_offset = self.allocator.row_blocks[
            result.row_block_index
        ].size
        result.count = min(
            self.allocator.row_blocks[result.row_block_index].remaining_rows(
                row_width
            ),
            append_count,
        )
        if not self.allocator.layout.all_constant():
            var sizes = state.heap_sizes.unsafe_ptr()
            var total = 0
            for i in range(result.count):
                total += sizes[unsafe_offset=append_offset + i]
            if total == 0:
                result.set_heap_empty()
            else:
                var first = sizes[unsafe_offset=append_offset]
                var last_room = self._heap_room()
                var remaining: Int
                if last_room >= first:
                    remaining = last_room
                else:
                    remaining = max(TUPLE_DATA_BLOCK_SIZE, first)
                if total <= remaining:
                    result.total_heap_size = total
                else:
                    # Not everything fits: as many rows as do.
                    result.total_heap_size = 0
                    for i in range(result.count):
                        var size = sizes[unsafe_offset=append_offset + i]
                        if result.total_heap_size + size > remaining:
                            result.count = i
                            break
                        result.total_heap_size += size
                if result.total_heap_size == 0:
                    result.set_heap_empty()
                else:
                    if self._heap_room() < first:
                        self.allocator.create_heap_block(
                            max(TUPLE_DATA_BLOCK_SIZE, first)
                        )
                    result.heap_block_index = (
                        len(self.allocator.heap_blocks) - 1
                    )
                    ref heap = self.allocator.heap_blocks[
                        result.heap_block_index
                    ]
                    result.heap_block_offset = heap.size
                    heap.size += result.total_heap_size
                    result.base_heap_ptr = heap.address()
        self.allocator.row_blocks[result.row_block_index].size += (
            result.count * row_width
        )
        return result

    def _heap_room(self) -> Int:
        """The remaining capacity of the last heap block; -1 without one."""
        var n = len(self.allocator.heap_blocks)
        if n == 0:
            return -1
        return self.allocator.heap_blocks[n - 1].remaining_capacity()

    def initialize_chunk_state(
        self,
        mut state: TupleDataChunkState,
        chunk_index: Int,
        init_heap: Bool,
    ) -> Int:
        """TupleDataAllocator::InitializeChunkState for a scan: the row
        pointers (and with `init_heap` the heap pointers and sizes) of chunk
        `chunk_index`; its row count."""
        ref chunk = self.chunks[chunk_index]
        state.chunk_parts.clear()
        for part_id in range(chunk.part_ids.start(), chunk.part_ids.end()):
            state.chunk_parts.append(part_id)
        self._initialize_chunk_state(state, 0, init_heap, init_heap)
        return chunk.count

    def _initialize_chunk_state(
        self,
        mut state: TupleDataChunkState,
        start: Int,
        init_heap_pointers: Bool,
        init_heap_sizes: Bool,
    ):
        """TupleDataAllocator::InitializeChunkStateInternal over the parts
        in `state.chunk_parts`, from row `start`."""
        ref layout = self.allocator.layout
        var row_width = layout.row_width()
        var all_constant = layout.all_constant()
        var heap_size_offset = layout.heap_size_offset()
        var rows = state.row_locations.unsafe_ptr()
        var sizes = state.heap_sizes.unsafe_ptr()
        var heaps = state.heap_locations.unsafe_ptr()
        var offset = start
        for p in range(len(state.chunk_parts)):
            var part = self.chunk_parts[state.chunk_parts[p]]
            var next = part.count
            var base = self.allocator.row_pointer(part)
            for i in range(next):
                rows[unsafe_offset=offset + i] = base + i * row_width
            if all_constant:
                offset += next
                continue
            if init_heap_sizes:
                for i in range(next):
                    sizes[unsafe_offset=offset + i] = Int(
                        _load_u64(
                            rows[unsafe_offset=offset + i] + heap_size_offset
                        )
                    )
            if part.total_heap_size == 0:
                offset += next
                continue
            if init_heap_pointers:
                heaps[unsafe_offset=offset] = (
                    part.base_heap_ptr + part.heap_block_offset
                )
                for i in range(1, next):
                    heaps[unsafe_offset=offset + i] = (
                        heaps[unsafe_offset=offset + i - 1]
                        + sizes[unsafe_offset=offset + i - 1]
                    )
            offset += next


@always_inline
def _load_u64(address: Int) -> UInt64:
    from std.memory import Pointer

    return Pointer[UInt64, MutAnyOrigin](
        unsafe_from_address=address
    ).unsafe_load[alignment=1]()
