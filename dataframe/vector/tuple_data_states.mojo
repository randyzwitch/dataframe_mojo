"""Track vector formats and row/heap pointers during append and scan.

Nested vector formats live in a flat list. The first column_count entries
are the top-level columns; child indices identify nested entries. Blocks
stay allocated at stable addresses, so pointers remain valid for the
collection's lifetime.
"""
from dataframe.vector.types import (
    PHYSICAL_OPAQUE,
    STANDARD_VECTOR_SIZE,
    SelectionVector,
    UnifiedVectorFormat,
)


struct TupleDataVectorFormat(Copyable, Movable):
    """One vector's unified format, its selection before a list slice, and the
    buffers this entry owns."""

    var physical: Int
    var unified: UnifiedVectorFormat
    var original_sel: SelectionVector
    var children: List[Int]
    # A list's (offset, length) entry per row, read from storage.
    var entries: List[UInt64]
    # Elements in a list's child.
    var list_size: Int
    # For a list in a list: the outer selection composed with the inner one, so
    # the inner child is read through a single selection.
    var sliced_sel: List[UInt32]
    # A list in a list's child entries, combined per appended row, and their
    # validity.
    var has_combined: Bool
    var combined: UnifiedVectorFormat
    var combined_entries: List[UInt64]
    var combined_validity: List[UInt8]

    def __init__(out self, physical: Int = PHYSICAL_OPAQUE):
        self.physical = physical
        self.unified = UnifiedVectorFormat()
        self.original_sel = SelectionVector()
        self.children = List[Int]()
        self.entries = List[UInt64]()
        self.list_size = 0
        self.sliced_sel = List[UInt32]()
        self.has_combined = False
        self.combined = UnifiedVectorFormat()
        self.combined_entries = List[UInt64]()
        self.combined_validity = List[UInt8]()


struct TupleDataChunkState(Copyable, Movable):
    """TupleDataChunkState: the formats of the chunk being appended, the
    columns appended or gathered, and per row its row pointer, heap
    pointer and heap size (STANDARD_VECTOR_SIZE of each)."""

    var vector_data: List[TupleDataVectorFormat]
    var column_ids: List[Int]
    var row_locations: List[Int]
    var heap_locations: List[Int]
    var heap_sizes: List[Int]
    # The chunk parts (indices into the segment's) the row pointers come
    # from, and for an append the chunk of each.
    var chunk_parts: List[Int]
    var chunk_part_chunks: List[Int]

    def __init__(out self):
        self.vector_data = List[TupleDataVectorFormat]()
        self.column_ids = List[Int]()
        self.row_locations = List[Int](
            unsafe_uninit_length=STANDARD_VECTOR_SIZE
        )
        self.heap_locations = List[Int](
            unsafe_uninit_length=STANDARD_VECTOR_SIZE
        )
        self.heap_sizes = List[Int](unsafe_uninit_length=STANDARD_VECTOR_SIZE)
        self.chunk_parts = List[Int]()
        self.chunk_part_chunks = List[Int]()

    @always_inline
    def rows(self) -> Int:
        """The address of the row pointers."""
        return Int(self.row_locations.unsafe_ptr())

    @always_inline
    def heaps(self) -> Int:
        """The address of the heap pointers."""
        return Int(self.heap_locations.unsafe_ptr())

    @always_inline
    def sizes(self) -> Int:
        """The address of the heap sizes."""
        return Int(self.heap_sizes.unsafe_ptr())


struct TupleDataAppendState(Movable):
    """TupleDataAppendState (the pin state has nothing to hold here)."""

    var chunk_state: TupleDataChunkState

    def __init__(out self):
        self.chunk_state = TupleDataChunkState()


struct TupleDataScanState(Movable):
    """TupleDataScanState: the next segment and chunk to scan."""

    var chunk_state: TupleDataChunkState
    var segment_index: Int
    var chunk_index: Int

    def __init__(out self):
        self.chunk_state = TupleDataChunkState()
        self.segment_index = 0
        self.chunk_index = 0
