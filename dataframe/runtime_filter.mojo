"""Membership filters a finished hash join build hands to the scan that
feeds its probe side (Polars' join runtime filters,
`polars-stream/src/nodes/joins/runtime_filter.rs`; DuckDB's join filter
pushdown). A split-block bloom filter over the build's integer keys, the
layout of Parquet's bloom filters: 32-byte blocks, and a key sets one bit in
each of the eight words of one block (`polars-utils/src/bloom_filter.rs`).
A probe row whose key the filter lacks joins nothing, so the scan drops it
before any step reads or gathers its other columns.
"""
from std.memory import ArcPointer, Pointer

from .bool_column import BoolColumn
from .column import Column
from .dtype import DataType
from .execution import _valid_lanes
from .series import Series

# Bits per key a filter is sized for, the smallest and largest filter
# built (`BLOOM_BITS_PER_KEY`, `BLOOM_MIN_BYTES`, `BLOOM_MAX_BYTES`).
comptime _BITS_PER_KEY = 8
comptime _MIN_BYTES = 64 * 1024
comptime _MAX_BYTES = 32 * 1024 * 1024
comptime _BLOCK_BYTES = 32
# Most build keys, as a share of the probe side's distinct keys, for which a
# filter is worth probing (`MAX_BUILD_PROBE_DISTINCT_RATIO`).
comptime MAX_BUILD_PROBE_DISTINCT_RATIO = 0.3
# A build side must be this many times smaller than the rows it filters
# (`LOPSIDED_FACTOR`).
comptime LOPSIDED_FACTOR = 8

comptime _SALT = SIMD[DType.uint32, 8](
    1203114875,
    1150766481,
    2284105051,
    2729912477,
    1884591559,
    770785867,
    2667333959,
    1550580529,
)


@always_inline
def _hash_key(value: Int64) -> UInt64:
    """A 64-bit mix of a key (the murmur finalizer, as the distinct-count
    sketch uses)."""
    var x = UInt64(value)
    x ^= x >> 32
    x *= UInt64(0xD6E8FEB86659FD93)
    x ^= x >> 32
    x *= UInt64(0xD6E8FEB86659FD93)
    x ^= x >> 32
    return x


@always_inline
def _block_mask(hash: UInt64) -> SIMD[DType.uint32, 8]:
    var key = UInt32(hash & 0xFFFFFFFF)
    var shifts = (SIMD[DType.uint32, 8](key) * _SALT) >> 27
    return SIMD[DType.uint32, 8](1) << shifts


@always_inline
def _block_index(hash: UInt64, blocks: Int) -> Int:
    return Int(((hash >> 32) * UInt64(blocks)) >> 32)


def bloom_blocks_for(keys: Int) -> Int:
    """Blocks of a filter for `keys` keys, a power of two; 0 when it would
    be larger than is built."""
    var wanted = max(keys, _MIN_BYTES * 8 // _BITS_PER_KEY)
    var bits = wanted * _BITS_PER_KEY
    var blocks = max(1, (bits + _BLOCK_BYTES * 8 - 1) // (_BLOCK_BYTES * 8))
    var size = 1
    while size < blocks:
        size *= 2
    if size * _BLOCK_BYTES > _MAX_BYTES:
        return 0
    return size


struct KeyBloom(Movable):
    """A split-block bloom filter over 64-bit integer keys."""

    var words: List[UInt32]
    var blocks: Int
    # Index of the first word on a 32-byte boundary.
    var start: Int

    def __init__(out self, blocks: Int):
        self.blocks = blocks
        self.words = List[UInt32](length=blocks * 8 + 8, fill=0)
        var address = Int(self.words.unsafe_ptr())
        self.start = ((32 - address % 32) % 32) // 4

    @always_inline
    def _base(self) -> Pointer[UInt32, MutAnyOrigin]:
        return Pointer[UInt32, MutAnyOrigin](
            unsafe_from_address=Int(self.words.unsafe_ptr()) + 4 * self.start
        )

    @always_inline
    def insert(self, key: Int64):
        var hash = _hash_key(key)
        var block = self._base() + _block_index(hash, self.blocks) * 8
        block.store(block.load[width=8]() | _block_mask(hash))

    @always_inline
    def contains(self, key: Int64) -> Bool:
        var hash = _hash_key(key)
        var block = self._base() + _block_index(hash, self.blocks) * 8
        var mask = _block_mask(hash)
        return (block.load[width=8]() & mask).eq(mask).reduce_and()

    def add_column[D: DType](self, column: Column[Scalar[D]]):
        """Insert the non-null values of a column."""
        var values = column._ptr()
        var all_valid = column.null_count() == 0
        for i in range(len(column)):
            if all_valid or column._valid(i):
                self.insert(Int64(values[i]))


def build_bloom(key: Series, rows: Int) -> Optional[ArcPointer[KeyBloom]]:
    """A filter over the non-null values of an integer key column of up to
    `rows` keys, or None when it would be larger than is built."""
    var blocks = bloom_blocks_for(rows)
    if blocks == 0:
        return None
    var bloom = KeyBloom(blocks)
    var physical = key.dtype().physical()
    for part in key.chunks():
        if physical == DataType.INT64:
            bloom.add_column(part._data[Column[Int64]])
        else:
            bloom.add_column(part._data[Column[Int32]])
    return Optional(ArcPointer(bloom^))


def _bloom_bits[
    D: DType
](
    column: Column[Scalar[D]],
    bloom: KeyBloom,
    target: Pointer[UInt8, MutAnyOrigin],
    selection: Pointer[UInt8, MutAnyOrigin],
    selected: Bool,
) -> Int:
    var values = column._ptr()
    var n = len(column)
    var all_valid = column.null_count() == 0
    var kept = 0
    if not selected:
        for i in range(n):
            if all_valid or column._valid(i):
                if bloom.contains(Int64(values[i])):
                    target[unsafe_offset=i >> 3] |= UInt8(1) << UInt8(i & 7)
                    kept += 1
        return kept
    # Only the rows an earlier filter kept are looked up.
    for byte in range((n + 7) // 8):
        var bits = selection[unsafe_offset=byte]
        if bits == 0:
            continue
        for lane in range(8):
            var i = byte * 8 + lane
            if i >= n:
                break
            if (bits >> UInt8(lane)) & 1 == 0:
                continue
            if all_valid or column._valid(i):
                if bloom.contains(Int64(values[i])):
                    target[unsafe_offset=i >> 3] |= UInt8(1) << UInt8(lane)
                    kept += 1
    return kept


def bloom_mask(
    series: Series,
    bloom: KeyBloom,
    selection: Optional[BoolColumn] = None,
) raises -> Tuple[BoolColumn, Int]:
    """The rows of an unchunked integer column whose value is valid and may
    be in the filter, as a packed mask, and how many. With a `selection`
    (a mask of no nulls and no offset) only its rows are looked up, and the
    mask holds only those."""
    var n = len(series)
    var values = List[UInt8](length=(n + 7) // 8, fill=0)
    var target = values.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var picked = Bool(selection)
    var rows = target
    if picked:
        rows = (
            selection.value()
            ._data[]
            .unsafe_ptr()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
    var kept = 0
    if series.dtype().physical() == DataType.INT64:
        kept = _bloom_bits(
            series._data[Column[Int64]], bloom, target, rows, picked
        )
    else:
        kept = _bloom_bits(
            series._data[Column[Int32]], bloom, target, rows, picked
        )
    return (BoolColumn(values=values^, bits=List[UInt8](), length=n), kept)
