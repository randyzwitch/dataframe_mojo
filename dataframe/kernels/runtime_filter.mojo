"""Membership filtering with register-blocked Bloom filters.

Hash each build key into a small block of bits. A probe missing any required
bit cannot be present; a possible hit still needs the join's exact key
comparison. Workers can build filters privately and merge their bits,
avoiding atomic operations on every build key. False positives only add
probe work, while false negatives would incorrectly discard result rows.
"""
from std.memory import ArcPointer, Pointer

from dataframe.storage.column import Column
from dataframe.dtype import DataType
from dataframe.parallel import Job, partitions, run_jobs, worker_count
from dataframe.series import Series

# Bloom-filter bit budget per key, minimum allocation and maximum allocation.
# The maximum bounds memory; declining a larger filter preserves correctness
# by leaving exact probing enabled.
comptime _BITS_PER_KEY = 12
comptime _MIN_BITS = 512
comptime _MAX_WORDS = 1 << 22
# Six bits of each of a hash's top four bytes pick the word's four bits.
comptime _SHIFT_MASK = UInt64(0x3F3F3F3F3F3F3F3F)


@always_inline
def _hash_keys[
    width: Int
](values: SIMD[DType.int64, width]) -> SIMD[DType.uint64, width]:
    """A 64-bit mix of each key (the murmur finalizer, as the distinct-count
    sketch uses)."""
    comptime m = SIMD[DType.uint64, width](0xD6E8FEB86659FD93)
    var x = values.cast[DType.uint64]()
    x ^= x >> 32
    x *= m
    x ^= x >> 32
    x *= m
    x ^= x >> 32
    return x


@always_inline
def _word_masks[
    width: Int
](hashes: SIMD[DType.uint64, width]) -> SIMD[DType.uint64, width]:
    """The four bits each hash sets in its word."""
    var shifts = hashes & SIMD[DType.uint64, width](_SHIFT_MASK)
    var one = SIMD[DType.uint64, width](1)
    var low = SIMD[DType.uint64, width](63)
    return (
        (one << ((shifts >> 32) & low))
        | (one << ((shifts >> 40) & low))
        | (one << ((shifts >> 48) & low))
        | (one << (shifts >> 56))
    )


def bloom_blocks_for(keys: Int) -> Int:
    """Words of a filter for `keys` keys, a power of two; 0 when it would be
    larger than is built."""
    var bits = max(_MIN_BITS, keys * _BITS_PER_KEY)
    var words = 1
    while words * 64 < bits:
        words *= 2
    if words > _MAX_WORDS:
        return 0
    return words


struct KeyBloom(Movable):
    """A register-blocked bloom filter over 64-bit integer keys: `blocks`
    64-bit words, a power of two."""

    var words: List[UInt64]
    var blocks: Int

    def __init__(out self, blocks: Int):
        self.blocks = blocks
        self.words = List[UInt64](length=blocks, fill=0)

    @always_inline
    def _base(self) -> Pointer[UInt64, MutAnyOrigin]:
        return Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=Int(self.words.unsafe_ptr())
        )

    @always_inline
    def insert(self, key: Int64):
        var hash = _hash_keys[1](SIMD[DType.int64, 1](key))
        var at = self._base().unsafe_offset(Int(hash & UInt64(self.blocks - 1)))
        at[] |= _word_masks[1](hash)[0]

    @always_inline
    def contains(self, key: Int64) -> Bool:
        var hash = _hash_keys[1](SIMD[DType.int64, 1](key))
        var mask = _word_masks[1](hash)[0]
        var word = self._base()[
            unsafe_offset=Int(hash & UInt64(self.blocks - 1))
        ]
        return word & mask == mask

    @always_inline
    def found[
        width: Int
    ](self, keys: SIMD[DType.int64, width]) -> SIMD[DType.bool, width]:
        """Which of `width` keys may be in the filter: hashed together, then
        one word read for each."""
        var hashes = _hash_keys[width](keys)
        var masks = _word_masks[width](hashes)
        var at = hashes & SIMD[DType.uint64, width](UInt64(self.blocks - 1))
        var base = self._base()
        var words = SIMD[DType.uint64, width](0)
        comptime for lane in range(width):
            words[lane] = base[unsafe_offset=Int(at[lane])]
        return (words & masks).eq(masks)

    def add_column[D: DType](self, column: Column[Scalar[D]]):
        """Insert the non-null values of a column."""
        var values = column._ptr()
        var all_valid = column.null_count() == 0
        for i in range(len(column)):
            if all_valid or column._valid(i):
                self.insert(Int64(values[unsafe_offset=i]))


struct _BloomBuild(Job):
    """One worker's filter over a slice of the build's keys, zeroed and
    filled on the worker."""

    var keys: Series
    var blocks: Int
    var words: List[UInt64]

    def __init__(out self, var keys: Series, blocks: Int):
        self.keys = keys^
        self.blocks = blocks
        self.words = List[UInt64]()

    def run(mut self) raises:
        var bloom = KeyBloom(self.blocks)
        var physical = self.keys.dtype().physical()
        for part in self.keys.chunks():
            if physical == DataType.INT64:
                bloom.add_column(part._data[Column[Int64]])
            else:
                bloom.add_column(part._data[Column[Int32]])
        swap(self.words, bloom.words)


struct _BloomMerge(Job):
    """Words [start, end) of the merged filter: every worker's or-ed."""

    var filters: List[Int]
    var target: Int
    var start: Int
    var end: Int

    def __init__(
        out self, filters: List[Int], target: Int, start: Int, end: Int
    ):
        self.filters = filters.copy()
        self.target = target
        self.start = start
        self.end = end

    def run(mut self) raises:
        var into = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=self.target
        )
        for f in range(len(self.filters)):
            var words = Pointer[UInt64, MutAnyOrigin](
                unsafe_from_address=self.filters[f]
            )
            for w in range(self.start, self.end):
                into[unsafe_offset=w] |= words[unsafe_offset=w]


def build_bloom(
    key: Series, rows: Int
) raises -> Optional[ArcPointer[KeyBloom]]:
    """A filter over the non-null values of an integer key column of up to
    `rows` keys, or None when it would be larger than is built. The keys
    are split over the workers, each filling a filter of its own, and the
    filters are merged by or-ing their words, each worker a range of them.
    Private filters avoid workers contending on shared atomic bit updates.
    Their bitwise union admits every build key; any extra bits can only
    increase false positives, which exact join comparisons reject."""
    var blocks = bloom_blocks_for(rows)
    if blocks == 0:
        return None
    var n = len(key)
    var parts = worker_count(n)
    var jobs = List[_BloomBuild](capacity=parts)
    var step = (n + parts - 1) // parts
    for p in range(parts):
        var offset = min(n, p * step)
        jobs.append(
            _BloomBuild(key.slice(offset, min(step, n - offset)), blocks)
        )
    if parts == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    var merged = KeyBloom(1)
    swap(merged.words, jobs[0].words)
    merged.blocks = blocks
    if parts > 1:
        var others = List[Int](capacity=parts - 1)
        for j in range(1, parts):
            others.append(Int(jobs[j].words.unsafe_ptr()))
        var bounds = partitions(blocks, parts, 8)
        var merges = List[_BloomMerge](capacity=parts)
        for p in range(parts):
            if bounds[p] < bounds[p + 1]:
                merges.append(
                    _BloomMerge(
                        others,
                        Int(merged.words.unsafe_ptr()),
                        bounds[p],
                        bounds[p + 1],
                    )
                )
        run_jobs(merges)
    _ = jobs^
    return Optional(ArcPointer(merged^))
