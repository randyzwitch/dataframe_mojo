"""Membership filtering with register-blocked Bloom filters.

Hash each build key into a small block of bits. A probe missing any required
bit cannot be present; a possible hit still needs the join's exact key
comparison. A key's bits come from the hash a join row stores for a single
integer key, so a hash table can fill the filter from its rows' hashes
without reading the key values again. Workers either fill private filters
and merge their bits, or set bits in one shared filter with atomic ORs.
False positives only add probe work, while false negatives would
incorrectly discard result rows.
"""
from std.atomic import Atomic
from std.memory import ArcPointer, Pointer, bitcast

from dataframe.hash_bits import _splitmix64

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
def key_hash(key: Int64) -> UInt64:
    """The hash a join row stores for a single integer key `key` (signed
    values sign-extended, as the join's key hash reads them)."""
    return _splitmix64(bitcast[DType.uint64](key))


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


# The join's bloom filter rule: a filter pays only when the build has
# fewer keys than the probe has rows, so that most probe rows can miss it.
# A build side that is not itself filtered usually holds every key of a
# foreign-key probe, so then its keys must also be few next to the probe
# (ratio) or few overall (count) for the filter to reject enough rows.
comptime BUILD_TO_PROBE_RATIO_THRESHOLD = 1.0
comptime NON_FILTERING_RATIO_THRESHOLD = 0.1
comptime NON_FILTERING_BUILD_SIDE_THRESHOLD = 4194304


def can_use_bloom_filter(
    equalities: Int, build_count: Int, probe_rows: Int, build_filtered: Bool
) -> Bool:
    """Whether a join's bloom filter is worth building: one equality key
    (the filter tests one probe column), and a build small next to its
    probe by the rule above."""
    if equalities != 1 or build_count == 0 or probe_rows <= 0:
        return False
    var ratio = Float64(build_count) / Float64(probe_rows)
    if ratio > BUILD_TO_PROBE_RATIO_THRESHOLD:
        return False
    if (
        not build_filtered
        and ratio > NON_FILTERING_RATIO_THRESHOLD
        and build_count > NON_FILTERING_BUILD_SIDE_THRESHOLD
    ):
        return False
    return True


@always_inline
def bloom_set_bits(words: Int, blocks: Int, hash: UInt64, shared: Bool):
    """Set `hash`'s bits in the filter whose `blocks` words are at
    `words`. `shared`: other workers set bits in the same filter at the
    same time, so the word is updated with an atomic OR and no bit another
    worker sets is lost."""
    var at = words + 8 * Int(hash & UInt64(blocks - 1))
    var mask = _word_masks[1](hash)[0]
    if shared:
        ref word = Pointer[Atomic[UInt64], MutAnyOrigin](
            unsafe_from_address=at
        )[]
        var seen = word.load()
        # A word that already has the bits needs no write; otherwise the
        # OR is retried until no other worker changed the word between.
        while seen & mask != mask:
            if word.compare_exchange(seen, seen | mask):
                break
    else:
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=at)[] |= mask


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
        self.insert_hash(key_hash(key))

    @always_inline
    def insert_hash(self, hash: UInt64):
        """Set `hash`'s bits (one writer at a time)."""
        bloom_set_bits(Int(self.words.unsafe_ptr()), self.blocks, hash, False)

    @always_inline
    def contains(self, key: Int64) -> Bool:
        var hash = key_hash(key)
        var mask = _word_masks[1](hash)[0]
        var word = self._base()[
            unsafe_offset=Int(hash & UInt64(self.blocks - 1))
        ]
        return word & mask == mask

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
