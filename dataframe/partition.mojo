"""Hash partitioning of rows by key, so grouping needs no merge (#104).

Rows are hashed by their key values and scattered into buckets by hash
range. Equal keys hash equally and so land in the same bucket, which means
each bucket can be grouped on its own: the dictionaries and reduction states
of different buckets never mention the same key, and nothing has to be
reconciled afterwards. That is what makes grouping parallel at every
cardinality; the earlier merge-based attempts (#8) paid workers x distinct
keys to reconcile private dictionaries and lost at high cardinality.

Hash equality follows key equality as `encode_rows` defines it: every NaN
is one value, -0.0 equals 0.0, strings compare by bytes, and a null key is
one ordinary value. The permutation is stable: within a bucket, rows keep
their input order, so first-occurrence order within a bucket is preserved.

Partitioning is not always worth it. Gathering the key and value columns
into bucket order costs a full materialization of the frame, which only
pays off when the serial encode it replaces is itself expensive -- that is,
when distinct keys are many. `low_cardinality` answers that beforehand by
hashing a strided sample on the calling thread and counting how much of
hash space it touches: keys spread across most slots mean many distinct
values. The sample is a few thousand rows, so deciding costs far less than
the hash pass it guards, and nothing is wasted when the answer is no.
"""
from std.memory import ArcPointer, Pointer, bitcast
from std.sys import size_of
from std.sys.intrinsics import prefetch
from std.collections import Dict

from .aggregate import float_key
from .bool_column import BoolColumn
from .column import Column, _validity_at
from .dtype import NUMERIC_DTYPES
from .gather import take_parallel
from .hashing import RowKeys, encode_rows
from .parallel import Job, partitions, run_jobs
from .series import Series
from .string_column import StringColumn

comptime _NULL_KEY = UInt64(0x9E3779B97F4A7C15)

# Histogram granularity. Always 256 slots regardless of the bucket count, so
# the number of occupied slots estimates distinct keys: each slot holds a
# 1/256 slice of hash space, so few occupied slots means few distinct keys.
comptime _SLOTS = 256
comptime _SLOT_SHIFT = 56

# Rows hashed to estimate cardinality before committing to a full pass.
comptime _SAMPLE_ROWS = 4096
# Keys with at most this many sampled values each, whose product is at
# most _SMALL_KEY_PRODUCT, group by row ranges (`small_key_product`).
comptime _SMALL_KEY_VALUES = 256
comptime _SMALL_KEY_PRODUCT = 65536


def _mix(value: UInt64) -> UInt64:
    """splitmix64's finalizer: spreads low-entropy keys across all bits."""
    var z = value
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _combine(seed: UInt64, key: UInt64) -> UInt64:
    return _mix(seed ^ (key + 0x9E3779B97F4A7C15 + (seed << 6) + (seed >> 2)))


def _short_hash(word: UInt64, length: Int) -> UInt64:
    """Encode up to eight bytes and their length before the final mix."""
    var mask = UInt64.MAX
    if length < 8:
        mask = (UInt64(1) << UInt64(length * 8)) - 1
    return (word & mask) ^ (UInt64(length) << 56) ^ UInt64(0xCBF29CE484222325)


def _hash_bytes(bytes: Span[UInt8, ImmutAnyOrigin]) -> UInt64:
    """Hash short strings by one packed word, longer strings in eight-byte blocks.

    Block mixing follows DuckDB 1.5.5 HashBytes (see THIRD_PARTY_NOTICES).
    Keep our existing short-key encoding and final column-level `_mix`.
    Every wide load lies entirely inside the supplied byte span.
    """
    if len(bytes) <= 8:
        var word = UInt64(0)
        for k in range(len(bytes)):
            word |= UInt64(bytes[k]) << UInt64(k * 8)
        return _short_hash(word, len(bytes))
    var h = UInt64(0xE17A1465) ^ (UInt64(len(bytes)) * 0xC6A4A7935BD1E995)
    var remainder = len(bytes) & 7
    var end = len(bytes) - remainder
    for k in range(0, end, 8):
        var word = bitcast[DType.uint64, 1](
            bytes.unsafe_ptr().unsafe_offset(k).unsafe_load[width=8]()
        )
        h = (h ^ word) * 0xD6E8FEB86659FD93
    if remainder:
        # Read the final eight bytes, then discard the bytes already mixed.
        # len(bytes) > 8 here, so this never reads before or after the span.
        var tail = bitcast[DType.uint64, 1](
            bytes.unsafe_ptr()
            .unsafe_offset(len(bytes) - 8)
            .unsafe_load[width=8]()
        )
        tail >>= UInt64((8 - remainder) * 8)
        h = (h ^ tail) * 0xD6E8FEB86659FD93
    return h


def _hash_column(
    series: Series,
    start: Int,
    end: Int,
    out_address: Int,
    first: Bool,
    output_offset: Int = 0,
) raises:
    """Hash rows [start, end) of one key column into `out`, combining with
    what earlier key columns wrote unless this is the first."""
    if series.dtype().is_nested():
        raise Error(
            "list and struct columns cannot be keys yet: " + series.name()
        )
    var p = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=out_address)

    @__parameter
    def write(i: Int, key: UInt64):
        var slot = p.unsafe_offset(i - output_offset)
        slot[] = _mix(key) if first else _combine(slot[], key)

    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            ref column = series._data[Column[Scalar[D]]]
            for i in range(start, end):
                if not column._valid(i):
                    write(i, _NULL_KEY)
                    continue
                var value = column._get(i)

                comptime if D.is_floating_point():
                    write(i, float_key(Float64(value)))
                elif D.is_unsigned():
                    write(i, UInt64(value))
                else:
                    write(i, bitcast[DType.uint64](Int64(value)))
            return
    if series._data.isa[Column[Int128]]():
        ref column = series._data[Column[Int128]]
        for i in range(start, end):
            if not column._valid(i):
                write(i, _NULL_KEY)
            else:
                var key = bitcast[DType.uint128](column._get(i))
                write(i, UInt64(key) ^ UInt64(key >> 64))
        return
    if series._data.isa[BoolColumn]():
        ref bools = series._data[BoolColumn]
        for i in range(start, end):
            if not bools._valid(i):
                write(i, _NULL_KEY)
            else:
                write(i, UInt64(1) if bools._get(i) else UInt64(2))
        return
    ref strings = series._data[StringColumn]
    if not strings._is_view_storage():
        # Offsets and bytes are read in place; the hash is the same one
        # _hash_bytes gives, so every storage of a string hashes alike.
        ref data = strings._bytes[]
        var size = len(data)
        var base = data.unsafe_ptr()
        var offsets = (
            strings._offsets[].unsafe_ptr().unsafe_offset(strings._offset)
        )
        # A missing bitmap means no nulls; this runs per sampled row too,
        # so it must not count them.
        var nulls = len(strings._bits[]) != 0
        for i in range(start, end):
            if nulls and not strings._valid(i):
                write(i, _NULL_KEY)
                continue
            var byte_start = Int(offsets[unsafe_offset=i])
            var length = Int(offsets[unsafe_offset=i + 1]) - byte_start
            if length <= 8 and byte_start <= size - 8:
                var word = bitcast[DType.uint64, 1](
                    base.unsafe_offset(byte_start).unsafe_load[width=8]()
                )
                write(i, _short_hash(word, length))
            else:
                write(
                    i,
                    _hash_bytes(
                        Span[UInt8, ImmutAnyOrigin](
                            unsafe_ptr=base.unsafe_offset(byte_start)
                            .unsafe_mut_cast[False]()
                            .unsafe_origin_cast[ImmutAnyOrigin](),
                            length=length,
                        )
                    ),
                )
        return
    for i in range(start, end):
        if not strings._valid(i):
            write(i, _NULL_KEY)
        else:
            write(i, _hash_bytes(strings._get(i).as_bytes()))


struct _HashJob(Job):
    """Hash one row range and count how many rows fall in each bucket."""

    var keys: List[Series]
    var start: Int
    var end: Int
    var out: Int
    var histogram: List[Int]

    def __init__(
        out self,
        keys: List[Series],
        start: Int,
        end: Int,
        out_address: Int,
    ):
        self.keys = keys.copy()
        self.start = start
        self.end = end
        self.out = out_address
        self.histogram = List[Int](length=_SLOTS, fill=0)

    def run(mut self) raises:
        for j in range(len(self.keys)):
            _hash_column(self.keys[j], self.start, self.end, self.out, j == 0)
        var p = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.out)
        for i in range(self.start, self.end):
            self.histogram[
                Int(p.unsafe_offset(i)[] >> UInt64(_SLOT_SHIFT))
            ] += 1


struct _ScatterJob(Job):
    """Write one row range's indices into their buckets' reserved slots."""

    var start: Int
    var end: Int
    var hashes: Int
    var order: Int
    # Where to write each row's hash in bucket order too (0: nowhere).
    var ordered_hashes: Int
    var fold: Int
    var next: List[Int]

    def __init__(
        out self,
        start: Int,
        end: Int,
        hashes: Int,
        order: Int,
        ordered_hashes: Int,
        fold: Int,
        var next: List[Int],
    ):
        self.start = start
        self.end = end
        self.hashes = hashes
        self.order = order
        self.ordered_hashes = ordered_hashes
        self.fold = fold
        self.next = next^

    def run(mut self) raises:
        var h = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.hashes)
        var o = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var oh = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=self.ordered_hashes
        )
        for i in range(self.start, self.end):
            var hash = h.unsafe_offset(i)[]
            var slot = Int(hash >> UInt64(_SLOT_SHIFT))
            var bucket = slot >> self.fold
            var at = self.next[bucket]
            o.unsafe_offset(at)[] = i
            if self.ordered_hashes != 0:
                oh.unsafe_offset(at)[] = hash
            self.next[bucket] = at + 1


@fieldwise_init
struct Partitioned(Movable):
    """A stable permutation of row indices grouped by hash bucket.

    Rows of bucket b are `order[bounds[b] : bounds[b + 1]]`, in input order.
    """

    var order: List[Int]
    var bounds: List[Int]
    # Each row's key hash in the same order as `order`, when requested.
    var hashes: List[UInt64]

    def buckets(self) -> Int:
        return len(self.bounds) - 1


def _prefer_whole_sample(
    occupied: Int, taken: Int, counts: List[Int], hashes: List[UInt64]
) -> Bool:
    if 2 * occupied < min(taken, _SLOTS):
        return True
    if taken < 128:
        return False
    var largest = 0
    for count in counts:
        largest = max(largest, count)
    if 4 * largest < taken:
        return False
    # A hot key alone does not imply cheap serial encoding: the remaining
    # rows may all be unique. Count exact hashes only for skewed samples.
    var distinct = Dict[UInt64, Bool]()
    for hash in hashes:
        distinct[hash] = True
    return 3 * len(distinct) <= taken


def low_cardinality(keys: List[Series]) raises -> Bool:
    """Whether whole-frame encoding is cheaper than hash partitioning.

    A bounded sample first counts occupied hash slots. If a key dominates,
    exact sampled hash cardinality checks whether the remaining domain is
    small enough to avoid the partition scatter and gather.
    """
    var rows = len(keys[0])
    if rows == 0:
        return True
    var sample = min(rows, _SAMPLE_ROWS)
    var stride = max(1, rows // sample)
    var picked = List[Int](capacity=sample)
    var i = 0
    while i < rows and len(picked) < sample:
        picked.append(i)
        var taken = len(picked)
        i = taken * stride + (taken * 7919) % stride
    var hashes = List[UInt64](length=len(picked), fill=0)
    for j in range(len(keys)):
        # Gather only the bounded sample, including from source chunks.
        # Resolve storage and dtype once per key rather than once per row.
        var values = keys[j].take(picked)
        _hash_column(values, 0, len(picked), Int(hashes.unsafe_ptr()), j == 0)
    var seen = List[Bool](length=_SLOTS, fill=False)
    var counts = List[Int](length=_SLOTS, fill=0)
    var occupied = 0
    for hash in hashes:
        var slot = Int(hash >> UInt64(_SLOT_SHIFT))
        counts[slot] += 1
        if not seen[slot]:
            seen[slot] = True
            occupied += 1
    return _prefer_whole_sample(occupied, len(picked), counts, hashes)


def small_key_product(keys: List[Series]) raises -> Bool:
    """Whether several keys, each with few values, can only form few groups.

    A sample of key rows looks mostly distinct when two keys of 100 values
    each form 10,000 groups (H2O q2), so `low_cardinality` says no, and the
    partitioned group-by is chosen for what per-range states aggregate in
    a third of the time. Each key on its own shows its few values: when
    every key's sample holds at most 256 distinct values, the groups can
    be no more than the product of those counts.
    """
    var rows = len(keys[0]) if len(keys) > 1 else 0
    if rows == 0:
        return False
    var sample = min(rows, _SAMPLE_ROWS)
    var picked = List[Int](capacity=sample)
    for k in range(sample):
        picked.append(k * rows // sample)
    var product = 1
    for key in keys:
        # One gather and one hashing pass over the sampled rows.
        var values = key.take(picked)
        var hashes = List[UInt64](length=sample, fill=0)
        _hash_column(
            values, 0, sample, Int(hashes.unsafe_ptr()), True, output_offset=0
        )
        sort(hashes)
        var distinct = 0
        for k in range(len(hashes)):
            if k == 0 or hashes[k] != hashes[k - 1]:
                distinct += 1
        if distinct > _SMALL_KEY_VALUES:
            return False
        product *= distinct
        if product > _SMALL_KEY_PRODUCT:
            return False
    return True


struct Partitioner(Movable):
    """One hash pass over the keys, plus the histogram that decides whether
    scattering is worth it."""

    var hashes: List[UInt64]
    var histogram: List[Int]
    var worker_histograms: List[List[Int]]
    var rows: Int
    # The key columns in one chunk each, as hashed; rows are compared here
    # when a bucket is encoded from hashes (encode_bucket).
    var keys: List[Series]

    def __init__(out self, keys: List[Series], workers: Int) raises:
        for key in keys:
            if key.dtype().is_nested():
                raise Error(
                    "list and struct columns cannot be keys yet: " + key.name()
                )
        var contiguous = List[Series](capacity=len(keys))
        for key in keys:
            contiguous.append(key.rechunk() if key.is_chunked() else key.copy())
        self.rows = len(contiguous[0])
        self.keys = contiguous.copy()
        # Every row's hash is written by exactly one _HashJob (#388).
        self.hashes = List[UInt64](unsafe_uninit_length=self.rows)
        self.histogram = List[Int](length=_SLOTS, fill=0)
        self.worker_histograms = List[List[Int]](capacity=workers)
        var bounds = partitions(self.rows, workers, 64)
        var jobs = List[_HashJob](capacity=workers)
        for w in range(workers):
            jobs.append(
                _HashJob(
                    contiguous,
                    bounds[w],
                    bounds[w + 1],
                    Int(self.hashes.unsafe_ptr()),
                )
            )
        run_jobs(jobs)
        for w in range(workers):
            var counts = jobs[w].histogram.copy()
            for s in range(_SLOTS):
                self.histogram[s] += counts[s]
            self.worker_histograms.append(counts^)

    def into_hashes(deinit self) -> List[UInt64]:
        """The row hashes, moved out (no 8-bytes-a-row copy)."""
        return self.hashes^

    def scatter(
        mut self, workers: Int, with_hashes: Bool = False
    ) raises -> Partitioned:
        """Build the stable permutation, folding slots into buckets; with
        `with_hashes`, also each row's hash in the same order."""
        var buckets = 1
        var fold = 8
        while buckets < 2 * workers and buckets < _SLOTS:
            buckets *= 2
            fold -= 1

        var starts = List[Int](length=buckets + 1, fill=0)
        for s in range(_SLOTS):
            starts[(s >> fold) + 1] += self.histogram[s]
        for b in range(buckets):
            starts[b + 1] += starts[b]

        # Each worker's write cursor within every bucket: a worker's rows
        # follow the previous workers' rows, which keeps the order stable.
        var bounds = partitions(self.rows, workers, 64)
        var per_worker = List[List[Int]](capacity=workers)
        for w in range(workers):
            var counts = List[Int](length=buckets, fill=0)
            for s in range(_SLOTS):
                counts[s >> fold] += self.worker_histograms[w][s]
            per_worker.append(counts^)
        # Both are permutations of every row, each slot written once by
        # the scatter jobs, so neither is filled first (#388).
        var order = List[Int](unsafe_uninit_length=self.rows)
        var ordered = List[UInt64](
            unsafe_uninit_length=self.rows if with_hashes else 0
        )
        var cursor = starts.copy()
        var jobs = List[_ScatterJob](capacity=workers)
        for w in range(workers):
            var next = List[Int](capacity=buckets)
            for b in range(buckets):
                next.append(cursor[b])
                cursor[b] += per_worker[w][b]
            jobs.append(
                _ScatterJob(
                    bounds[w],
                    bounds[w + 1],
                    Int(self.hashes.unsafe_ptr()),
                    Int(order.unsafe_ptr()),
                    Int(ordered.unsafe_ptr()) if with_hashes else 0,
                    fold,
                    next^,
                )
            )
        run_jobs(jobs)
        return Partitioned(order^, starts^, ordered^)


struct _EncodeJob(Job):
    """Encode one bucket's keys with a private dictionary."""

    var keys: List[Series]
    var nulls_equal: Bool
    var ids: List[Int]
    var count: Int

    def __init__(out self, var keys: List[Series], nulls_equal: Bool):
        self.keys = keys^
        self.nulls_equal = nulls_equal
        self.ids = List[Int]()
        self.count = 0

    def run(mut self) raises:
        var keys = encode_rows(self.keys, self.nulls_equal)
        self.count = keys.count()
        swap(self.ids, keys.ids)


struct _PlaceJob(Job):
    """Write one bucket's local ids back to row order, offset to global.

    Buckets hold disjoint rows and disjoint id ranges, so every bucket
    writes its own slots of the shared outputs on its own thread.
    """

    var local: List[Int]
    var lo: Int
    var base: Int
    var order: Int
    var ids: Int
    var representatives: Int

    def __init__(
        out self,
        var local: List[Int],
        lo: Int,
        base: Int,
        order: Int,
        ids: Int,
        representatives: Int,
    ):
        self.local = local^
        self.lo = lo
        self.base = base
        self.order = order
        self.ids = ids
        self.representatives = representatives

    def run(mut self) raises:
        var o = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var out = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.ids)
        var reps = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=self.representatives
        )
        for i in range(len(self.local)):
            var local = self.local[i]
            # -1 marks a null key that must not match; it carries through.
            if local >= 0:
                var row = o.unsafe_offset(self.lo + i)[]
                out.unsafe_offset(row)[] = local + self.base
                # One row per id, for callers that need a key value back.
                reps.unsafe_offset(self.base + local)[] = row


def encode_partitioned(
    keys: List[Series], workers: Int, nulls_equal: Bool
) raises -> RowKeys:
    """Dense ids for distinct key rows, encoded one hash bucket at a time.

    Equal keys share a bucket, so each bucket's private dictionary is
    disjoint from every other's and local ids only need an offset to become
    globally unique. That keeps each dictionary small, which is the whole
    point: a single dictionary over hundreds of thousands of distinct keys
    spends most of its time missing cache.

    Unlike `encode_rows`, the numbering is unspecified rather than
    first-occurrence, so this suits callers that only need equal keys to
    share an id -- a join's row order comes from iterating rows, not from
    the ids. `representatives` still names one row per id.
    """
    var rows = len(keys[0])
    var partitioner = Partitioner(keys, workers)
    var parts = partitioner.scatter(workers)
    var gathered = take_parallel(keys, parts.order.copy(), workers)

    var jobs = List[_EncodeJob]()
    var offsets = List[Int]()
    for b in range(parts.buckets()):
        var lo = parts.bounds[b]
        var hi = parts.bounds[b + 1]
        if hi == lo:
            continue
        var slices = List[Series](capacity=len(gathered))
        for column in gathered:
            slices.append(column.slice(lo, hi - lo))
        jobs.append(_EncodeJob(slices^, nulls_equal))
        offsets.append(lo)
    run_jobs(jobs)

    var ids = List[Int](length=rows, fill=-1)
    var total = 0
    for j in range(len(jobs)):
        total += jobs[j].count
    var representatives = List[Int](length=total, fill=0)
    var places = List[_PlaceJob](capacity=len(jobs))
    var base = 0
    for j in range(len(jobs)):
        var local = List[Int]()
        swap(local, jobs[j].ids)
        places.append(
            _PlaceJob(
                local^,
                offsets[j],
                base,
                Int(parts.order.unsafe_ptr()),
                Int(ids.unsafe_ptr()),
                Int(representatives.unsafe_ptr()),
            )
        )
        base += jobs[j].count
    run_jobs(places)
    _ = parts^
    return RowKeys(ids^, representatives^)


def _same_value(series: Series, a: Int, b: Int) -> Bool:
    """Whether rows a and b hold equal keys, as `encode_rows` compares them:
    nulls are equal to each other, every NaN is one value, -0.0 equals 0.0,
    and strings compare by bytes."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if series._data.isa[Column[Scalar[D]]]():
            ref column = series._data[Column[Scalar[D]]]
            var present = column._valid(a)
            if present != column._valid(b):
                return False
            if not present:
                return True
            comptime if D.is_floating_point():
                return float_key(Float64(column._get(a))) == float_key(
                    Float64(column._get(b))
                )
            else:
                return column._get(a) == column._get(b)
    if series._data.isa[Column[Int128]]():
        ref column = series._data[Column[Int128]]
        var present = column._valid(a)
        if present != column._valid(b):
            return False
        return not present or column._get(a) == column._get(b)
    if series._data.isa[BoolColumn]():
        ref bools = series._data[BoolColumn]
        var present = bools._valid(a)
        if present != bools._valid(b):
            return False
        return not present or bools._get(a) == bools._get(b)
    ref strings = series._data[StringColumn]
    var present = strings._valid(a)
    if present != strings._valid(b):
        return False
    return not present or strings._equal_at_valid(strings, a, b)


def _numeric_key(key: Series) -> Bool:
    """A fixed-width number whose 64-bit hash key `_hash_column` writes
    without folding (every numeric dtype; not Int128 decimals)."""
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        if key._data.isa[Column[Scalar[D]]]():
            return True
    return False


# How one key column is compared on a hash match, resolved once per
# bucket instead of once per row through `_same_value`'s dtype dispatch.
comptime _KEY_GENERIC = 0
comptime _KEY_FIXED = 1
comptime _KEY_FLOAT64 = 2
comptime _KEY_FLOAT32 = 3
comptime _KEY_STRING = 4
comptime _KEY_VIEW = 5


struct _KeyView(Copyable, Movable):
    """One key column as raw addresses for comparing rows (#486).

    `values` is row 0 of a fixed-width payload (`width` bytes an element)
    or of a string column's offsets; `bytes` is a string column's bytes.
    `bits` is the validity bitmap when the column has nulls, with
    `bit_offset` the bit of row 0, and 0 otherwise. A view-backed string
    keeps its long-value buffers' addresses. Resolved once per bucket, so
    confirming a hash match is typed loads of the two rows' keys rather
    than a dtype dispatch per key per row.
    """

    var kind: Int
    var values: Int
    var bytes: Int
    var width: Int
    var bits: Int
    var bit_offset: Int
    # A view-backed string's long-value buffers, by buffer index.
    var buffers: List[Int]

    def __init__(out self, key: Series):
        self.kind = _KEY_GENERIC
        self.values = 0
        self.bytes = 0
        self.width = 0
        self.bits = 0
        self.bit_offset = 0
        self.buffers = List[Int]()
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if key._data.isa[Column[Scalar[D]]]():
                ref column = key._data[Column[Scalar[D]]]
                comptime if D == DType.float64:
                    self.kind = _KEY_FLOAT64
                elif D == DType.float32:
                    self.kind = _KEY_FLOAT32
                else:
                    self.kind = _KEY_FIXED
                self.width = size_of[Scalar[D]]()
                self.values = Int(column.unsafe_values())
                if len(column._bits[]) != 0:
                    self.bits = Int(column._bits[].unsafe_ptr())
                    self.bit_offset = column._offset
                return
        if key._data.isa[Column[Int128]]():
            ref column = key._data[Column[Int128]]
            self.kind = _KEY_FIXED
            self.width = 16
            self.values = Int(column.unsafe_values())
            if len(column._bits[]) != 0:
                self.bits = Int(column._bits[].unsafe_ptr())
                self.bit_offset = column._offset
            return
        if key._data.isa[StringColumn]():
            ref strings = key._data[StringColumn]
            if strings._is_view_storage():
                # Arrow binary views: 16 bytes a row, the length and the
                # first four bytes in the first word, and the rest of a
                # value through 12 bytes (zero padded) or the buffer index
                # and offset of a longer one in the second (TPC-DS q39's
                # w_warehouse_name).
                ref storage = strings._view_storage.value()
                self.kind = _KEY_VIEW
                self.values = Int(
                    storage._views[].unsafe_ptr().unsafe_offset(strings._offset)
                )
                for buffer in storage._buffers[]:
                    self.buffers.append(Int(buffer[].unsafe_ptr()))
                if len(strings._bits[]) != 0:
                    self.bits = Int(strings._bits[].unsafe_ptr())
                    self.bit_offset = strings._offset
                return
            self.kind = _KEY_STRING
            self.values = (
                Int(strings._offsets[].unsafe_ptr()) + 8 * strings._offset
            )
            self.bytes = Int(strings._bytes[].unsafe_ptr())
            if len(strings._bits[]) != 0:
                self.bits = Int(strings._bits[].unsafe_ptr())
                self.bit_offset = strings._offset

    @always_inline
    def _valid(self, row: Int) -> Bool:
        return _validity_at(
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.bits),
            self.bit_offset + row,
        )

    @always_inline
    def _word(self, row: Int) -> UInt64:
        """A fixed-width row's payload as 64 bits (the low half of an
        Int128; `_high` has the rest), or a float's equality key."""
        var base = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.values)
        if self.kind == _KEY_FLOAT64:
            return float_key(
                base.unsafe_offset(8 * row).unsafe_bitcast[Float64]()[]
            )
        if self.kind == _KEY_FLOAT32:
            return float_key(
                Float64(base.unsafe_offset(4 * row).unsafe_bitcast[Float32]()[])
            )
        if self.width == 8 or self.width == 16:
            return base.unsafe_offset(self.width * row).unsafe_bitcast[
                UInt64
            ]()[]
        if self.width == 4:
            return UInt64(
                base.unsafe_offset(4 * row).unsafe_bitcast[UInt32]()[]
            )
        if self.width == 2:
            return UInt64(
                base.unsafe_offset(2 * row).unsafe_bitcast[UInt16]()[]
            )
        return UInt64(base.unsafe_offset(row)[])

    @always_inline
    def _high(self, row: Int) -> UInt64:
        var base = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.values)
        return base.unsafe_offset(16 * row + 8).unsafe_bitcast[UInt64]()[]

    @always_inline
    def _view_words(self, row: Int) -> Tuple[UInt64, UInt64]:
        var at = Pointer[UInt64, MutAnyOrigin](
            unsafe_from_address=self.values
        ).unsafe_offset(2 * row)
        return (at[], at.unsafe_offset(1)[])

    @always_inline
    def _view_bytes(self, word: UInt64) -> Pointer[UInt8, MutAnyOrigin]:
        """Where a long value's bytes start, from its view's second word
        (buffer index in the low half, offset in the high half)."""
        return Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.buffers[Int(word & 0xFFFFFFFF)]
        ).unsafe_offset(Int(word >> 32))

    @always_inline
    def _string_bounds(self, row: Int) -> Tuple[Int, Int]:
        var offsets = Pointer[Int64, MutAnyOrigin](
            unsafe_from_address=self.values
        )
        return (
            Int(offsets.unsafe_offset(row)[]),
            Int(offsets.unsafe_offset(row + 1)[]),
        )

    @always_inline
    def same(self, first: Int, row: Int) -> Bool:
        """Whether rows `first` and `row` hold equal keys (nulls equal each
        other, every NaN one value, -0.0 equal to 0.0, strings by bytes)."""
        if self.bits != 0:
            var present = self._valid(row)
            if present != self._valid(first):
                return False
            if not present:
                return True
        if self.kind == _KEY_VIEW:
            var words = self._view_words(row)
            var other = self._view_words(first)
            if words[0] != other[0]:
                return False
            var length = Int(words[0] & 0xFFFFFFFF)
            if length <= 12:
                return words[1] == other[1]
            # Equal lengths and first four bytes: compare the rest.
            return _same_bytes(
                self._view_bytes(words[1]),
                self._view_bytes(other[1]),
                4,
                length,
            )
        if self.kind == _KEY_STRING:
            var bounds = self._string_bounds(row)
            var other = self._string_bounds(first)
            var length = bounds[1] - bounds[0]
            if length != other[1] - other[0]:
                return False
            var bytes = Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=self.bytes
            )
            return _same_bytes(
                bytes.unsafe_offset(bounds[0]),
                bytes.unsafe_offset(other[0]),
                0,
                length,
            )
        if self._word(row) != self._word(first):
            return False
        return self.width != 16 or self._high(row) == self._high(first)


@always_inline
def _same_bytes(
    a: Pointer[UInt8, MutAnyOrigin],
    b: Pointer[UInt8, MutAnyOrigin],
    start: Int,
    length: Int,
) -> Bool:
    """Whether bytes [start, length) of `a` and `b` are equal, eight at a
    time."""
    var i = start
    while i + 8 <= length:
        if (
            a.unsafe_offset(i).unsafe_bitcast[UInt64]().unsafe_load()
            != b.unsafe_offset(i).unsafe_bitcast[UInt64]().unsafe_load()
        ):
            return False
        i += 8
    while i < length:
        if a.unsafe_offset(i)[] != b.unsafe_offset(i)[]:
            return False
        i += 1
    return True


def encode_bucket(
    keys: List[Series],
    hashes: Span[UInt64, _],
    rows: Span[Int, _],
    mut ids: List[Int],
    mut firsts: List[Int],
    exact_hashes: Bool = False,
):
    """Group ids for one bucket's rows, in first-occurrence order, from the
    key hashes the partitioner already computed (`hashes[p]` belongs to
    `rows[p]`). A hash match is confirmed by comparing the two rows' keys in
    place, so the key columns are never gathered and never hashed again.
    `firsts` gets each group's first row (a row of `keys`, not a bucket
    position)."""
    var m = len(rows)
    ids.resize(m, 0)
    firsts.clear()
    var group_hashes = List[UInt64]()
    var capacity = 1024
    var table = List[Int32](length=capacity, fill=-1)
    var out = ids.unsafe_ptr()
    # One offsets-backed string key without nulls, the common case, is
    # compared directly against the group's first row.
    var direct = (
        len(keys) == 1
        and keys[0]._data.isa[StringColumn]()
        and not keys[0]._data[StringColumn]._is_view_storage()
        and len(keys[0]._data[StringColumn]._bits[]) == 0
    )
    # With `exact_hashes` (the hashes are `_hash_column`'s), one fixed-width
    # numeric key without nulls is hashed by `_mix` alone, a bijection on
    # its 64 bits: equal hashes are equal keys, and the rows need no
    # comparison (no random read of either row's key). Other callers' hashes
    # only find candidates.
    var exact = (
        exact_hashes
        and len(keys) == 1
        and _numeric_key(keys[0])
        and keys[0].null_count() == 0
    )
    var offsets_at = 0
    var bytes_at = 0
    if direct:
        ref strings = keys[0]._data[StringColumn]
        offsets_at = Int(strings._offsets[].unsafe_ptr()) + 8 * strings._offset
        bytes_at = Int(strings._bytes[].unsafe_ptr())
    var offsets = Pointer[Int64, MutAnyOrigin](unsafe_from_address=offsets_at)
    var bytes = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=bytes_at)
    # Every offsets-backed string key, for prefetching its rows ahead.
    var fetch_offsets = List[Int]()
    var fetch_bytes = List[Int]()
    for key in keys:
        if key._data.isa[StringColumn]():
            ref strings = key._data[StringColumn]
            if not strings._is_view_storage():
                fetch_offsets.append(
                    Int(strings._offsets[].unsafe_ptr()) + 8 * strings._offset
                )
                fetch_bytes.append(Int(strings._bytes[].unsafe_ptr()))
    # Every other shape compares through a view of each key resolved once
    # here: typed loads of the row's key against the group's key kept
    # beside the table (#486), or `_same_value` for storage without one.
    var views = List[_KeyView]()
    if not direct and not exact:
        for key in keys:
            views.append(_KeyView(key))
    for p in range(m):
        var row = rows[p]
        var hash = hashes[p]
        # Rows arrive in bucket order, so their strings are scattered:
        # fetch the offsets of a row 16 ahead and the bytes of one 8 ahead
        # (its offsets are in cache by now) while this one probes.
        for k in range(len(fetch_offsets)):
            var key_offsets = Pointer[Int64, MutAnyOrigin](
                unsafe_from_address=fetch_offsets[k]
            )
            if p + 16 < m:
                prefetch(key_offsets.unsafe_offset(rows[p + 16]))
            if p + 8 < m:
                var ahead = Int(key_offsets.unsafe_offset(rows[p + 8])[])
                prefetch(
                    Pointer[UInt8, MutAnyOrigin](
                        unsafe_from_address=fetch_bytes[k]
                    ).unsafe_offset(ahead)
                )
        var mask = capacity - 1
        var slot = Int(hash) & mask
        var cells = table.unsafe_ptr()
        var known = group_hashes.unsafe_ptr()
        var first_rows = firsts.unsafe_ptr()
        while True:
            var g = Int(cells[unsafe_offset=slot])
            if g < 0:
                g = len(firsts)
                cells[unsafe_offset=slot] = Int32(g)
                firsts.append(row)
                group_hashes.append(hash)
                out[unsafe_offset=p] = g
                break
            if known[unsafe_offset=g] == hash:
                var same: Bool
                if exact:
                    same = True
                elif direct:
                    var other = first_rows[unsafe_offset=g]
                    var a = Int(offsets.unsafe_offset(row)[])
                    var length = Int(offsets.unsafe_offset(row + 1)[]) - a
                    var b = Int(offsets.unsafe_offset(other)[])
                    same = length == Int(offsets.unsafe_offset(other + 1)[]) - b
                    # Eight bytes at a time: URLs average about 70 bytes,
                    # and a byte loop was most of ClickBench q33's grouping.
                    var i = 0
                    while same and i + 8 <= length:
                        same = (
                            bytes.unsafe_offset(a + i)
                            .unsafe_bitcast[UInt64]()
                            .unsafe_load()
                            == bytes.unsafe_offset(b + i)
                            .unsafe_bitcast[UInt64]()
                            .unsafe_load()
                        )
                        i += 8
                    while same and i < length:
                        same = (
                            bytes.unsafe_offset(a + i)[]
                            == bytes.unsafe_offset(b + i)[]
                        )
                        i += 1
                else:
                    same = True
                    for k in range(len(views)):
                        if views[k].kind == _KEY_GENERIC:
                            same = _same_value(
                                keys[k], first_rows[unsafe_offset=g], row
                            )
                        else:
                            same = views[k].same(
                                first_rows[unsafe_offset=g], row
                            )
                        if not same:
                            break
                if same:
                    out[unsafe_offset=p] = g
                    break
            slot = (slot + 1) & mask
        if 2 * len(firsts) > capacity:
            # Rehash from the stored group hashes; ids do not change.
            capacity *= 2
            table = List[Int32](length=capacity, fill=-1)
            var grown = table.unsafe_ptr()
            for g in range(len(firsts)):
                var at = Int(group_hashes[g]) & (capacity - 1)
                while grown[unsafe_offset=at] >= 0:
                    at = (at + 1) & (capacity - 1)
                grown[unsafe_offset=at] = Int32(g)
