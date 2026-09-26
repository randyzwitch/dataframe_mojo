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
from std.collections import Dict

from .aggregate import float_key
from .bool_column import BoolColumn
from .column import Column
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
    """Hash short strings by one packed word, longer strings by FNV-1a."""
    if len(bytes) <= 8:
        var word = UInt64(0)
        for k in range(len(bytes)):
            word |= UInt64(bytes[k]) << UInt64(k * 8)
        return _short_hash(word, len(bytes))
    var h = UInt64(0xCBF29CE484222325)
    for k in range(len(bytes)):
        h = (h ^ UInt64(bytes[k])) * 0x100000001B3
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
        ref data = strings._bytes[]
        for i in range(start, end):
            if not strings._valid(i):
                write(i, _NULL_KEY)
                continue
            var byte_start = strings._start(i)
            var length = strings._end(i) - byte_start
            if length <= 8 and byte_start <= len(data) - 8:
                var word = bitcast[DType.uint64, 1](
                    data.unsafe_ptr()
                    .unsafe_offset(byte_start)
                    .unsafe_load[width=8]()
                )
                write(i, _short_hash(word, length))
            else:
                write(i, _hash_bytes(strings._get(i).as_bytes()))
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
    var fold: Int
    var next: List[Int]

    def __init__(
        out self,
        start: Int,
        end: Int,
        hashes: Int,
        order: Int,
        fold: Int,
        var next: List[Int],
    ):
        self.start = start
        self.end = end
        self.hashes = hashes
        self.order = order
        self.fold = fold
        self.next = next^

    def run(mut self) raises:
        var h = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=self.hashes)
        var o = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        for i in range(self.start, self.end):
            var slot = Int(h.unsafe_offset(i)[] >> UInt64(_SLOT_SHIFT))
            var bucket = slot >> self.fold
            o.unsafe_offset(self.next[bucket])[] = i
            self.next[bucket] += 1


@fieldwise_init
struct Partitioned(Movable):
    """A stable permutation of row indices grouped by hash bucket.

    Rows of bucket b are `order[bounds[b] : bounds[b + 1]]`, in input order.
    """

    var order: List[Int]
    var bounds: List[Int]

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
    for key in keys:
        if key.is_chunked():
            return _low_cardinality_chunked(keys)
    var rows = len(keys[0])
    if rows == 0:
        return True
    var sample = min(rows, _SAMPLE_ROWS)
    var stride = max(1, rows // sample)
    var hash = List[UInt64](length=1, fill=0)
    var address = Int(hash.unsafe_ptr())
    var seen = List[Bool](length=_SLOTS, fill=False)
    var counts = List[Int](length=_SLOTS, fill=0)
    var hashes = List[UInt64](capacity=sample)
    var occupied = 0
    var taken = 0
    var i = 0
    while i < rows and taken < sample:
        for j in range(len(keys)):
            _hash_column(keys[j], i, i + 1, address, j == 0, output_offset=i)
        var slot = Int(hash[0] >> UInt64(_SLOT_SHIFT))
        counts[slot] += 1
        hashes.append(hash[0])
        if not seen[slot]:
            seen[slot] = True
            occupied += 1
        taken += 1
        i = taken * stride + (taken * 7919) % stride
    return _prefer_whole_sample(occupied, taken, counts, hashes)


def _low_cardinality_chunked(keys: List[Series]) raises -> Bool:
    """Sample source chunks without materializing whole key columns."""
    var rows = len(keys[0])
    if rows == 0:
        return True
    var sample = min(rows, _SAMPLE_ROWS)
    var stride = max(1, rows // sample)
    var parts = List[List[Series]](capacity=len(keys))
    var indexes = List[Int](length=len(keys), fill=0)
    var ends = List[Int](capacity=len(keys))
    for key in keys:
        var chunks = key.chunks()
        ends.append(len(chunks[0]))
        parts.append(chunks^)
    var hash = List[UInt64](length=1, fill=0)
    var address = Int(hash.unsafe_ptr())
    var seen = List[Bool](length=_SLOTS, fill=False)
    var counts = List[Int](length=_SLOTS, fill=0)
    var hashes = List[UInt64](capacity=sample)
    var occupied = 0
    var taken = 0
    var i = 0
    while i < rows and taken < sample:
        for j in range(len(keys)):
            while i >= ends[j]:
                indexes[j] += 1
                ends[j] += len(parts[j][indexes[j]])
            var part = parts[j][indexes[j]].copy()
            var local = i - (ends[j] - len(part))
            _hash_column(
                part,
                local,
                local + 1,
                address,
                j == 0,
                output_offset=local,
            )
        var slot = Int(hash[0] >> UInt64(_SLOT_SHIFT))
        counts[slot] += 1
        hashes.append(hash[0])
        if not seen[slot]:
            seen[slot] = True
            occupied += 1
        taken += 1
        i = taken * stride + (taken * 7919) % stride
    return _prefer_whole_sample(occupied, taken, counts, hashes)


struct Partitioner(Movable):
    """One hash pass over the keys, plus the histogram that decides whether
    scattering is worth it."""

    var hashes: List[UInt64]
    var histogram: List[Int]
    var worker_histograms: List[List[Int]]
    var rows: Int

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
        self.hashes = List[UInt64](length=self.rows, fill=0)
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

    def scatter(mut self, workers: Int) raises -> Partitioned:
        """Build the stable permutation, folding slots into buckets."""
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
        var order = List[Int](length=self.rows, fill=0)
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
                    fold,
                    next^,
                )
            )
        run_jobs(jobs)
        return Partitioned(order^, starts^)


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
        self.ids = keys.ids.copy()
        self.count = keys.count()


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
    var representatives = List[Int]()
    var base = 0
    for j in range(len(jobs)):
        var lo = offsets[j]
        for i in range(len(jobs[j].ids)):
            var local = jobs[j].ids[i]
            # -1 marks a null key that must not match; it carries through.
            if local >= 0:
                ids[parts.order[lo + i]] = local + base
        for _ in range(jobs[j].count):
            representatives.append(0)
        # One row per id, for callers that need a key value back.
        for i in range(len(jobs[j].ids)):
            var local = jobs[j].ids[i]
            if local >= 0:
                representatives[base + local] = parts.order[lo + i]
        base += jobs[j].count
    return RowKeys(ids^, representatives^)
