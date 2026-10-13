"""Window partitions laid out contiguously, as DuckDB's PhysicalWindow lays
out its hash groups: the rows of each partition next to each other, in their
input order, so each partition is evaluated on its own and many partitions
are evaluated at once.

The layout is a stable sort of row positions by partition id, done in two
parallel passes: rows into bins of consecutive ids (each worker counting and
scattering its own row range, so rows stay in input order within a bin),
then a counting sort of each bin by id (bins in parallel). A partition's
rows are then `order[starts[g]:starts[g + 1]]`, and `inverse` maps a row
back to its place, to return results to row order.
"""
from std.memory import Pointer

from .huge_pages import huge_uninit
from .parallel import Job, partitions, run_jobs
from .partition import Partitioner, encode_bucket
from .series import Series

# Id ranges the first pass scatters rows into (DuckDB's window sink uses
# 16 to 256 hash groups).
comptime _BINS = 256


struct PartitionLayout(Movable):
    """Rows grouped by partition id, in input order within a partition."""

    # Each row's partition id, in input order.
    var ids: List[Int]
    var count: Int
    # Row positions grouped by partition, and where each partition starts
    # among them (count + 1 entries).
    var order: List[Int]
    var starts: List[Int]
    # Each row's position in `order`.
    var inverse: List[Int]
    # Whether `order` is every row in input order: the rows already sit
    # partition by partition (a sort on the keys came before), so nothing
    # needs gathering into the layout or back.
    var in_place: Bool

    def __init__(
        out self,
        var ids: List[Int],
        count: Int,
        var order: List[Int],
        var starts: List[Int],
        var inverse: List[Int],
    ):
        self.ids = ids^
        self.count = count
        self.order = order^
        self.starts = starts^
        self.inverse = inverse^
        var p = self.order.unsafe_ptr()
        self.in_place = True
        for k in range(len(self.order)):
            if p[unsafe_offset=k] != k:
                self.in_place = False
                break


@always_inline
def _bin(id: Int, count: Int, bins: Int) -> Int:
    return id * bins // count


@always_inline
def _first_id(bin: Int, count: Int, bins: Int) -> Int:
    """The smallest id in bin `bin` (the inverse of `_bin`)."""
    return (bin * count + bins - 1) // bins


struct _CountJob(Job):
    """Pass 1a: one row range's count of rows per bin."""

    var ids: Int
    var lo: Int
    var hi: Int
    var count: Int
    var bins: Int
    var counts: List[Int]

    def __init__(out self, ids: Int, lo: Int, hi: Int, count: Int, bins: Int):
        self.ids = ids
        self.lo = lo
        self.hi = hi
        self.count = count
        self.bins = bins
        self.counts = List[Int](length=bins, fill=0)

    def run(mut self) raises:
        var ids = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.ids)
        var counts = self.counts.unsafe_ptr()
        for row in range(self.lo, self.hi):
            counts[
                unsafe_offset=_bin(
                    ids[unsafe_offset=row], self.count, self.bins
                )
            ] += 1


struct _ScatterJob(Job):
    """Pass 1b: one row range's rows into their bins, from the place its
    count of each bin begins."""

    var ids: Int
    var binned: Int
    var lo: Int
    var hi: Int
    var count: Int
    var bins: Int
    var at: List[Int]

    def __init__(
        out self,
        ids: Int,
        binned: Int,
        lo: Int,
        hi: Int,
        count: Int,
        bins: Int,
        var at: List[Int],
    ):
        self.ids = ids
        self.binned = binned
        self.lo = lo
        self.hi = hi
        self.count = count
        self.bins = bins
        self.at = at^

    def run(mut self) raises:
        var ids = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.ids)
        var binned = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.binned)
        var at = self.at.unsafe_ptr()
        for row in range(self.lo, self.hi):
            var b = _bin(ids[unsafe_offset=row], self.count, self.bins)
            binned[unsafe_offset=at[unsafe_offset=b]] = row
            at[unsafe_offset=b] += 1


struct _BinJob(Job):
    """Pass 2: one bin's rows by id, counting sort, keeping row order within
    an id; writes the bin's ids' starts."""

    var ids: Int
    var binned: Int
    var order: Int
    var starts: Int
    var lo: Int
    var hi: Int
    var first: Int
    var last: Int

    def __init__(
        out self,
        ids: Int,
        binned: Int,
        order: Int,
        starts: Int,
        lo: Int,
        hi: Int,
        first: Int,
        last: Int,
    ):
        self.ids = ids
        self.binned = binned
        self.order = order
        self.starts = starts
        self.lo = lo
        self.hi = hi
        self.first = first
        self.last = last

    def run(mut self) raises:
        var ids = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.ids)
        var binned = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.binned)
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var starts = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.starts)
        var span = self.last - self.first
        var at = List[Int](length=span + 1, fill=0)
        var place = at.unsafe_ptr()
        for k in range(self.lo, self.hi):
            var id = ids[unsafe_offset=binned[unsafe_offset=k]]
            place[unsafe_offset=id - self.first + 1] += 1
        place[unsafe_offset=0] = self.lo
        for g in range(span):
            place[unsafe_offset=g + 1] += place[unsafe_offset=g]
            starts[unsafe_offset=self.first + g] = place[unsafe_offset=g]
        for k in range(self.lo, self.hi):
            var row = binned[unsafe_offset=k]
            var local = ids[unsafe_offset=row] - self.first
            order[unsafe_offset=place[unsafe_offset=local]] = row
            place[unsafe_offset=local] += 1


struct _InverseJob(Job):
    var order: Int
    var inverse: Int
    var lo: Int
    var hi: Int

    def __init__(out self, order: Int, inverse: Int, lo: Int, hi: Int):
        self.order = order
        self.inverse = inverse
        self.lo = lo
        self.hi = hi

    def run(mut self) raises:
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var inverse = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=self.inverse
        )
        for k in range(self.lo, self.hi):
            inverse[unsafe_offset=order[unsafe_offset=k]] = k


def partition_layout(
    var ids: List[Int], count: Int, workers: Int
) raises -> PartitionLayout:
    """The layout of rows whose partition ids, in [0, count), are `ids`."""
    var n = len(ids)
    var bins = max(1, min(_BINS, count, 4 * workers))
    var lanes = max(1, min(workers, n // 2048))
    var bounds = partitions(n, lanes, 64)
    var ids_at = Int(ids.unsafe_ptr())
    # Pass 1: rows into bins, each range's rows after the earlier ranges'.
    var counting = List[_CountJob](capacity=lanes)
    for w in range(lanes):
        counting.append(
            _CountJob(ids_at, bounds[w], bounds[w + 1], count, bins)
        )
    run_jobs(counting)
    var bin_starts = List[Int](length=bins + 1, fill=0)
    var scattering = List[_ScatterJob](capacity=lanes)
    var running = 0
    var at = List[List[Int]](capacity=lanes)
    for _ in range(lanes):
        at.append(List[Int](length=bins, fill=0))
    for b in range(bins):
        bin_starts[b] = running
        for w in range(lanes):
            at[w][b] = running
            running += counting[w].counts[b]
    bin_starts[bins] = running
    var binned = huge_uninit[Int](n)
    for w in range(lanes):
        scattering.append(
            _ScatterJob(
                ids_at,
                Int(binned.unsafe_ptr()),
                bounds[w],
                bounds[w + 1],
                count,
                bins,
                at[w].copy(),
            )
        )
    run_jobs(scattering)
    # Pass 2: each bin by id.
    var order = huge_uninit[Int](n)
    var starts = List[Int](length=count + 1, fill=n)
    var sorting = List[_BinJob](capacity=bins)
    for b in range(bins):
        sorting.append(
            _BinJob(
                ids_at,
                Int(binned.unsafe_ptr()),
                Int(order.unsafe_ptr()),
                Int(starts.unsafe_ptr()),
                bin_starts[b],
                bin_starts[b + 1],
                _first_id(b, count, bins),
                _first_id(b + 1, count, bins),
            )
        )
    run_jobs(sorting)
    _ = binned^
    var inverse = huge_uninit[Int](n)
    var inverting = List[_InverseJob](capacity=lanes)
    for w in range(lanes):
        inverting.append(
            _InverseJob(
                Int(order.unsafe_ptr()),
                Int(inverse.unsafe_ptr()),
                bounds[w],
                bounds[w + 1],
            )
        )
    run_jobs(inverting)
    return PartitionLayout(ids^, count, order^, starts^, inverse^)


struct _NumberJob(Job):
    """One hash bucket's partitions numbered from the hashes already
    computed, rows compared in place (`encode_bucket`), in the order they
    first appear in the bucket."""

    var keys: List[Series]
    var hashes: Int
    var order: Int
    var lo: Int
    var hi: Int
    var ids: List[Int]
    var count: Int

    def __init__(
        out self, keys: List[Series], hashes: Int, order: Int, lo: Int, hi: Int
    ):
        self.keys = keys.copy()
        self.hashes = hashes
        self.order = order
        self.lo = lo
        self.hi = hi
        self.ids = List[Int]()
        self.count = 0

    def run(mut self) raises:
        ref hashes = Pointer[List[UInt64], MutAnyOrigin](
            unsafe_from_address=self.hashes
        )[]
        ref order = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.order
        )[]
        var firsts = List[Int]()
        encode_bucket(
            self.keys,
            Span(hashes)[self.lo : self.hi],
            Span(order)[self.lo : self.hi],
            self.ids,
            firsts,
            exact_hashes=True,
        )
        self.count = len(firsts)


struct _PlaceJob(Job):
    """One bucket's rows into contiguous partitions: a counting sort of its
    rows by local id, keeping row order; writes each row's global id and
    each partition's start."""

    var local: List[Int]
    var count: Int
    var lo: Int
    var base: Int
    var bucketed: Int
    var ids: Int
    var order: Int
    var starts: Int

    def __init__(
        out self,
        var local: List[Int],
        count: Int,
        lo: Int,
        base: Int,
        bucketed: Int,
        ids: Int,
        order: Int,
        starts: Int,
    ):
        self.local = local^
        self.count = count
        self.lo = lo
        self.base = base
        self.bucketed = bucketed
        self.ids = ids
        self.order = order
        self.starts = starts

    def run(mut self) raises:
        var bucketed = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=self.bucketed
        )
        var ids = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.ids)
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var starts = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.starts)
        var places = List[Int](length=self.count + 1, fill=0)
        var place = places.unsafe_ptr()
        var local = self.local.unsafe_ptr()
        var rows = len(self.local)
        for k in range(rows):
            place[unsafe_offset=local[unsafe_offset=k] + 1] += 1
        place[unsafe_offset=0] = self.lo
        for g in range(self.count):
            place[unsafe_offset=g + 1] += place[unsafe_offset=g]
            starts[unsafe_offset=self.base + g] = place[unsafe_offset=g]
        for k in range(rows):
            var id = local[unsafe_offset=k]
            var row = bucketed[unsafe_offset=self.lo + k]
            ids[unsafe_offset=row] = self.base + id
            order[unsafe_offset=place[unsafe_offset=id]] = row
            place[unsafe_offset=id] += 1


def keyed_layout(keys: List[Series], lanes: Int) raises -> PartitionLayout:
    """The layout of the rows of `keys`' partitions (nulls equal), built as
    DuckDB's window sink builds its hash groups: keys hashed and rows
    scattered into hash buckets in parallel, then each bucket's partitions
    numbered and laid out on their own, buckets in parallel. No global
    numbering pass: on one thread that was most of a window step."""
    var n = len(keys[0])
    var flat = List[Series](capacity=len(keys))
    for key in keys:
        flat.append(key.rechunk() if key.is_chunked() else key.copy())
    var partitioner = Partitioner(flat, lanes)
    var parts = partitioner.scatter(lanes, with_hashes=True)
    var numbering = List[_NumberJob]()
    var lows = List[Int]()
    for b in range(parts.buckets()):
        var lo = parts.bounds[b]
        var hi = parts.bounds[b + 1]
        if hi == lo:
            continue
        numbering.append(
            _NumberJob(
                flat,
                Int(Pointer(to=parts.hashes)),
                Int(Pointer(to=parts.order)),
                lo,
                hi,
            )
        )
        lows.append(lo)
    run_jobs(numbering)
    var count = 0
    for j in range(len(numbering)):
        count += numbering[j].count
    var ids = huge_uninit[Int](n)
    var order = huge_uninit[Int](n)
    var starts = List[Int](length=count + 1, fill=n)
    var placing = List[_PlaceJob](capacity=len(numbering))
    var base = 0
    for j in range(len(numbering)):
        var local = List[Int]()
        swap(local, numbering[j].ids)
        placing.append(
            _PlaceJob(
                local^,
                numbering[j].count,
                lows[j],
                base,
                Int(parts.order.unsafe_ptr()),
                Int(ids.unsafe_ptr()),
                Int(order.unsafe_ptr()),
                Int(starts.unsafe_ptr()),
            )
        )
        base += numbering[j].count
    run_jobs(placing)
    _ = parts^
    var inverse = huge_uninit[Int](n)
    var bounds = partitions(n, lanes, 64)
    var inverting = List[_InverseJob](capacity=lanes)
    for w in range(lanes):
        inverting.append(
            _InverseJob(
                Int(order.unsafe_ptr()),
                Int(inverse.unsafe_ptr()),
                bounds[w],
                bounds[w + 1],
            )
        )
    run_jobs(inverting)
    return PartitionLayout(ids^, count, order^, starts^, inverse^)
