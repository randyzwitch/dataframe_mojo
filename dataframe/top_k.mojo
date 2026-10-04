"""Filters that keep each partition's first k rows by ordinal rank.

`col(v).rank("ordinal", descending=...).over(keys) <= k` keeps, in every
partition, the k rows whose (value, row) come first, the "top N per group"
idiom. Ranking every row sorts every partition and scatters the whole
column; here each partition only keeps its k best rows while the rows are
scanned once, and the filter's mask marks them. A row whose value is null
has a null rank and is not kept, as the comparison would have it.

Partitions are numbered as `over()` numbers them. When groups * k fits
within N candidates, workers use row-local heaps if all replicas still fit
that bound, or route rows to group owners otherwise. Larger requested state
uses bucketed groups and one reusable heap per worker. Both strategies use
O(N + groups) scratch independent of groups * k, and visit each input row
once per pass. The order is `rank`'s own key (`rank._key`), so NaN and -0.0 rank
as they do there, and ties keep the earlier row, as ordinal rank does.
"""
from std.collections import Optional
from std.memory import Pointer

from .binding import BoundExpr
from .bool_column import BoolColumn
from .column import Column
from .dtype import NUMERIC_DTYPES
from .expr import COL, LE, LIT_INT, LT, OVER, RANK, SEP
from .hashing import encode_rows, encode_rows_parallel
from .parallel import Job, run_jobs, worker_count
from .partition import encode_partitioned, low_cardinality
from .rank import _key, _pack
from .series import Series


@always_inline
def _keep_best(
    heap: Pointer[UInt128, MutAnyOrigin], k: Int, count: Int, item: UInt128
) -> Int:
    """Insert into a bounded max-heap; return its new length."""
    if count < k:
        var at = count
        while at > 0:
            var parent = (at - 1) // 2
            if heap[unsafe_offset=parent] >= item:
                break
            heap[unsafe_offset=at] = heap[unsafe_offset=parent]
            at = parent
        heap[unsafe_offset=at] = item
        return count + 1
    if item < heap[unsafe_offset=0]:
        var at = 0
        while at < count // 2:
            var child = at * 2 + 1
            if (
                child + 1 < count
                and heap[unsafe_offset=child + 1] > heap[unsafe_offset=child]
            ):
                child += 1
            if heap[unsafe_offset=child] <= item:
                break
            heap[unsafe_offset=at] = heap[unsafe_offset=child]
            at = child
        heap[unsafe_offset=at] = item
    return count


@fieldwise_init
struct _LocalTopKJob[D: DType](Job):
    """Scan a disjoint row range into private, bounded per-group heaps."""

    var values: Int
    var nulls: Bool
    var ids: Int
    var members: Int
    var group_offset: Int
    var first: Int
    var last: Int
    var k: Int
    var descending: Bool
    var heaps: Int
    var counts: Int

    def run(mut self) raises:
        ref column = Pointer[Column[Scalar[Self.D]], MutAnyOrigin](
            unsafe_from_address=self.values
        )[]
        var ids = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.ids)
        var heaps = Pointer[UInt128, MutAnyOrigin](
            unsafe_from_address=self.heaps
        )
        var counts = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.counts)
        var data = column._ptr()
        var members = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=self.members
        )
        for p in range(self.first, self.last):
            var row = p if self.members == 0 else members[unsafe_offset=p]
            if self.nulls and not column._valid(row):
                continue
            var g = ids[unsafe_offset=row] - self.group_offset
            var item = _pack(
                _key[Self.D](data[unsafe_offset=row], self.descending), row
            )
            counts[unsafe_offset=g] = _keep_best(
                heaps.unsafe_offset(g * self.k),
                self.k,
                counts[unsafe_offset=g],
                item,
            )


@fieldwise_init
struct _MergeTopKJob(Job):
    """Merge local candidates, visiting each candidate once."""

    var heaps: Int
    var counts: Int
    var groups: Int
    var workers: Int
    var first: Int
    var last: Int
    var k: Int
    var mask: Int

    def run(mut self) raises:
        var heaps = Pointer[UInt128, MutAnyOrigin](
            unsafe_from_address=self.heaps
        )
        var counts = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.counts)
        var mask = Pointer[Bool, MutAnyOrigin](unsafe_from_address=self.mask)
        if self.workers == 1:
            for g in range(self.first, self.last):
                for p in range(counts[unsafe_offset=g]):
                    var item = heaps[unsafe_offset=g * self.k + p]
                    mask[unsafe_offset=Int(item & UInt128(UInt64.MAX))] = True
            return
        var heap = List[UInt128](length=self.k, fill=0)
        for g in range(self.first, self.last):
            var count = 0
            for w in range(self.workers):
                var slot = w * self.groups + g
                for p in range(counts[unsafe_offset=slot]):
                    count = _keep_best(
                        heap.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                        self.k,
                        count,
                        heaps[unsafe_offset=slot * self.k + p],
                    )
            for p in range(count):
                mask[unsafe_offset=Int(heap[p] & UInt128(UInt64.MAX))] = True


@fieldwise_init
struct _TopKJob[D: DType](Job):
    """Visit only this worker's partitions; reuse one bounded heap."""

    var values: Int
    var nulls: Bool
    var members: Int
    var starts: Int
    var lo: Int
    var hi: Int
    var k: Int
    var descending: Bool
    var mask: Int

    def run(mut self) raises:
        ref column = Pointer[Column[Scalar[Self.D]], MutAnyOrigin](
            unsafe_from_address=self.values
        )[]
        var members = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=self.members
        )
        var starts = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.starts)
        var mask = Pointer[Bool, MutAnyOrigin](unsafe_from_address=self.mask)
        var capacity = 0
        for g in range(self.lo, self.hi):
            var size = starts[unsafe_offset=g + 1] - starts[unsafe_offset=g]
            if self.k < size:
                capacity = max(capacity, self.k)
        var heap = List[UInt128](length=capacity, fill=0)
        var data = column._ptr()
        for g in range(self.lo, self.hi):
            var first = starts[unsafe_offset=g]
            var last = starts[unsafe_offset=g + 1]
            # No ordering is necessary when every valid row qualifies.
            if self.k >= last - first:
                for p in range(first, last):
                    var row = members[unsafe_offset=p]
                    if not self.nulls or column._valid(row):
                        mask[unsafe_offset=row] = True
                continue
            var count = 0
            for p in range(first, last):
                var row = members[unsafe_offset=p]
                if self.nulls and not column._valid(row):
                    continue
                var item = _pack(
                    _key[Self.D](data[unsafe_offset=row], self.descending), row
                )
                count = _keep_best(
                    heap.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                    self.k,
                    count,
                    item,
                )
            for p in range(count):
                mask[unsafe_offset=Int(heap[p] & UInt128(UInt64.MAX))] = True


def _pattern(bound: BoundExpr) -> Tuple[Int, Int, Int]:
    """(column node, over node, k) of `rank(ordinal).over(...) <= k` (or
    `< k + 1`), or (-1, -1, 0)."""
    ref nodes = bound.expr._nodes
    var miss = (-1, -1, 0)
    var root = len(nodes) - 1
    if root < 0:
        return miss
    ref compare = nodes[root]
    if compare.op != LE and compare.op != LT:
        return miss
    if compare.left < 0 or compare.right < 0:
        return miss
    ref limit = nodes[compare.right]
    if limit.op != LIT_INT:
        return miss
    ref over = nodes[compare.left]
    if over.op != OVER or over.left < 0:
        return miss
    ref rank = nodes[over.left]
    if rank.op != RANK or rank.text != "ordinal" or rank.left < 0:
        return miss
    if nodes[rank.left].op != COL:
        return miss
    # Negative limits cannot keep a positive ordinal rank. Clamp before
    # subtracting so `< Int64.MIN` cannot wrap to a huge positive k.
    var k = max(0, Int(limit.integer))
    if compare.op == LT and k > 0:
        k -= 1
    return (rank.left, compare.left, k)


def top_k_mask(
    bound: BoundExpr, columns: List[Series], height: Int
) raises -> Optional[BoolColumn]:
    """The filter mask of a top-k-per-partition predicate, or None when
    the predicate has another shape."""
    var found = _pattern(bound)
    var column_node = found[0]
    if column_node < 0:
        return None
    ref nodes = bound.expr._nodes
    var k = found[2]
    var value = columns[bound.sources[column_node]].copy()
    var descending = nodes[nodes[found[1]].left].min_count == 1
    var keys = List[Series]()
    for name in nodes[found[1]].text2.split(SEP):
        var matched = False
        for column in columns:
            if column.name() == String(name):
                keys.append(column.copy())
                matched = True
        if not matched:
            return None
    var mask = List[Bool](length=height, fill=False)
    if k <= 0 or height == 0:
        return BoolColumn(mask^)
    var whole = value.rechunk() if value.is_chunked() else value.copy()
    var workers = worker_count(height)
    var partitions = encode_rows(keys, nulls_equal=True) if workers <= 1 else (
        encode_rows_parallel(keys, True, workers) if low_cardinality(
            keys
        ) else encode_partitioned(keys, workers, nulls_equal=True)
    )
    var count = partitions.count()
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        if whole._data.isa[Column[Scalar[D]]]():
            ref data = whole._data[Column[Scalar[D]]]
            var nulls = data.null_count() > 0
            # At most one candidate slot per input row. Replicate group
            # heaps for row-sharded workers only if all copies still fit;
            # otherwise route rows to group owners with O(workers) counters.
            # Division checks the bound before any groups*k multiplication.
            if count > 0 and k <= height // count:
                var copies = workers if k <= height // count // workers else 1
                var slots = count * copies
                var candidates = slots * k
                # Local heaps + counts + merge heaps need <= 40*N bytes.
                # Ownership routing needs <= 32*N + 16*(workers+1), itself
                # bounded by 48*N+16. Check before allocating these arrays.
                if height > (Int.MAX - 16) // 48:
                    raise Error("grouped top-k scratch size overflow")
                var heaps = List[UInt128](length=candidates, fill=0)
                var counts = List[Int](length=slots, fill=0)
                var shares = min(workers, count)
                var width = count // shares + Int(count % shares != 0)
                var members = List[Int]()
                var starts = List[Int]()
                if copies == 1 and workers > 1:
                    starts = List[Int](length=shares + 1, fill=0)
                    for g in partitions.ids:
                        starts[g // width + 1] += 1
                    for w in range(shares):
                        starts[w + 1] += starts[w]
                    var next = starts.copy()
                    members = List[Int](length=height, fill=0)
                    for row in range(height):
                        var owner = partitions.ids[row] // width
                        members[next[owner]] = row
                        next[owner] += 1
                var local = List[_LocalTopKJob[D]](capacity=workers)
                var jobs = workers if copies > 1 else shares
                for w in range(jobs):
                    var indirect = len(members) > 0
                    var group_offset = min(count, w * width) if indirect else 0
                    var slot = w * count if copies > 1 else group_offset
                    var first = starts[w] if indirect else height // jobs * w
                    var last = starts[w + 1] if indirect else (
                        height if w == jobs - 1 else height // jobs * (w + 1)
                    )
                    local.append(
                        _LocalTopKJob[D](
                            Int(Pointer(to=data)),
                            nulls,
                            Int(partitions.ids.unsafe_ptr()),
                            Int(members.unsafe_ptr()) if indirect else 0,
                            group_offset,
                            first,
                            last,
                            k,
                            descending,
                            Int(heaps.unsafe_ptr().unsafe_offset(slot * k)),
                            Int(counts.unsafe_ptr().unsafe_offset(slot)),
                        )
                    )
                run_jobs(local)
                _ = members^
                _ = starts^
                var merge = List[_MergeTopKJob](capacity=shares)
                for w in range(shares):
                    merge.append(
                        _MergeTopKJob(
                            Int(heaps.unsafe_ptr()),
                            Int(counts.unsafe_ptr()),
                            count,
                            copies,
                            count // shares * w,
                            count if w
                            == shares - 1 else count // shares * (w + 1),
                            k,
                            Int(mask.unsafe_ptr()),
                        )
                    )
                run_jobs(merge)
                _ = heaps^
                _ = counts^
                _ = partitions^
                _ = whole^
                return BoolColumn(mask^)
            # Bucket indices once; workers only visit their own groups.
            # Members use N slots, starts/cursors use G+1 slots each, and
            # all worker heaps together use <= N UInt128 slots (each heap
            # is bounded by its worker's largest group).
            # Check the linear scratch bound before any of these allocations.
            if count == Int.MAX or count + 1 > Int.MAX // 16:
                raise Error("grouped top-k scratch size overflow")
            var group_bytes = (count + 1) * 16
            if height > (Int.MAX - group_bytes) // 24:
                raise Error("grouped top-k scratch size overflow")
            var starts = List[Int](length=count + 1, fill=0)
            for id in partitions.ids:
                starts[id + 1] += 1
            for g in range(count):
                starts[g + 1] += starts[g]
            var next = starts.copy()
            var members = List[Int](length=height, fill=0)
            for row in range(height):
                var g = partitions.ids[row]
                members[next[g]] = row
                next[g] += 1
            _ = next^
            var shares = max(1, min(workers, count))
            var jobs = List[_TopKJob[D]](capacity=shares)
            var first = 0
            for w in range(shares):
                # Whole groups stay on one worker; balance by rows rather
                # than group count when groups are skewed.
                var target = height // shares * (w + 1)
                var last = first
                while last < count and (
                    starts[last] < target or w == shares - 1
                ):
                    last += 1
                if last > first:
                    jobs.append(
                        _TopKJob[D](
                            Int(Pointer(to=data)),
                            nulls,
                            Int(members.unsafe_ptr()),
                            Int(starts.unsafe_ptr()),
                            first,
                            last,
                            k,
                            descending,
                            Int(mask.unsafe_ptr()),
                        )
                    )
                first = last
            run_jobs(jobs)
            _ = members^
            _ = starts^
            _ = partitions^
            _ = whole^
            return BoolColumn(mask^)
    return None
