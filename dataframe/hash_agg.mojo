"""Grouped reductions over a high-cardinality key, aggregated in place.

The partitioned group-by (`group_by.partitioned`) scatters row numbers by key
hash and gathers every aggregated column into bucket order before reducing
it: on H2O's group-by queries that gather was a third of the time. DuckDB
updates each group's state from the rows as they arrive instead
(`radix_partitioned_hashtable.cpp`), and this does the same, in two phases:

1. Each worker numbers its own row range with an open-addressing table and
   reduces the range's values in row order through `Reducer.update`.
2. The workers' groups are split by hash into parts; each part merges its
   groups across workers on its own thread, combining states directly.

Groups from earlier ranges are merged first, so each group keeps its first
row and first/tie semantics match a single pass. The first version serves
one Int64 key without nulls and SUM, MEAN, MIN, MAX, COUNT and LEN of Int64
or Float64 columns; anything else returns None and takes the other paths.
"""
from std.collections import Optional
from std.memory import Pointer, bitcast

from .aggregate import Reducer
from .binding import BoundExpr
from .column import Column
from .dtype import DataType
from .execution import _new_reducer
from .expr import COL, COUNT, LEN, MAX, MEAN, MIN, SUM
from .parallel import Job, partitions, run_jobs
from .partition import _mix
from .reductions import FloatSumState, IntSumState
from .series import Series


def _supported(bound: BoundExpr, columns: List[Series]) -> Bool:
    """A reduction of one Int64 or Float64 column that this module merges."""
    ref nodes = bound.expr._nodes
    if len(nodes) != 2:
        return False
    ref node = nodes[1]
    if node.op not in [SUM, MEAN, MIN, MAX, COUNT, LEN]:
        return False
    if nodes[node.left].op != COL:
        return False
    var dtype = columns[bound.sources[node.left]].dtype()
    return dtype == DataType.INT64 or dtype == DataType.FLOAT64


def _moderate_groups(key: Series) raises -> Bool:
    """Whether the key's groups are few enough for per-range tables to
    shrink the data: under 85% of 65,536 stratified sample rows distinct.
    With millions of groups a range holds nearly as
    many groups as rows, the merge re-inserts almost every row, and hash
    partitioning first is faster (H2O q5 at 1M groups: 292 against 168 ms).
    """
    var n = len(key)
    var sample = min(n, 65536)
    if sample == 0:
        return False
    # A chunked key's merged array is cached (#419): later steps reuse it.
    var whole = key.rechunk() if key.is_chunked() else key.copy()
    var values = whole._data[Column[Int64]]._ptr()
    var capacity = 1 << 17
    var table = List[UInt64](length=capacity, fill=0)
    var used = List[Bool](length=capacity, fill=False)
    var cells = table.unsafe_ptr()
    var occupied = used.unsafe_ptr()
    var distinct = 0
    var first = 0
    var cutoff = 85 * sample
    for k in range(sample):
        # One row from every stratum, with a reproducible mixed offset.
        # A fixed offset aliases periodic keys (including two repeated
        # half-frames), understating the number of groups. Multiply-high
        # maps the mixed index into [0, last - first) without a division.
        var last = (k + 1) * n // sample
        var offset = Int(
            (UInt128(_mix(UInt64(k))) * UInt128(last - first)) >> UInt128(64)
        )
        var hash = _mix(
            bitcast[DType.uint64](values[unsafe_offset=first + offset])
        )
        first = last
        var slot = Int(hash) & (capacity - 1)
        while (
            occupied[unsafe_offset=slot] and cells[unsafe_offset=slot] != hash
        ):
            slot = (slot + 1) & (capacity - 1)
        if not occupied[unsafe_offset=slot]:
            occupied[unsafe_offset=slot] = True
            cells[unsafe_offset=slot] = hash
            distinct += 1
        # Both bounds prove the result of the complete bounded sample:
        # distinct never decreases, and every remaining row adds at most
        # one group. Low-cardinality controls need not scan all 65K rows.
        if 100 * distinct >= cutoff:
            return False
        if 100 * (distinct + sample - k - 1) < cutoff:
            return True
    return 100 * distinct < cutoff


def hash_agg_eligible(
    key: Series, bound: List[BoundExpr], columns: List[Series]
) raises -> Bool:
    if key.dtype() != DataType.INT64:
        return False
    if len(bound) == 0:
        return False
    for expression in bound:
        if not _supported(expression, columns):
            return False
    return _moderate_groups(key)


struct _RangeTableJob(Job):
    """Phase 1: number rows [start, end) and reduce their values."""

    var key: Series
    var values: List[Series]
    var bound: List[BoundExpr]
    var start: Int
    var end: Int
    var keys: List[Int64]
    var hashes: List[UInt64]
    var firsts: List[Int]
    var reducers: List[Reducer]
    # The range's null-key group, -1 when it has no null key. Null keys are
    # one group, kept out of the hash table (whose hashes are values').
    var null_group: Int
    var bits: Int
    # This range's groups by hash part, each in first-row order.
    var members: List[List[Int]]

    def __init__(
        out self,
        key: Series,
        values: List[Series],
        bound: List[BoundExpr],
        start: Int,
        end: Int,
        bits: Int,
    ):
        self.key = key.copy()
        self.values = values.copy()
        self.bound = bound.copy()
        self.start = start
        self.end = end
        self.keys = List[Int64]()
        self.hashes = List[UInt64]()
        self.firsts = List[Int]()
        self.reducers = List[Reducer]()
        self.null_group = -1
        self.bits = bits
        self.members = List[List[Int]]()

    def run(mut self) raises:
        var n = self.end - self.start
        var key = self.key.slice(self.start, n)
        if key.is_chunked():
            key = key.rechunk()
        ref column = key._data[Column[Int64]]
        var values = column._ptr()
        var nulls = column.null_count() > 0
        var ids = List[Int](unsafe_uninit_length=n)
        var out = ids.unsafe_ptr()
        var capacity = 1024
        var table = List[Int32](length=capacity, fill=-1)
        for i in range(n):
            if nulls and not column._valid(i):
                if self.null_group < 0:
                    self.null_group = len(self.keys)
                    self.keys.append(0)
                    self.hashes.append(0)
                    self.firsts.append(self.start + i)
                out[unsafe_offset=i] = self.null_group
                continue
            var bits = bitcast[DType.uint64](values[unsafe_offset=i])
            var hash = _mix(bits)
            var mask = capacity - 1
            var slot = Int(hash) & mask
            var cells = table.unsafe_ptr()
            while True:
                var g = Int(cells[unsafe_offset=slot])
                if g < 0:
                    g = len(self.keys)
                    cells[unsafe_offset=slot] = Int32(g)
                    self.keys.append(values[unsafe_offset=i])
                    self.hashes.append(hash)
                    self.firsts.append(self.start + i)
                    out[unsafe_offset=i] = g
                    break
                # `_mix` is a bijection: equal hashes are equal keys.
                if self.hashes[g] == hash and g != self.null_group:
                    out[unsafe_offset=i] = g
                    break
                slot = (slot + 1) & mask
            if 2 * len(self.keys) > capacity:
                capacity *= 2
                table = List[Int32](length=capacity, fill=-1)
                var grown = table.unsafe_ptr()
                for g in range(len(self.keys)):
                    if g == self.null_group:
                        continue
                    var at = Int(self.hashes[g]) & (capacity - 1)
                    while grown[unsafe_offset=at] >= 0:
                        at = (at + 1) & (capacity - 1)
                    grown[unsafe_offset=at] = Int32(g)
        var groups = len(self.keys)
        var parts = 1 << self.bits
        var shift = UInt64(64 - self.bits)
        self.members = List[List[Int]](length=parts, fill=List[Int]())
        for g in range(groups):
            if g != self.null_group:
                self.members[Int(self.hashes[g] >> shift)].append(g)
        for j in range(len(self.bound)):
            ref node = self.bound[j].expr._nodes[1]
            var reducer = _new_reducer(self.bound[j], node, groups)
            reducer.update(self.values[j].slice(self.start, n), 0, True, ids)
            self.reducers.append(reducer^)


def _combine(mut into: Reducer, f: Int, other: Reducer, g: Int):
    """Fold group g of `other` into group f of `into`, for the operations
    `_supported` admits; the same rules as `Reducer.merge`."""
    var op = into.op
    if op == COUNT or op == LEN:
        into.counts[f] += other.counts[g]
    elif op == SUM or op == MEAN:
        if into.dtype == DataType.INT64:
            into.int_sums[f].merge(other.int_sums[g])
        else:
            into.float_sums[f].merge(other.float_sums[g])
    else:
        var is_max = op == MAX
        if into.dtype == DataType.FLOAT64:
            into.nan_seen[f] = into.nan_seen[f] or other.nan_seen[g]
        if not other.seen[g]:
            return
        var take = not into.seen[f]
        if not take:
            if into.dtype == DataType.INT64:
                take = (
                    other.ints[g]
                    > into.ints[f] if is_max else other.ints[g]
                    < into.ints[f]
                )
            else:
                take = (
                    other.floats[g]
                    > into.floats[f] if is_max else other.floats[g]
                    < into.floats[f]
                )
        if take:
            into.seen[f] = True
            if into.dtype == DataType.INT64:
                into.ints[f] = other.ints[g]
            else:
                into.floats[f] = other.floats[g]


def _combine_all(
    mut into: Reducer, other: Reducer, targets: List[Int], sources: List[Int]
):
    """`_combine` for every (target, source) pair, with the operation and
    state type chosen once and states read through pointers: the generic
    per-group merge was most of the time at millions of groups."""
    var n = len(sources)
    var to = targets.unsafe_ptr()
    var from_ = sources.unsafe_ptr()
    var op = into.op
    if op == COUNT or op == LEN:
        var a = into.counts.unsafe_ptr()
        var b = other.counts.unsafe_ptr()
        for k in range(n):
            a[unsafe_offset=to[unsafe_offset=k]] += b[
                unsafe_offset=from_[unsafe_offset=k]
            ]
        return
    if op == SUM or op == MEAN:
        if into.dtype == DataType.INT64:
            var a = into.int_sums.unsafe_ptr()
            var b = other.int_sums.unsafe_ptr()
            for k in range(n):
                ref x = a[unsafe_offset=to[unsafe_offset=k]]
                ref y = b[unsafe_offset=from_[unsafe_offset=k]]
                x.total += y.total
                x.count += y.count
        else:
            var a = into.float_sums.unsafe_ptr()
            var b = other.float_sums.unsafe_ptr()
            for k in range(n):
                ref x = a[unsafe_offset=to[unsafe_offset=k]]
                ref y = b[unsafe_offset=from_[unsafe_offset=k]]
                x.total += y.total
                x.count += y.count
        return
    for k in range(n):
        _combine(into, to[unsafe_offset=k], other, from_[unsafe_offset=k])


struct _PartMergeJob(Job):
    """Phase 2: merge one hash part's groups from every range."""

    var ranges: Int
    var part: Int
    var bound: List[BoundExpr]
    var reducers: List[Reducer]
    var firsts: List[Int]

    def __init__(out self, ranges: Int, part: Int, bound: List[BoundExpr]):
        self.ranges = ranges
        self.part = part
        self.bound = bound.copy()
        self.reducers = List[Reducer]()
        self.firsts = List[Int]()

    def run(mut self) raises:
        ref jobs = Pointer[List[_RangeTableJob], MutAnyOrigin](
            unsafe_from_address=self.ranges
        )[]
        # This part's groups, range by range, in each range's order.
        var members = List[List[Int]](capacity=len(jobs))
        var total = 0
        for w in range(len(jobs)):
            members.append(jobs[w].members[self.part].copy())
            total += len(members[w])
        var capacity = 16
        while capacity < 2 * total:
            capacity *= 2
        var mask = capacity - 1
        var table = List[Int32](length=capacity, fill=-1)
        var final_hashes = List[UInt64](capacity=total)
        # (range, group) of each final group's first appearance.
        var origin_range = List[Int](capacity=total)
        var origin_group = List[Int](capacity=total)
        var maps = List[List[Int]](capacity=len(jobs))
        for w in range(len(jobs)):
            ref hashes = jobs[w].hashes
            var map = List[Int](capacity=len(members[w]))
            for g in members[w]:
                var hash = hashes[g]
                var slot = Int(hash) & mask
                while True:
                    var f = Int(table[slot])
                    if f < 0:
                        f = len(final_hashes)
                        table[slot] = Int32(f)
                        final_hashes.append(hash)
                        origin_range.append(w)
                        origin_group.append(g)
                        map.append(f)
                        break
                    if final_hashes[f] == hash:
                        map.append(f)
                        break
                    slot = (slot + 1) & mask
            maps.append(map^)
        # Part 0 also holds the null-key group, merged from every range.
        var null_final = -1
        if self.part == 0:
            for w in range(len(jobs)):
                var g = jobs[w].null_group
                if g < 0:
                    continue
                if null_final < 0:
                    null_final = len(final_hashes)
                    final_hashes.append(0)
                    origin_range.append(w)
                    origin_group.append(g)
                members[w].append(g)
                maps[w].append(null_final)
        var count = len(final_hashes)
        for f in range(count):
            self.firsts.append(jobs[origin_range[f]].firsts[origin_group[f]])
        for j in range(len(self.bound)):
            ref node = self.bound[j].expr._nodes[1]
            var into = _new_reducer(self.bound[j], node, count)
            for w in range(len(jobs)):
                _combine_all(into, jobs[w].reducers[j], maps[w], members[w])
            self.reducers.append(into^)


def hash_aggregate(
    key: Series,
    bound: List[BoundExpr],
    columns: List[Series],
    workers: Int,
) raises -> Tuple[List[Int], List[List[Series]]]:
    """Each part's first rows (one per group) and finished aggregate
    columns, parts in hash order and groups within a part in first-row
    order. The caller takes the key column at the first rows."""
    var height = len(key)
    var bounds = partitions(height, workers, 64)
    var values = List[Series](capacity=len(bound))
    for expression in bound:
        values.append(
            columns[expression.sources[expression.expr._nodes[1].left]].copy()
        )
    var bits = 1
    while (1 << bits) < 4 * workers:
        bits += 1
    var jobs = List[_RangeTableJob](capacity=workers)
    for w in range(workers):
        if bounds[w + 1] > bounds[w]:
            jobs.append(
                _RangeTableJob(
                    key, values, bound, bounds[w], bounds[w + 1], bits
                )
            )
    run_jobs(jobs)
    var merges = List[_PartMergeJob](capacity=1 << bits)
    for p in range(1 << bits):
        merges.append(_PartMergeJob(Int(Pointer(to=jobs)), p, bound))
    run_jobs(merges)
    _ = jobs^
    var firsts = List[Int]()
    var parts = List[List[Series]](capacity=len(merges))
    for p in range(len(merges)):
        for row in merges[p].firsts:
            firsts.append(row)
        var finished = List[Series](capacity=len(bound))
        for j in range(len(bound)):
            finished.append(merges[p].reducers[j].finish())
        parts.append(finished^)
    return (firsts^, parts^)
