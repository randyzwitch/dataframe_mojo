"""DuckDB 2.0's window partitioning: the sort strategies of PhysicalWindow
(common/sort/sort_strategy.cpp, hashed_sort.cpp, full_sort.cpp,
natural_sort.cpp) and the boundary masks of WindowHashGroup
(physical_window.cpp, ComputeMasks).

- HashedSort (partition keys): every row's partition keys are hashed and
  the rows are radix-partitioned by hash into 2^bits hash groups, stable
  within a group. Each hash group is then sorted on its own, groups in
  parallel, by the partition keys (ascending, nulls first) and then the
  order keys. A group with no order keys whose rows all share one key
  needs no sort (PartitionKeyTracker::CanBypass).
- FullSort (order keys only): one hash group, sorted as a whole.
- NaturalSort (neither): one hash group in input order.

The sorted layout is the hash groups one after another. Walking it marks
where each partition starts (DuckDB's partition_mask) and where each run
of rows with equal partition and order keys starts (its order_mask, the
peer boundaries). Every partition lies inside one hash group, so groups
are evaluated independently.

Sorts are DuckDB's (`dataframe.sorting.sort_rows`), kept stable (ties in
input order), as `over(order_by=...)` keeps input order on ties.
"""
from std.memory import Pointer

from dataframe.storage.allocation import huge_uninit
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.column import Column
from dataframe.storage.string import StringColumn
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.kernels.partition import Partitioner, _same_value
from dataframe.parallel import Job, configured_workers, partitions, run_jobs
from dataframe.series import Series
from dataframe.sorting.sort import sort_rows
from dataframe.kernels.gather import take_parallel
from dataframe.kernels.binary import validity
from dataframe.sorting.sort_key import STANDARD_VECTOR_SIZE

# hashed_sort.cpp, HashedSortGlobalSinkState::Rehash: partitions start at
# 4 radix bits and grow by 2 while the average hash group holds more than
# DEFAULT_ROW_GROUP_SIZE rows, up to max_bits (8 for any thread memory
# above 512 MB: thread_pages >> max_bits > 1).
comptime _START_BITS = 4
comptime _MAX_BITS = 8
comptime DEFAULT_ROW_GROUP_SIZE = 122880
# Boundary flags per sorted position.
comptime PARTITION_START = UInt8(1)
comptime PEER_START = UInt8(2)


def radix_bits(rows: Int) -> Int:
    """Hash-group radix bits for `rows` rows (HashedSort Rehash)."""
    var bits = _START_BITS
    while bits < _MAX_BITS and rows // (1 << bits) > DEFAULT_ROW_GROUP_SIZE:
        bits = min(bits + 2, _MAX_BITS)
    return bits


struct WindowLayout(Movable):
    """Rows sorted by hash group, then partition and order keys, with the
    partition and peer boundaries of the sorted order."""

    # Sorted position -> input row.
    var order: List[Int]
    # Input row -> sorted position (empty when `in_place`).
    var inverse: List[Int]
    # Whether `order` is the identity: nothing to gather or scatter.
    var in_place: Bool
    # Per sorted position: PARTITION_START and PEER_START flags.
    var flags: List[UInt8]
    # Where each partition starts in the sorted order (count + 1 entries).
    var starts: List[Int]
    var count: Int
    # Partition index bounds of the hash groups (groups + 1 entries).
    var group_parts: List[Int]
    # Input row -> its partition's index (in sorted partition order).
    var row_partition: List[Int]

    def __init__(out self):
        self.order = List[Int]()
        self.inverse = List[Int]()
        self.in_place = True
        self.flags = List[UInt8]()
        self.starts = List[Int]()
        self.count = 0
        self.group_parts = List[Int]()
        self.row_partition = List[Int]()

    def partition_ids(self) -> List[Int]:
        """Each sorted position's partition index."""
        var ids = List[Int](unsafe_uninit_length=len(self.order))
        var p = ids.unsafe_ptr()
        for g in range(self.count):
            for k in range(self.starts[g], self.starts[g + 1]):
                p[k] = g
        return ids^


@always_inline
def _numeric_changes[
    D: DType
](
    column: Column[Scalar[D]],
    flags: Pointer[UInt8, MutAnyOrigin],
    lo: Int,
    hi: Int,
    bit: UInt8,
):
    """Mark positions of [lo + 1, hi) whose value differs from the one
    before (integers: nulls equal to each other)."""
    var values = column._ptr()
    if len(column._bits[]) == 0:
        for k in range(lo + 1, hi):
            if values[unsafe_offset=k] != values[unsafe_offset=k - 1]:
                flags[unsafe_offset=k] |= bit
        return
    for k in range(lo + 1, hi):
        var a = column._valid(k - 1)
        var b = column._valid(k)
        if a != b or (
            b and values[unsafe_offset=k] != values[unsafe_offset=k - 1]
        ):
            flags[unsafe_offset=k] |= bit


def mark_changes(
    key: Series,
    flags_address: Int,
    lo: Int,
    hi: Int,
    bit: UInt8,
) raises:
    """Set `bit` in flags[k] for every k in (lo, hi) where the sorted key
    column's row k differs from row k - 1, as the sort compares them:
    nulls equal, NaN one value, -0.0 equal to 0.0 (VectorOperations::
    DistinctFrom in ComputeMasks)."""
    var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=flags_address)
    var column = key.rechunk() if key.is_chunked() else key.copy()
    comptime for t in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[t]
        comptime if not D.is_floating_point():
            if column._data.isa[Column[Scalar[D]]]():
                _numeric_changes[D](
                    column._data[Column[Scalar[D]]], flags, lo, hi, bit
                )
                return
    if column._data.isa[StringColumn]():
        ref strings = column._data[StringColumn]
        for k in range(lo + 1, hi):
            var a = strings._valid(k - 1)
            var b = strings._valid(k)
            if a != b or (b and not strings._equal_at_valid(strings, k, k - 1)):
                flags[unsafe_offset=k] |= bit
        return
    for k in range(lo + 1, hi):
        if not _same_value(column, k, k - 1):
            flags[unsafe_offset=k] |= bit


def sort_key_rows(
    keys: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
    stable: Bool = True,
) raises -> List[Int]:
    """The sorted order of the key columns' rows: DuckDB's sort
    (`sort_rows`), with ties in input order when `stable`, as
    over(order_by=...) keeps them. Threads come from the shared budget, so
    groups sorted on several workers at once share it."""
    return sort_rows(keys, descending, nulls_last, maintain_order=stable)


struct _GroupSortJob(Job):
    """Sort one hash group and mark its boundaries (the SORT, MATERIALIZE
    and MASK stages of one WindowHashGroup)."""

    var keys: List[Series]
    var partition_keys: Int
    var descending: List[Bool]
    var nulls_last: List[Bool]
    # The group's input rows, in input order, at rows[lo:hi].
    var rows: Int
    var lo: Int
    var hi: Int
    # Output: order[lo:hi] and flags[lo:hi].
    var order: Int
    var flags: Int
    var stable: Bool

    def __init__(
        out self,
        keys: List[Series],
        partition_keys: Int,
        descending: List[Bool],
        nulls_last: List[Bool],
        rows: Int,
        lo: Int,
        hi: Int,
        order: Int,
        flags: Int,
        stable: Bool,
    ):
        self.keys = keys.copy()
        self.partition_keys = partition_keys
        self.descending = descending.copy()
        self.nulls_last = nulls_last.copy()
        self.rows = rows
        self.lo = lo
        self.hi = hi
        self.order = order
        self.flags = flags
        self.stable = stable

    def run(mut self) raises:
        var n = self.hi - self.lo
        if n == 0:
            return
        var rows = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.rows)
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var flags = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.flags + self.lo
        )
        var mine = List[Int](unsafe_uninit_length=n)
        for k in range(n):
            mine[k] = rows[unsafe_offset=self.lo + k]
        var gathered = List[Series](capacity=len(self.keys))
        var identity = True
        for k in range(n):
            identity = identity and mine[k] == self.lo + k
        for key in self.keys:
            gathered.append(
                key.copy() if identity and n == len(key) else key.take(mine)
            )
        var sorted = gathered^
        var count = len(self.keys)
        var bypass = count == self.partition_keys
        if bypass:
            # PartitionKeyTracker: one key in the group skips the sort.
            for k in range(n):
                flags[unsafe_offset=k] = 0
            for c in range(count):
                mark_changes(
                    sorted[c], self.flags + self.lo, 0, n, PARTITION_START
                )
            for k in range(1, n):
                if flags[unsafe_offset=k] != 0:
                    bypass = False
                    break
        if bypass:
            for k in range(n):
                order[unsafe_offset=self.lo + k] = mine[k]
        else:
            var perm = sort_key_rows(
                sorted, self.descending, self.nulls_last, self.stable
            )
            var in_order = True
            for k in range(n):
                in_order = in_order and perm[k] == k
            if not in_order:
                for c in range(count):
                    sorted[c] = sorted[c].take(perm)
            for k in range(n):
                order[unsafe_offset=self.lo + k] = mine[perm[k]]
        for k in range(n):
            flags[unsafe_offset=k] = 0
        for c in range(self.partition_keys):
            mark_changes(sorted[c], self.flags + self.lo, 0, n, PARTITION_START)
        for k in range(n):
            if flags[unsafe_offset=k] != 0:
                flags[unsafe_offset=k] = PARTITION_START | PEER_START
        for c in range(self.partition_keys, count):
            mark_changes(sorted[c], self.flags + self.lo, 0, n, PEER_START)
        flags[unsafe_offset=0] = PARTITION_START | PEER_START


struct _MaskJob(Job):
    """Mark peer boundaries over one range of a fully sorted key set (the
    MASK stage, split across threads for one large hash group)."""

    var keys: List[Series]
    var lo: Int
    var hi: Int
    var flags: Int

    def __init__(out self, keys: List[Series], lo: Int, hi: Int, flags: Int):
        self.keys = keys.copy()
        self.lo = lo
        self.hi = hi
        self.flags = flags

    def run(mut self) raises:
        var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.flags)
        for k in range(self.lo, self.hi):
            flags[unsafe_offset=k] = 0
        var from_ = max(self.lo - 1, 0)
        for key in self.keys:
            mark_changes(key, self.flags, from_, self.hi, PEER_START)


def window_layout(
    partition_keys: List[Series],
    order_keys: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
    height: Int,
    stable: Bool = True,
) raises -> WindowLayout:
    """Sort `height` rows by partition and order keys into hash groups and
    mark their boundaries. `descending` and `nulls_last` hold one flag per
    order key. Rows with equal keys keep input order when `stable`."""
    var layout = WindowLayout()
    layout.group_parts.append(0)
    if height == 0:
        layout.starts.append(0)
        return layout^
    var workers = configured_workers()
    var order = huge_uninit[Int](height)
    var flags = huge_uninit[UInt8](height)
    var bounds = List[Int]()
    if len(partition_keys) > 0:
        # HashedSort: radix-partition rows by their keys' hash.
        # The sink runs a thread per source morsel (DEFAULT_ROW_GROUP_SIZE
        # rows), each hashing and partitioning its own rows.
        var sink = max(
            1,
            min(
                workers,
                (height + DEFAULT_ROW_GROUP_SIZE - 1) // DEFAULT_ROW_GROUP_SIZE,
            ),
        )
        var partitioner = Partitioner(partition_keys, sink)
        var bits = radix_bits(height)
        var parts = partitioner.scatter(sink, bucket_count=1 << bits)
        var keys = partitioner.keys.copy()
        for key in order_keys:
            keys.append(key.rechunk() if key.is_chunked() else key.copy())
        var desc = List[Bool](length=len(partition_keys), fill=False)
        var last = List[Bool](length=len(partition_keys), fill=False)
        for k in range(len(order_keys)):
            desc.append(descending[k])
            last.append(nulls_last[k])
        # Largest groups first, as CreateTaskList schedules them.
        var by_size = List[Int]()
        for b in range(parts.buckets()):
            if parts.bounds[b + 1] > parts.bounds[b]:
                by_size.append(b)
        for i in range(1, len(by_size)):
            var b = by_size[i]
            var j = i
            while j > 0 and (
                parts.bounds[by_size[j - 1] + 1] - parts.bounds[by_size[j - 1]]
                < parts.bounds[b + 1] - parts.bounds[b]
            ):
                by_size[j] = by_size[j - 1]
                j -= 1
            by_size[j] = b
        var jobs = List[_GroupSortJob](capacity=len(by_size))
        for b in by_size:
            jobs.append(
                _GroupSortJob(
                    keys,
                    len(partition_keys),
                    desc,
                    last,
                    Int(parts.order.unsafe_ptr()),
                    parts.bounds[b],
                    parts.bounds[b + 1],
                    Int(order.unsafe_ptr()),
                    Int(flags.unsafe_ptr()),
                    stable,
                )
            )
        run_jobs(jobs)
        for b in range(parts.buckets()):
            if parts.bounds[b + 1] > parts.bounds[b]:
                bounds.append(parts.bounds[b])
        _ = parts^
    elif len(order_keys) > 0:
        # FullSort: one group, sorted on every thread.
        var perm = sort_key_rows(order_keys, descending, nulls_last, stable)
        var keys = List[Series](capacity=len(order_keys))
        var identity = True
        for k in range(height):
            identity = identity and perm[k] == k
        for key in order_keys:
            keys.append(key.copy())
        if not identity:
            keys = take_parallel(keys, perm.copy(), workers)
        for k in range(height):
            order[k] = perm[k]
        var ranges = partitions(height, workers, 4096)
        var jobs = List[_MaskJob](capacity=workers)
        for w in range(workers):
            if ranges[w + 1] > ranges[w]:
                jobs.append(
                    _MaskJob(
                        keys, ranges[w], ranges[w + 1], Int(flags.unsafe_ptr())
                    )
                )
        run_jobs(jobs)
        flags[0] = PARTITION_START | PEER_START
        bounds.append(0)
    else:
        # NaturalSort: input order, one partition of peers.
        for k in range(height):
            order[k] = k
            flags[k] = 0
        flags[0] = PARTITION_START | PEER_START
        bounds.append(0)
    bounds.append(height)
    # Partition starts, then each row's partition and sorted position, over
    # ranges of sorted positions in parallel.
    var ranges = partitions(height, workers, STANDARD_VECTOR_SIZE)
    var finish = List[_FinishJob](capacity=workers)
    for w in range(workers):
        if ranges[w + 1] > ranges[w]:
            finish.append(
                _FinishJob(
                    Int(flags.unsafe_ptr()),
                    Int(order.unsafe_ptr()),
                    ranges[w],
                    ranges[w + 1],
                )
            )
    if len(finish) > 1:
        run_jobs(finish)
    else:
        finish[0].run()
    layout.in_place = True
    var first = List[Int](capacity=len(finish))
    for j in range(len(finish)):
        layout.in_place = layout.in_place and finish[j].in_place
        first.append(len(layout.starts) - 1)
        layout.starts.extend(Span(finish[j].starts))
    layout.count = len(layout.starts)
    layout.starts.append(height)
    # The groups' partition ranges: every group starts a partition.
    var g = 1
    for p in range(layout.count):
        if g < len(bounds) - 1 and layout.starts[p] == bounds[g]:
            layout.group_parts.append(p)
            g += 1
    layout.group_parts.append(layout.count)
    layout.row_partition = List[Int](unsafe_uninit_length=height)
    if not layout.in_place:
        layout.inverse = List[Int](unsafe_uninit_length=height)
    for j in range(len(finish)):
        finish[j].partition = first[j]
        finish[j].row_partition = Int(layout.row_partition.unsafe_ptr())
        finish[j].inverse = 0 if layout.in_place else Int(
            layout.inverse.unsafe_ptr()
        )
    if len(finish) > 1:
        run_jobs(finish)
    else:
        finish[0].run()
    layout.order = order^
    layout.flags = flags^
    return layout^


struct _FinishJob(Job):
    """One range of sorted positions [lo, hi): first its partition starts
    and whether order is the identity there; then (`row_partition` set)
    each row's partition index and sorted position."""

    var flags: Int
    var order: Int
    var lo: Int
    var hi: Int
    var starts: List[Int]
    var in_place: Bool
    # Second pass: the partition holding lo, and the outputs.
    var partition: Int
    var row_partition: Int
    var inverse: Int

    def __init__(out self, flags: Int, order: Int, lo: Int, hi: Int):
        self.flags = flags
        self.order = order
        self.lo = lo
        self.hi = hi
        self.starts = List[Int]()
        self.in_place = True
        self.partition = 0
        self.row_partition = 0
        self.inverse = 0

    def run(mut self) raises:
        var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.flags)
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        if self.row_partition == 0:
            var identity = True
            for k in range(self.lo, self.hi):
                if (flags[unsafe_offset=k] & PARTITION_START) != 0:
                    self.starts.append(k)
                identity = identity and order[unsafe_offset=k] == k
            self.in_place = identity
            return
        var rows = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=self.row_partition
        )
        var p = self.partition
        if self.inverse == 0:
            for k in range(self.lo, self.hi):
                if (flags[unsafe_offset=k] & PARTITION_START) != 0:
                    p += 1
                rows[unsafe_offset=order[unsafe_offset=k]] = p
            return
        var inverse = Pointer[Int, MutAnyOrigin](
            unsafe_from_address=self.inverse
        )
        for k in range(self.lo, self.hi):
            if (flags[unsafe_offset=k] & PARTITION_START) != 0:
                p += 1
            var row = order[unsafe_offset=k]
            rows[unsafe_offset=row] = p
            inverse[unsafe_offset=row] = k


# Rank methods (rank()'s `method`).
comptime RANK_ORDINAL = 0
comptime RANK_MIN = 1
comptime RANK_MAX = 2
comptime RANK_DENSE = 3
comptime RANK_AVERAGE = 4


def rank_method(method: String) raises -> Int:
    if method == "ordinal":
        return RANK_ORDINAL
    if method == "min":
        return RANK_MIN
    if method == "max":
        return RANK_MAX
    if method == "dense":
        return RANK_DENSE
    if method == "average":
        return RANK_AVERAGE
    raise Error("unknown rank method: " + method)


def rank_order_keys(
    input: Series,
    descending: Bool,
    mut keys: List[Series],
    mut key_descending: List[Bool],
    mut key_nulls_last: List[Bool],
) raises:
    """Append the order keys that rank `input`: its value (a struct's
    fields in turn), nulls first in each partition so that ranks count
    from the first value. A struct field's nulls are its smallest values
    (Polars); a null struct sorts before every value."""
    var column = input.rechunk() if input.is_chunked() else input.copy()
    if column.dtype().is_struct():
        var structs = column.struct_column()
        if column.null_count() > 0:
            var present = validity(column)
            var flags = List[UInt8](unsafe_uninit_length=len(present))
            for k in range(len(present)):
                flags[k] = UInt8(1) if present[k] else UInt8(0)
            keys.append(Series("", Column[UInt8](flags^)))
            key_descending.append(False)
            key_nulls_last.append(False)
        for k in range(structs.field_count()):
            keys.append(structs.field(k))
            key_descending.append(descending)
            key_nulls_last.append(descending)
        return
    keys.append(column^)
    key_descending.append(descending)
    key_nulls_last.append(False)


struct _RankScanJob(Job):
    """Ranks over sorted positions [lo, hi) from the partition and peer
    boundaries (WindowPeerLocalState::NextRank, and the peer end for max
    and average). The first pass summarizes the range; the second writes
    ranks from the state carried in from the ranges before it."""

    var flags: Int
    var order: Int
    # Input row -> Bool validity (0: every row valid).
    var valid: Int
    var lo: Int
    var hi: Int
    var method: Int
    var write: Bool
    # Carried in (second pass): the partition's first valid position (-1
    # while in its nulls), the current peer group's start, the dense rank.
    var base: Int
    var peer: Int
    var dense: Int
    # The first peer boundary at or after hi.
    var next_start: Int
    # Summary (first pass).
    var has_partition: Bool
    var first_peer: Int
    var last_peer: Int
    var first_valid_peer: Int
    var local_base: Int
    var local_dense: Int
    # Outputs, by input row.
    var ints: Int
    var floats: Int
    var out_valid: Int

    def __init__(
        out self,
        flags: Int,
        order: Int,
        valid: Int,
        lo: Int,
        hi: Int,
        method: Int,
        ints: Int,
        floats: Int,
        out_valid: Int,
    ):
        self.flags = flags
        self.order = order
        self.valid = valid
        self.lo = lo
        self.hi = hi
        self.method = method
        self.write = False
        self.base = -1
        self.peer = lo
        self.dense = 0
        self.next_start = hi
        self.has_partition = False
        self.first_peer = -1
        self.last_peer = -1
        self.first_valid_peer = -1
        self.local_base = -1
        self.local_dense = 0
        self.ints = ints
        self.floats = floats
        self.out_valid = out_valid

    @always_inline
    def _valid_at(self, position: Int) -> Bool:
        if self.valid == 0:
            return True
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var valid = Pointer[Bool, MutAnyOrigin](unsafe_from_address=self.valid)
        return valid[unsafe_offset=order[unsafe_offset=position]]

    def run(mut self) raises:
        var flags = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=self.flags)
        if not self.write:
            for k in range(self.lo, self.hi):
                var f = flags[unsafe_offset=k]
                if (f & PEER_START) == 0:
                    continue
                if self.first_peer < 0:
                    self.first_peer = k
                self.last_peer = k
                if (f & PARTITION_START) != 0:
                    self.has_partition = True
                    self.local_base = -1
                    self.local_dense = 0
                if self._valid_at(k):
                    if self.has_partition:
                        if self.local_base < 0:
                            self.local_base = k
                    elif self.first_valid_peer < 0:
                        self.first_valid_peer = k
                    self.local_dense += 1
            return
        var order = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.order)
        var ints = Pointer[Int64, MutAnyOrigin](unsafe_from_address=self.ints)
        var floats = Pointer[Float64, MutAnyOrigin](
            unsafe_from_address=self.floats
        )
        var out_valid = Pointer[Bool, MutAnyOrigin](
            unsafe_from_address=self.out_valid
        )
        var method = self.method
        var base = self.base
        var peer = self.peer
        var dense = self.dense
        var k = self.lo
        while k < self.hi:
            var f = flags[unsafe_offset=k]
            var present: Bool
            if (f & PEER_START) != 0:
                if (f & PARTITION_START) != 0:
                    base = -1
                    dense = 0
                peer = k
                present = self._valid_at(k)
                if present:
                    if base < 0:
                        base = k
                    dense += 1
            else:
                # The peer group carried in from the range before.
                present = base >= 0 and self._valid_at(k)
            var e = k + 1
            while e < self.hi and (flags[unsafe_offset=e] & PEER_START) == 0:
                e += 1
            var end = e if e < self.hi else self.next_start
            if self.out_valid != 0:
                for j in range(k, e):
                    out_valid[unsafe_offset=order[unsafe_offset=j]] = present
            if not present:
                k = e
                continue
            if method == RANK_AVERAGE:
                var value = Float64(peer + 1 + end - 2 * base) / 2
                for j in range(k, e):
                    floats[unsafe_offset=order[unsafe_offset=j]] = value
            elif method == RANK_ORDINAL:
                for j in range(k, e):
                    ints[unsafe_offset=order[unsafe_offset=j]] = Int64(
                        j - base + 1
                    )
            else:
                var value: Int64
                if method == RANK_MIN:
                    value = Int64(peer - base + 1)
                elif method == RANK_MAX:
                    value = Int64(end - base)
                else:
                    value = Int64(dense)
                for j in range(k, e):
                    ints[unsafe_offset=order[unsafe_offset=j]] = value
            k = e


def rank_from_layout(
    layout: WindowLayout, input: Series, method: String
) raises -> Series:
    """Every row's rank within its partition, read off a layout whose last
    order keys are `rank_order_keys(input)`: runs of equal values are
    peers, as DuckDB's rank executors read them from the order mask. Nulls
    rank as null; ranks count from a partition's first value. Ordinal
    ranks peers in layout order (input order when the layout is stable).
    Ranges of sorted positions are ranked in parallel, each starting from
    the state its predecessors carry in."""
    var code = rank_method(method)
    var n = len(layout.order)
    var average = code == RANK_AVERAGE
    var ints = List[Int64](unsafe_uninit_length=0 if average else n)
    var floats = List[Float64](unsafe_uninit_length=n if average else 0)
    var present = List[Bool]()
    var has_nulls = input.null_count() > 0
    if has_nulls:
        present = validity(input)
    var out_valid = List[Bool](unsafe_uninit_length=n if has_nulls else 0)
    var workers = configured_workers()
    var bounds = partitions(n, workers, STANDARD_VECTOR_SIZE)
    var jobs = List[_RankScanJob](capacity=workers)
    for w in range(workers):
        if bounds[w + 1] > bounds[w]:
            jobs.append(
                _RankScanJob(
                    Int(layout.flags.unsafe_ptr()),
                    Int(layout.order.unsafe_ptr()),
                    Int(present.unsafe_ptr()) if has_nulls else 0,
                    bounds[w],
                    bounds[w + 1],
                    code,
                    Int(ints.unsafe_ptr()),
                    Int(floats.unsafe_ptr()),
                    Int(out_valid.unsafe_ptr()) if has_nulls else 0,
                )
            )
    if len(jobs) > 1:
        run_jobs(jobs)
        # Carry the state across ranges, then the next boundary backwards.
        var base = -1
        var peer = 0
        var dense = 0
        for j in range(len(jobs)):
            jobs[j].base = base
            jobs[j].peer = peer
            jobs[j].dense = dense
            if jobs[j].has_partition:
                base = jobs[j].local_base
                dense = jobs[j].local_dense
            else:
                if base < 0 and jobs[j].first_valid_peer >= 0:
                    base = jobs[j].first_valid_peer
                dense += jobs[j].local_dense
            if jobs[j].last_peer >= 0:
                peer = jobs[j].last_peer
        var next_start = n
        for r in range(len(jobs)):
            var j = len(jobs) - 1 - r
            jobs[j].next_start = next_start
            if jobs[j].first_peer >= 0:
                next_start = jobs[j].first_peer
    elif len(jobs) == 1:
        jobs[0].next_start = n
    for j in range(len(jobs)):
        jobs[j].write = True
    if len(jobs) > 1:
        run_jobs(jobs)
    elif len(jobs) == 1:
        jobs[0].run()
    _ = present^
    if average:
        if has_nulls:
            return Series("", Column[Float64](floats^, out_valid))
        return Series("", Column[Float64](floats^))
    if has_nulls:
        return Series("", Column[Int64](ints^, out_valid))
    return Series("", Column[Int64](ints^))


def rank_rows(
    input: Series, ids: List[Int], method: String, descending: Bool
) raises -> Series:
    """Rank `input` within the partitions `ids` gives (none: one
    partition), through the window layout: rank() OVER (PARTITION BY ids
    ORDER BY input)."""
    var n = len(input)
    var partition_keys = List[Series]()
    if len(ids) > 0:
        var codes = List[Int64](unsafe_uninit_length=n)
        for k in range(n):
            codes[k] = Int64(ids[k])
        partition_keys.append(Series("", Column[Int64](codes^)))
    var keys = List[Series]()
    var key_descending = List[Bool]()
    var key_nulls_last = List[Bool]()
    rank_order_keys(input, descending, keys, key_descending, key_nulls_last)
    var layout = window_layout(
        partition_keys,
        keys,
        key_descending,
        key_nulls_last,
        n,
        stable=method == "ordinal",
    )
    return rank_from_layout(layout, input, method)
