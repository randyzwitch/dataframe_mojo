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
) raises -> List[Int]:
    """The stable sorted order of the key columns' rows: DuckDB's sort
    (`sort_rows`) with ties in input order, as over(order_by=...) keeps
    them. Threads come from the shared budget, so groups sorted on
    several workers at once share it."""
    return sort_rows(keys, descending, nulls_last, maintain_order=True)


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
            var perm = sort_key_rows(sorted, self.descending, self.nulls_last)
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
) raises -> WindowLayout:
    """Sort `height` rows by partition and order keys into hash groups and
    mark their boundaries. `descending` and `nulls_last` hold one flag per
    order key."""
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
        var partitioner = Partitioner(partition_keys, workers)
        var bits = radix_bits(height)
        var parts = partitioner.scatter(workers, bucket_count=1 << bits)
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
                )
            )
        run_jobs(jobs)
        for b in range(parts.buckets()):
            if parts.bounds[b + 1] > parts.bounds[b]:
                bounds.append(parts.bounds[b])
        _ = parts^
    elif len(order_keys) > 0:
        # FullSort: one group, sorted on every thread.
        var perm = sort_key_rows(order_keys, descending, nulls_last)
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
    # Partition starts and the groups' partition ranges.
    var fp = flags.unsafe_ptr()
    var g = 0
    for k in range(height):
        if k == bounds[g + 1]:
            g += 1
            layout.group_parts.append(len(layout.starts))
        if (fp[unsafe_offset=k] & PARTITION_START) != 0:
            layout.starts.append(k)
    layout.count = len(layout.starts)
    layout.starts.append(height)
    layout.group_parts.append(layout.count)
    var op = order.unsafe_ptr()
    layout.in_place = True
    for k in range(height):
        if op[unsafe_offset=k] != k:
            layout.in_place = False
            break
    layout.row_partition = List[Int](unsafe_uninit_length=height)
    var rp = layout.row_partition.unsafe_ptr()
    for p in range(layout.count):
        for k in range(layout.starts[p], layout.starts[p + 1]):
            rp[unsafe_offset=op[unsafe_offset=k]] = p
    if not layout.in_place:
        layout.inverse = List[Int](unsafe_uninit_length=height)
        var ip = layout.inverse.unsafe_ptr()
        for k in range(height):
            ip[unsafe_offset=op[unsafe_offset=k]] = k
    layout.order = order^
    layout.flags = flags^
    return layout^
