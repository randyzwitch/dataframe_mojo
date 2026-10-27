"""Stable join row layouts, pair ordering, and worker jobs."""
from dataframe.expr import (
    MAX,
)
from dataframe.parallel import (
    Job,
    partitions,
    run_jobs,
)
from dataframe.trace import (
    trace_path,
)
from std.memory import (
    ArcPointer,
    Pointer,
)


struct _OrderCountJob(Job):
    """Rows of each left id among pairs [lo, hi)."""

    var ids: Int
    var lo: Int
    var hi: Int
    var counts: List[Int]

    def __init__(out self, ids: Int, lo: Int, hi: Int, count: Int):
        self.ids = ids
        self.lo = lo
        self.hi = hi
        self.counts = List[Int](length=count, fill=0)

    def run(mut self) raises:
        var ids = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.ids)
        var counts = self.counts.unsafe_ptr()
        for k in range(self.lo, self.hi):
            counts[unsafe_offset=ids[unsafe_offset=k]] += 1


struct _OrderScatterJob(Job):
    """Pairs [lo, hi) to their left id's group, each id's next position
    in `cursor` (this range's own, after earlier ranges' rows)."""

    var ids: Int
    var rows: Int
    var out: Int
    var lo: Int
    var hi: Int
    var cursor: List[Int]

    def __init__(
        out self,
        ids: Int,
        rows: Int,
        out_rows: Int,
        lo: Int,
        hi: Int,
        var cursor: List[Int],
    ):
        self.ids = ids
        self.rows = rows
        self.out = out_rows
        self.lo = lo
        self.hi = hi
        self.cursor = cursor^

    def run(mut self) raises:
        var ids = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.ids)
        var rows = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.rows)
        var out = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.out)
        var at = self.cursor.unsafe_ptr()
        for k in range(self.lo, self.hi):
            var id = ids[unsafe_offset=k]
            var position = at[unsafe_offset=id]
            at[unsafe_offset=id] = position + 1
            out[unsafe_offset=position] = rows[unsafe_offset=k]


struct _OrderFillJob(Job):
    """Each output position of left ids [lo, hi) given its id."""

    var starts: Int
    var out: Int
    var lo: Int
    var hi: Int

    def __init__(out self, starts: Int, out_ids: Int, lo: Int, hi: Int):
        self.starts = starts
        self.out = out_ids
        self.lo = lo
        self.hi = hi

    def run(mut self) raises:
        var starts = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.starts)
        var out = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.out)
        for id in range(self.lo, self.hi):
            for position in range(
                starts[unsafe_offset=id], starts[unsafe_offset=id + 1]
            ):
                out[unsafe_offset=position] = id


def _parallel_left_major(
    ids: List[Int], rows: List[Int], count: Int, lanes: Int
) raises -> Tuple[List[Int], List[Int]]:
    """`_smaller_build_join_rows`' stable counting sort of pairs by left id,
    on `lanes` workers: each counts its contiguous range of pairs per id,
    one pass turns the counts into each range's own write positions (after
    the earlier ranges' rows of the same id), and each scatters its range
    there, as a parallel radix sort's scatter does. Ranges follow probe
    order, so every group keeps its pairs in probe order, as the serial
    sort does."""
    var n = len(ids)
    var bounds = partitions(n, lanes, 1)
    var counting = List[_OrderCountJob](capacity=lanes)
    for j in range(lanes):
        counting.append(
            _OrderCountJob(
                Int(ids.unsafe_ptr()), bounds[j], bounds[j + 1], count
            )
        )
    run_jobs(counting)
    var starts = List[Int](unsafe_uninit_length=count + 1)
    var total = 0
    for id in range(count):
        starts[id] = total
        for j in range(lanes):
            var c = counting[j].counts[id]
            counting[j].counts[id] = total
            total += c
    starts[count] = total
    var left_rows = List[Int](unsafe_uninit_length=n)
    var right_rows = List[Int](unsafe_uninit_length=n)
    var scatter = List[_OrderScatterJob](capacity=lanes)
    for j in range(lanes):
        var cursor = List[Int]()
        swap(cursor, counting[j].counts)
        scatter.append(
            _OrderScatterJob(
                Int(ids.unsafe_ptr()),
                Int(rows.unsafe_ptr()),
                Int(right_rows.unsafe_ptr()),
                bounds[j],
                bounds[j + 1],
                cursor^,
            )
        )
    run_jobs(scatter)
    var id_bounds = partitions(count, lanes, 1)
    var fill = List[_OrderFillJob](capacity=lanes)
    for j in range(lanes):
        if id_bounds[j] < id_bounds[j + 1]:
            fill.append(
                _OrderFillJob(
                    Int(starts.unsafe_ptr()),
                    Int(left_rows.unsafe_ptr()),
                    id_bounds[j],
                    id_bounds[j + 1],
                )
            )
    run_jobs(fill)
    _ = starts^
    return (left_rows^, right_rows^)


def _group_index(ids: List[Int], count: Int) -> List[Int]:
    """Start offset per key id, in a flat CSR layout (count + 1 entries)."""
    var starts = List[Int](length=count + 1, fill=0)
    var counts = starts.unsafe_ptr()
    var source = ids.unsafe_ptr()
    for k in range(len(ids)):
        var id = source[unsafe_offset=k]
        if id >= 0:
            counts[unsafe_offset=id + 1] += 1
    for g in range(count):
        counts[unsafe_offset=g + 1] += counts[unsafe_offset=g]
    return starts^


def _group_rows(ids: List[Int], starts: List[Int]) -> List[Int]:
    """Row indices grouped by key id, each group in increasing row order."""
    var rows = List[Int](unsafe_uninit_length=starts[len(starts) - 1])
    var cursor = starts.copy()
    var at = cursor.unsafe_ptr()
    var out = rows.unsafe_ptr()
    var source = ids.unsafe_ptr()
    for i in range(len(ids)):
        var id = source[unsafe_offset=i]
        if id >= 0:
            var position = at[unsafe_offset=id]
            at[unsafe_offset=id] = position + 1
            out[unsafe_offset=position] = i
    return rows^


# Rows from which a Float64 comparison filter runs directly on aligned
# chunks instead of materializing a mask. Swept on 2026-09-25 with the
# overall benchmark's filter (32 threads, best of 7, two rounds): the mask
# path won at 5k and 20k rows (0.18 vs 0.25 ms, 0.70 vs 0.73 ms) and the


# Measured on a Threadripper 3970X (32 cores, 16 MiB L3 per cache domain),
# Mojo 1.2, at 32 workers with an 8-worker cross-check; see
# docs/design/join-cutoffs.md for the alternating sweeps and their limitations.
# CSR: 750k distinct rows lose (15 vs 18 ms), 1M win (33 vs 21-26 ms).
# At 1.5M rows / 93,750 groups serial wins; at 1.75M / 109,375 parallel wins.
# Use the conservative 16 MiB working-set boundary: earlier dispatch helps
# uniform random IDs but regressed the 1M-row dictionary-encoded full join.
comptime _CSR_PARALLEL_WORKING_BYTES = 16 * 1024 * 1024
# At 6M rows, 64/512 groups win serial (63/35 vs 83/67 ms), 4,098
# groups are level (64 vs 62 ms), and 5,859 groups win parallel (127 vs 71).
# Keep the cursor below 32 KiB on the serial path.
comptime _CSR_MIN_PARALLEL_GROUPS = 4096


def _parallel_csr_fits(ids: List[Int], groups: Int, workers: Int) -> Bool:
    """Choose stable scatter from footprint, cardinality and sampled locality.

    The sample only selects between equivalent algorithms; it is not an
    ordering proof. Missing a rare inversion merely keeps the serial path.
    """
    var rows = len(ids)
    if (
        workers <= 1
        or groups < _CSR_MIN_PARALLEL_GROUPS
        or rows < max(0, _CSR_PARALLEL_WORKING_BYTES // 8 - groups)
    ):
        return False
    # Ordered IDs write almost sequentially: 1M/2M/4M distinct rows take
    # 6/13/25 ms serial vs 19/40/69 ms parallel. At most 64 adjacent pairs
    # sample locality; random IDs usually reject ordering at the first pair.
    var step = max(1, (rows - 1) // 64)
    for sample in range(64):
        var row = 1 + sample * step
        if row >= rows:
            break
        if ids[row] >= 0 and ids[row - 1] > ids[row]:
            return True
    return False


def _id_range_first(bucket: Int, groups: Int, buckets: Int) -> Int:
    return (bucket * groups + buckets - 1) // buckets


def _id_range_bucket(id: Int, groups: Int, buckets: Int) -> Int:
    var bucket = 0
    while bucket + 1 < buckets and id >= _id_range_first(
        bucket + 1, groups, buckets
    ):
        bucket += 1
    return bucket


struct _CSRCountJob(Job):
    """Count one input range into contiguous global-id buckets."""

    var ids: Int
    var start: Int
    var end: Int
    var groups: Int
    var buckets: Int
    var counts: List[Int]

    def __init__(
        out self, ids: Int, start: Int, end: Int, groups: Int, buckets: Int
    ):
        self.ids = ids
        self.start = start
        self.end = end
        self.groups = groups
        self.buckets = buckets
        self.counts = List[Int]()

    def run(mut self) raises:
        ref ids = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.ids
        )[]
        self.counts = List[Int](length=self.buckets, fill=0)
        for row in range(self.start, self.end):
            if ids[row] >= 0:
                self.counts[
                    _id_range_bucket(ids[row], self.groups, self.buckets)
                ] += 1


struct _CSRScatterJob(Job):
    """Stably scatter one input range into its global-id bucket spans."""

    var ids: Int
    var output: Int
    var start: Int
    var end: Int
    var groups: Int
    var buckets: Int
    var cursor: List[Int]

    def __init__(
        out self,
        ids: Int,
        output: Int,
        start: Int,
        end: Int,
        groups: Int,
        buckets: Int,
        var cursor: List[Int],
    ):
        self.ids = ids
        self.output = output
        self.start = start
        self.end = end
        self.groups = groups
        self.buckets = buckets
        self.cursor = cursor^

    def run(mut self) raises:
        ref ids = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.ids
        )[]
        ref output = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.output
        )[]
        for row in range(self.start, self.end):
            var id = ids[row]
            if id >= 0:
                var bucket = _id_range_bucket(id, self.groups, self.buckets)
                output[self.cursor[bucket]] = row
                self.cursor[bucket] += 1


struct _CSRFillJob(Job):
    """Fill the final CSR groups for one disjoint contiguous id range."""

    var ids: Int
    var order: Int
    var output: Int
    var first: Int
    var last: Int
    var group_first: Int
    var cursor: List[Int]

    def __init__(
        out self,
        ids: Int,
        order: Int,
        output: Int,
        first: Int,
        last: Int,
        group_first: Int,
        var cursor: List[Int],
    ):
        self.ids = ids
        self.order = order
        self.output = output
        self.first = first
        self.last = last
        self.group_first = group_first
        self.cursor = cursor^

    def run(mut self) raises:
        ref ids = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.ids
        )[]
        ref order = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.order
        )[]
        ref output = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.output
        )[]
        for at in range(self.first, self.last):
            var row = order[at]
            var id = ids[row]
            output[self.cursor[id - self.group_first]] = row
            self.cursor[id - self.group_first] += 1


def _parallel_group_rows(
    ids: List[Int], starts: List[Int], workers: Int
) raises -> List[Int]:
    """Stable counting-sort rows into CSR groups without random writes.

    The first scatter is only by a small contiguous-id bucket, with worker
    spans laid out in input order. Each bucket then owns whole CSR groups and
    writes its final segment independently. For one id, its input row order
    therefore survives both passes exactly.
    """
    var groups = len(starts) - 1
    var rows = starts[len(starts) - 1]
    if rows == 0 or groups == 0:
        return List[Int]()
    var buckets = min(workers, groups)
    var bounds = partitions(len(ids), workers, 1)
    var counts = List[_CSRCountJob](capacity=workers)
    for worker in range(workers):
        counts.append(
            _CSRCountJob(
                Int(Pointer(to=ids)),
                bounds[worker],
                bounds[worker + 1],
                groups,
                buckets,
            )
        )
    run_jobs(counts)
    var bucket_starts = List[Int](length=buckets + 1, fill=0)
    for bucket in range(buckets):
        for worker in range(workers):
            bucket_starts[bucket + 1] += counts[worker].counts[bucket]
        bucket_starts[bucket + 1] += bucket_starts[bucket]
    var cursors = List[Int](capacity=buckets)
    for bucket in range(buckets):
        cursors.append(bucket_starts[bucket])
    var scatters = List[_CSRScatterJob](capacity=workers)
    var order = List[Int](length=rows, fill=0)
    for worker in range(workers):
        var cursor = cursors.copy()
        for bucket in range(buckets):
            cursors[bucket] += counts[worker].counts[bucket]
        scatters.append(
            _CSRScatterJob(
                Int(Pointer(to=ids)),
                Int(Pointer(to=order)),
                bounds[worker],
                bounds[worker + 1],
                groups,
                buckets,
                cursor^,
            )
        )
    run_jobs(scatters)
    var flat = List[Int](length=rows, fill=0)
    var fills = List[_CSRFillJob](capacity=buckets)
    for bucket in range(buckets):
        var first_group = _id_range_first(bucket, groups, buckets)
        var last_group = _id_range_first(bucket + 1, groups, buckets)
        var cursor = List[Int](capacity=last_group - first_group)
        for group in range(first_group, last_group):
            cursor.append(starts[group])
        fills.append(
            _CSRFillJob(
                Int(Pointer(to=ids)),
                Int(Pointer(to=order)),
                Int(Pointer(to=flat)),
                bucket_starts[bucket],
                bucket_starts[bucket + 1],
                first_group,
                cursor^,
            )
        )
    run_jobs(fills)
    # `fills` carries `order` only as an address. Keep the owning list live
    # until its workers have consumed that address.
    _ = order^
    return flat^


struct _JoinCountJob(Job):
    """Count inner-join output rows for one contiguous left range."""

    var left_ids: ArcPointer[List[Int]]
    var right_starts: ArcPointer[List[Int]]
    var start: Int
    var end: Int
    var count: Int
    var include_unmatched: Bool

    def __init__(
        out self,
        left_ids: ArcPointer[List[Int]],
        right_starts: ArcPointer[List[Int]],
        start: Int,
        end: Int,
        include_unmatched: Bool = False,
    ):
        self.left_ids = left_ids.copy()
        self.right_starts = right_starts.copy()
        self.start = start
        self.end = end
        self.count = 0
        self.include_unmatched = include_unmatched

    def run(mut self) raises:
        for row in range(self.start, self.end):
            var key = self.left_ids[][row]
            var matches = 0
            if key >= 0:
                matches = (
                    self.right_starts[][key + 1] - self.right_starts[][key]
                )
            if matches == 0 and self.include_unmatched:
                matches = 1
            if matches > Int.MAX - self.count:
                raise Error("Join output row count overflows")
            self.count += matches


struct _JoinFillJob(Job):
    """Fill one already-counted inner-join output span."""

    var left_ids: ArcPointer[List[Int]]
    var right_starts: ArcPointer[List[Int]]
    var right_flat: ArcPointer[List[Int]]
    var start: Int
    var end: Int
    var output: Int
    var left_output: Int
    var right_output: Int
    var include_unmatched: Bool

    def __init__(
        out self,
        left_ids: ArcPointer[List[Int]],
        right_starts: ArcPointer[List[Int]],
        right_flat: ArcPointer[List[Int]],
        start: Int,
        end: Int,
        output: Int,
        left_output: Int,
        right_output: Int,
        include_unmatched: Bool,
    ):
        self.left_ids = left_ids.copy()
        self.right_starts = right_starts.copy()
        self.right_flat = right_flat.copy()
        self.start = start
        self.end = end
        self.output = output
        self.left_output = left_output
        self.right_output = right_output
        self.include_unmatched = include_unmatched

    def run(mut self) raises:
        ref left_rows = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.left_output
        )[]
        ref right_rows = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.right_output
        )[]
        var output = self.output
        for row in range(self.start, self.end):
            var key = self.left_ids[][row]
            if (
                key >= 0
                and self.right_starts[][key + 1] > self.right_starts[][key]
            ):
                for at in range(
                    self.right_starts[][key], self.right_starts[][key + 1]
                ):
                    left_rows[output] = row
                    right_rows[output] = self.right_flat[][at]
                    output += 1
            elif self.include_unmatched:
                left_rows[output] = row
                right_rows[output] = -1
                output += 1


def _parallel_join_rows(
    left_ids: List[Int],
    right_starts: List[Int],
    right_flat: List[Int],
    workers: Int,
    include_unmatched: Bool,
) raises -> Tuple[List[Int], List[Int]]:
    """Inner or left pairs in the same left-major order as the serial loop.

    Each worker owns a contiguous left-row range. Counting it first assigns
    a disjoint output span; prefixing spans in left-range order preserves
    the public order contract, and each CSR group is itself right-row order.
    """
    var shared_left = ArcPointer(left_ids.copy())
    var shared_starts = ArcPointer(right_starts.copy())
    var shared_flat = ArcPointer(right_flat.copy())
    var bounds = partitions(len(left_ids), workers, 1)
    var counts = List[_JoinCountJob](capacity=workers)
    for worker in range(workers):
        counts.append(
            _JoinCountJob(
                shared_left,
                shared_starts,
                bounds[worker],
                bounds[worker + 1],
                include_unmatched,
            )
        )
    run_jobs(counts)
    var outputs = List[Int](capacity=workers)
    var total = 0
    for worker in range(workers):
        outputs.append(total)
        if counts[worker].count > Int.MAX - total:
            raise Error("Join output row count overflows")
        total += counts[worker].count
    var left_rows = List[Int](length=total, fill=0)
    var right_rows = List[Int](length=total, fill=0)
    var fills = List[_JoinFillJob](capacity=workers)
    for worker in range(workers):
        fills.append(
            _JoinFillJob(
                shared_left,
                shared_starts,
                shared_flat,
                bounds[worker],
                bounds[worker + 1],
                outputs[worker],
                Int(Pointer(to=left_rows)),
                Int(Pointer(to=right_rows)),
                include_unmatched,
            )
        )
    run_jobs(fills)
    return (left_rows^, right_rows^)


struct _UnmatchedRightCountJob(Job):
    """Count right rows whose key never occurred on the left."""

    var right_ids: ArcPointer[List[Int]]
    var left_present: ArcPointer[List[Bool]]
    var start: Int
    var end: Int
    var count: Int

    def __init__(
        out self,
        right_ids: ArcPointer[List[Int]],
        left_present: ArcPointer[List[Bool]],
        start: Int,
        end: Int,
    ):
        self.right_ids = right_ids.copy()
        self.left_present = left_present.copy()
        self.start = start
        self.end = end
        self.count = 0

    def run(mut self) raises:
        for row in range(self.start, self.end):
            var key = self.right_ids[][row]
            if key < 0 or not self.left_present[][key]:
                self.count += 1


struct _UnmatchedRightFillJob(Job):
    """Append unmatched right rows in their original input order."""

    var right_ids: ArcPointer[List[Int]]
    var left_present: ArcPointer[List[Bool]]
    var start: Int
    var end: Int
    var output: Int
    var left_output: Int
    var right_output: Int

    def __init__(
        out self,
        right_ids: ArcPointer[List[Int]],
        left_present: ArcPointer[List[Bool]],
        start: Int,
        end: Int,
        output: Int,
        left_output: Int,
        right_output: Int,
    ):
        self.right_ids = right_ids.copy()
        self.left_present = left_present.copy()
        self.start = start
        self.end = end
        self.output = output
        self.left_output = left_output
        self.right_output = right_output

    def run(mut self) raises:
        ref left_rows = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.left_output
        )[]
        ref right_rows = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.right_output
        )[]
        var output = self.output
        for row in range(self.start, self.end):
            var key = self.right_ids[][row]
            if key < 0 or not self.left_present[][key]:
                left_rows[output] = -1
                right_rows[output] = row
                output += 1


def _parallel_full_join_rows(
    left_ids: List[Int],
    right_ids: List[Int],
    right_starts: List[Int],
    right_flat: List[Int],
    count: Int,
    workers: Int,
) raises -> Tuple[List[Int], List[Int]]:
    """Full pairs in left-major order, then unmatched right input order."""
    trace_path("join.full_parallel")
    var present = List[Bool](length=count, fill=False)
    for key in left_ids:
        if key >= 0:
            present[key] = True
    var shared_left = ArcPointer(left_ids.copy())
    var shared_right = ArcPointer(right_ids.copy())
    var shared_starts = ArcPointer(right_starts.copy())
    var shared_flat = ArcPointer(right_flat.copy())
    var shared_present = ArcPointer(present^)
    var left_bounds = partitions(len(left_ids), workers, 1)
    var right_bounds = partitions(len(right_ids), workers, 1)
    var left_counts = List[_JoinCountJob](capacity=workers)
    var right_counts = List[_UnmatchedRightCountJob](capacity=workers)
    for worker in range(workers):
        left_counts.append(
            _JoinCountJob(
                shared_left,
                shared_starts,
                left_bounds[worker],
                left_bounds[worker + 1],
                True,
            )
        )
        right_counts.append(
            _UnmatchedRightCountJob(
                shared_right,
                shared_present,
                right_bounds[worker],
                right_bounds[worker + 1],
            )
        )
    run_jobs(left_counts)
    run_jobs(right_counts)
    var left_outputs = List[Int](capacity=workers)
    var right_outputs = List[Int](capacity=workers)
    var total = 0
    for worker in range(workers):
        left_outputs.append(total)
        if left_counts[worker].count > Int.MAX - total:
            raise Error("Join output row count overflows")
        total += left_counts[worker].count
    for worker in range(workers):
        right_outputs.append(total)
        if right_counts[worker].count > Int.MAX - total:
            raise Error("Join output row count overflows")
        total += right_counts[worker].count
    var left_rows = List[Int](length=total, fill=0)
    var right_rows = List[Int](length=total, fill=0)
    var left_fills = List[_JoinFillJob](capacity=workers)
    var right_fills = List[_UnmatchedRightFillJob](capacity=workers)
    for worker in range(workers):
        left_fills.append(
            _JoinFillJob(
                shared_left,
                shared_starts,
                shared_flat,
                left_bounds[worker],
                left_bounds[worker + 1],
                left_outputs[worker],
                Int(Pointer(to=left_rows)),
                Int(Pointer(to=right_rows)),
                True,
            )
        )
        right_fills.append(
            _UnmatchedRightFillJob(
                shared_right,
                shared_present,
                right_bounds[worker],
                right_bounds[worker + 1],
                right_outputs[worker],
                Int(Pointer(to=left_rows)),
                Int(Pointer(to=right_rows)),
            )
        )
    run_jobs(left_fills)
    run_jobs(right_fills)
    return (left_rows^, right_rows^)
