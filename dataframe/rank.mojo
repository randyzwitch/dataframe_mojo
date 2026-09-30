"""`rank` of numeric columns, globally or per partition (#330).

Each valid row becomes one 128-bit integer: an order-preserving 64-bit key
of its value above its row index. Sorting those integers orders a group by
value with ties in row order, so one native sort replaces the dense-rank
sort and the index sort the general path does, and a single pass over runs
of equal keys assigns every method. NaN sorts above every number, and -0.0
equals 0.0, as the general path and Polars have it. Nulls are left out and
rank as null.

Rows are bucketed by partition with a counting sort, and partitions are
ranked on workers in contiguous ranges of about equal row counts. A
partition too large to share a worker with others (a global rank) is sorted
in parallel chunks that are then merged.
"""
from std.memory import bitcast

from .column import Column
from .dtype import DataType, NUMERIC_DTYPES
from .parallel import Job, configured_workers, run_jobs
from .series import Series
from .trace import trace_path

comptime _ORDINAL = 0
comptime _MIN = 1
comptime _MAX = 2
comptime _DENSE = 3
comptime _AVERAGE = 4

comptime _SIGN = UInt64(1) << 63
comptime _LOW = (UInt128(1) << 64) - 1
# A partition this large is sorted on all workers rather than one.
comptime _LARGE = 1 << 16
# Below this many rows, insertion sort beats the general sorter.
comptime _SMALL = 24


def _method_code(method: String) raises -> Int:
    if method == "ordinal":
        return _ORDINAL
    if method == "min":
        return _MIN
    if method == "max":
        return _MAX
    if method == "dense":
        return _DENSE
    if method == "average":
        return _AVERAGE
    raise Error("unknown rank method: " + method)


@always_inline
def _key[D: DType](value: Scalar[D], descending: Bool) -> UInt64:
    """An unsigned key that orders as the value does."""
    var key: UInt64
    comptime if D.is_floating_point():
        var x = value.cast[DType.float64]()
        if x != x:
            key = UInt64(0x7FF8000000000000) ^ _SIGN  # above +inf
        elif x == 0:
            key = _SIGN  # -0.0 ranks with 0.0
        else:
            var bits = bitcast[DType.uint64](x)
            key = ~bits if (bits & _SIGN) != 0 else bits ^ _SIGN
    elif D.is_unsigned():
        key = value.cast[DType.uint64]()
    else:
        key = value.cast[DType.int64]().cast[DType.uint64]() ^ _SIGN
    return ~key if descending else key


@always_inline
def _pack(key: UInt64, row: Int) -> UInt128:
    return (UInt128(key) << 64) | UInt128(row)


def _sort_small(mut values: List[UInt128]):
    for i in range(1, len(values)):
        var item = values[i]
        var j = i
        while j > 0 and values[j - 1] > item:
            values[j] = values[j - 1]
            j -= 1
        values[j] = item


struct _Outputs(ImplicitlyCopyable, Movable):
    """Addresses of the shared result buffers; rows never overlap."""

    var ints: Int
    var floats: Int
    var valid: Int
    var method: Int

    def __init__(out self, ints: Int, floats: Int, valid: Int, method: Int):
        self.ints = ints
        self.floats = floats
        self.valid = valid
        self.method = method

    def assign(self, sorted: Span[UInt128, _]):
        """Rank one sorted partition."""
        self.assign_runs(
            sorted.unsafe_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin](),
            len(sorted),
            0,
            len(sorted),
            0,
        )

    def assign_runs(
        self,
        sorted: Pointer[UInt128, MutAnyOrigin],
        size: Int,
        first: Int,
        last: Int,
        dense_before: Int,
        row_bits: Int = 64,
    ):
        """Rank the runs of equal keys that start in [first, last) of a
        sorted partition of `size` values; a run may end past `last`.
        `dense_before` counts the runs before `first`. The low `row_bits`
        of each value are its row and the bits above them its key."""
        var mask = (UInt128(1) << UInt128(row_bits)) - 1
        var shift = UInt128(row_bits)
        var ints = Pointer[Int64, MutAnyOrigin](unsafe_from_address=self.ints)
        var floats = Pointer[Float64, MutAnyOrigin](
            unsafe_from_address=self.floats
        )
        var valid = Pointer[Bool, MutAnyOrigin](unsafe_from_address=self.valid)
        if self.method == _ORDINAL:
            for p in range(first, last):
                var row = Int(sorted[unsafe_offset=p] & mask)
                valid[unsafe_offset=row] = True
                ints[unsafe_offset=row] = Int64(p + 1)
            return
        var start = first
        var dense = dense_before
        while start < last:
            var key = sorted[unsafe_offset=start] >> shift
            var end = start + 1
            while end < size and sorted[unsafe_offset=end] >> shift == key:
                end += 1
            dense += 1
            for p in range(start, end):
                var row = Int(sorted[unsafe_offset=p] & mask)
                valid[unsafe_offset=row] = True
                if self.method == _AVERAGE:
                    floats[unsafe_offset=row] = Float64(start + 1 + end) / 2
                elif self.method == _MIN:
                    ints[unsafe_offset=row] = Int64(start + 1)
                elif self.method == _MAX:
                    ints[unsafe_offset=row] = Int64(end)
                else:
                    ints[unsafe_offset=row] = Int64(dense)
            start = end


struct _RankGroupsJob(Job):
    """Sort and rank partitions [first, last) of the packed values in
    place; each partition is a contiguous slice."""

    var packed: Int
    var starts: Int
    var first: Int
    var last: Int
    var outputs: _Outputs

    def __init__(
        out self,
        packed: Int,
        starts: Int,
        first: Int,
        last: Int,
        outputs: _Outputs,
    ):
        self.packed = packed
        self.starts = starts
        self.first = first
        self.last = last
        self.outputs = outputs

    def run(mut self) raises:
        ref packed = Pointer[List[UInt128], MutAnyOrigin](
            unsafe_from_address=self.packed
        )[]
        var base = packed.unsafe_ptr()
        var starts = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.starts
        )[].unsafe_ptr()
        for g in range(self.first, self.last):
            var start = starts[unsafe_offset=g]
            var m = starts[unsafe_offset=g + 1] - start
            if m >= _LARGE:
                continue  # sorted on every worker afterwards
            var slice = base.unsafe_offset(start)
            if m <= _SMALL:
                for i in range(1, m):
                    var item = slice[unsafe_offset=i]
                    var j = i
                    while j > 0 and slice[unsafe_offset=j - 1] > item:
                        slice[unsafe_offset=j] = slice[unsafe_offset=j - 1]
                        j -= 1
                    slice[unsafe_offset=j] = item
            else:
                sort(Span(packed)[start : start + m])
            self.outputs.assign_runs(slice, m, 0, m, 0)


struct _CountJob(Job):
    """Count valid rows per partition in rows [first, last)."""

    var column: Int
    var ids: Int
    var first: Int
    var last: Int
    var counts: List[UInt32]
    var all_valid: Bool

    def __init__(
        out self,
        column: Int,
        ids: Int,
        first: Int,
        last: Int,
        groups: Int,
        all_valid: Bool,
    ):
        self.column = column
        self.ids = ids
        self.first = first
        self.last = last
        self.counts = List[UInt32](length=groups, fill=0)
        self.all_valid = all_valid

    def run(mut self) raises:
        ref bits = Pointer[List[Bool], MutAnyOrigin](
            unsafe_from_address=self.column
        )[]
        ref ids = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.ids
        )[]
        var partitioned = len(ids) > 0
        var counts = self.counts.unsafe_ptr()
        if not partitioned:
            for i in range(self.first, self.last):
                if self.all_valid or bits[i]:
                    counts[unsafe_offset=0] += 1
            return
        var id = ids.unsafe_ptr()
        for i in range(self.first, self.last):
            if self.all_valid or bits[i]:
                counts[unsafe_offset=id[unsafe_offset=i]] += 1


struct _ScatterJob[D: DType](Job):
    """Write the packed key of each valid row in [first, last) to its
    partition's next slot; `next` holds this job's slot per partition."""

    var column: Int
    var valid: Int
    var ids: Int
    var packed: Int
    var first: Int
    var last: Int
    var next: List[UInt32]
    var descending: Bool
    var all_valid: Bool

    def __init__(
        out self,
        column: Int,
        valid: Int,
        ids: Int,
        packed: Int,
        first: Int,
        last: Int,
        var next: List[UInt32],
        descending: Bool,
        all_valid: Bool,
    ):
        self.column = column
        self.valid = valid
        self.ids = ids
        self.packed = packed
        self.first = first
        self.last = last
        self.next = next^
        self.descending = descending
        self.all_valid = all_valid

    def run(mut self) raises:
        ref column = Pointer[Column[Scalar[Self.D]], MutAnyOrigin](
            unsafe_from_address=self.column
        )[]
        ref valid = Pointer[List[Bool], MutAnyOrigin](
            unsafe_from_address=self.valid
        )[]
        ref ids = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.ids
        )[]
        var out = Pointer[List[UInt128], MutAnyOrigin](
            unsafe_from_address=self.packed
        )[].unsafe_ptr()
        var partitioned = len(ids) > 0
        var next = self.next.unsafe_ptr()
        var id = ids.unsafe_ptr()
        for i in range(self.first, self.last):
            if not self.all_valid and not valid[i]:
                continue
            var g = id[unsafe_offset=i] if partitioned else 0
            var slot = next[unsafe_offset=g]
            out[unsafe_offset=Int(slot)] = _pack(
                _key[Self.D](column._get(i), self.descending), i
            )
            next[unsafe_offset=g] = slot + 1


struct _SortRunJob(Job):
    """Sort one run of a large partition in place."""

    var data: Int
    var start: Int
    var end: Int

    def __init__(out self, data: Int, start: Int, end: Int):
        self.data = data
        self.start = start
        self.end = end

    def run(mut self) raises:
        ref data = Pointer[List[UInt128], MutAnyOrigin](
            unsafe_from_address=self.data
        )[]
        var run = List[UInt128](capacity=self.end - self.start)
        run.extend(Span(data)[self.start : self.end])
        sort(run)
        for i in range(len(run)):
            data[self.start + i] = run[i]


struct _MergeRunsJob(Job):
    """Write outputs [first, last) of merging adjacent sorted runs
    [start, mid) and [mid, end). Packed values are distinct (each holds its
    row), so a piece's inputs are found by co-ranking with no tie rule, and
    one merge can be split across every worker."""

    var source: Int
    var target: Int
    var start: Int
    var mid: Int
    var end: Int
    var first: Int
    var last: Int

    def __init__(
        out self,
        source: Int,
        target: Int,
        start: Int,
        mid: Int,
        end: Int,
        first: Int,
        last: Int,
    ):
        self.source = source
        self.target = target
        self.start = start
        self.mid = mid
        self.end = end
        self.first = first
        self.last = last

    def _split(self, src: Pointer[UInt128, MutAnyOrigin], k: Int) -> Int:
        """How many of the first k merged values come from the left run."""
        var left = self.mid - self.start
        var right = self.end - self.mid
        var lo = max(0, k - right)
        var hi = min(k, left)
        while lo < hi:
            var i = (lo + hi) // 2
            if (
                src[unsafe_offset=self.start + i]
                < src[unsafe_offset=self.mid + k - i - 1]
            ):
                lo = i + 1
            else:
                hi = i
        return lo

    def run(mut self) raises:
        var src = Pointer[List[UInt128], MutAnyOrigin](
            unsafe_from_address=self.source
        )[].unsafe_ptr()
        var dst = Pointer[List[UInt128], MutAnyOrigin](
            unsafe_from_address=self.target
        )[].unsafe_ptr()
        var k = self.first - self.start
        var from_left = self._split(src, k)
        var i = self.start + from_left
        var j = self.mid + (k - from_left)
        for at in range(self.first, self.last):
            if j >= self.end or (
                i < self.mid and src[unsafe_offset=i] < src[unsafe_offset=j]
            ):
                dst[unsafe_offset=at] = src[unsafe_offset=i]
                i += 1
            else:
                dst[unsafe_offset=at] = src[unsafe_offset=j]
                j += 1


def _sort_parallel(
    var data: List[UInt128], workers: Int
) raises -> List[UInt128]:
    """Sort on `workers` threads: one run each, then merge rounds whose
    merges are split so every round keeps all workers busy."""
    var n = len(data)
    var bounds = List[Int]()
    for w in range(workers + 1):
        bounds.append(n * w // workers)
    var jobs = List[_SortRunJob]()
    for w in range(workers):
        jobs.append(
            _SortRunJob(Int(Pointer(to=data)), bounds[w], bounds[w + 1])
        )
    run_jobs(jobs)
    var other = List[UInt128](length=n, fill=0)
    var from_data = True
    while len(bounds) > 2:
        var merges = List[_MergeRunsJob]()
        var next = List[Int]()
        var source = Int(Pointer(to=data)) if from_data else Int(
            Pointer(to=other)
        )
        var target = Int(Pointer(to=other)) if from_data else Int(
            Pointer(to=data)
        )
        var pairs = (len(bounds) - 1 + 1) // 2
        var pieces = max(1, workers // pairs)
        var k = 0
        while k + 1 < len(bounds):
            var start = bounds[k]
            var mid = bounds[k + 1]
            var end = bounds[k + 2] if k + 2 < len(bounds) else mid
            for p in range(pieces):
                merges.append(
                    _MergeRunsJob(
                        source,
                        target,
                        start,
                        mid,
                        end,
                        start + (end - start) * p // pieces,
                        start + (end - start) * (p + 1) // pieces,
                    )
                )
            next.append(start)
            k += 2
        next.append(n)
        run_jobs(merges)
        bounds = next^
        from_data = not from_data
    if from_data:
        return data^
    return other^


struct _CountRunsJob(Job):
    """Count positions in [first, last) that start a run of equal keys."""

    var data: Int
    var first: Int
    var last: Int
    var count: Int

    def __init__(out self, data: Int, first: Int, last: Int):
        self.data = data
        self.first = first
        self.last = last
        self.count = 0

    def run(mut self) raises:
        var sorted = Pointer[List[UInt128], MutAnyOrigin](
            unsafe_from_address=self.data
        )[].unsafe_ptr()
        for p in range(self.first, self.last):
            if (
                p == 0
                or sorted[unsafe_offset=p] >> 64
                != sorted[unsafe_offset=p - 1] >> 64
            ):
                self.count += 1


struct _AssignJob(Job):
    """Rank the runs of equal keys that start in [first, last)."""

    var data: Int
    var size: Int
    var first: Int
    var last: Int
    var dense: Int
    var outputs: _Outputs

    def __init__(
        out self,
        data: Int,
        size: Int,
        first: Int,
        last: Int,
        dense: Int,
        outputs: _Outputs,
    ):
        self.data = data
        self.size = size
        self.first = first
        self.last = last
        self.dense = dense
        self.outputs = outputs

    def run(mut self) raises:
        var sorted = Pointer[List[UInt128], MutAnyOrigin](
            unsafe_from_address=self.data
        )[].unsafe_ptr()
        var start = self.first
        # A run of ties begun before this range belongs to the previous
        # job; ordinal ranks each position alone, so it takes them all.
        while (
            self.outputs.method != _ORDINAL
            and start < self.last
            and start > 0
            and sorted[unsafe_offset=start] >> 64
            == sorted[unsafe_offset=start - 1] >> 64
        ):
            start += 1
        self.outputs.assign_runs(
            sorted, self.size, start, self.last, self.dense
        )


def _assign_parallel(
    sorted: List[UInt128], outputs: _Outputs, workers: Int
) raises:
    var n = len(sorted)
    var address = Int(Pointer(to=sorted))
    var counts = List[_CountRunsJob]()
    for w in range(workers):
        counts.append(
            _CountRunsJob(address, n * w // workers, n * (w + 1) // workers)
        )
    if outputs.method == _DENSE:
        run_jobs(counts)
    var jobs = List[_AssignJob]()
    var dense = 0
    for w in range(workers):
        jobs.append(
            _AssignJob(
                address,
                n,
                n * w // workers,
                n * (w + 1) // workers,
                dense,
                outputs,
            )
        )
        if outputs.method == _DENSE:
            dense += counts[w].count
    run_jobs(jobs)


struct _PackGroupedJob[D: DType](Job):
    """Pack rows [first, last) as partition:32 | key:64 | row:32 into
    their slots (each job's valid rows go to a precomputed offset)."""

    var column: Int
    var valid: Int
    var ids: Int
    var packed: Int
    var first: Int
    var last: Int
    var offset: Int
    var descending: Bool

    def __init__(
        out self,
        column: Int,
        valid: Int,
        ids: Int,
        packed: Int,
        first: Int,
        last: Int,
        offset: Int,
        descending: Bool,
    ):
        self.column = column
        self.valid = valid
        self.ids = ids
        self.packed = packed
        self.first = first
        self.last = last
        self.offset = offset
        self.descending = descending

    def run(mut self) raises:
        ref column = Pointer[Column[Scalar[Self.D]], MutAnyOrigin](
            unsafe_from_address=self.column
        )[]
        ref valid = Pointer[List[Bool], MutAnyOrigin](
            unsafe_from_address=self.valid
        )[]
        ref ids = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.ids
        )[]
        var out = Pointer[List[UInt128], MutAnyOrigin](
            unsafe_from_address=self.packed
        )[].unsafe_ptr()
        var at = self.offset
        for i in range(self.first, self.last):
            if not valid[i]:
                continue
            var key = _key[Self.D](column._get(i), self.descending)
            out[unsafe_offset=at] = (
                (UInt128(ids[i]) << 96) | (UInt128(key) << 32) | UInt128(i)
            )
            at += 1


struct _AssignGroupedJob(Job):
    """Rank every partition that starts in [first, last) of the sorted
    partition:key:row values; boundaries are moved to partition starts."""

    var data: Int
    var first: Int
    var last: Int
    var outputs: _Outputs

    def __init__(out self, data: Int, first: Int, last: Int, outputs: _Outputs):
        self.data = data
        self.first = first
        self.last = last
        self.outputs = outputs

    def run(mut self) raises:
        ref data = Pointer[List[UInt128], MutAnyOrigin](
            unsafe_from_address=self.data
        )[]
        var sorted = data.unsafe_ptr()
        var n = len(data)
        var start = self.first
        while start < self.last:
            var group = sorted[unsafe_offset=start] >> 96
            var end = start + 1
            while end < n and sorted[unsafe_offset=end] >> 96 == group:
                end += 1
            self.outputs.assign_runs(
                sorted.unsafe_offset(start), end - start, 0, end - start, 0, 32
            )
            start = end


def _rank_by_one_sort[
    D: DType
](
    column: Column[Scalar[D]],
    ids: List[Int],
    valid: List[Bool],
    outputs: _Outputs,
    descending: Bool,
    workers: Int,
) raises:
    """Many small partitions: sort every row once by (partition, key, row)
    on all workers, then rank each partition's run."""
    var n = len(column)
    var threads = max(1, min(workers, n // 65536))
    var bounds = List[Int]()
    for w in range(threads + 1):
        bounds.append(n * w // threads)
    var offsets = List[Int](length=threads + 1, fill=0)
    for w in range(threads):
        var present = 0
        for i in range(bounds[w], bounds[w + 1]):
            if valid[i]:
                present += 1
        offsets[w + 1] = offsets[w] + present
    var packed = List[UInt128](length=offsets[threads], fill=0)
    var jobs = List[_PackGroupedJob[D]]()
    for w in range(threads):
        jobs.append(
            _PackGroupedJob[D](
                Int(Pointer(to=column)),
                Int(Pointer(to=valid)),
                Int(Pointer(to=ids)),
                Int(Pointer(to=packed)),
                bounds[w],
                bounds[w + 1],
                offsets[w],
                descending,
            )
        )
    run_jobs(jobs)
    var sorted = _sort_parallel(packed^, threads)
    var total = len(sorted)
    # Each worker takes whole partitions: move boundaries to a partition
    # start so no partition is split.
    var cuts = List[Int]()
    for w in range(threads + 1):
        var cut = total * w // threads
        while (
            cut > 0
            and cut < total
            and (sorted[cut] >> 96 == sorted[cut - 1] >> 96)
        ):
            cut += 1
        cuts.append(cut)
    var assign = List[_AssignGroupedJob]()
    for w in range(threads):
        if cuts[w] < cuts[w + 1]:
            assign.append(
                _AssignGroupedJob(
                    Int(Pointer(to=sorted)), cuts[w], cuts[w + 1], outputs
                )
            )
    run_jobs(assign)
    _ = sorted^


def _rank_typed[
    D: DType
](
    column: Column[Scalar[D]],
    ids: List[Int],
    method: Int,
    descending: Bool,
) raises -> Series:
    var n = len(column)
    var count = 1
    for id in ids:
        count = max(count, id + 1)
    var valid = List[Bool](capacity=n)
    var all_valid = True
    for i in range(n):
        var present = column._valid(i)
        valid.append(present)
        all_valid = all_valid and present
    var workers = configured_workers()
    # Per-worker counts cost workers x partitions; with very many
    # partitions one worker buckets alone.
    var threads = max(1, min(workers, n // 65536))
    if (
        count * threads > 4 * n
        and threads > 1
        and n < (1 << 32)
        and count < (1 << 32)
    ):
        # Too many partitions to count per worker: one parallel sort with
        # the partition in the key's top bits instead.
        var average = method == _AVERAGE
        var ints = List[Int64](length=0 if average else n, fill=0)
        var floats = List[Float64](length=n if average else 0, fill=0)
        var out_valid = List[Bool](length=n, fill=False)
        var outputs = _Outputs(
            Int(ints.unsafe_ptr()),
            Int(floats.unsafe_ptr()),
            Int(out_valid.unsafe_ptr()),
            method,
        )
        trace_path("rank.one_sort")
        _rank_by_one_sort[D](column, ids, valid, outputs, descending, workers)
        if average:
            return Series("", Column[Float64](floats^, out_valid))
        return Series("", Column[Int64](ints^, out_valid))
    if count * threads > 4 * n:
        threads = 1
    var column_address = Int(Pointer(to=column))
    var valid_address = Int(Pointer(to=valid))
    var ids_address = Int(Pointer(to=ids))
    var counts = List[_CountJob]()
    for w in range(threads):
        counts.append(
            _CountJob(
                valid_address,
                ids_address,
                n * w // threads,
                n * (w + 1) // threads,
                count,
                all_valid,
            )
        )
    run_jobs(counts)
    # Partition g occupies [starts[g], starts[g + 1]); within it, worker w's
    # rows come after those of workers before it, keeping row order.
    var starts = List[Int](length=count + 1, fill=0)
    var offsets = List[List[UInt32]](length=threads, fill=List[UInt32]())
    for w in range(threads):
        offsets[w] = List[UInt32](length=count, fill=0)
    var total = 0
    for g in range(count):
        starts[g] = total
        for w in range(threads):
            offsets[w][g] = UInt32(total)
            total += Int(counts[w].counts[g])
    starts[count] = total
    var packed = List[UInt128](length=total, fill=0)
    var scatter = List[_ScatterJob[D]]()
    for w in range(threads):
        scatter.append(
            _ScatterJob[D](
                column_address,
                valid_address,
                ids_address,
                Int(Pointer(to=packed)),
                n * w // threads,
                n * (w + 1) // threads,
                offsets[w].copy(),
                descending,
                all_valid,
            )
        )
    run_jobs(scatter)
    var average = method == _AVERAGE
    var ints = List[Int64](length=0 if average else n, fill=0)
    var floats = List[Float64](length=n if average else 0, fill=0)
    var out_valid = List[Bool](length=n, fill=False)
    var outputs = _Outputs(
        Int(ints.unsafe_ptr()),
        Int(floats.unsafe_ptr()),
        Int(out_valid.unsafe_ptr()),
        method,
    )
    var parts = max(1, min(workers, total // 4096))
    var jobs = List[_RankGroupsJob]()
    var first = 0
    for w in range(parts):
        var target = total * (w + 1) // parts
        var last = first
        while last < count and (starts[last] < target or w == parts - 1):
            last += 1
        jobs.append(
            _RankGroupsJob(
                Int(Pointer(to=packed)),
                Int(Pointer(to=starts)),
                first,
                last,
                outputs,
            )
        )
        first = last
    run_jobs(jobs)
    for g in range(count):
        var size = starts[g + 1] - starts[g]
        if size < _LARGE:
            continue
        var part = List[UInt128](capacity=size)
        part.extend(Span(packed)[starts[g] : starts[g + 1]])
        var threads_here = max(1, min(workers, size // 4096))
        var sorted = _sort_parallel(part^, threads_here)
        _assign_parallel(sorted, outputs, threads_here)
    # Jobs reached these by address; keep them alive until every job ran.
    _ = valid^
    _ = packed^
    _ = starts^
    if average:
        return Series("", Column[Float64](floats^, out_valid))
    return Series("", Column[Int64](ints^, out_valid))


def rank_numeric(
    input: Series, ids: List[Int], method: String, descending: Bool
) raises -> Optional[Series]:
    """Ranks per partition (`ids`, or one partition when empty) for a
    fixed-width numeric or temporal column; None for other storage."""
    var code = _method_code(method)
    if len(input) >= (1 << 32):
        return None  # rows are packed into 32 or 64 bits
    comptime for i in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[i]
        if input._data.isa[Column[Scalar[D]]]():
            return _rank_typed[D](
                input._data[Column[Scalar[D]]], ids, code, descending
            )
    return None
