"""Parallel stable row sorting shared by rank and normalized-key orders."""
from std.memory import ArcPointer, Pointer
from .parallel import Job, Pool, configured_workers, partitions
from .trace import trace_path


trait KeyOrder(Copyable, Deinitable, Movable):
    def less(self, a: Int, b: Int) -> Bool:
        ...


def _merge_runs[
    K: KeyOrder
](
    ranks: K,
    source: List[Int],
    mut target: List[Int],
    start: Int,
    mid: Int,
    end: Int,
):
    """Merge two adjacent sorted runs, preferring the earlier on ties."""
    var left = start
    var right = mid
    for dest in range(start, end):
        if left < mid and (
            right >= end or not ranks.less(source[right], source[left])
        ):
            target[dest] = source[left]
            left += 1
        else:
            target[dest] = source[right]
            right += 1


def _co_rank[
    K: KeyOrder
](ranks: K, source: List[Int], start: Int, mid: Int, end: Int, k: Int,) -> Int:
    """How many of the first `k` merged outputs come from the left run.

    Splitting a merge across workers needs each worker to know where its
    output slice begins in *both* runs. For output position k there is
    exactly one split (i from the left, k - i from the right), because
    `less` breaks ties by row index and is therefore a total order --
    no two distinct rows compare equal, so no split is ambiguous. That also
    means the slices reproduce the serial merge exactly, including which run
    an equal-keyed row came from, so stability needs no special handling.

    Found by binary search on i, the classic merge-path co-rank.
    """
    var left_len = mid - start
    var right_len = end - mid
    var low = max(0, k - right_len)
    var high = min(k, left_len)
    while low < high:
        var i = (low + high) // 2
        var j = k - i
        # source[mid + j - 1] belongs before source[start + i]: take more
        # from the left run.
        if j > 0 and ranks.less(source[start + i], source[mid + j - 1]):
            low = i + 1
        else:
            high = i
    return low


def _merge_slice[
    K: KeyOrder
](
    ranks: K,
    source: List[Int],
    mut target: List[Int],
    start: Int,
    mid: Int,
    end: Int,
    first: Int,
    last: Int,
):
    """Merge only outputs [first, last) of merging [start, mid) and
    [mid, end), where both are offsets from `start`."""
    var i = _co_rank(ranks, source, start, mid, end, first)
    var j = first - i
    var left = start + i
    var right = mid + j
    for dest in range(start + first, start + last):
        if left < mid and (
            right >= end or not ranks.less(source[right], source[left])
        ):
            target[dest] = source[left]
            left += 1
        else:
            target[dest] = source[right]
            right += 1


def _sort_range[K: KeyOrder](ranks: K, start: Int, end: Int) -> List[Int]:
    """Stable bottom-up mergesort of rows [start, end), returned in order."""
    var n = end - start
    var indices = List[Int](capacity=n)
    for i in range(start, end):
        indices.append(i)
    var scratch = indices.copy()
    var width = 1
    while width < n:
        var at = 0
        while at < n:
            var mid = min(at + width, n)
            var stop = min(at + 2 * width, n)
            _merge_runs(ranks, indices, scratch, at, mid, stop)
            at = stop
        var old = indices^
        indices = scratch^
        scratch = old^
        width *= 2
    return indices^


struct _SortRangeJob[K: KeyOrder](Job):
    """Sort one contiguous row range into the shared output."""

    var ranks: ArcPointer[Self.K]
    var start: Int
    var end: Int
    var rows: List[Int]

    def __init__(out self, ranks: ArcPointer[Self.K], start: Int, end: Int):
        self.ranks = ranks.copy()
        self.start = start
        self.end = end
        self.rows = List[Int]()

    def run(mut self) raises:
        # The standard sorter costs less than full merge passes within a
        # worker's private run. The comparator uses row indices to break ties,
        # so its result is stable even if the sorter itself is not.
        var rows = List[Int](capacity=self.end - self.start)
        for i in range(self.start, self.end):
            rows.append(i)
        var shared = self.ranks.copy()
        ref ranks = shared[]

        def less(a: Int, b: Int) {imm ranks} -> Bool:
            return ranks.less(a, b)

        sort(rows, less)
        self.rows = rows^


struct _MergeJob[K: KeyOrder](Job):
    """Merge outputs [first, last) of two adjacent sorted runs of `source`
    into `target`, where first and last are offsets from `start`.

    A whole merge is the slice [0, end - start); splitting it lets one merge
    occupy every worker, which matters most in the last round, where the
    pairwise tree has only one merge left and it spans the whole array.
    """

    var ranks: ArcPointer[Self.K]
    var source: Int
    var target: Int
    var start: Int
    var mid: Int
    var end: Int
    var first: Int
    var last: Int

    def __init__(
        out self,
        ranks: ArcPointer[Self.K],
        source: Int,
        target: Int,
        start: Int,
        mid: Int,
        end: Int,
        first: Int,
        last: Int,
    ):
        self.ranks = ranks.copy()
        self.source = source
        self.target = target
        self.start = start
        self.mid = mid
        self.end = end
        self.first = first
        self.last = last

    def run(mut self) raises:
        ref out = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.target
        )[]
        ref src = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.source
        )[]
        _merge_slice(
            self.ranks[],
            src,
            out,
            self.start,
            self.mid,
            self.end,
            self.first,
            self.last,
        )


comptime _MIN_ROWS_PER_RUN = 8192


def comparison_arg_sort[
    K: KeyOrder
](var ranks: K, n: Int, first: List[Int] = List[Int]()) raises -> List[Int]:
    var workers = configured_workers()
    # One run per thread, not one per MIN_ROWS_PER_WORKER rows: that minimum
    # is sized for a linear scan, and it both caps a 1M-row sort at 15 runs
    # however many cores are free and leaves a 100k-row sort entirely serial.
    # Sorting a run is n log n, so shorter runs still repay their scheduling,
    # and the merge rounds below are themselves split across threads and so
    # do not lengthen as runs are added.
    var target = max(1, min(workers, n // _MIN_ROWS_PER_RUN))
    if target <= 1 or n < 2:
        return _sort_range(ranks, 0, n)

    var shared = ArcPointer(ranks^)
    if len(first) == n:
        var bucketed = _prefix_buckets(shared, first, workers)
        if bucketed:
            trace_path("sort.prefix_buckets")
            return bucketed.take()
    var bounds = partitions(n, target, 1)
    # partitions() can leave empty trailing ranges; keep only real ones.
    var starts = List[Int]()
    for w in range(len(bounds) - 1):
        if bounds[w + 1] > bounds[w]:
            starts.append(bounds[w])
    starts.append(n)
    var runs = len(starts) - 1
    if runs <= 1:
        return _sort_range(shared[], 0, n)

    # One pool for the run pass and every merge round that follows. Creating
    # threads per round cost about 1.27 ms of the sort at 32 threads, against
    # 32 us to wake this pool's, and a sort runs one round plus log2(runs)
    # merge rounds. The pool is released before returning, on every path.
    var pool = Pool(workers)
    var jobs = List[_SortRangeJob[K]](capacity=runs)
    for r in range(runs):
        jobs.append(_SortRangeJob[K](shared, starts[r], starts[r + 1]))
    pool.run(jobs)
    var indices = List[Int](length=n, fill=0)
    for r in range(runs):
        var at = starts[r]
        for i in range(len(jobs[r].rows)):
            indices[at + i] = jobs[r].rows[i]

    # Merge adjacent runs in rounds, alternating buffers. Each round halves
    # the number of merges, so the later rounds have fewer merges than there
    # are workers -- the last has one, spanning the whole array. Every merge
    # is therefore split into output slices, enough that a round has about
    # one slice per worker however few merges it contains.
    var scratch = List[Int](length=n, fill=0)
    var stride = 1
    while stride < runs:
        var source_address = Int(Pointer(to=indices))
        var merges = List[_MergeJob[K]]()
        var pending = (runs + 2 * stride - 1) // (2 * stride)
        var slices = max(1, (workers + pending - 1) // pending)
        var r = 0
        while r < runs:
            var start = starts[r]
            var mid = starts[min(r + stride, runs)]
            var end = starts[min(r + 2 * stride, runs)]
            if mid < end:
                var width = end - start
                var cuts = min(slices, width)
                for s in range(cuts):
                    var first = (width * s) // cuts
                    var last = (width * (s + 1)) // cuts
                    if last > first:
                        merges.append(
                            _MergeJob[K](
                                shared,
                                source_address,
                                Int(Pointer(to=scratch)),
                                start,
                                mid,
                                end,
                                first,
                                last,
                            )
                        )
            else:
                for i in range(start, end):
                    scratch[i] = indices[i]
            r += 2 * stride
        pool.run(merges)
        var old = indices^
        indices = scratch^
        scratch = old^
        stride *= 2
    pool.release()
    return indices^


struct _PrefixSortJob[K: KeyOrder](Job):
    var order: ArcPointer[Self.K]
    var rows: List[Int]

    def __init__(out self, order: ArcPointer[Self.K], var rows: List[Int]):
        self.order = order.copy()
        self.rows = rows^

    def run(mut self) raises:
        var shared = self.order.copy()
        ref order = shared[]

        def less(a: Int, b: Int) {imm order} -> Bool:
            return order.less(a, b)

        var rows = self.rows^
        sort(rows, less)
        self.rows = rows^


def _prefix_buckets[
    K: KeyOrder
](
    order: ArcPointer[K],
    first: List[Int],
    workers: Int,
) raises -> Optional[
    List[Int]
]:
    """Partition on the highest varying bits of the leading encoded word.

    Every more-significant bit is constant, so bucket concatenation is in
    sort order. A comparison sort within each bucket resolves all remaining
    keys, including full strings. Reject skew that would serialize work.
    """
    var n = len(first)
    var low = first[0]
    var high = low
    for value in first:
        low = min(low, value)
        high = max(high, value)
    # At most an 8 KiB histogram; use its wider prefix only when at least
    # 128 input rows amortize each potential bucket's job/allocation.
    var bucket_count = 1024 if n >= 1024 * 128 else 256
    var mask = UInt64(bucket_count - 1)
    var varying = UInt64(low) ^ UInt64(high)
    var shift = UInt64(0)
    while varying > mask:
        varying >>= 1
        shift += 1
    var counts = List[Int](length=bucket_count, fill=0)
    for value in first:
        var digit = Int(
            ((UInt64(value) ^ UInt64(0x8000_0000_0000_0000)) >> shift) & mask
        )
        counts[digit] += 1
    var occupied = 0
    var largest = 0
    for count in counts:
        occupied += Int(count > 0)
        largest = max(largest, count)
    if occupied < 4 or occupied > n // 64 or largest > n // min(4, workers):
        return None
    var buckets = List[List[Int]]()
    for count in counts:
        buckets.append(List[Int](capacity=count))
    for row in range(n):
        var digit = Int(
            ((UInt64(first[row]) ^ UInt64(0x8000_0000_0000_0000)) >> shift)
            & mask
        )
        buckets[digit].append(row)
    var jobs = List[_PrefixSortJob[K]]()
    while len(buckets) > 0:
        var rows = buckets.pop(0)
        if len(rows) > 0:
            jobs.append(_PrefixSortJob[K](order, rows^))
    var pool = Pool(min(workers, len(jobs)))
    pool.run(jobs)
    pool.release()
    var result = List[Int](capacity=n)
    for i in range(len(jobs)):
        result.extend(Span(jobs[i].rows))
    return result^
