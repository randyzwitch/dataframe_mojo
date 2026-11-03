"""DuckDB 2.0's sort (common/sorting/sort.cpp, sorted_run.cpp,
sorted_run_merger.cpp) and top-N heap (physical_top_n.cpp).

Sort, as a sink and a source:
- Sink: every thread encodes the rows it takes into a run of key records
  (`encode_keys`), and Combine sorts the run: vergesort, whose fallback for
  unsorted stretches is the in-place radix sort `ska_sort`
  (SortedRun::Finalize).
- Finalize: partitions of DEFAULT_ROW_GROUP_SIZE rows of the output.
- Source (SortedRunMerger): each partition's run boundaries come from a
  K-way merge path over the runs; each partition copies its records from
  every run and sorts them with vergesort, which finds the runs and merges
  them; partitions run in parallel and are scanned out in order. The
  payload is the row index, so the scan-out is the gather of the output
  columns by row (the row ids LateMaterialization sorts).

Top-N (PhysicalTopN): one heap per thread over the full key, holding the
best `limit` keys and their rows; the threads' heaps combine into one,
which is sorted. The heap reduces its key storage once it holds more than
ReduceThreshold entries' worth.

Callers come through `sort_rows`, which picks the top-N for a limit as
DuckDB's TopN optimizer does (topn_optimizer.cpp: a full sort once the
limit is over 5,000 rows and 0.7% of the input).
"""
from std.ffi import external_call
from std.memory import Pointer, unsafe_memcpy

from dataframe.parallel import Job, configured_workers, partitions, run_jobs
from dataframe.series import Series
from dataframe.sorting.algorithms import (
    MergeBuffer,
    Rec,
    SkaInfo,
    SortKeys,
    pdqsort_branchless,
    vergesort,
)
from dataframe.sorting.sort_key import (
    DEFAULT_ROW_GROUP_SIZE,
    STANDARD_VECTOR_SIZE,
    KeyBlock,
    SortKeyLayout,
    encode_keys,
)

# topn_optimizer.cpp, TopN::CanOptimize: a limit over this many rows and
# over 0.7% of the input sorts the whole input instead.
comptime _TOPN_LARGE_LIMIT = 5000
comptime _TOPN_INPUT_SHARE = 0.007
# physical_top_n.cpp, TopNHeap: SMALL_HEAP_THRESHOLD is not needed (it
# chooses how payload is copied; rows are the payload here).


def _sink_threads(rows: Int, threads: Int) -> Int:
    """Threads sinking `rows` input rows: one per row group (the scan's
    morsel), at most `threads` (0: the configured thread count)."""
    var groups = (rows + DEFAULT_ROW_GROUP_SIZE - 1) // DEFAULT_ROW_GROUP_SIZE
    var cap = threads if threads > 0 else configured_workers()
    return max(1, min(cap, groups))


# ===----------------------------------------------------------------------=== #
# Sort sink: one sorted run per thread
# ===----------------------------------------------------------------------=== #


struct _SinkRunJob[W: Int, VAR: Bool](Job):
    """One thread's Sink of rows [lo, hi) and Combine (sort the run)."""

    var layout: Int
    var lo: Int
    var hi: Int
    var block: KeyBlock

    def __init__(out self, layout: Int, lo: Int, hi: Int):
        self.layout = layout
        self.lo = lo
        self.hi = hi
        self.block = KeyBlock()

    def run(mut self) raises:
        ref layout = Pointer[SortKeyLayout, MutAnyOrigin](
            unsafe_from_address=self.layout
        )[]
        self.block = encode_keys(layout, self.lo, self.hi)
        var n = self.block.count
        var buffer = MergeBuffer(n, layout.record_words())
        var keys = SortKeys[Self.W, Self.VAR](self.block.address())
        vergesort[Self.W, Self.VAR, True](
            keys, 0, n, layout.info, buffer.address()
        )
        _ = buffer^


# ===----------------------------------------------------------------------=== #
# SortedRunMerger
# ===----------------------------------------------------------------------=== #


@always_inline
def _boundary_delta(total_remaining: Int, k: Int, begin: Int, end: Int) -> Int:
    """ComputeBoundaryDelta: ceil(total_remaining / k), or what the run
    has left."""
    return min((total_remaining + k - 1) // k, end - begin)


def _partition_ends[
    W: Int, VAR: Bool
](runs: List[Int], counts: List[Int], target: Int) -> List[Int]:
    """TemplatedComputePartitionBoundaries from the start of every run:
    per run, how many of its rows fall in the first `target` rows of the
    merged output (K-way Merge Path)."""
    comptime K = SortKeys[W, VAR]
    var num_runs = len(runs)
    var begins = List[Int](length=num_runs, fill=0)
    var ends = List[Int](capacity=num_runs)
    var active = List[Int]()
    for r in range(num_runs):
        ends.append(min(counts[r], target))
        if ends[r] != 0:
            active.append(r)
    var total_remaining = target
    while total_remaining != 0:
        var k = len(active)
        var min_idx = active[0]
        var min_delta = _boundary_delta(
            total_remaining, k, begins[min_idx], ends[min_idx]
        )
        var min_value = K(runs[min_idx]).get(begins[min_idx] + min_delta - 1)
        for i in range(1, len(active)):
            var r = active[i]
            var delta = _boundary_delta(total_remaining, k, begins[r], ends[r])
            var value = K(runs[r]).get(begins[r] + delta - 1)
            if K.less(value, min_value):
                min_idx = r
                min_delta = delta
                min_value = value
        begins[min_idx] += min_delta
        if begins[min_idx] == ends[min_idx]:
            for i in range(len(active)):
                if active[i] == min_idx:
                    _ = active.pop(i)
                    break
        total_remaining -= min_delta
    return begins^


struct _BoundaryJob[W: Int, VAR: Bool](Job):
    var runs: List[Int]
    var counts: List[Int]
    var target: Int
    var ends: List[Int]

    def __init__(out self, runs: List[Int], counts: List[Int], target: Int):
        self.runs = runs.copy()
        self.counts = counts.copy()
        self.target = target
        self.ends = List[Int]()

    def run(mut self) raises:
        self.ends = _partition_ends[Self.W, Self.VAR](
            self.runs, self.counts, self.target
        )


struct _MergePartitionJob[W: Int, VAR: Bool](Job):
    """TemplatedMergePartition and the partition's scan: its records from
    every run, sorted by vergesort (pdqsort for unsorted stretches), then
    its row ids written at the partition's place in the output."""

    var runs: List[Int]
    var begins: List[Int]
    var ends: List[Int]
    var info: SkaInfo
    var output: Int
    var start: Int

    def __init__(
        out self,
        runs: List[Int],
        begins: List[Int],
        ends: List[Int],
        info: SkaInfo,
        output: Int,
        start: Int,
    ):
        self.runs = runs.copy()
        self.begins = begins.copy()
        self.ends = ends.copy()
        self.info = info
        self.output = output
        self.start = start

    def run(mut self) raises:
        comptime K = SortKeys[Self.W, Self.VAR]
        comptime R = K.R
        var count = 0
        var active = 0
        for r in range(len(self.runs)):
            var n = self.ends[r] - self.begins[r]
            count += n
            if n > 0:
                active += 1
        var merged = List[UInt64](unsafe_uninit_length=max(1, count * R))
        var at = Int(merged.unsafe_ptr())
        for r in range(len(self.runs)):
            var n = self.ends[r] - self.begins[r]
            if n == 0:
                continue
            unsafe_memcpy(
                dest=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at),
                src=Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=K(self.runs[r]).addr(self.begins[r])
                ),
                count=n * R * 8,
            )
            at += n * R * 8
        var keys = K(Int(merged.unsafe_ptr()))
        if active > 1:
            var buffer = MergeBuffer(count, R)
            vergesort[Self.W, Self.VAR, False](
                keys, 0, count, self.info, buffer.address()
            )
            _ = buffer^
        var output = Pointer[Int, MutAnyOrigin](unsafe_from_address=self.output)
        for i in range(count):
            output[unsafe_offset=self.start + i] = keys.payload(i)
        _ = merged^


def _sorted_rows[
    W: Int, VAR: Bool
](layout: SortKeyLayout, threads_cap: Int) raises -> List[Int]:
    var n = layout.rows
    var order = List[Int](unsafe_uninit_length=n)
    if n == 0:
        return order^
    # Sink + Combine: a sorted run per thread.
    var threads = _sink_threads(n, threads_cap)
    var bounds = partitions(n, threads, STANDARD_VECTOR_SIZE)
    var jobs = List[_SinkRunJob[W, VAR]](capacity=threads)
    var layout_address = Int(Pointer(to=layout))
    for t in range(threads):
        if bounds[t] < bounds[t + 1]:
            jobs.append(
                _SinkRunJob[W, VAR](layout_address, bounds[t], bounds[t + 1])
            )
    if len(jobs) > 1:
        run_jobs(jobs)
    else:
        jobs[0].run()
    var runs = List[Int](capacity=len(jobs))
    var counts = List[Int](capacity=len(jobs))
    for j in range(len(jobs)):
        runs.append(jobs[j].block.address())
        counts.append(jobs[j].block.count)
    var output = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=Int(order.unsafe_ptr())
    )
    if len(runs) == 1:
        var keys = SortKeys[W, VAR](runs[0])
        for i in range(n):
            output[unsafe_offset=i] = keys.payload(i)
        _ = jobs^
        return order^
    # Finalize: partitions of the output; Source: each one merged.
    var partition_size = min(n, DEFAULT_ROW_GROUP_SIZE)
    var num_partitions = (n + partition_size - 1) // partition_size
    var boundary_jobs = List[_BoundaryJob[W, VAR]](capacity=num_partitions)
    for p in range(num_partitions - 1):
        boundary_jobs.append(
            _BoundaryJob[W, VAR](runs, counts, (p + 1) * partition_size)
        )
    if len(boundary_jobs) > 1:
        run_jobs(boundary_jobs)
    elif len(boundary_jobs) == 1:
        boundary_jobs[0].run()
    var merges = List[_MergePartitionJob[W, VAR]](capacity=num_partitions)
    var previous = List[Int](length=len(runs), fill=0)
    for p in range(num_partitions):
        var ends = (
            counts.copy() if p
            == num_partitions - 1 else boundary_jobs[p].ends.copy()
        )
        merges.append(
            _MergePartitionJob[W, VAR](
                runs,
                previous,
                ends,
                layout.info,
                Int(order.unsafe_ptr()),
                p * partition_size,
            )
        )
        previous = ends^
    if len(merges) > 1:
        run_jobs(merges)
    else:
        merges[0].run()
    _ = jobs^
    return order^


# ===----------------------------------------------------------------------=== #
# Top-N
# ===----------------------------------------------------------------------=== #


struct _KeyArena(Movable):
    """StringHeap for the heap's variable keys: blocks never move, so a
    key's address stays valid while the arena lives."""

    var blocks: List[List[UInt8]]
    var used: Int

    def __init__(out self):
        self.blocks = List[List[UInt8]]()
        self.used = 0

    def add(mut self, address: Int, size: Int) -> Int:
        comptime BLOCK = 1 << 16
        if len(self.blocks) == 0 or self.used + size > len(
            self.blocks[len(self.blocks) - 1]
        ):
            self.blocks.append(
                List[UInt8](unsafe_uninit_length=max(BLOCK, size))
            )
            self.used = 0
        var block = Int(self.blocks[len(self.blocks) - 1].unsafe_ptr())
        var at = block + self.used
        unsafe_memcpy(
            dest=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=at),
            src=Pointer[UInt8, MutAnyOrigin](unsafe_from_address=address),
            count=size,
        )
        self.used += size
        return at


struct TopNHeap[W: Int, VAR: Bool](Movable):
    """TopNHeap: a max-heap of the best `heap_size` key records seen (the
    worst on top), each with its row."""

    comptime K = SortKeys[Self.W, Self.VAR]
    var heap_size: Int
    var records: List[UInt64]
    var size: Int
    var arena: _KeyArena
    # Entries added since the last Reduce (heap_data's size in DuckDB).
    var added: Int

    def __init__(out self, heap_size: Int):
        self.heap_size = heap_size
        self.records = List[UInt64](
            length=max(1, (heap_size + 1) * Self.K.R), fill=0
        )
        self.size = 0
        self.arena = _KeyArena()
        self.added = 0

    def keys(self) -> Self.K:
        return Self.K(Int(self.records.unsafe_ptr()))

    def reduce_threshold(self) -> Int:
        return max(STANDARD_VECTOR_SIZE * 5, 2 * self.heap_size)

    @always_inline
    def should_add(self, rec: Rec) -> Bool:
        """EntryShouldBeAdded."""
        if self.heap_size == 0:
            return False
        if self.size < self.heap_size:
            return True
        return Self.K.less(rec, self.keys().get(0))

    def _sift_up(self, i_in: Int):
        var k = self.keys()
        var i = i_in
        var value = k.get(i)
        while i > 0:
            var parent = (i - 1) // 2
            var p = k.get(parent)
            if not Self.K.less(p, value):
                break
            k.put(i, p)
            i = parent
        k.put(i, value)

    def _sift_down(self, i_in: Int):
        var k = self.keys()
        var i = i_in
        var value = k.get(i)
        while True:
            var child = 2 * i + 1
            if child >= self.size:
                break
            var c = k.get(child)
            if child + 1 < self.size:
                var right = k.get(child + 1)
                if Self.K.less(c, right):
                    child += 1
                    c = right
            if not Self.K.less(value, c):
                break
            k.put(i, c)
            i = child
        k.put(i, value)

    def add(mut self, var rec: Rec):
        """AddEntryToHeap: replace the worst entry once full."""
        comptime if Self.VAR:
            rec[2] = UInt64(self.arena.add(Int(rec[2]), Int(rec[1])))
        self.added += 1
        if self.size >= self.heap_size:
            self.keys().put(0, rec)
            self._sift_down(0)
            return
        self.keys().put(self.size, rec)
        self.size += 1
        self._sift_up(self.size - 1)

    def sink(mut self, block: KeyBlock):
        """Sink: every record of a chunk that beats the heap's worst."""
        var k = Self.K(block.address())
        for i in range(block.count):
            var rec = k.get(i)
            if self.should_add(rec):
                self.add(rec)

    def reduce(mut self):
        """Reduce: past the threshold, keep only the live entries' keys."""
        comptime if Self.VAR:
            if self.added < self.reduce_threshold():
                return
            var arena = _KeyArena()
            var k = self.keys()
            for i in range(self.size):
                var rec = k.get(i)
                rec[2] = UInt64(arena.add(Int(rec[2]), Int(rec[1])))
                k.put(i, rec)
            self.arena = arena^
        self.added = self.size

    def finalize(self):
        """Finalize: the entries in key order."""
        pdqsort_branchless(self.keys(), 0, self.size)

    def combine(mut self, other: Self):
        """Combine: another thread's sorted heap, best first, until an
        entry can no longer enter."""
        var k = other.keys()
        for i in range(other.size):
            var rec = k.get(i)
            if not self.should_add(rec):
                break
            self.add(rec)
        self.reduce()


struct _TopNJob[W: Int, VAR: Bool](Job):
    """One thread's top-N Sink over rows [lo, hi), a vector at a time, and
    its Finalize."""

    var layout: Int
    var lo: Int
    var hi: Int
    var heap: TopNHeap[Self.W, Self.VAR]

    def __init__(out self, layout: Int, lo: Int, hi: Int, limit: Int):
        self.layout = layout
        self.lo = lo
        self.hi = hi
        self.heap = TopNHeap[Self.W, Self.VAR](limit)

    def run(mut self) raises:
        ref layout = Pointer[SortKeyLayout, MutAnyOrigin](
            unsafe_from_address=self.layout
        )[]
        var at = self.lo
        while at < self.hi:
            var end = min(self.hi, at + STANDARD_VECTOR_SIZE)
            var block = encode_keys(layout, at, end)
            self.heap.sink(block)
            self.heap.reduce()
            _ = block^
            at = end
        self.heap.finalize()


def _top_rows[
    W: Int, VAR: Bool
](layout: SortKeyLayout, limit: Int, threads_cap: Int) raises -> List[Int]:
    var n = layout.rows
    var threads = _sink_threads(n, threads_cap)
    var bounds = partitions(n, threads, STANDARD_VECTOR_SIZE)
    var jobs = List[_TopNJob[W, VAR]](capacity=threads)
    var layout_address = Int(Pointer(to=layout))
    for t in range(threads):
        if bounds[t] < bounds[t + 1]:
            jobs.append(
                _TopNJob[W, VAR](
                    layout_address, bounds[t], bounds[t + 1], limit
                )
            )
    if len(jobs) > 1:
        run_jobs(jobs)
    elif len(jobs) == 1:
        jobs[0].run()
    var order = List[Int]()
    if len(jobs) == 0:
        return order^
    var heap = TopNHeap[W, VAR](limit)
    for j in range(len(jobs)):
        heap.combine(jobs[j].heap)
    heap.finalize()
    order = List[Int](capacity=heap.size)
    var k = heap.keys()
    for i in range(heap.size):
        order.append(k.payload(i))
    _ = jobs^
    _ = heap^
    return order^


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


def sort_rows(
    keys: List[Series],
    descending: List[Bool],
    nulls_last: List[Bool],
    *,
    maintain_order: Bool = False,
    limit: Int = -1,
    threads: Int = 0,
) raises -> List[Int]:
    """The rows of the key columns in sorted order: all of them, or the
    first `limit` (a top-N). Equal keys come in unspecified order unless
    `maintain_order`, which keeps their input order. `threads` caps the
    sink's threads (0: the configured count)."""
    var layout = SortKeyLayout(keys, descending, nulls_last, maintain_order)
    var n = layout.rows
    var top = limit >= 0 and not (
        limit > _TOPN_LARGE_LIMIT
        and Float64(limit) > Float64(n) * _TOPN_INPUT_SHARE
    )
    if top:
        var take = min(limit, n)
        if layout.variable:
            return _top_rows[1, True](layout, take, threads)
        if layout.words == 1:
            return _top_rows[1, False](layout, take, threads)
        if layout.words == 2:
            return _top_rows[2, False](layout, take, threads)
        return _top_rows[3, False](layout, take, threads)
    var order: List[Int]
    if layout.variable:
        order = _sorted_rows[1, True](layout, threads)
    elif layout.words == 1:
        order = _sorted_rows[1, False](layout, threads)
    elif layout.words == 2:
        order = _sorted_rows[2, False](layout, threads)
    else:
        order = _sorted_rows[3, False](layout, threads)
    if limit >= 0 and limit < len(order):
        order.shrink(limit)
    return order^
