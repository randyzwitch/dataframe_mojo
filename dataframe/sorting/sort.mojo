"""Sort normalized key records in runs and merge their row order.

Encode ordering, null placement and stability in the keys. Sort independent
runs, then merge them into a global order. A small requested prefix uses a
bounded top-N heap; a larger prefix uses the full-sort route to avoid the
heap's repeated maintenance cost when most rows would be retained.
"""
from std.collections import Optional
from std.ffi import external_call
from std.memory import Pointer, unsafe_memcpy

from dataframe.parallel import (
    Job,
    Rounds,
    configured_workers,
    partitions,
    round_width,
    run_jobs,
    run_rounds,
)
from std.memory import ArcPointer
from dataframe.series import Series
from dataframe.sorting.algorithms import (
    MergeBuffer,
    Rec,
    SkaInfo,
    SortKeys,
    pdqsort_branchless,
    vergesort,
)
from dataframe.sorting.top_n import TopNBoundary, TopNHeap
from dataframe.sorting.sort_key import (
    DEFAULT_ROW_GROUP_SIZE,
    STANDARD_VECTOR_SIZE,
    KeyBlock,
    SortKeyLayout,
    encode_keys,
    key_prefixes,
    encode_selected,
    has_row_key,
)

# A limit over this many rows and over 0.7% of the input sorts the whole input
# instead: a top-N heap tests every row and does log N work for each that
# enters, and with a large N many rows enter, so the parallel full sort is
# cheaper. Both are empirical choices.
comptime _TOPN_LARGE_LIMIT = 5000
comptime _TOPN_INPUT_SHARE = 0.007


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
# Merging sorted runs
# ===----------------------------------------------------------------------=== #


@always_inline
def _boundary_delta(total_remaining: Int, k: Int, begin: Int, end: Int) -> Int:
    """The rows per merge task, ceil(total_remaining / k), or what the run has
    left."""
    return min((total_remaining + k - 1) // k, end - begin)


def _partition_ends[
    W: Int, VAR: Bool
](runs: List[Int], counts: List[Int], target: Int) -> List[Int]:
    """From the start of every run: per run, how many of its rows fall in the
    first `target` rows of the merged output (K-way Merge Path), found by binary
    search so each merge partition can be built independently."""
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
    """Merge one partition and scan it: its records from every run, sorted by
    vergesort (pdqsort for unsorted stretches), then its row ids written at the
    partition's place in the output."""

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


# The stages of a `SortRounds`: the round that ran last.
comptime _SORT_START = 0
comptime _SORT_SINK = 1
comptime _SORT_BOUNDARY = 2
comptime _SORT_MERGE = 3
comptime _SORT_DONE = 4


def _sink_task[
    W: Int, VAR: Bool
](layout: Int, lo: Int, hi: Int) raises -> KeyBlock:
    """Rows [lo, hi) encoded and sorted into one run."""
    var job = _SinkRunJob[W, VAR](layout, lo, hi)
    job.run()
    var block = KeyBlock()
    swap(block, job.block)
    return block^


def _boundary_task[
    W: Int, VAR: Bool
](runs: List[Int], counts: List[Int], target: Int) -> List[Int]:
    """Per run, its rows among the first `target` of the merged order."""
    return _partition_ends[W, VAR](runs, counts, target)


def _merge_task[
    W: Int, VAR: Bool
](
    runs: List[Int],
    begins: List[Int],
    ends: List[Int],
    info: SkaInfo,
    output: Int,
    start: Int,
) raises:
    """One partition of the merged order, written from `start`."""
    var job = _MergePartitionJob[W, VAR](
        runs, begins, ends, info, output, start
    )
    job.run()


struct SortRounds(Movable, Rounds):
    """The sorted order of the key columns' rows, as rounds of tasks: each
    task of the first round encodes the keys of one range of rows and
    sorts them into a run; the second finds, for every partition of the
    merged order (one row group of rows each), where it starts in every
    run, by binary search; the third merges each partition's records from
    every run and writes its rows at its place in the order. Each round
    needs only the one before it, so the rounds can run as tasks of a
    query's schedule or, with `run_rounds`, as jobs. `take_order` once no
    round is left."""

    var layout: ArcPointer[SortKeyLayout]
    var threads_cap: Int
    var stage: Int
    var bounds: List[Int]
    var blocks: List[KeyBlock]
    var runs: List[Int]
    var counts: List[Int]
    var partition_size: Int
    var partitions: Int
    # By partition but the last: per run, its rows before the end of the
    # partition.
    var ends: List[List[Int]]
    var width: Int
    var order: List[Int]

    def __init__(
        out self,
        keys: List[Series],
        descending: List[Bool],
        nulls_last: List[Bool],
        maintain_order: Bool,
        threads: Int = 0,
    ) raises:
        """The order of `keys`' rows; equal keys keep input order when
        `maintain_order`. `threads` caps the first round's tasks (0: the
        configured count)."""
        self = Self(
            SortKeyLayout(keys, descending, nulls_last, maintain_order),
            threads,
        )

    def __init__(out self, var layout: SortKeyLayout, threads: Int = 0):
        """The order of the rows `layout` encodes."""
        self.layout = ArcPointer(layout^)
        self.threads_cap = threads
        self.stage = _SORT_START
        self.bounds = List[Int]()
        self.blocks = List[KeyBlock]()
        self.runs = List[Int]()
        self.counts = List[Int]()
        self.partition_size = 0
        self.partitions = 0
        self.ends = List[List[Int]]()
        self.width = 0
        self.order = List[Int]()

    def _variant(self) -> Int:
        """Which records the layout's keys encode as: 0 for variable
        width, else the words per record (1, 2 or 3)."""
        ref layout = self.layout[]
        if layout.variable:
            return 0
        if layout.words == 1 or layout.words == 2:
            return layout.words
        return 3

    def next_round(mut self) raises -> Int:
        var n = self.layout[].rows
        if self.stage == _SORT_START:
            self.order = List[Int](unsafe_uninit_length=n)
            if n == 0:
                self.stage = _SORT_DONE
                return 0
            var threads = _sink_threads(n, self.threads_cap)
            var bounds = partitions(n, threads, STANDARD_VECTOR_SIZE)
            self.bounds.append(0)
            for t in range(threads):
                if bounds[t] < bounds[t + 1]:
                    self.bounds.append(bounds[t + 1])
            var tasks = len(self.bounds) - 1
            for _ in range(tasks):
                self.blocks.append(KeyBlock())
            self.stage = _SORT_SINK
            return tasks
        if self.stage == _SORT_SINK:
            for b in range(len(self.blocks)):
                self.runs.append(self.blocks[b].address())
                self.counts.append(self.blocks[b].count)
            if len(self.runs) == 1:
                # One run is the order already. The copy reads the run's
                # records only through their address, so the blocks move
                # to a local that is dropped after it: a value ends at its
                # last named use, and reassigning the field here would let
                # the records be freed before the copy reads them.
                var blocks = List[KeyBlock]()
                swap(blocks, self.blocks)
                var output = Int(self.order.unsafe_ptr())
                var variant = self._variant()
                if variant == 0:
                    _copy_payloads[1, True](self.runs[0], n, output)
                elif variant == 1:
                    _copy_payloads[1, False](self.runs[0], n, output)
                elif variant == 2:
                    _copy_payloads[2, False](self.runs[0], n, output)
                else:
                    _copy_payloads[3, False](self.runs[0], n, output)
                _ = blocks^
                self.stage = _SORT_DONE
                return 0
            self.partition_size = min(n, DEFAULT_ROW_GROUP_SIZE)
            self.partitions = (
                n + self.partition_size - 1
            ) // self.partition_size
            self.ends = List[List[Int]](
                length=self.partitions - 1, fill=List[Int]()
            )
            self.stage = _SORT_BOUNDARY
            if self.partitions > 1:
                self.width = round_width(self.partitions - 1)
                return self.width
        if self.stage == _SORT_BOUNDARY:
            self.width = round_width(self.partitions)
            self.stage = _SORT_MERGE
            return self.width
        if self.stage == _SORT_MERGE:
            self.stage = _SORT_DONE
            self.blocks = List[KeyBlock]()
            self.runs = List[Int]()
            self.ends = List[List[Int]]()
        return 0

    def round_task(mut self, task: Int) raises:
        var variant = self._variant()
        if self.stage == _SORT_SINK:
            var address = Int(Pointer(to=self.layout[]))
            var lo = self.bounds[task]
            var hi = self.bounds[task + 1]
            if variant == 0:
                self.blocks[task] = _sink_task[1, True](address, lo, hi)
            elif variant == 1:
                self.blocks[task] = _sink_task[1, False](address, lo, hi)
            elif variant == 2:
                self.blocks[task] = _sink_task[2, False](address, lo, hi)
            else:
                self.blocks[task] = _sink_task[3, False](address, lo, hi)
        elif self.stage == _SORT_BOUNDARY:
            for p in range(task, self.partitions - 1, self.width):
                var target = (p + 1) * self.partition_size
                if variant == 0:
                    self.ends[p] = _boundary_task[1, True](
                        self.runs, self.counts, target
                    )
                elif variant == 1:
                    self.ends[p] = _boundary_task[1, False](
                        self.runs, self.counts, target
                    )
                elif variant == 2:
                    self.ends[p] = _boundary_task[2, False](
                        self.runs, self.counts, target
                    )
                else:
                    self.ends[p] = _boundary_task[3, False](
                        self.runs, self.counts, target
                    )
        elif self.stage == _SORT_MERGE:
            var output = Int(self.order.unsafe_ptr())
            var info = self.layout[].info
            for p in range(task, self.partitions, self.width):
                var begins = List[Int](length=len(self.runs), fill=0)
                if p > 0:
                    begins = self.ends[p - 1].copy()
                var ends = self.counts.copy()
                if p < self.partitions - 1:
                    ends = self.ends[p].copy()
                var start = p * self.partition_size
                if variant == 0:
                    _merge_task[1, True](
                        self.runs, begins, ends, info, output, start
                    )
                elif variant == 1:
                    _merge_task[1, False](
                        self.runs, begins, ends, info, output, start
                    )
                elif variant == 2:
                    _merge_task[2, False](
                        self.runs, begins, ends, info, output, start
                    )
                else:
                    _merge_task[3, False](
                        self.runs, begins, ends, info, output, start
                    )

    def take_order(mut self) -> List[Int]:
        """The rows in sorted order, moved out once no round is left."""
        var taken = List[Int]()
        swap(taken, self.order)
        return taken^


def _copy_payloads[W: Int, VAR: Bool](run: Int, n: Int, output: Int):
    """The row ids of a run's `n` records, in its order, to `output`."""
    var keys = SortKeys[W, VAR](run)
    var out = Pointer[Int, MutAnyOrigin](unsafe_from_address=output)
    for i in range(n):
        out[unsafe_offset=i] = keys.payload(i)


# ===----------------------------------------------------------------------=== #
# Top-N
# ===----------------------------------------------------------------------=== #


struct _TopNJob[W: Int, VAR: Bool](Job):
    """One thread's top-N Sink over rows [lo, hi), a vector at a time, and
    its Finalize; full heaps share their boundary."""

    var layout: Int
    var lo: Int
    var hi: Int
    var heap: TopNHeap
    var boundary: TopNBoundary

    def __init__(
        out self,
        layout: Int,
        lo: Int,
        hi: Int,
        limit: Int,
        boundary: TopNBoundary,
    ):
        self.layout = layout
        self.lo = lo
        self.hi = hi
        self.heap = TopNHeap(
            limit, SortKeys[Self.W, Self.VAR].R, variable=Self.VAR
        )
        self.boundary = boundary

    def run(mut self) raises:
        ref layout = Pointer[SortKeyLayout, MutAnyOrigin](
            unsafe_from_address=self.layout
        )[]
        var at = self.lo
        var prefixes = List[UInt64]()
        var row_key = has_row_key(layout)
        while at < self.hi:
            var end = min(self.hi, at + STANDARD_VECTOR_SIZE)
            var block: KeyBlock
            var entering = Optional[List[Int]]()
            if key_prefixes(layout, at, end, prefixes):
                entering = self.heap.entering[Self.W, Self.VAR](
                    prefixes, end - at
                )
            if entering and len(entering.value()) == 0:
                # No row of the vector can be in the result.
                at = end
                continue
            if entering and len(entering.value()) < end - at and not row_key:
                # Sort keys for the rows that pass alone.
                block = encode_selected(layout, at, entering.value())
            else:
                block = encode_keys(layout, at, end)
            self.heap.sink[Self.W, Self.VAR](block)
            _ = self.heap.publish[Self.W, Self.VAR](self.boundary)
            _ = block^
            at = end
            if self.heap.added >= self.heap.reduce_threshold():
                # Reduce: the payload is the row itself, so only the
                # keys' bytes are compacted.
                self.heap.compact_keys[Self.W, Self.VAR]()
        self.heap.finalize[Self.W, Self.VAR]()


def top_n(rows: Int, limit: Int) -> Bool:
    """Whether the first `limit` of `rows` rows are found with top-N heaps
    rather than by sorting every row (see `_TOPN_LARGE_LIMIT`)."""
    return limit >= 0 and not (
        limit > _TOPN_LARGE_LIMIT
        and Float64(limit) > Float64(rows) * _TOPN_INPUT_SHARE
    )


# The stages of a `TopNRounds`: the round that ran last.
comptime _TOPN_START = 0
comptime _TOPN_SINK = 1
comptime _TOPN_DONE = 2


struct TopNRounds(Movable, Rounds):
    """The first `limit` rows of the key columns in sorted order, as a
    round of tasks: each task keeps the best rows of one range of rows in
    its own heap, a vector at a time, the tasks sharing the boundary a row
    must beat; then the heaps are combined into one and sorted, here.
    `take_order` once no round is left."""

    var layout: ArcPointer[SortKeyLayout]
    var limit: Int
    var threads_cap: Int
    var stage: Int
    var boundary: TopNBoundary
    # One thread's sink per range of rows, for the layout's record shape
    # (only one list is used).
    var narrow: List[_TopNJob[1, False]]
    var variable: List[_TopNJob[1, True]]
    var two: List[_TopNJob[2, False]]
    var three: List[_TopNJob[3, False]]
    var order: List[Int]

    def __init__(
        out self,
        keys: List[Series],
        descending: List[Bool],
        nulls_last: List[Bool],
        maintain_order: Bool,
        limit: Int,
        threads: Int = 0,
    ) raises:
        """The first `limit` rows of `keys`' order; equal keys keep input
        order when `maintain_order`. `threads` caps the round's tasks (0:
        the configured count)."""
        self = Self(
            SortKeyLayout(keys, descending, nulls_last, maintain_order),
            limit,
            threads,
        )

    def __init__(
        out self, var layout: SortKeyLayout, limit: Int, threads: Int = 0
    ):
        self.limit = min(limit, layout.rows)
        self.layout = ArcPointer(layout^)
        self.threads_cap = threads
        self.stage = _TOPN_START
        self.boundary = TopNBoundary.none()
        self.narrow = List[_TopNJob[1, False]]()
        self.variable = List[_TopNJob[1, True]]()
        self.two = List[_TopNJob[2, False]]()
        self.three = List[_TopNJob[3, False]]()
        self.order = List[Int]()

    def next_round(mut self) raises -> Int:
        if self.stage == _TOPN_START:
            var n = self.layout[].rows
            var threads = _sink_threads(n, self.threads_cap)
            var bounds = partitions(n, threads, STANDARD_VECTOR_SIZE)
            if threads > 1:
                self.boundary = TopNBoundary.new()
            var address = Int(Pointer(to=self.layout[]))
            var tasks = 0
            for t in range(threads):
                if bounds[t] < bounds[t + 1]:
                    tasks += 1
                    if self.layout[].variable:
                        self.variable.append(
                            _TopNJob[1, True](
                                address,
                                bounds[t],
                                bounds[t + 1],
                                self.limit,
                                self.boundary,
                            )
                        )
                    elif self.layout[].words == 1:
                        self.narrow.append(
                            _TopNJob[1, False](
                                address,
                                bounds[t],
                                bounds[t + 1],
                                self.limit,
                                self.boundary,
                            )
                        )
                    elif self.layout[].words == 2:
                        self.two.append(
                            _TopNJob[2, False](
                                address,
                                bounds[t],
                                bounds[t + 1],
                                self.limit,
                                self.boundary,
                            )
                        )
                    else:
                        self.three.append(
                            _TopNJob[3, False](
                                address,
                                bounds[t],
                                bounds[t + 1],
                                self.limit,
                                self.boundary,
                            )
                        )
            self.stage = _TOPN_SINK
            if tasks > 0:
                return tasks
        if self.stage == _TOPN_SINK:
            self.boundary.free()
            self.boundary = TopNBoundary.none()
            if self.layout[].variable:
                self.order = _combined_heaps[1, True](self.variable, self.limit)
                self.variable = List[_TopNJob[1, True]]()
            elif self.layout[].words == 1:
                self.order = _combined_heaps[1, False](self.narrow, self.limit)
                self.narrow = List[_TopNJob[1, False]]()
            elif self.layout[].words == 2:
                self.order = _combined_heaps[2, False](self.two, self.limit)
                self.two = List[_TopNJob[2, False]]()
            else:
                self.order = _combined_heaps[3, False](self.three, self.limit)
                self.three = List[_TopNJob[3, False]]()
            self.stage = _TOPN_DONE
        return 0

    def round_task(mut self, task: Int) raises:
        if self.layout[].variable:
            self.variable[task].run()
        elif self.layout[].words == 1:
            self.narrow[task].run()
        elif self.layout[].words == 2:
            self.two[task].run()
        else:
            self.three[task].run()

    def take_order(mut self) -> List[Int]:
        """The first rows in sorted order, moved out once no round is
        left."""
        var taken = List[Int]()
        swap(taken, self.order)
        return taken^


def _combined_heaps[
    W: Int, VAR: Bool
](mut jobs: List[_TopNJob[W, VAR]], limit: Int) raises -> List[Int]:
    """The threads' heaps combined into one and sorted: its rows in
    order."""
    var order = List[Int]()
    if len(jobs) == 0:
        return order^
    var heap = TopNHeap(limit, SortKeys[W, VAR].R, variable=VAR)
    for j in range(len(jobs)):
        heap.combine[W, VAR](jobs[j].heap)
    heap.finalize[W, VAR]()
    order = heap.payloads[W, VAR]()
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
    if top_n(layout.rows, limit):
        var top = TopNRounds(layout^, limit, threads)
        run_rounds(top)
        return top.take_order()
    var rounds = SortRounds(layout^, threads)
    run_rounds(rounds)
    var order = rounds.take_order()
    if limit >= 0 and limit < len(order):
        order.shrink(limit)
    return order^
