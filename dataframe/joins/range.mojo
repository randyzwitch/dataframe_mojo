"""Bounded-integer and progression join paths and their existing budgets."""
from dataframe.column import (
    Column,
)
from dataframe.dtype import (
    DataType,
)
from dataframe.expr import (
    MAX,
    MIN,
)
from dataframe.frame import (
    DataFrame,
)
from dataframe.gather import (
    filter_range_int64_chunks,
)
from dataframe.huge_pages import (
    huge_capacity,
    huge_uninit,
)
from dataframe.join_fuse import (
    FusedJoin,
    MatchJob,
    fused_gather,
    probe_blocks,
)
from dataframe.join_hash import (
    RowsCopyJob,
    int64_progression,
)
from dataframe.parallel import (
    Job,
    Pool,
    partitions,
    run_jobs,
    worker_count,
)
from dataframe.series import (
    Series,
)
from dataframe.trace import (
    trace_path,
)
from std.collections import (
    Optional,
)
from std.memory import (
    ArcPointer,
    Pointer,
)
from std.sys import (
    num_physical_cores,
)


struct _JoinRangeProbeJob(Job):
    """Collect a stable right-row subsequence within proven left-key bounds."""

    var key: Column[Int64]
    var low: Int64
    var high: Int64
    var offset: Int
    var first: Int
    var last: Int
    var rows: List[Int]

    def __init__(
        out self,
        key: Column[Int64],
        low: Int64,
        high: Int64,
        offset: Int,
        first: Int,
        last: Int,
    ):
        self.key = key.copy()
        self.low = low
        self.high = high
        self.offset = offset
        self.first = first
        self.last = last
        self.rows = List[Int]()

    def run(mut self) raises:
        var all_valid = len(self.key._bits[]) == 0
        for i in range(self.first, self.last):
            if all_valid or self.key._valid(i):
                var value = self.key._get(i)
                if value >= self.low and value <= self.high:
                    self.rows.append(self.offset + i)


def _join_range_possible(right: Series, low: Int64, high: Int64) -> Int:
    """The existing 256-point range sample without per-point Series copies.

    Only physical Int64 keys enter this helper. Chunk ends locate each
    point directly; empty chunks and nonzero physical offsets retain the
    same row/validity interpretation as `_chunk_at`.
    """
    var samples = min(256, len(right))
    if samples == 0:
        return 0
    var step = len(right) // samples
    var possible = 0
    if not right.is_chunked():
        ref key = right._data[Column[Int64]]
        var values = key._ptr()
        var all_valid = len(key._bits[]) == 0
        for i in range(samples):
            var row = i * step
            if all_valid or key._valid(row):
                var value = values[unsafe_offset=row]
                possible += Int(value >= low and value <= high)
        return possible
    ref chunks = right._chunked.value()[]
    for i in range(samples):
        var row = i * step
        var lo = 0
        var hi = len(chunks.ends)
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            if row < chunks.ends[mid]:
                hi = mid
            else:
                lo = mid + 1
        var start = 0 if lo == 0 else chunks.ends[lo - 1]
        ref key = chunks.arrays[lo][Column[Int64]]
        var local = row - start
        if key._valid(local):
            var value = key._get(local)
            possible += Int(value >= low and value <= high)
    return possible


def _join_range_rows(
    left: Series,
    right: Series,
) raises -> Optional[List[Int]]:
    """Conservative physical Int64 bounds; None means no worthwhile filter.

    Bounds are exact. Sampling only chooses whether to attempt filtering;
    every retained row is checked and the ordinary join resolves equality.
    No subtraction is used, so signed extremes and temporal storage are safe.
    """
    if (
        not left._data.isa[Column[Int64]]()
        or not right._data.isa[Column[Int64]]()
    ):
        return None
    var low = Int64.MAX
    var high = Int64.MIN
    var found = False
    for part in left.chunks():
        ref key = part._data[Column[Int64]]
        for i in range(len(key)):
            if key._valid(i):
                var value = key._get(i)
                low = min(low, value)
                high = max(high, value)
                found = True
    if not found or len(right) == 0:
        return List[Int]()
    var samples = min(256, len(right))
    var possible = _join_range_possible(right, low, high)
    if possible > samples // 4:
        return None
    var jobs = List[_JoinRangeProbeJob]()
    var offset = 0
    for part in right.chunks():
        var workers = worker_count(len(part))
        var bounds = partitions(len(part), workers, 1)
        for worker in range(workers):
            jobs.append(
                _JoinRangeProbeJob(
                    part._data[Column[Int64]],
                    low,
                    high,
                    offset,
                    bounds[worker],
                    bounds[worker + 1],
                )
            )
        offset += len(part)
    var pool = Pool(min(worker_count(len(right)), len(jobs)))
    pool.run(jobs)
    pool.release()
    var count = 0
    for i in range(len(jobs)):
        count += len(jobs[i].rows)
    if count > len(right) // 4:
        return None
    var rows = List[Int](capacity=count)
    for i in range(len(jobs)):
        for row in jobs[i].rows:
            rows.append(row)
    trace_path("join.range_filter")
    return rows^


struct _ProgressionProbeJob(Job):
    """Probe one left-row range against an ascending constant-step build."""

    var left: Column[Int64]
    var base: Int64
    var stride: UInt64
    var right_count: Int
    var start: Int
    var end: Int
    var include_unmatched: Bool
    var left_rows: List[Int]
    var right_rows: List[Int]

    def __init__(
        out self,
        left: Column[Int64],
        base: Int64,
        stride: UInt64,
        right_count: Int,
        start: Int,
        end: Int,
        include_unmatched: Bool,
    ):
        self.left = left.copy()
        self.base = base
        self.stride = stride
        self.right_count = right_count
        self.start = start
        self.end = end
        self.include_unmatched = include_unmatched
        self.left_rows = List[Int](capacity=end - start)
        self.right_rows = List[Int](capacity=end - start)

    def run(mut self) raises:
        var all_valid = len(self.left._bits[]) == 0
        var right_limit = UInt64(self.right_count)
        for i in range(self.start, self.end):
            var row = -1
            if all_valid or self.left._valid(i):
                var value = self.left._get(i)
                if value >= self.base:
                    var distance = _int64_distance(self.base, value)
                    if distance % self.stride == 0:
                        var slot = distance // self.stride
                        if slot < right_limit:
                            row = Int(slot)
            if row >= 0 or self.include_unmatched:
                self.left_rows.append(i)
                self.right_rows.append(row)


def _parallel_progression_rows(
    left: Series,
    base: Int64,
    stride: UInt64,
    right_count: Int,
    include_unmatched: Bool,
) raises -> Tuple[List[Int], List[Int]]:
    var left_values = left.int64()
    var workers = worker_count(len(left_values))
    var bounds = partitions(len(left_values), workers, 1)
    var jobs = List[_ProgressionProbeJob](capacity=workers)
    for worker in range(workers):
        jobs.append(
            _ProgressionProbeJob(
                left_values,
                base,
                stride,
                right_count,
                bounds[worker],
                bounds[worker + 1],
                include_unmatched,
            )
        )
    run_jobs(jobs)
    var total = 0
    for worker in range(workers):
        total += len(jobs[worker].left_rows)
    var left_rows = List[Int](capacity=total)
    var right_rows = List[Int](capacity=total)
    for worker in range(workers):
        for i in range(len(jobs[worker].left_rows)):
            left_rows.append(jobs[worker].left_rows[i])
            right_rows.append(jobs[worker].right_rows[i])
    return (left_rows^, right_rows^)


def _dense_right_int64_rows(
    left: Series, right: Series, include_unmatched: Bool = False
) raises -> Tuple[Bool, List[Int], List[Int]]:
    """Direct matches when the build keys are an ascending progression.

    Logical types must agree, including temporal units. Sorted sequential
    IDs and calendar grids (one row per constant step) qualify; see
    int64_progression. Any repeated, null, descending or irregular build
    key declines, and the caller falls back to an index.
    """
    if (
        left.dtype() != right.dtype()
        or right.dtype().physical() != DataType.INT64
    ):
        return (False, List[Int](), List[Int]())
    var progression = int64_progression(right)
    if not progression[0]:
        return (False, List[Int](), List[Int]())
    var base = progression[1]
    var stride = progression[2]
    # Rechunking and row-list merging pay off once the probe working set is large.
    if (
        stride > 1
        and worker_count(len(left)) > 1
        and not left.is_chunked()
        and len(left) >= _PROGRESSION_PARALLEL_KEY_BYTES // 8
    ):
        var pairs = _parallel_progression_rows(
            left, base, stride, len(right), include_unmatched
        )
        trace_path("join.progression")
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        swap(left_rows, pairs[0])
        swap(right_rows, pairs[1])
        return (True, left_rows^, right_rows^)
    var row = 0
    if stride == 1:
        var right_limit = UInt64(len(right))
        var left_rows = List[Int](capacity=len(left))
        var right_rows = List[Int](capacity=len(left))
        for part in left.chunks():
            ref column = part._data[Column[Int64]]
            var values = column.unsafe_values()
            if len(column._bits[]) == 0:
                for i in range(len(column)):
                    var matched_row = -1
                    var value = values.unsafe_load(i)
                    if value >= base:
                        var index = UInt64(value) - UInt64(base)
                        if index < right_limit:
                            matched_row = Int(index)
                    if matched_row >= 0 or include_unmatched:
                        left_rows.append(row)
                        right_rows.append(matched_row)
                    row += 1
            else:
                for i in range(len(column)):
                    var matched_row = -1
                    if column._valid(i):
                        var value = values.unsafe_load(i)
                        if value >= base:
                            var index = UInt64(value) - UInt64(base)
                            if index < right_limit:
                                matched_row = Int(index)
                    if matched_row >= 0 or include_unmatched:
                        left_rows.append(row)
                        right_rows.append(matched_row)
                    row += 1
        trace_path("join.progression")
        return (True, left_rows^, right_rows^)
    var right_limit = UInt64(len(right))
    var left_rows = List[Int](capacity=len(left))
    var right_rows = List[Int](capacity=len(left))
    for part in left.chunks():
        ref column = part._data[Column[Int64]]
        var values = column.unsafe_values()
        if len(column._bits[]) == 0:
            for i in range(len(column)):
                var matched_row = -1
                var value = values.unsafe_load(i)
                if value >= base:
                    var distance = _int64_distance(base, value)
                    if distance % stride == 0:
                        var index = distance // stride
                        if index < right_limit:
                            matched_row = Int(index)
                if matched_row >= 0 or include_unmatched:
                    left_rows.append(row)
                    right_rows.append(matched_row)
                row += 1
        else:
            for i in range(len(column)):
                var matched_row = -1
                if column._valid(i):
                    var value = values.unsafe_load(i)
                    if value >= base:
                        var distance = _int64_distance(base, value)
                        if distance % stride == 0:
                            var index = distance // stride
                            if index < right_limit:
                                matched_row = Int(index)
                if matched_row >= 0 or include_unmatched:
                    left_rows.append(row)
                    right_rows.append(matched_row)
                row += 1
    trace_path("join.progression")
    return (True, left_rows^, right_rows^)


# Rechunk + row-list merge lose at 1.5-2M probes and win at 3-6M on 32
# workers. The 8-worker run also loses at 2M and wins at 4M: this follows
# probe-buffer size, not rows per worker. Use the conservative 24 MiB edge.
# Chunked probes stay serial: rechunking loses at 2M, 4M, 6M, 8M and 10M.
comptime _PROGRESSION_PARALLEL_KEY_BYTES = 24 * 1024 * 1024

# Allocation policy shared by all direct-address paths: the domain table
# costs at most four times the Int64 input-key bytes and at most 128 MiB.
# Row-chain heads cost 8 B/slot; CSR starts plus cursors cost 16 B/slot;
# membership flags cost 1 B/slot. Row-proportional buffers are separate.
# The absolute budget admits the winning 80 MB dense row-chain table at
# 10M keys but rejects the 160 MB table that is only level with hashing.
comptime _RANGE_TABLE_MAX_BYTES = 128 * 1024 * 1024
comptime _RANGE_TABLE_BYTES_PER_KEY = 4 * 8
# A 25%-dense head table wins at 100k keys (3.2 MB), but loses at 500k
# (16 MB). Beyond this cache allowance require at least 50% row density.
comptime _RANGE_SMALL_TABLE_BYTES = 8 * 1024 * 1024
# Membership's serial random stores hit a separate cache cliff: at 2M keys
# 10/12 MB take 32/38 ms vs hash's 39 ms; 14/16 MB take 46/55 ms.
comptime _RANGE_MEMBERSHIP_MAX_BYTES = 12_000_000
# Building a large random head table in slot partitions starts paying off
# at 750k keys / 12 MB heads and 1M / 8 MB, not 500k / 8 MB. Account for
# keys, next-row links and heads; tiny head tables still stay serial.
comptime _RANGE_PARALLEL_WORKING_BYTES = 20 * 1024 * 1024
comptime _RANGE_PARALLEL_MIN_HEAD_BYTES = 8_000_000


def _range_join_table_capacity(
    rows: Int,
    slot_bytes: Int,
    max_bytes: Int = _RANGE_TABLE_MAX_BYTES,
    bytes_per_key: Int = _RANGE_TABLE_BYTES_PER_KEY,
) -> Int:
    """Domain slots fitting both an absolute and input-relative byte budget.

    Bound before multiplying, so even theoretical Int.MAX row counts do
    not overflow. Callers account for the actual per-domain-slot storage.
    """
    if rows <= 0 or slot_bytes <= 0 or max_bytes <= 0:
        return 0
    var slots = max_bytes // slot_bytes
    var per_row = bytes_per_key // slot_bytes
    if per_row <= 0:
        return 0
    return slots if rows > slots // per_row else rows * per_row


def _parallel_range_build_workers(rows: Int, slots: Int) -> Int:
    """Split a random index build only when its working set warrants it."""
    var entries = _RANGE_PARALLEL_WORKING_BYTES // 8
    if (
        slots < _RANGE_PARALLEL_MIN_HEAD_BYTES // 8
        or slots < entries - 2 * min(rows, entries // 2)
    ):
        return 1
    # 4/8/16-worker calibration on 3970X and M1: retain the 16-worker
    # ceiling, but never oversubscribe physical cores for this memory-bound
    # build. M1 1M keys: 8 builders 12.5 ms vs cap 16 (15 effective) 15.1 ms.
    return min(16, min(max(1, num_physical_cores()), worker_count(rows)))


def _range_join_span_fits(low: Int64, high: Int64, cap: Int) -> Bool:
    """Whether `[low, high]` has at most `cap` Int64 values, safely."""
    if cap <= 0:
        return False
    if low >= 0 or high < 0:
        # Same-sign subtraction cannot overflow here. The largest negative
        # span is Int64.MAX (`-1 - Int64.MIN`).
        return high - low < Int64(cap)
    # A span crossing zero needs unsigned addition. `-Int64.MIN` itself is
    # not representable, and its range cannot meet our small cap anyway.
    if low == Int64.MIN:
        return False
    return UInt64(high) + UInt64(-low) < UInt64(cap)


def _int64_distance(low: Int64, high: Int64) -> UInt64:
    """Nonnegative distance between ordered Int64 values, without overflow."""
    if low >= 0 or high < 0:
        return UInt64(high - low)
    return UInt64(-(low + 1)) + UInt64(high) + 1


struct _RangeMembershipJob(Job):
    """Select one left range using either an interval or presence table."""

    var left: Column[Int64]
    var present: ArcPointer[List[UInt8]]
    var low: Int64
    var high: Int64
    var consecutive: Bool
    var want_match: Bool
    var start: Int
    var end: Int
    var rows: List[Int]

    def __init__(
        out self,
        left: Column[Int64],
        present: ArcPointer[List[UInt8]],
        low: Int64,
        high: Int64,
        consecutive: Bool,
        want_match: Bool,
        start: Int,
        end: Int,
    ):
        self.left = left.copy()
        self.present = present.copy()
        self.low = low
        self.high = high
        self.consecutive = consecutive
        self.want_match = want_match
        self.start = start
        self.end = end
        self.rows = List[Int]()

    def run(mut self) raises:
        var all_valid = len(self.left._bits[]) == 0
        for row in range(self.start, self.end):
            var matched = False
            if all_valid or self.left._valid(row):
                var value = self.left._get(row)
                if value >= self.low and value <= self.high:
                    matched = self.consecutive or (
                        self.present[][Int(value - self.low)] != 0
                    )
            if matched == self.want_match:
                self.rows.append(row)


struct _RangeJoinProbeJob(MatchJob):
    """Match rows [start, end) of one left chunk, which begins at row
    `base`, against a bounded right Int64 index."""

    var left: Column[Int64]
    var base: Int
    var heads: ArcPointer[List[Int]]
    var next_rows: ArcPointer[List[Int]]
    var low: Int64
    var high: Int64
    var unique_keys: Bool
    var start: Int
    var end: Int
    var include_unmatched: Bool
    var left_rows: List[Int]
    var right_rows: List[Int]

    def __init__(
        out self,
        left: Column[Int64],
        heads: ArcPointer[List[Int]],
        next_rows: ArcPointer[List[Int]],
        low: Int64,
        high: Int64,
        unique_keys: Bool,
        start: Int,
        end: Int,
        include_unmatched: Bool,
        base: Int = 0,
    ):
        self.left = left.copy()
        self.base = base
        self.heads = heads.copy()
        self.next_rows = next_rows.copy()
        self.low = low
        self.high = high
        self.unique_keys = unique_keys
        self.start = start
        self.end = end
        self.include_unmatched = include_unmatched
        self.left_rows = huge_capacity[Int](end - start)
        self.right_rows = huge_capacity[Int](end - start)

    def take_matches(
        mut self, mut left: List[Int], mut right: List[Int]
    ) -> Bool:
        swap(left, self.left_rows)
        swap(right, self.right_rows)
        return False

    def row_range(self) -> Tuple[Int, Int]:
        return (self.base + self.start, self.base + self.end)

    def run(mut self) raises:
        var all_valid = len(self.left._bits[]) == 0
        var values = self.left._ptr()
        var heads = self.heads[].unsafe_ptr()
        var next_rows = self.next_rows[].unsafe_ptr()
        var base = self.base
        for i in range(self.start, self.end):
            var j = -1
            if all_valid or self.left._valid(i):
                var value = values.unsafe_offset(i)[]
                if value >= self.low and value <= self.high:
                    j = heads.unsafe_offset(Int(value - self.low))[]
            if j >= 0:
                if self.unique_keys:
                    self.left_rows.append(base + i)
                    self.right_rows.append(j)
                else:
                    while j >= 0:
                        self.left_rows.append(base + i)
                        self.right_rows.append(j)
                        j = next_rows.unsafe_offset(j)[]
            elif self.include_unmatched:
                self.left_rows.append(base + i)
                self.right_rows.append(-1)


struct _RangeIndexBuildJob(Job):
    """Own a disjoint slot range while scanning build rows in reverse order."""

    var values: Column[Int64]
    var heads: Int
    var next_rows: Int
    var low: Int64
    var first_slot: Int
    var last_slot: Int
    var unique_keys: Bool

    def __init__(
        out self,
        values: Column[Int64],
        heads: Int,
        next_rows: Int,
        low: Int64,
        first_slot: Int,
        last_slot: Int,
    ):
        self.values = values.copy()
        self.heads = heads
        self.next_rows = next_rows
        self.low = low
        self.first_slot = first_slot
        self.last_slot = last_slot
        self.unique_keys = True

    def run(mut self) raises:
        ref heads = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.heads
        )[]
        ref next_rows = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.next_rows
        )[]
        for j in range(len(self.values) - 1, -1, -1):
            if self.values._valid(j):
                var value = self.values._get(j)
                if value < self.low:
                    continue
                var slot = Int(value - self.low)
                if slot >= self.first_slot and slot < self.last_slot:
                    if heads[slot] >= 0:
                        self.unique_keys = False
                    next_rows[j] = heads[slot]
                    heads[slot] = j


def _bounded_int64_join_rows(
    left: Series,
    right: Series,
    include_unmatched: Bool,
    mut left_rows: List[Int],
    mut right_rows: List[Int],
    mut request: FusedJoin,
) raises -> Bool:
    """Direct-address right-row chains when its Int64 domain is compact.
    On success the matched row pairs are written to `left_rows` and
    `right_rows` (returned in place: a tuple's lists had to be copied out,
    about 9M rows each on H2O's joins)."""
    if (
        left.dtype().physical() != DataType.INT64
        or right.dtype().physical() != DataType.INT64
        or len(right) == 0
    ):
        return False
    var cap = _range_join_table_capacity(len(right), 8)
    # Beyond the cache-sized allowance, a sparse head table only ties or
    # loses to hashing. Both limits are computed without row-count overflow.
    cap = min(
        cap,
        max(
            _RANGE_SMALL_TABLE_BYTES // 8,
            _range_join_table_capacity(len(right), 8, bytes_per_key=16),
        ),
    )
    # A sample can prove a domain is too wide without rechunking or
    # scanning the full right key. The exact scan below still decides hits.
    if right.dtype() == DataType.INT64:
        var sample_found = False
        var sample_low = Int64(0)
        var sample_high = Int64(0)
        var sample_step = max(1, len(right) // 256)
        var row = 0
        while row < len(right):
            var cell = right.get(row)
            if not cell.is_null():
                var value = cell.int64()
                if not sample_found:
                    sample_low = value
                    sample_high = value
                    sample_found = True
                else:
                    sample_low = min(sample_low, value)
                    sample_high = max(sample_high, value)
                if not _range_join_span_fits(sample_low, sample_high, cap):
                    return False
            row += sample_step
    var right_values = right.int64()
    var found = False
    var low = Int64(0)
    var high = Int64(0)
    for j in range(len(right_values)):
        if not right_values._valid(j):
            continue
        var value = right_values._get(j)
        if not found:
            low = value
            high = value
            found = True
        else:
            low = min(low, value)
            high = max(high, value)
    if not found or not _range_join_span_fits(low, high, cap):
        return False
    var heads = List[Int](length=Int(high - low) + 1, fill=-1)
    var next_rows = List[Int](length=len(right_values), fill=-1)
    var unique_keys = True
    var build_workers = _parallel_range_build_workers(
        len(right_values), len(heads)
    )
    if build_workers > 1:
        var slot_bounds = partitions(len(heads), build_workers, 1)
        var build_jobs = List[_RangeIndexBuildJob](capacity=build_workers)
        for worker in range(build_workers):
            build_jobs.append(
                _RangeIndexBuildJob(
                    right_values,
                    Int(Pointer(to=heads)),
                    Int(Pointer(to=next_rows)),
                    low,
                    slot_bounds[worker],
                    slot_bounds[worker + 1],
                )
            )
        run_jobs(build_jobs)
        for worker in range(build_workers):
            unique_keys = unique_keys and build_jobs[worker].unique_keys
    else:
        for j in range(len(right_values) - 1, -1, -1):
            if right_values._valid(j):
                var slot = Int(right_values._get(j) - low)
                if heads[slot] >= 0:
                    unique_keys = False
                next_rows[j] = heads[slot]
                heads[slot] = j
    # Probe each chunk of the left key in place (a Parquet column is
    # chunked; copying it into one buffer first was a serial pass), in
    # pieces of about equal rows that never span a chunk.
    var height = len(left)
    var workers = worker_count(height)
    var step = max(1, (height + workers - 1) // workers)
    if request.enabled:
        step = min(
            step,
            (height + probe_blocks(height, request) - 1)
            // probe_blocks(height, request),
        )
    var shared_heads = ArcPointer(heads^)
    var shared_next = ArcPointer(next_rows^)
    var jobs = List[_RangeJoinProbeJob]()
    var base = 0
    for part in left.chunks():
        ref chunk = part._data[Column[Int64]]
        var at = 0
        while at < len(chunk):
            var end = min(len(chunk), at + step)
            jobs.append(
                _RangeJoinProbeJob(
                    chunk,
                    shared_heads,
                    shared_next,
                    low,
                    high,
                    unique_keys,
                    at,
                    end,
                    include_unmatched,
                    base,
                )
            )
            at = end
        base += len(chunk)
    if request.enabled:
        # Each block gathers its own columns: no row list of the output.
        fused_gather(jobs, request, include_unmatched)
        trace_path("join.bounded_index")
        return True
    run_jobs(jobs)
    # Concatenate on every worker: each job's matches are copied to their
    # offset in the output. One thread appending them row by row was a
    # third of a join's time on 10M rows.
    var offsets = List[Int](capacity=len(jobs) + 1)
    offsets.append(0)
    for job in range(len(jobs)):
        offsets.append(offsets[job] + len(jobs[job].left_rows))
    var total = offsets[len(jobs)]
    left_rows = huge_uninit[Int](total)
    right_rows = huge_uninit[Int](total)
    var copies = List[RowsCopyJob](capacity=len(jobs))
    for job in range(len(jobs)):
        copies.append(
            RowsCopyJob(
                Int(Pointer(to=jobs[job].left_rows)),
                Int(Pointer(to=jobs[job].right_rows)),
                0,
                len(jobs[job].left_rows),
                Int(Pointer(to=left_rows)),
                Int(Pointer(to=right_rows)),
                offsets[job],
            )
        )
    run_jobs(copies)
    _ = jobs^
    trace_path("join.bounded_index")
    return True


@fieldwise_init
struct _RangeMembershipDomain(Copyable):
    var supported: Bool
    var found: Bool
    var low: Int64
    var high: Int64
    var consecutive: Bool
    var present: List[UInt8]


def _range_int64_membership_domain(
    right: Series,
) raises -> _RangeMembershipDomain:
    """Describe a bounded right-key set, or report a wide domain."""
    if right.dtype().physical() != DataType.INT64:
        return _RangeMembershipDomain(False, False, 0, 0, False, List[UInt8]())
    var right_values = right.int64()
    var found = False
    var low = Int64(0)
    var high = Int64(0)
    var previous = Int64(0)
    var consecutive = True
    for row in range(len(right_values)):
        if not right_values._valid(row):
            consecutive = False
            continue
        var value = right_values._get(row)
        if not found:
            low = value
            high = value
            found = True
        else:
            if previous == Int64.MAX or value != previous + 1:
                consecutive = False
            low = min(low, value)
            high = max(high, value)
        previous = value
    var cap = _range_join_table_capacity(
        len(right_values), 1, _RANGE_MEMBERSHIP_MAX_BYTES
    )
    if found and not consecutive and not _range_join_span_fits(low, high, cap):
        return _RangeMembershipDomain(
            False, found, low, high, consecutive, List[UInt8]()
        )
    var present = List[UInt8]()
    if found and not consecutive:
        present = List[UInt8](length=Int(high - low) + 1, fill=0)
        for row in range(len(right_values)):
            if right_values._valid(row):
                present[Int(right_values._get(row) - low)] = 1
    return _RangeMembershipDomain(True, found, low, high, consecutive, present^)


def _range_int64_membership_chunks(
    left: DataFrame, key_column: Int, right: Series, want_match: Bool
) raises -> Tuple[Bool, List[Series]]:
    """Filter aligned left chunks from a bounded right Int64 domain."""
    if left._columns[key_column].dtype().physical() != DataType.INT64:
        return (False, List[Series]())
    var domain = _range_int64_membership_domain(right)
    if not domain.supported:
        return (False, List[Series]())
    trace_path("join.bounded_membership")
    return (
        True,
        filter_range_int64_chunks(
            left._columns,
            key_column,
            domain.present.copy(),
            domain.low,
            domain.high,
            domain.consecutive,
            domain.found,
            want_match,
        ),
    )


def _range_int64_membership_rows(
    left: Series, right: Series, want_match: Bool
) raises -> Tuple[Bool, List[Int]]:
    """Direct-address membership for a bounded Int64 key range.

    Consecutive sorted right keys need only two bounds. Other bounded
    domains use one presence flag per key; duplicates are idempotent.
    """
    if left.dtype().physical() != DataType.INT64:
        return (False, List[Int]())
    var domain = _range_int64_membership_domain(right)
    if not domain.supported:
        return (False, List[Int]())
    var left_values = left.int64()
    var rows = List[Int](capacity=len(left_values))
    if not domain.found:
        if not want_match:
            for row in range(len(left_values)):
                rows.append(row)
        trace_path("join.bounded_membership")
        return (True, rows^)
    var low = domain.low
    var high = domain.high
    var consecutive = domain.consecutive
    var present = domain.present.copy()
    var workers = worker_count(len(left_values))
    if workers == 1:
        var all_valid = len(left_values._bits[]) == 0
        for row in range(len(left_values)):
            var matched = False
            if all_valid or left_values._valid(row):
                var value = left_values._get(row)
                if value >= low and value <= high:
                    matched = consecutive or (present[Int(value - low)] != 0)
            if matched == want_match:
                rows.append(row)
        trace_path("join.bounded_membership")
        return (True, rows^)
    var shared = ArcPointer(present^)
    var bounds = partitions(len(left_values), workers, 1)
    var jobs = List[_RangeMembershipJob](capacity=workers)
    for worker in range(workers):
        jobs.append(
            _RangeMembershipJob(
                left_values,
                shared,
                low,
                high,
                consecutive,
                want_match,
                bounds[worker],
                bounds[worker + 1],
            )
        )
    run_jobs(jobs)
    for worker in range(workers):
        for row in jobs[worker].rows:
            rows.append(row)
    trace_path("join.bounded_membership")
    return (True, rows^)


def _bounded_int64_join_ids(
    left: DataFrame, right: DataFrame, left_column: Int, right_column: Int
) raises -> Tuple[Bool, List[Int], List[Int], Int]:
    """Dense direct ids for one small Int64 value range, or `False`.

    This is deliberately a range guard, not a general direct-address table:
    `count` feeds the join CSR's `starts` allocation, so a sparse or wide
    range must use the normal dictionary encoder even when few values occur.
    """
    if (
        left._columns[left_column].dtype().physical() != DataType.INT64
        or right._columns[right_column].dtype().physical() != DataType.INT64
    ):
        return (False, List[Int](), List[Int](), 0)
    # The relative cap below is only an allocation guard. Avoid overflowing
    # its row-count input on theoretical maximal frames; those must use the
    # dictionary path.
    if left.height() > Int.MAX - right.height():
        return (False, List[Int](), List[Int](), 0)
    var total = left.height() + right.height()
    # Starts and its cursor copy each have count + 1 entries.
    var cap = _range_join_table_capacity(total, 16, _RANGE_TABLE_MAX_BYTES - 16)
    var found = False
    var low = Int64(0)
    var high = Int64(0)
    var left_values = left._columns[left_column].int64()
    var right_values = right._columns[right_column].int64()
    for row in range(len(left_values)):
        if left_values._valid(row):
            var value = left_values._get(row)
            if not found:
                low = value
                high = value
                found = True
            else:
                low = min(low, value)
                high = max(high, value)
    for row in range(len(right_values)):
        if right_values._valid(row):
            var value = right_values._get(row)
            if not found:
                low = value
                high = value
                found = True
            else:
                low = min(low, value)
                high = max(high, value)
    if not found:
        trace_path("join.dense_ids")
        return (
            True,
            List[Int](length=len(left_values), fill=-1),
            List[Int](length=len(right_values), fill=-1),
            0,
        )
    if not _range_join_span_fits(low, high, cap):
        return (False, List[Int](), List[Int](), 0)
    var count = Int(high - low) + 1
    var left_ids = List[Int](capacity=len(left_values))
    var right_ids = List[Int](capacity=len(right_values))
    for row in range(len(left_values)):
        left_ids.append(
            Int(left_values._get(row) - low) if left_values._valid(row) else -1
        )
    for row in range(len(right_values)):
        right_ids.append(
            Int(right_values._get(row) - low) if right_values._valid(
                row
            ) else -1
        )
    trace_path("join.dense_ids")
    return (True, left_ids^, right_ids^, count)
