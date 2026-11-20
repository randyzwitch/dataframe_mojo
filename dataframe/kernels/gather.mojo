"""Parallel row selection: mask-to-index compaction and row gathers (#7).

Both steps split work into contiguous ranges and preserve row order.

- `true_rows` scans the mask in row partitions; each worker collects the
  indices of valid true rows, and the per-partition lists are concatenated
  in order.
- `take_parallel` gathers rows for every column. Fixed-width columns (every
  numeric type, Bool, temporal) are preallocated and each worker fills a
  disjoint range of *output* positions whose bounds are multiples of 8, so
  no two workers write the same validity byte. String columns gather per
  range and append the pieces in order (bulk byte copies).

Inputs are shared read-only (reference-counted buffers); outputs are only
published after every worker succeeds, so a failure never exposes a
partially built column.
"""
from dataframe.storage.buffer import _Buffer, _BufferData
from std.bit import count_trailing_zeros
from std.memory import ArcPointer, Pointer
from dataframe.storage.boolean import BoolColumn, _bits64
from dataframe.storage.column import Column
from dataframe.storage.allocation import huge_uninit

# Fixed-width payloads a gather copies: the numeric types plus Int128, the
# storage of decimal columns, which joins used to return as nulls.
comptime GATHER_DTYPES: Array[DType, 11] = [
    DType.int64,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float32,
    DType.int128,
]
from dataframe.expr import GT, LT, GE, LE, EQ
from dataframe.parallel import (
    _performance_core_count,
    Job,
    Pool,
    Rounds,
    configured_workers,
    partitions,
    round_width,
    run_jobs,
    run_rounds,
    task_count,
    worker_count,
)
from dataframe.series import ChunkMerge, Series
from dataframe.storage.string import StringColumn
from dataframe.trace import trace_path


struct _MaskJob(Job):
    var mask: BoolColumn
    var start: Int
    var end: Int
    var rows: List[Int]

    def __init__(out self, mask: BoolColumn, start: Int, end: Int):
        self.mask = mask.copy()
        self.start = start
        self.end = end
        self.rows = List[Int]()

    def run(mut self) raises:
        # Values and validity are both LSB-first bitmaps: read 64 rows at a
        # time and walk the set bits.
        ref values = self.mask._data[]
        ref bits = self.mask._bits[]
        var base = self.mask._offset
        var end = base + self.end
        var i = self.start
        while i < self.end:
            var word = _bits64(values, base + i, end) & _bits64(
                bits, base + i, end
            )
            while word != 0:
                self.rows.append(i + Int(count_trailing_zeros(word)))
                word &= word - 1
            i += 64

    def into_rows(deinit self) -> List[Int]:
        return self.rows^


def true_rows(mask: BoolColumn) raises -> List[Int]:
    """Indices of valid true entries, in order."""
    var n = len(mask)
    var workers = worker_count(n)
    var jobs = List[_MaskJob](capacity=workers)
    var bounds = partitions(n, workers, 64)
    for w in range(workers):
        jobs.append(_MaskJob(mask, bounds[w], bounds[w + 1]))
    if workers == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    var count = 0
    for i in range(len(jobs)):
        count += len(jobs[i].rows)
    var rows = List[Int](capacity=count)
    while len(jobs) > 0:
        rows.extend(Span(jobs.pop(0).into_rows()))
    return rows^


def _float_compare(op: Int, value: Float64, literal: Float64) -> Bool:
    if op == GT:
        return value > literal
    if op == LT:
        return value < literal
    if op == GE:
        return value >= literal
    if op == LE:
        return value <= literal
    if op == EQ:
        return value == literal
    return value != literal


def _scan_float_compare(
    mut rows: List[Int],
    column: Column[Float64],
    offset: Int,
    lo: Int,
    hi: Int,
    op: Int,
    literal: Float64,
):
    var values = column.unsafe_values()
    for row in range(lo, hi):
        var local = row - offset
        if column._valid(local) and _float_compare(
            op, values.unsafe_load(local), literal
        ):
            rows.append(row)


struct _FloatCompareRowsJob(Job):
    """Select matching Float64 rows from one contiguous row partition."""

    var source: Series
    var start: Int
    var end: Int
    var op: Int
    var literal: Float64
    var rows: List[Int]

    def __init__(
        out self,
        source: Series,
        start: Int,
        end: Int,
        op: Int,
        literal: Float64,
    ):
        self.source = source.copy()
        self.start = start
        self.end = end
        self.op = op
        self.literal = literal
        self.rows = List[Int]()

    def run(mut self) raises:
        var source = self.source.copy()
        var rows = List[Int]()
        var op = self.op
        var literal = self.literal
        if not source.is_chunked():
            _scan_float_compare(
                rows,
                source._data[Column[Float64]],
                0,
                self.start,
                self.end,
                op,
                literal,
            )
            self.rows = rows^
            return
        ref chunks = source._chunked.value()[]
        var first = 0
        var upper = len(chunks.ends)
        while first < upper:
            var mid = (first + upper) // 2
            if chunks.ends[mid] <= self.start:
                first = mid + 1
            else:
                upper = mid
        for i in range(first, len(chunks.ends)):
            var offset = 0 if i == 0 else chunks.ends[i - 1]
            if offset >= self.end:
                break
            _scan_float_compare(
                rows,
                chunks.arrays[i][Column[Float64]],
                offset,
                max(self.start, offset),
                min(self.end, chunks.ends[i]),
                op,
                literal,
            )
        self.rows = rows^

    def into_rows(deinit self) -> List[Int]:
        return self.rows^


def float_compare_rows(
    source: Series, op: Int, literal: Float64
) raises -> List[Int]:
    """Select valid Float64 rows without materializing a Boolean column."""
    trace_path("filter.float_rows")
    var n = len(source)
    var workers = worker_count(n)
    var bounds = partitions(n, workers, 1)
    var jobs = List[_FloatCompareRowsJob](capacity=workers)
    for w in range(workers):
        jobs.append(
            _FloatCompareRowsJob(source, bounds[w], bounds[w + 1], op, literal)
        )
    if workers == 1:
        jobs[0].run()
    else:
        run_jobs(jobs)
    var count = 0
    for i in range(len(jobs)):
        count += len(jobs[i].rows)
    var rows = List[Int](capacity=count)
    while len(jobs) > 0:
        rows.extend(Span(jobs.pop(0).into_rows()))
    return rows^


def can_filter_aligned_chunks(columns: List[Series]) -> Bool:
    """Whether all columns share physical row boundaries for local filtering."""
    if len(columns) == 0 or not columns[0].is_chunked():
        return False
    ref first = columns[0]._chunked.value()[].ends
    if len(first) < 2:
        return False
    for c in range(1, len(columns)):
        if not columns[c].is_chunked():
            return False
        ref ends = columns[c]._chunked.value()[].ends
        if len(ends) != len(first):
            return False
        for i in range(len(first)):
            if ends[i] != first[i]:
                return False
    return True


struct _RangeMembershipChunkFilterJob(Job):
    """Select one aligned chunk using Int64 membership and gather locally."""

    var columns: ArcPointer[List[Series]]
    var predicate: Int
    var chunk: Int
    var present: ArcPointer[List[UInt8]]
    var low: Int64
    var high: Int64
    var consecutive: Bool
    var found: Bool
    var want_match: Bool
    var selected: List[Series]

    def __init__(
        out self,
        columns: ArcPointer[List[Series]],
        predicate: Int,
        chunk: Int,
        present: ArcPointer[List[UInt8]],
        low: Int64,
        high: Int64,
        consecutive: Bool,
        found: Bool,
        want_match: Bool,
    ):
        self.columns = columns.copy()
        self.predicate = predicate
        self.chunk = chunk
        self.present = present.copy()
        self.low = low
        self.high = high
        self.consecutive = consecutive
        self.found = found
        self.want_match = want_match
        self.selected = List[Series]()

    def run(mut self) raises:
        ref columns = self.columns[]
        ref key = (
            columns[self.predicate]
            ._chunked.value()[]
            .arrays[self.chunk][Column[Int64]]
        )
        var rows = List[Int]()
        var all_valid = len(key._bits[]) == 0
        for i in range(len(key)):
            var matched = False
            if self.found and (all_valid or key._valid(i)):
                var value = key._get(i)
                if value >= self.low and value <= self.high:
                    matched = self.consecutive or (
                        self.present[][Int(value - self.low)] != 0
                    )
            if matched == self.want_match:
                rows.append(i)
        var selected = List[Series](capacity=len(columns))
        for column in columns:
            var part = Series(
                column.name(),
                column._chunked.value()[].arrays[self.chunk].copy(),
                column.dtype(),
                column._field_metadata,
            )
            if len(rows) > 0:
                selected.append(part.take(rows.copy()))
            else:
                selected.append(part.slice(0, 0))
        self.selected = selected^


def filter_range_int64_chunks(
    columns: List[Series],
    predicate: Int,
    var present: List[UInt8],
    low: Int64,
    high: Int64,
    consecutive: Bool,
    found: Bool,
    want_match: Bool,
) raises -> List[Series]:
    """Fuse bounded membership selection with aligned chunk gathers."""
    var shared_columns = ArcPointer(columns.copy())
    var shared_present = ArcPointer(present^)
    var jobs = List[_RangeMembershipChunkFilterJob]()
    for i in range(columns[0].n_chunks()):
        jobs.append(
            _RangeMembershipChunkFilterJob(
                shared_columns,
                predicate,
                i,
                shared_present,
                low,
                high,
                consecutive,
                found,
                want_match,
            )
        )
    var pool = Pool(min(configured_workers(), len(jobs)))
    pool.run(jobs)
    pool.release()
    var output = List[Series](capacity=len(columns))
    for c in range(len(columns)):
        var pieces = List[Series]()
        for i in range(len(jobs)):
            ref part = jobs[i].selected[c]
            if len(part) > 0:
                pieces.append(part.copy())
        if len(pieces) == 0:
            output.append(columns[c].slice(0, 0))
        else:
            output.append(Series._from_chunks(pieces))
    return output^


struct _SortedChunkTakeJob(Job):
    """Gather ordered filter indices from one column without rechunking it."""

    var source: Series
    var indices: ArcPointer[List[Int]]
    var allow_repeats: Bool
    var result: Series

    def __init__(
        out self,
        source: Series,
        indices: ArcPointer[List[Int]],
        allow_repeats: Bool,
    ):
        self.source = source.copy()
        self.indices = indices.copy()
        self.allow_repeats = allow_repeats
        self.result = source.copy()

    def run(mut self) raises:
        ref rows = self.indices[]
        if len(rows) == 0:
            self.result = self.source.slice(0, 0)
            return
        if not self.source.is_chunked():
            self.result = self.source.take(rows.copy())
            return
        var selected = List[Series]()
        var offset = 0
        var next_row = 0
        for chunk in self.source.chunks():
            var end = offset + len(chunk)
            # Filter indices are strictly increasing. Matching both endpoints
            # proves that this entire chunk is selected; retain its storage.
            if (
                not self.allow_repeats
                and len(chunk) > 0
                and next_row + len(chunk) <= len(rows)
                and rows[next_row] == offset
                and rows[next_row + len(chunk) - 1] == end - 1
            ):
                selected.append(chunk.copy())
                next_row += len(chunk)
                offset = end
                continue
            # The indices are sorted: find this chunk's share by bisection
            # and gather it in place, without copying it out (#328).
            var low = next_row
            var high = len(rows)
            while low < high:
                var middle = (low + high) // 2
                if rows[middle] < end:
                    low = middle + 1
                else:
                    high = middle
            if low > next_row:
                var gathered = chunk._take_range(
                    rows, next_row, low, offset, Optional(self.indices)
                )
                if gathered:
                    selected.append(gathered.take())
                else:
                    var local = List[Int](capacity=low - next_row)
                    for at in range(next_row, low):
                        local.append(rows[at] - offset)
                    selected.append(chunk.take(local))
            next_row = low
            offset = end
        self.result = Series._from_chunks(selected)

    def into_result(deinit self) -> Series:
        return self.result^


struct _SortedChunkPartJob(Job):
    """Gather one column from a contiguous range of physical chunks."""

    var source: Series
    var indices: ArcPointer[List[Int]]
    var first: Int
    var last: Int
    var allow_repeats: Bool
    var result: Series

    def __init__(
        out self,
        source: Series,
        indices: ArcPointer[List[Int]],
        first: Int,
        last: Int,
        allow_repeats: Bool,
    ):
        self.source = source.copy()
        self.indices = indices.copy()
        self.first = first
        self.last = last
        self.allow_repeats = allow_repeats
        self.result = source.copy()

    def run(mut self) raises:
        ref chunks = self.source._chunked.value()[]
        ref rows = self.indices[]
        var chunk_start = 0 if self.first == 0 else chunks.ends[self.first - 1]
        var lower = 0
        var upper = len(rows)
        while lower < upper:
            var mid = (lower + upper) // 2
            if rows[mid] < chunk_start:
                lower = mid + 1
            else:
                upper = mid
        var next_row = lower
        var selected = List[Series]()
        for i in range(self.first, self.last):
            var end = chunks.ends[i]
            var size = end - chunk_start
            if (
                not self.allow_repeats
                and size > 0
                and next_row + size <= len(rows)
                and rows[next_row] == chunk_start
                and rows[next_row + size - 1] == end - 1
            ):
                selected.append(
                    Series(
                        self.source.name(),
                        chunks.arrays[i].copy(),
                        self.source.dtype(),
                        self.source._field_metadata,
                    )
                )
                next_row += size
                chunk_start = end
                continue
            # The indices are sorted: this chunk's share ends at the first
            # row past it. Gather in place where the storage allows, so the
            # rows are neither copied out nor rebased.
            var low = next_row
            var high = len(rows)
            while low < high:
                var middle = (low + high) // 2
                if rows[middle] < end:
                    low = middle + 1
                else:
                    high = middle
            if low > next_row:
                var part = Series(
                    self.source.name(),
                    chunks.arrays[i].copy(),
                    self.source.dtype(),
                    self.source._field_metadata,
                )
                var gathered = part._take_range(
                    rows, next_row, low, chunk_start, Optional(self.indices)
                )
                if gathered:
                    selected.append(gathered.take())
                else:
                    var local = List[Int](capacity=low - next_row)
                    for at in range(next_row, low):
                        local.append(rows[at] - chunk_start)
                    selected.append(part.take(local))
            next_row = low
            chunk_start = end
        if len(selected) == 0:
            self.result = self.source.slice(0, 0)
        else:
            self.result = Series._from_chunks(selected)


def _take_sorted_chunked_partitioned(
    columns: List[Series],
    var indices: List[Int],
    parts: Int,
    allow_repeats: Bool = False,
) raises -> List[Series]:
    var shared = ArcPointer(indices^)
    var jobs = List[_SortedChunkPartJob](capacity=len(columns) * parts)
    for column in columns:
        var bounds = partitions(column.n_chunks(), parts, 1)
        for p in range(parts):
            jobs.append(
                _SortedChunkPartJob(
                    column, shared, bounds[p], bounds[p + 1], allow_repeats
                )
            )
    run_jobs(jobs)
    var result = List[Series](capacity=len(columns))
    for c in range(len(columns)):
        var pieces = List[Series](capacity=parts)
        for p in range(parts):
            ref part = jobs[c * parts + p].result
            if len(part) > 0:
                pieces.append(part.copy())
        if len(pieces) == 0:
            result.append(columns[c].slice(0, 0))
        else:
            result.append(Series._from_chunks(pieces))
    return result^


def take_sorted_chunked(
    columns: List[Series],
    var indices: List[Int],
    workers: Int,
    allow_repeats: Bool = False,
) raises -> List[Series]:
    """Filter source-ordered rows within physical chunks.

    Generic ``take_parallel`` still handles arbitrary join indices. Jobs run
    by column, so the selected chunks remain in source order in the result.
    """
    for column in columns:
        if column.dtype().is_nested():
            var gathered = List[Series](capacity=len(columns))
            for c in columns:
                gathered.append(c.take(indices))
            return gathered^
    if workers > 1 and len(columns) > 1:
        # One task per column and row range, a task per thread at most and
        # each at least one vector of rows. On a heterogeneous CPU only the
        # performance cores count: a range on a slower core finishes last.
        var parts = task_count(
            len(indices),
            min(configured_workers(), _performance_core_count())
            // len(columns),
        )
        if parts > 1:
            var chunked = True
            for column in columns:
                if not column.is_chunked():
                    chunked = False
                    break
            if chunked:
                return _take_sorted_chunked_partitioned(
                    columns, indices^, parts, allow_repeats
                )
    var jobs = List[_SortedChunkTakeJob](capacity=len(columns))
    var shared = ArcPointer(indices^)
    for column in columns:
        jobs.append(_SortedChunkTakeJob(column, shared, allow_repeats))
    if workers > 1 and len(jobs) > 1:
        run_jobs(jobs)
    else:
        for i in range(len(jobs)):
            jobs[i].run()
    var result = List[Series](capacity=len(jobs))
    while len(jobs) > 0:
        result.append(jobs.pop(0).into_result())
    return result^


struct _GatherJob(Job):
    """Rows `indices[start:end]` of one column into output positions
    [start, end), writing through raw output buffer addresses."""

    var source: Series
    var indices: ArcPointer[List[Int]]
    var start: Int
    var end: Int
    var values: Int  # address of the output payloads (fixed-width columns)
    var bits: Int  # address of the output validity bytes
    var piece: Series  # string columns: the gathered range
    # A join's unmatched rows carry index -1, meaning "no row on this side".
    var or_null: Bool
    var skip_validity: Bool
    var share_validity: Bool

    def __init__(
        out self,
        source: Series,
        indices: ArcPointer[List[Int]],
        start: Int,
        end: Int,
        values: Int,
        bits: Int,
        or_null: Bool = False,
        skip_validity: Bool = False,
        share_validity: Bool = False,
    ):
        self.source = source.copy()
        self.indices = indices
        self.start = start
        self.end = end
        self.values = values
        self.bits = bits
        self.piece = source.copy()
        self.or_null = or_null
        self.skip_validity = skip_validity
        self.share_validity = share_validity

    def run(mut self) raises:
        ref rows = self.indices[]
        if self.source._data.isa[StringColumn]():
            self.piece = Series(
                self.source.name(),
                self.source._data[StringColumn]._take_range(
                    rows,
                    self.start,
                    self.end,
                    self.or_null,
                    Optional(self.indices),
                ),
            )
            # Keep the logical type: binary shares the string layout.
            self.piece._dtype = self.source.dtype()
            self.piece._field_metadata = self.source._field_metadata
            return
        var out_bits = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=self.bits
        )
        if self.source._data.isa[BoolColumn]():
            # Values are bits too; output ranges start on byte boundaries.
            ref column = self.source._data[BoolColumn]
            var out = Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=self.values
            )
            if self.skip_validity:
                for k in range(self.start, self.end):
                    var row = rows[k]
                    if self.or_null and row < 0:
                        continue
                    var mask = UInt8(1) << UInt8(k % 8)
                    if column._get(row):
                        out.unsafe_offset(k // 8)[] |= mask
                    if self.or_null and not self.share_validity:
                        out_bits.unsafe_offset(k // 8)[] |= mask
                return
            for k in range(self.start, self.end):
                var row = rows[k]
                if self.or_null and row < 0:
                    continue  # output starts null, so leaving it is the null
                var mask = UInt8(1) << UInt8(k % 8)
                if column._get(row):
                    out.unsafe_offset(k // 8)[] |= mask
                if column._valid(row):
                    out_bits.unsafe_offset(k // 8)[] |= mask
            return
        comptime for d in range(len(GATHER_DTYPES)):
            comptime D = GATHER_DTYPES[d]
            if self.source._data.isa[Column[Scalar[D]]]():
                ref column = self.source._data[Column[Scalar[D]]]
                var out = Pointer[Scalar[D], MutAnyOrigin](
                    unsafe_from_address=self.values
                )
                # Callers validate indices or construct them from source row positions.
                var input = column._ptr()
                # The output starts uninitialized (see `_allocate`), so a
                # null-extended slot is written too, as zero. Indices through a
                # pointer and the flags decided once: a bounds check and two
                # field reads per row keep the CPU from overlapping the loop's
                # scattered loads, whose cache misses are the cost of a gather.
                var positions = rows.unsafe_ptr()
                if self.skip_validity and not self.or_null:
                    for k in range(self.start, self.end):
                        out.unsafe_offset(k)[] = input.unsafe_offset(
                            positions.unsafe_offset(k)[]
                        )[]
                    return
                if self.skip_validity:
                    var mark = not self.share_validity
                    for k in range(self.start, self.end):
                        var row = positions.unsafe_offset(k)[]
                        if row < 0:
                            out.unsafe_offset(k)[] = 0
                            continue
                        out.unsafe_offset(k)[] = input.unsafe_offset(row)[]
                        if mark:
                            out_bits.unsafe_offset(k // 8)[] |= UInt8(
                                1
                            ) << UInt8(k % 8)
                    return
                # Assemble validity bytes in local words, then write each
                # output word once. Updating a single bit at a time would
                # repeatedly read and modify the same byte.
                var or_null = self.or_null
                var source_bits = column.unsafe_validity()
                var first = column.validity_offset()
                var k = self.start
                while k < self.end:
                    var stop = min(k + 8, self.end)
                    var byte = UInt8(0)
                    for j in range(k, stop):
                        var row = positions.unsafe_offset(j)[]
                        if or_null and row < 0:
                            out.unsafe_offset(j)[] = 0
                            continue
                        out.unsafe_offset(j)[] = input.unsafe_offset(row)[]
                        var bit = first + row
                        byte |= (
                            (
                                source_bits.unsafe_offset(bit >> 3)[]
                                >> UInt8(bit & 7)
                            )
                            & 1
                        ) << UInt8(j - k)
                    out_bits.unsafe_offset(k // 8)[] = byte
                    k = stop


struct _RechunkJob(Job):
    """Materialize one input column before a parallel gather."""

    var source: Series
    var result: Series

    def __init__(out self, source: Series):
        self.source = source.copy()
        self.result = source.copy()

    def run(mut self) raises:
        self.result = self.source.rechunk()

    def into_result(deinit self) -> Series:
        return self.result^


def take_parallel(
    columns: List[Series],
    var indices: List[Int],
    workers: Int,
    or_null: Bool = False,
) raises -> List[Series]:
    """Gather `indices` (already bounds-checked) from every column.

    With `or_null`, a negative index yields a null instead of a row, which
    is how a join names the side that has no matching row.
    """
    var rounds = GatherRounds(columns, indices^, workers, or_null)
    run_rounds(rounds)
    return rounds.take_result()


# The stages of a `GatherRounds`: the round that ran last.
comptime _GATHER_START = 0
comptime _GATHER_RECHUNK = 1
comptime _GATHER_ROWS = 2
comptime _GATHER_DONE = 3


struct GatherRounds(Movable, Rounds):
    """`take_parallel` as rounds of tasks: chunked columns are first made
    contiguous, a fixed-width number column by ranges of its rows (each
    a piece of the round), any other column whole; then each task
    gathers one range of output rows of every column, writing straight
    into the outputs. A
    constant column (unless `or_null`) is the same value over the gathered
    rows, and list and struct columns gather through their own take, so
    neither is gathered by the tasks. `take_result` once no round is
    left."""

    var columns: List[Series]
    var indices: ArcPointer[List[Int]]
    var workers: Int
    var or_null: Bool
    var stage: Int
    # By column: whether it is resized rather than gathered; the gathered
    # columns, made contiguous by the first round.
    var constant: List[Bool]
    var gathered: List[Series]
    var chunked: List[Int]
    # By chunked column: its merge by row ranges (none: rechunked whole);
    # the round's pieces (chunked column, first row, end row; -1 for the
    # whole column).
    var merges: List[Optional[ChunkMerge]]
    var pieces: List[Tuple[Int, Int, Int]]
    var width: Int
    # The second round's ranges, outputs and validity, and its jobs (by
    # gathered column, then by range).
    var active: Int
    var outputs: List[Series]
    var bits: List[List[UInt8]]
    var skip_validity: List[Bool]
    var share_validity: Bool
    var shared_bits: _Buffer[UInt8]
    var jobs: List[_GatherJob]
    var result: List[Series]

    def __init__(
        out self,
        columns: List[Series],
        var indices: List[Int],
        workers: Int,
        or_null: Bool = False,
    ):
        self.columns = columns.copy()
        self.indices = ArcPointer(indices^)
        self.workers = workers
        self.or_null = or_null
        self.stage = _GATHER_START
        self.constant = List[Bool]()
        self.gathered = List[Series]()
        self.chunked = List[Int]()
        self.merges = List[Optional[ChunkMerge]]()
        self.pieces = List[Tuple[Int, Int, Int]]()
        self.width = 0
        self.active = 0
        self.outputs = List[Series]()
        self.bits = List[List[UInt8]]()
        self.skip_validity = List[Bool]()
        self.share_validity = False
        self.shared_bits = _Buffer(List[UInt8]())
        self.jobs = List[_GatherJob]()
        self.result = List[Series]()

    def next_round(mut self) raises -> Int:
        if self.stage == _GATHER_START:
            for column in self.columns:
                var resized = not self.or_null and column.is_constant()
                self.constant.append(resized)
                if not resized:
                    self.gathered.append(column.copy())
            for column in self.gathered:
                if column.dtype().is_nested():
                    # List and struct columns gather through their own
                    # take; the payload scatter below only knows
                    # fixed-width and string buffers.
                    var taken = List[Series](capacity=len(self.gathered))
                    ref rows = self.indices[]
                    for c in self.gathered:
                        taken.append(
                            c.take_or_null(rows) if self.or_null else c.take(
                                rows
                            )
                        )
                    self.gathered = taken^
                    self._finish()
                    return 0
            for c in range(len(self.gathered)):
                if not self.gathered[c].is_chunked():
                    continue
                var k = len(self.chunked)
                self.chunked.append(c)
                ref column = self.gathered[c]
                var merge = ChunkMerge.of(column)
                if merge:
                    # Its rows split over the workers, as a column copied
                    # on its own would split them.
                    var n = len(column)
                    var bounds = partitions(n, worker_count(n), 64)
                    for w in range(len(bounds) - 1):
                        if bounds[w] < bounds[w + 1]:
                            self.pieces.append((k, bounds[w], bounds[w + 1]))
                else:
                    self.pieces.append((k, 0, -1))
                self.merges.append(merge^)
            self.stage = _GATHER_RECHUNK
            if len(self.pieces) > 0:
                self.width = round_width(len(self.pieces))
                return self.width
        if self.stage == _GATHER_RECHUNK:
            for k in range(len(self.chunked)):
                if self.merges[k]:
                    self.gathered[self.chunked[k]] = (
                        self.merges[k].value().merged.copy()
                    )
            self.merges = List[Optional[ChunkMerge]]()
            self.pieces = List[Tuple[Int, Int, Int]]()
            return self._start_rows()
        if self.stage == _GATHER_ROWS:
            self._finish_rows()
        return 0

    def round_task(mut self, task: Int) raises:
        if self.stage == _GATHER_RECHUNK:
            for p in range(task, len(self.pieces), self.width):
                var piece = self.pieces[p]
                var k = piece[0]
                if self.merges[k]:
                    self.merges[k].value().copy_rows(piece[1], piece[2])
                else:
                    var c = self.chunked[k]
                    self.gathered[c] = self.gathered[c].rechunk()
        elif self.stage == _GATHER_ROWS:
            for c in range(len(self.gathered)):
                self.jobs[c * self.active + task].run()

    def _start_rows(mut self) raises -> Int:
        ref columns = self.gathered
        var m = len(self.indices[])
        self.active = task_count(m, self.workers)
        var nonnullable_fixed = 0
        if self.or_null and self.active < self.workers:
            for column in columns:
                if (
                    not column._data.isa[StringColumn]()
                    and column.null_count() == 0
                ):
                    nonnullable_fixed += 1
        # A shared bitmap amortizes the index scan when several fixed-width
        # columns would otherwise write the same null-extension bits.
        self.share_validity = nonnullable_fixed >= 3
        var common_bits = List[UInt8]()
        if self.share_validity:
            common_bits = List[UInt8](length=(m + 7) // 8, fill=0)
            ref rows = self.indices[]
            for i in range(m):
                if rows[i] >= 0:
                    common_bits[i // 8] |= UInt8(1) << UInt8(i % 8)
        self.shared_bits = _Buffer(common_bits^)
        var bounds = partitions(m, self.active, 8)
        # Preallocate each fixed-width output; strings are assembled from
        # pieces.
        for column in columns:
            # An absent output bitmap means every gathered row is valid.
            # Check once per column instead of reading and writing
            # validity per row.
            var source_all_valid = column.null_count() == 0
            self.skip_validity.append(source_all_valid)
            var needs_own_bits = not source_all_valid or (
                self.or_null and not self.share_validity
            )
            self.bits.append(
                List[UInt8](
                    length=(m + 7) // 8, fill=0
                ) if needs_own_bits else List[UInt8]()
            )
            self.outputs.append(_allocate(column, m))
        for c in range(len(columns)):
            var values = _payload_address(self.outputs[c])
            for w in range(self.active):
                self.jobs.append(
                    _GatherJob(
                        columns[c],
                        self.indices,
                        bounds[w],
                        bounds[w + 1],
                        values,
                        Int(self.bits[c].unsafe_ptr()),
                        self.or_null,
                        self.skip_validity[c],
                        self.share_validity and self.skip_validity[c],
                    )
                )
        self.stage = _GATHER_ROWS
        if len(columns) == 0:
            self._finish_rows()
            return 0
        return self.active

    def _finish_rows(mut self) raises:
        var result = List[Series](capacity=len(self.gathered))
        for c in range(len(self.gathered)):
            if self.gathered[c]._data.isa[StringColumn]():
                # Keep each task's gathered view as a physical chunk.
                # Appending views here repeatedly copies the growing
                # descriptor array.
                var pieces = List[Series](capacity=self.active)
                for w in range(self.active):
                    pieces.append(self.jobs[c * self.active + w].piece.copy())
                result.append(Series._from_chunks(pieces))
                continue
            var output = self.outputs[c].copy()
            if self.share_validity and self.skip_validity[c]:
                _set_shared_bits(output, self.shared_bits)
            else:
                _set_bits(output, self.bits[c].copy())
            result.append(output^)
        self.jobs = List[_GatherJob]()
        self.bits = List[List[UInt8]]()
        self.outputs = List[Series]()
        self.gathered = result^
        self._finish()

    def _finish(mut self):
        var m = len(self.indices[])
        var next = 0
        for c in range(len(self.columns)):
            if self.constant[c]:
                self.result.append(self.columns[c]._constant_resized(m))
            else:
                self.result.append(self.gathered[next].copy())
                next += 1
        self.gathered = List[Series]()
        self.stage = _GATHER_DONE

    def take_result(mut self) -> List[Series]:
        """The gathered columns, moved out once no round is left."""
        var taken = List[Series]()
        swap(taken, self.result)
        return taken^


def _allocate(column: Series, m: Int) raises -> Series:
    """An m-row column of column's dtype whose payloads workers overwrite.

    Fixed-width payloads are left uninitialized: every gather job writes
    each position in its range, so filling them first only adds a serial
    pass -- about 35 ms per 10M-row column, most of a sort's gather -- and
    leaves the workers' first touch of each page to happen on one thread.
    The validity bitmap is set by the caller afterwards.
    """
    if column._data.isa[StringColumn]():
        return column.copy()
    comptime for d in range(len(GATHER_DTYPES)):
        comptime D = GATHER_DTYPES[d]
        if column._data.isa[Column[Scalar[D]]]():
            var output = Series(
                column.name(),
                Column[Scalar[D]](huge_uninit[Scalar[D]](m)),
            )
            output._dtype = column.dtype()
            output._field_metadata = column._field_metadata
            return output^
    var output = Series.full_null(column.name(), column.dtype(), m)
    output._field_metadata = column._field_metadata
    return output^


def _payload_address(series: Series) -> Int:
    if series._data.isa[BoolColumn]():
        return Int(series._data[BoolColumn]._data[].unsafe_ptr())
    comptime for d in range(len(GATHER_DTYPES)):
        comptime D = GATHER_DTYPES[d]
        if series._data.isa[Column[Scalar[D]]]():
            return Int(series._data[Column[Scalar[D]]]._ptr())
    return 0


def _set_bits(mut series: Series, var bits: List[UInt8]):
    if series._data.isa[BoolColumn]():
        series._data[BoolColumn]._bits = _Buffer(bits^)
        return
    comptime for d in range(len(GATHER_DTYPES)):
        comptime D = GATHER_DTYPES[d]
        if series._data.isa[Column[Scalar[D]]]():
            series._data[Column[Scalar[D]]]._bits = _Buffer(bits^)
            return


def _set_shared_bits(mut series: Series, bits: _Buffer[UInt8]):
    if series._data.isa[BoolColumn]():
        series._data[BoolColumn]._bits = bits.copy()
        return
    comptime for d in range(len(GATHER_DTYPES)):
        comptime D = GATHER_DTYPES[d]
        if series._data.isa[Column[Scalar[D]]]():
            series._data[Column[Scalar[D]]]._bits = bits.copy()
            return
