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
from std.memory import ArcPointer, Pointer
from .bool_column import BoolColumn
from .column import Column, _bit, _validity_bit
from .dtype import DataType, NUMERIC_DTYPES
from .expr import GT, LT, GE, LE, EQ, NE
from .parallel import (
    Job,
    Pool,
    configured_workers,
    partitions,
    run_jobs,
    worker_count,
)
from .series import Series
from .string_column import StringColumn


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
        ref values = self.mask._data[]
        ref bits = self.mask._bits[]
        var base = self.mask._offset
        for i in range(self.start, self.end):
            var row = base + i
            # Values and validity are both LSB-first bitmaps.
            if _bit(values, row) and _validity_bit(bits, row):
                self.rows.append(i)

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


def can_filter_float_chunks(columns: List[Series]) -> Bool:
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


struct _FloatChunkFilterJob(Job):
    """Scan one predicate chunk and gather its columns from local row IDs."""

    var columns: ArcPointer[List[Series]]
    var predicate: Int
    var chunk: Int
    var op: Int
    var literal: Float64
    var selected: List[Series]

    def __init__(
        out self,
        columns: ArcPointer[List[Series]],
        predicate: Int,
        chunk: Int,
        op: Int,
        literal: Float64,
    ):
        self.columns = columns.copy()
        self.predicate = predicate
        self.chunk = chunk
        self.op = op
        self.literal = literal
        self.selected = List[Series]()

    def run(mut self) raises:
        ref columns = self.columns[]
        var pred = Series(
            columns[self.predicate].name(),
            columns[self.predicate]
            ._chunked.value()[]
            .arrays[self.chunk]
            .copy(),
            columns[self.predicate].dtype(),
        )
        var rows = List[Int]()
        _scan_float_compare(
            rows,
            pred._data[Column[Float64]],
            0,
            0,
            len(pred),
            self.op,
            self.literal,
        )
        var selected = List[Series](capacity=len(columns))
        for column in columns:
            var part = Series(
                column.name(),
                column._chunked.value()[].arrays[self.chunk].copy(),
                column.dtype(),
            )
            if len(rows) > 0:
                selected.append(part.take(rows.copy()))
            else:
                selected.append(part.slice(0, 0))
        self.selected = selected^


def filter_float_chunks(
    columns: List[Series], predicate: Int, op: Int, literal: Float64
) raises -> List[Series]:
    """Filter aligned source chunks independently, keeping their row order."""
    var shared = ArcPointer(columns.copy())
    var jobs = List[_FloatChunkFilterJob]()
    for i in range(columns[0].n_chunks()):
        jobs.append(_FloatChunkFilterJob(shared, predicate, i, op, literal))
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
            output.append(Series._from_chunks(pieces^))
    return output^


struct _SortedChunkTakeJob(Job):
    """Gather ordered filter indices from one column without rechunking it."""

    var source: Series
    var indices: ArcPointer[List[Int]]
    var result: Series

    def __init__(out self, source: Series, indices: ArcPointer[List[Int]]):
        self.source = source.copy()
        self.indices = indices.copy()
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
            var local = List[Int]()
            while next_row < len(rows) and rows[next_row] < end:
                local.append(rows[next_row] - offset)
                next_row += 1
            if len(local) > 0:
                selected.append(chunk.take(local))
            offset = end
        self.result = Series._from_chunks(selected^)

    def into_result(deinit self) -> Series:
        return self.result^


struct _SortedChunkPartJob(Job):
    """Gather one column from a contiguous range of physical chunks."""

    var source: Series
    var indices: ArcPointer[List[Int]]
    var first: Int
    var last: Int
    var result: Series

    def __init__(
        out self,
        source: Series,
        indices: ArcPointer[List[Int]],
        first: Int,
        last: Int,
    ):
        self.source = source.copy()
        self.indices = indices.copy()
        self.first = first
        self.last = last
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
            var local = List[Int]()
            while next_row < len(rows) and rows[next_row] < end:
                local.append(rows[next_row] - chunk_start)
                next_row += 1
            if len(local) > 0:
                var part = Series(
                    self.source.name(),
                    chunks.arrays[i].copy(),
                    self.source.dtype(),
                )
                selected.append(part.take(local))
            chunk_start = end
        if len(selected) == 0:
            self.result = self.source.slice(0, 0)
        else:
            self.result = Series._from_chunks(selected^)


def _take_sorted_chunked_partitioned(
    columns: List[Series], var indices: List[Int], parts: Int
) raises -> List[Series]:
    var shared = ArcPointer(indices^)
    var jobs = List[_SortedChunkPartJob](capacity=len(columns) * parts)
    for column in columns:
        var bounds = partitions(column.n_chunks(), parts, 1)
        for p in range(parts):
            jobs.append(
                _SortedChunkPartJob(column, shared, bounds[p], bounds[p + 1])
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
            result.append(Series._from_chunks(pieces^))
    return result^


def take_sorted_chunked(
    columns: List[Series], var indices: List[Int], workers: Int
) raises -> List[Series]:
    """Filter source-ordered rows within physical chunks.

    Generic ``take_parallel`` still handles arbitrary join indices. Jobs run
    by column, so the selected chunks remain in source order in the result.
    """
    if len(indices) >= 2_000_000 and len(columns) > 1:
        var parts = min(4, configured_workers() // len(columns))
        if parts > 1:
            var chunked = True
            for column in columns:
                if not column.is_chunked() or column.n_chunks() < 16:
                    chunked = False
                    break
            if chunked:
                return _take_sorted_chunked_partitioned(
                    columns, indices^, parts
                )
    var jobs = List[_SortedChunkTakeJob](capacity=len(columns))
    var shared = ArcPointer(indices^)
    for column in columns:
        jobs.append(_SortedChunkTakeJob(column, shared))
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

    def __init__(
        out self,
        source: Series,
        indices: ArcPointer[List[Int]],
        start: Int,
        end: Int,
        values: Int,
        bits: Int,
        or_null: Bool = False,
    ):
        self.source = source.copy()
        self.indices = indices
        self.start = start
        self.end = end
        self.values = values
        self.bits = bits
        self.piece = source.copy()
        self.or_null = or_null

    def run(mut self) raises:
        ref rows = self.indices[]
        if self.source._data.isa[StringColumn]():
            var subset = List[Int](capacity=self.end - self.start)
            for k in range(self.start, self.end):
                subset.append(rows[k])
            self.piece = self.source.take_or_null(
                subset
            ) if self.or_null else self.source.take(subset)
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
        comptime for d in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[d]
            if self.source._data.isa[Column[Scalar[D]]]():
                ref column = self.source._data[Column[Scalar[D]]]
                var out = Pointer[Scalar[D], MutAnyOrigin](
                    unsafe_from_address=self.values
                )
                for k in range(self.start, self.end):
                    var row = rows[k]
                    if self.or_null and row < 0:
                        continue
                    out.unsafe_offset(k)[] = column._get(row)
                    if column._valid(row):
                        out_bits.unsafe_offset(k // 8)[] |= UInt8(1) << UInt8(
                            k % 8
                        )

    def into_piece(deinit self) -> Series:
        return self.piece^


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
    var contiguous = List[Series](capacity=len(columns))
    var rechunk = List[_RechunkJob]()
    var positions = List[Int]()
    for c in range(len(columns)):
        contiguous.append(columns[c].copy())
        if columns[c].is_chunked():
            positions.append(c)
            rechunk.append(_RechunkJob(columns[c]))
    if len(rechunk) > 0:
        if len(rechunk) == 1:
            rechunk[0].run()
        else:
            run_jobs(rechunk)
        while len(rechunk) > 0:
            contiguous[positions.pop(0)] = rechunk.pop(0).into_result()
        return take_parallel(contiguous^, indices^, workers, or_null)
    var m = len(indices)
    var shared = ArcPointer(indices^)
    var bounds = partitions(m, workers, 8)
    # Preallocate each fixed-width output; strings are assembled from pieces.
    var outputs = List[Series](capacity=len(columns))
    var bits = List[List[UInt8]](capacity=len(columns))
    for column in columns:
        bits.append(List[UInt8](length=(m + 7) // 8, fill=0))
        outputs.append(_allocate(column, m))
    var jobs = List[_GatherJob](capacity=len(columns) * workers)
    for c in range(len(columns)):
        var values = _payload_address(outputs[c])
        for w in range(workers):
            jobs.append(
                _GatherJob(
                    columns[c],
                    shared,
                    bounds[w],
                    bounds[w + 1],
                    values,
                    Int(bits[c].unsafe_ptr()),
                    or_null,
                )
            )
    run_jobs(jobs)
    var result = List[Series](capacity=len(columns))
    for c in range(len(columns)):
        if columns[c]._data.isa[StringColumn]():
            # Keep each worker's gathered view as a physical chunk. Appending
            # views here repeatedly copies the growing descriptor array.
            var pieces = List[Series](capacity=workers)
            for w in range(workers):
                pieces.append(jobs[c * workers + w].piece.copy())
            result.append(Series._from_chunks(pieces^))
            continue
        var output = outputs[c].copy()
        _set_bits(output, bits[c].copy())
        result.append(output^)
    _ = bits^
    _ = outputs^
    return result^


def _allocate(column: Series, m: Int) raises -> Series:
    """An m-row column of column's dtype whose payloads workers overwrite."""
    if column._data.isa[StringColumn]():
        return column.copy()
    return Series.full_null(column.name(), column.dtype(), m)


def _payload_address(series: Series) -> Int:
    if series._data.isa[BoolColumn]():
        return Int(series._data[BoolColumn]._data[].unsafe_ptr())
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if series._data.isa[Column[Scalar[D]]]():
            return Int(series._data[Column[Scalar[D]]]._ptr())
    return 0


def _set_bits(mut series: Series, var bits: List[UInt8]):
    if series._data.isa[BoolColumn]():
        series._data[BoolColumn]._bits = ArcPointer(bits^)
        return
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if series._data.isa[Column[Scalar[D]]]():
            series._data[Column[Scalar[D]]]._bits = ArcPointer(bits^)
            return
