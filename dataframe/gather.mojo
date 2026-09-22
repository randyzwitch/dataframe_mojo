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
from .parallel import Job, partitions, run_jobs, worker_count
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
    var rows = jobs.pop(0).into_rows()
    while len(jobs) > 0:
        rows.extend(Span(jobs.pop(0).into_rows()))
    return rows^


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


def take_sorted_chunked(
    columns: List[Series], var indices: List[Int], workers: Int
) raises -> List[Series]:
    """Filter source-ordered rows within physical chunks.

    Generic ``take_parallel`` still handles arbitrary join indices. Jobs run
    by column, so the selected chunks remain in source order in the result.
    """
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
    for column in columns:
        if column.is_chunked():
            var contiguous = List[Series](capacity=len(columns))
            for item in columns:
                contiguous.append(item.rechunk())
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
            var assembled = jobs[c * workers].piece.copy()
            for w in range(1, workers):
                assembled._append_series(jobs[c * workers + w].piece)
            result.append(assembled^)
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
