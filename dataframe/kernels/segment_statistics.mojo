"""Segment statistics of column buffers: computed, published, and read
back for a window.

Statistics belong to a buffer, not to a column window: every window onto
the buffer reads the same `_StatsRecord` (`storage/buffer.mojo`), one entry
per `STATISTICS_VECTOR_SIZE` rows from the buffer's first element and one
per `ROW_GROUP_VECTORS` vectors. A window that covers part of a vector reads
that vector's statistics, which hold every value of the window and possibly
more: wider bounds rule fewer rows out but never a row that is there.

Loaded tables receive statistics when read (`compute_table_statistics`).
Other frames have none unless requested (`ensure_statistics`). Without
statistics a scan reads every region, because it has no bounds that could
prove rows irrelevant.
"""
from std.memory import ArcPointer, Pointer
from std.collections import Optional

from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.kernels.validity import _valid_lanes
from dataframe.parallel import Job, run_jobs, configured_workers
from dataframe.series import Series, Storage
from dataframe.storage.boolean import BoolColumn
from dataframe.storage.buffer import _Buffer, _StatsRecord
from dataframe.storage.column import Column
from dataframe.storage.statistics import (
    BASE_STATS,
    BaseStatistics,
    FLOAT_STATS,
    MAX_STRING_MINMAX_SIZE,
    NUMERIC_STATS,
    ROW_GROUP_VECTORS,
    STATISTICS_VECTOR_SIZE,
    STRING_STATS,
)
from dataframe.storage.string import StringColumn


def _null_flags(
    mut stats: BaseStatistics,
    bits: Pointer[UInt8, MutAnyOrigin],
    has_bits: Bool,
    start: Int,
    end: Int,
):
    """has_null / has_no_null of buffer rows [start, end)."""
    if not has_bits:
        stats.has_null = False
        stats.has_no_null = end > start
        return
    var valid = 0
    for r in range(start, end):
        valid += Int((bits[unsafe_offset=r >> 3] >> UInt8(r & 7)) & 1)
    stats.has_null = valid < end - start
    stats.has_no_null = valid > 0


def _numeric_vector[
    D: DType
](
    values: Pointer[Scalar[D], MutAnyOrigin],
    bits: Pointer[UInt8, MutAnyOrigin],
    has_bits: Bool,
    start: Int,
    end: Int,
) -> BaseStatistics:
    """Statistics of buffer rows [start, end) of a numeric buffer, eight
    rows at a time."""
    comptime is_float = D.is_floating_point()
    var stats = BaseStatistics.empty(FLOAT_STATS if is_float else NUMERIC_STATS)
    _null_flags(stats, bits, has_bits, start, end)
    if not stats.has_no_null:
        return stats
    comptime W = 8
    # Infinities for floats.
    var least = SIMD[D, W](Scalar[D].MAX)
    var most = SIMD[D, W](Scalar[D].MIN)
    var nan = SIMD[DType.bool, W](fill=False)
    var any = SIMD[DType.bool, W](fill=False)
    var i = start
    while i + W <= end:
        var v = values.unsafe_load[width=W](i)
        var valid = SIMD[DType.bool, W](fill=True)
        if has_bits:
            valid = _valid_lanes(bits, i)
        comptime if is_float:
            var is_nan = v.ne(v)
            nan |= valid & is_nan
            valid &= ~is_nan
        least = valid.select(min(least, v), least)
        most = valid.select(max(most, v), most)
        any |= valid
        i += W
    var low = least.reduce_min()
    var high = most.reduce_max()
    var found = any.reduce_or()
    var saw_nan = nan.reduce_or()
    while i < end:
        var valid = not has_bits or Bool(
            (bits[unsafe_offset=i >> 3] >> UInt8(i & 7)) & 1
        )
        if valid:
            var v = values[unsafe_offset=i]
            comptime if is_float:
                if v != v:
                    saw_nan = True
                    i += 1
                    continue
            low = min(low, v)
            high = max(high, v)
            found = True
        i += 1
    comptime if is_float:
        # NaN is kept apart from the bounds (`check_float_zonemap`).
        stats.has_nan = saw_nan
        if not found:
            return stats
        stats.has_min_max = True
        stats.float_min = Float64(low)
        stats.float_max = Float64(high)
    else:
        if not found:
            return stats
        stats.has_min_max = True
        stats.int_min = Int128(low)
        stats.int_max = Int128(high)
    return stats


def _wide_vector(
    values: Pointer[Int128, MutAnyOrigin],
    bits: Pointer[UInt8, MutAnyOrigin],
    has_bits: Bool,
    start: Int,
    end: Int,
) -> BaseStatistics:
    var stats = BaseStatistics.empty(NUMERIC_STATS)
    _null_flags(stats, bits, has_bits, start, end)
    var found = False
    var low = Int128.MAX
    var high = Int128.MIN
    for i in range(start, end):
        if has_bits and not Bool(
            (bits[unsafe_offset=i >> 3] >> UInt8(i & 7)) & 1
        ):
            continue
        var v = values[unsafe_offset=i]
        low = min(low, v)
        high = max(high, v)
        found = True
    if found:
        stats.has_min_max = True
        stats.int_min = low
        stats.int_max = high
    return stats


def _less(a: Span[UInt8, _], b: Span[UInt8, _]) -> Bool:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return a[i] < b[i]
    return len(a) < len(b)


def _prefix(bytes: Span[UInt8, _]) -> SIMD[DType.uint8, 16]:
    var out = SIMD[DType.uint8, 16](0)
    for i in range(min(len(bytes), MAX_STRING_MINMAX_SIZE)):
        out[i] = bytes[i]
    return out


def _string_vector(
    window: StringColumn, has_bits: Bool, start: Int, end: Int
) -> BaseStatistics:
    """Statistics of rows [start, end) of a string column's buffers
    (`window` is the column at offset 0). The least and greatest value are
    kept to their first MAX_STRING_MINMAX_SIZE bytes, truncated beyond."""
    var stats = BaseStatistics.empty(STRING_STATS)
    var bits = window.unsafe_validity()
    _null_flags(stats, bits, has_bits, start, end)
    if not stats.has_no_null:
        return stats
    var views = window._is_view_storage()
    var byte_count = 0 if views else len(window._bytes[])
    var low = Span[UInt8, ImmutAnyOrigin](unsafe_ptr=window._base(), length=0)
    var high = low
    var found = False
    var unicode = False
    var shortest = Int.MAX
    var longest = 0
    for r in range(start, end):
        if has_bits and not Bool(
            (bits[unsafe_offset=r >> 3] >> UInt8(r & 7)) & 1
        ):
            continue
        if not views:
            var first = window._start(r)
            var last = window._end(r)
            if first < 0 or last < first or last > byte_count:
                # Offsets outside the bytes: nothing is known.
                return BaseStatistics.unknown()
        var bytes = window._row_bytes(r)
        shortest = min(shortest, len(bytes))
        longest = max(longest, len(bytes))
        if not unicode:
            for b in bytes:
                if b >= 0x80:
                    unicode = True
                    break
        if not found:
            low = bytes
            high = bytes
            found = True
        else:
            if _less(bytes, low):
                low = bytes
            if _less(high, bytes):
                high = bytes
    stats.has_unicode = unicode
    stats.min_string_length = shortest
    stats.max_string_length = longest
    if found:
        stats.has_min_max = True
        stats.str_min = _prefix(low)
        stats.str_max = _prefix(high)
        stats.str_min_size = min(len(low), MAX_STRING_MINMAX_SIZE)
        stats.str_max_size = min(len(high), MAX_STRING_MINMAX_SIZE)
        stats.min_exact = len(low) <= MAX_STRING_MINMAX_SIZE
        stats.max_exact = len(high) <= MAX_STRING_MINMAX_SIZE
    return stats


@fieldwise_init
struct _Layout(Copyable, Movable):
    """Which buffer rows [first, rows) a storage's statistics describe, and
    with which bitmap. `kind` is -1 when it has none."""

    var kind: Int
    var first: Int
    var rows: Int
    var validity: Int
    var has_bits: Bool


comptime _KIND_STRING = 100
comptime _KIND_BOOL = 101
comptime _KIND_WIDE = 102


def _layout(storage: Storage) -> _Layout:
    """The buffer rows a storage's statistics cover: a column's declared
    window (`_BufferFacts`), or a string buffer's every row; rows past the
    validity bitmap's end are left out."""
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if storage.isa[Column[Scalar[D]]]():
            ref column = storage[Column[Scalar[D]]]
            var rows = column._facts[].last(len(column._data[]))
            var has_bits = len(column._bits[]) > 0
            if has_bits:
                rows = min(rows, 8 * len(column._bits[]))
            return _Layout(
                d,
                column._facts[].first(),
                rows,
                Int(column._bits.ptr()) if has_bits else 0,
                has_bits,
            )
    if storage.isa[Column[Int128]]():
        ref column = storage[Column[Int128]]
        var rows = column._facts[].last(len(column._data[]))
        var has_bits = len(column._bits[]) > 0
        if has_bits:
            rows = min(rows, 8 * len(column._bits[]))
        return _Layout(
            _KIND_WIDE,
            column._facts[].first(),
            rows,
            Int(column._bits.ptr()) if has_bits else 0,
            has_bits,
        )
    if storage.isa[StringColumn]():
        ref column = storage[StringColumn]
        var rows: Int
        if column._is_view_storage():
            ref held = column._view_storage.value()
            if held._selects():
                return _Layout(-1, 0, 0, 0, False)
            rows = len(held._views[])
        else:
            rows = len(column._offsets[]) - 1
        var has_bits = len(column._bits[]) > 0
        if has_bits:
            rows = min(rows, 8 * len(column._bits[]))
        return _Layout(
            _KIND_STRING,
            0,
            max(rows, 0),
            Int(column._bits.ptr()) if has_bits else 0,
            has_bits,
        )
    return _Layout(-1, 0, 0, 0, False)


def _record_of(storage: Storage) -> Optional[ArcPointer[_StatsRecord]]:
    """The statistics published on the storage's values buffer."""
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if storage.isa[Column[Scalar[D]]]():
            return storage[Column[Scalar[D]]]._data[].statistics()
    if storage.isa[Column[Int128]]():
        return storage[Column[Int128]]._data[].statistics()
    if storage.isa[StringColumn]():
        ref column = storage[StringColumn]
        if column._is_view_storage():
            return column._view_storage.value()._views[].statistics()
        return column._offsets[].statistics()
    return None


def _publish(storage: Storage, var record: _StatsRecord):
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if storage.isa[Column[Scalar[D]]]():
            storage[Column[Scalar[D]]]._data[].publish_statistics(record^)
            return
    if storage.isa[Column[Int128]]():
        storage[Column[Int128]]._data[].publish_statistics(record^)
        return
    if storage.isa[StringColumn]():
        ref column = storage[StringColumn]
        if column._is_view_storage():
            column._view_storage.value()._views[].publish_statistics(record^)
        else:
            column._offsets[].publish_statistics(record^)


def _pin(storage: Storage) -> Optional[_Buffer[UInt8]]:
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if storage.isa[Column[Scalar[D]]]():
            return storage[Column[Scalar[D]]]._bits.copy()
    if storage.isa[Column[Int128]]():
        return storage[Column[Int128]]._bits.copy()
    if storage.isa[StringColumn]():
        return storage[StringColumn]._bits.copy()
    return None


def _vectors(
    storage: Storage, layout: _Layout, first: Int, last: Int
) -> List[BaseStatistics]:
    """Statistics of vectors [first, last) of a storage's buffer."""
    var out = List[BaseStatistics](capacity=last - first)
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if layout.kind == d:
            ref column = storage[Column[Scalar[D]]]
            var values = column._data[].unsafe_ptr()
            var bits = column._bits[].unsafe_ptr()
            for v in range(first, last):
                out.append(
                    _numeric_vector[D](
                        values,
                        bits,
                        layout.has_bits,
                        max(v * STATISTICS_VECTOR_SIZE, layout.first),
                        min((v + 1) * STATISTICS_VECTOR_SIZE, layout.rows),
                    )
                )
            return out^
    if layout.kind == _KIND_WIDE:
        ref column = storage[Column[Int128]]
        var values = column._data[].unsafe_ptr()
        var bits = column._bits[].unsafe_ptr()
        for v in range(first, last):
            out.append(
                _wide_vector(
                    values,
                    bits,
                    layout.has_bits,
                    max(v * STATISTICS_VECTOR_SIZE, layout.first),
                    min((v + 1) * STATISTICS_VECTOR_SIZE, layout.rows),
                )
            )
        return out^
    if layout.kind == _KIND_STRING:
        var window = storage[StringColumn].copy()
        window._offset = 0
        window._length = layout.rows
        for v in range(first, last):
            out.append(
                _string_vector(
                    window,
                    layout.has_bits,
                    v * STATISTICS_VECTOR_SIZE,
                    min((v + 1) * STATISTICS_VECTOR_SIZE, layout.rows),
                )
            )
        return out^
    return out^


def _groups(vectors: List[BaseStatistics]) -> List[BaseStatistics]:
    """One merged entry per ROW_GROUP_VECTORS vectors."""
    var groups = List[BaseStatistics]()
    var g = 0
    while g < len(vectors):
        var merged = vectors[g]
        for v in range(g + 1, min(g + ROW_GROUP_VECTORS, len(vectors))):
            merged.merge(vectors[v])
        groups.append(merged)
        g += ROW_GROUP_VECTORS
    return groups^


struct _VectorsJob(Job):
    """Statistics of a run of vectors of one storage's buffer."""

    var storage: Storage
    var layout: _Layout
    var first: Int
    var last: Int
    var result: List[BaseStatistics]

    def __init__(
        out self, storage: Storage, layout: _Layout, first: Int, last: Int
    ):
        self.storage = storage.copy()
        self.layout = layout.copy()
        self.first = first
        self.last = last
        self.result = List[BaseStatistics]()

    def run(mut self) raises:
        self.result = _vectors(self.storage, self.layout, self.first, self.last)


def _missing(series: Series) -> List[Storage]:
    """The arrays under `series` that can hold statistics and have none."""
    var out = List[Storage]()
    for part in series.chunks():
        var layout = _layout(part._data)
        if layout.kind < 0 or layout.rows <= layout.first:
            continue
        if _record_of(part._data):
            continue
        out.append(part._data.copy())
    return out^


def _compute(arrays: List[Storage]) raises:
    """Compute and publish statistics for `arrays`, a job per run of
    ROW_GROUP_VECTORS vectors."""
    var jobs = List[_VectorsJob]()
    var owners = List[Int]()
    for a in range(len(arrays)):
        var layout = _layout(arrays[a])
        var count = (
            layout.rows + STATISTICS_VECTOR_SIZE - 1
        ) // STATISTICS_VECTOR_SIZE
        var v = 0
        while v < count:
            var last = min(v + ROW_GROUP_VECTORS, count)
            jobs.append(_VectorsJob(arrays[a], layout, v, last))
            owners.append(a)
            v = last
    if len(jobs) > 1 and configured_workers() > 1:
        run_jobs(jobs)
    else:
        for j in range(len(jobs)):
            jobs[j].run()
    var k = 0
    for a in range(len(arrays)):
        var layout = _layout(arrays[a])
        var vectors = List[BaseStatistics]()
        while k < len(jobs) and owners[k] == a:
            vectors.extend(jobs[k].result.copy())
            k += 1
        var groups = _groups(vectors)
        _publish(
            arrays[a],
            _StatsRecord(
                layout.validity,
                _pin(arrays[a]),
                layout.first,
                layout.rows,
                vectors^,
                groups^,
            ),
        )


def ensure_statistics(series: Series) raises:
    """Compute the statistics of every buffer under `series` that has none."""
    _compute(_missing(series))


def compute_table_statistics(columns: List[Series]) raises:
    """Statistics for every column of a loaded table, all computed at once."""
    var arrays = List[Storage]()
    for column in columns:
        arrays.extend(_missing(column))
    _compute(arrays)


@fieldwise_init
struct WindowStatistics(Copyable, Movable):
    """The statistics a storage window reads: `vectors` are the record's,
    vector `v` covering window rows from `v * STATISTICS_VECTOR_SIZE -
    shift` (clipped to the window); `first`/`last` the vectors it touches."""

    var record: ArcPointer[_StatsRecord]
    var shift: Int
    var first: Int
    var last: Int


def window_statistics(part: Series) -> Optional[WindowStatistics]:
    """The statistics of an unchunked series' window, when its buffer has
    them for the validity bitmap it reads."""
    if part.is_chunked() or len(part) == 0:
        return None
    var layout = _layout(part._data)
    if layout.kind < 0:
        return None
    var record = _record_of(part._data)
    if not record:
        return None
    if record.value()[].validity != layout.validity:
        return None
    var offset = _window_offset(part._data)
    var end = offset + len(part)
    if offset < record.value()[].first or end > record.value()[].end:
        return None
    return WindowStatistics(
        record.value(),
        offset,
        offset // STATISTICS_VECTOR_SIZE,
        (end - 1) // STATISTICS_VECTOR_SIZE + 1,
    )


def _window_offset(storage: Storage) -> Int:
    comptime for d in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[d]
        if storage.isa[Column[Scalar[D]]]():
            return storage[Column[Scalar[D]]]._offset
    if storage.isa[Column[Int128]]():
        return storage[Column[Int128]]._offset
    if storage.isa[StringColumn]():
        return storage[StringColumn]._offset
    return 0


def series_statistics(series: Series) -> Optional[BaseStatistics]:
    """Statistics of every row of `series`, merged from its buffers'; None
    when a buffer has none."""
    var merged = Optional[BaseStatistics]()
    for part in series.chunks():
        if len(part) == 0:
            continue
        var found = window_statistics(part)
        if not found:
            return None
        ref w = found.value()
        ref record = w.record[]
        var stats = record.vectors[w.first]
        for v in range(w.first + 1, w.last):
            stats.merge(record.vectors[v])
        if merged:
            merged.value().merge(stats)
        else:
            merged = stats
    return merged^
