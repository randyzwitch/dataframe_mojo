"""Derive probe-side filters from a finished hash join build.

Build-key bounds and membership can reject probe rows that cannot match.
Apply these filters only where dropping unmatched probe rows preserves the
join's semantics. Each worker collects local bounds; merge them before
probing so no key from another build partition is excluded.
"""
from std.memory import ArcPointer, Pointer, bitcast

from dataframe.dtype import DataType
from dataframe.join_type import (
    JOIN_INNER,
    JOIN_RIGHT,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
)
from dataframe.joins.hash_table import JoinHashTable
from dataframe.kernels.runtime_filter import (
    KeyBitmap,
    KeyBloom,
    build_bloom,
    can_use_bloom_filter,
)
from dataframe.parallel import Job, partitions, run_jobs, worker_count
from dataframe.series import Series
from dataframe.storage.column import Column, _validity_at
from dataframe.storage.statistics import (
    COMPARE_EQUAL,
    COMPARE_GREATERTHANOREQUALTO,
    COMPARE_LESSTHANOREQUALTO,
)
from dataframe.table_filter import (
    DynamicFilterData,
    FilterConstants,
    TableFilterSet,
    column_domain,
    constant_in_domain,
)

# The most distinct build keys pushed to the probe scan as an IN list: each list
# value is one more comparison per zone map and per row, so a long list costs
# more than the rows it removes. An empirical choice.
comptime DYNAMIC_OR_FILTER_THRESHOLD = 50
# A bloom filter's selectivity is measured over its first 6 vectors; when it
# keeps half the rows or more it costs more than it saves, and it pauses for 10
# vectors (longer after each failed check) before measuring again, since the
# data can change along the scan. Empirical choices.
comptime BF_THRESHOLD = 0.5
comptime BF_CHECK_N = 6
comptime VECTOR_PAUSE = 10


@fieldwise_init
struct SelectivityStats(Copyable, Movable):
    """One thread's selectivity of an optional row filter, which turns the
    filter off while it removes too few rows to pay for itself."""

    var n_vectors_to_check: Int
    var threshold: Float64
    var accepted: Int
    var processed: Int
    var vectors: Int
    var active: Bool
    var pause_multiplier: Int

    @staticmethod
    def of(n_vectors_to_check: Int, threshold: Float64) -> Self:
        return Self(n_vectors_to_check, threshold, 0, 0, 0, True, 0)

    def update(mut self, accepted: Int, processed: Int):
        self.vectors += 1
        self.accepted += accepted
        self.processed += processed
        if self.vectors == max(self.pause_multiplier, 1) * VECTOR_PAUSE:
            self.vectors = 0
            self.accepted = 0
            self.processed = 0
            self.active = True
        elif self.active and self.vectors >= self.n_vectors_to_check:
            var share = 0.0
            if self.processed > 0:
                share = Float64(self.accepted) / Float64(self.processed)
            if share >= self.threshold:
                self.active = False
                self.pause_multiplier += 1
            else:
                self.pause_multiplier = 0


@fieldwise_init
struct JoinFilter(Copyable, Movable):
    """The filters one join key gives its probe side's source column."""

    var column: String
    # The join's step among the pipeline's steps.
    var step: Int
    # False when the build holds no non-null key: no row joins.
    var any: Bool
    # A single build value, compared with each row.
    var equal: Bool
    var value: Int64
    # Applied to rows, with its pause: a bloom filter, or the exact bitmap
    # of a perfect hash table's keys.
    var bloom: Optional[ArcPointer[KeyBloom]]
    var bitmap: Optional[ArcPointer[KeyBitmap]]


def _row_key(dtype: DataType) -> Bool:
    """Keys whose rows are compared as their physical integers."""
    var physical = dtype.physical()
    return physical == DataType.INT64 or physical == DataType.INT32


struct _BoundsJob(Job):
    """Collect minimum and maximum values from each build key. Merge worker-
    local bounds before using them to prune probe-side ranges; every build
    key must remain within the final bounds."""

    var key: Series
    var least: Int64
    var most: Int64
    var found: Bool

    def __init__(out self, var key: Series):
        self.key = key^
        self.least = Int64.MAX
        self.most = Int64.MIN
        self.found = False

    def run(mut self) raises:
        var physical = self.key.dtype().physical()
        for part in self.key.chunks():
            if physical == DataType.INT64:
                self._scan(part._data[Column[Int64]])
            else:
                self._scan(part._data[Column[Int32]])

    def _scan[D: DType](mut self, column: Column[Scalar[D]]):
        var values = column._ptr()
        var n = len(column)
        var least = self.least
        var most = self.most
        if column.null_count() == 0:
            for i in range(n):
                var v = Int64(values[unsafe_offset=i])
                least = min(least, v)
                most = max(most, v)
            self.found = self.found or n > 0
        else:
            for i in range(n):
                if column._valid(i):
                    var v = Int64(values[unsafe_offset=i])
                    least = min(least, v)
                    most = max(most, v)
                    self.found = True
        self.least = least
        self.most = most


def _bounds(key: Series) raises -> Optional[Tuple[Int64, Int64]]:
    """The least and greatest non-null value of an integer key: each worker's
    slice's, combined."""
    var n = len(key)
    var parts = worker_count(n)
    var bounds = partitions(n, parts, 1)
    var jobs = List[_BoundsJob](capacity=parts)
    for p in range(parts):
        if bounds[p] < bounds[p + 1]:
            jobs.append(
                _BoundsJob(key.slice(bounds[p], bounds[p + 1] - bounds[p]))
            )
    if len(jobs) == 1:
        jobs[0].run()
    elif len(jobs) > 1:
        run_jobs(jobs)
    var least = Int64.MAX
    var most = Int64.MIN
    var found = False
    for j in range(len(jobs)):
        if jobs[j].found:
            least = min(least, jobs[j].least)
            most = max(most, jobs[j].most)
            found = True
    if not found:
        return None
    return (least, most)


def _distinct(key: Series) -> List[Int64]:
    """The distinct non-null values of an integer key, ascending."""
    var values = List[Int64]()
    var physical = key.dtype().physical()
    for part in key.chunks():
        for i in range(len(part)):
            var v: Int64
            if physical == DataType.INT64:
                ref column = part._data[Column[Int64]]
                if not column._valid(i):
                    continue
                v = column._get(i)
            else:
                ref column = part._data[Column[Int32]]
                if not column._valid(i):
                    continue
                v = Int64(column._get(i))
            var at = 0
            while at < len(values) and values[at] < v:
                at += 1
            if at < len(values) and values[at] == v:
                continue
            values.insert(at, v)
    return values^


def _constant(dtype: DataType, value: Int64) raises -> Series:
    """A one-row constant of the key's type."""
    if dtype.physical() == DataType.INT64:
        return Series("c", Column[Int64]([value])).with_dtype(dtype)
    return Series("c", Column[Int32]([Int32(value)])).with_dtype(dtype)


def _dynamic(
    dtype: DataType, op: Int, values: List[Int64]
) raises -> Optional[ArcPointer[DynamicFilterData]]:
    """A set dynamic filter comparing with `values` (one, or an IN list's,
    compared for equality)."""
    var data = DynamicFilterData(dtype, op)
    var constants = FilterConstants.of(column_domain(dtype), True)
    for v in values:
        var converted = constant_in_domain(dtype, _constant(dtype, v))
        if not converted:
            return None
        constants.append(converted.value())
    data.constants = constants^
    data.initialized.store(1)
    return ArcPointer(data^)


struct _BuildKey(Movable):
    """A build key's values, read only when a filter needs them: a key
    column, or a key of a hash table (`JoinHashTable.key_column`)."""

    var series: Optional[Series]
    var table: Optional[JoinHashTable]
    var key: Int

    def __init__(out self, series: Series):
        self.series = series.copy()
        self.table = None
        self.key = 0

    def __init__(out self, table: JoinHashTable, key: Int):
        self.series = None
        self.table = table.copy()
        self.key = key

    def column(self) raises -> Series:
        if self.series:
            return self.series.value().copy()
        return self.table.value().key_column(self.key, "")


def finalize_filters(
    probe_column: String,
    probe_dtype: DataType,
    key: Series,
    step: Int,
    equalities: Int,
    build_count: Int,
    probe_rows: Int,
    build_filtered: Bool,
    mut scan: TableFilterSet,
) raises -> Optional[JoinFilter]:
    """The filters one build key column gives the probe side: statistics
    filters go into `scan` (one value as an equality, a few scattered
    values as an IN list, else the range between the least and greatest
    value), and the row filters the probe applies come back (None when
    the key's type takes no filter). No probe row outside them can match
    a build key, so dropping it early never changes the join."""
    if key.dtype() != probe_dtype or not _row_key(key.dtype()):
        return None
    return _finalize(
        probe_column,
        probe_dtype,
        _bounds(key),
        _BuildKey(key),
        step,
        equalities,
        build_count,
        probe_rows,
        build_filtered,
        scan,
    )


def finalize_table_filters(
    probe_column: String,
    probe_dtype: DataType,
    table: JoinHashTable,
    key: Int,
    step: Int,
    equalities: Int,
    probe_rows: Int,
    mut scan: TableFilterSet,
) raises -> Optional[JoinFilter]:
    """`finalize_filters` for key `key` of a finished build's hash table:
    its bounds are the ones the build recorded as its rows came in, its
    values are read from the table only for a short IN list, and its row
    filter is the one the table holds: the bloom filter it filled from its
    rows' hashes, or a perfect table's own bitmap of its keys."""
    var dtype = table.key_types[key]
    if dtype != probe_dtype or not _row_key(dtype):
        return None
    return _finalize(
        probe_column,
        probe_dtype,
        table.key_bounds(key),
        _BuildKey(table, key),
        step,
        equalities,
        table.inserted,
        probe_rows,
        table.build_filtered,
        scan,
    )


def _finalize(
    probe_column: String,
    probe_dtype: DataType,
    bounds: Optional[Tuple[Int64, Int64]],
    key: _BuildKey,
    step: Int,
    equalities: Int,
    build_count: Int,
    probe_rows: Int,
    build_filtered: Bool,
    mut scan: TableFilterSet,
) raises -> Optional[JoinFilter]:
    if not bounds:
        return JoinFilter(probe_column, step, False, False, 0, None, None)
    var low = bounds.value()[0]
    var high = bounds.value()[1]
    if low == high:
        var one: List[Int64] = [low]
        var equal = _dynamic(probe_dtype, COMPARE_EQUAL, one)
        if equal:
            scan.push_dynamic(probe_column, equal.value())
        return JoinFilter(probe_column, step, True, True, low, None, None)
    var bloom = Optional[ArcPointer[KeyBloom]]()
    var bitmap = Optional[ArcPointer[KeyBitmap]]()
    var perfect = Bool(key.table) and key.table.value().perfect
    if perfect:
        # A perfect hash table already holds an exact bitmap of its keys
        # over their range, which the probe side shares as its row filter:
        # no hashing, no false positives, under the same rule as a bloom
        # filter (a build small next to its probe). When the unique keys
        # fill their whole range it is the range itself, checked per row as
        # one comparison; the statistics filter alone only skips vectors.
        if can_use_bloom_filter(
            equalities, build_count, probe_rows, build_filtered
        ):
            bitmap = key.table.value().key_bits.copy()
    elif key.table:
        # A hash table fills its filter from its rows' hashes while it
        # finalizes, when the rule held then.
        bloom = key.table.value().bloom.copy()
    elif can_use_bloom_filter(
        equalities, build_count, probe_rows, build_filtered
    ):
        bloom = build_bloom(key.column(), build_count)
    var pushed_in = False
    if build_count > 1 and build_count <= DYNAMIC_OR_FILTER_THRESHOLD:
        var values = _distinct(key.column())
        # A dense run of values filters no more than its range does.
        if Int64(len(values)) - 1 != high - low:
            var listed = _dynamic(probe_dtype, COMPARE_EQUAL, values)
            if listed:
                scan.push_dynamic(probe_column, listed.value())
                pushed_in = True
    if bitmap:
        # The bitmap rules out a vector whose value range holds no key,
        # which covers the vectors outside the keys' range too.
        scan.push_key_bitmap(probe_column, bitmap.value())
    elif not pushed_in:
        var least: List[Int64] = [low]
        var most: List[Int64] = [high]
        var lower = _dynamic(probe_dtype, COMPARE_GREATERTHANOREQUALTO, least)
        var upper = _dynamic(probe_dtype, COMPARE_LESSTHANOREQUALTO, most)
        if lower and upper:
            scan.push_dynamic(probe_column, lower.value())
            scan.push_dynamic(probe_column, upper.value())
    return JoinFilter(probe_column, step, True, False, 0, bloom^, bitmap^)


def supports_join_filters(how: Int) -> Bool:
    """Whether a join among the pipeline's steps can filter its probe side: a
    probe row that matches nothing makes no output row."""
    return (
        how == JOIN_INNER
        or how == JOIN_SEMI
        or how == JOIN_RIGHT
        or how == JOIN_RIGHT_SEMI
    )


def keep_rows(
    filter: JoinFilter,
    key: Series,
    start: Int,
    count: Int,
    rows: Int,
    dense: Bool,
    all_valid: Bool,
) -> Int:
    """Narrow a selection to the rows of an unchunked integer key that
    `filter` keeps (its equality, its key bitmap or its bloom filter).

    The selection is `count` UInt32 row numbers of `key` at address `rows`;
    when `dense` it is rows [start, start + count) and is not read, only
    written. The kept rows are written in order to the selection's front
    (in place: a row is written no later than it is read), and their number
    returned. Each filter reads only the rows the ones before it kept, and
    no row is copied: what passes on is the selection. `all_valid`: the
    key has no null, found once for the key rather than per selection."""
    if key.dtype().physical() == DataType.INT64:
        return _keep_rows(
            filter,
            key._data[Column[Int64]],
            start,
            count,
            rows,
            dense,
            all_valid,
        )
    return _keep_rows(
        filter, key._data[Column[Int32]], start, count, rows, dense, all_valid
    )


def _keep_rows[
    D: DType
](
    filter: JoinFilter,
    column: Column[Scalar[D]],
    start: Int,
    count: Int,
    rows: Int,
    dense: Bool,
    all_valid: Bool,
) -> Int:
    var sel = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=rows)
    var values = column._ptr()
    var bits = column.unsafe_validity()
    var offset = column._offset
    var kept = 0
    if filter.bitmap:
        ref bitmap = filter.bitmap.value()[]
        if bitmap.dense:
            if dense and all_valid:
                return _keep_in_range[D, True, True](
                    bitmap, values, bits, offset, start, count, rows
                )
            if dense:
                return _keep_in_range[D, True, False](
                    bitmap, values, bits, offset, start, count, rows
                )
            if all_valid:
                return _keep_in_range[D, False, True](
                    bitmap, values, bits, offset, start, count, rows
                )
            return _keep_in_range[D, False, False](
                bitmap, values, bits, offset, start, count, rows
            )
        if dense and all_valid:
            return _keep_in_bitmap[D, True, True](
                bitmap, values, bits, offset, start, count, rows
            )
        if dense:
            return _keep_in_bitmap[D, True, False](
                bitmap, values, bits, offset, start, count, rows
            )
        if all_valid:
            return _keep_in_bitmap[D, False, True](
                bitmap, values, bits, offset, start, count, rows
            )
        return _keep_in_bitmap[D, False, False](
            bitmap, values, bits, offset, start, count, rows
        )
    for i in range(count):
        var row = start + i if dense else Int(sel[unsafe_offset=i])
        var ok = all_valid or _validity_at(bits, offset + row)
        if ok:
            var v = Int64(values[unsafe_offset=row])
            if filter.equal:
                ok = v == filter.value
            else:
                ok = filter.bloom.value()[].contains(v)
        sel[unsafe_offset=kept] = UInt32(row)
        kept += Int(ok)
    return kept


def _keep_in_bitmap[
    D: DType, dense: Bool, all_valid: Bool
](
    bitmap: KeyBitmap,
    keys: Pointer[Scalar[D], MutAnyOrigin],
    bits: Pointer[UInt8, MutAnyOrigin],
    offset: Int,
    start: Int,
    count: Int,
    rows: Int,
) -> Int:
    """`_keep_rows` for a key bitmap, with the selection's form and the
    key's validity fixed: a key's bit is its distance from the bitmap's
    base, and a key outside the range (below the base it wraps past the
    span) reads the last slot's bit, which is then masked off."""
    var sel = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=rows)
    var set = bitmap.bits.unsafe_ptr()
    var base = bitmap.base
    var span = bitmap.span
    var kept = 0
    for i in range(count):
        var row: Int
        comptime if dense:
            row = start + i
        else:
            row = Int(sel[unsafe_offset=i])
        var at = bitcast[DType.uint64](Int64(keys[unsafe_offset=row])) - base
        var slot = Int(min(at, span))
        var bit = (set[unsafe_offset=slot >> 3] >> UInt8(slot & 7)) & 1
        var ok = (at <= span) & (bit != 0)
        comptime if not all_valid:
            ok = ok & _validity_at(bits, offset + row)
        sel[unsafe_offset=kept] = UInt32(row)
        kept += Int(ok)
    return kept


def _keep_in_range[
    D: DType, dense: Bool, all_valid: Bool
](
    bitmap: KeyBitmap,
    keys: Pointer[Scalar[D], MutAnyOrigin],
    bits: Pointer[UInt8, MutAnyOrigin],
    offset: Int,
    start: Int,
    count: Int,
    rows: Int,
) -> Int:
    """`_keep_in_bitmap` for a bitmap with every bit set: a key is kept
    when its distance from the base is within the span."""
    var sel = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=rows)
    var base = bitmap.base
    var span = bitmap.span
    var kept = 0
    for i in range(count):
        var row: Int
        comptime if dense:
            row = start + i
        else:
            row = Int(sel[unsafe_offset=i])
        var at = bitcast[DType.uint64](Int64(keys[unsafe_offset=row])) - base
        var ok = at <= span
        comptime if not all_valid:
            ok = ok & _validity_at(bits, offset + row)
        sel[unsafe_offset=kept] = UInt32(row)
        kept += Int(ok)
    return kept


@fieldwise_init
struct PushedJoinFilter(Copyable, Movable):
    """A finished build's key for a scan below an aggregate on the join's probe
    side. A filter on a group key passes below the aggregate, since removing a
    group's rows removes only that group: `column` names the group key in the
    aggregate's input, which the input's pipeline follows to its source column,
    where `finalize_filters` makes the key's filters."""

    var column: String
    var key: Series
    # The join's equality conditions, and its build's rows with no null key and
    # whether the build side filters (for `can_use_bloom_filter`).
    var equalities: Int
    var build_count: Int
    var build_filtered: Bool


@fieldwise_init
struct PlannedJoinFilter(Copyable, Movable):
    """A join whose build's keys filter the input of an aggregate below
    its probe side, decided before anything runs: the join's plan node, which of its keys reach a group
    key (`positions`) and that key's name in the aggregate (`grouped`).
    The build publishes the filters when it finishes."""

    var join: Int
    var positions: List[Int]
    var grouped: List[String]
    var right_keys: List[String]
    var build_filtered: Bool


struct PushedJoinFilters(Movable):
    """The keys pushed into the input of the aggregate at plan node
    `node`, for its pipeline to filter its source by: `planned` are the
    joins they come from, `filters` the keys once those builds finished."""

    var node: Int
    var planned: List[PlannedJoinFilter]
    var filters: List[PushedJoinFilter]

    def __init__(
        out self,
        node: Int,
        var planned: List[PlannedJoinFilter],
        var filters: List[PushedJoinFilter],
    ):
        self.node = node
        self.planned = planned^
        self.filters = filters^
