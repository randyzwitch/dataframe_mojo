"""Derive probe-side filters from a finished hash join build.

Build-key bounds and membership can reject probe rows that cannot match.
Apply these filters only where dropping unmatched probe rows preserves the
join's semantics. Each worker collects local bounds; merge them before
probing so no key from another build partition is excluded.
"""
from std.memory import ArcPointer, Pointer

from dataframe.dtype import DataType
from dataframe.join_type import (
    JOIN_INNER,
    JOIN_RIGHT,
    JOIN_RIGHT_SEMI,
    JOIN_SEMI,
)
from dataframe.kernels.runtime_filter import KeyBloom, build_bloom
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

# DynamicOrFilterThresholdSetting's default.
comptime DYNAMIC_OR_FILTER_THRESHOLD = 50
# physical_hash_join.cpp CanUseBloomFilter.
comptime BUILD_TO_PROBE_RATIO_THRESHOLD = 1.0
comptime NON_FILTERING_RATIO_THRESHOLD = 0.1
comptime NON_FILTERING_BUILD_SIDE_THRESHOLD = 4194304
# GetThresholdAndVectorsToCheck (BF) and SelectivityStats::Update.
comptime BF_THRESHOLD = 0.5
comptime BF_CHECK_N = 6
comptime VECTOR_PAUSE = 10


@fieldwise_init
struct SelectivityStats(Copyable, Movable):
    """One thread's selectivity of an optional row filter
    (`SelectivityOptionalFilterState::SelectivityStats`)."""

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
    # Applied to rows, with its pause.
    var bloom: Optional[ArcPointer[KeyBloom]]


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
    """The least and greatest non-null value of an integer key: each
    worker's slice's, combined (JoinFilterGlobalState's Combine)."""
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


def can_use_bloom_filter(
    equalities: Int, build_count: Int, probe_rows: Int, build_filtered: Bool
) -> Bool:
    """CanUseBloomFilter on a join's sizes."""
    if equalities != 1 or build_count == 0 or probe_rows <= 0:
        return False
    var ratio = Float64(build_count) / Float64(probe_rows)
    if ratio > BUILD_TO_PROBE_RATIO_THRESHOLD:
        return False
    if (
        not build_filtered
        and ratio > NON_FILTERING_RATIO_THRESHOLD
        and build_count > NON_FILTERING_BUILD_SIDE_THRESHOLD
    ):
        return False
    return True


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
    """JoinFilterPushdownInfo::FinalizeFilters for one key: its statistics
    filters go into `scan`, and the row filters it applies come back (None
    when the key's type takes no filter)."""
    if key.dtype() != probe_dtype or not _row_key(key.dtype()):
        return None
    var bounds = _bounds(key)
    if not bounds:
        return JoinFilter(probe_column, step, False, False, 0, None)
    var low = bounds.value()[0]
    var high = bounds.value()[1]
    if low == high:
        var one: List[Int64] = [low]
        var equal = _dynamic(probe_dtype, COMPARE_EQUAL, one)
        if equal:
            scan.push_dynamic(probe_column, equal.value())
        return JoinFilter(probe_column, step, True, True, low, None)
    var pushed_in = False
    if build_count > 1 and build_count <= DYNAMIC_OR_FILTER_THRESHOLD:
        var values = _distinct(key)
        # Only when the values are not dense (FilterCombiner::IsDenseRange).
        if Int64(len(values)) - 1 != high - low:
            var listed = _dynamic(probe_dtype, COMPARE_EQUAL, values)
            if listed:
                scan.push_dynamic(probe_column, listed.value())
                pushed_in = True
    if not pushed_in:
        var least: List[Int64] = [low]
        var most: List[Int64] = [high]
        var lower = _dynamic(probe_dtype, COMPARE_GREATERTHANOREQUALTO, least)
        var upper = _dynamic(probe_dtype, COMPARE_LESSTHANOREQUALTO, most)
        if lower and upper:
            scan.push_dynamic(probe_column, lower.value())
            scan.push_dynamic(probe_column, upper.value())
    var bloom = Optional[ArcPointer[KeyBloom]]()
    if can_use_bloom_filter(
        equalities, build_count, probe_rows, build_filtered
    ):
        bloom = build_bloom(key, build_count)
    return JoinFilter(probe_column, step, True, False, 0, bloom^)


def supports_join_filters(how: Int) -> Bool:
    """JoinFilterPushdownUtil::JoinTypeIsSupported among the pipeline's
    join steps: a probe row that matches nothing makes no output row."""
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
    end: Int,
    mut keep: List[UInt8],
) -> Tuple[Int, Int]:
    """AND rows [start, end) of an unchunked integer key that `filter` keeps
    into `keep` (indexed from `start`): its equality, or its bloom filter.
    Returns the rows looked at (those `keep` still held) and those kept."""
    var at = keep.unsafe_ptr()
    var looked = 0
    var kept = 0
    var physical = key.dtype().physical()
    var offset = 0
    var all_valid = key.null_count() == 0
    if physical == DataType.INT64:
        ref column = key._data[Column[Int64]]
        var bits = column.unsafe_validity()
        if not all_valid:
            offset = column._offset
        var values = column._ptr()
        for i in range(start, end):
            var k = i - start
            if at[unsafe_offset=k] == 0:
                continue
            looked += 1
            var ok = all_valid or _validity_at(bits, offset + i)
            if ok:
                var v = values[unsafe_offset=i]
                if filter.equal:
                    ok = v == filter.value
                else:
                    ok = filter.bloom.value()[].contains(v)
            at[unsafe_offset=k] = UInt8(1) if ok else UInt8(0)
            kept += Int(ok)
    else:
        ref column = key._data[Column[Int32]]
        var bits = column.unsafe_validity()
        if not all_valid:
            offset = column._offset
        var values = column._ptr()
        for i in range(start, end):
            var k = i - start
            if at[unsafe_offset=k] == 0:
                continue
            looked += 1
            var ok = all_valid or _validity_at(bits, offset + i)
            if ok:
                var v = Int64(values[unsafe_offset=i])
                if filter.equal:
                    ok = v == filter.value
                else:
                    ok = filter.bloom.value()[].contains(v)
            at[unsafe_offset=k] = UInt8(1) if ok else UInt8(0)
            kept += Int(ok)
    return (looked, kept)


@fieldwise_init
struct PushedJoinFilter(Copyable, Movable):
    """A finished build's key for a scan below an aggregate on the join's
    probe side (`JoinFilterPushdownOptimizer::GetPushdownFilterTargets`,
    the LOGICAL_AGGREGATE_AND_GROUP_BY case): `column` names the group key
    in the aggregate's input, which the input's pipeline follows to its
    source column, where `finalize_filters` makes the key's filters."""

    var column: String
    var key: Series
    # The join's equality conditions, and its build's rows with no null
    # key and whether the build side filters (for CanUseBloomFilter).
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
