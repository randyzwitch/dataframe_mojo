"""The scan order of a top-N's source (DuckDB 2.0's `RowGroupPruner`,
optimizer/row_group_pruner.cpp, and `RowGroupReorderer`,
storage/table/row_group_reorderer.cpp).

For `ORDER BY c LIMIT n [OFFSET m]` over a scanned column `c`, the source's
row groups are read in the order of their statistics on `c`: ascending by
each group's minimum, descending by its maximum (`OrderByStatistics`).
Row groups that cannot hold any of the first `n + m` rows are not read
(`AddRowGroups`), and leading row groups that lie wholly inside the offset
are dropped with the offset cut by their rows (`GetOffsetAfterPruning`).
The TopN's heap then fills from the best row groups first, so its boundary
is tight from the start.

A row group is `ROW_GROUP_VECTORS` vectors of `STATISTICS_VECTOR_SIZE`
rows (122,880, DuckDB's DEFAULT_ROW_GROUP_SIZE), counted from the start of
each buffer of the column, as its segment statistics are kept. Workers
take the ordered row groups one at a time, as DuckDB's
`RowGroupCollection::NextParallelScan` gives each thread the reorderer's
next row group whole.
"""

from std.collections import Optional

from dataframe.dtype import DataType
from dataframe.frame import DataFrame
from dataframe.kernels.segment_statistics import window_statistics
from dataframe.series import Series
from dataframe.storage.statistics import (
    BaseStatistics,
    FLOAT_STATS,
    NUMERIC_STATS,
    ROW_GROUP_VECTORS,
    STATISTICS_VECTOR_SIZE,
    STRING_STATS,
)

comptime ROW_GROUP_ROWS = ROW_GROUP_VECTORS * STATISTICS_VECTOR_SIZE


@fieldwise_init
struct _StatValue(Copyable, Movable):
    """`RetrieveStat`'s value: none, an integer (numeric statistics), a
    float, or a string's stored bytes with whether they are the whole
    value (a truncated maximum is only a prefix of it)."""

    var kind: Int
    var int_value: Int128
    var float_value: Float64
    var bytes: List[UInt8]
    var exact: Bool

    @staticmethod
    def none() -> Self:
        return Self(0, 0, 0.0, List[UInt8](), True)

    def is_null(self) -> Bool:
        return self.kind == 0


comptime _INT = 1
comptime _FLOAT = 2
comptime _STRING = 3


def _retrieve(stats: BaseStatistics, use_max: Bool) -> _StatValue:
    """`RowGroupReorderer::RetrieveStat`: the minimum or maximum, or none
    when the statistics hold no min/max."""
    if not stats.has_min_max:
        return _StatValue.none()
    if stats.stats_type == NUMERIC_STATS:
        return _StatValue(
            _INT,
            stats.int_max if use_max else stats.int_min,
            0.0,
            List[UInt8](),
            True,
        )
    if stats.stats_type == FLOAT_STATS:
        return _StatValue(
            _FLOAT,
            0,
            stats.float_max if use_max else stats.float_min,
            List[UInt8](),
            True,
        )
    if stats.stats_type == STRING_STATS:
        var bytes = List[UInt8]()
        var size = stats.str_max_size if use_max else stats.str_min_size
        for i in range(size):
            bytes.append(stats.str_max[i] if use_max else stats.str_min[i])
        return _StatValue(
            _STRING,
            0,
            0.0,
            bytes^,
            stats.max_exact if use_max else stats.min_exact,
        )
    return _StatValue.none()


def _bytes_compare(a: List[UInt8], b: List[UInt8]) -> Int:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return -1 if a[i] < b[i] else 1
    if len(a) == len(b):
        return 0
    return -1 if len(a) < len(b) else 1


def _compare(a: _StatValue, b: _StatValue) -> Int:
    """The order of two values of one kind (Value's comparison)."""
    if a.kind == _INT:
        return 0 if a.int_value == b.int_value else (
            -1 if a.int_value < b.int_value else 1
        )
    if a.kind == _FLOAT:
        return 0 if a.float_value == b.float_value else (
            -1 if a.float_value < b.float_value else 1
        )
    return _bytes_compare(a.bytes, b.bytes)


def _differs_below(upper: _StatValue, lower: _StatValue) -> Bool:
    """Whether every string with the truncated prefix `upper` is below
    `lower`: they differ at a byte both hold, `upper`'s the smaller."""
    for i in range(min(len(upper.bytes), len(lower.bytes))):
        if upper.bytes[i] != lower.bytes[i]:
            return upper.bytes[i] < lower.bytes[i]
    return False


def _beyond(current: _StatValue, boundary: _StatValue, use_max: Bool) -> Bool:
    """`CompareValues`: whether the row group keyed `current` comes wholly
    after the row group whose opposite statistic is `boundary` (MIN:
    current > boundary; MAX: current < boundary). A truncated string
    maximum stands for every string it starts, so it compares only where
    it differs from the other value."""
    if current.is_null() or boundary.is_null():
        return False
    if use_max:
        # current: a row group's maximum; boundary: another's minimum.
        if current.kind == _STRING and not current.exact:
            return _differs_below(current, boundary)
        return _compare(current, boundary) < 0
    # current: a row group's minimum; boundary: another's maximum.
    if boundary.kind == _STRING and not boundary.exact:
        return _differs_below(boundary, current)
    return _compare(current, boundary) > 0


def _null_only(stats: BaseStatistics) -> Bool:
    return not stats.can_have_no_null()


@fieldwise_init
struct _RowGroup(Copyable, Movable):
    var start: Int
    var count: Int
    var stats: Optional[BaseStatistics]


def _group_statistics(window: Series) -> Optional[BaseStatistics]:
    """A row group's statistics, merged from its vectors'. Float bounds
    leave NaN out (`BaseStatistics.has_nan`), and NaN sorts above every
    number, so a row group holding NaN has no min/max to order by."""
    var found = window_statistics(window)
    if not found:
        return None
    ref w = found.value()
    ref record = w.record[]
    var stats = record.vectors[w.first]
    for v in range(w.first + 1, w.last):
        stats.merge(record.vectors[v])
    if stats.stats_type == FLOAT_STATS and stats.has_nan:
        stats.has_min_max = False
    return stats


def _row_groups(column: Series) raises -> List[_RowGroup]:
    """The column's row groups with their statistics (None when a buffer
    has none): ROW_GROUP_ROWS rows from the start of each chunk."""
    var groups = List[_RowGroup]()
    var start = 0
    for part in column.chunks():
        var length = len(part)
        var at = 0
        while at < length:
            var count = min(ROW_GROUP_ROWS, length - at)
            groups.append(
                _RowGroup(
                    start + at,
                    count,
                    _group_statistics(part.slice(at, count)),
                )
            )
            at += count
        start += length
    return groups^


def _minimum_qualifying(group: _RowGroup, is_string: Bool) -> Int:
    """`GetMinimumQualifyingTupleCount`: the rows a row group surely holds
    at its ordering statistic or better."""
    ref stats = group.stats.value()
    if not stats.can_have_null():
        return group.count
    if is_string:
        # Truncated string statistics may not be values of the group.
        return 0
    if not stats.has_min_max:
        return 0
    if stats.stats_type == FLOAT_STATS:
        return 1 if stats.float_min == stats.float_max else 2
    if stats.int_min == stats.int_max:
        return 1
    return 2


def _sorted_by_key(indexes: List[Int], keys: List[_StatValue]) -> List[Int]:
    """`indexes` ordered by their keys, equal keys in insertion order (a
    multimap's)."""
    var out = indexes.copy()
    for i in range(1, len(out)):
        var j = i
        var moving = out[i]
        while j > 0 and _compare(keys[moving], keys[out[j - 1]]) < 0:
            out[j] = out[j - 1]
            j -= 1
        out[j] = moving
    return out^


def _add_row_groups(
    seq: List[Int],
    groups: List[_RowGroup],
    keys: List[_StatValue],
    row_limit: Int,
    is_string: Bool,
    use_max: Bool,
    mut ordered: List[Int],
) -> Bool:
    """`AddRowGroups`: the row groups in `seq` until the earlier ones hold
    `row_limit` rows that every later one comes after; whether any were
    left out."""
    var last = 0
    var boundary = _retrieve(groups[seq[0]].stats.value(), not use_max)
    var qualifying = 0
    var qualify_later = 0
    var last_sum = _minimum_qualifying(groups[seq[0]], is_string)
    for i in range(len(seq)):
        ref key = keys[seq[i]]
        while last != i:
            if not _beyond(key, boundary, use_max):
                if _compare(key, keys[seq[i - 1]]) != 0:
                    # Row groups overlap: one more qualifying row only.
                    qualifying += qualify_later
                    qualify_later = 0
                    qualifying += 1
                else:
                    # The same order value: a qualifying row later.
                    qualify_later += 1
                break
            # No overlap: the earlier row group's rows qualify.
            qualifying = last_sum
            last += 1
            last_sum += _minimum_qualifying(groups[seq[last]], is_string)
            boundary = _retrieve(groups[seq[last]].stats.value(), not use_max)
        if qualifying >= row_limit:
            return True
        ordered.append(seq[i])
    return False


@fieldwise_init
struct _OffsetPruning(Copyable, Movable):
    """`OffsetPruningResult`."""

    var offset_remainder: Int
    var pruned_row_group_count: Int
    var leading_null_group_offset: Int


def _offset_prunable(
    seq: List[Int],
    groups: List[_RowGroup],
    keys: List[_StatValue],
    use_max: Bool,
    row_offset: Int,
) -> _OffsetPruning:
    """`FindOffsetPrunableChunks`: the leading row groups that every later
    one comes after and that lie inside the offset."""
    if len(seq) == 0:
        return _OffsetPruning(row_offset, 0, 0)
    var last = 0
    var boundary = _retrieve(groups[seq[0]].stats.value(), not use_max)
    var seen = 0
    var remainder = row_offset
    var pruned = 0
    for i in range(len(seq)):
        seen += groups[seq[i]].count
        while last != i:
            if not _beyond(keys[seq[i]], boundary, use_max):
                break
            if groups[seq[last]].stats.value().can_have_null():
                # Only a prefix of row groups with no nulls is pruned.
                return _OffsetPruning(remainder, pruned, 0)
            pruned += 1
            remainder -= groups[seq[last]].count
            last += 1
            boundary = _retrieve(groups[seq[last]].stats.value(), not use_max)
        if seen > row_offset:
            break
    return _OffsetPruning(remainder, pruned, 0)


def _exact(stats: BaseStatistics) -> Bool:
    """`MinMaxIsExact`: string statistics may hold a truncated prefix.
    Numeric and float min/max are values of the rows (a vector holding NaN
    has none)."""
    if stats.stats_type == STRING_STATS:
        return stats.min_exact and stats.max_exact
    return True


def _offset_after_pruning(
    groups: List[_RowGroup],
    use_max: Bool,
    descending: Bool,
    nulls_last: Bool,
    row_offset: Int,
) -> _OffsetPruning:
    """`RowGroupReorderer::GetOffsetAfterPruning`."""
    var remainder = row_offset
    var leading = 0
    var maybe_null = False
    var entries = List[Int]()
    var keys = List[_StatValue]()
    for g in range(len(groups)):
        keys.append(_StatValue.none())
        if not groups[g].stats:
            return _OffsetPruning(remainder, 0, leading)
        ref stats = groups[g].stats.value()
        if not _exact(stats) and not _null_only(stats):
            return _OffsetPruning(row_offset, 0, 0)
        if not nulls_last and _null_only(stats):
            if remainder < groups[g].count:
                return _OffsetPruning(remainder, 0, leading)
            remainder -= groups[g].count
            leading += 1
            continue
        var value = _retrieve(stats, use_max)
        if value.is_null():
            if nulls_last and _null_only(stats):
                continue
            if not nulls_last:
                maybe_null = True
                continue
            return _OffsetPruning(remainder, 0, leading)
        if not nulls_last and stats.can_have_null():
            maybe_null = True
            continue
        keys[g] = value^
        entries.append(g)
    if not nulls_last and maybe_null:
        return _OffsetPruning(remainder, 0, leading)
    var seq = _sorted_by_key(entries, keys)
    if descending:
        seq.reverse()
    var result = _offset_prunable(seq, groups, keys, use_max, remainder)
    result.leading_null_group_offset = leading
    return result^


@fieldwise_init
struct RowGroupOrder(Copyable, Movable):
    """The scan order: `ranges` are (offset, length) pairs of the row
    groups to read, in order; `remaining_offset` is the limit's offset after the
    row groups it covered were dropped."""

    var ranges: List[Int]
    var remaining_offset: Int


def orderable(dtype: DataType) -> Bool:
    """`FindLogicalOrder`: numeric, temporal and VARCHAR sort columns."""
    return (
        dtype == DataType.STRING
        or dtype.is_integer()
        or dtype.is_decimal()
        or dtype.is_float()
        or dtype.is_date()
        or dtype.is_datetime()
        or dtype.is_duration()
        or dtype.is_time()
    )


def row_group_order(
    column: Series,
    descending: Bool,
    nulls_last: Bool,
    limit: Int,
    offset: Int,
    prune: Bool,
) raises -> Optional[RowGroupOrder]:
    """The row groups of a scan whose first sort column is `column`, for
    `limit` rows after `offset` (`CreateRowGroupReordererOptions`, then
    `GetRootSegment`). Without `prune` (a filter between the scan and the
    order, or no limit) they are only ordered. None when the column has
    no row group statistics to order by."""
    var groups = _row_groups(column)
    var any_stats = False
    for g in groups:
        if g.stats:
            any_stats = True
            break
    if not any_stats or len(groups) < 2:
        return None
    var use_max = descending
    var is_string = column.dtype() == DataType.STRING
    var can_prune = prune and limit >= 0
    var row_limit = limit + max(offset, 0) if can_prune else -1
    var row_group_offset = 0
    var leading_null_groups = 0
    var remainder = offset
    if can_prune and offset > 0:
        var found = _offset_after_pruning(
            groups, use_max, descending, nulls_last, offset
        )
        if (
            found.pruned_row_group_count > 0
            or found.leading_null_group_offset > 0
        ):
            remainder = found.offset_remainder
            row_limit = limit + remainder
            row_group_offset = found.pruned_row_group_count
            leading_null_groups = found.leading_null_group_offset

    var null_only = List[Int]()
    var ambiguous = List[Int]()
    var mapped = List[Int]()
    var keys = List[_StatValue]()
    for g in range(len(groups)):
        keys.append(_StatValue.none())
        if not groups[g].stats:
            ambiguous.append(g)
            continue
        ref stats = groups[g].stats.value()
        if _null_only(stats):
            null_only.append(g)
            continue
        var value = _retrieve(stats, use_max)
        if value.is_null() or (not nulls_last and stats.can_have_null()):
            ambiguous.append(g)
            continue
        keys[g] = value^
        mapped.append(g)
    var seq = _sorted_by_key(mapped, keys)
    if descending:
        seq.reverse()
    var rest = List[Int]()
    for i in range(min(row_group_offset, len(seq)), len(seq)):
        rest.append(seq[i])

    var ordered = List[Int]()
    if not nulls_last:
        for i in range(leading_null_groups, len(null_only)):
            ordered.append(null_only[i])
        ordered.extend(ambiguous.copy())
        if len(rest) > 0:
            if row_limit >= 0:
                _ = _add_row_groups(
                    rest, groups, keys, row_limit, is_string, use_max, ordered
                )
            else:
                ordered.extend(rest.copy())
    else:
        var pruned = False
        if len(rest) > 0:
            if row_limit >= 0:
                pruned = _add_row_groups(
                    rest, groups, keys, row_limit, is_string, use_max, ordered
                )
            else:
                ordered.extend(rest.copy())
        ordered.extend(ambiguous.copy())
        # An ascending limit met by non-null row groups leaves no room
        # for null-only ones.
        if not pruned or descending:
            ordered.extend(null_only.copy())
    var ranges = List[Int](capacity=2 * len(ordered))
    for g in ordered:
        ranges.append(groups[g].start)
        ranges.append(groups[g].count)
    return RowGroupOrder(ranges^, remainder)
