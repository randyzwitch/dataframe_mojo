"""An eager CPU dataframe with runtime schema and positional row semantics."""
from .time_windows import (
    positive_period,
    closed_sides,
    negate_period,
    shift_time,
    truncate_time,
    check_time_index,
    sort_time_groups,
    search_time,
    rolling_bounds,
)
from .temporal import zone_of, parse_interval
from .window import partitions as window_partitions
from .dtype import DataType, NUMERIC_DTYPES
from .decimal import pow10
from std.collections import Dict, Optional
from std.memory import ArcPointer, Pointer
from dataframe.storage.boolean import BoolColumn, both_true, true_count
from dataframe.storage.column import Column
from dataframe.storage.string import StringColumn
from .series import Series, sort_indices
from dataframe.kernels.distinct_statistics import TableStatistics
from dataframe.kernels.fused_compare import SelectionTree, fused_mask
from dataframe.fused_compare import fused_comparisons, selection_tree
from dataframe.sorted_range import sorted_window
from .expr import (
    Expr,
    MODE,
    VALUE_COUNTS,
    conjuncts,
    MIN,
    MAX,
    col,
    lit,
    null,
    when,
    COL,
    SELECTOR,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_STRING,
    LIT_NULL,
    GT,
    LT,
    GE,
    LE,
    EQ,
    NE,
    UNTYPED,
)
from .binding import bind, BoundExpr, ROWS, AGGREGATE
from dataframe.execution import WindowLayouts, evaluate
from dataframe.kernels.sampling import sample_size, sample_indices
from dataframe.kernels.mask_filter import filter_columns
from dataframe.kernels.gather import (
    _RechunkJob,
    take_parallel,
    take_sorted_chunked,
    true_rows,
    float_compare_rows,
    can_filter_aligned_chunks,
)
from .parallel import (
    Job,
    run_jobs,
    worker_count,
)
from dataframe.kernels.partition import (
    encode_partitioned,
    low_cardinality,
)
from dataframe.kernels.join_hash import (
    PreparedHashIndex,
)
from .join_type import (
    JOIN_CROSS,
    JOIN_INNER,
    join_code,
    join_type,
)
from dataframe.storage.nested import ListColumn, StructColumn
from .trace import trace_path
from dataframe.kernels.row_encode import encodable, encode_sort_keys
from .categorical import (
    recode,
    sort_ranks,
    string_sort_ranks,
    unify,
    union_of,
)
from dataframe.kernels.packed_sort import (
    chunked_top_rows,
    packed_arg_sort,
    packed_top_rows,
)

# A head of a sort at least 1/this of the rows sorts fully (#332).
comptime _TOP_K_FRACTION = 10
from .value import AnyValue
from .frame_iter import RowIterator, ColumnIterator, SliceIterator
from .frame_size import frame_size
from .asof import asof_rows
from dataframe.kernels.hashing import (
    RowKeys,
    encode_rows,
)
from .groups import GroupIndices
from dataframe.kernels.top_k import top_k_mask
from dataframe.kernels.row_sort import row_arg_sort, row_top_rows
from dataframe.kernels.binary import validity
from .selectors import expand, expand_all
from .lazy import LazyFrame, _concat_plans
from .display import render_frame, render_glimpse
from dataframe.aggregation.keys import (
    _KEY_SEP,
    _KEY_VALID,
    _expand_struct_keys,
    _group_keys,
    _key_parts,
    _key_prefix,
    _pack_struct_keys,
)
from dataframe.aggregation.expressions import (
    _bind_all,
    _column_names,
    _row_local,
    _stream_reductions,
)
from dataframe.aggregation.group_by import (
    GroupBy,
    GroupingSets,
    _BucketJob,
    _EncodeJob,
    _HashedBucketJob,
    _PARTITION_ROWS_PER_WORKER,
    _PERFECT_WINDOW,
    _PerfectAggJob,
    _RangeAggJob,
    _encode_parallel,
    _reduce_column,
    group_by_workers,
    range_states,
)
from dataframe.aggregation.state import (
    _FinishJob,
    _KeyIndex,
    _ReductionPlan,
    _StreamMergeJob,
    _StreamReduction,
    _encode_in_order,
    _equality_words,
    _finish_parts,
    _merge_ascending_runs,
    _reduction_batch,
    _table_input,
    _word_hash,
    table_frame,
    table_inputs,
    table_specs,
    table_supported,
)
from dataframe.joins.eager import (
    _left_builds,
    _smaller_build_join_rows,
    cross_join_frames,
    join_frames,
)
from dataframe.joins.rows import (
    _CSRCountJob,
    _CSRFillJob,
    _CSRScatterJob,
    _CSR_MIN_PARALLEL_GROUPS,
    _CSR_PARALLEL_WORKING_BYTES,
    _JoinCountJob,
    _JoinFillJob,
    _OrderCountJob,
    _OrderFillJob,
    _OrderScatterJob,
    _UnmatchedRightCountJob,
    _UnmatchedRightFillJob,
    _group_index,
    _group_rows,
    _id_range_bucket,
    _id_range_first,
    _parallel_csr_fits,
    _parallel_full_join_rows,
    _parallel_group_rows,
    _parallel_join_rows,
    _parallel_left_major,
)
from dataframe.joins.range import (
    _JoinRangeProbeJob,
    _PROGRESSION_PARALLEL_KEY_BYTES,
    _ProgressionProbeJob,
    _RANGE_MEMBERSHIP_MAX_BYTES,
    _RANGE_PARALLEL_MIN_HEAD_BYTES,
    _RANGE_PARALLEL_WORKING_BYTES,
    _RANGE_SMALL_TABLE_BYTES,
    _RANGE_TABLE_BYTES_PER_KEY,
    _RANGE_TABLE_MAX_BYTES,
    _RangeIndexBuildJob,
    _RangeJoinProbeJob,
    _RangeMembershipDomain,
    _RangeMembershipJob,
    _bounded_int64_join_ids,
    _bounded_int64_join_rows,
    _dense_right_int64_rows,
    _int64_distance,
    _join_range_possible,
    _join_range_rows,
    _parallel_progression_rows,
    _parallel_range_build_workers,
    _range_int64_membership_chunks,
    _range_int64_membership_domain,
    _range_int64_membership_rows,
    _range_join_span_fits,
    _range_join_table_capacity,
)
from dataframe.joins.keys import (
    _joint_key_ids,
)


@fieldwise_init
struct Field(Copyable):
    """One schema entry: a column name and its dtype."""

    var name: String
    var dtype: DataType


# Up to this many columns, names are checked pair by pair: no dictionary is
# built for a frame of a few columns, and frames are built for every step.
comptime _PAIRWISE_NAMES = 32


def _check_unique_names(columns: List[Series]) raises:
    """Raise on a repeated column name."""
    var n = len(columns)
    if n <= _PAIRWISE_NAMES:
        for i in range(1, n):
            ref name = columns[i]._name
            for j in range(i):
                if columns[j]._name == name:
                    raise Error("Duplicate column name: " + name)
        return
    var names = Dict[String, Bool]()
    for column in columns:
        var name = column.name()
        if name in names:
            raise Error("Duplicate column name: " + name)
        names[name] = True


struct DataFrame(Copyable, Sized, Writable):
    """Own equal-length, uniquely named columns; transformations copy storage."""

    var _columns: List[Series]
    var _height: Int
    # A loaded table's distinct-count statistics, shared by its copies
    # (`read_parquet`, `read_csv`); none for a frame made any other way.
    var _statistics: Optional[ArcPointer[TableStatistics]]

    def __init__(
        out self, var columns: List[Series], *, height: Int = -1
    ) raises:
        if height < -1:
            raise Error("Height must be nonnegative or inferred")
        var inferred = 0
        if len(columns) > 0:
            inferred = len(columns[0])
        elif height >= 0:
            inferred = height
        if height >= 0 and inferred != height:
            raise Error("Explicit height does not match columns")
        for column in columns:
            if len(column) != inferred:
                raise Error("DataFrame columns must have equal lengths")
        _check_unique_names(columns)
        self._columns = columns^
        self._height = inferred
        self._statistics = None

    def _with_statistics(var self) -> Self:
        """Mark this as a loaded table so column distinct counts are
        estimated on first request and shared across copies."""
        self._statistics = Optional(ArcPointer(TableStatistics()))
        return self^

    def _distinct_count(self, name: String) raises -> Int:
        """Column `name`'s estimated distinct count from the table's
        statistics, or -1 when this frame has none or no such column."""
        if not self._statistics:
            return -1
        for column in self._columns:
            if column.name() == name:
                var box = self._statistics.value()
                return box[].distinct(column)
        return -1

    def _sampled(self, names: List[String], runs: Int, run: Int) raises -> Self:
        """The named columns' rows at `runs` evenly spaced runs of `run`
        rows. A loaded table keeps each column's merged runs with its
        statistics and reads them back; any other frame merges them
        again."""
        var columns = List[Series](capacity=len(names))
        for name in names:
            var column = self[name]
            if self._statistics:
                var box = self._statistics.value()
                columns.append(box[].sample(column, runs, run))
            else:
                var parts = List[Series](capacity=runs)
                for k in range(runs):
                    parts.append(column.slice(k * (self._height // runs), run))
                columns.append(Series._from_chunks(parts).rechunk())
        return Self(columns^, height=runs * run)

    def rechunk(self) raises -> Self:
        """Return one contiguous array per column, copying only chunked
        data; chunked columns merge in parallel, one job each (a 159K-row
        join build with five string columns: 1 ms serial)."""
        var jobs = List[_RechunkJob]()
        var at = List[Int]()
        for c in range(self.width()):
            if self._columns[c].is_chunked():
                jobs.append(_RechunkJob(self._columns[c]))
                at.append(c)
        if len(jobs) > 1:
            run_jobs(jobs)
        elif len(jobs) == 1:
            jobs[0].run()
        var columns = List[Series](capacity=self.width())
        var next = 0
        for c in range(self.width()):
            if next < len(at) and at[next] == c:
                columns.append(jobs[next].result.copy())
                next += 1
            else:
                columns.append(self._columns[c].copy())
        return Self(columns^, height=self._height)

    def height(self) -> Int:
        return self._height

    def write_to(self, mut writer: Some[Writer]):
        writer.write(render_frame(self._columns, self._height, 10, 12, 32))

    def to_string(
        self,
        *,
        max_rows: Int = 10,
        max_columns: Int = 12,
        max_string_length: Int = 32,
    ) -> String:
        """Render a bounded table. Negative limits mean unlimited."""
        return render_frame(
            self._columns,
            self._height,
            max_rows,
            max_columns,
            max_string_length,
        )

    def glimpse(
        self, *, max_width: Int = 100, max_string_length: Int = 32
    ) -> String:
        """Transposed summary: one line per column with leading values."""
        return render_glimpse(
            self._columns, self._height, max_width, max_string_length
        )

    def width(self) -> Int:
        return len(self._columns)

    def schema(self) -> List[Field]:
        var fields = List[Field](capacity=self.width())
        for column in self._columns:
            fields.append(Field(column.name(), column.dtype()))
        return fields^

    def shape(self) -> Tuple[Int, Int]:
        return (self._height, self.width())

    def __len__(self) -> Int:
        return self._height

    def is_empty(self) -> Bool:
        return self._height == 0

    def columns(self) -> List[String]:
        var names = List[String](capacity=self.width())
        for column in self._columns:
            names.append(column.name())
        return names^

    def dtypes(self) -> List[DataType]:
        var types = List[DataType](capacity=self.width())
        for column in self._columns:
            types.append(column.dtype())
        return types^

    def _index(self, name: String) raises -> Int:
        for i in range(self.width()):
            if self._columns[i].name() == name:
                return i
        raise Error("Unknown column: " + name)

    def column(self, name: String) raises -> Series:
        """Return an owned copy. Column lookup is linear in the schema width."""
        return self._columns[self._index(name)].copy()

    def lazy(self) -> LazyFrame:
        """Start a lazy query over this frame; see LazyFrame."""
        return LazyFrame(self)

    def get_column(self, name: String) raises -> Series:
        return self.column(name)

    def __getitem__(self, name: String) raises -> Series:
        return self.column(name)

    def null_count(self) raises -> Self:
        """One row holding each column's null count as Int64."""
        var columns = List[Series](capacity=self.width())
        for column in self._columns:
            columns.append(
                Series(
                    column.name(), Column[Int64]([Int64(column.null_count())])
                )
            )
        return Self(columns^, height=1)

    def row(self, index: Int) raises -> List[AnyValue]:
        """Return one row as tagged values, in schema order."""
        if index < 0 or index >= self._height:
            raise Error("Row index out of bounds")
        var values = List[AnyValue](capacity=self.width())
        for column in self._columns:
            values.append(column.get(index))
        return values^

    def rows(self) raises -> List[List[AnyValue]]:
        """Materialize every row; intended for small frames and tests."""
        var result = List[List[AnyValue]](capacity=self._height)
        for i in range(self._height):
            result.append(self.row(i))
        return result^

    def to_dict(self) -> Dict[String, Series]:
        """Map column names to owned Series sharing their immutable buffers."""
        var result = Dict[String, Series]()
        for column in self._columns:
            result[column.name()] = column.copy()
        return result^

    def to_dicts(self) raises -> List[Dict[String, AnyValue]]:
        """Materialize every row as a name-to-value dictionary."""
        var result = List[Dict[String, AnyValue]](capacity=self._height)
        for row in range(self._height):
            var values = Dict[String, AnyValue]()
            for column in self._columns:
                values[column.name()] = column.get(row)
            result.append(values^)
        return result^

    def iter_rows(self) -> RowIterator:
        """Iterate owned row views, converting cells only when accessed.

        row[index] and row[name] return AnyValue and may raise a conversion
        error. row.to_list() materializes just that row. Views retain the
        source buffers, including after this frame or iterator is released.
        """
        return RowIterator(self)

    def iter_columns(self) -> ColumnIterator:
        """Yield one owned Series at a time, sharing immutable buffers."""
        return ColumnIterator(self)

    def iter_slices(self, n: Int = 10000) raises -> SliceIterator:
        """Yield frames of at most n rows without materializing all slices.

        n must be positive. Each slice shares the input's column buffers;
        the final slice may be shorter, and an empty input yields none.
        """
        return SliceIterator(self, n)

    def estimated_size(self, unit: String = "b") raises -> Float64:
        """Estimate visible data-buffer bytes in b/kb/mb/gb/tb (powers of 1024).

        Includes validity, chunks, nested children, and foreign buffers;
        excludes capacity, object metadata, caches, and categorical mappings.
        Shared buffers may be counted more than once. Binary-view strings
        count visible non-null value bytes, following Polars' estimate.
        Slices count their visible ranges, not retained allocations. Storage
        layouts differ from Polars, so estimates can differ for equal data.
        """
        return frame_size(self, unit)

    def item(self) raises -> AnyValue:
        """Return the only cell of a 1x1 dataframe."""
        if self._height != 1 or self.width() != 1:
            raise Error("item() requires a dataframe with exactly one cell")
        return self._columns[0].get(0)

    def item(self, row: Int, column: String) raises -> AnyValue:
        return self._columns[self._index(column)].get(row)

    def _shares_buffers_with(self, other: Self) -> Bool:
        """Whether both frames are the same columns over the same Arrow
        buffers (the same frame, however many times copied)."""
        if self.width() != other.width() or self._height != other._height:
            return False
        for c in range(self.width()):
            if self._columns[c].name() != other._columns[c].name():
                return False
            if not self._columns[c]._shares_buffers_with(other._columns[c]):
                return False
        return True

    def equals(self, other: Self, *, null_equal: Bool = True) -> Bool:
        """Same names, dtypes, order, height, and cells (NaN equals NaN)."""
        if self._height != other._height or self.width() != other.width():
            return False
        for i in range(self.width()):
            if not self._columns[i].equals(
                other._columns[i], null_equal=null_equal, check_names=True
            ):
                return False
        return True

    def slice(self, offset: Int, length: Int = -1) raises -> Self:
        """Rows [offset, offset + length), clipped to the frame.

        A negative offset counts from the end. length=-1 takes all remaining
        rows; other negative lengths raise.
        """
        if length < -1:
            raise Error("Slice length must be nonnegative")
        var start = offset
        if start < 0:
            start = max(self._height + start, 0)
        start = min(start, self._height)
        var available = self._height - start
        var count = available if length == -1 else min(length, available)
        var columns = List[Series](capacity=self.width())
        for column in self._columns:
            columns.append(column.slice(start, count))
        return Self(columns^, height=count)

    def head(self, n: Int = 5) raises -> Self:
        """First n rows; a negative n drops the last -n rows."""
        if n < 0:
            return self.slice(0, max(self._height + n, 0))
        return self.slice(0, n)

    def tail(self, n: Int = 5) raises -> Self:
        """Last n rows; a negative n drops the first -n rows."""
        if n < 0:
            return self.slice(min(-n, self._height))
        return self.slice(self._height - min(n, self._height))

    def limit(self, n: Int = 5) raises -> Self:
        return self.head(n)

    def reverse(self) raises -> Self:
        var columns = List[Series](capacity=self.width())
        for column in self._columns:
            columns.append(column.reverse())
        return Self(columns^, height=self._height)

    def shift(
        self, n: Int = 1, *, fill_value: Optional[Expr] = None
    ) raises -> Self:
        """Every column's values moved n rows later (earlier when n is
        negative). Height and schema stay the
        same. The min(|n|, height) rows left empty at the start (or, for
        a negative n, the end) are null, or `fill_value` cast to each
        column's type. `fill_value` is a scalar expression evaluated
        against this frame: a literal, or a reduction such as
        `col("a").max()`."""
        var height = self._height
        var vacated = height if n >= height or n <= -height else abs(n)
        if vacated == 0:
            return self.copy()
        var fills = Optional[Self]()
        if fill_value and self.width() > 0:
            if bind(fill_value.value(), self._columns).shape() == ROWS:
                raise Error("shift fill_value must be a scalar value")
            var exprs = List[Expr](capacity=self.width())
            for column in self._columns:
                exprs.append(
                    fill_value.value().cast(column.dtype()).alias(column.name())
                )
            fills = self.select_exprs(exprs)
        # The source row of each output row; vacated rows read -1 (null)
        # or, with a fill value, the fill row appended after the last.
        var empty_row = height if fills else -1
        var rows = List[Int](capacity=height)
        for i in range(height):
            if n > 0:
                rows.append(i - vacated if i >= vacated else empty_row)
            else:
                rows.append(i + vacated if i + vacated < height else empty_row)
        var columns = List[Series](capacity=self.width())
        for k in range(self.width()):
            ref column = self._columns[k]
            if fills:
                columns.append(
                    column.append(fills.value()._columns[k]).take(rows)
                )
            else:
                columns.append(column.take_or_null(rows))
        return Self(columns^, height=height)

    def vstack(self, other: Self) raises -> Self:
        """Append other's rows; schemas must match exactly."""
        return concat([self.copy(), other.copy()], "vertical")

    def hstack(self, other: Self) raises -> Self:
        """Append other's columns; heights must match, names stay unique."""
        return concat([self.copy(), other.copy()], "horizontal")

    def hstack(self, columns: List[Series]) raises -> Self:
        return self.hstack(Self(columns.copy(), height=self._height))

    def clear(self, n: Int = 0) raises -> Self:
        """The same schema with n rows of nulls (zero rows by default)."""
        if n < 0:
            raise Error("clear: n must be nonnegative")
        if n == 0:
            return self.slice(0, 0)
        var columns = List[Series](capacity=self.width())
        for column in self._columns:
            columns.append(Series.full_null(column.name(), column.dtype(), n))
        return Self(columns^, height=n)

    def drop(self, name: String) raises -> Self:
        return self.drop([name])

    def drop(self, names: List[String]) raises -> Self:
        """Remove columns; every name must exist and appear once."""
        var dropped = Dict[String, Bool]()
        for name in names:
            _ = self._index(name)
            if name in dropped:
                raise Error("Column listed twice in drop: " + name)
            dropped[name] = True
        var columns = List[Series]()
        for column in self._columns:
            if column.name() not in dropped:
                columns.append(column.copy())
        return Self(columns^, height=self._height)

    def rename(
        self, mapping: Dict[String, String], *, strict: Bool = True
    ) raises -> Self:
        """Rename columns by old name; all names are validated first. With
        strict=False, names this frame lacks are ignored."""
        if strict:
            for item in mapping.items():
                _ = self._index(item.key)
        var columns = List[Series](capacity=self.width())
        var seen = Dict[String, Bool]()
        for column in self._columns:
            var name = column.name()
            if name in mapping:
                name = mapping[name]
            if name in seen:
                raise Error("Rename produces duplicate column name: " + name)
            seen[name] = True
            columns.append(column.renamed(name))
        return Self(columns^, height=self._height)

    def rename(
        self, function: def(String) raises thin -> String
    ) raises -> Self:
        """Rename every column using a Mojo function; reject duplicate outputs."""
        var mapping = Dict[String, String]()
        for name in self.columns():
            mapping[name] = function(name)
        return self.rename(mapping)

    def rename(self, function: def(String) thin -> String) raises -> Self:
        """Rename every column using a Mojo function; reject duplicate outputs."""
        var mapping = Dict[String, String]()
        for name in self.columns():
            mapping[name] = function(name)
        return self.rename(mapping)

    def insert_column(self, index: Int, column: Series) raises -> Self:
        """Insert a column by position, returning a frame with shared buffers.

        Negative positions count from the end. Names must be unique and
        heights must match; a 0x0 frame takes its first column's height.
        """
        var position = index + self.width() if index < 0 else index
        if position < 0 or position > self.width():
            raise Error("Column insertion index out of bounds")
        var height = self._height
        if height == 0 and self.width() == 0:
            height = len(column)
        if len(column) != height:
            raise Error("Inserted column must match dataframe height")
        var columns = List[Series](capacity=self.width() + 1)
        for i in range(self.width() + 1):
            if i == position:
                columns.append(column.copy())
            if i < self.width():
                columns.append(self._columns[i].copy())
        return Self(columns^, height=height)

    def gather_every(self, n: Int, offset: Int = 0) raises -> Self:
        """Take every nth row from a nonnegative offset; n must be positive."""
        if n <= 0 or offset < 0:
            raise Error("gather_every needs positive n and nonnegative offset")
        var indices = List[Int]()
        var row = offset
        while row < self._height:
            indices.append(row)
            if n >= self._height - row:
                break
            row += n
        return self.take(indices)

    def with_row_index(
        self, name: String = "index", offset: Int64 = 0
    ) raises -> Self:
        """Prepend an Int64 row index starting at offset."""
        for column in self._columns:
            if column.name() == name:
                raise Error("Row index name collides with column: " + name)
        var values = List[Int64](capacity=self._height)
        for i in range(self._height):
            values.append(offset + Int64(i))
        var columns = List[Series](capacity=self.width() + 1)
        columns.append(Series(name, Column[Int64](values^)))
        for column in self._columns:
            columns.append(column.copy())
        return Self(columns^, height=self._height)

    def select(self, names: List[String]) raises -> Self:
        var columns = List[Series](capacity=len(names))
        for name in names:
            columns.append(self.column(name))
        return Self(columns^, height=self._height)

    def sample(
        self,
        n: Optional[Int] = None,
        *,
        fraction: Optional[Float64] = None,
        with_replacement: Bool = False,
        shuffle: Bool = False,
        seed: Optional[Int] = None,
    ) raises -> Self:
        """A random sample of rows: n of them, or floor(fraction * height),
        or one row when neither is given.

        Without replacement no row repeats and asking for more rows than the
        frame has raises. Rows keep their original order unless shuffle is
        set. A seed makes the sample reproducible across runs and
        platforms; without one the generator is seeded from the clock.
        The generator is SplitMix64.
        """
        var count = sample_size(self._height, n, fraction, with_replacement)
        return self.take(
            sample_indices(self._height, count, with_replacement, shuffle, seed)
        )

    def sum(self) raises -> Self:
        """Sum each numeric column; unsupported columns are null."""
        return self._reduce_frame("sum")

    def mean(self) raises -> Self:
        """Mean each numeric column; unsupported columns are null."""
        return self._reduce_frame("mean")

    def min(self) raises -> Self:
        """Minimum of each ordered column; nested columns are null."""
        return self._reduce_frame("min")

    def max(self) raises -> Self:
        """Maximum of each ordered column; nested columns are null."""
        return self._reduce_frame("max")

    def median(self) raises -> Self:
        """Median of each numeric column; unsupported columns are null."""
        return self._reduce_frame("median")

    def count(self) raises -> Self:
        """Count non-null values in every column, as UInt32."""
        return self._reduce_frame("count")

    def product(self) raises -> Self:
        """Product of each numeric column; unsupported columns are null."""
        return self._reduce_frame("product")

    def std(self, ddof: Int = 1) raises -> Self:
        """Column-wise standard deviation, with nonnegative ddof."""
        return self._reduce_frame("std", ddof=ddof)

    def var(self, ddof: Int = 1) raises -> Self:
        """Column-wise variance, with nonnegative ddof."""
        return self._reduce_frame("var", ddof=ddof)

    def quantile(
        self, quantile: Float64, interpolation: String = "nearest"
    ) raises -> Self:
        """Numeric column quantiles; interpolation defaults to nearest."""
        return self._reduce_frame(
            "quantile", quantile=quantile, interpolation=interpolation
        )

    def _reduce_frame(
        self,
        reduction: String,
        *,
        ddof: Int = 1,
        quantile: Float64 = 0.5,
        interpolation: String = "nearest",
    ) raises -> Self:
        var expressions = self._frame_reduction_exprs(
            reduction, ddof, quantile, interpolation
        )
        if len(expressions) == 0:
            return Self(List[Series](), height=1)
        return self.select_exprs(expressions)

    def _frame_reduction_exprs(
        self,
        reduction: String,
        ddof: Int,
        quantile: Float64,
        interpolation: String,
    ) raises -> List[Expr]:
        """Choose frame reductions from schema alone, shared with LazyFrame."""
        if (reduction == "std" or reduction == "var") and ddof < 0:
            raise Error("ddof must be nonnegative")
        if reduction == "quantile":
            if not (quantile >= 0 and quantile <= 1):
                raise Error("quantile must be between 0 and 1")
            if interpolation not in [
                String("linear"),
                "lower",
                "higher",
                "nearest",
                "midpoint",
            ]:
                raise Error("Invalid quantile interpolation")
        var expressions = List[Expr](capacity=self.width())
        for column in self._columns:
            var dtype = column.dtype()
            var c = col(column.name())
            var numeric = dtype.is_numeric() or dtype == DataType.BOOL
            var expression: Expr
            if reduction == "count":
                expression = c.is_not_null().sum()
            elif reduction == "min" or reduction == "max":
                expression = (
                    null(dtype) if dtype.is_nested()
                    or dtype
                    == DataType.NULL else c.min() if reduction
                    == "min" else c.max()
                )
                if dtype.is_categorical():
                    expression = expression.cast(dtype)
            elif reduction == "sum":
                expression = (
                    c.sum() if numeric or dtype.is_duration() else null(dtype)
                )
            elif reduction == "product":
                expression = c.product() if numeric else null(DataType.NULL)
            elif reduction == "mean":
                expression = c.mean() if numeric else null(dtype)
            elif numeric:
                var input = (
                    c.cast(DataType.FLOAT64) if dtype == DataType.BOOL
                    or dtype.is_decimal() else c.copy()
                )
                if reduction == "std":
                    expression = input.std(ddof)
                elif reduction == "var":
                    expression = input.var(ddof)
                elif reduction == "median":
                    expression = input.median()
                else:
                    expression = input.quantile(quantile, interpolation)
            else:
                expression = null(dtype)
            if dtype == DataType.FLOAT32 and reduction in [
                String("mean"),
                "std",
                "var",
                "median",
                "quantile",
            ]:
                expression = expression.cast(DataType.FLOAT32)
            expressions.append(expression.alias(column.name()))
        return expressions^

    def describe(
        self,
        percentiles: List[Float64] = [0.25, 0.5, 0.75],
        interpolation: String = "nearest",
    ) raises -> Self:
        """Summary statistics per column.

        Rows: count, null_count, mean, std, min, one row per percentile in
        ascending order (labelled like "25%"), then max. Pass an empty list
        for no percentile rows. Numeric and Bool columns give Float64 (Bool
        mean is the share of true values; std and percentiles are null).
        String columns give String: counts, min and max. Temporal columns
        give String: counts, the mean (a datetime for Date columns), min,
        percentiles and max, formatted as a cast to String formats them.
        Nested columns give Float64 counts and nulls elsewhere. min and max
        skip NaN; mean and std propagate it.
        """
        var qs = _describe_percentiles(percentiles)
        var exprs = self._describe_exprs(qs, interpolation)
        var stats = self.select_exprs(exprs) if len(exprs) > 0 else Self(
            List[Series](), height=1
        )
        return self._described(qs, stats)

    def _describe_exprs(
        self, qs: List[Float64], interpolation: String
    ) -> List[Expr]:
        """The reductions `describe` computes, chosen from the column names
        and types alone, so LazyFrame.describe computes the same ones over
        its plan."""
        var exprs = List[Expr]()
        for i in range(self.width()):
            ref column = self._columns[i]
            var dtype = column.dtype()
            var c = col(column.name())
            var p = String(i) + "_"
            exprs.append(c.count().alias(p + "count"))
            exprs.append(c.null_count().alias(p + "null_count"))
            if dtype.is_numeric() or dtype == DataType.BOOL:
                var x = c.cast(DataType.FLOAT64)
                # min and max skip NaN unless every valid value is NaN, as
                # the min and max expressions do (#816).
                exprs.append(x.mean().alias(p + "mean"))
                exprs.append(x.min().alias(p + "min"))
                exprs.append(x.max().alias(p + "max"))
                if dtype.is_numeric():
                    exprs.append(x.std().alias(p + "std"))
                    for k in range(len(qs)):
                        exprs.append(
                            x.quantile(qs[k], interpolation).alias(
                                p + "q" + String(k)
                            )
                        )
            elif dtype.physical() == DataType.STRING:
                exprs.append(c.min().alias(p + "min"))
                exprs.append(c.max().alias(p + "max"))
            elif dtype.is_temporal():
                var x = c.cast(DataType.INT64)
                exprs.append(x.cast(DataType.FLOAT64).mean().alias(p + "mean"))
                exprs.append(x.min().alias(p + "min"))
                exprs.append(x.max().alias(p + "max"))
                for k in range(len(qs)):
                    exprs.append(
                        x.cast(DataType.FLOAT64)
                        .quantile(qs[k], interpolation)
                        .alias(p + "q" + String(k))
                    )
        return exprs^

    def _described(self, qs: List[Float64], stats: DataFrame) raises -> Self:
        """`describe`'s frame from the reductions `_describe_exprs` named,
        computed into one row of `stats`; only this frame's names and types
        are read."""
        var labels: List[String] = ["count", "null_count", "mean", "std", "min"]
        for q in qs:
            labels.append(_percent_label(q))
        labels.append("max")
        var rows = len(labels)
        var present = Dict[String, Bool]()
        for name in stats.columns():
            present[name] = True
        var columns = List[Series](capacity=self.width() + 1)
        columns.append(Series("statistic", Column[String](labels.copy())))
        for i in range(self.width()):
            ref column = self._columns[i]
            var dtype = column.dtype()
            var p = String(i) + "_"
            var count = stats.column(p + "count").get(0).int64()
            var nulls = stats.column(p + "null_count").get(0).int64()
            # Stat name for each output row: "count", ..., "q0", ..., "max".
            var keys: List[String] = ["count", "null_count", "mean", "std"]
            keys.append("min")
            for k in range(len(qs)):
                keys.append("q" + String(k))
            keys.append("max")
            if (
                dtype.is_numeric()
                or dtype == DataType.BOOL
                or dtype.is_nested()
            ):
                var values = List[Float64](capacity=rows)
                var valid = List[Bool](capacity=rows)
                for key in keys:
                    if key == "count":
                        values.append(Float64(count))
                        valid.append(True)
                    elif key == "null_count":
                        values.append(Float64(nulls))
                        valid.append(True)
                    elif (p + key) in present:
                        var value = stats.column(p + key).get(0)
                        valid.append(not value.is_null())
                        values.append(0 if value.is_null() else value.float64())
                    else:
                        values.append(0)
                        valid.append(False)
                columns.append(
                    Series(column.name(), Column[Float64](values^, valid))
                )
                continue
            var texts = List[String](capacity=rows)
            var valid = List[Bool](capacity=rows)
            if dtype.is_temporal():
                # Physical values for min, percentiles and max, formatted
                # together by one cast; the mean gets its own type.
                var ticks = List[Int64]()
                var ticks_valid = List[Bool]()
                for key in keys:
                    if key == "min" or key == "max" or key.startswith("q"):
                        var value = stats.column(p + key).get(0)
                        ticks_valid.append(not value.is_null())
                        ticks.append(
                            0 if value.is_null() else _round_ticks(
                                value.float64()
                            ) if key.startswith("q") else value.int64()
                        )
                var formatted = (
                    Series("", Column[Int64](ticks^, ticks_valid))
                    .with_dtype(dtype)
                    .cast(DataType.STRING)
                )
                var mean = stats.column(p + "mean").get(0)
                var mean_type = dtype
                var mean_ticks = Int64(0)
                if not mean.is_null():
                    var ticks_mean = mean.float64()
                    if dtype.is_date():
                        mean_type = DataType.datetime("us")
                        ticks_mean *= 86400000000.0
                    mean_ticks = _round_ticks(ticks_mean)
                var mean_text = (
                    Series(
                        "",
                        Column[Int64]([mean_ticks], [not mean.is_null()]),
                    )
                    .with_dtype(mean_type)
                    .cast(DataType.STRING)
                    .get(0)
                )
                var at = 0
                for key in keys:
                    if key == "count":
                        texts.append(String(count))
                        valid.append(True)
                    elif key == "null_count":
                        texts.append(String(nulls))
                        valid.append(True)
                    elif key == "mean":
                        valid.append(not mean_text.is_null())
                        texts.append(
                            "" if mean_text.is_null() else mean_text.string()
                        )
                    elif key == "std":
                        texts.append("")
                        valid.append(False)
                    else:
                        var value = formatted.get(at)
                        at += 1
                        valid.append(not value.is_null())
                        texts.append("" if value.is_null() else value.string())
            else:
                for key in keys:
                    if key == "count":
                        texts.append(String(count))
                        valid.append(True)
                    elif key == "null_count":
                        texts.append(String(nulls))
                        valid.append(True)
                    elif (p + key) in present:
                        var value = stats.column(p + key).get(0)
                        valid.append(not value.is_null())
                        # Binary min and max print as b"...", as values do.
                        texts.append(
                            "" if value.is_null() else (
                                String(
                                    value
                                ) if dtype._is_bytes() else value.string()
                            )
                        )
                    else:
                        texts.append("")
                        valid.append(False)
            columns.append(Series(column.name(), StringColumn(texts, valid)))
        return Self(columns^, height=rows)

    def gather(
        self, indices: List[Int], *, null_on_oob: Bool = False
    ) raises -> Self:
        """The rows at `indices`, in that order: a
        negative index counts from the end. An index out of range raises,
        or with null_on_oob=True gives a row of nulls."""
        return self._gather(indices, List[Bool](), null_on_oob)

    def _gather(
        self, indices: List[Int], valid: List[Bool], null_on_oob: Bool
    ) raises -> Self:
        """`gather`, where a false entry of `valid` (when given) is a null
        index, which gives a row of nulls."""
        var rows = List[Int](capacity=len(indices))
        var nulls = False
        for k in range(len(indices)):
            var index = indices[k]
            var row = index + self._height if index < 0 else index
            if len(valid) > 0 and not valid[k]:
                row = -1
            elif row < 0 or row >= self._height:
                if not null_on_oob:
                    raise Error(
                        "gather index "
                        + String(index)
                        + " is out of bounds for height "
                        + String(self._height)
                    )
                row = -1
            nulls = nulls or row < 0
            rows.append(row)
        if not nulls:
            return self.take(rows)
        var columns = List[Series](capacity=self.width())
        for column in self._columns:
            columns.append(column.take_or_null(rows))
        return Self(columns^, height=len(rows))

    def take(self, indices: List[Int]) raises -> Self:
        for i in indices:
            if i < 0 or i >= self._height:
                raise Error("Row index out of bounds")
        var workers = worker_count(len(indices))
        # At medium row counts, columns can run independently even when
        # there are too few rows to split each column across workers.
        if self.width() > 0 and (
            workers > 1 or (len(indices) >= 65536 and self.width() >= 4)
        ):
            return Self(
                take_parallel(self._columns, indices.copy(), workers),
                height=len(indices),
            )
        var columns = List[Series](capacity=self.width())
        for column in self._columns:
            columns.append(column.take(indices))
        return Self(columns^, height=len(indices))

    def filter(self, mask: Column[Bool]) raises -> Self:
        """Filter with a byte-per-value Boolean column (packed first)."""
        return self.filter(BoolColumn(mask))

    def filter(self, mask: BoolColumn) raises -> Self:
        """Keep true rows, dropping false and null mask entries, in input order.

        Fixed-width and Boolean columns are filtered from the mask's words
        directly (#376); string and nested columns from a list of rows.
        """
        if len(mask) != self._height:
            raise Error("Filter mask must match dataframe height")
        if _keeps_every_row(mask):
            # Nothing to drop (`is_not_null()` on a column without nulls):
            # the frame's columns are shared, not copied.
            return self.copy()
        var filtered = filter_columns(self._columns, mask)
        ref others = filtered[1]
        var count = filtered[2]
        var rest = List[Series]()
        if len(others) > 0:
            var subset = List[Series](capacity=len(others))
            for i in others:
                subset.append(self._columns[i].copy())
            rest = (
                Self(subset^, height=self._height)
                ._filter_rows(true_rows(mask))
                ._columns.copy()
            )
        var columns = List[Series](capacity=self.width())
        var next_fixed = 0
        var next_other = 0
        for i in range(self.width()):
            if next_other < len(others) and others[next_other] == i:
                columns.append(rest[next_other].copy())
                next_other += 1
            else:
                columns.append(filtered[0][next_fixed].copy())
                next_fixed += 1
        return Self(columns^, height=count)

    def explode(self, column: String) raises -> Self:
        return self.explode([column])

    def explode(self, columns: List[String]) raises -> Self:
        """One output row per list element; other columns repeat. An empty
        or null list gives one row holding null. Several columns explode
        together and must have the same element count in every row."""
        if len(columns) == 0:
            raise Error("explode needs at least one column")
        var positions = List[Int](capacity=len(columns))
        var lists = List[ListColumn](capacity=len(columns))
        for name in columns:
            var index = self._index(name)
            for p in positions:
                if p == index:
                    raise Error("explode: column listed twice: " + name)
            if not self._columns[index].dtype().is_list():
                raise Error(
                    "explode needs list columns; "
                    + name
                    + " is "
                    + self._columns[index].dtype().name()
                )
            positions.append(index)
            lists.append(self._columns[index].list_column())
        var repeat = List[Int]()
        var child_rows = List[List[Int]](capacity=len(lists))
        for _ in range(len(lists)):
            child_rows.append(List[Int]())
        for i in range(self._height):
            var count = lists[0].element_count(i)
            for k in range(1, len(lists)):
                if lists[k].element_count(i) != count:
                    raise Error(
                        "explode: columns have different element counts at"
                        " row " + String(i)
                    )
            if count == 0:
                repeat.append(i)
                for k in range(len(lists)):
                    child_rows[k].append(-1)
                continue
            for e in range(count):
                repeat.append(i)
                for k in range(len(lists)):
                    child_rows[k].append(lists[k]._start(i) + e)
        var out = List[Series](capacity=self.width())
        for c in range(self.width()):
            var exploded = -1
            for k in range(len(positions)):
                if positions[k] == c:
                    exploded = k
            if exploded < 0:
                out.append(self._columns[c].take(repeat))
            else:
                out.append(
                    lists[exploded]
                    .child()
                    .take_or_null(child_rows[exploded])
                    .renamed(self._columns[c].name())
                )
        return Self(out^, height=len(repeat))

    def unnest(self, column: String) raises -> Self:
        """Replace a struct column with its fields as top-level columns, in
        its position. Field names must not clash with other columns."""
        var index = self._index(column)
        if not self._columns[index].dtype().is_struct():
            raise Error(
                "unnest needs a struct column; "
                + column
                + " is "
                + self._columns[index].dtype().name()
            )
        var structs = self._columns[index].struct_column()
        var out = List[Series](capacity=self.width() + structs.field_count())
        for c in range(self.width()):
            if c != index:
                out.append(self._columns[c].copy())
                continue
            for f in range(structs.field_count()):
                var field = structs.field(f)
                if structs.null_count() > 0:
                    var rows = List[Int](capacity=len(structs))
                    for i in range(len(structs)):
                        rows.append(i if structs._valid(i) else -1)
                    field = field.take_or_null(rows)
                for other in range(self.width()):
                    if (
                        other != index
                        and self._columns[other].name() == field.name()
                    ):
                        raise Error(
                            "unnest: field "
                            + field.name()
                            + " clashes with an existing column"
                        )
                out.append(field^)
        return Self(out^, height=self._height)

    def pack_struct(self, name: String, columns: List[String]) raises -> Self:
        """Add a struct column built from existing columns (kept as they
        are); unnest(name) gives them back."""
        var fields = List[Series](capacity=len(columns))
        for c in columns:
            fields.append(self.column(c))
        return self.with_column(Series(name, StructColumn(fields^)))

    def _filter_rows(self, var rows: List[Int]) raises -> Self:
        var max_chunks = 1
        for column in self._columns:
            max_chunks = max(max_chunks, column.n_chunks())
        # For tiny CSV ranges, chunk-local gathers cost more than a
        # single contiguous gather. Larger ranges repay that setup by avoiding
        # a full input rechunk before selecting the rows.
        if max_chunks > 1 and self._height // max_chunks >= 512:
            var count = len(rows)
            return Self(
                take_sorted_chunked(self._columns, rows^, worker_count(count)),
                height=count,
            )
        return self.take(rows)

    def with_column(self, var column: Series) raises -> Self:
        """Replace by name or append; the input dataframe is unchanged."""
        if len(column) != self._height:
            raise Error("New column must match dataframe height")
        var columns = self._columns.copy()
        for i in range(len(columns)):
            if columns[i].name() == column.name():
                columns[i] = column^
                return Self(columns^, height=self._height)
        columns.append(column^)
        return Self(columns^, height=self._height)

    def sort(
        self,
        by: String,
        descending: Bool = False,
        nulls_last: Bool = True,
        *,
        maintain_order: Bool = False,
    ) raises -> Self:
        """Single-column sort. NaNs follow non-null numbers. Rows with equal
        keys come out in no particular order unless `maintain_order` keeps
        their input order."""
        return self.sort(
            [by], descending, nulls_last, maintain_order=maintain_order
        )

    def sort(
        self,
        by: List[String],
        descending: Bool = False,
        nulls_last: Bool = True,
        *,
        maintain_order: Bool = False,
    ) raises -> Self:
        """Lexicographic sort by several columns, one direction; equal keys
        in no particular order unless `maintain_order`."""
        _ = maintain_order
        return self.take(self.arg_sort(by, descending, nulls_last))

    def sort(
        self,
        by: List[String],
        *,
        descending: List[Bool],
        nulls_last: List[Bool],
        maintain_order: Bool = False,
    ) raises -> Self:
        """Per-column direction and null placement; list lengths match by.
        Equal keys in no particular order unless `maintain_order`."""
        _ = maintain_order
        return self.take(
            self.arg_sort(by, descending=descending, nulls_last=nulls_last)
        )

    def _sort_key_exprs(self, by: List[Expr]) raises -> List[Expr]:
        var expressions = expand_all(by, self._columns)
        var names = self.columns()
        for i in range(len(expressions)):
            var name = String("__dataframe_sort_key_") + String(i)
            while name in names:
                name += "_"
            names.append(name)
            expressions[i] = expressions[i].alias(name)
        return expressions^

    def sort(
        self,
        by: Expr,
        descending: Bool = False,
        nulls_last: Bool = True,
        *,
        maintain_order: Bool = False,
    ) raises -> Self:
        """Sort by an expression, evaluated once before sorting."""
        return self.sort(
            [by.copy()], descending, nulls_last, maintain_order=maintain_order
        )

    def sort(
        self,
        by: List[Expr],
        descending: Bool = False,
        nulls_last: Bool = True,
        *,
        maintain_order: Bool = False,
    ) raises -> Self:
        """Sort by expression keys with common direction and null placement."""
        var keys = self._sort_key_exprs(by)
        return self._sort_by_keys(
            keys,
            List[Bool](length=len(keys), fill=descending),
            List[Bool](length=len(keys), fill=nulls_last),
            maintain_order,
        )

    def sort(
        self,
        by: List[Expr],
        *,
        descending: List[Bool],
        nulls_last: List[Bool],
        maintain_order: Bool = False,
    ) raises -> Self:
        """Sort by expression keys with a direction and null placement each."""
        return self._sort_by_keys(
            self._sort_key_exprs(by), descending, nulls_last, maintain_order
        )

    def _sort_by_keys(
        self,
        keys: List[Expr],
        descending: List[Bool],
        nulls_last: List[Bool],
        maintain_order: Bool,
    ) raises -> Self:
        _ = maintain_order
        if len(descending) != len(keys) or len(nulls_last) != len(keys):
            raise Error("sort needs one descending and nulls_last flag per key")
        if len(keys) == 0:
            return self.copy()
        var names = List[String](capacity=len(keys))
        for key in keys:
            names.append(key._name)
        var evaluated = self.with_columns(keys).select(names)
        return self.take(
            evaluated.arg_sort(
                names, descending=descending, nulls_last=nulls_last
            )
        )

    def arg_sort(
        self,
        by: List[String],
        descending: Bool = False,
        nulls_last: Bool = True,
    ) raises -> List[Int]:
        return self.arg_sort(
            by,
            descending=List[Bool](length=len(by), fill=descending),
            nulls_last=List[Bool](length=len(by), fill=nulls_last),
        )

    def arg_sort(
        self,
        by: List[String],
        *,
        descending: List[Bool],
        nulls_last: List[Bool],
    ) raises -> List[Int]:
        """Row order of a stable sort; equal keys keep input order."""
        var ranked = self._categorical_ranks(by)
        if ranked:
            return ranked.value().arg_sort(
                by, descending=descending, nulls_last=nulls_last
            )
        for name in by:
            if self.column(name).dtype().is_nested():
                raise Error(
                    "cannot sort by a "
                    + self.column(name).dtype().name()
                    + " column: "
                    + name
                )
        if (
            len(by) > 0
            and len(descending) == len(by)
            and len(nulls_last) == len(by)
        ):
            var keys = List[Series](capacity=len(by))
            for name in by:
                keys.append(self.column(name))
            # String keys whose values repeat sort by their ranks: each
            # distinct value is compared once, and the ranks pack with the
            # other keys where the strings would not.
            var ranked = keys.copy()
            var ranked_any = False
            for k in range(len(ranked)):
                var ranks = string_sort_ranks(ranked[k])
                if ranks:
                    ranked[k] = ranks.take()
                    ranked_any = True
            if ranked_any:
                var packed = packed_arg_sort(ranked, descending, nulls_last)
                if packed:
                    trace_path("sort.string_ranks")
                    return packed.take()
            var packed = packed_arg_sort(keys, descending, nulls_last)
            if packed:
                trace_path("sort.packed")
                return packed.take()
        var keys = List[Series]()
        var fixed = (
            len(by) > 0
            and len(descending) == len(by)
            and len(nulls_last) == len(by)
        )
        for name in by:
            keys.append(self.column(name))
            fixed = fixed and encodable(keys[len(keys) - 1])
        if fixed:
            return sort_indices(encode_sort_keys(keys, descending, nulls_last))
        trace_path("sort.row_prefix")
        return row_arg_sort(keys, descending, nulls_last)

    def top_k(
        self, k: Int, by: List[String], *, reverse: Bool = False
    ) raises -> Self:
        """The k rows with greatest values by `by` (ascending when
        `reverse=True`).

        Nulls rank last. Selection is about one pass rather than a full sort.
        """
        var n = len(by)
        return self.take(
            self._arg_sort_head(
                by,
                List[Bool](length=n, fill=not reverse),
                List[Bool](length=n, fill=True),
                k,
            )
        )

    def top_k(
        self, k: Int, by: String, *, reverse: Bool = False
    ) raises -> Self:
        return self.top_k(k, [by], reverse=reverse)

    def bottom_k(
        self, k: Int, by: List[String], *, reverse: Bool = False
    ) raises -> Self:
        """The k rows that sort(by) would put first (sort(by,
        descending=True) with reverse=True); nulls rank last."""
        var n = len(by)
        return self.take(
            self._arg_sort_head(
                by,
                List[Bool](length=n, fill=reverse),
                List[Bool](length=n, fill=True),
                k,
            )
        )

    def _categorical_ranks(self, by: List[String]) raises -> Optional[Self]:
        """This frame with each categorical sort key replaced by the rank of
        its value, or None when no key is categorical: codes number values
        in the order they first appeared, not in sorted order (#106)."""
        var found = False
        for name in by:
            found = found or self.column(name).dtype().is_categorical()
        if not found:
            return None
        var columns = self._columns.copy()
        for i in range(len(columns)):
            if columns[i].name() in by and columns[i].dtype().is_categorical():
                columns[i] = sort_ranks(columns[i])
        return Self(columns^, height=self._height)

    def _arg_sort_head(
        self,
        by: List[String],
        descending: List[Bool],
        nulls_last: List[Bool],
        k: Int,
        threads: Int = 0,
    ) raises -> List[Int]:
        """The first k rows of arg_sort(by, ...), in order (#332).

        `threads` caps the selection's workers (0: the configured count); a
        streaming batch, already on a worker, passes 1.


        Small k selects the rows instead of sorting all of them: with packed
        keys when they pack, else a heap of row sort words per range
        (`row_top_rows`). Past a tenth of the rows selection stops paying
        and the full sort is cut short.
        """
        if k < 0:
            raise Error("k must be nonnegative")
        var ranked = self._categorical_ranks(by)
        if ranked:
            return ranked.value()._arg_sort_head(
                by, descending, nulls_last, k, threads
            )
        var take = min(k, self._height)
        if take * _TOP_K_FRACTION >= self._height:
            var order = self.arg_sort(
                by, descending=descending, nulls_last=nulls_last
            )
            order.shrink(take)
            return order^
        for name in by:
            if self.column(name).dtype().is_nested():
                raise Error(
                    "cannot sort by a "
                    + self.column(name).dtype().name()
                    + " column: "
                    + name
                )
        if (
            len(by) > 0
            and len(descending) == len(by)
            and len(nulls_last) == len(by)
        ):
            var keys = List[Series](capacity=len(by))
            for name in by:
                keys.append(self.column(name))
            var packed = packed_top_rows(
                keys, descending, nulls_last, take, threads
            ) if threads > 0 else chunked_top_rows(
                keys, descending, nulls_last, take
            )
            if packed:
                trace_path("sort.top_k")
                return packed.take()
        var keys = List[Series](capacity=len(by))
        for name in by:
            keys.append(self.column(name))
        trace_path("sort.row_top_k")
        return row_top_rows(keys, descending, nulls_last, take, threads)

    def bottom_k(
        self, k: Int, by: String, *, reverse: Bool = False
    ) raises -> Self:
        return self.bottom_k(k, [by], reverse=reverse)

    def _sort_ranks(
        self,
        by: List[String],
        descending: List[Bool],
        nulls_last: List[Bool],
    ) raises -> List[List[Int]]:
        if len(by) == 0:
            raise Error("sort requires at least one column")
        if len(descending) != len(by) or len(nulls_last) != len(by):
            raise Error(
                "descending and nulls_last must have one entry per sort column"
            )
        # Fixed-width keys need no ranking at all: each value maps to an
        # Int ordered as the value is, in one linear pass, which is what
        # `sort_indices` compares anyway. Ranking exists to handle strings,
        # whose order cannot be carried in a fixed-width word.
        var keys = List[Series](capacity=len(by))
        var fixed = True
        for i in range(len(by)):
            ref column = self._columns[self._index(by[i])]
            if not encodable(column):
                fixed = False
            keys.append(column.copy())
        if fixed:
            return encode_sort_keys(keys, descending, nulls_last)

        # Small top-k and window consumers still need rank-compatible
        # words. Only unsupported keys require a value sort: retain direct
        # encoding for every other key when one long string/decimal is here.
        var jobs = List[_RankJob]()
        var positions = List[Int](length=len(by), fill=-1)
        for i in range(len(by)):
            if not encodable(keys[i]):
                positions[i] = len(jobs)
                jobs.append(
                    _RankJob(keys[i].copy(), descending[i], nulls_last[i])
                )
        if worker_count(self.height()) > 1:
            run_jobs(jobs)
        else:
            for j in range(len(jobs)):
                jobs[j].run()
        var ranks = List[List[Int]]()
        for i in range(len(by)):
            if positions[i] >= 0:
                ranks.append(jobs[positions[i]].ranks.copy())
            else:
                var words = encode_sort_keys(
                    [keys[i].copy()], [descending[i]], [nulls_last[i]]
                )
                while len(words) > 0:
                    ranks.append(words.pop(0))
        return ranks^

    def join_asof(
        self,
        right: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: List[String] = List[String](),
        strategy: String = "backward",
        tolerance: Optional[AnyValue] = None,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        """Match each left row to an ordered right key, optionally within `by`.

        Strategies are backward (last <=), forward (first >=), and nearest
        (ties prefer the later key and last duplicate). Null ordered keys
        and null group keys do not match. Inputs need not be sorted: a side
        whose keys are out of order within a group is matched through a
        stable (group, key) order with null keys last. Output keeps left row
        order.
        Keys must have the same integer, float, Date, Datetime or Duration
        dtype. Use `on`, or both `left_on` and `right_on`. Right keys with a
        different name are retained; common keys and `by` columns coalesce.
        Tolerance is a nonnegative number in key units or an AnyValue
        Duration (for temporal keys), and None means no limit. Numeric
        overloads accept plain integers/floats; AnyValue also retains an
        explicitly typed unsigned tolerance. `by` is a list of column names.
        """
        var left_key = left_on
        var right_key = right_on
        if on != "":
            if left_key != "" or right_key != "":
                raise Error("join_asof: use on or left_on/right_on, not both")
            left_key = on
            right_key = on
        if left_key == "" or right_key == "":
            raise Error("join_asof requires on or both left_on and right_on")
        var lhs = self.column(left_key)
        var rhs = right.column(right_key)
        var groups = by.copy()
        var seen = Dict[String, Bool]()
        var stacked = List[Series]()
        for name in groups:
            if name in seen or name == left_key or name == right_key:
                raise Error("join_asof duplicate or ordered key in by: " + name)
            seen[name] = True
            var l = self.column(name)
            var r = right.column(name)
            if l.dtype().is_categorical() or r.dtype().is_categorical():
                unify(l, r)
            if l.dtype() != r.dtype():
                raise Error("join_asof by dtypes must match: " + name)
            stacked.append(l.append(r))
        var left_ids = List[Int]()
        var right_ids = List[Int]()
        var matchable = List[Bool]([True])
        if len(groups):
            var keys = encode_rows(_expand_struct_keys(stacked), True)
            matchable = List[Bool](length=keys.count(), fill=True)
            for g in range(keys.count()):
                for key in stacked:
                    if key.get(keys.representatives[g]).is_null():
                        matchable[g] = False
            for row in range(self.height()):
                left_ids.append(keys.ids[row])
            for row in range(right.height()):
                right_ids.append(keys.ids[self.height() + row])
        var rows = asof_rows(
            lhs,
            rhs,
            left_ids,
            right_ids,
            matchable,
            strategy,
            tolerance,
            allow_exact_matches,
        )
        var columns = self._columns.copy()
        var names = Dict[String, Bool]()
        for name in self.columns():
            names[name] = True
        var left_names = names.copy()
        for column in right._columns:
            var name = column.name()
            if name in seen or (name == right_key and left_key == right_key):
                continue
            if name in left_names:
                name += suffix
            if name in names:
                raise Error("join_asof output name collision: " + name)
            names[name] = True
            columns.append(column.take_or_null(rows).renamed(name))
        return Self(columns^, height=self.height())

    def join_asof(
        self,
        right: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: List[String] = List[String](),
        strategy: String = "backward",
        tolerance: Int,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            right,
            on,
            left_on=left_on,
            right_on=right_on,
            by=by,
            strategy=strategy,
            tolerance=Optional(AnyValue(Int64(tolerance))),
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        right: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: List[String] = List[String](),
        strategy: String = "backward",
        tolerance: Float64,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            right,
            on,
            left_on=left_on,
            right_on=right_on,
            by=by,
            strategy=strategy,
            tolerance=Optional(AnyValue(tolerance)),
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        right: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: String,
        strategy: String = "backward",
        tolerance: Optional[AnyValue] = None,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            right,
            on,
            left_on=left_on,
            right_on=right_on,
            by=[by],
            strategy=strategy,
            tolerance=tolerance,
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        right: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: String,
        strategy: String = "backward",
        tolerance: Int,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            right,
            on,
            left_on=left_on,
            right_on=right_on,
            by=[by],
            strategy=strategy,
            tolerance=tolerance,
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        right: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: String,
        strategy: String = "backward",
        tolerance: Float64,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            right,
            on,
            left_on=left_on,
            right_on=right_on,
            by=[by],
            strategy=strategy,
            tolerance=tolerance,
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        right: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: NoneType,
        strategy: String = "backward",
        tolerance: Optional[AnyValue] = None,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            right,
            on,
            left_on=left_on,
            right_on=right_on,
            by=List[String](),
            strategy=strategy,
            tolerance=tolerance,
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        right: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: NoneType,
        strategy: String = "backward",
        tolerance: Int,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            right,
            on,
            left_on=left_on,
            right_on=right_on,
            by=List[String](),
            strategy=strategy,
            tolerance=tolerance,
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join_asof(
        self,
        right: Self,
        on: String = "",
        *,
        left_on: String = "",
        right_on: String = "",
        by: NoneType,
        strategy: String = "backward",
        tolerance: Float64,
        suffix: String = "_right",
        allow_exact_matches: Bool = True,
    ) raises -> Self:
        return self.join_asof(
            right,
            on,
            left_on=left_on,
            right_on=right_on,
            by=List[String](),
            strategy=strategy,
            tolerance=tolerance,
            suffix=suffix,
            allow_exact_matches=allow_exact_matches,
        )

    def join(
        self,
        right: Self,
        on: String,
        how: String = "inner",
        suffix: String = "_right",
        coalesce: Bool = True,
    ) raises -> Self:
        return self.join(
            right,
            left_on=[on],
            right_on=[on],
            how=how,
            suffix=suffix,
            coalesce=coalesce,
        )

    def join(
        self,
        right: Self,
        on: List[String],
        how: String = "inner",
        suffix: String = "_right",
        coalesce: Bool = True,
    ) raises -> Self:
        return self.join(
            right,
            left_on=on,
            right_on=on,
            how=how,
            suffix=suffix,
            coalesce=coalesce,
        )

    def join(
        self,
        right: Self,
        *,
        left_on: List[String],
        right_on: List[String],
        how: String = "inner",
        suffix: String = "_right",
        coalesce: Bool = True,
    ) raises -> Self:
        """Hash join on key columns of any dtype; null keys never match.

        how: inner, left, right, full, semi, anti. Row order: inner, left,
        semi, and anti follow left rows (each left row's matches in right
        order); right follows right rows (matches in left order); full is the
        left join followed by unmatched right rows in right order.
        Output: left columns, then right non-key columns, with suffix on
        names that collide with left names. Key columns keep left names and
        take whichever side is present. With coalesce=False (inner, left,
        right and full joins), the right keys are kept as columns of their
        own instead, after the left columns and suffixed where their names
        collide, and the left keys keep the left side's values: a left
        join's unmatched rows then have a null right key.
        Semi and anti joins output left columns only, so the flag does not
        apply to them.
        """
        return self._join_impl(
            right,
            left_on=left_on,
            right_on=right_on,
            how=join_type(how),
            suffix=suffix,
            coalesce=coalesce,
        )

    def _join_impl(
        self,
        right: Self,
        *,
        left_on: List[String],
        right_on: List[String],
        how: Int = JOIN_INNER,
        suffix: String = "_right",
        coalesce: Bool = True,
        prepared: Optional[PreparedHashIndex] = None,
        range_filtered: Bool = False,
        keep_order: Bool = True,
        workers: Int = 0,
        marks: Int = 0,
    ) raises -> Self:
        """`keep_order=False` lets an inner join emit its pairs in whatever
        order its probe produced them, for a caller whose consumers ignore
        row order (a lazy plan feeding an aggregation). A right join
        probing a `prepared` index emits only its matched pairs, and sets
        the byte at `marks` (the address of one byte per build row, when
        not 0) of every build row it matched: the caller emits the rest."""
        if how == JOIN_CROSS:
            raise Error(
                "A cross join takes no keys; use join(right, how='cross')"
            )
        if len(left_on) == 0 or len(left_on) != len(right_on):
            raise Error(
                "Join requires the same nonzero number of left and right keys"
            )
        return join_frames(
            self,
            right,
            left_on=left_on,
            right_on=right_on,
            how=how,
            suffix=suffix,
            coalesce=coalesce,
            prepared=prepared,
            range_filtered=range_filtered,
            keep_order=keep_order,
            workers=workers,
            marks=marks,
        )

    def join(
        self, right: Self, *, how: String, suffix: String = "_right"
    ) raises -> Self:
        """Cross join: every left row paired with every right row, left-major.

        Right names that collide with left names gain the suffix.
        """
        if join_code(how) != JOIN_CROSS:
            raise Error("Join how='" + how + "' requires key columns")
        var total = self._height * right._height
        if right._height != 0 and total // right._height != self._height:
            raise Error("Cross join row count overflows")
        var names = Dict[String, Bool]()
        for column in self._columns:
            names[column.name()] = True
        var left_names = names.copy()
        var right_names = List[String]()
        for column in right._columns:
            var name = column.name()
            if name in left_names:
                name += suffix
            if name in names:
                raise Error("Join output name collision: " + name)
            names[name] = True
            right_names.append(name)
        return cross_join_frames(self, right, total, right_names)

    def select(
        self, expression: Expr, *, batch_size: Int = 8192
    ) raises -> Self:
        return self.select_exprs([expression.copy()], batch_size=batch_size)

    def select_exprs(
        self, expressions: List[Expr], *, batch_size: Int = 8192
    ) raises -> Self:
        """Evaluate against the original frame. Scalar-only output has one row.

        Rows are evaluated in batches of `batch_size`. The default, as for
        with_columns, amortizes each expression node's fixed per-batch cost:
        with bitmap comparisons (#327) that cost dominated at 1024 rows.

        If any expression is row-valued, scalar results broadcast to height,
        including zero rows. An empty selection preserves the input height.
        """
        var bound = _bind_all(expressions, self._columns)
        var has_rows = len(expressions) == 0
        for expression in bound:
            has_rows = has_rows or expression.shape() == ROWS
        var height = self._height if has_rows else 1
        var columns = List[Series]()
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        # A top-level mode() yields one row per mode, as in Polars; scalar
        # siblings repeat, and row-valued siblings cannot align with it.
        var modes = List[String]()
        for expression in bound:
            ref nodes = expression.expr._nodes
            var top = nodes[len(nodes) - 1].op
            if top == MODE or top == VALUE_COUNTS:
                if has_rows:
                    raise Error(
                        "mode() and value_counts() cannot be selected with"
                        " row-valued expressions; their length differs"
                    )
                modes.append(expression.expr._name)
        # Window partitions laid out once per key set for all expressions,
        # as in with_columns.
        var layouts = WindowLayouts()
        for expression in bound:
            var result = evaluate(
                expression,
                self._columns,
                self._height,
                batch_size=batch_size,
                layouts=Int(Pointer(to=layouts)),
            )
            columns.append(result^)
        _ = layouts^
        var row_height = -1
        for k in range(len(bound)):
            if bound[k].shape() == ROWS:
                if row_height >= 0 and len(columns[k]) != row_height:
                    raise Error("Selected expressions have different lengths")
                row_height = len(columns[k])
        if row_height >= 0:
            height = row_height
        for k in range(len(bound)):
            if bound[k].shape() != ROWS and has_rows:
                columns[k] = columns[k]._broadcast(height)
        var frame = Self(columns^, height=height)
        if len(modes) > 0:
            # An empty input has no modes: zero rows, not one null row.
            var empty = frame.select(col(modes[0]).list().len()).item()
            if empty.int64() == 0:
                return frame.explode(modes).clear()
            return frame.explode(modes)
        return frame^

    def with_columns(
        self, expression: Expr, *, batch_size: Int = 8192
    ) raises -> Self:
        return self.with_columns([expression.copy()], batch_size=batch_size)

    def with_columns(
        self, expressions: List[Expr], *, batch_size: Int = 8192
    ) raises -> Self:
        """All siblings see the original schema and data; aliases are outputs.

        The bounded default batch amortizes fused expression setup on large
        inputs. Callers can still choose a smaller batch_size explicitly.
        """
        var bound = _bind_all(expressions, self._columns)
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        var columns = self._columns.copy()
        # Window partitions laid out once per key set for all expressions.
        var layouts = WindowLayouts()
        for expression in bound:
            var result = evaluate(
                expression,
                self._columns,
                self._height,
                batch_size=batch_size,
                layouts=Int(Pointer(to=layouts)),
            )
            if expression.shape() != ROWS:
                result = result._broadcast(self._height)
            var replacement = -1
            for i in range(len(columns)):
                if columns[i].name() == result.name():
                    replacement = i
                    break
            if replacement < 0:
                columns.append(result^)
            else:
                columns[replacement] = result^
        _ = layouts^
        return Self(columns^, height=self._height)

    def filter(
        self,
        predicate: Expr,
        *,
        batch_size: Int = 8192,
        keep: List[String] = List[String](),
    ) raises -> Self:
        """The rows `predicate` keeps. Given `keep`, only those columns are
        returned: the predicate still reads any column, but columns only it
        reads are never copied (a lazy filter whose later steps read fewer
        columns than it does)."""
        # A literal true keeps every row as it is (a filter whose parts an
        # ordered column answered leaves one behind).
        if (
            len(predicate._nodes) == 1
            and predicate._nodes[0].op == LIT_BOOL
            and predicate._nodes[0].integer != 0
        ):
            return self._kept(keep)
        var predicates = expand(predicate, self._columns)
        if len(predicates) != 1:
            raise Error("A filter selector must match exactly one column")
        var bound = bind(predicates[0], self._columns)
        if bound.dtypes[len(bound.dtypes) - 1] != DataType.BOOL:
            raise Error("Filter expression must return Boolean values")
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        # Parts comparing a column stored in ascending order with a constant
        # keep one run of rows: two binary searches and a zero-copy slice
        # (ClickBench q36-q42 filter CounterID and EventDate, the order the
        # hits table is stored in).
        if self._height > 0 and _row_local(predicates):
            var parts = conjuncts(predicates[0])
            var window = sorted_window(self._columns, self._height, parts)
            var any = False
            for answered in window.used:
                any = any or answered
            if any:
                trace_path("filter.sorted_range")
                var narrowed = self.slice(window.low, window.high - window.low)
                var rest = List[Expr]()
                for k in range(len(parts)):
                    if not window.used[k]:
                        rest.append(parts[k].copy())
                if len(rest) == 0 or narrowed.height() == 0:
                    return narrowed._kept(keep)
                var remaining = rest[0].copy()
                for k in range(1, len(rest)):
                    remaining = remaining & rest[k]
                return narrowed.filter(
                    remaining, batch_size=batch_size, keep=keep
                )
        # Each partition's first k rows by ordinal rank: no rank computed.
        var top = top_k_mask(bound, self._columns, self._height)
        if top:
            trace_path("filter.top_k_per_partition")
            return self._kept(keep).filter(top.value())
        if len(bound.expr._nodes) == 3:
            ref node = bound.expr._nodes[2]
            if (
                (
                    node.op == GT
                    or node.op == LT
                    or node.op == GE
                    or node.op == LE
                    or node.op == EQ
                    or node.op == NE
                )
                and bound.expr._nodes[node.left].op == COL
                and bound.expr._nodes[node.right].op == LIT_FLOAT
                and self._columns[bound.sources[node.left]].dtype()
                == DataType.FLOAT64
            ):
                # Aligned chunks take the per-chunk path below instead.
                if not can_filter_aligned_chunks(self._columns):
                    return self._kept(keep)._filter_rows(
                        float_compare_rows(
                            self._columns[bound.sources[node.left]],
                            node.op,
                            bound.expr._nodes[node.right].floating,
                        )
                    )
        # Columns chunked at the same rows (Parquet row groups, a
        # partitioned group-by's buckets) filter chunk by chunk in parallel,
        # each chunk taking whichever path suits it, with no merge first.
        if (
            bound.shape() == ROWS
            and _row_local(predicates)
            and can_filter_aligned_chunks(self._columns)
        ):
            trace_path("filter.aligned_chunks")
            return self._filter_chunks(predicates[0], batch_size, keep)
        var parts = conjuncts(predicates[0])
        if (
            len(parts) > 1
            and self._height >= SELECTIVE_FILTER_ROWS
            and _row_local(parts)
        ):
            trace_path("filter.selective_and")
            return self._filter_selective(parts, batch_size, keep)
        var mask = self._predicate_mask(bound, batch_size)
        return self._kept(keep).filter(mask)

    def _selection(
        self, predicate: Expr, batch_size: Int, source_rows: Int = 0
    ) raises -> BoolColumn:
        """The rows `predicate` keeps, as a mask over this frame, chosen the
        way `filter` chooses its path: a run answered by a column stored in
        ascending order, a top-k per partition, selective conjuncts (the
        cheap parts first, strings only on the rows they kept, fused
        comparisons), or the predicate's own mask. A pipeline's morsel
        narrows its selection with this and copies nothing; it passes its
        source's height as `source_rows`, so the route is the one the
        whole source would take, however finely it is cut."""
        if (
            len(predicate._nodes) == 1
            and predicate._nodes[0].op == LIT_BOOL
            and predicate._nodes[0].integer != 0
        ):
            return _rows_mask(List[Int](), self._height, every=True)
        var predicates = expand(predicate, self._columns)
        if len(predicates) != 1:
            raise Error("A filter selector must match exactly one column")
        var bound = bind(predicates[0], self._columns)
        if bound.dtypes[len(bound.dtypes) - 1] != DataType.BOOL:
            raise Error("Filter expression must return Boolean values")
        if self._height > 0 and _row_local(predicates):
            var parts = conjuncts(predicates[0])
            var window = sorted_window(self._columns, self._height, parts)
            var any = False
            for answered in window.used:
                any = any or answered
            if any:
                trace_path("filter.sorted_range")
                var low = window.low
                var length = window.high - window.low
                var rest = List[Expr]()
                for k in range(len(parts)):
                    if not window.used[k]:
                        rest.append(parts[k].copy())
                if len(rest) == 0 or length == 0:
                    return _window_mask(low, length, self._height)
                var remaining = rest[0].copy()
                for k in range(1, len(rest)):
                    remaining = remaining & rest[k]
                var inner = self.slice(low, length)._selection(
                    remaining, batch_size, source_rows
                )
                return _placed_mask(inner, low, self._height)
        var top = top_k_mask(bound, self._columns, self._height)
        if top:
            trace_path("filter.top_k_per_partition")
            return top.take()
        var parts = conjuncts(predicates[0])
        if (
            len(parts) > 1
            and max(self._height, source_rows) >= SELECTIVE_FILTER_ROWS
            and _row_local(parts)
        ):
            trace_path("filter.selective_and")
            return self._selective_mask(parts, batch_size)
        return self._predicate_mask(bound, batch_size)

    def _kept(self, keep: List[String]) raises -> Self:
        """This frame, or only the `keep` columns when given."""
        if len(keep) == 0:
            return self.copy()
        return self.select(keep)

    def _filter_chunks(
        self, predicate: Expr, batch_size: Int, keep: List[String]
    ) raises -> Self:
        """Filter each aligned chunk on its own, in parallel, and keep the
        surviving pieces as the output's chunks. The predicate must be
        row-local: a chunk sees only its own rows."""
        var chunks = self._columns[0].n_chunks()
        var jobs = List[_ChunkFilterJob](capacity=chunks)
        for i in range(chunks):
            var parts = List[Series](capacity=len(self._columns))
            for column in self._columns:
                parts.append(
                    Series(
                        column.name(),
                        column._chunked.value()[].arrays[i].copy(),
                        column.dtype(),
                    )
                )
            jobs.append(
                _ChunkFilterJob(Self(parts^), predicate, batch_size, keep)
            )
        run_jobs(jobs)
        var shape = self._kept(keep)
        var output = List[Series](capacity=len(shape._columns))
        for c in range(len(shape._columns)):
            var pieces = List[Series]()
            for i in range(len(jobs)):
                if jobs[i].result.height() > 0:
                    pieces.append(jobs[i].result._columns[c].copy())
            if len(pieces) == 0:
                output.append(shape._columns[c].slice(0, 0))
            elif len(pieces) == 1:
                output.append(pieces[0].copy())
            else:
                output.append(Series._from_chunks(pieces))
        return Self(output^)

    def _predicate_mask(
        self, bound: BoundExpr, batch_size: Int
    ) raises -> BoolColumn:
        var result = evaluate(
            bound, self._columns, self._height, batch_size=batch_size
        )
        if bound.shape() != ROWS:
            result = result._broadcast(self._height)
        return result.bool()

    def _filter_selective(
        self, parts: List[Expr], batch_size: Int, keep: List[String]
    ) raises -> Self:
        """Filter by the AND of row-local `parts` (`_selective_mask`)."""
        return self._kept(keep).filter(self._selective_mask(parts, batch_size))

    def _selective_mask(
        self, parts: List[Expr], batch_size: Int
    ) raises -> BoolColumn:
        """The mask of the AND of row-local `parts`, each evaluated only on
        rows kept by earlier parts: ClickBench q37 compared all 10M titles
        with "" although its counter and date parts keep few rows.

        Parts reading only numbers run before parts reading strings, and
        otherwise in written order. Narrowing gathers the columns later
        parts read, which pays when a part keeps at most an eighth of its
        rows, or half when a string comparison is still to come. Past that
        the remaining parts run together on the same rows and the masks
        combine: PDS-H q6's year of lineitem dates keeps a seventh, and its
        cheap numeric parts were 36% slower on gathered rows.

        Parts comparing a numeric column with a constant go first, all of
        them in one pass (`fused_comparisons`), as if they were one part.
        When every part still to run is an AND/OR of such comparisons and
        null tests (`selection_tree`), those parts read the source columns
        at the kept rows instead of a gathered copy.
        """
        var fused = fused_comparisons(self._columns, self._height, parts)
        var order = List[Int]()
        for late in [False, True]:
            for k in range(len(parts)):
                if fused.used[k]:
                    continue
                if _reads_strings(parts[k], self) == late:
                    order.append(k)
        var current = self.copy()
        var rows = List[Int]()
        var whole = True
        var k = 0
        var pending = len(fused.mask) > 0
        if pending:
            trace_path("filter.fused_compare")
        while pending or k < len(order):
            var mask: BoolColumn
            if pending:
                mask = fused_mask(fused.mask, self._height)
                pending = False
            else:
                mask = current._predicate_mask(
                    bind(parts[order[k]], current._columns), batch_size
                )
                k += 1
            if whole and k == len(order):
                return mask^
            if k < len(order):
                var strings = _reads_strings(parts[order[len(order) - 1]], self)
                var limit = current._height // (2 if strings else 8)
                if true_count(mask) > limit:
                    var rest = parts[order[k]].copy()
                    for j in range(k + 1, len(order)):
                        rest = rest & parts[order[j]]
                    var other = current._predicate_mask(
                        bind(rest, current._columns), batch_size
                    )
                    var both = both_true(mask, other)
                    if whole:
                        return both^
                    mask = both^
                    k = len(order)
            var kept = true_rows(mask)
            if not whole:
                for i in range(len(kept)):
                    kept[i] = rows[kept[i]]
            if k < len(order):
                # Later parts made of comparisons and null tests narrow the
                # kept rows in place, reading the source columns at those
                # rows (DuckDB's selection vectors): nothing is gathered.
                var trees = List[SelectionTree]()
                for j in range(k, len(order)):
                    var tree = selection_tree(parts[order[j]], self._columns)
                    if not tree:
                        break
                    trees.append(tree.take())
                if len(trees) == len(order) - k:
                    trace_path("filter.selection_tree")
                    for t in range(len(trees)):
                        kept = trees[t].select(kept)
                    return _rows_mask(kept, self._height)
                # Only the columns later parts read are carried forward.
                var names = List[String]()
                for j in range(k, len(order)):
                    for name in _column_names(parts[order[j]]):
                        if name not in names:
                            names.append(name)
                if len(names) == 0:
                    names.append(self._columns[0].name())
                current = self.select(names)._filter_rows(kept.copy())
            rows = kept^
            whole = False
        return _rows_mask(rows, self._height)

    def unpivot(
        self,
        on: List[String] = List[String](),
        index: List[String] = List[String](),
        *,
        variable_name: String = "variable",
        value_name: String = "value",
    ) raises -> Self:
        """Wide to long: one row per (input row, `on` column).

        `on` defaults to every non-index column; all `on` columns must share
        one dtype. Output rows follow `on` order, then input row order.
        """
        var index_set = Dict[String, Bool]()
        for name in index:
            _ = self._index(name)
            if name in index_set:
                raise Error("Column listed twice in index: " + name)
            index_set[name] = True
        var columns_on = on.copy()
        if len(columns_on) == 0:
            for column in self._columns:
                if column.name() not in index_set:
                    columns_on.append(column.name())
        if len(columns_on) == 0:
            raise Error("unpivot requires at least one value column")
        var dtype = self._columns[self._index(columns_on[0])].dtype()
        for name in columns_on:
            if name in index_set:
                raise Error("Column is both index and on: " + name)
            var actual = self._columns[self._index(name)].dtype()
            if actual != dtype:
                raise Error(
                    "unpivot columns must share one dtype; "
                    + name
                    + " is "
                    + actual.name()
                    + ", expected "
                    + dtype.name()
                )
        if (
            variable_name == value_name
            or variable_name in index_set
            or value_name in index_set
        ):
            raise Error(
                "unpivot output names must be distinct from each other and the"
                " index"
            )
        var repeated = List[Int](capacity=self._height * len(columns_on))
        for _ in range(len(columns_on)):
            for row in range(self._height):
                repeated.append(row)
        var output = List[Series]()
        for name in index:
            output.append(self._columns[self._index(name)].take(repeated))
        var labels = List[String](capacity=len(repeated))
        for name in columns_on:
            for _ in range(self._height):
                labels.append(name)
        output.append(Series(variable_name, StringColumn(labels)))
        var values = self._columns[self._index(columns_on[0])].renamed(
            value_name
        )
        for k in range(1, len(columns_on)):
            values._append_series(self._columns[self._index(columns_on[k])])
        output.append(values^)
        return Self(output^, height=len(repeated))

    def pivot(
        self,
        on: String,
        *,
        index: List[String],
        values: String,
        aggregate_function: String = "",
        sort_columns: Bool = False,
        batch_size: Int = 1024,
    ) raises -> Self:
        """Long to wide: one row per distinct index key, one column per
        distinct `on` value (first-occurrence order unless sort_columns).

        Without aggregate_function, each (index, on) cell must hold at most
        one row. Otherwise it is one of first, last, sum, mean, min, max,
        count, len, or median. Missing cells are null. New column names are
        the `on` values' text, with null rendered as "null".
        """
        var agg = _pivot_aggregate(values, aggregate_function)
        var seen = Dict[String, Bool]()
        for name in index:
            _ = self._index(name)
            if name in seen:
                raise Error("Column listed twice in index: " + name)
            seen[name] = True
        if on in seen or values in seen:
            raise Error("pivot on/values columns cannot also be index columns")
        var on_column = self._columns[self._index(on)].copy()
        _ = self._index(values)
        var bound = bind(agg, self._columns)
        # Row keys, column keys, and (row, column) cells.
        var row_ids = List[Int](length=self._height, fill=0)
        var row_reps = List[Int]()
        if len(index) > 0:
            var keys = List[Series]()
            for name in index:
                keys.append(self._columns[self._index(name)].copy())
            var encoded = encode_rows(keys, nulls_equal=True)
            row_ids = encoded.ids.copy()
            row_reps = encoded.representatives.copy()
        elif self._height > 0:
            row_reps.append(0)
        var column_keys = encode_rows([on_column.copy()], nulls_equal=True)
        var cell_keys = List[Series]()
        cell_keys.append(Series("row", Column[Int64](_as_int64(row_ids))))
        cell_keys.append(
            Series("column", Column[Int64](_as_int64(column_keys.ids)))
        )
        var cells = encode_rows(cell_keys, nulls_equal=True)
        if aggregate_function.byte_length() == 0:
            var counts = List[Int](length=cells.count(), fill=0)
            for id in cells.ids:
                counts[id] += 1
            for c in counts:
                if c > 1:
                    raise Error(
                        "pivot found several rows for one cell; pass an"
                        " aggregate_function"
                    )
        var aggregated = evaluate(
            bound,
            self._columns,
            self._height,
            batch_size=batch_size,
            grouped=True,
            groups=cells.ids,
            group_count=cells.count(),
        )
        var n_rows = len(row_reps)
        var n_cols = column_keys.count()
        var matrix = List[Int](length=n_rows * n_cols, fill=-1)
        for c in range(cells.count()):
            var rep = cells.representatives[c]
            matrix[row_ids[rep] * n_cols + column_keys.ids[rep]] = c
        var order = List[Int]()
        for k in range(n_cols):
            order.append(k)
        if sort_columns:
            order = on_column.take(column_keys.representatives).argsort()
        var output = List[Series]()
        var names = Dict[String, Bool]()
        for name in index:
            output.append(self._columns[self._index(name)].take(row_reps))
            names[name] = True
        for k in order:
            var label = String(on_column.get(column_keys.representatives[k]))
            if label in names:
                raise Error("pivot column name collides: " + label)
            names[label] = True
            var picks = List[Int](capacity=n_rows)
            for r in range(n_rows):
                picks.append(matrix[r * n_cols + k])
            output.append(aggregated.take_or_null(picks).renamed(label))
        return Self(output^, height=n_rows)

    def cast(
        self, dtypes: Dict[String, String], *, strict: Bool = True
    ) raises -> Self:
        """Cast named columns in place of the originals; order is kept."""
        var casts = List[Expr]()
        for item in dtypes.items():
            _ = self._index(item.key)
            casts.append(col(item.key).cast(item.value, strict))
        return self.with_columns(casts)

    def _subset_keys(self, subset: List[String]) raises -> List[Series]:
        var names = subset.copy() if len(subset) > 0 else self.columns()
        if len(names) == 0:
            raise Error("Row uniqueness requires at least one column")
        var seen = Dict[String, Bool]()
        var keys = List[Series](capacity=len(names))
        for name in names:
            if name in seen:
                raise Error("Column listed twice in subset: " + name)
            seen[name] = True
            keys.append(self._columns[self._index(name)].copy())
        return _expand_struct_keys(keys)

    def _subset_row_keys(self, subset: List[String]) raises -> RowKeys:
        """Ids for subset's distinct rows, with each id's representative its
        first row. The numbering is first-occurrence on one worker; across
        workers a high-cardinality key encodes one hash bucket at a time,
        so ids are in no particular order."""
        var keys = self._subset_keys(subset)
        var workers = worker_count(self._height)
        if workers > 1 and not low_cardinality(keys):
            # The bucket scatter is stable, so a bucket's first row for a key
            # is the key's first row in the frame.
            return encode_partitioned(keys, workers, nulls_equal=True)
        return encode_rows(keys, nulls_equal=True)

    def _key_counts(
        self, subset: List[String]
    ) raises -> Tuple[List[Int], List[Int]]:
        var keys = self._subset_row_keys(subset)
        var counts = List[Int](length=keys.count(), fill=0)
        for id in keys.ids:
            counts[id] += 1
        return (keys.ids.copy(), counts^)

    def unique(
        self,
        subset: List[String] = List[String](),
        *,
        keep: String = "any",
        maintain_order: Bool = False,
    ) raises -> Self:
        """Drop duplicate rows compared on subset (default: every column).

        keep: any or first (first occurrence), last, or none (drop every
        duplicated row). Nulls equal nulls; NaN equals NaN; -0.0 equals 0.0.
        Output order is unspecified unless maintain_order=True, which keeps
        surviving rows in input order.
        """
        if (
            keep != "any"
            and keep != "first"
            and keep != "last"
            and keep != "none"
        ):
            raise Error("keep must be 'any', 'first', 'last', or 'none'")
        var whole_rows = len(subset) == 0
        if not whole_rows:
            whole_rows = True
            for name in self.columns():
                whole_rows = whole_rows and name in subset
        if (
            keep == "any"
            and not maintain_order
            and whole_rows
            and worker_count(self._height) > 1
        ):
            # Distinct whole rows in any order are a group_by on every
            # column with no aggregates, which encodes on every worker
            # (307K rows, 23K distinct: 3.5 ms against 10 through one
            # dictionary; TPC-DS q35 takes three such uniques).
            return self.group_by(self.columns()).agg(List[Expr]())
        if keep == "any" or keep == "first":
            # Each key's first row is its id's representative. One worker
            # numbers ids by first occurrence, so those rows are already in
            # input order; bucketed ids are not, and the rows are put back
            # in input order by marking them. No counts or second pass over
            # the ids (TPC-DS q35: three 80K-307K-row uniques).
            var encoded = self._subset_row_keys(subset)
            ref firsts = encoded.representatives
            var ordered = True
            for g in range(1, len(firsts)):
                if firsts[g] < firsts[g - 1]:
                    ordered = False
                    break
            if ordered:
                return self.take(firsts)
            var chosen = List[Bool](length=self._height, fill=False)
            for row in firsts:
                chosen[row] = True
            var rows = List[Int](capacity=len(firsts))
            for i in range(self._height):
                if chosen[i]:
                    rows.append(i)
            return self.take(rows)
        var keyed = self._key_counts(subset)
        ref ids = keyed[0]
        ref counts = keyed[1]
        var rows = List[Int]()
        if keep == "none":
            for i in range(len(ids)):
                if counts[ids[i]] == 1:
                    rows.append(i)
        elif keep == "last":
            var last = List[Int](length=len(counts), fill=-1)
            for i in range(len(ids)):
                last[ids[i]] = i
            var chosen = List[Bool](length=len(ids), fill=False)
            for row in last:
                chosen[row] = True
            for i in range(len(ids)):
                if chosen[i]:
                    rows.append(i)
        else:
            var seen = List[Bool](length=len(counts), fill=False)
            for i in range(len(ids)):
                if not seen[ids[i]]:
                    seen[ids[i]] = True
                    rows.append(i)
        return self.take(rows)

    def n_unique(self, subset: List[String] = List[String]()) raises -> Int:
        """Number of distinct rows on subset (default: every column)."""
        if self._height == 0:
            return 0
        return len(self._key_counts(subset)[1])

    def is_duplicated(
        self, subset: List[String] = List[String]()
    ) raises -> Series:
        """True for every row whose key occurs more than once."""
        var keyed = self._key_counts(subset)
        var flags = List[Bool](capacity=self._height)
        for id in keyed[0]:
            flags.append(keyed[1][id] > 1)
        return Series("is_duplicated", BoolColumn(flags))

    def is_unique(self, subset: List[String] = List[String]()) raises -> Series:
        """True for every row whose key occurs exactly once."""
        var keyed = self._key_counts(subset)
        var flags = List[Bool](capacity=self._height)
        for id in keyed[0]:
            flags.append(keyed[1][id] == 1)
        return Series("is_unique", BoolColumn(flags))

    def drop_nulls(self, subset: List[String] = List[String]()) raises -> Self:
        """Keep rows with no null in subset (default: every column)."""
        var names = subset.copy() if len(subset) > 0 else self.columns()
        var keep = List[Bool](length=self._height, fill=True)
        for name in names:
            ref column = self._columns[self._index(name)]
            var valid = validity(column)
            for i in range(self._height):
                keep[i] = keep[i] and valid[i]
        var rows = List[Int]()
        for i in range(self._height):
            if keep[i]:
                rows.append(i)
        return self.take(rows)

    def fill_null(
        self, value: String, subset: List[String] = List[String]()
    ) raises -> Self:
        """Fill nulls in string columns with a string."""
        return self.fill_null(lit(value), subset)

    def fill_null(
        self, value: Expr, subset: List[String] = List[String]()
    ) raises -> Self:
        """Fill nulls with a scalar value.

        Without subset, only columns whose dtype matches the value change.
        Every listed subset column must match the value's dtype. A bare
        number (`fill_null(0)`) is untyped: it fills every column it can
        adopt (an integer: every numeric column; a float: every float
        column), range-checked per column.
        """
        return self.with_columns(self._fill_null_exprs(value, subset))

    def fill_null(
        self,
        *,
        strategy: String,
        limit: Optional[Int] = None,
        subset: List[String] = List[String](),
    ) raises -> Self:
        """Fill nulls with forward/backward, min/max/mean, zero, or one.

        limit bounds each forward/backward null run. Integer means truncate
        toward zero, preserving dtype. Columns with no nulls are unchanged;
        all-null columns stay null for forward/backward/min/max/mean.
        """
        var run_limit = self._fill_strategy_limit(strategy, limit, subset)
        var fills = List[Expr]()
        for column in self._columns:
            var name = column.name()
            if len(subset) > 0 and name not in subset:
                continue
            var nulls = column.null_count()
            if nulls == 0 or (
                nulls == len(column) and strategy not in [String("zero"), "one"]
            ):
                continue
            var dtype = column.dtype()
            var c = col(name)
            var expression: Expr
            if strategy == "forward":
                expression = c.forward_fill(run_limit)
            elif strategy == "backward":
                expression = c.backward_fill(run_limit)
            elif dtype.is_nested():
                raise Error(
                    "fill_null strategy "
                    + strategy
                    + " is not supported for "
                    + dtype.name()
                )
            elif strategy == "min" or strategy == "max":
                expression = c.fill_null(
                    c.min() if strategy == "min" else c.max()
                )
            elif strategy == "mean":
                if not dtype.is_numeric() and not dtype.is_temporal():
                    raise Error(
                        "fill_null mean is not supported for " + dtype.name()
                    )
                expression = c.fill_null(self._mean_fill(c, dtype))
            else:
                var value: Expr
                if dtype.physical() == DataType.STRING:
                    if strategy != "zero":
                        raise Error(
                            "fill_null one is not supported for " + dtype.name()
                        )
                    value = lit("").cast(dtype)
                elif (
                    dtype.is_numeric()
                    or dtype.is_temporal()
                    or dtype == DataType.BOOL
                ):
                    value = lit(Int64(1 if strategy == "one" else 0)).cast(
                        dtype
                    )
                else:
                    raise Error(
                        "fill_null "
                        + strategy
                        + " is not supported for "
                        + dtype.name()
                    )
                expression = c.fill_null(value)
            if dtype.is_categorical():
                expression = expression.cast(dtype)
            fills.append(expression.alias(name))
        return self.with_columns(fills)

    def _fill_strategy_limit(
        self, strategy: String, limit: Optional[Int], subset: List[String]
    ) raises -> Int:
        if strategy not in [
            String("forward"),
            "backward",
            "min",
            "max",
            "mean",
            "zero",
            "one",
        ]:
            raise Error("Unknown fill_null strategy: " + strategy)
        if limit and (strategy != "forward" and strategy != "backward"):
            raise Error("limit is only valid for forward/backward fill")
        var run_limit = limit.value() if limit else -1
        if run_limit < -1 or (limit and run_limit < 0):
            raise Error("fill limit must be nonnegative")
        var seen = Dict[String, Bool]()
        for name in subset:
            _ = self._index(name)
            if name in seen:
                raise Error("Column listed twice in subset: " + name)
            seen[name] = True
        return run_limit

    def _mean_fill(self, c: Expr, dtype: DataType) raises -> Expr:
        var work = c.cast(DataType.INT64) if dtype.is_temporal() else c.copy()
        var mean = lit(self.select(work.mean()).item().float64())
        if dtype.is_decimal():
            var scale = lit(Float64(pow10(dtype.scale())))
            return ((mean * scale).truncate() / scale).cast(dtype)
        var physical = DataType.INT64 if dtype.is_temporal() else dtype
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            comptime if D.is_integral():
                if physical == DataType.of(D):
                    var low = lit(Scalar[D].MIN)
                    var high = lit(Scalar[D].MAX)
                    var value = (
                        when(mean >= high.cast(DataType.FLOAT64))
                        .then(high)
                        .otherwise(
                            when(mean <= low.cast(DataType.FLOAT64))
                            .then(low)
                            .otherwise(mean.cast(physical))
                        )
                    )
                    return value.cast(dtype) if dtype.is_temporal() else value^
        return mean.cast(dtype)

    def fill_nan(self, value: Float64) raises -> Self:
        """Replace floating NaNs with a numeric scalar, preserving column dtypes."""
        return self.fill_nan(lit(value))

    def fill_nan(self, value: Expr) raises -> Self:
        """Replace NaNs in floating columns; nulls and other columns stay.

        value is cast to each floating dtype and may be a scalar or a
        row-valued expression.
        """
        return self.with_columns(self._fill_nan_exprs(value))

    def _fill_nan_exprs(self, value: Expr) -> List[Expr]:
        var expressions = List[Expr]()
        for column in self._columns:
            if column.dtype().is_float():
                expressions.append(
                    col(column.name())
                    .fill_nan(value.cast(column.dtype()))
                    .alias(column.name())
                )
        return expressions^

    def drop_nans(self, subset: List[String] = List[String]()) raises -> Self:
        """Remove rows with a NaN in subset's floating columns; keep nulls."""
        var predicate = self._drop_nan_predicate(subset)
        return self.filter(predicate.value()) if predicate else self.copy()

    def _drop_nan_predicate(
        self, subset: List[String]
    ) raises -> Optional[Expr]:
        for name in subset:
            _ = self._index(name)
        var predicate: Optional[Expr] = None
        for column in self._columns:
            if not column.dtype().is_float() or (
                len(subset) > 0 and column.name() not in subset
            ):
                continue
            var keep = col(column.name()).is_not_nan().fill_null(lit(True))
            predicate = predicate.value() & keep if predicate else keep^
        return predicate^

    def _fill_null_exprs(
        self, value: Expr, subset: List[String]
    ) raises -> List[Expr]:
        """`fill_null`'s per-column fills, chosen from the column names and
        types alone, so LazyFrame.fill_null makes the same choice from its
        input's schema."""
        var bound = bind(value, self._columns)
        if bound.shape() == ROWS:
            raise Error("fill_null on a dataframe requires a scalar value")
        var dtype = bound.dtypes[len(bound.dtypes) - 1]
        var names = List[String]()
        if _is_untyped(value):
            for column in self._columns:
                var name = column.name()
                if len(subset) > 0 and name not in subset:
                    continue
                if len(subset) > 0:
                    names.append(name)  # binding reports a failed adoption
                    continue
                try:
                    _ = bind(col(name).fill_null(value), self._columns)
                    names.append(name)
                except:
                    pass
        elif len(subset) > 0:
            for name in subset:
                var actual = self._columns[self._index(name)].dtype()
                if actual != dtype:
                    raise Error(
                        "fill_null value is "
                        + dtype.name()
                        + " but column "
                        + name
                        + " is "
                        + actual.name()
                    )
                names.append(name)
        else:
            for column in self._columns:
                if column.dtype() == dtype:
                    names.append(column.name())
        for name in subset:
            _ = self._index(name)  # unknown subset names raise
        var fills = List[Expr]()
        for name in names:
            fills.append(col(name).fill_null(value))
        return fills^

    def group_by_dynamic(
        self,
        index_column: String,
        every: String,
        period: Optional[String] = None,
        offset: Optional[String] = None,
        closed: String = "left",
        label: String = "left",
        group_by: List[String] = List[String](),
    ) raises -> TimeGroupBy:
        """Group temporal rows into calendar-aligned, possibly overlapping windows.

        The index need not be sorted: each group's rows are taken in stable
        index order. Groups appear in first-appearance order, and windows
        within a group in time order.
        """
        return TimeGroupBy(
            self.copy(),
            index_column,
            every,
            period.value() if period else every,
            offset.value() if offset else "0d",
            closed,
            label,
            group_by.copy(),
            False,
        )

    def rolling(
        self,
        index_column: String,
        period: String,
        offset: Optional[String] = None,
        closed: String = "right",
        group_by: List[String] = List[String](),
    ) raises -> TimeGroupBy:
        """Aggregate a duration window for each temporal input row.

        The index need not be sorted: windows are formed from each group's
        rows in stable index order, and output rows keep input row order
        within each group.
        """
        return TimeGroupBy(
            self.copy(),
            index_column,
            "",
            period,
            offset.value() if offset else negate_period(period),
            closed,
            "datapoint",
            group_by.copy(),
            True,
        )

    def group_by_dynamic(
        self,
        index_column: String,
        every: String,
        period: Optional[String] = None,
        offset: Optional[String] = None,
        closed: String = "left",
        label: String = "left",
        *,
        group_by: String,
    ) raises -> TimeGroupBy:
        return self.group_by_dynamic(
            index_column,
            every,
            period,
            offset,
            closed,
            label,
            List[String]([group_by]),
        )

    def rolling(
        self,
        index_column: String,
        period: String,
        offset: Optional[String] = None,
        closed: String = "right",
        *,
        group_by: String,
    ) raises -> TimeGroupBy:
        return self.rolling(
            index_column, period, offset, closed, List[String]([group_by])
        )

    def group_by_dynamic(
        self,
        index_column: String,
        every: String,
        period: Optional[String] = None,
        offset: Optional[String] = None,
        closed: String = "left",
        label: String = "left",
        *,
        group_by: NoneType,
    ) raises -> TimeGroupBy:
        return self.group_by_dynamic(
            index_column, every, period, offset, closed, label, List[String]()
        )

    def rolling(
        self,
        index_column: String,
        period: String,
        offset: Optional[String] = None,
        closed: String = "right",
        *,
        group_by: NoneType,
    ) raises -> TimeGroupBy:
        return self.rolling(
            index_column, period, offset, closed, List[String]()
        )

    def group_by(
        self, key: String, *, maintain_order: Bool = False
    ) raises -> GroupBy:
        return self.group_by([key], maintain_order=maintain_order)

    def group_by(
        self, keys: List[String], *, maintain_order: Bool = False
    ) raises -> GroupBy:
        """Group by one or more columns of any dtype.

        Group output order is unspecified unless maintain_order=True, which
        guarantees first-occurrence order. Null keys form their own groups.
        """
        if len(keys) == 0:
            raise Error("group_by requires at least one key")
        var seen = Dict[String, Bool]()
        var columns = List[Series](capacity=len(keys))
        for key in keys:
            if key in seen:
                raise Error("Duplicate group_by key: " + key)
            seen[key] = True
            columns.append(self._columns[self._index(key)].copy())
        return GroupBy(self.copy(), _group_keys(columns), maintain_order)

    def group_by_rollup(self, keys: List[String]) -> GroupingSets:
        """Group by the rollup sets (k1, k2, ...), (k1, ..., k(n-1)),
        ..., (k1), and (). Finish with `agg`; see GroupingSets for the
        output. Runs the lazy plan."""
        return GroupingSets(self.lazy().group_by_rollup(keys))

    def group_by_cube(self, keys: List[String]) -> GroupingSets:
        """Every subset of the keys
        as a grouping set. Finish it with `agg`."""
        return GroupingSets(self.lazy().group_by_cube(keys))

    def group_by_grouping_sets(self, sets: List[List[String]]) -> GroupingSets:
        """One grouping
        per listed set, an empty set being the grand total. Finish it with
        `agg`."""
        return GroupingSets(self.lazy().group_by_grouping_sets(sets))

    def group_indices(self, key: String) raises -> GroupIndices:
        return self.group_indices([key])

    def group_indices(self, keys: List[String]) raises -> GroupIndices:
        """Which rows belong to which group, without aggregating them.

        For callers that want each group's rows rather than one summary row
        per group: `take(groups.rows(g))` builds a sub-frame, and
        `representative(g)` says where to read that group's key values.
        Groups are numbered in first-occurrence order, and null keys form
        their own group, as in `group_by`.
        """
        if len(keys) == 0:
            raise Error("group_indices requires at least one key")
        var seen = Dict[String, Bool]()
        var columns = List[Series](capacity=len(keys))
        for key in keys:
            if key in seen:
                raise Error("Duplicate group_indices key: " + key)
            seen[key] = True
            columns.append(self._columns[self._index(key)].copy())
        var encoded = encode_rows(_group_keys(columns), True)
        return GroupIndices(encoded.ids.copy(), encoded.representatives.copy())

    def group_by(
        self,
        keys: List[Expr],
        *,
        maintain_order: Bool = False,
        batch_size: Int = 1024,
    ) raises -> GroupBy:
        """Group by computed keys; each key is evaluated once and named by
        its output name. Aggregations still see the original columns."""
        if len(keys) == 0:
            raise Error("group_by requires at least one key")
        var bound = _bind_all(keys, self._columns)
        var columns = List[Series](capacity=len(keys))
        for expression in bound:
            if expression.shape() == AGGREGATE:
                raise Error("group_by keys must not be aggregates")
            var result = evaluate(
                expression, self._columns, self._height, batch_size=batch_size
            )
            if expression.shape() != ROWS:
                result = result._broadcast(self._height)
            columns.append(result^)
        return GroupBy(self.copy(), _group_keys(columns), maintain_order)


def _pivot_aggregate(values: String, aggregate_function: String) raises -> Expr:
    """The reduction `pivot` applies to each cell's `values` (first when
    no aggregate_function is given, which then allows one row a cell)."""
    var value = col(values)
    var function = (
        aggregate_function if aggregate_function.byte_length() > 0 else "first"
    )
    if function == "first":
        return value.first()
    if function == "last":
        return value.last()
    if function == "sum":
        return value.sum()
    if function == "mean":
        return value.mean()
    if function == "min":
        return value.min()
    if function == "max":
        return value.max()
    if function == "count":
        return value.count()
    if function == "len":
        return value.len()
    if function == "median":
        return value.median()
    raise Error(
        "aggregate_function must be first, last, sum, mean, min, max,"
        " count, len, or median"
    )


def _as_int64(values: List[Int]) -> List[Int64]:
    var out = List[Int64](capacity=len(values))
    for v in values:
        out.append(Int64(v))
    return out^


# aligned path won from 100k up (2.1 vs 3.8 ms; 4.7 vs 7.9 ms at 1M). The
# old 2,000,000 was chosen without a sweep and cost 1M-row filters 1.7x.
struct _ChunkFilterJob(Job):
    """Filter one aligned chunk's frame (`DataFrame._filter_chunks`)."""

    var frame: DataFrame
    var predicate: Expr
    var batch_size: Int
    var keep: List[String]
    var result: DataFrame

    def __init__(
        out self,
        var frame: DataFrame,
        predicate: Expr,
        batch_size: Int,
        keep: List[String],
    ) raises:
        self.frame = frame^
        self.predicate = predicate.copy()
        self.batch_size = batch_size
        self.keep = keep.copy()
        self.result = DataFrame(List[Series]())

    def run(mut self) raises:
        self.result = self.frame.filter(
            self.predicate, batch_size=self.batch_size, keep=self.keep
        )


def _shared_dictionaries(
    frames: List[DataFrame], how: String
) raises -> List[DataFrame]:
    """The frames with each categorical column, matched by position
    (vertical) or name (diagonal), on the union of its dictionaries, so the
    codes of every piece mean the same values (#106). Frames whose
    dictionaries already match are returned as they are."""
    var out = frames.copy()
    if how != "vertical" and how != "diagonal":
        return out^
    var names = List[String]()
    for frame in frames:
        for column in frame._columns:
            if column.dtype().is_categorical() and column.name() not in names:
                names.append(column.name())
    for name in names:
        var dtype = Optional[DataType]()
        var differ = False
        for frame in frames:
            if name not in frame.columns():
                continue
            var d = frame.column(name).dtype()
            if not d.is_categorical() or not d.has_dictionary():
                continue
            if not dtype:
                dtype = d
            elif not dtype.value() == d:
                differ = True
        if not differ or not dtype:
            continue
        var merged = dtype.value().dictionary()[].copy_values()
        for frame in frames:
            if name in frame.columns():
                var d = frame.column(name).dtype()
                if d.is_categorical() and d.has_dictionary():
                    merged = union_of(merged, d.dictionary()[])
        var target = DataType.categorical(merged^)
        for f in range(len(out)):
            if name in out[f].columns():
                var i = out[f]._index(name)
                if out[f]._columns[i].dtype().is_categorical():
                    out[f]._columns[i] = recode(out[f]._columns[i], target)
    return out^


def concat(
    frames: List[DataFrame], how: String = "vertical"
) raises -> DataFrame:
    """Combine frames: 'vertical', 'diagonal', or 'horizontal'.

    Vertical requires identical names, order, and dtypes. Diagonal unions
    columns by name in first-seen order and fills missing columns with nulls;
    shared names must share a dtype. Horizontal requires equal heights and
    unique names. All inputs are validated before any column is built.
    Categorical columns with different dictionaries are first moved onto
    the union of them.
    """
    var shared = _shared_dictionaries(frames, how)
    if len(shared) == 0:
        raise Error("concat requires at least one dataframe")
    if how == "vertical":
        var first = shared[0].schema()
        var height = 0
        for f in range(len(shared)):
            var schema = shared[f].schema()
            if len(schema) != len(first):
                raise Error(
                    "concat vertical: frame "
                    + String(f)
                    + " has "
                    + String(len(schema))
                    + " columns, expected "
                    + String(len(first))
                )
            for c in range(len(first)):
                if (
                    schema[c].name != first[c].name
                    or schema[c].dtype != first[c].dtype
                ):
                    raise Error(
                        "concat vertical: frame "
                        + String(f)
                        + " column "
                        + String(c)
                        + " is "
                        + schema[c].name
                        + ":"
                        + schema[c].dtype.name()
                        + ", expected "
                        + first[c].name
                        + ":"
                        + first[c].dtype.name()
                    )
            height += shared[f].height()
        # Polars accumulate_dataframes_vertical / vstack_mut_owned:
        # append Arrow array references; keep the output multi-chunk.
        var columns = List[Series](capacity=len(first))
        for c in range(len(first)):
            var parts = List[Series](capacity=len(shared))
            for frame in shared:
                parts.append(frame._columns[c].copy())
            columns.append(Series._from_chunks(parts))
        return DataFrame(columns^, height=height)
    if how == "diagonal":
        var names = List[String]()
        var dtypes = Dict[String, DataType]()
        for f in range(len(shared)):
            for field in shared[f].schema():
                if field.name not in dtypes:
                    dtypes[field.name] = field.dtype
                    names.append(field.name)
                elif dtypes[field.name] != field.dtype:
                    raise Error(
                        "concat diagonal: column "
                        + field.name
                        + " is "
                        + field.dtype.name()
                        + " in frame "
                        + String(f)
                        + ", expected "
                        + dtypes[field.name].name()
                    )
        var aligned = List[DataFrame](capacity=len(shared))
        for frame in shared:
            var columns = List[Series](capacity=len(names))
            for name in names:
                var found = -1
                for i in range(frame.width()):
                    if frame._columns[i].name() == name:
                        found = i
                        break
                if found >= 0:
                    columns.append(frame._columns[found].copy())
                else:
                    var missing = Series.full_null(
                        name, dtypes[name], frame.height()
                    )
                    for source in shared:
                        if name in source.columns():
                            missing._field_metadata = source.column(
                                name
                            )._field_metadata
                            break
                    columns.append(missing^)

            aligned.append(DataFrame(columns^, height=frame.height()))
        return concat(aligned, "vertical")
    if how == "horizontal":
        var height = shared[0].height()
        var seen = Dict[String, Bool]()
        for f in range(len(shared)):
            if shared[f].height() != height:
                raise Error(
                    "concat horizontal: frame "
                    + String(f)
                    + " has height "
                    + String(shared[f].height())
                    + ", expected "
                    + String(height)
                )
            for column in shared[f]._columns:
                if column.name() in seen:
                    raise Error(
                        "concat horizontal: duplicate column "
                        + column.name()
                        + " in frame "
                        + String(f)
                    )
                seen[column.name()] = True
        var columns = List[Series]()
        for frame in shared:
            for column in frame._columns:
                columns.append(column.copy())
        return DataFrame(columns^, height=height)
    raise Error("concat how must be 'vertical', 'diagonal', or 'horizontal'")


def concat(
    frames: List[LazyFrame], how: String = "vertical"
) raises -> LazyFrame:
    """Combine lazy plans into one plan, with the eager concat's results
    and errors for 'vertical', 'diagonal' and 'horizontal'.

    A diagonal concat is a vertical one over inputs each selected to
    the union of the columns (nulls filling those an
    input lacks), so filters and column selections above it reach every
    input, and a horizontal one is its own operator. The inputs' names and
    types are resolved here, as collect_schema resolves them.
    """
    return _concat_plans(frames, how)


struct _RankJob(Job):
    """Compute one sort key column's dense ranks."""

    var column: Series
    var descending: Bool
    var nulls_last: Bool
    var ranks: List[Int]

    def __init__(
        out self, var column: Series, descending: Bool, nulls_last: Bool
    ):
        self.column = column^
        self.descending = descending
        self.nulls_last = nulls_last
        self.ranks = List[Int]()

    def run(mut self) raises:
        self.ranks = self.column._sort_ranks(self.descending, self.nulls_last)


def _keeps_every_row(mask: BoolColumn) -> Bool:
    """Whether every row of a byte-aligned mask is a valid true, checked a
    byte (eight rows) at a time. Other masks answer False."""
    if mask._offset % 8 != 0:
        return False
    var n = mask._length
    var first = mask._offset // 8
    var values = mask._data[].unsafe_ptr().unsafe_offset(first)
    var has_bits = len(mask._bits[]) > 0
    var bits = mask._bits[].unsafe_ptr().unsafe_offset(first if has_bits else 0)
    var full = n // 8
    for k in range(full):
        var byte = values[unsafe_offset=k]
        if has_bits:
            byte &= bits[unsafe_offset=k]
        if byte != 255:
            return False
    var tail = n - 8 * full
    if tail > 0:
        var want = UInt8((1 << tail) - 1)
        var byte = values[unsafe_offset=full]
        if has_bits:
            byte &= bits[unsafe_offset=full]
        if byte & want != want:
            return False
    return True


def _is_untyped(value: Expr) -> Bool:
    """Whether an expression is built only from untyped numeric literals."""
    for node in value._nodes:
        if node.op == COL or node.op == SELECTOR:
            return False
        if (
            node.op == LIT_INT or node.op == LIT_FLOAT
        ) and node.text != UNTYPED:
            return False
        if node.op == LIT_BOOL or node.op == LIT_STRING or node.op == LIT_NULL:
            return False
    return True


comptime SELECTIVE_FILTER_ROWS = 4096
"""Frames shorter than this evaluate an AND filter whole."""


def _rows_mask(
    rows: List[Int], height: Int, *, every: Bool = False
) raises -> BoolColumn:
    """A mask of `height` rows with the (ascending) `rows` set, or every
    row set."""
    var fill = UInt8(255) if every else UInt8(0)
    var values = List[UInt8](length=(height + 7) // 8, fill=fill)
    if every:
        var rest = height % 8
        if rest > 0:
            values[len(values) - 1] = UInt8((1 << rest) - 1)
    else:
        var bits = values.unsafe_ptr()
        for r in rows:
            bits[unsafe_offset=r >> 3] |= UInt8(1) << UInt8(r & 7)
    return BoolColumn(values=values^, bits=List[UInt8](), length=height)


def _window_mask(low: Int, length: Int, height: Int) raises -> BoolColumn:
    """A mask of `height` rows with rows [low, low + length) set."""
    var values = List[UInt8](length=(height + 7) // 8, fill=0)
    var bits = values.unsafe_ptr()
    var r = low
    var end = low + length
    while r < end and (r & 7) != 0:
        bits[unsafe_offset=r >> 3] |= UInt8(1) << UInt8(r & 7)
        r += 1
    while r + 8 <= end:
        bits[unsafe_offset=r >> 3] = 255
        r += 8
    while r < end:
        bits[unsafe_offset=r >> 3] |= UInt8(1) << UInt8(r & 7)
        r += 1
    return BoolColumn(values=values^, bits=List[UInt8](), length=height)


def _placed_mask(inner: BoolColumn, low: Int, height: Int) raises -> BoolColumn:
    """`inner` (a mask over rows [low, low + len(inner))) as a mask over
    `height` rows, zero elsewhere."""
    var values = List[UInt8](length=(height + 7) // 8, fill=0)
    var bits = values.unsafe_ptr()
    for i in range(len(inner)):
        if inner._get(i) and inner._valid(i):
            var r = low + i
            bits[unsafe_offset=r >> 3] |= UInt8(1) << UInt8(r & 7)
    return BoolColumn(values=values^, bits=List[UInt8](), length=height)


def _reads_strings(expr: Expr, frame: DataFrame) raises -> Bool:
    for name in _column_names(expr):
        if frame.column(name).dtype().physical() == DataType.STRING:
            return True
    return False


def _round_ticks(value: Float64) -> Int64:
    """Nearest integer tick, halves away from zero."""
    return Int64(value + 0.5) if value >= 0 else -Int64(-value + 0.5)


def _describe_percentiles(percentiles: List[Float64]) raises -> List[Float64]:
    """describe()'s percentiles checked and in ascending order."""
    var qs = percentiles.copy()
    for q in qs:
        if not (q >= 0 and q <= 1):
            raise Error("describe() percentiles must lie in [0, 1]")
    sort(qs)
    return qs^


def _percent_label(q: Float64) -> String:
    """A percentile's row label as Python formats f"{q * 100:g}%": six
    significant digits, trailing zeros dropped (0.25 gives "25%")."""
    var value = q * 100
    if value == 0:
        return "0%"
    var decimals = 5
    var scale = value
    while scale >= 10:
        scale /= 10
        decimals -= 1
    while scale < 1:
        scale *= 10
        decimals += 1
    var scaled = value
    for _ in range(decimals):
        scaled *= 10
    var digits = Int(scaled + 0.5)
    while decimals > 0 and digits % 10 == 0:
        digits //= 10
        decimals -= 1
    var factor = 1
    for _ in range(decimals):
        factor *= 10
    if decimals == 0:
        return String(digits) + "%"
    var fraction = String(digits % factor)
    while fraction.byte_length() < decimals:
        fraction = "0" + fraction
    return String(digits // factor) + "." + fraction + "%"


@fieldwise_init
struct TimeGroupBy(Copyable):
    """A snapshot plus a duration grouping request, evaluated by agg.

    Overlapping membership is gathered once, then existing grouped expression
    reducers process all windows together. Empty rolling windows are retained;
    empty dynamic windows are omitted.
    """

    var _frame: DataFrame
    var _index: String
    var _every: String
    var _period: String
    var _offset: String
    var _closed: String
    var _label: String
    var _by: List[String]
    var _rolling: Bool

    def agg(
        self, expression: Expr, *, batch_size: Int = 1024
    ) raises -> DataFrame:
        return self.agg([expression.copy()], batch_size=batch_size)

    def agg(
        self, expressions: List[Expr], *, batch_size: Int = 1024
    ) raises -> DataFrame:
        positive_period(self._period)
        _ = parse_interval(self._offset)
        var sides = closed_sides(self._closed)
        if self._label not in ["left", "right", "datapoint"]:
            raise Error("label must be left, right or datapoint")
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        var index = self._frame.column(self._index).rechunk()
        var keys = List[Series]()
        var names = List[String]([self._index])
        for name in self._by:
            if name in names:
                raise Error("duplicate time grouping key: " + name)
            names.append(name)
            keys.append(self._frame.column(name))
        var ids = List[Int]()
        if len(keys):
            ids = encode_rows(_expand_struct_keys(keys), True).ids.copy()
        var groups = window_partitions(self._frame.height(), ids)
        # Unsorted groups are put in index order; rolling output rows still
        # follow input row order within each group.
        var inputs = List[List[Int]]()
        if not check_time_index(index, groups):
            if self._rolling:
                inputs = groups.copy()
            sort_time_groups(index, groups)
        var dtype = index.dtype()
        var zone = zone_of(dtype)
        ref times = index._data[Column[Int64]]
        var gathered = List[Int]()
        var window_ids = List[Int]()
        var representatives = List[Int]()
        var labels = List[Int64]()
        if self._rolling:
            var bounds = rolling_bounds(
                index, groups, self._period, self._offset, self._closed
            )
            for g in range(len(groups)):
                ref rows = groups[g]
                for j in range(len(rows)):
                    var row = inputs[g][j] if len(inputs) else rows[j]
                    var id = len(labels)
                    labels.append(times._get(row))
                    representatives.append(row)
                    var begin = bounds[0][row]
                    var end = bounds[1][row]
                    for k in range(begin, end):
                        gathered.append(rows[k])
                        window_ids.append(id)
        else:
            positive_period(self._every)
            for rows in groups:
                if len(rows) == 0:
                    continue
                var first = times._get(rows[0])
                var last = times._get(rows[len(rows) - 1])
                var start = shift_time(
                    truncate_time(first, dtype, self._every, zone),
                    dtype,
                    self._offset,
                    zone,
                )
                while start > first or (start == first and not sides[0]):
                    var previous = shift_time(
                        start, dtype, negate_period(self._every), zone
                    )
                    if previous >= start:
                        raise Error("every does not advance the time index")
                    start = previous
                # Large negative offsets can start before the first useful window.
                while shift_time(start, dtype, self._period, zone) < first:
                    start = shift_time(start, dtype, self._every, zone)
                while start <= last:
                    var end = shift_time(start, dtype, self._period, zone)
                    var lo = search_time(times, rows, start, not sides[0])
                    var hi = search_time(times, rows, end, sides[1])
                    if lo < hi:
                        var id = len(labels)
                        labels.append(
                            start if self._label
                            == "left" else end if self._label
                            == "right" else times._get(rows[lo])
                        )
                        representatives.append(rows[lo])
                        for k in range(lo, hi):
                            gathered.append(rows[k])
                            window_ids.append(id)
                    var next = shift_time(start, dtype, self._every, zone)
                    if next <= start:
                        raise Error("every does not advance the time index")
                    start = next
        var bound = _bind_all(expressions, self._frame._columns)
        for expression in bound:
            if expression.shape() != AGGREGATE:
                raise Error(
                    "Time grouping requires scalar aggregate expressions"
                )
            if expression.expr._name in names:
                raise Error(
                    "Aggregate output name collides with time grouping key: "
                    + expression.expr._name
                )
        var columns = List[Series]()
        for key in keys:
            columns.append(key.take(representatives))
        columns.append(
            Series(self._index, Column[Int64](labels.copy())).with_dtype(dtype)
        )
        var input = self._frame.take(gathered)
        for expression in bound:
            columns.append(
                evaluate(
                    expression,
                    input._columns,
                    input.height(),
                    batch_size=batch_size,
                    grouped=True,
                    groups=window_ids,
                    group_count=len(labels),
                )
            )
        return DataFrame(columns^, height=len(labels))
