"""An eager CPU dataframe with runtime schema and positional row semantics."""
from .dtype import DataType
from std.collections import Dict, Optional
from std.sys import num_physical_cores
from std.memory import ArcPointer, Pointer, unsafe_memcpy
from .bool_column import BoolColumn
from .column import Column, _append_validity, _pack_bits
from .string_column import StringColumn, StringBuilder
from .series import Series, sort_indices, smallest_indices
from .expr import (
    Expr,
    ARG_MIN,
    ARG_MAX,
    MODE,
    VALUE_COUNTS,
    SKEW,
    KURTOSIS,
    CORR,
    COV,
    is_reduction,
    is_window,
    subtree,
    OVER,
    MIN,
    MAX,
    FIRST,
    LAST,
    STD,
    VAR,
    LEN,
    ANY,
    ALL,
    NULL_COUNT,
    N_UNIQUE,
    col,
    lit,
    null,
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
    SUM,
    COUNT,
    MEAN,
    UNTYPED,
)
from .binding import bind, BoundExpr, ROWS, AGGREGATE
from .execution import evaluate, _ReduceJob
from .aggregate import Reducer
from .sampling import sample_size, sample_indices
from .mask_filter import filter_columns
from .gather import (
    SORTED_GATHER_MIN_CHUNKS,
    take_parallel,
    take_sorted_chunked,
    true_rows,
    float_compare_rows,
    can_filter_aligned_chunks,
    filter_float_chunks,
    filter_range_int64_chunks,
)
from .parallel import Job, Pool, partitions, run_jobs, worker_count
from .partition import (
    Partitioner,
    encode_bucket,
    encode_partitioned,
    low_cardinality,
    small_key_product,
)
from .join_hash import (
    RowsCopyJob,
    direct_hash_join_rows,
    direct_hash_semi_anti_rows,
    int64_progression,
    PreparedHashIndex,
    prefer_left_build,
    prepared_hash_join_rows,
    prepared_hash_semi_anti_rows,
)
from .join_type import (
    JOIN_ANTI,
    JOIN_CROSS,
    JOIN_FULL,
    JOIN_INNER,
    JOIN_LEFT,
    JOIN_RIGHT,
    JOIN_SEMI,
    join_code,
    join_name,
    join_type,
)
from .nested_column import ListColumn, StructColumn
from .trace import trace_path
from .row_encode import STRING_PREFIX_BYTES, encodable, encode_sort_keys
from .categorical import decode, recode, sort_ranks, unify, union_of
from .packed_sort import (
    chunked_top_rows,
    packed_arg_sort,
    packed_top_rows,
)

# A head of a sort at least 1/this of the rows sorts fully (#332).
comptime _TOP_K_FRACTION = 10
from .value import AnyValue
from .hashing import (
    RowKeys,
    encode_rows,
    encode_rows_parallel,
    encode_string_rows_parallel,
)
from .groups import GroupIndices
from .top_k import top_k_mask
from .hash_agg import hash_agg_eligible, hash_aggregate
from .indexed_reduce import indexed_reductions, reduce_indexed
from .expr_kernels import choose, validity
from .selectors import expand, expand_all
from .lazy import LazyFrame
from .display import render_frame, render_glimpse

# A struct key groups, joins and deduplicates by its fields plus its own
# validity, so a null struct is one group and a struct of nulls another.
# The expanded columns carry the struct's name, this separator, and the
# field name (or _KEY_VALID); _key_columns packs them back.
comptime _KEY_SEP = "\x1f"
comptime _KEY_VALID = "\x1evalid"


def _expand_struct_keys(columns: List[Series]) raises -> List[Series]:
    """Replace each struct key column by its fields and validity."""
    var out = List[Series](capacity=len(columns))
    for column in columns:
        if not column.dtype().is_struct():
            out.append(column.copy())
            continue
        var structs = column.struct_column()
        for i in range(structs.field_count()):
            var field = structs.field(i)
            if field.dtype().is_nested():
                raise Error(
                    "struct keys with nested fields are not supported yet: "
                    + column.name()
                    + "."
                    + field.name()
                )
            out.append(field.renamed(column.name() + _KEY_SEP + field.name()))
        out.append(
            Series(
                column.name() + _KEY_SEP + _KEY_VALID,
                BoolColumn(structs.validity()),
            )
        )
    return out^


def _key_prefix(name: String) -> String:
    """The struct name an expanded key column belongs to, or "" if plain."""
    var at = name.find(_KEY_SEP)
    if at < 0:
        return ""
    return String(name[byte=0:at])


def _pack_struct_keys(frame: DataFrame) raises -> DataFrame:
    """Pack runs of expanded key columns (see _expand_struct_keys) back into
    their struct, in place; other columns are unchanged."""
    var columns = List[Series](capacity=frame.width())
    var k = 0
    while k < frame.width():
        var prefix = _key_prefix(frame._columns[k].name())
        if prefix == "":
            columns.append(frame._columns[k].copy())
            k += 1
            continue
        var fields = List[Series]()
        var bits = List[UInt8]()
        while (
            k < frame.width()
            and _key_prefix(frame._columns[k].name()) == prefix
        ):
            ref column = frame._columns[k]
            var suffix = String(
                column.name()[byte = prefix.byte_length() + 1 :]
            )
            if suffix == _KEY_VALID:
                # The helper's values (not its bitmap) are the struct's
                # validity.
                var flags = column.bool()
                var values = List[Bool](capacity=len(flags))
                for i in range(len(flags)):
                    values.append(flags._get(i))
                bits = _pack_bits(values)
            else:
                fields.append(column.renamed(suffix))
            k += 1
        columns.append(Series(prefix, StructColumn(fields^, bits^)))
    return DataFrame(columns^, height=frame.height())


from .reductions import FloatSumState


@fieldwise_init
struct Field(Copyable):
    """One schema entry: a column name and its dtype."""

    var name: String
    var dtype: DataType


struct DataFrame(Copyable, Sized, Writable):
    """Own equal-length, uniquely named columns; transformations copy storage."""

    var _columns: List[Series]
    var _height: Int

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
        var names = Dict[String, Bool]()
        for column in columns:
            if len(column) != inferred:
                raise Error("DataFrame columns must have equal lengths")
            var name = column.name()
            if name in names:
                raise Error("Duplicate column name: " + name)
            names[name] = True
        self._columns = columns^
        self._height = inferred

    def rechunk(self) raises -> Self:
        """Return one contiguous array per column, copying only chunked data."""
        var columns = List[Series](capacity=self.width())
        for column in self._columns:
            columns.append(column.rechunk())
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

    def item(self) raises -> AnyValue:
        """Return the only cell of a 1x1 dataframe."""
        if self._height != 1 or self.width() != 1:
            raise Error("item() requires a dataframe with exactly one cell")
        return self._columns[0].get(0)

    def item(self, row: Int, column: String) raises -> AnyValue:
        return self._columns[self._index(column)].get(row)

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

    def vstack(self, other: Self) raises -> Self:
        """Append other's rows; schemas must match exactly."""
        return concat([self.copy(), other.copy()], "vertical")

    def hstack(self, other: Self) raises -> Self:
        """Append other's columns; heights must match, names stay unique."""
        return concat([self.copy(), other.copy()], "horizontal")

    def hstack(self, columns: List[Series]) raises -> Self:
        return self.hstack(Self(columns.copy(), height=self._height))

    def clear(self) raises -> Self:
        """Zero rows with the same schema."""
        return self.slice(0, 0)

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

    def rename(self, mapping: Dict[String, String]) raises -> Self:
        """Rename columns by old name; all names are validated first."""
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
        set (Polars' order without shuffle is unspecified). A seed makes the
        sample reproducible across runs and platforms; without one the
        generator is seeded from the clock. The generator is SplitMix64, so
        seeded samples differ from Polars' for the same seed.
        """
        var count = sample_size(self._height, n, fraction, with_replacement)
        return self.take(
            sample_indices(self._height, count, with_replacement, shuffle, seed)
        )

    def describe(
        self,
        percentiles: List[Float64] = [0.25, 0.5, 0.75],
        interpolation: String = "nearest",
    ) raises -> Self:
        """Summary statistics per column, laid out as Polars' describe().

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
        var qs = percentiles.copy()
        for q in qs:
            if not (q >= 0 and q <= 1):
                raise Error("describe() percentiles must lie in [0, 1]")
        sort(qs)
        var labels: List[String] = ["count", "null_count", "mean", "std", "min"]
        for q in qs:
            labels.append(_percent_label(q))
        labels.append("max")
        var rows = len(labels)
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
                # min and max skip NaN, as Polars' describe does; a column
                # of only NaN gives NaN (restored below).
                var numbers = x.fill_nan(null(DataType.FLOAT64))
                exprs.append(x.mean().alias(p + "mean"))
                exprs.append(numbers.min().alias(p + "min"))
                exprs.append(numbers.max().alias(p + "max"))
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
        var stats = self.select_exprs(exprs) if len(exprs) > 0 else Self(
            List[Series](), height=1
        )
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
                        if (
                            value.is_null()
                            and count > 0
                            and (key == "min" or key == "max")
                        ):
                            # Every valid value was NaN.
                            values.append(Float64(0) / Float64(0))
                            valid.append(True)
                            continue
                        valid.append(not value.is_null())
                        values.append(0 if value.is_null() else value.float64())
                    else:
                        values.append(0)
                        valid.append(False)
                columns.append(
                    Series(column.name(), Column[Float64](values^, valid^))
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
                    Series("", Column[Int64](ticks^, ticks_valid^))
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
                                ) if dtype.is_binary() else value.string()
                            )
                        )
                    else:
                        texts.append("")
                        valid.append(False)
            columns.append(Series(column.name(), StringColumn(texts, valid)))
        return Self(columns^, height=rows)

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
        return self.take(rows^)

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
        self, by: String, descending: Bool = False, nulls_last: Bool = True
    ) raises -> Self:
        """Stable single-column sort. NaNs follow non-null numbers."""
        return self.sort([by], descending, nulls_last)

    def sort(
        self,
        by: List[String],
        descending: Bool = False,
        nulls_last: Bool = True,
    ) raises -> Self:
        """Stable lexicographic sort by several columns, one direction."""
        return self.take(self.arg_sort(by, descending, nulls_last))

    def sort(
        self,
        by: List[String],
        *,
        descending: List[Bool],
        nulls_last: List[Bool],
    ) raises -> Self:
        """Per-column direction and null placement; list lengths match by."""
        return self.take(
            self.arg_sort(by, descending=descending, nulls_last=nulls_last)
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
            var packed = packed_arg_sort(keys, descending, nulls_last)
            if packed:
                trace_path("sort.packed")
                return packed.take()
        return sort_indices(self._sort_ranks(by, descending, nulls_last))

    def top_k(self, k: Int, by: List[String]) raises -> Self:
        """The k rows that sort(by, descending=True) would put first.

        Nulls rank last. Selection is about one pass rather than a full sort.
        """
        var n = len(by)
        return self.take(
            self._arg_sort_head(
                by,
                List[Bool](length=n, fill=True),
                List[Bool](length=n, fill=True),
                k,
            )
        )

    def top_k(self, k: Int, by: String) raises -> Self:
        return self.top_k(k, [by])

    def bottom_k(self, k: Int, by: List[String]) raises -> Self:
        """The k rows that sort(by) would put first; nulls rank last."""
        var n = len(by)
        return self.take(
            self._arg_sort_head(
                by,
                List[Bool](length=n, fill=False),
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
        keys when they pack, else a heap over dense ranks. Past a tenth of
        the rows selection stops paying and the full sort is cut short.
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
        return smallest_indices(
            self._sort_ranks(by, descending, nulls_last), take
        )

    def bottom_k(self, k: Int, by: String) raises -> Self:
        return self.bottom_k(k, [by])

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

        # Ranking a column is serial and is the largest part of a sort, so
        # several key columns are ranked at once. One column per job, not
        # one row range per job: ranking sorts the values, which a row range
        # cannot do independently. This is also why it is one `run_jobs`
        # call -- each costs about 1.3 ms in thread creation at 32 threads,
        # there being no pool yet (#103), which is enough to swallow the
        # gain if it is paid per round.
        if len(by) > 1 and worker_count(self.height()) > 1:
            var jobs = List[_RankJob](capacity=len(by))
            for i in range(len(by)):
                jobs.append(
                    _RankJob(
                        self._columns[self._index(by[i])].copy(),
                        descending[i],
                        nulls_last[i],
                    )
                )
            run_jobs(jobs)
            var ranked = List[List[Int]](capacity=len(by))
            for i in range(len(jobs)):
                ranked.append(jobs[i].ranks.copy())
            return ranked^

        var ranks = List[List[Int]](capacity=len(by))
        for i in range(len(by)):
            ranks.append(
                self._columns[self._index(by[i])]._sort_ranks(
                    descending[i], nulls_last[i]
                )
            )
        return ranks^

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
        take whichever side is present; with how="full" and coalesce=False,
        right keys are kept as separate columns instead.
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
    ) raises -> Self:
        if how == JOIN_CROSS:
            raise Error(
                "A cross join takes no keys; use join(right, how='cross')"
            )
        if len(left_on) == 0 or len(left_on) != len(right_on):
            raise Error(
                "Join requires the same nonzero number of left and right keys"
            )
        # A categorical key joins on codes, which compare only on a shared
        # dictionary (#106): both key columns move onto the union of theirs
        # (a String key is encoded into it), leaving the left codes as they
        # are. Unchanged pairs join as before.
        var unified = False
        var left_frame = self.copy()
        var right_frame = right.copy()
        for i in range(len(left_on)):
            var l = self._index(left_on[i])
            var r = right._index(right_on[i])
            var ltype = self._columns[l].dtype()
            var rtype = right._columns[r].dtype()
            if (ltype.is_categorical() or rtype.is_categorical()) and not (
                ltype.is_categorical()
                and rtype.is_categorical()
                and ltype == rtype
            ):
                var lkey = left_frame._columns[l].copy()
                var rkey = right_frame._columns[r].copy()
                if (
                    ltype.is_categorical()
                    or ltype.physical() == DataType.STRING
                ):
                    unify(lkey, rkey)
                    left_frame._columns[l] = lkey^
                    right_frame._columns[r] = rkey^
                    unified = True
        if unified:
            return left_frame._join_impl(
                right_frame,
                left_on=left_on,
                right_on=right_on,
                how=how,
                suffix=suffix,
                coalesce=coalesce,
                prepared=None,
                range_filtered=range_filtered,
            )
        var left_keys = List[Int]()
        var right_keys = List[Int]()
        var seen = Dict[String, Bool]()
        for i in range(len(left_on)):
            if left_on[i] in seen:
                raise Error("Duplicate join key: " + left_on[i])
            seen[left_on[i]] = True
            left_keys.append(self._index(left_on[i]))
            right_keys.append(right._index(right_on[i]))
            var ltype = self._columns[left_keys[i]].dtype()
            var rtype = right._columns[right_keys[i]].dtype()
            if ltype != rtype:
                raise Error(
                    "Join key dtypes differ: "
                    + left_on[i]
                    + " is "
                    + ltype.name()
                    + " but "
                    + right_on[i]
                    + " is "
                    + rtype.name()
                )
        var has_struct_key = False
        for k in left_keys:
            if self._columns[k].dtype().is_struct():
                has_struct_key = True
        if has_struct_key:
            # Join on the struct's fields and validity as ordinary key
            # columns, then drop those helpers. The right struct column is
            # dropped up front so it does not come through as an extra
            # output column.
            if (
                how != JOIN_INNER
                and how != JOIN_LEFT
                and how != JOIN_SEMI
                and how != JOIN_ANTI
            ):
                raise Error(
                    "struct join keys support inner, left, semi and anti"
                    " joins; found " + join_name(how)
                )
            var left_side = self.copy()
            var right_side = right.copy()
            var new_left_on = List[String]()
            var new_right_on = List[String]()
            var helpers = List[String]()
            for i in range(len(left_on)):
                if not self._columns[left_keys[i]].dtype().is_struct():
                    new_left_on.append(left_on[i])
                    new_right_on.append(right_on[i])
                    continue
                var left_parts = _expand_struct_keys(
                    [self._columns[left_keys[i]].copy()]
                )
                var right_parts = _expand_struct_keys(
                    [right._columns[right_keys[i]].renamed(left_on[i])]
                )
                right_side = right_side.drop([right_on[i]])
                for p in range(len(left_parts)):
                    left_side = left_side.with_column(left_parts[p].copy())
                    right_side = right_side.with_column(right_parts[p].copy())
                    new_left_on.append(left_parts[p].name())
                    new_right_on.append(left_parts[p].name())
                    helpers.append(left_parts[p].name())
            return left_side._join_impl(
                right_side,
                left_on=new_left_on,
                right_on=new_right_on,
                how=how,
                suffix=suffix,
                coalesce=coalesce,
            ).drop(helpers)
        var keep_right_keys = how == JOIN_FULL and not coalesce
        var right_output = List[Int]()
        var right_names = List[String]()
        if how != JOIN_SEMI and how != JOIN_ANTI:
            var names = Dict[String, Bool]()
            for column in self._columns:
                names[column.name()] = True
            var left_names = names.copy()
            for i in range(right.width()):
                if not keep_right_keys and i in right_keys:
                    continue
                var name = right._columns[i].name()
                if name in left_names:
                    name += suffix
                if name in names:
                    raise Error("Join output name collision: " + name)
                names[name] = True
                right_output.append(i)
                right_names.append(name)
        if (
            not range_filtered
            and not prepared
            and (how == JOIN_INNER or how == JOIN_LEFT)
            and prefer_left_build(self.height(), right.height())
        ):
            for k in range(len(left_keys)):
                var selected = _join_range_rows(
                    self._columns[left_keys[k]], right._columns[right_keys[k]]
                )
                if selected:
                    var filtered = right._filter_rows(selected.take())
                    return self._join_impl(
                        filtered,
                        left_on=left_on,
                        right_on=right_on,
                        how=how,
                        suffix=suffix,
                        coalesce=coalesce,
                        range_filtered=True,
                    )
        if prepared:
            var sources = List[Series](capacity=len(left_keys))
            for k in left_keys:
                sources.append(self._columns[k].copy())
            if how == JOIN_SEMI or how == JOIN_ANTI:
                return self._filter_rows(
                    prepared_hash_semi_anti_rows(
                        sources, prepared.value(), how == JOIN_SEMI
                    )
                )
            var pairs = prepared_hash_join_rows(
                sources, prepared.value(), how == JOIN_LEFT, omit_identity=True
            )
            var workers = worker_count(len(pairs[1]))
            var columns = self._columns.copy()
            var identity = pairs[2] or len(pairs[0]) == self.height()
            if identity and not pairs[2]:
                for i in range(len(pairs[0])):
                    if pairs[0][i] != i:
                        identity = False
                        break
            # The row lists are swapped out of the tuple: copying them
            # doubled the index traffic of a 10M-row join (#335).
            var output_height = len(pairs[1])
            var left_rows = List[Int]()
            var right_rows = List[Int]()
            swap(left_rows, pairs[0])
            swap(right_rows, pairs[1])
            if not identity:
                columns = take_parallel(
                    columns^, left_rows^, workers, or_null=False
                )
            var right_sources = List[Series]()
            for c in right_output:
                right_sources.append(right._columns[c].copy())
            var gathered = take_parallel(
                right_sources,
                right_rows^,
                workers,
                or_null=how == JOIN_LEFT,
            )
            for k in range(len(right_output)):
                columns.append(gathered[k].renamed(right_names[k]))
            return Self(columns^, height=output_height)
        if (how == JOIN_INNER or how == JOIN_LEFT) and len(left_keys) == 1:
            var dense = _dense_right_int64_rows(
                self._columns[left_keys[0]],
                right._columns[right_keys[0]],
                how == JOIN_LEFT,
            )
            if dense[0]:
                var left_rows = dense[1].copy()
                var right_rows = dense[2].copy()
                var workers = worker_count(len(left_rows))
                var columns = self._columns.copy()
                var left_identity = len(left_rows) == self.height()
                if left_identity:
                    for i in range(len(left_rows)):
                        if left_rows[i] != i:
                            left_identity = False
                            break
                var output_height = len(left_rows)
                if not left_identity:
                    var ordered_chunks = how == JOIN_INNER
                    for column in columns:
                        if (
                            not column.is_chunked()
                            or column.n_chunks() < SORTED_GATHER_MIN_CHUNKS
                        ):
                            ordered_chunks = False
                    if ordered_chunks:
                        columns = take_sorted_chunked(
                            columns^,
                            left_rows^,
                            workers,
                            allow_repeats=True,
                        )
                    else:
                        columns = take_parallel(
                            columns^, left_rows^, workers, or_null=False
                        )
                var right_sources = List[Series]()
                for c in right_output:
                    right_sources.append(right._columns[c].copy())
                var gathered = take_parallel(
                    right_sources,
                    right_rows^,
                    workers,
                    or_null=how == JOIN_LEFT,
                )
                for k in range(len(right_output)):
                    columns.append(gathered[k].renamed(right_names[k]))
                return Self(columns^, height=output_height)
        if (how == JOIN_SEMI or how == JOIN_ANTI) and len(left_keys) == 1:
            var aligned_chunks = can_filter_aligned_chunks(self._columns)
            if aligned_chunks:
                var chunk_membership = _range_int64_membership_chunks(
                    self,
                    left_keys[0],
                    right._columns[right_keys[0]],
                    how == JOIN_SEMI,
                )
                if chunk_membership[0]:
                    var selected = chunk_membership[1].copy()
                    return Self(selected^, height=len(selected[0]))
            else:
                var membership = _range_int64_membership_rows(
                    self._columns[left_keys[0]],
                    right._columns[right_keys[0]],
                    how == JOIN_SEMI,
                )
                if membership[0]:
                    return self._filter_rows(membership[1].copy())
        # A small left side against a large right one: hash the left keys
        # and scan the right rows, marking the left rows they match, as
        # DuckDB builds on the smaller side. PDS-H q4 hashed 3.8M late
        # lineitem rows to test about 57K orders.
        if (
            (how == JOIN_SEMI or how == JOIN_ANTI)
            and right.height() >= 524288
            and self.height() > 0
            and 4 * self.height() <= right.height()
            and self.height() <= Int(Int32.MAX)
        ):
            trace_path("join.membership_left_build")
            var left_sources = List[Series](capacity=len(left_keys))
            var right_sources = List[Series](capacity=len(right_keys))
            for k in range(len(left_keys)):
                left_sources.append(self._columns[left_keys[k]].copy())
                right_sources.append(right._columns[right_keys[k]].copy())
            var pairs = direct_hash_join_rows(
                right_sources, left_sources, False
            )
            var marked = List[Bool](length=self.height(), fill=False)
            for row in pairs[1]:
                marked[row] = True
            var keep = how == JOIN_SEMI
            var rows = List[Int]()
            for i in range(self.height()):
                if marked[i] == keep:
                    rows.append(i)
            return self._filter_rows(rows^)
        # Semi and anti joins on keys the bounded path declined: probe a
        # right-row hash index for membership only. The dictionary path
        # below would encode both inputs and group every right row first.
        if (
            (how == JOIN_SEMI or how == JOIN_ANTI)
            and worker_count(self.height()) > 1
            and right.height() <= Int(Int32.MAX)
        ):
            var left_sources = List[Series](capacity=len(left_keys))
            var right_sources = List[Series](capacity=len(right_keys))
            for k in range(len(left_keys)):
                left_sources.append(self._columns[left_keys[k]].copy())
                right_sources.append(right._columns[right_keys[k]].copy())
            if not low_cardinality(right_sources):
                return self._filter_rows(
                    direct_hash_semi_anti_rows(
                        left_sources, right_sources, how == JOIN_SEMI
                    )
                )
        # Dense ids over both inputs materialize and re-encode every key.
        # For high-cardinality right keys, a row index probes the original
        # columns directly and preserves exact equality across collisions.
        var build_left = prefer_left_build(self.height(), right.height())
        if (how == JOIN_INNER or how == JOIN_LEFT) and (
            build_left or worker_count(self.height()) > 1
        ):
            var left_sources = List[Series](capacity=len(left_keys))
            var right_sources = List[Series](capacity=len(right_keys))
            for k in range(len(left_keys)):
                left_sources.append(self._columns[left_keys[k]].copy())
                right_sources.append(right._columns[right_keys[k]].copy())
            var left_rows = List[Int]()
            var right_rows = List[Int]()
            var direct = False
            var direct_identity = False
            if build_left:
                var pairs = _smaller_build_join_rows(
                    left_sources, right_sources, how == JOIN_LEFT
                )
                swap(left_rows, pairs[0])
                swap(right_rows, pairs[1])
                direct = True
            if not direct and len(left_keys) == 1:
                direct = _bounded_int64_join_rows(
                    left_sources[0],
                    right_sources[0],
                    how == JOIN_LEFT,
                    left_rows,
                    right_rows,
                )
            if (
                not direct
                and right.height() <= Int(Int32.MAX)
                and not low_cardinality(right_sources)
            ):
                var pairs = direct_hash_join_rows(
                    left_sources,
                    right_sources,
                    how == JOIN_LEFT,
                    omit_identity=True,
                )
                direct = True
                direct_identity = pairs[2]
                swap(left_rows, pairs[0])
                swap(right_rows, pairs[1])
            if direct:
                var workers = worker_count(len(right_rows))
                var columns = self._columns.copy()
                var left_identity = (
                    direct_identity or len(left_rows) == self.height()
                )
                if left_identity and not direct_identity:
                    for i in range(len(left_rows)):
                        if left_rows[i] != i:
                            left_identity = False
                            break
                if not left_identity:
                    var ordered_chunks = how == JOIN_INNER
                    for column in columns:
                        if (
                            not column.is_chunked()
                            or column.n_chunks() < SORTED_GATHER_MIN_CHUNKS
                        ):
                            ordered_chunks = False
                    if ordered_chunks:
                        columns = take_sorted_chunked(
                            columns^,
                            left_rows^,
                            workers,
                            allow_repeats=True,
                        )
                    else:
                        columns = take_parallel(
                            columns^, left_rows^, workers, or_null=False
                        )
                var right_output_sources = List[Series]()
                for c in right_output:
                    right_output_sources.append(right._columns[c].copy())
                var output_height = len(right_rows)
                var gathered = take_parallel(
                    right_output_sources,
                    right_rows^,
                    workers,
                    or_null=how == JOIN_LEFT,
                )
                for k in range(len(right_output)):
                    columns.append(gathered[k].renamed(right_names[k]))
                return Self(columns^, height=output_height)
        # A right join is a left-major probe from the right input. Build on
        # the original left rows for high-cardinality keys, then swap the
        # resulting row lists back to the public output column order.
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        var direct_right = False
        if how == JOIN_RIGHT and worker_count(right.height()) > 1:
            var right_probe_keys = List[Series](capacity=len(right_keys))
            var left_build_keys = List[Series](capacity=len(left_keys))
            for k in range(len(left_keys)):
                right_probe_keys.append(right._columns[right_keys[k]].copy())
                left_build_keys.append(self._columns[left_keys[k]].copy())
            if len(left_keys) == 1:
                direct_right = _bounded_int64_join_rows(
                    right_probe_keys[0],
                    left_build_keys[0],
                    True,
                    right_rows,
                    left_rows,
                )
            if (
                not direct_right
                and self.height() <= Int(Int32.MAX)
                and not low_cardinality(left_build_keys)
            ):
                var pairs = direct_hash_join_rows(
                    right_probe_keys, left_build_keys, True
                )
                direct_right = True
                swap(right_rows, pairs[0])
                swap(left_rows, pairs[1])
        if not direct_right:
            trace_path("join.dictionary")
            var ids = _joint_key_ids(self, right, left_keys, right_keys)
            var left_ids = ids[0].copy()
            var right_ids = ids[1].copy()
            var count = ids[2]
            # Rows per key id in a flat CSR layout: one list per distinct key
            # would be hundreds of thousands of heap allocations on a
            # high-cardinality join. Ids appear in increasing row order, so a
            # counting sort preserves the match order the contract documents.
            var right_starts = List[Int]()
            var right_flat = List[Int]()
            if how != JOIN_RIGHT:
                right_starts = _group_index(right_ids, count)
                var csr_workers = worker_count(len(right_ids))
                # Use the output/cursor working set and key cardinality to
                # decide whether the extra stable scatter passes pay off.
                right_flat = _parallel_group_rows(
                    right_ids, right_starts, csr_workers
                ) if (
                    (how == JOIN_INNER or how == JOIN_FULL)
                    and _parallel_csr_fits(right_ids, count, csr_workers)
                ) else _group_rows(
                    right_ids, right_starts
                )
            if how == JOIN_SEMI or how == JOIN_ANTI:
                for i in range(len(left_ids)):
                    var id = left_ids[i]
                    var matched = (
                        id >= 0 and right_starts[id + 1] > right_starts[id]
                    )
                    if matched == (how == JOIN_SEMI):
                        left_rows.append(i)
                return self.take(left_rows)
            if how == JOIN_RIGHT:
                var left_starts = _group_index(left_ids, count)
                var left_workers = worker_count(len(left_ids))
                var left_flat = _parallel_group_rows(
                    left_ids, left_starts, left_workers
                ) if (
                    _parallel_csr_fits(left_ids, count, left_workers)
                ) else _group_rows(
                    left_ids, left_starts
                )
                var right_workers = worker_count(len(right_ids))
                if right_workers > 1:
                    var pairs = _parallel_join_rows(
                        right_ids, left_starts, left_flat, right_workers, True
                    )
                    swap(right_rows, pairs[0])
                    swap(left_rows, pairs[1])
                else:
                    for j in range(len(right_ids)):
                        var id = right_ids[j]
                        if id >= 0 and left_starts[id + 1] > left_starts[id]:
                            for k in range(
                                left_starts[id], left_starts[id + 1]
                            ):
                                left_rows.append(left_flat[k])
                                right_rows.append(j)
                        else:
                            left_rows.append(-1)
                            right_rows.append(j)
            elif how == JOIN_FULL and worker_count(len(left_ids)) > 1:
                var pairs = _parallel_full_join_rows(
                    left_ids,
                    right_ids,
                    right_starts,
                    right_flat,
                    count,
                    worker_count(len(left_ids)),
                )
                swap(left_rows, pairs[0])
                swap(right_rows, pairs[1])
            elif (how == JOIN_INNER or how == JOIN_LEFT) and worker_count(
                len(left_ids)
            ) > 1:
                var pairs = _parallel_join_rows(
                    left_ids,
                    right_starts,
                    right_flat,
                    worker_count(len(left_ids)),
                    how == JOIN_LEFT,
                )
                swap(left_rows, pairs[0])
                swap(right_rows, pairs[1])
            else:
                var right_matched = List[Bool](
                    length=len(right_ids), fill=False
                )
                for i in range(len(left_ids)):
                    var id = left_ids[i]
                    if id >= 0 and right_starts[id + 1] > right_starts[id]:
                        for k in range(right_starts[id], right_starts[id + 1]):
                            var j = right_flat[k]
                            left_rows.append(i)
                            right_rows.append(j)
                            right_matched[j] = True
                    elif how != JOIN_INNER:
                        left_rows.append(i)
                        right_rows.append(-1)
                if how == JOIN_FULL:
                    for j in range(len(right_ids)):
                        if not right_matched[j]:
                            left_rows.append(-1)
                            right_rows.append(j)
        if how == JOIN_RIGHT:
            # Every output row has a right row. Its key is the coalesced key
            # even when no left row matched, so never gather a left key or
            # choose between duplicate key columns after materialization.
            var gather_workers = worker_count(len(left_rows))
            var left_sources = List[Series]()
            for c in range(self.width()):
                if c not in left_keys:
                    left_sources.append(self._columns[c].copy())
            var left_identity = len(left_rows) == self.height()
            if left_identity:
                for i in range(len(left_rows)):
                    if left_rows[i] != i:
                        left_identity = False
                        break
            var left_gathered = left_sources^
            if not left_identity:
                left_gathered = take_parallel(
                    left_gathered^,
                    left_rows.copy(),
                    gather_workers,
                    or_null=True,
                )
            var right_sources = List[Series]()
            for k in range(len(right_keys)):
                right_sources.append(right._columns[right_keys[k]].copy())
            for c in right_output:
                right_sources.append(right._columns[c].copy())
            var right_gathered = take_parallel(
                right_sources, right_rows^, gather_workers, or_null=False
            )
            var columns = List[Series](
                capacity=self.width() + len(right_output)
            )
            var left_source = 0
            for c in range(self.width()):
                var key = -1
                for k in range(len(left_keys)):
                    if left_keys[k] == c:
                        key = k
                        break
                if key >= 0:
                    columns.append(
                        right_gathered[key].renamed(self._columns[c].name())
                    )
                else:
                    columns.append(left_gathered[left_source].copy())
                    left_source += 1
            for k in range(len(right_output)):
                columns.append(
                    right_gathered[len(right_keys) + k].renamed(right_names[k])
                )
            return Self(columns^, height=len(left_rows))
        # Assembling the output is about half of a join, and it used to
        # gather one column at a time on the calling thread. take_parallel
        # writes disjoint output ranges, so every column of both sides goes
        # at once; `or_null` is what lets it carry the -1 that means "no row
        # on this side".
        var gather_workers = worker_count(len(left_rows))
        var sides_mixed = how == JOIN_RIGHT or how == JOIN_FULL
        var left_sources = List[Series](capacity=self.width())
        for c in range(self.width()):
            left_sources.append(self._columns[c].copy())
        # When every left row appears once in input order, its columns are
        # already the exact output. Sharing them avoids a full-frame gather.
        var left_identity = len(left_rows) == self.height()
        if left_identity:
            for i in range(len(left_rows)):
                if left_rows[i] != i:
                    left_identity = False
                    break
        var columns = left_sources^
        if not left_identity:
            columns = take_parallel(
                columns^,
                left_rows.copy(),
                gather_workers,
                or_null=sides_mixed,
            )

        # Right-side columns: the non-key output columns, plus any key
        # column that has to be coalesced with its left counterpart.
        var right_sources = List[Series]()
        var coalesced = List[Int]()
        for c in range(self.width()):
            for k in range(len(left_keys)):
                if left_keys[k] == c and sides_mixed and not keep_right_keys:
                    coalesced.append(c)
                    right_sources.append(right._columns[right_keys[k]].copy())
        for k in range(len(right_output)):
            right_sources.append(right._columns[right_output[k]].copy())
        var from_right = take_parallel(
            right_sources,
            right_rows.copy(),
            gather_workers,
            or_null=how == JOIN_LEFT or how == JOIN_FULL,
        )

        for j in range(len(coalesced)):
            var c = coalesced[j]
            var use_left = List[Bool](capacity=len(left_rows))
            for i in left_rows:
                use_left.append(i >= 0)
            columns[c] = choose(use_left, columns[c], from_right[j]).renamed(
                self._columns[c].name()
            )
        for k in range(len(right_output)):
            columns.append(
                from_right[len(coalesced) + k].renamed(right_names[k])
            )
        return Self(columns^, height=len(left_rows))

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
        var left_rows = List[Int](capacity=total)
        var right_rows = List[Int](capacity=total)
        for i in range(self._height):
            for j in range(right._height):
                left_rows.append(i)
                right_rows.append(j)
        var columns = List[Series]()
        for column in self._columns:
            columns.append(column.take(left_rows))
        for k in range(right.width()):
            columns.append(
                right._columns[k].take(right_rows).renamed(right_names[k])
            )
        return Self(columns^, height=total)

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
        for expression in bound:
            var result = evaluate(
                expression, self._columns, self._height, batch_size=batch_size
            )
            if expression.shape() != ROWS and has_rows:
                result = result._broadcast(height)
            columns.append(result^)
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
        for expression in bound:
            var result = evaluate(
                expression, self._columns, self._height, batch_size=batch_size
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
        return Self(columns^, height=self._height)

    def filter(self, predicate: Expr, *, batch_size: Int = 8192) raises -> Self:
        var predicates = expand(predicate, self._columns)
        if len(predicates) != 1:
            raise Error("A filter selector must match exactly one column")
        var bound = bind(predicates[0], self._columns)
        if bound.dtypes[len(bound.dtypes) - 1] != DataType.BOOL:
            raise Error("Filter expression must return Boolean values")
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        # Each partition's first k rows by ordinal rank: no rank computed.
        var top = top_k_mask(bound, self._columns, self._height)
        if top:
            trace_path("filter.top_k_per_partition")
            return self.filter(top.value())
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
                if (
                    self._height >= ALIGNED_FILTER_ROWS
                    and can_filter_aligned_chunks(self._columns)
                ):
                    var filtered = filter_float_chunks(
                        self._columns,
                        bound.sources[node.left],
                        node.op,
                        bound.expr._nodes[node.right].floating,
                    )
                    return Self(filtered^)
                return self._filter_rows(
                    float_compare_rows(
                        self._columns[bound.sources[node.left]],
                        node.op,
                        bound.expr._nodes[node.right].floating,
                    )
                )
        var result = evaluate(
            bound, self._columns, self._height, batch_size=batch_size
        )
        if bound.shape() != ROWS:
            result = result._broadcast(self._height)
        return self.filter(result.bool())

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
        var agg: Expr
        var value = col(values)
        var function = (
            aggregate_function if aggregate_function.byte_length()
            > 0 else "first"
        )
        if function == "first":
            agg = value.first()
        elif function == "last":
            agg = value.last()
        elif function == "sum":
            agg = value.sum()
        elif function == "mean":
            agg = value.mean()
        elif function == "min":
            agg = value.min()
        elif function == "max":
            agg = value.max()
        elif function == "count":
            agg = value.count()
        elif function == "len":
            agg = value.len()
        elif function == "median":
            agg = value.median()
        else:
            raise Error(
                "aggregate_function must be first, last, sum, mean, min, max,"
                " count, len, or median"
            )
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

    def _key_counts(
        self, subset: List[String]
    ) raises -> Tuple[List[Int], List[Int]]:
        var keys = encode_rows(self._subset_keys(subset), nulls_equal=True)
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
        return Series("is_duplicated", BoolColumn(flags^))

    def is_unique(self, subset: List[String] = List[String]()) raises -> Series:
        """True for every row whose key occurs exactly once."""
        var keyed = self._key_counts(subset)
        var flags = List[Bool](capacity=self._height)
        for id in keyed[0]:
            flags.append(keyed[1][id] == 1)
        return Series("is_unique", BoolColumn(flags^))

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
        return self.with_columns(fills)

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
        return GroupBy(
            self.copy(), _expand_struct_keys(columns), maintain_order
        )

    def group_indices(self, key: String) raises -> GroupIndices:
        return self.group_indices([key])

    def group_indices(self, keys: List[String]) raises -> GroupIndices:
        """Which rows belong to which group, without aggregating them.

        For callers that want each group's rows rather than one summary row
        per group: `take(groups.rows(g))` builds a sub-frame, and
        `representative(g)` says where to read that group's key values.
        Groups are numbered in first-occurrence order, and null keys form
        their own group, both as in `group_by`.
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
        var encoded = encode_rows(_expand_struct_keys(columns), True)
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
        return GroupBy(
            self.copy(), _expand_struct_keys(columns), maintain_order
        )


def _as_int64(values: List[Int]) -> List[Int64]:
    var out = List[Int64](capacity=len(values))
    for v in values:
        out.append(Int64(v))
    return out^


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
    var possible = 0
    for i in range(samples):
        var row = i * (len(right) // samples)
        var part = right.copy()
        if right.is_chunked():
            var located = right._chunk_at(row)
            part = located[0].copy()
            row = located[1]
        ref key = part._data[Column[Int64]]
        if key._valid(row):
            var value = key._get(row)
            possible += Int(value >= low and value <= high)
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


def _smaller_build_join_rows(
    left: List[Series], right: List[Series], include_unmatched: Bool
) raises -> Tuple[List[Int], List[Int]]:
    """Build on logical left, then stably restore left-major/right-input order.

    The physical probe emits right-major pairs. Stable counting scatter by
    logical left row preserves the original right-row ordering within every
    left group, including duplicate keys on either side.
    """
    trace_path("join.smaller_build")
    var pairs = direct_hash_join_rows(right, left, False)
    var starts = _group_index(pairs[1], len(left[0]))
    var order = _group_rows(pairs[1], starts)
    var left_rows = List[Int](capacity=len(order))
    var right_rows = List[Int](capacity=len(order))
    for i in range(len(left[0])):
        if starts[i] == starts[i + 1] and include_unmatched:
            left_rows.append(i)
            right_rows.append(-1)
        for at in range(starts[i], starts[i + 1]):
            left_rows.append(i)
            right_rows.append(pairs[0][order[at]])
    return (left_rows^, right_rows^)


def _group_index(ids: List[Int], count: Int) -> List[Int]:
    """Start offset per key id, in a flat CSR layout (count + 1 entries)."""
    var starts = List[Int](length=count + 1, fill=0)
    for id in ids:
        if id >= 0:
            starts[id + 1] += 1
    for g in range(count):
        starts[g + 1] += starts[g]
    return starts^


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


def _group_rows(ids: List[Int], starts: List[Int]) -> List[Int]:
    """Row indices grouped by key id, each group in increasing row order."""
    var rows = List[Int](length=starts[len(starts) - 1], fill=0)
    var cursor = starts.copy()
    for i in range(len(ids)):
        var id = ids[i]
        if id >= 0:
            rows[cursor[id]] = i
            cursor[id] += 1
    return rows^


# Rows from which a Float64 comparison filter runs directly on aligned
# chunks instead of materializing a mask. Swept on 2026-09-25 with the
# overall benchmark's filter (32 threads, best of 7, two rounds): the mask
# path won at 5k and 20k rows (0.18 vs 0.25 ms, 0.70 vs 0.73 ms) and the
# aligned path won from 100k up (2.1 vs 3.8 ms; 4.7 vs 7.9 ms at 1M). The
# old 2,000,000 was chosen without a sweep and cost 1M-row filters 1.7x.
comptime ALIGNED_FILTER_ROWS = 50_000

# Measured on a Threadripper 3970X (32 cores, 16 MiB L3 per cache domain),
# Mojo 1.2, at 32 workers with an 8-worker cross-check; see
# docs/join-cutoffs.md for the alternating sweeps and their limitations.
# CSR: 750k distinct rows lose (15 vs 18 ms), 1M win (33 vs 21-26 ms).
# At 1.5M rows / 93,750 groups serial wins; at 1.75M / 109,375 parallel wins.
# Use the conservative 16 MiB working-set boundary: earlier dispatch helps
# uniform random IDs but regressed the 1M-row dictionary-encoded full join.
comptime _CSR_PARALLEL_WORKING_BYTES = 16 * 1024 * 1024
# At 6M rows, 64/512 groups win serial (63/35 vs 83/67 ms), 4,098
# groups are level (64 vs 62 ms), and 5,859 groups win parallel (127 vs 71).
# Keep the cursor below 32 KiB on the serial path.
comptime _CSR_MIN_PARALLEL_GROUPS = 4096
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


def _parallel_csr_fits(ids: List[Int], groups: Int, workers: Int) -> Bool:
    """Choose stable scatter from footprint, cardinality and sampled locality.

    The sample only selects between equivalent algorithms; it is not an
    ordering proof. Missing a rare inversion merely keeps the serial path.
    """
    var rows = len(ids)
    if (
        workers <= 1
        or groups < _CSR_MIN_PARALLEL_GROUPS
        or rows < max(0, _CSR_PARALLEL_WORKING_BYTES // 8 - groups)
    ):
        return False
    # Ordered IDs write almost sequentially: 1M/2M/4M distinct rows take
    # 6/13/25 ms serial vs 19/40/69 ms parallel. At most 64 adjacent pairs
    # sample locality; random IDs usually reject ordering at the first pair.
    var step = max(1, (rows - 1) // 64)
    for sample in range(64):
        var row = 1 + sample * step
        if row >= rows:
            break
        if ids[row] >= 0 and ids[row - 1] > ids[row]:
            return True
    return False


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


struct _RangeJoinProbeJob(Job):
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
        self.left_rows = List[Int](capacity=end - start)
        self.right_rows = List[Int](capacity=end - start)

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
    run_jobs(jobs)
    # Concatenate on every worker: each job's matches are copied to their
    # offset in the output. One thread appending them row by row was a
    # third of a join's time on 10M rows.
    var offsets = List[Int](capacity=len(jobs) + 1)
    offsets.append(0)
    for job in range(len(jobs)):
        offsets.append(offsets[job] + len(jobs[job].left_rows))
    var total = offsets[len(jobs)]
    left_rows = List[Int](unsafe_uninit_length=total)
    right_rows = List[Int](unsafe_uninit_length=total)
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


def _id_range_first(bucket: Int, groups: Int, buckets: Int) -> Int:
    return (bucket * groups + buckets - 1) // buckets


def _id_range_bucket(id: Int, groups: Int, buckets: Int) -> Int:
    var bucket = 0
    while bucket + 1 < buckets and id >= _id_range_first(
        bucket + 1, groups, buckets
    ):
        bucket += 1
    return bucket


struct _CSRCountJob(Job):
    """Count one input range into contiguous global-id buckets."""

    var ids: Int
    var start: Int
    var end: Int
    var groups: Int
    var buckets: Int
    var counts: List[Int]

    def __init__(
        out self, ids: Int, start: Int, end: Int, groups: Int, buckets: Int
    ):
        self.ids = ids
        self.start = start
        self.end = end
        self.groups = groups
        self.buckets = buckets
        self.counts = List[Int]()

    def run(mut self) raises:
        ref ids = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.ids
        )[]
        self.counts = List[Int](length=self.buckets, fill=0)
        for row in range(self.start, self.end):
            if ids[row] >= 0:
                self.counts[
                    _id_range_bucket(ids[row], self.groups, self.buckets)
                ] += 1


struct _CSRScatterJob(Job):
    """Stably scatter one input range into its global-id bucket spans."""

    var ids: Int
    var output: Int
    var start: Int
    var end: Int
    var groups: Int
    var buckets: Int
    var cursor: List[Int]

    def __init__(
        out self,
        ids: Int,
        output: Int,
        start: Int,
        end: Int,
        groups: Int,
        buckets: Int,
        var cursor: List[Int],
    ):
        self.ids = ids
        self.output = output
        self.start = start
        self.end = end
        self.groups = groups
        self.buckets = buckets
        self.cursor = cursor^

    def run(mut self) raises:
        ref ids = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.ids
        )[]
        ref output = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.output
        )[]
        for row in range(self.start, self.end):
            var id = ids[row]
            if id >= 0:
                var bucket = _id_range_bucket(id, self.groups, self.buckets)
                output[self.cursor[bucket]] = row
                self.cursor[bucket] += 1


struct _CSRFillJob(Job):
    """Fill the final CSR groups for one disjoint contiguous id range."""

    var ids: Int
    var starts: Int
    var order: Int
    var output: Int
    var first: Int
    var last: Int
    var group_first: Int
    var cursor: List[Int]

    def __init__(
        out self,
        ids: Int,
        starts: Int,
        order: Int,
        output: Int,
        first: Int,
        last: Int,
        group_first: Int,
        var cursor: List[Int],
    ):
        self.ids = ids
        self.starts = starts
        self.order = order
        self.output = output
        self.first = first
        self.last = last
        self.group_first = group_first
        self.cursor = cursor^

    def run(mut self) raises:
        ref ids = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.ids
        )[]
        ref order = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.order
        )[]
        ref output = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.output
        )[]
        for at in range(self.first, self.last):
            var row = order[at]
            var id = ids[row]
            output[self.cursor[id - self.group_first]] = row
            self.cursor[id - self.group_first] += 1


def _parallel_group_rows(
    ids: List[Int], starts: List[Int], workers: Int
) raises -> List[Int]:
    """Stable counting-sort rows into CSR groups without random writes.

    The first scatter is only by a small contiguous-id bucket, with worker
    spans laid out in input order. Each bucket then owns whole CSR groups and
    writes its final segment independently. For one id, its input row order
    therefore survives both passes exactly.
    """
    var groups = len(starts) - 1
    var rows = starts[len(starts) - 1]
    if rows == 0 or groups == 0:
        return List[Int]()
    var buckets = min(workers, groups)
    var bounds = partitions(len(ids), workers, 1)
    var counts = List[_CSRCountJob](capacity=workers)
    for worker in range(workers):
        counts.append(
            _CSRCountJob(
                Int(Pointer(to=ids)),
                bounds[worker],
                bounds[worker + 1],
                groups,
                buckets,
            )
        )
    run_jobs(counts)
    var bucket_starts = List[Int](length=buckets + 1, fill=0)
    for bucket in range(buckets):
        for worker in range(workers):
            bucket_starts[bucket + 1] += counts[worker].counts[bucket]
        bucket_starts[bucket + 1] += bucket_starts[bucket]
    var cursors = List[Int](capacity=buckets)
    for bucket in range(buckets):
        cursors.append(bucket_starts[bucket])
    var scatters = List[_CSRScatterJob](capacity=workers)
    var order = List[Int](length=rows, fill=0)
    for worker in range(workers):
        var cursor = cursors.copy()
        for bucket in range(buckets):
            cursors[bucket] += counts[worker].counts[bucket]
        scatters.append(
            _CSRScatterJob(
                Int(Pointer(to=ids)),
                Int(Pointer(to=order)),
                bounds[worker],
                bounds[worker + 1],
                groups,
                buckets,
                cursor^,
            )
        )
    run_jobs(scatters)
    var flat = List[Int](length=rows, fill=0)
    var fills = List[_CSRFillJob](capacity=buckets)
    for bucket in range(buckets):
        var first_group = _id_range_first(bucket, groups, buckets)
        var last_group = _id_range_first(bucket + 1, groups, buckets)
        var cursor = List[Int](capacity=last_group - first_group)
        for group in range(first_group, last_group):
            cursor.append(starts[group])
        fills.append(
            _CSRFillJob(
                Int(Pointer(to=ids)),
                Int(Pointer(to=starts)),
                Int(Pointer(to=order)),
                Int(Pointer(to=flat)),
                bucket_starts[bucket],
                bucket_starts[bucket + 1],
                first_group,
                cursor^,
            )
        )
    run_jobs(fills)
    # `fills` carries `order` only as an address. Keep the owning list live
    # until its workers have consumed that address.
    _ = order^
    return flat^


struct _JoinCountJob(Job):
    """Count inner-join output rows for one contiguous left range."""

    var left_ids: ArcPointer[List[Int]]
    var right_starts: ArcPointer[List[Int]]
    var start: Int
    var end: Int
    var count: Int
    var include_unmatched: Bool

    def __init__(
        out self,
        left_ids: ArcPointer[List[Int]],
        right_starts: ArcPointer[List[Int]],
        start: Int,
        end: Int,
        include_unmatched: Bool = False,
    ):
        self.left_ids = left_ids.copy()
        self.right_starts = right_starts.copy()
        self.start = start
        self.end = end
        self.count = 0
        self.include_unmatched = include_unmatched

    def run(mut self) raises:
        for row in range(self.start, self.end):
            var key = self.left_ids[][row]
            var matches = 0
            if key >= 0:
                matches = (
                    self.right_starts[][key + 1] - self.right_starts[][key]
                )
            if matches == 0 and self.include_unmatched:
                matches = 1
            if matches > Int.MAX - self.count:
                raise Error("Join output row count overflows")
            self.count += matches


struct _JoinFillJob(Job):
    """Fill one already-counted inner-join output span."""

    var left_ids: ArcPointer[List[Int]]
    var right_starts: ArcPointer[List[Int]]
    var right_flat: ArcPointer[List[Int]]
    var start: Int
    var end: Int
    var output: Int
    var left_output: Int
    var right_output: Int
    var include_unmatched: Bool

    def __init__(
        out self,
        left_ids: ArcPointer[List[Int]],
        right_starts: ArcPointer[List[Int]],
        right_flat: ArcPointer[List[Int]],
        start: Int,
        end: Int,
        output: Int,
        left_output: Int,
        right_output: Int,
        include_unmatched: Bool,
    ):
        self.left_ids = left_ids.copy()
        self.right_starts = right_starts.copy()
        self.right_flat = right_flat.copy()
        self.start = start
        self.end = end
        self.output = output
        self.left_output = left_output
        self.right_output = right_output
        self.include_unmatched = include_unmatched

    def run(mut self) raises:
        ref left_rows = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.left_output
        )[]
        ref right_rows = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.right_output
        )[]
        var output = self.output
        for row in range(self.start, self.end):
            var key = self.left_ids[][row]
            if (
                key >= 0
                and self.right_starts[][key + 1] > self.right_starts[][key]
            ):
                for at in range(
                    self.right_starts[][key], self.right_starts[][key + 1]
                ):
                    left_rows[output] = row
                    right_rows[output] = self.right_flat[][at]
                    output += 1
            elif self.include_unmatched:
                left_rows[output] = row
                right_rows[output] = -1
                output += 1


def _parallel_join_rows(
    left_ids: List[Int],
    right_starts: List[Int],
    right_flat: List[Int],
    workers: Int,
    include_unmatched: Bool,
) raises -> Tuple[List[Int], List[Int]]:
    """Inner or left pairs in the same left-major order as the serial loop.

    Each worker owns a contiguous left-row range. Counting it first assigns
    a disjoint output span; prefixing spans in left-range order preserves
    the public order contract, and each CSR group is itself right-row order.
    """
    var shared_left = ArcPointer(left_ids.copy())
    var shared_starts = ArcPointer(right_starts.copy())
    var shared_flat = ArcPointer(right_flat.copy())
    var bounds = partitions(len(left_ids), workers, 1)
    var counts = List[_JoinCountJob](capacity=workers)
    for worker in range(workers):
        counts.append(
            _JoinCountJob(
                shared_left,
                shared_starts,
                bounds[worker],
                bounds[worker + 1],
                include_unmatched,
            )
        )
    run_jobs(counts)
    var outputs = List[Int](capacity=workers)
    var total = 0
    for worker in range(workers):
        outputs.append(total)
        if counts[worker].count > Int.MAX - total:
            raise Error("Join output row count overflows")
        total += counts[worker].count
    var left_rows = List[Int](length=total, fill=0)
    var right_rows = List[Int](length=total, fill=0)
    var fills = List[_JoinFillJob](capacity=workers)
    for worker in range(workers):
        fills.append(
            _JoinFillJob(
                shared_left,
                shared_starts,
                shared_flat,
                bounds[worker],
                bounds[worker + 1],
                outputs[worker],
                Int(Pointer(to=left_rows)),
                Int(Pointer(to=right_rows)),
                include_unmatched,
            )
        )
    run_jobs(fills)
    return (left_rows^, right_rows^)


struct _UnmatchedRightCountJob(Job):
    """Count right rows whose key never occurred on the left."""

    var right_ids: ArcPointer[List[Int]]
    var left_present: ArcPointer[List[Bool]]
    var start: Int
    var end: Int
    var count: Int

    def __init__(
        out self,
        right_ids: ArcPointer[List[Int]],
        left_present: ArcPointer[List[Bool]],
        start: Int,
        end: Int,
    ):
        self.right_ids = right_ids.copy()
        self.left_present = left_present.copy()
        self.start = start
        self.end = end
        self.count = 0

    def run(mut self) raises:
        for row in range(self.start, self.end):
            var key = self.right_ids[][row]
            if key < 0 or not self.left_present[][key]:
                self.count += 1


struct _UnmatchedRightFillJob(Job):
    """Append unmatched right rows in their original input order."""

    var right_ids: ArcPointer[List[Int]]
    var left_present: ArcPointer[List[Bool]]
    var start: Int
    var end: Int
    var output: Int
    var left_output: Int
    var right_output: Int

    def __init__(
        out self,
        right_ids: ArcPointer[List[Int]],
        left_present: ArcPointer[List[Bool]],
        start: Int,
        end: Int,
        output: Int,
        left_output: Int,
        right_output: Int,
    ):
        self.right_ids = right_ids.copy()
        self.left_present = left_present.copy()
        self.start = start
        self.end = end
        self.output = output
        self.left_output = left_output
        self.right_output = right_output

    def run(mut self) raises:
        ref left_rows = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.left_output
        )[]
        ref right_rows = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.right_output
        )[]
        var output = self.output
        for row in range(self.start, self.end):
            var key = self.right_ids[][row]
            if key < 0 or not self.left_present[][key]:
                left_rows[output] = -1
                right_rows[output] = row
                output += 1


def _parallel_full_join_rows(
    left_ids: List[Int],
    right_ids: List[Int],
    right_starts: List[Int],
    right_flat: List[Int],
    count: Int,
    workers: Int,
) raises -> Tuple[List[Int], List[Int]]:
    """Full pairs in left-major order, then unmatched right input order."""
    trace_path("join.full_parallel")
    var present = List[Bool](length=count, fill=False)
    for key in left_ids:
        if key >= 0:
            present[key] = True
    var shared_left = ArcPointer(left_ids.copy())
    var shared_right = ArcPointer(right_ids.copy())
    var shared_starts = ArcPointer(right_starts.copy())
    var shared_flat = ArcPointer(right_flat.copy())
    var shared_present = ArcPointer(present^)
    var left_bounds = partitions(len(left_ids), workers, 1)
    var right_bounds = partitions(len(right_ids), workers, 1)
    var left_counts = List[_JoinCountJob](capacity=workers)
    var right_counts = List[_UnmatchedRightCountJob](capacity=workers)
    for worker in range(workers):
        left_counts.append(
            _JoinCountJob(
                shared_left,
                shared_starts,
                left_bounds[worker],
                left_bounds[worker + 1],
                True,
            )
        )
        right_counts.append(
            _UnmatchedRightCountJob(
                shared_right,
                shared_present,
                right_bounds[worker],
                right_bounds[worker + 1],
            )
        )
    run_jobs(left_counts)
    run_jobs(right_counts)
    var left_outputs = List[Int](capacity=workers)
    var right_outputs = List[Int](capacity=workers)
    var total = 0
    for worker in range(workers):
        left_outputs.append(total)
        if left_counts[worker].count > Int.MAX - total:
            raise Error("Join output row count overflows")
        total += left_counts[worker].count
    for worker in range(workers):
        right_outputs.append(total)
        if right_counts[worker].count > Int.MAX - total:
            raise Error("Join output row count overflows")
        total += right_counts[worker].count
    var left_rows = List[Int](length=total, fill=0)
    var right_rows = List[Int](length=total, fill=0)
    var left_fills = List[_JoinFillJob](capacity=workers)
    var right_fills = List[_UnmatchedRightFillJob](capacity=workers)
    for worker in range(workers):
        left_fills.append(
            _JoinFillJob(
                shared_left,
                shared_starts,
                shared_flat,
                left_bounds[worker],
                left_bounds[worker + 1],
                left_outputs[worker],
                Int(Pointer(to=left_rows)),
                Int(Pointer(to=right_rows)),
                True,
            )
        )
        right_fills.append(
            _UnmatchedRightFillJob(
                shared_right,
                shared_present,
                right_bounds[worker],
                right_bounds[worker + 1],
                right_outputs[worker],
                Int(Pointer(to=left_rows)),
                Int(Pointer(to=right_rows)),
            )
        )
    run_jobs(left_fills)
    run_jobs(right_fills)
    return (left_rows^, right_rows^)


def _joint_key_ids(
    left: DataFrame,
    right: DataFrame,
    left_keys: List[Int],
    right_keys: List[Int],
) raises -> Tuple[List[Int], List[Int], Int]:
    """Encode both sides' keys in one id space; null keys get id -1."""
    if len(left_keys) == 1:
        var direct = _bounded_int64_join_ids(
            left, right, left_keys[0], right_keys[0]
        )
        if direct[0]:
            return (direct[1].copy(), direct[2].copy(), direct[3])
    var stacked = List[Series](capacity=len(left_keys))
    for k in range(len(left_keys)):
        stacked.append(
            left._columns[left_keys[k]].append(right._columns[right_keys[k]])
        )
    # A join's row order comes from iterating rows, not from the id
    # numbering, so ids may be assigned in any consistent order. On a
    # high-cardinality key a single dictionary over both sides is the
    # dominant cost of the whole join, so encode per hash bucket instead.
    var total = left.height() + right.height()
    var workers = worker_count(total)
    var keys = encode_partitioned(
        stacked, workers, nulls_equal=False
    ) if workers > 1 and not low_cardinality(stacked) else encode_rows(
        stacked, nulls_equal=False
    )
    var n = left.height()
    var left_ids = List[Int](capacity=n)
    var right_ids = List[Int](capacity=right.height())
    for i in range(len(keys.ids)):
        if i < n:
            left_ids.append(keys.ids[i])
        else:
            right_ids.append(keys.ids[i])
    return (left_ids^, right_ids^, keys.count())


comptime _CONCAT_COLUMN = 0
comptime _CONCAT_STRING_BYTES = 1
comptime _CONCAT_STRING_OFFSETS = 2
comptime _CONCAT_STRING_BITS = 3


struct _ConcatJob(Job):
    """Build one fixed column or one independent string-buffer part."""

    var frames: ArcPointer[List[DataFrame]]
    var column: Int
    var part: Int
    var result: Series
    var bytes: List[UInt8]
    var offsets: List[Int64]
    var bits: List[UInt8]

    def __init__(
        out self, frames: ArcPointer[List[DataFrame]], column: Int, part: Int
    ):
        self.frames = frames.copy()
        self.column = column
        self.part = part
        self.result = frames[][0]._columns[column].copy()
        self.bytes = List[UInt8]()
        self.offsets = List[Int64]()
        self.bits = List[UInt8]()

    def run(mut self) raises:
        if self.part == _CONCAT_COLUMN:
            _concat_column(self.frames[], self.column, self.result)
        elif self.part == _CONCAT_STRING_BYTES:
            self.bytes = _concat_string_bytes(self.frames[], self.column)
        elif self.part == _CONCAT_STRING_OFFSETS:
            self.offsets = _concat_string_offsets(self.frames[], self.column)
        else:
            self.bits = _concat_string_bits(self.frames[], self.column)

    def into_column(deinit self) -> Series:
        return self.result^

    def into_bytes(deinit self) -> List[UInt8]:
        return self.bytes^

    def into_offsets(deinit self) -> List[Int64]:
        return self.offsets^

    def into_bits(deinit self) -> List[UInt8]:
        return self.bits^


def _concat_string_bytes(frames: List[DataFrame], column: Int) -> List[UInt8]:
    var total = 0
    for f in range(len(frames)):
        total += frames[f]._columns[column]._text_bytes()
    var bytes = List[UInt8](capacity=total)
    for f in range(len(frames)):
        ref source = frames[f]._columns[column]._data[StringColumn]
        if len(source) == 0:
            continue
        var first = source._start(0)
        var count = source._value_bytes()
        bytes.extend(
            Span[UInt8, ImmutAnyOrigin](
                unsafe_ptr=source.unsafe_bytes().unsafe_offset(first),
                length=count,
            )
        )
    return bytes^


def _concat_string_offsets(frames: List[DataFrame], column: Int) -> List[Int64]:
    var rows = 0
    for f in range(len(frames)):
        rows += frames[f].height()
    var offsets = List[Int64](capacity=rows + 1)
    offsets.append(0)
    var bytes = 0
    for f in range(len(frames)):
        ref source = frames[f]._columns[column]._data[StringColumn]
        var count = len(source)
        if count == 0:
            continue
        var first = source._start(0)
        var target = len(offsets)
        offsets.resize(target + count, 0)
        var incoming = source.unsafe_offsets().unsafe_offset(
            source.validity_offset() + 1
        )
        var destination = offsets.unsafe_ptr().unsafe_offset(target)
        var shift = Int64(bytes - first)
        var i = 0
        var shifts = SIMD[DType.int64, 4](shift)
        while i + 4 <= count:
            destination.unsafe_store[width=4](
                i, incoming.unsafe_load[width=4](i) + shifts
            )
            i += 4
        while i < count:
            offsets[target + i] = incoming.unsafe_load(i) + shift
            i += 1
        bytes += source._value_bytes()
    return offsets^


def _concat_string_bits(frames: List[DataFrame], column: Int) -> List[UInt8]:
    var rows = 0
    for f in range(len(frames)):
        rows += frames[f].height()
    var bits = List[UInt8](capacity=(rows + 7) // 8)
    var offset = 0
    for f in range(len(frames)):
        ref source = frames[f]._columns[column]._data[StringColumn]
        var count = len(source)
        _append_validity(
            bits,
            offset,
            source._bits[],
            source.validity_offset(),
            count,
        )
        offset += count
    return bits^


def _concat_column(
    frames: List[DataFrame], column: Int, mut into: Series
) raises:
    """Append one column of every frame after the first, into `into`.

    The final height and text size are both known from the inputs, so the
    output is sized once here. Letting it grow geometrically instead copied
    the whole column again on every doubling: reassembling 1M rows of 8
    columns from 32 ranges of a parallel CSV read took 21 ms that way and
    7 ms this way.
    """
    var rows = 0
    var text_bytes = 0
    for f in range(len(frames)):
        rows += frames[f].height()
        text_bytes += frames[f]._columns[column]._text_bytes()
    into._reserve_rows(rows, text_bytes)
    for f in range(1, len(frames)):
        into._append_series(frames[f]._columns[column])


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
                    columns.append(
                        Series.full_null(name, dtypes[name], frame.height())
                    )
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


def _bind_all(
    expressions: List[Expr], columns: List[Series]
) raises -> List[BoundExpr]:
    """Expand selectors, reject duplicate output names, then bind all."""
    var expanded = expand_all(expressions, columns)
    var bound = List[BoundExpr]()
    var names = Dict[String, Bool]()
    for expression in expanded:
        if expression._name in names:
            raise Error("Duplicate expression output name: " + expression._name)
        names[expression._name] = True
        bound.append(bind(expression, columns))
    return bound^


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


struct _EncodeJob(Job):
    """Encode one row range of a heavy bucket with a private dictionary."""

    var keys: List[Series]
    var ids: List[Int]
    var representatives: List[Int]

    def __init__(out self, var keys: List[Series]):
        self.keys = keys^
        self.ids = List[Int]()
        self.representatives = List[Int]()

    def run(mut self) raises:
        var local = encode_rows(self.keys, nulls_equal=True)
        self.ids = local.ids.copy()
        self.representatives = local.representatives.copy()


def _encode_parallel(keys: List[Series], workers: Int) raises -> RowKeys:
    """Dense ids in first-occurrence order, encoded per row range and then
    reconciled through the representative rows.

    Only worth it when distinct keys are few relative to rows, which is
    what makes a bucket heavy: the reconciliation re-encodes one row per
    (worker, distinct key), so its cost is workers x distinct keys.
    """
    var n = len(keys[0])
    var bounds = partitions(n, workers, 64)
    var jobs = List[_EncodeJob](capacity=workers)
    for w in range(workers):
        var slices = List[Series](capacity=len(keys))
        for key in keys:
            slices.append(key.slice(bounds[w], bounds[w + 1] - bounds[w]))
        jobs.append(_EncodeJob(slices^))
    run_jobs(jobs)

    # One row per (worker, local id), in worker order, so encoding those
    # rows yields each local id's global id at a known position.
    var rows = List[Int]()
    var offsets = List[Int](capacity=workers + 1)
    for w in range(workers):
        offsets.append(len(rows))
        for r in jobs[w].representatives:
            rows.append(bounds[w] + r)
    offsets.append(len(rows))
    var samples = List[Series](capacity=len(keys))
    for key in keys:
        samples.append(key.take(rows))
    var merged = encode_rows(samples, nulls_equal=True)

    var ids = List[Int](length=n, fill=0)
    for w in range(workers):
        var base = offsets[w]
        var start = bounds[w]
        for i in range(len(jobs[w].ids)):
            ids[start + i] = merged.ids[base + jobs[w].ids[i]]
    var representatives = List[Int](capacity=merged.count())
    for r in merged.representatives:
        representatives.append(rows[r])
    return RowKeys(ids^, representatives^)


struct _RangeAggJob(Job):
    """Reduce rows [start, end) into mergeable per-group state."""

    var frame: DataFrame
    var keys: List[Series]
    var expressions: List[Expr]
    var key_names: Dict[String, Bool]
    var start: Int
    var end: Int
    var budget: Int
    # None when the range held more groups than the budget.
    var state: Optional[_StreamReduction]

    def __init__(
        out self,
        frame: DataFrame,
        keys: List[Series],
        expressions: List[Expr],
        key_names: Dict[String, Bool],
        start: Int,
        end: Int,
        budget: Int,
    ):
        self.frame = frame.copy()
        self.keys = keys.copy()
        self.expressions = expressions.copy()
        self.key_names = key_names.copy()
        self.start = start
        self.end = end
        self.budget = budget
        self.state = None

    def run(mut self) raises:
        var length = self.end - self.start
        var frame = self.frame.slice(self.start, length)
        var keys = List[Series](capacity=len(self.keys))
        for key in self.keys:
            keys.append(key.slice(self.start, length))
        var state = _StreamReduction(
            frame, self.expressions, keys, self.key_names
        )
        if state.keys.height() <= self.budget:
            self.state = state^


struct _BucketJob(Job):
    """Group one hash bucket on its own: encode its keys, evaluate the
    aggregates, and remember each group's first row for reordering."""

    var keys: List[Series]
    var columns: List[Series]
    var expressions: List[Expr]
    var batch_size: Int
    var height: Int
    # Workers for the encode; 1 inside a batch, more when run on the caller.
    var encoders: Int
    var result: List[Series]
    var firsts: List[Int]

    def __init__(
        out self,
        var keys: List[Series],
        var columns: List[Series],
        expressions: List[Expr],
        batch_size: Int,
        height: Int,
        encoders: Int = 1,
    ):
        self.keys = keys^
        self.columns = columns^
        self.expressions = expressions.copy()
        self.batch_size = batch_size
        self.height = height
        self.encoders = encoders
        self.result = List[Series]()
        self.firsts = List[Int]()

    def run(mut self) raises:
        var groups = _encode_parallel(
            self.keys, self.encoders
        ) if self.encoders > 1 else encode_rows(self.keys, nulls_equal=True)
        for key in self.keys:
            self.result.append(key.take(groups.representatives))
        # Bind against this bucket's columns so source indices line up.
        var bound = _bind_all(self.expressions, self.columns)
        for expression in bound:
            self.result.append(
                evaluate(
                    expression,
                    self.columns,
                    self.height,
                    batch_size=self.batch_size,
                    grouped=True,
                    groups=groups.ids,
                    group_count=groups.count(),
                )
            )
        self.firsts = groups.representatives.copy()


struct _HashedBucketJob(Job):
    """Group one hash bucket from the partitioner's key hashes (#334): the
    keys stay in the source columns, compared in place on a hash match, and
    only the aggregated columns were gathered into bucket order."""

    var keys: List[Series]
    var hashes: Int
    var order: Int
    var lo: Int
    var hi: Int
    var columns: List[Series]
    var expressions: List[Expr]
    var batch_size: Int
    # Whether `columns` are the whole source columns, read through `order`
    # (`indexed_reduce.mojo`), rather than this bucket's gathered slices.
    var indexed: Bool
    var result: List[Series]
    var firsts: List[Int]

    def __init__(
        out self,
        keys: List[Series],
        hashes: Int,
        order: Int,
        lo: Int,
        hi: Int,
        var columns: List[Series],
        expressions: List[Expr],
        batch_size: Int,
        indexed: Bool = False,
    ):
        self.keys = keys.copy()
        self.hashes = hashes
        self.order = order
        self.lo = lo
        self.hi = hi
        self.columns = columns^
        self.expressions = expressions.copy()
        self.batch_size = batch_size
        self.indexed = indexed
        self.result = List[Series]()
        self.firsts = List[Int]()

    def run(mut self) raises:
        ref hashes = Pointer[List[UInt64], MutAnyOrigin](
            unsafe_from_address=self.hashes
        )[]
        ref order = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.order
        )[]
        var ids = List[Int]()
        var firsts = List[Int]()
        encode_bucket(
            self.keys,
            Span(hashes)[self.lo : self.hi],
            Span(order)[self.lo : self.hi],
            ids,
            firsts,
            exact_hashes=True,
        )
        for key in self.keys:
            self.result.append(key.take(firsts))
        var bound = _bind_all(self.expressions, self.columns)
        if self.indexed:
            var rows = order.unsafe_ptr().unsafe_offset(self.lo)
            for expression in bound:
                self.result.append(
                    reduce_indexed(
                        expression, self.columns, rows, ids, len(firsts)
                    ).renamed(expression.expr._name)
                )
            self.firsts = firsts^
            return
        if self.indexed:
            var rows = order.unsafe_ptr().unsafe_offset(self.lo)
            for expression in bound:
                self.result.append(
                    reduce_indexed(
                        expression, self.columns, rows, ids, len(firsts)
                    ).renamed(expression.expr._name)
                )
            self.firsts = firsts^
            return
        for expression in bound:
            self.result.append(
                evaluate(
                    expression,
                    self.columns,
                    self.hi - self.lo,
                    batch_size=self.batch_size,
                    grouped=True,
                    groups=ids,
                    group_count=len(firsts),
                )
            )
        self.firsts = firsts^


@fieldwise_init
struct GroupBy(Copyable):
    """An eager grouping request. No per-group dataframe materialization.

    It owns a snapshot of the input and the evaluated key columns.
    """

    var _frame: DataFrame
    var _keys: List[Series]
    var _maintain_order: Bool

    def _coded_keys(self) raises -> Optional[GroupBy]:
        """This grouping with each String key that carries dictionary codes
        replaced by those codes as a categorical; None when no key does."""
        var keys = List[Series](capacity=len(self._keys))
        var any = False
        for key in self._keys:
            var codes = key._dictionary_codes()
            if codes:
                keys.append(codes.take())
                any = True
            else:
                keys.append(key.copy())
        if not any:
            return None
        return GroupBy(self._frame.copy(), keys^, self._maintain_order)

    def _key_names(self) -> Dict[String, Bool]:
        var names = Dict[String, Bool]()
        for key in self._keys:
            var prefix = _key_prefix(key.name())
            names[prefix if prefix != "" else key.name()] = True
        return names^

    def _key_columns(self, groups: RowKeys) raises -> List[Series]:
        var columns = List[Series](capacity=len(self._keys))
        for key in self._keys:
            columns.append(key.take(groups.representatives))
        return columns^

    def agg(
        self, expression: Expr, *, batch_size: Int = 1024
    ) raises -> DataFrame:
        return self.agg([expression.copy()], batch_size=batch_size)

    def agg(
        self, expressions: List[Expr], *, batch_size: Int = 1024
    ) raises -> DataFrame:
        # String keys that carry their source's dictionary codes (a Parquet
        # scan) group by the codes, as categorical keys do, and come back as
        # strings: H2O q2's two string keys 102 -> 49 ms.
        var coded = self._coded_keys()
        if coded:
            var result = coded.value().agg(expressions, batch_size=batch_size)
            var columns = List[Series](capacity=result.width())
            for column in result._columns:
                if column.dtype().is_categorical() and column.name() in (
                    self._key_names()
                ):
                    columns.append(decode(column))
                else:
                    columns.append(column.copy())
            return DataFrame(columns^)
        var bound = _bind_all(expressions, self._frame._columns)
        if batch_size <= 0:
            raise Error("batch_size must be positive")
        var key_names = self._key_names()
        for expression in bound:
            if expression.shape() != AGGREGATE:
                raise Error(
                    "Group aggregation requires scalar aggregate expressions"
                )
            if expression.expr._name in key_names:
                raise Error(
                    "Aggregate output name collides with grouping key: "
                    + expression.expr._name
                )
        var workers = worker_count(self._frame.height())
        var result: DataFrame
        if (
            workers > 1
            and len(self._keys) == 1
            and not low_cardinality(self._keys)
            and hash_agg_eligible(self._keys[0], bound, self._frame._columns)
        ):
            return _pack_struct_keys(self._agg_hash(bound, workers))
        var ranged = Optional[DataFrame]()
        if (
            workers > 1
            and _stream_reductions(expressions)
            and not self._many_distinct(expressions)
        ):
            ranged = self._agg_ranges(expressions, key_names, workers)
        if ranged:
            result = ranged.take()
        elif workers > 1:
            result = self._agg_partitioned(
                expressions, bound, batch_size, workers
            )
        else:
            result = self._agg_whole(bound, batch_size)
        return _pack_struct_keys(result)

    def _agg_hash(
        self, bound: List[BoundExpr], workers: Int
    ) raises -> DataFrame:
        """Many groups of one Int64 key: states updated in place per worker
        range, then merged by hash part (`hash_agg.mojo`), with no column
        gathered into bucket order."""
        trace_path("group_by.hash_agg")
        var parts = hash_aggregate(
            self._keys[0], bound, self._frame._columns, workers
        )
        ref firsts = parts[0]
        var columns = List[Series](capacity=len(bound) + 1)
        columns.append(self._keys[0].take(firsts))
        for j in range(len(bound)):
            var pieces = List[Series](capacity=len(parts[1]))
            for p in range(len(parts[1])):
                pieces.append(parts[1][p][j].copy())
            var merged = Series._from_chunks(pieces).rechunk()
            columns.append(merged.renamed(bound[j].expr._name))
        var grouped = DataFrame(columns^, height=len(firsts))
        if not self._maintain_order:
            return grouped^
        return grouped.take(sort_indices([firsts.copy()]))

    def _many_distinct(self, expressions: List[Expr]) raises -> Bool:
        """Whether some `n_unique` reads many distinct values (#336).

        Per-range sets of (group, value) pairs merge on one thread, which
        is cheap only when values repeat: grouping IsRefresh by RegionID's
        3,586 values takes 57 ms that way and 75 ms through the whole-frame
        path's hash partitions, while UserID's 1.5M values take 314 ms and
        87 ms. A sample of evenly spaced rows tells them apart: at most a
        tenth of it distinct (RegionID shows 9%, SearchPhrase 14%, UserID
        all) keeps the ranges. Any input that is not a column is assumed
        to have many values.
        """
        var height = self._frame.height()
        var sample = min(height, 4096)
        if sample == 0:
            return False
        var rows = List[Int](capacity=sample)
        for k in range(sample):
            rows.append(k * height // sample)
        for expression in expressions:
            ref nodes = expression._nodes
            for node in nodes:
                if node.op != N_UNIQUE:
                    continue
                ref input = nodes[node.left]
                if input.op != COL:
                    return True
                var picked = self._frame.column(input.text).take(rows)
                var distinct = encode_rows([picked^], nulls_equal=True).count()
                if 10 * distinct > sample:
                    return True
        return False

    def _agg_ranges(
        self,
        expressions: List[Expr],
        key_names: Dict[String, Bool],
        workers: Int,
    ) raises -> Optional[DataFrame]:
        """Reduce each worker's row range to per-group state, then merge.

        Every row-local reduction the streaming executor supports keeps a
        mergeable state (see _StreamReduction), so this serves any list of
        them over any key types. Ranges are zero-copy slices; no column is
        gathered. Private states must stay small for the serial merge, so
        only low-cardinality keys qualify, and any range holding more groups
        than the budget falls back to hash partitioning. Groups keep
        first-occurrence order, as the whole-frame path does.
        """
        var height = self._frame.height()
        var budget = max(1, 1_000_000 // (workers * len(expressions)))
        # The same sampled estimate that picks whole-frame encoding over
        # hash partitioning: every range of a high-cardinality key would
        # hold most groups, and merging those states serially loses.
        if not (low_cardinality(self._keys) or small_key_product(self._keys)):
            return None
        var bounds = partitions(height, workers, 64)
        var jobs = List[_RangeAggJob](capacity=workers)
        for w in range(workers):
            jobs.append(
                _RangeAggJob(
                    self._frame,
                    self._keys,
                    expressions,
                    key_names,
                    bounds[w],
                    bounds[w + 1],
                    budget,
                )
            )
        run_jobs(jobs)
        for w in range(len(jobs)):
            if not jobs[w].state:
                return None
        var merged = jobs[0].state.take()
        for w in range(1, len(jobs)):
            merged.merge(jobs[w].state.value())
        trace_path("group_by.ranges")
        return merged.finish()

    def _referenced(self, bound: List[BoundExpr]) -> List[Series]:
        """The frame columns the aggregates read, in frame order; every
        column if a selector is still unexpanded."""
        var wanted = Dict[String, Bool]()
        var everything = False
        for expression in bound:
            for node in expression.expr._nodes:
                if node.op == COL:
                    wanted[node.text] = True
                elif node.op == SELECTOR:
                    everything = True
        var columns = List[Series]()
        for column in self._frame._columns:
            if everything or column.name() in wanted:
                columns.append(column.copy())
        return columns^

    def _agg_partitioned(
        self,
        expressions: List[Expr],
        bound: List[BoundExpr],
        batch_size: Int,
        workers: Int,
    ) raises -> DataFrame:
        """Group by hash bucket in parallel; see dataframe/partition.mojo.

        Each bucket holds a disjoint set of keys, so buckets are encoded
        and reduced independently and their outputs concatenated. Output
        order is bucket order unless maintain_order, which sorts groups by
        their first input row afterwards (O(groups), not O(rows)).

        A bucket holding several times its share of rows (a skewed key)
        would serialize the batch, so such buckets run on the calling
        thread, where their reduce can use the pool, and only the light
        buckets are jobs.
        """
        var height = self._frame.height()
        # A sampled estimate decides whether scattering is worth its gather;
        # on a low-cardinality key the serial encode it replaces is cheap.
        var whole = low_cardinality(self._keys)
        if whole:
            return self._agg_whole(bound, batch_size)
        trace_path("group_by.partitioned")
        var partitioner = Partitioner(self._keys, workers)
        var parts = partitioner.scatter(workers, with_hashes=True)
        var buckets = parts.buckets()
        var referenced = self._referenced(bound)
        # Heavy: a bucket big enough to serialize the batch on its own. It
        # must be large relative to the frame, not only to its share, or
        # low cardinality (few occupied buckets) would count as heavy.
        var heavy_rows = max(height // 8, 4 * height // buckets)
        var any_heavy = False
        for b in range(buckets):
            if parts.bounds[b + 1] - parts.bounds[b] > heavy_rows:
                any_heavy = True
        if not any_heavy:
            # Encode each bucket from the key hashes. Plain reductions read
            # values at their source rows; others gather values into bucket
            # order first.
            var indexed = indexed_reductions(bound, self._frame._columns)
            var values: List[Series]
            if indexed:
                trace_path("group_by.partitioned.indexed")
                values = List[Series](capacity=len(referenced))
                for column in referenced:
                    values.append(
                        column.rechunk() if column.is_chunked() else column.copy()
                    )
            else:
                values = take_parallel(referenced, parts.order.copy(), workers)
            var jobs = List[_HashedBucketJob]()
            for b in range(buckets):
                var lo = parts.bounds[b]
                var hi = parts.bounds[b + 1]
                if hi == lo:
                    continue
                var bucket_columns = List[Series](capacity=len(values))
                for column in values:
                    bucket_columns.append(
                        column.copy() if indexed else column.slice(lo, hi - lo)
                    )
                jobs.append(
                    _HashedBucketJob(
                        partitioner.keys,
                        Int(Pointer(to=parts.hashes)),
                        Int(Pointer(to=parts.order)),
                        lo,
                        hi,
                        bucket_columns^,
                        expressions,
                        batch_size,
                        indexed,
                    )
                )
            run_jobs(jobs)
            var frames = List[DataFrame](capacity=len(jobs))
            var starts = List[Int]()
            for j in range(len(jobs)):
                frames.append(
                    DataFrame(jobs[j].result.copy(), height=len(jobs[j].firsts))
                )
                for row in jobs[j].firsts:
                    starts.append(row)
            _ = partitioner^
            _ = parts^
            var grouped = concat(frames)
            if not self._maintain_order:
                return grouped^
            return grouped.take(sort_indices([starts^]))
        var sources = List[Series](capacity=len(self._keys) + len(referenced))
        for key in self._keys:
            sources.append(key.copy())
        for column in referenced:
            sources.append(column.copy())
        var gathered = take_parallel(sources, parts.order.copy(), workers)
        var keys = List[Series](capacity=len(self._keys))
        var columns = List[Series](capacity=len(referenced))
        for i in range(len(gathered)):
            if i < len(self._keys):
                keys.append(gathered[i].copy())
            else:
                columns.append(gathered[i].copy())
        var light = List[_BucketJob]()
        var light_offsets = List[Int]()
        var done = List[_BucketJob]()
        var done_offsets = List[Int]()
        for b in range(buckets):
            var lo = parts.bounds[b]
            var hi = parts.bounds[b + 1]
            if hi == lo:
                continue
            var bucket_keys = List[Series](capacity=len(keys))
            for key in keys:
                bucket_keys.append(key.slice(lo, hi - lo))
            var bucket_columns = List[Series](capacity=len(columns))
            for column in columns:
                bucket_columns.append(column.slice(lo, hi - lo))
            var heavy = hi - lo > heavy_rows
            var job = _BucketJob(
                bucket_keys^,
                bucket_columns^,
                expressions,
                batch_size,
                hi - lo,
                worker_count(hi - lo) if heavy else 1,
            )
            if heavy:
                # On the caller, so its encode and reduce can use the pool.
                job.run()
                done.append(job^)
                done_offsets.append(lo)
            else:
                light.append(job^)
                light_offsets.append(lo)
        run_jobs(light)
        while len(light) > 0:
            done.append(light.pop(0))
            done_offsets.append(light_offsets.pop(0))

        var frames = List[DataFrame](capacity=len(done))
        var firsts = List[Int]()
        for j in range(len(done)):
            var groups = len(done[j].firsts)
            frames.append(DataFrame(done[j].result.copy(), height=groups))
            for r in done[j].firsts:
                firsts.append(parts.order[done_offsets[j] + r])
        var result = concat(frames)
        if not self._maintain_order:
            return result^
        return result.take(sort_indices([firsts^]))

    def _agg_whole(
        self, bound: List[BoundExpr], batch_size: Int
    ) raises -> DataFrame:
        """Serial key encoding, then the parallel per-group reduce."""
        trace_path("group_by.whole")
        var groups: RowKeys
        if len(self._keys) == 1 and self._keys[0]._data.isa[StringColumn]():
            groups = encode_string_rows_parallel(
                self._keys[0], True, worker_count(self._frame.height())
            )
        else:
            groups = encode_rows_parallel(
                self._keys, True, worker_count(self._frame.height())
            )
        # First-occurrence representatives are ordered by source row.
        var columns = take_sorted_chunked(
            self._keys, groups.representatives.copy(), 1
        )
        for expression in bound:
            columns.append(
                evaluate(
                    expression,
                    self._frame._columns,
                    self._frame.height(),
                    batch_size=batch_size,
                    grouped=True,
                    groups=groups.ids,
                    group_count=groups.count(),
                )
            )
        return DataFrame(columns^, height=groups.count())

    def len(self, name: String = "len") raises -> DataFrame:
        """Row count per group, including rows with null values."""
        if name in self._key_names():
            raise Error(
                "Aggregate output name collides with grouping key: " + name
            )
        var groups = encode_rows(self._keys, nulls_equal=True)
        var counts = List[Int64](length=groups.count(), fill=0)
        for id in groups.ids:
            counts[id] += 1
        var columns = take_sorted_chunked(
            self._keys, groups.representatives.copy(), 1
        )
        columns.append(Series(name, Column[Int64](counts^)))
        return _pack_struct_keys(DataFrame(columns^, height=groups.count()))


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


def _row_local(exprs: List[Expr]) -> Bool:
    """True when every output row depends only on the same input row."""
    for e in exprs:
        for node in e._nodes:
            if is_reduction(node.op) or is_window(node.op) or node.op == OVER:
                return False
    return True


def _stream_reductions(expressions: List[Expr]) -> Bool:
    if len(expressions) == 0:
        return False
    for expression in expressions:
        ref nodes = expression._nodes
        var reachable = List[Bool](length=len(nodes), fill=False)
        reachable[len(nodes) - 1] = True
        var saw_reduction = False
        for reverse in range(len(nodes)):
            var i = len(nodes) - 1 - reverse
            ref node = nodes[i]
            if is_window(node.op) or node.op == OVER or node.op == SELECTOR:
                return False
            if not reachable[i]:
                continue
            if is_reduction(node.op):
                if node.op not in [
                    SUM,
                    COUNT,
                    MIN,
                    MAX,
                    MEAN,
                    FIRST,
                    LAST,
                    STD,
                    VAR,
                    LEN,
                    ANY,
                    ALL,
                    NULL_COUNT,
                    N_UNIQUE,
                    ARG_MIN,
                    ARG_MAX,
                    SKEW,
                    KURTOSIS,
                    CORR,
                    COV,
                ]:
                    return False
                # Reduction inputs must be row-local; reduction-of-reduction
                # and windows retain the materializing evaluator.
                if not _row_local([subtree(expression, node.left)]):
                    return False
                if node.right >= 0 and not _row_local(
                    [subtree(expression, node.right)]
                ):
                    return False
                saw_reduction = True
                continue
            if node.op == COL:
                return False
            for child in [node.left, node.right, node.extra]:
                if child >= 0:
                    reachable[child] = True
        if not saw_reduction:
            return False
    return True


def _encode_in_order(keys: List[Series]) raises -> RowKeys:
    """Dense ids in first-occurrence order, as `encode_rows` gives, using
    the hash-partitioned parallel encoder for large high-cardinality keys.

    A streaming merge encodes up to every group seen so far, and on 10M
    distinct keys the single-dictionary `encode_rows` took 6 s where the
    partitioned encoder takes a fraction of that (#326). Its numbering is
    unspecified, so one pass over the ids renumbers them by first
    occurrence and records each group's first row.
    """
    var n = len(keys[0])
    var workers = worker_count(n)
    if workers <= 1 or low_cardinality(keys):
        return encode_rows(keys, nulls_equal=True)
    var encoded = encode_partitioned(keys, workers, nulls_equal=True)
    var renumbered = List[Int](length=encoded.count(), fill=-1)
    var ids = List[Int](capacity=n)
    var representatives = List[Int](capacity=encoded.count())
    for row in range(n):
        var id = encoded.ids[row]
        var target = renumbered[id]
        if target < 0:
            target = len(representatives)
            renumbered[id] = target
            representatives.append(row)
        ids.append(target)
    return RowKeys(ids^, representatives^)


def _equality_words(columns: List[Series]) raises -> List[List[Int]]:
    """Words that are equal for two rows exactly when their keys are equal
    as grouping defines it (every NaN one value, -0.0 equal to 0.0, a null
    one value), word-major.

    The values come from the sort-key encoder (row_encode.mojo), which
    already canonicalises NaN and -0.0 and pads strings of up to
    STRING_PREFIX_BYTES. It adds a rank word only for columns with nulls or
    floats, which would give batches different layouts, so every column's
    rank (0 value, 1 NaN, 2 null) is packed into one leading word instead,
    two bits per column. Callers check `encodable` and at most 32 columns.
    """
    var rows = len(columns[0])
    var rank = List[Int](length=rows, fill=0)
    var words = List[List[Int]]()
    words.append(List[Int]())
    for k in range(len(columns)):
        var encoded = encode_sort_keys([columns[k].copy()], [False], [True])
        var expected = 2
        if columns[k].dtype().physical() == DataType.STRING:
            expected += STRING_PREFIX_BYTES // 8
        if len(encoded) == expected:
            ref ranks = encoded[0]
            for i in range(rows):
                rank[i] |= ranks[i] << (2 * k)
            _ = encoded.pop(0)
        while len(encoded) > 0:
            words.append(encoded.pop(0))
    words[0] = rank^
    return words^


def _word_hash(words: List[List[Int]], row: Int) -> UInt64:
    var h = UInt64(0x9E3779B97F4A7C15)
    for w in range(len(words)):
        var z = h ^ UInt64(words[w][row])
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        h = z ^ (z >> 31)
    return h


struct _KeyIndex(Movable):
    """Group ids by key words, kept across streaming merges (#326).

    Each group's words and hash are stored once; an open-addressing table
    maps a hash to candidate ids. Merging a batch probes only that batch's
    groups, so no merge re-encodes the groups already accumulated.
    """

    var words: List[List[Int]]
    var hashes: List[UInt64]
    var slots: List[Int]

    def __init__(out self):
        self.words = List[List[Int]]()
        self.hashes = List[UInt64]()
        self.slots = List[Int]()

    def __init__(out self, distinct: List[List[Int]]):
        """Index rows already known to be distinct, as ids 0, 1, ..."""
        var n = len(distinct[0])
        self.words = List[List[Int]](capacity=len(distinct))
        for _ in range(len(distinct)):
            self.words.append(List[Int](capacity=n))
        self.hashes = List[UInt64](capacity=n)
        var capacity = 1024
        while capacity < 2 * n:
            capacity *= 2
        self.slots = List[Int](length=capacity, fill=-1)
        for row in range(n):
            _ = self.find_or_insert(distinct, row)

    def count(self) -> Int:
        return len(self.hashes)

    def _grow(mut self):
        var capacity = max(1024, 2 * len(self.slots))
        self.slots = List[Int](length=capacity, fill=-1)
        var mask = capacity - 1
        for id in range(len(self.hashes)):
            var slot = Int(self.hashes[id]) & mask
            while self.slots[slot] >= 0:
                slot = (slot + 1) & mask
            self.slots[slot] = id

    def find_or_insert(mut self, source: List[List[Int]], row: Int) -> Int:
        """The id of source's row, adding it as the next id if new."""
        if len(self.words) == 0:
            for _ in range(len(source)):
                self.words.append(List[Int]())
        if 2 * (len(self.hashes) + 1) > len(self.slots):
            self._grow()
        var h = _word_hash(source, row)
        var mask = len(self.slots) - 1
        var slot = Int(h) & mask
        while True:
            var id = self.slots[slot]
            if id < 0:
                id = len(self.hashes)
                self.slots[slot] = id
                self.hashes.append(h)
                for w in range(len(source)):
                    self.words[w].append(source[w][row])
                return id
            if self.hashes[id] == h:
                var same = True
                for w in range(len(source)):
                    if self.words[w][id] != source[w][row]:
                        same = False
                        break
                if same:
                    return id
            slot = (slot + 1) & mask


struct _StreamMergeJob(Job):
    """Merge some hash parts' pending batch states into their accumulated
    states. Parts never share a key, so they merge independently (#326)."""

    var states: List[_StreamReduction]
    var pending: List[List[_StreamReduction]]

    def __init__(out self):
        self.states = List[_StreamReduction]()
        self.pending = List[List[_StreamReduction]]()

    def run(mut self) raises:
        for i in range(len(self.states)):
            self.states[i].merge_all(self.pending[i], parallel=False)
        self.pending = List[List[_StreamReduction]]()


struct _FinishJob(Job):
    """Finish one hash part's state into its output frame."""

    var state: _StreamReduction
    var frame: DataFrame

    def __init__(out self, var state: _StreamReduction) raises:
        self.state = state^
        self.frame = DataFrame(List[Series](), height=0)

    def run(mut self) raises:
        self.frame = self.state.finish()


def _finish_parts(var parts: List[_StreamReduction]) raises -> DataFrame:
    """Finish every hash part and interleave their groups by first
    occurrence, the order a single state would have produced.

    Parts finish on their own threads. Each part's groups are already in
    first-occurrence order, so the interleave is a k-way merge on `firsts`
    with a binary heap of part cursors.
    """
    var bases = List[Int](capacity=len(parts))
    var total = 0
    for p in range(len(parts)):
        bases.append(total)
        total += parts[p].group_count()
    var firsts = List[List[Int]](capacity=len(parts))
    var jobs = List[_FinishJob](capacity=len(parts))
    while len(parts) > 0:
        var part = parts.pop(0)
        firsts.append(part.firsts.copy())
        jobs.append(_FinishJob(part^))
    run_jobs(jobs)
    var frames = List[DataFrame](capacity=len(jobs))
    for j in range(len(jobs)):
        frames.append(jobs[j].frame.copy())
    var sizes = List[Int](capacity=len(firsts))
    for p in range(len(firsts)):
        sizes.append(len(firsts[p]))
    # Heap entries: the part and its next first-occurrence row, cached so a
    # comparison reads two flat lists.
    var heap_part = List[Int]()
    var heap_key = List[Int]()
    var cursor = List[Int](length=len(sizes), fill=0)
    for p in range(len(sizes)):
        if sizes[p] > 0:
            heap_part.append(p)
            heap_key.append(firsts[p][0])
    var size = len(heap_part)

    @always_inline
    def sift_down(
        mut part: List[Int], mut keys: List[Int], size: Int, start: Int
    ):
        var i = start
        while True:
            var smallest = i
            var left = 2 * i + 1
            var right = left + 1
            if left < size and keys[left] < keys[smallest]:
                smallest = left
            if right < size and keys[right] < keys[smallest]:
                smallest = right
            if smallest == i:
                return
            var swap_key = keys[i]
            keys[i] = keys[smallest]
            keys[smallest] = swap_key
            var swap_part = part[i]
            part[i] = part[smallest]
            part[smallest] = swap_part
            i = smallest

    for i in range(size // 2 - 1, -1, -1):
        sift_down(heap_part, heap_key, size, i)
    var order = List[Int](capacity=total)
    while size > 0:
        var part = heap_part[0]
        order.append(bases[part] + cursor[part])
        cursor[part] += 1
        if cursor[part] == sizes[part]:
            size -= 1
            heap_part[0] = heap_part[size]
            heap_key[0] = heap_key[size]
        else:
            heap_key[0] = firsts[part][cursor[part]]
        sift_down(heap_part, heap_key, size, 0)
    return concat(frames^).rechunk().take(order^)


struct _StreamReduction(Movable):
    var keys: DataFrame
    var states: List[Reducer]
    var names: List[String]
    var dtypes: List[DataType]
    var outputs: List[Expr]
    var grouped: Bool
    # Row of each group's first occurrence, relative to the reduced frame
    # until the stream shifts it to the whole input; restores group order
    # when the state is split by key hash (#326).
    var firsts: List[Int]
    # Rows the state was reduced from.
    var rows: Int
    # Persistent key index for merges: 0 not built yet, 1 in use, -1 given
    # up (a key the word encoding cannot hold, such as a long string).
    var index: _KeyIndex
    var indexing: Int

    def __init__(
        out self,
        *,
        var keys: DataFrame,
        var states: List[Reducer],
        names: List[String],
        dtypes: List[DataType],
        outputs: List[Expr],
        grouped: Bool,
        var firsts: List[Int],
        rows: Int,
    ):
        self.keys = keys^
        self.states = states^
        self.names = names.copy()
        self.dtypes = dtypes.copy()
        self.outputs = outputs.copy()
        self.grouped = grouped
        self.firsts = firsts^
        self.rows = rows
        self.index = _KeyIndex()
        self.indexing = 0

    def __init__(
        out self, frame: DataFrame, expressions: List[Expr], names: List[String]
    ) raises:
        var keys = List[Series]()
        if len(names) > 0:
            keys = _expand_struct_keys(frame._subset_keys(names))
        var key_names = Dict[String, Bool]()
        for name in names:
            key_names[name] = True
        self = Self(frame, expressions, keys, key_names)

    def __init__(
        out self,
        frame: DataFrame,
        expressions: List[Expr],
        keys: List[Series],
        key_names: Dict[String, Bool],
    ) raises:
        """Reduce every row of frame, grouped by keys (already expanded and
        row-aligned with frame; none means one global group)."""
        self.grouped = len(keys) > 0
        self.rows = frame.height()
        self.index = _KeyIndex()
        self.indexing = 0
        var ids = List[Int]()
        var count = 1
        if self.grouped:
            var groups = encode_rows(keys, nulls_equal=True)
            count = groups.count()
            ids = groups.ids.copy()
            var columns = List[Series]()
            for key in keys:
                columns.append(key.take(groups.representatives.copy()))
            self.keys = DataFrame(columns^, height=count)
            self.firsts = groups.representatives.copy()
        else:
            self.keys = DataFrame(List[Series](), height=1)
            self.firsts = [0]
        self.outputs = List[Expr]()
        self.states = List[Reducer]()
        self.names = List[String]()
        self.dtypes = List[DataType]()
        var shared = ArcPointer(ids^)
        for expression in expressions:
            var bound = bind(expression, frame._columns)
            if bound.shape() != AGGREGATE:
                raise Error(
                    "Streaming aggregate requires scalar aggregate expressions"
                )
            if expression._name in key_names:
                raise Error(
                    "Aggregate output name collides with grouping key: "
                    + expression._name
                )
            var rewritten = expression.copy()
            for i in range(len(expression._nodes)):
                if not is_reduction(expression._nodes[i].op):
                    continue
                var job = _ReduceJob[8](
                    bound,
                    frame._columns,
                    List[Series](),
                    expression._nodes[i],
                    0,
                    frame.height(),
                    1024,
                    self.grouped,
                    shared,
                    count,
                )
                job.run()
                var name = "__stream_reduction_" + String(len(self.states))
                self.states.append(job^.into_reducer())
                self.names.append(name)
                self.dtypes.append(bound.dtypes[i])
                rewritten._nodes[i] = col(name)._nodes[0].copy()
            self.outputs.append(subtree(rewritten, len(rewritten._nodes) - 1))

    def merge_all(mut self, others: List[Self], parallel: Bool = True) raises:
        """Merge several partial states in one pass.

        The accumulated keys and every other state's keys are encoded
        together once, so merging k batches costs time in proportion to
        their groups plus the accumulated ones, not k times the accumulated
        groups. Accumulated keys come first and are distinct, so existing
        groups keep their ids and new ones follow in first-occurrence order.
        """
        if len(others) == 0:
            return
        if self.grouped and self.indexing >= 0:
            if self._indexable(others):
                self._merge_indexed(others)
                return
            self.indexing = -1
            self.index = _KeyIndex()
        var count = 1
        var mappings = List[List[Int]](capacity=len(others))
        if self.grouped:
            var parts = List[DataFrame](capacity=len(others) + 1)
            parts.append(self.keys.copy())
            for j in range(len(others)):
                parts.append(others[j].keys.copy())
            var combined = concat(parts^)
            var groups = _encode_in_order(
                combined._columns
            ) if parallel else encode_rows(combined._columns, nulls_equal=True)
            count = groups.count()
            var start = self.keys.height()
            for j in range(len(others)):
                var height = others[j].keys.height()
                var mapping = List[Int](capacity=height)
                for i in range(height):
                    mapping.append(groups.ids[start + i])
                start += height
                mappings.append(mapping^)
            var firsts = self.firsts.copy()
            for j in range(len(others)):
                firsts.extend(others[j].firsts.copy())
            var kept = List[Int](capacity=count)
            for r in groups.representatives:
                kept.append(firsts[r])
            self.firsts = kept^
            self.keys = combined.take(groups.representatives^)
        else:
            for _ in range(len(others)):
                mappings.append(List[Int]())
        for i in range(len(self.states)):
            self.states[i].grow(count)
            for j in range(len(others)):
                self.states[i].merge(others[j].states[i], mappings[j])

    def _indexable(self, others: List[Self]) raises -> Bool:
        if self.keys.width() == 0 or self.keys.width() > 32:
            return False
        if self.indexing == 0:
            for column in self.keys._columns:
                if not encodable(column):
                    return False
        for j in range(len(others)):
            for column in others[j].keys._columns:
                if not encodable(column):
                    return False
        return True

    def _merge_indexed(mut self, others: List[Self]) raises:
        """Merge through the persistent key index: each other state's
        groups are probed once, new ones appended in first-occurrence
        order, and the accumulated keys are never encoded again."""
        if self.indexing == 0:
            if self.keys.height() > 0:
                self.index = _KeyIndex(_equality_words(self.keys._columns))
            self.indexing = 1
        var key_parts = List[DataFrame](capacity=len(others) + 1)
        key_parts.append(self.keys.copy())
        var mappings = List[List[Int]](capacity=len(others))
        for j in range(len(others)):
            ref other = others[j]
            var n = other.keys.height()
            var mapping = List[Int](capacity=n)
            if n == 0:
                mappings.append(mapping^)
                continue
            var words = _equality_words(other.keys._columns)
            var added = List[Int]()
            for g in range(n):
                var before = self.index.count()
                var id = self.index.find_or_insert(words, g)
                mapping.append(id)
                if id == before:
                    added.append(g)
                    self.firsts.append(other.firsts[g])
            if len(added) > 0:
                key_parts.append(other.keys.take(added))
            mappings.append(mapping^)
        if len(key_parts) > 1:
            self.keys = concat(key_parts^)
            if self.keys._columns[0].n_chunks() > 64:
                self.keys = self.keys.rechunk()
        var count = self.index.count()
        for i in range(len(self.states)):
            self.states[i].grow(count)
            for j in range(len(others)):
                self.states[i].merge(others[j].states[i], mappings[j])

    def group_count(self) -> Int:
        return self.keys.height()

    def shift_firsts(mut self, offset: Int):
        for i in range(len(self.firsts)):
            self.firsts[i] += offset

    def split(self, bits: Int) raises -> List[Self]:
        """This state's groups divided into 2^bits parts by key hash.

        The hash is the partitioner's (partition.mojo), so equal keys land
        in the same part in every batch, and each part can be merged on its
        own. Each part keeps its groups' first-occurrence rows.
        """
        var count = 1 << bits
        var n = self.keys.height()
        var members = List[List[Int]](length=count, fill=List[Int]())
        if n > 0:
            var hashed = Partitioner(self.keys._columns, 1)
            var shift = UInt64(64 - bits)
            for g in range(n):
                members[Int(hashed.hashes[g] >> shift)].append(g)
        var parts = List[Self](capacity=count)
        for p in range(count):
            var states = List[Reducer](capacity=len(self.states))
            for i in range(len(self.states)):
                ref source = self.states[i]
                var state = Reducer(
                    source.op,
                    source.input,
                    len(members[p]),
                    source.min_count,
                    source.integer,
                    source.floating,
                    source.text,
                    source.logical,
                )
                if len(members[p]) > 0:
                    state.merge(source, sources=members[p])
                states.append(state^)
            var firsts = List[Int](capacity=len(members[p]))
            for g in members[p]:
                firsts.append(self.firsts[g])
            parts.append(
                Self(
                    keys=self.keys.take(members[p]),
                    states=states^,
                    names=self.names,
                    dtypes=self.dtypes,
                    outputs=self.outputs,
                    grouped=True,
                    firsts=firsts^,
                    rows=0,
                )
            )
        return parts^

    def merge(mut self, other: Self) raises:
        var mapping = List[Int]()
        var count = 1
        if self.grouped:
            var old = self.keys.height()
            var combined = concat([self.keys.copy(), other.keys.copy()])
            var groups = encode_rows(combined._columns, nulls_equal=True)
            count = groups.count()
            for i in range(other.keys.height()):
                mapping.append(groups.ids[old + i])
            var firsts = self.firsts.copy()
            for f in other.firsts:
                firsts.append(f + self.rows)
            var kept = List[Int](capacity=count)
            for r in groups.representatives:
                kept.append(firsts[r])
            self.firsts = kept^
            self.keys = combined.take(groups.representatives^)
        self.rows += other.rows
        for i in range(len(self.states)):
            self.states[i].grow(count)
            self.states[i].merge(other.states[i], mapping)

    def finish(self) raises -> DataFrame:
        var reduced = List[Series]()
        for i in range(len(self.states)):
            reduced.append(
                self.states[i]
                .finish()
                .with_dtype(self.dtypes[i])
                .renamed(self.names[i])
            )
        var values = DataFrame(
            reduced^, height=self.keys.height()
        ).select_exprs(self.outputs)
        var columns = self.keys._columns.copy()
        for column in values._columns:
            columns.append(column.copy())
        return _pack_struct_keys(DataFrame(columns^, height=self.keys.height()))


def _round_ticks(value: Float64) -> Int64:
    """Nearest integer tick, halves away from zero."""
    return Int64(value + 0.5) if value >= 0 else -Int64(-value + 0.5)


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
