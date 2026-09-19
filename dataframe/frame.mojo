"""An eager CPU dataframe with runtime schema and positional row semantics."""
from std.collections import Dict
from .column import Column
from .series import Series, sort_indices, smallest_indices
from .kernels import checked_add
from .expr import Expr
from .binding import bind, BoundExpr, ROWS, AGGREGATE
from .execution import evaluate
from .value import AnyValue
from .hashing import RowKeys, encode_rows
from .display import render_frame, render_glimpse


@fieldwise_init
struct Field(Copyable):
    var name: String
    var dtype: String


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

    def dtypes(self) -> List[String]:
        var types = List[String](capacity=self.width())
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

    def get_column(self, name: String) raises -> Series:
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

    def take(self, indices: List[Int]) raises -> Self:
        for i in indices:
            if i < 0 or i >= self._height:
                raise Error("Row index out of bounds")
        var columns = List[Series](capacity=self.width())
        for column in self._columns:
            columns.append(column.take(indices))
        return Self(columns^, height=len(indices))

    def filter(self, mask: Column[Bool]) raises -> Self:
        """Keep true rows, dropping false and null mask entries, in input order."""
        if len(mask) != self._height:
            raise Error("Filter mask must match dataframe height")
        var indices = List[Int]()
        for i in range(self._height):
            if not mask.is_null(i) and mask.value(i):
                indices.append(i)
        return self.take(indices)

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

    def group_by_sum(
        self, key: String, value: String, output: String = "sum"
    ) raises -> Self:
        """Sum one numeric column by one string key, in first-seen order.

        Null keys form a group. Empty/all-null groups produce null sums.
        Int64 sums are checked, Float64 sums accumulate in input order.
        """
        if output == key:
            raise Error(
                "Aggregate output name must differ from the grouping key"
            )
        var key_index = self._index(key)
        var value_index = self._index(value)
        var keys = self._columns[key_index].string()
        var dtype = self._columns[value_index].dtype()
        if dtype != "int64" and dtype != "float64":
            raise Error("Grouped sum requires an int64 or float64 value column")
        var lookup = Dict[String, Int]()
        var representatives = List[Int]()
        var groups = List[Int](capacity=self._height)
        var null_group = -1
        for i in range(self._height):
            var group: Int
            if keys.is_null(i):
                if null_group == -1:
                    null_group = len(representatives)
                    representatives.append(i)
                group = null_group
            else:
                var label = keys.value(i)
                if label not in lookup:
                    lookup[label] = len(representatives)
                    representatives.append(i)
                group = lookup[label]
            groups.append(group)
        var result = List[Series]()
        result.append(Series(key, keys.take(representatives)))
        var valid = List[Bool](length=len(representatives), fill=False)
        if dtype == "int64":
            var values = self._columns[value_index].int64()
            var totals = List[Int64](length=len(representatives), fill=0)
            for i in range(self._height):
                if not values.is_null(i):
                    var g = groups[i]
                    totals[g] = checked_add(totals[g], values.value(i))
                    valid[g] = True
            result.append(Series(output, Column[Int64](totals^, valid)))
        else:
            var values = self._columns[value_index].float64()
            var totals = List[Float64](length=len(representatives), fill=0)
            for i in range(self._height):
                if not values.is_null(i):
                    var g = groups[i]
                    totals[g] += values.value(i)
                    valid[g] = True
            result.append(Series(output, Column[Float64](totals^, valid)))
        return Self(result^)

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
        return sort_indices(self._sort_ranks(by, descending, nulls_last))

    def top_k(self, k: Int, by: List[String]) raises -> Self:
        """The k rows that sort(by, descending=True) would put first.

        Nulls rank last. Selection is O(n log k) rather than a full sort.
        """
        var n = len(by)
        return self.take(
            smallest_indices(
                self._sort_ranks(
                    by,
                    List[Bool](length=n, fill=True),
                    List[Bool](length=n, fill=True),
                ),
                k,
            )
        )

    def top_k(self, k: Int, by: String) raises -> Self:
        return self.top_k(k, [by])

    def bottom_k(self, k: Int, by: List[String]) raises -> Self:
        """The k rows that sort(by) would put first; nulls rank last."""
        var n = len(by)
        return self.take(
            smallest_indices(
                self._sort_ranks(
                    by,
                    List[Bool](length=n, fill=False),
                    List[Bool](length=n, fill=True),
                ),
                k,
            )
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
    ) raises -> Self:
        """Hash join on one shared string column; 'inner' and 'left' supported.

        Null keys never match. Emit left rows in input order and each row's
        right matches in right input order. Duplicate keys produce all pairs.
        Overlapping right names gain suffix; remaining collisions raise.
        """
        if how != "inner" and how != "left":
            raise Error("Join how must be 'inner' or 'left'")
        var left_key_index = self._index(on)
        var right_key_index = right._index(on)
        var left_keys = self._columns[left_key_index].string()
        var right_keys = right._columns[right_key_index].string()
        var names = Dict[String, Bool]()
        for column in self._columns:
            names[column.name()] = True
        var left_names = names.copy()
        var right_names = List[String]()
        for i in range(right.width()):
            if i == right_key_index:
                continue
            var name = right._columns[i].name()
            if name in left_names:
                name += suffix
            # Validate against both original left and preceding output names.
            if name in names:
                raise Error("Join output name collision: " + name)
            names[name] = True
            right_names.append(name)
        var lookup = Dict[String, List[Int]]()
        for i in range(right.height()):
            if not right_keys.is_null(i):
                var key = right_keys.value(i)
                if key not in lookup:
                    lookup[key] = List[Int]()
                lookup[key].append(i)
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        for i in range(self.height()):
            var matched = False
            if not left_keys.is_null(i):
                var key = left_keys.value(i)
                if key in lookup:
                    for j in lookup[key]:
                        left_rows.append(i)
                        right_rows.append(j)
                    matched = True
            if not matched and how == "left":
                left_rows.append(i)
                right_rows.append(-1)
        var columns = List[Series]()
        for column in self._columns:
            columns.append(column.take(left_rows))
        var name_index = 0
        for i in range(right.width()):
            if i != right_key_index:
                columns.append(
                    right._columns[i]
                    .take_or_null(right_rows)
                    .renamed(right_names[name_index])
                )
                name_index += 1
        return Self(columns^, height=len(left_rows))

    def select(
        self, expression: Expr, *, batch_size: Int = 1024
    ) raises -> Self:
        return self.select_exprs([expression.copy()], batch_size=batch_size)

    def select_exprs(
        self, expressions: List[Expr], *, batch_size: Int = 1024
    ) raises -> Self:
        """Evaluate against the original frame. Scalar-only output has one row.

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
        for expression in bound:
            var result = evaluate(
                expression, self._columns, self._height, batch_size=batch_size
            )
            if expression.shape() != ROWS and has_rows:
                result = result._broadcast(height)
            columns.append(result^)
        return Self(columns^, height=height)

    def with_columns(
        self, expression: Expr, *, batch_size: Int = 1024
    ) raises -> Self:
        return self.with_columns([expression.copy()], batch_size=batch_size)

    def with_columns(
        self, expressions: List[Expr], *, batch_size: Int = 1024
    ) raises -> Self:
        """All siblings see the original schema and data; aliases are outputs."""
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

    def filter(self, predicate: Expr, *, batch_size: Int = 1024) raises -> Self:
        var bound = bind(predicate, self._columns)
        if bound.dtypes[len(bound.dtypes) - 1] != "bool":
            raise Error("Filter expression must return Boolean values")
        var result = evaluate(
            bound, self._columns, self._height, batch_size=batch_size
        )
        if bound.shape() != ROWS:
            result = result._broadcast(self._height)
        return self.filter(result._data[Column[Bool]])

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
        return GroupBy(self.copy(), columns^, maintain_order)

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
        return GroupBy(self.copy(), columns^, maintain_order)


def concat(
    frames: List[DataFrame], how: String = "vertical"
) raises -> DataFrame:
    """Combine frames: 'vertical', 'diagonal', or 'horizontal'.

    Vertical requires identical names, order, and dtypes. Diagonal unions
    columns by name in first-seen order and fills missing columns with nulls;
    shared names must share a dtype. Horizontal requires equal heights and
    unique names. All inputs are validated before any column is built.
    """
    if len(frames) == 0:
        raise Error("concat requires at least one dataframe")
    if how == "vertical":
        var first = frames[0].schema()
        var height = 0
        for f in range(len(frames)):
            var schema = frames[f].schema()
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
                        + schema[c].dtype
                        + ", expected "
                        + first[c].name
                        + ":"
                        + first[c].dtype
                    )
            height += frames[f].height()
        var columns = List[Series](capacity=len(first))
        for c in range(len(first)):
            var column = frames[0]._columns[c].copy()
            for f in range(1, len(frames)):
                column._append_series(frames[f]._columns[c])
            columns.append(column^)
        return DataFrame(columns^, height=height)
    if how == "diagonal":
        var names = List[String]()
        var dtypes = Dict[String, String]()
        for f in range(len(frames)):
            for field in frames[f].schema():
                if field.name not in dtypes:
                    dtypes[field.name] = field.dtype
                    names.append(field.name)
                elif dtypes[field.name] != field.dtype:
                    raise Error(
                        "concat diagonal: column "
                        + field.name
                        + " is "
                        + field.dtype
                        + " in frame "
                        + String(f)
                        + ", expected "
                        + dtypes[field.name]
                    )
        var aligned = List[DataFrame](capacity=len(frames))
        for frame in frames:
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
        var height = frames[0].height()
        var seen = Dict[String, Bool]()
        for f in range(len(frames)):
            if frames[f].height() != height:
                raise Error(
                    "concat horizontal: frame "
                    + String(f)
                    + " has height "
                    + String(frames[f].height())
                    + ", expected "
                    + String(height)
                )
            for column in frames[f]._columns:
                if column.name() in seen:
                    raise Error(
                        "concat horizontal: duplicate column "
                        + column.name()
                        + " in frame "
                        + String(f)
                    )
                seen[column.name()] = True
        var columns = List[Series]()
        for frame in frames:
            for column in frame._columns:
                columns.append(column.copy())
        return DataFrame(columns^, height=height)
    raise Error("concat how must be 'vertical', 'diagonal', or 'horizontal'")


def _bind_all(
    expressions: List[Expr], columns: List[Series]
) raises -> List[BoundExpr]:
    var bound = List[BoundExpr]()
    var names = Dict[String, Bool]()
    for expression in expressions:
        if expression._name in names:
            raise Error("Duplicate expression output name: " + expression._name)
        names[expression._name] = True
        bound.append(bind(expression, columns))
    return bound^


@fieldwise_init
struct GroupBy(Copyable):
    """An eager grouping request. No per-group dataframe materialization.

    It owns a snapshot of the input and the evaluated key columns.
    """

    var _frame: DataFrame
    var _keys: List[Series]
    var _maintain_order: Bool

    def _key_names(self) -> Dict[String, Bool]:
        var names = Dict[String, Bool]()
        for key in self._keys:
            names[key.name()] = True
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
        var groups = encode_rows(self._keys, nulls_equal=True)
        var columns = self._key_columns(groups)
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
        var columns = self._key_columns(groups)
        columns.append(Series(name, Column[Int64](counts^)))
        return DataFrame(columns^, height=groups.count())
