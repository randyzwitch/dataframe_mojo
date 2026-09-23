"""An eager CPU dataframe with runtime schema and positional row semantics."""
from .dtype import DataType
from std.collections import Dict
from std.memory import ArcPointer, Pointer
from .bool_column import BoolColumn
from .column import Column, _append_validity
from .string_column import StringColumn, StringBuilder
from .series import Series, sort_indices, smallest_indices
from .expr import (
    Expr,
    col,
    lit,
    COL,
    SELECTOR,
    LIT_INT,
    LIT_FLOAT,
    LIT_BOOL,
    LIT_STRING,
    LIT_NULL,
    UNTYPED,
)
from .binding import bind, BoundExpr, ROWS, AGGREGATE
from .execution import evaluate
from .gather import take_parallel, take_sorted_chunked, true_rows
from .parallel import Job, partitions, run_jobs, worker_count
from .partition import Partitioner, encode_partitioned, low_cardinality
from .row_encode import encodable, encode_sort_keys
from .value import AnyValue
from .hashing import RowKeys, encode_rows
from .groups import GroupIndices
from .expr_kernels import choose, validity
from .selectors import expand, expand_all
from .lazy import LazyFrame
from .display import render_frame, render_glimpse


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

    def take(self, indices: List[Int]) raises -> Self:
        for i in indices:
            if i < 0 or i >= self._height:
                raise Error("Row index out of bounds")
        var workers = worker_count(len(indices))
        if workers > 1 and self.width() > 0:
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
        """Keep true rows, dropping false and null mask entries, in input order."""
        if len(mask) != self._height:
            raise Error("Filter mask must match dataframe height")
        var rows = true_rows(mask)
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
        if how == "cross":
            raise Error(
                "A cross join takes no keys; use join(right, how='cross')"
            )
        if (
            how != "inner"
            and how != "left"
            and how != "right"
            and how != "full"
            and how != "semi"
            and how != "anti"
        ):
            raise Error(
                "Join how must be inner, left, right, full, semi, anti, or"
                " cross"
            )
        if len(left_on) == 0 or len(left_on) != len(right_on):
            raise Error(
                "Join requires the same nonzero number of left and right keys"
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
        var keep_right_keys = how == "full" and not coalesce
        var right_output = List[Int]()
        var right_names = List[String]()
        if how != "semi" and how != "anti":
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
        var ids = _joint_key_ids(self, right, left_keys, right_keys)
        var left_ids = ids[0].copy()
        var right_ids = ids[1].copy()
        var count = ids[2]
        # Rows per key id in a flat CSR layout: one list per distinct key
        # would be hundreds of thousands of heap allocations on a
        # high-cardinality join. Ids appear in increasing row order, so a
        # counting sort preserves the match order the contract documents.
        var right_starts = _group_index(right_ids, count)
        var csr_workers = worker_count(len(right_ids))
        # The stable range scatter adds an order list and one cursor per key.
        # It wins only once its parallel scatter repays those buffers; keep
        # the compact serial CSR below the measured 2M-right-row crossover.
        var right_flat = _parallel_group_rows(
            right_ids, right_starts, csr_workers
        ) if (
            how == "inner" and csr_workers > 1 and len(right_ids) >= 2_000_000
        ) else _group_rows(
            right_ids, right_starts
        )
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        if how == "semi" or how == "anti":
            for i in range(len(left_ids)):
                var id = left_ids[i]
                var matched = (
                    id >= 0 and right_starts[id + 1] > right_starts[id]
                )
                if matched == (how == "semi"):
                    left_rows.append(i)
            return self.take(left_rows)
        if how == "right":
            var left_starts = _group_index(left_ids, count)
            var left_flat = _group_rows(left_ids, left_starts)
            for j in range(len(right_ids)):
                var id = right_ids[j]
                if id >= 0 and left_starts[id + 1] > left_starts[id]:
                    for k in range(left_starts[id], left_starts[id + 1]):
                        left_rows.append(left_flat[k])
                        right_rows.append(j)
                else:
                    left_rows.append(-1)
                    right_rows.append(j)
        elif how == "inner" and worker_count(len(left_ids)) > 1:
            var pairs = _parallel_inner_rows(
                left_ids, right_starts, right_flat, worker_count(len(left_ids))
            )
            left_rows = pairs[0].copy()
            right_rows = pairs[1].copy()
        else:
            var right_matched = List[Bool](length=len(right_ids), fill=False)
            for i in range(len(left_ids)):
                var id = left_ids[i]
                if id >= 0 and right_starts[id + 1] > right_starts[id]:
                    for k in range(right_starts[id], right_starts[id + 1]):
                        var j = right_flat[k]
                        left_rows.append(i)
                        right_rows.append(j)
                        right_matched[j] = True
                elif how != "inner":
                    left_rows.append(i)
                    right_rows.append(-1)
            if how == "full":
                for j in range(len(right_ids)):
                    if not right_matched[j]:
                        left_rows.append(-1)
                        right_rows.append(j)
        # Assembling the output is about half of a join, and it used to
        # gather one column at a time on the calling thread. take_parallel
        # writes disjoint output ranges, so every column of both sides goes
        # at once; `or_null` is what lets it carry the -1 that means "no row
        # on this side".
        var gather_workers = worker_count(len(left_rows))
        var sides_mixed = how == "right" or how == "full"
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
                columns^, left_rows.copy(), gather_workers, or_null=True
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
            right_sources, right_rows.copy(), gather_workers, or_null=True
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
        if how != "cross":
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
        var predicates = expand(predicate, self._columns)
        if len(predicates) != 1:
            raise Error("A filter selector must match exactly one column")
        var bound = bind(predicates[0], self._columns)
        if bound.dtypes[len(bound.dtypes) - 1] != DataType.BOOL:
            raise Error("Filter expression must return Boolean values")
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
        return keys^

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
        return GroupBy(self.copy(), columns^, maintain_order)

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
        var encoded = encode_rows(columns, True)
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
        return GroupBy(self.copy(), columns^, maintain_order)


def _as_int64(values: List[Int]) -> List[Int64]:
    var out = List[Int64](capacity=len(values))
    for v in values:
        out.append(Int64(v))
    return out^


def _group_index(ids: List[Int], count: Int) -> List[Int]:
    """Start offset per key id, in a flat CSR layout (count + 1 entries)."""
    var starts = List[Int](length=count + 1, fill=0)
    for id in ids:
        if id >= 0:
            starts[id + 1] += 1
    for g in range(count):
        starts[g + 1] += starts[g]
    return starts^


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


comptime _RANGE_JOIN_MAX_IDS = 8_000_000


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
    var cap = _RANGE_JOIN_MAX_IDS
    if total < cap // 4:
        cap = total * 4
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

    def __init__(
        out self,
        left_ids: ArcPointer[List[Int]],
        right_starts: ArcPointer[List[Int]],
        start: Int,
        end: Int,
    ):
        self.left_ids = left_ids.copy()
        self.right_starts = right_starts.copy()
        self.start = start
        self.end = end
        self.count = 0

    def run(mut self) raises:
        for row in range(self.start, self.end):
            var key = self.left_ids[][row]
            if key >= 0:
                var matches = (
                    self.right_starts[][key + 1] - self.right_starts[][key]
                )
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
    ):
        self.left_ids = left_ids.copy()
        self.right_starts = right_starts.copy()
        self.right_flat = right_flat.copy()
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
            var key = self.left_ids[][row]
            if key >= 0:
                for at in range(
                    self.right_starts[][key], self.right_starts[][key + 1]
                ):
                    left_rows[output] = row
                    right_rows[output] = self.right_flat[][at]
                    output += 1


def _parallel_inner_rows(
    left_ids: List[Int],
    right_starts: List[Int],
    right_flat: List[Int],
    workers: Int,
) raises -> Tuple[List[Int], List[Int]]:
    """Inner pairs in the same left-major order as the serial loop.

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
            )
        )
    run_jobs(fills)
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
                        + schema[c].dtype.name()
                        + ", expected "
                        + first[c].name
                        + ":"
                        + first[c].dtype.name()
                    )
            height += frames[f].height()
        # Polars accumulate_dataframes_vertical / vstack_mut_owned:
        # append Arrow array references; keep the output multi-chunk.
        var columns = List[Series](capacity=len(first))
        for c in range(len(first)):
            var parts = List[Series](capacity=len(frames))
            for frame in frames:
                parts.append(frame._columns[c].copy())
            columns.append(Series._from_chunks(parts))
        return DataFrame(columns^, height=height)
    if how == "diagonal":
        var names = List[String]()
        var dtypes = Dict[String, DataType]()
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
                        + field.dtype.name()
                        + " in frame "
                        + String(f)
                        + ", expected "
                        + dtypes[field.name].name()
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
        var workers = worker_count(self._frame.height())
        if workers > 1:
            return self._agg_partitioned(
                expressions, bound, batch_size, workers
            )
        return self._agg_whole(bound, batch_size)

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
        if low_cardinality(self._keys):
            return self._agg_whole(bound, batch_size)
        var partitioner = Partitioner(self._keys, workers)
        var parts = partitioner.scatter(workers)
        var buckets = parts.buckets()
        var keys = take_parallel(self._keys, parts.order.copy(), workers)
        var columns = take_parallel(
            self._referenced(bound), parts.order.copy(), workers
        )
        # Heavy: a bucket big enough to serialize the batch on its own. It
        # must be large relative to the frame, not only to its share, or
        # low cardinality (few occupied buckets) would count as heavy.
        var heavy_rows = max(height // 8, 4 * height // buckets)
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
