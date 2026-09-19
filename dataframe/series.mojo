"""Runtime-tagged, named columns without per-element type erasure."""
from .dtype import DataType
from std.utils import Variant
from .column import Column
from .value import AnyValue
from .display import render_series
from .cast import cast_series
from .expr import Expr, col, lit
from .frame import DataFrame

# Element types in DataType code order. Storage holds Column[E] for each, and
# methods dispatch with one compile-time loop over Elements instead of an
# if-chain per type; adding a dtype means extending both lists.
comptime Elements = Variant[Int64, Float64, Bool, String]
comptime Storage = Variant[
    Column[Int64], Column[Float64], Column[Bool], Column[String]
]


struct Series(Copyable, Sized, Writable):
    """A named column of one supported dtype, plus expression-backed methods."""

    var _name: String
    var _data: Storage
    # The logical type. Temporal types are stored in Column[Int64].
    var _dtype: DataType

    def __init__(out self, var name: String, var column: Column[Int64]):
        self._name = name^
        self._data = Storage(column^)
        self._dtype = DataType.INT64

    def __init__(out self, var name: String, var column: Column[Float64]):
        self._name = name^
        self._data = Storage(column^)
        self._dtype = DataType.FLOAT64

    def __init__(out self, var name: String, var column: Column[Bool]):
        self._name = name^
        self._data = Storage(column^)
        self._dtype = DataType.BOOL

    def __init__(out self, var name: String, var column: Column[String]):
        self._name = name^
        self._data = Storage(column^)
        self._dtype = DataType.STRING

    @staticmethod
    def _wrap[
        E: Copyable & Deinitable
    ](var name: String, var column: Column[E]) -> Self:
        """Build a series from any storable column type."""
        var result = Self(name^, Column[Int64]([]))
        result._data = Storage(column^)
        comptime for i in range(len(Elements.Ts)):
            comptime T: Copyable & Deinitable = Elements.Ts[i]
            if result._data.isa[Column[T]]():
                result._dtype = DataType(i, 0)
        return result^

    def with_dtype(self, dtype: DataType) raises -> Self:
        """The same values tagged with another logical type that shares their
        storage (temporal types and INT64)."""
        if dtype.physical() != self._dtype.physical():
            raise Error(
                "cannot tag "
                + self._dtype.name()
                + " storage as "
                + dtype.name()
            )
        var result = self.copy()
        result._dtype = dtype
        return result^

    def name(self) -> String:
        return self._name

    def write_to(self, mut writer: Some[Writer]):
        writer.write(render_series(self, 10, 32))

    def to_string(
        self, *, max_rows: Int = 10, max_string_length: Int = 32
    ) -> String:
        """Render at most max_rows values; negative means unlimited."""
        return render_series(self, max_rows, max_string_length)

    def cast(self, dtype: DataType, strict: Bool = True) raises -> Self:
        """Convert to another dtype; see Expr.cast."""
        return cast_series(self, dtype, strict, 0, List[Bool]())

    def cast(self, dtype: String, strict: Bool = True) raises -> Self:
        if not DataType.is_known(dtype):
            raise Error("Unknown cast dtype: " + dtype)
        return self.cast(DataType.parse(dtype), strict)

    def renamed(self, var name: String) -> Self:
        var result = self.copy()
        result._name = name^
        return result^

    def dtype(self) -> DataType:
        return self._dtype

    def __len__(self) -> Int:
        comptime for i in range(len(Elements.Ts)):
            comptime E: Copyable & Deinitable = Elements.Ts[i]
            if self._data.isa[Column[E]]():
                return len(self._data[Column[E]])
        return 0

    def null_count(self) -> Int:
        comptime for i in range(len(Elements.Ts)):
            comptime E: Copyable & Deinitable = Elements.Ts[i]
            if self._data.isa[Column[E]]():
                return self._data[Column[E]].null_count()
        return 0

    def get(self, index: Int) raises -> AnyValue:
        """Return one cell as a tagged value; raises when out of bounds."""
        if self._data.isa[Column[Int64]]():
            if self._data[Column[Int64]].is_null(index):
                return AnyValue.null(self._dtype)
            return AnyValue(
                self._dtype,
                True,
                self._data[Column[Int64]]._values[index],
                0,
                False,
                "",
            )
        if self._data.isa[Column[Float64]]():
            if self._data[Column[Float64]].is_null(index):
                return AnyValue.null(DataType.FLOAT64)
            return AnyValue(self._data[Column[Float64]]._values[index])
        if self._data.isa[Column[Bool]]():
            if self._data[Column[Bool]].is_null(index):
                return AnyValue.null(DataType.BOOL)
            return AnyValue(self._data[Column[Bool]]._values[index])
        if self._data[Column[String]].is_null(index):
            return AnyValue.null(DataType.STRING)
        return AnyValue(self._data[Column[String]]._values[index])

    def equals(
        self, other: Self, *, null_equal: Bool = True, check_names: Bool = False
    ) -> Bool:
        """Structural equality: NaN equals NaN and -0.0 equals 0.0.

        With null_equal=False, any null in either input makes them unequal.
        """
        if self.dtype() != other.dtype() or len(self) != len(other):
            return False
        if check_names and self._name != other._name:
            return False
        if not null_equal and (self.null_count() > 0 or other.null_count() > 0):
            return False
        if self._data.isa[Column[Int64]]():
            return _equal_columns(
                self._data[Column[Int64]], other._data[Column[Int64]]
            )
        if self._data.isa[Column[Float64]]():
            ref a = self._data[Column[Float64]]
            ref b = other._data[Column[Float64]]
            for i in range(len(a)):
                if a._valid(i) != b._valid(i):
                    return False
                if a._valid(i):
                    var x = a._values[i]
                    var y = b._values[i]
                    if x != y and not (x != x and y != y):
                        return False
            return True
        if self._data.isa[Column[Bool]]():
            return _equal_columns(
                self._data[Column[Bool]], other._data[Column[Bool]]
            )
        return _equal_columns(
            self._data[Column[String]], other._data[Column[String]]
        )

    # Expression-backed operations. Each evaluates the matching expression
    # over a one-column frame, so Series and expressions share every kernel
    # and contract. Binary operations pair rows positionally.

    def _frame(self) raises -> DataFrame:
        return DataFrame([self.copy()])

    def _apply(self, expr: Expr) raises -> Self:
        return self._frame().select(expr.alias(self._name)).column(self._name)

    def _pair(self, other: Self, op: String) raises -> Self:
        if len(self) != len(other):
            raise Error(
                "Series lengths differ: "
                + String(len(self))
                + " and "
                + String(len(other))
            )
        var frame = DataFrame([self.copy(), other.renamed("__right")])
        var left = col(self._name)
        var right = col("__right")
        var e: Expr
        if op == "+":
            e = left + right
        elif op == "-":
            e = left - right
        elif op == "*":
            e = left * right
        elif op == "/":
            e = left / right
        elif op == "//":
            e = left // right
        elif op == "%":
            e = left % right
        elif op == "**":
            e = left**right
        elif op == "<":
            e = left < right
        elif op == "<=":
            e = left <= right
        elif op == ">":
            e = left > right
        elif op == ">=":
            e = left >= right
        elif op == "eq":
            e = left.eq(right)
        elif op == "ne":
            e = left.ne(right)
        elif op == "&":
            e = left & right
        elif op == "|":
            e = left | right
        else:
            e = left ^ right
        return frame.select(e.alias(self._name)).column(self._name)

    def _scalar(self, expr: Expr) raises -> AnyValue:
        return self._frame().select(expr.alias(self._name)).item()

    def __add__(self, other: Self) raises -> Self:
        return self._pair(other, "+")

    def __add__(self, other: Expr) raises -> Self:
        return self._apply(col(self._name) + other)

    def __sub__(self, other: Self) raises -> Self:
        return self._pair(other, "-")

    def __sub__(self, other: Expr) raises -> Self:
        return self._apply(col(self._name) - other)

    def __mul__(self, other: Self) raises -> Self:
        return self._pair(other, "*")

    def __mul__(self, other: Expr) raises -> Self:
        return self._apply(col(self._name) * other)

    def __truediv__(self, other: Self) raises -> Self:
        return self._pair(other, "/")

    def __truediv__(self, other: Expr) raises -> Self:
        return self._apply(col(self._name) / other)

    def __floordiv__(self, other: Self) raises -> Self:
        return self._pair(other, "//")

    def __floordiv__(self, other: Expr) raises -> Self:
        return self._apply(col(self._name) // other)

    def __mod__(self, other: Self) raises -> Self:
        return self._pair(other, "%")

    def __mod__(self, other: Expr) raises -> Self:
        return self._apply(col(self._name) % other)

    def __pow__(self, other: Self) raises -> Self:
        return self._pair(other, "**")

    def __pow__(self, other: Expr) raises -> Self:
        return self._apply(col(self._name) ** other)

    def __neg__(self) raises -> Self:
        return self._apply(-col(self._name))

    def __lt__(self, other: Self) raises -> Self:
        return self._pair(other, "<")

    def __lt__(self, other: Expr) raises -> Self:
        return self._apply(col(self._name) < other)

    def __le__(self, other: Self) raises -> Self:
        return self._pair(other, "<=")

    def __le__(self, other: Expr) raises -> Self:
        return self._apply(col(self._name) <= other)

    def __gt__(self, other: Self) raises -> Self:
        return self._pair(other, ">")

    def __gt__(self, other: Expr) raises -> Self:
        return self._apply(col(self._name) > other)

    def __ge__(self, other: Self) raises -> Self:
        return self._pair(other, ">=")

    def __ge__(self, other: Expr) raises -> Self:
        return self._apply(col(self._name) >= other)

    def eq(self, other: Self) raises -> Self:
        return self._pair(other, "eq")

    def eq(self, other: Expr) raises -> Self:
        return self._apply(col(self._name).eq(other))

    def ne(self, other: Self) raises -> Self:
        return self._pair(other, "ne")

    def ne(self, other: Expr) raises -> Self:
        return self._apply(col(self._name).ne(other))

    def __and__(self, other: Self) raises -> Self:
        return self._pair(other, "&")

    def __or__(self, other: Self) raises -> Self:
        return self._pair(other, "|")

    def __xor__(self, other: Self) raises -> Self:
        return self._pair(other, "^")

    def __invert__(self) raises -> Self:
        return self._apply(~col(self._name))

    def __getitem__(self, index: Int) raises -> AnyValue:
        """One cell; raises when out of bounds. Negative indices count
        from the end."""
        return self.get(index + len(self) if index < 0 else index)

    def apply(self, expr: Expr) raises -> Self:
        """Evaluate any expression written against this series' name."""
        return self._apply(expr)

    def is_null(self) raises -> Self:
        return self._apply(col(self._name).is_null())

    def is_not_null(self) raises -> Self:
        return self._apply(col(self._name).is_not_null())

    def fill_null(self, value: Expr) raises -> Self:
        return self._apply(col(self._name).fill_null(value))

    def abs(self) raises -> Self:
        return self._apply(col(self._name).abs())

    def round(self, decimals: Int = 0) raises -> Self:
        return self._apply(col(self._name).round(decimals))

    def sum(self) raises -> AnyValue:
        return self._scalar(col(self._name).sum())

    def mean(self) raises -> AnyValue:
        return self._scalar(col(self._name).mean())

    def min(self) raises -> AnyValue:
        return self._scalar(col(self._name).min())

    def max(self) raises -> AnyValue:
        return self._scalar(col(self._name).max())

    def median(self) raises -> AnyValue:
        return self._scalar(col(self._name).median())

    def quantile(
        self, quantile: Float64, interpolation: String = "linear"
    ) raises -> AnyValue:
        return self._scalar(col(self._name).quantile(quantile, interpolation))

    def std(self, ddof: Int = 1) raises -> AnyValue:
        return self._scalar(col(self._name).std(ddof))

    def var(self, ddof: Int = 1) raises -> AnyValue:
        return self._scalar(col(self._name).var(ddof))

    def count(self) raises -> Int:
        return Int(self._scalar(col(self._name).count()).int64())

    def n_unique(self) raises -> Int:
        return Int(self._scalar(col(self._name).n_unique()).int64())

    def first(self) raises -> AnyValue:
        return self._scalar(col(self._name).first())

    def last(self) raises -> AnyValue:
        return self._scalar(col(self._name).last())

    def any(self, ignore_nulls: Bool = True) raises -> AnyValue:
        return self._scalar(col(self._name).any(ignore_nulls))

    def all(self, ignore_nulls: Bool = True) raises -> AnyValue:
        return self._scalar(col(self._name).all(ignore_nulls))

    def head(self, n: Int = 5) raises -> Self:
        return self._frame().head(n).column(self._name)

    def tail(self, n: Int = 5) raises -> Self:
        return self._frame().tail(n).column(self._name)

    def sort(
        self, descending: Bool = False, nulls_last: Bool = True
    ) raises -> Self:
        return self.take(self.argsort(descending, nulls_last))

    def unique(self, maintain_order: Bool = False) raises -> Self:
        """Distinct values, null counted once."""
        return (
            self._frame()
            .unique(keep="first", maintain_order=maintain_order)
            .column(self._name)
        )

    def value_counts(
        self, sort: Bool = True, name: String = "count"
    ) raises -> DataFrame:
        """Distinct values and their row counts (nulls included), most
        frequent first when sort=True; ties keep first-occurrence order."""
        var counts = (
            self._frame().group_by(self._name, maintain_order=True).len(name)
        )
        if sort:
            return counts.sort([name], descending=True)
        return counts^

    def to_values(self) raises -> List[AnyValue]:
        var values = List[AnyValue](capacity=len(self))
        for i in range(len(self)):
            values.append(self.get(i))
        return values^

    def int64(self) raises -> Column[Int64]:
        """Return an owned typed copy, raising on a dtype mismatch."""
        if not self._data.isa[Column[Int64]]():
            raise Error("Expected int64 column")
        return self._data[Column[Int64]].copy()

    def float64(self) raises -> Column[Float64]:
        """Return an owned typed copy, raising on a dtype mismatch."""
        if not self._data.isa[Column[Float64]]():
            raise Error("Expected float64 column")
        return self._data[Column[Float64]].copy()

    def bool(self) raises -> Column[Bool]:
        """Return an owned typed copy, raising on a dtype mismatch."""
        if not self._data.isa[Column[Bool]]():
            raise Error("Expected bool column")
        return self._data[Column[Bool]].copy()

    def string(self) raises -> Column[String]:
        """Return an owned typed copy, raising on a dtype mismatch."""
        if not self._data.isa[Column[String]]():
            raise Error("Expected string column")
        return self._data[Column[String]].copy()

    def take(self, indices: List[Int]) raises -> Self:
        var result = self._take_storage(indices)
        result._dtype = self._dtype
        return result^

    def _take_storage(self, indices: List[Int]) raises -> Self:
        comptime for i in range(len(Elements.Ts)):
            comptime E: Copyable & Deinitable = Elements.Ts[i]
            if self._data.isa[Column[E]]():
                return Self._wrap(
                    self._name, self._data[Column[E]].take(indices)
                )
        raise Error("Unknown column type")

    def take_or_null(self, indices: List[Int]) raises -> Self:
        var result = self._take_or_null_storage(indices)
        result._dtype = self._dtype
        return result^

    def _take_or_null_storage(self, indices: List[Int]) raises -> Self:
        if self._data.isa[Column[Int64]]():
            return Self(
                self._name,
                self._data[Column[Int64]].take_or_null(indices, Int64(0)),
            )
        if self._data.isa[Column[Float64]]():
            return Self(
                self._name,
                self._data[Column[Float64]].take_or_null(indices, Float64(0)),
            )
        if self._data.isa[Column[Bool]]():
            return Self(
                self._name,
                self._data[Column[Bool]].take_or_null(indices, False),
            )
        if self._data.isa[Column[String]]():
            return Self(
                self._name,
                self._data[Column[String]].take_or_null(indices, String("")),
            )
        raise Error("Unknown column type")

    def _less(
        self, a: Int, b: Int, descending: Bool, nulls_last: Bool
    ) raises -> Bool:
        if self._data.isa[Column[Int64]]():
            var a_null = self._data[Column[Int64]].is_null(a)
            var b_null = self._data[Column[Int64]].is_null(b)
            if a_null or b_null:
                return a_null != b_null and (b_null if nulls_last else a_null)
            var x = self._data[Column[Int64]].value(a)
            var y = self._data[Column[Int64]].value(b)
            return y < x if descending else x < y
        if self._data.isa[Column[Float64]]():
            var a_null = self._data[Column[Float64]].is_null(a)
            var b_null = self._data[Column[Float64]].is_null(b)
            if a_null or b_null:
                return a_null != b_null and (b_null if nulls_last else a_null)
            var x = self._data[Column[Float64]].value(a)
            var y = self._data[Column[Float64]].value(b)
            # NaNs follow finite/infinite values in either direction.
            if x != x or y != y:
                return x == x and y != y
            return y < x if descending else x < y
        if self._data.isa[Column[Bool]]():
            var a_null = self._data[Column[Bool]].is_null(a)
            var b_null = self._data[Column[Bool]].is_null(b)
            if a_null or b_null:
                return a_null != b_null and (b_null if nulls_last else a_null)
            var x = self._data[Column[Bool]].value(a)
            var y = self._data[Column[Bool]].value(b)
            return Int(y) < Int(x) if descending else Int(x) < Int(y)
        if self._data.isa[Column[String]]():
            var a_null = self._data[Column[String]].is_null(a)
            var b_null = self._data[Column[String]].is_null(b)
            if a_null or b_null:
                return a_null != b_null and (b_null if nulls_last else a_null)
            var x = self._data[Column[String]].value(a)
            var y = self._data[Column[String]].value(b)
            return y < x if descending else x < y
        raise Error("Unknown column type")

    def argsort(
        self, descending: Bool = False, nulls_last: Bool = True
    ) raises -> List[Int]:
        """Stable sort order: ranks are resolved once, then merged by Int."""
        return sort_indices([self._sort_ranks(descending, nulls_last)])

    def _sort_ranks(
        self, descending: Bool, nulls_last: Bool
    ) raises -> List[Int]:
        """Dense per-row ranks encoding direction, NaN, and null placement.

        Non-null, non-NaN values rank 0..d-1 (reversed when descending).
        NaN ranks d in either direction; null ranks -1 or d + 1.
        """
        var n = len(self)
        var ranks = List[Int](length=n, fill=0)
        var valid = List[Bool](length=n, fill=False)
        var nan = List[Bool](length=n, fill=False)
        var distinct: Int
        if self._data.isa[Column[Int64]]():
            ref column = self._data[Column[Int64]]
            for i in range(n):
                valid[i] = column._valid(i)
            distinct = _dense_ranks(column._values, valid, ranks)
        elif self._data.isa[Column[Float64]]():
            ref column = self._data[Column[Float64]]
            var usable = List[Bool](length=n, fill=False)
            for i in range(n):
                valid[i] = column._valid(i)
                var x = column._values[i]
                nan[i] = valid[i] and x != x
                usable[i] = valid[i] and not nan[i]
            distinct = _dense_ranks(column._values, usable, ranks)
        elif self._data.isa[Column[Bool]]():
            ref column = self._data[Column[Bool]]
            for i in range(n):
                valid[i] = column._valid(i)
                ranks[i] = Int(column._values[i])
            distinct = 2
        else:
            ref column = self._data[Column[String]]
            for i in range(n):
                valid[i] = column._valid(i)
            distinct = _dense_ranks(column._values, valid, ranks)
        for i in range(n):
            if not valid[i]:
                ranks[i] = distinct + 1 if nulls_last else -1
            elif nan[i]:
                ranks[i] = distinct
            elif descending:
                ranks[i] = distinct - 1 - ranks[i]
        return ranks^

    def _argsort_reference(
        self, descending: Bool = False, nulls_last: Bool = True
    ) raises -> List[Int]:
        """Comparator mergesort kept as a test oracle and benchmark baseline."""
        var indices = List[Int](capacity=len(self))
        for i in range(len(self)):
            indices.append(i)
        var scratch = indices.copy()
        var width = 1
        while width < len(self):
            var start = 0
            while start < len(self):
                var mid = min(start + width, len(self))
                var end = min(start + 2 * width, len(self))
                var left = start
                var right = mid
                for dest in range(start, end):
                    if left < mid and (
                        right >= end
                        or not self._less(
                            indices[right],
                            indices[left],
                            descending,
                            nulls_last,
                        )
                    ):
                        scratch[dest] = indices[left]
                        left += 1
                    else:
                        scratch[dest] = indices[right]
                        right += 1
                start = end
            var old = indices^
            indices = scratch^
            scratch = old^
            width *= 2
        return indices^

    def slice(self, offset: Int, length: Int) raises -> Self:
        var result = self._slice_storage(offset, length)
        result._dtype = self._dtype
        return result^

    def _slice_storage(self, offset: Int, length: Int) raises -> Self:
        comptime for i in range(len(Elements.Ts)):
            comptime E: Copyable & Deinitable = Elements.Ts[i]
            if self._data.isa[Column[E]]():
                return Self._wrap(
                    self._name, self._data[Column[E]].slice(offset, length)
                )
        raise Error("Unknown column type")

    def _broadcast(self, length: Int) raises -> Self:
        var result = self._broadcast_storage(length)
        result._dtype = self._dtype
        return result^

    def _broadcast_storage(self, length: Int) raises -> Self:
        comptime for i in range(len(Elements.Ts)):
            comptime E: Copyable & Deinitable = Elements.Ts[i]
            if self._data.isa[Column[E]]():
                return Self._wrap(
                    self._name, self._data[Column[E]]._broadcast(length)
                )
        raise Error("Unknown column type")

    @staticmethod
    def full_null(var name: String, dtype: String, length: Int) raises -> Self:
        return Self.full_null(name^, DataType.parse(dtype), length)

    @staticmethod
    def full_null(
        var name: String, dtype: DataType, length: Int
    ) raises -> Self:
        """A column of `length` nulls with the requested dtype."""
        if dtype.is_temporal():
            return Self(name^, Column[Int64]._nulls(length, 0)).with_dtype(
                dtype
            )
        if dtype == DataType.INT64:
            return Self(name^, Column[Int64]._nulls(length, 0))
        if dtype == DataType.FLOAT64:
            return Self(name^, Column[Float64]._nulls(length, 0))
        if dtype == DataType.BOOL:
            return Self(name^, Column[Bool]._nulls(length, False))
        return Self(name^, Column[String]._nulls(length, ""))

    def append(self, other: Self) raises -> Self:
        """Return a new series with other's rows after this one's."""
        if self.dtype() != other.dtype():
            raise Error(
                "Cannot append "
                + other.dtype().name()
                + " to "
                + self.dtype().name()
                + " series '"
                + self._name
                + "'"
            )
        var result = self.copy()
        result._append_series(other)
        return result^

    def reverse(self) raises -> Self:
        var indices = List[Int](capacity=len(self))
        for i in range(len(self)):
            indices.append(len(self) - 1 - i)
        return self.take(indices)

    def _append_series(mut self, other: Self) raises:
        if self._dtype.physical() != other._dtype.physical():
            raise Error("Cannot append different dtypes")
        comptime for i in range(len(Elements.Ts)):
            comptime E: Copyable & Deinitable = Elements.Ts[i]
            if self._data.isa[Column[E]]():
                self._data[Column[E]]._append_column(other._data[Column[E]])


def _equal_columns[
    T: Copyable & Deinitable & Equatable
](a: Column[T], b: Column[T]) -> Bool:
    for i in range(len(a)):
        if a._valid(i) != b._valid(i):
            return False
        if a._valid(i) and not (a._values[i] == b._values[i]):
            return False
    return True


def _dense_ranks[
    T: Copyable & Deinitable & Comparable
](values: List[T], usable: List[Bool], mut ranks: List[Int]) -> Int:
    """Rank usable values densely by sorted order; returns the rank count."""
    var ordered = List[T]()
    for i in range(len(values)):
        if usable[i]:
            ordered.append(values[i].copy())
    sort(ordered)
    var distinct = List[T]()
    for value in ordered:
        if len(distinct) == 0 or not (distinct[len(distinct) - 1] == value):
            distinct.append(value.copy())
    for i in range(len(values)):
        if not usable[i]:
            continue
        var low = 0
        var high = len(distinct)
        while low < high:
            var mid = (low + high) // 2
            if distinct[mid] < values[i]:
                low = mid + 1
            else:
                high = mid
        ranks[i] = low
    return len(distinct)


def _rank_less(ranks: List[List[Int]], a: Int, b: Int) -> Bool:
    """Lexicographic rank order with the row index as the final tie-break."""
    for key in ranks:
        if key[a] != key[b]:
            return key[a] < key[b]
    return a < b


def sort_indices(ranks: List[List[Int]]) raises -> List[Int]:
    """Stable bottom-up mergesort of row indices by lexicographic ranks."""
    if len(ranks) == 0:
        raise Error("Sorting requires at least one key")
    var n = len(ranks[0])
    var indices = List[Int](capacity=n)
    for i in range(n):
        indices.append(i)
    var scratch = indices.copy()
    var width = 1
    while width < n:
        var start = 0
        while start < n:
            var mid = min(start + width, n)
            var end = min(start + 2 * width, n)
            var left = start
            var right = mid
            for dest in range(start, end):
                if left < mid and (
                    right >= end
                    or not _rank_less(ranks, indices[right], indices[left])
                ):
                    scratch[dest] = indices[left]
                    left += 1
                else:
                    scratch[dest] = indices[right]
                    right += 1
            start = end
        var old = indices^
        indices = scratch^
        scratch = old^
        width *= 2
    return indices^


def smallest_indices(ranks: List[List[Int]], k: Int) raises -> List[Int]:
    """The k first rows of sort_indices(ranks), in order, in O(n log k).

    A max-heap holds the best k rows seen so far under the same total order
    (ranks, then row index), so ties resolve exactly as a stable sort would.
    """
    if len(ranks) == 0:
        raise Error("Sorting requires at least one key")
    if k < 0:
        raise Error("k must be nonnegative")
    var n = len(ranks[0])
    var heap = List[Int](capacity=min(k, n))
    for row in range(n):
        if len(heap) < k:
            heap.append(row)
            var child = len(heap) - 1
            while child > 0:
                var parent = (child - 1) // 2
                if not _rank_less(ranks, heap[parent], heap[child]):
                    break
                var tmp = heap[parent]
                heap[parent] = heap[child]
                heap[child] = tmp
                child = parent
        elif k > 0 and _rank_less(ranks, row, heap[0]):
            heap[0] = row
            var parent = 0
            while True:
                var largest = parent
                var left = 2 * parent + 1
                var right = left + 1
                if left < len(heap) and _rank_less(
                    ranks, heap[largest], heap[left]
                ):
                    largest = left
                if right < len(heap) and _rank_less(
                    ranks, heap[largest], heap[right]
                ):
                    largest = right
                if largest == parent:
                    break
                var tmp = heap[parent]
                heap[parent] = heap[largest]
                heap[largest] = tmp
                parent = largest
    var selected = List[List[Int]]()
    for key in ranks:
        var subset = List[Int](capacity=len(heap))
        for row in heap:
            subset.append(key[row])
        selected.append(subset^)
    # Ranks alone may tie; re-sort by (ranks, original row) to keep stability.
    var positions = List[Int](capacity=len(heap))
    for row in heap:
        positions.append(row)
    selected.append(positions^)
    var order = sort_indices(selected)
    var result = List[Int](capacity=len(heap))
    for i in order:
        result.append(heap[i])
    return result^
