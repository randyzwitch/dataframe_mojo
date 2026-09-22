"""Runtime-tagged, named columns without per-element type erasure."""
from .dtype import DataType, NUMERIC_DTYPES
from std.utils import Variant
from .bool_column import BoolColumn
from .column import Column
from .string_column import StringColumn
from .string_view import StringViewStorage
from .value import AnyValue
from .display import render_series
from .parallel import (
    Job,
    Pool,
    configured_workers,
    partitions,
    run_jobs,
)
from std.memory import ArcPointer
from .cast import cast_series
from .expr import Expr, col, lit
from .frame import DataFrame

# Storage holds Column[Scalar[D]] for each D in NUMERIC_DTYPES, BoolColumn
# (bit-packed values),
# and StringColumn (Arrow large_utf8). Type-agnostic methods (length, take,
# slice, append) loop over FixedElements; numeric kernels loop over
# NUMERIC_DTYPES so each iteration sees a concrete Scalar[D]. Adding a
# numeric type means extending NUMERIC_DTYPES and both lists below.
comptime FixedElements = Variant[
    Int64,
    Float64,
    Int8,
    Int16,
    Int32,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
]
comptime Storage = Variant[
    Column[Int64],
    Column[Float64],
    BoolColumn,
    StringColumn,
    Column[Int8],
    Column[Int16],
    Column[Int32],
    Column[UInt8],
    Column[UInt16],
    Column[UInt32],
    Column[UInt64],
    Column[Float32],
]


@fieldwise_init
struct _SeriesChunks(Copyable):
    var arrays: List[Storage]
    var ends: List[Int]


struct Series(Copyable, Sized, Writable):
    """A named column of one supported dtype, plus expression-backed methods."""

    var _name: String
    var _data: Storage
    # The logical type. Temporal types are stored in Column[Int64].
    var _dtype: DataType
    # Immutable array metadata is shared too: cloning a chunked Series is O(1).
    var _chunked: Optional[ArcPointer[_SeriesChunks]]

    def __init__[
        D: DType
    ](out self, var name: String, var column: Column[Scalar[D]]):
        self._name = name^
        self._data = Storage(column^)
        self._dtype = DataType.of(D)
        self._chunked = None

    def __init__(out self, var name: String, var column: BoolColumn):
        self._name = name^
        self._data = Storage(column^)
        self._dtype = DataType.BOOL
        self._chunked = None

    def __init__(out self, var name: String, column: Column[Bool]) raises:
        """Pack a byte-per-value Boolean column into bits."""
        self = Self(name^, BoolColumn(column))

    def __init__(out self, var name: String, var column: StringColumn):
        self._name = name^
        self._data = Storage(column^)
        self._dtype = DataType.STRING
        self._chunked = None

    def __init__(out self, var name: String, column: Column[String]):
        """Convert list-backed strings to the contiguous UTF-8 layout."""
        self = Self(name^, StringColumn(column))

    @staticmethod
    def _wrap[
        E: Copyable & Deinitable
    ](var name: String, var column: Column[E]) -> Self:
        """Build a series from any storable column type."""
        var result = Self(name^, Column[Int64]([]))
        result._data = Storage(column^)
        result._dtype = result._storage_dtype()
        return result^

    def __init__(
        out self, var name: String, var storage: Storage, dtype: DataType
    ):
        self._name = name^
        self._data = storage^
        self._dtype = dtype
        self._chunked = None

    def is_chunked(self) -> Bool:
        return True if self._chunked else False

    def n_chunks(self) -> Int:
        """Number of physical Arrow arrays backing this series."""
        return len(self._chunked.value()[].arrays) if self.is_chunked() else 1

    def chunks(self) -> List[Self]:
        """Owned column views sharing the immutable buffers of each array."""
        if not self.is_chunked():
            return [self.copy()]
        var result = List[Self](capacity=len(self._chunked.value()[].arrays))
        for storage in self._chunked.value()[].arrays:
            var part = Self(self._name, storage.copy(), self._dtype)
            result.append(part^)
        return result^

    @staticmethod
    def _from_chunks(parts: List[Self]) raises -> Self:
        """Append array references without copying values or validity bits."""
        if len(parts) == 0:
            raise Error("Chunked series requires at least one array")
        var result = parts[0].copy()
        result._chunked = None
        var arrays = List[Storage]()
        var ends = List[Int]()
        var height = 0
        for part in parts:
            if part.dtype() != result.dtype():
                raise Error("Chunked series arrays must have the same dtype")
            # CSV decode jobs produce one array each. Append that array
            # directly, as Polars vstack_mut_owned does, without allocating a
            # temporary one-element chunk list for every input part.
            if not part.is_chunked():
                if len(part) == 0:
                    continue
                if len(part) > Int.MAX - height:
                    raise Error("Chunked series length overflows")
                height += len(part)
                arrays.append(part._data.copy())
                ends.append(height)
                continue
            for chunk in part.chunks():
                if len(chunk) == 0:
                    continue
                if len(chunk) > Int.MAX - height:
                    raise Error("Chunked series length overflows")
                height += len(chunk)
                arrays.append(chunk._data.copy())
                ends.append(height)
        if len(arrays) > 0:
            result._data = arrays[0].copy()
        if len(arrays) > 1:
            result._chunked = ArcPointer(_SeriesChunks(arrays^, ends^))
        return result^

    def rechunk(self) raises -> Self:
        """Materialize one contiguous Arrow array, preserving name and dtype."""
        if not self.is_chunked():
            return self.copy()
        var parts = self.chunks()
        var result = parts[0].copy()
        # Utf8View chunks concatenate descriptors and Arc byte blocks. Calling
        # the legacy reserve path would materialize their payloads first.
        if (
            result._data.isa[StringColumn]()
            and result._data[StringColumn]._is_view_storage()
        ):
            var storages = List[StringViewStorage](capacity=len(parts))
            var offsets = List[Int](capacity=len(parts))
            var lengths = List[Int](capacity=len(parts))
            for part in parts:
                ref column = part._data[StringColumn]
                if not column._is_view_storage():
                    # The mixed path below uses the explicit large_utf8 adapter.
                    result._reserve_rows(len(self), self._text_bytes())
                    for i in range(1, len(parts)):
                        result._append_series(parts[i])
                    return result^
                storages.append(column._view_storage_unchecked())
                offsets.append(column._offset)
                lengths.append(len(column))
            var merged = StringViewStorage._concat_many(
                storages^, offsets^, lengths^
            )
            return Self(self._name, StringColumn(merged^))
        result._reserve_rows(len(self), self._text_bytes())
        for i in range(1, len(parts)):
            result._append_series(parts[i])
        return result^

    def _chunk_at(self, row: Int) raises -> Tuple[Self, Int]:
        if row < 0 or row >= len(self):
            raise Error("Column index out of bounds")
        var lo = 0
        var hi = len(self._chunked.value()[].ends)
        while lo < hi:
            var mid = lo + (hi - lo) // 2
            if row < self._chunked.value()[].ends[mid]:
                hi = mid
            else:
                lo = mid + 1
        var part = Self(
            self._name, self._chunked.value()[].arrays[lo].copy(), self._dtype
        )
        var start = 0 if lo == 0 else self._chunked.value()[].ends[lo - 1]
        return (part^, row - start)

    def _storage_dtype(self) -> DataType:
        """The physical DataType of the stored column."""
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if self._data.isa[Column[Scalar[D]]]():
                return DataType.of(D)
        if self._data.isa[BoolColumn]():
            return DataType.BOOL
        return DataType.STRING

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
        if self.is_chunked():
            return self._chunked.value()[].ends[
                len(self._chunked.value()[].ends) - 1
            ]
        comptime for i in range(len(FixedElements.Ts)):
            comptime E: Copyable & Deinitable = FixedElements.Ts[i]
            if self._data.isa[Column[E]]():
                return len(self._data[Column[E]])
        if self._data.isa[BoolColumn]():
            return len(self._data[BoolColumn])
        return len(self._data[StringColumn])

    def null_count(self) -> Int:
        if self.is_chunked():
            var count = 0
            for chunk in self.chunks():
                count += chunk.null_count()
            return count
        comptime for i in range(len(FixedElements.Ts)):
            comptime E: Copyable & Deinitable = FixedElements.Ts[i]
            if self._data.isa[Column[E]]():
                return self._data[Column[E]].null_count()
        if self._data.isa[BoolColumn]():
            return self._data[BoolColumn].null_count()
        return self._data[StringColumn].null_count()

    def get(self, index: Int) raises -> AnyValue:
        """Return one cell as a tagged value; raises when out of bounds."""
        if self.is_chunked():
            var part = self._chunk_at(index)
            return part[0].get(part[1])
        if self._dtype.is_temporal():
            ref column = self._data[Column[Int64]]
            if column.is_null(index):
                return AnyValue.null(self._dtype)
            return AnyValue(self._dtype, True, column._get(index), 0, False, "")
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if self._data.isa[Column[Scalar[D]]]():
                ref column = self._data[Column[Scalar[D]]]
                if column.is_null(index):
                    return AnyValue.null(self._dtype)
                return AnyValue(column._get(index))
        if self._data.isa[BoolColumn]():
            if self._data[BoolColumn].is_null(index):
                return AnyValue.null(DataType.BOOL)
            return AnyValue(self._data[BoolColumn]._get(index))
        if self._data[StringColumn].is_null(index):
            return AnyValue.null(DataType.STRING)
        return AnyValue(String(self._data[StringColumn]._get(index)))

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
        if self.is_chunked() or other.is_chunked():
            var a = self.chunks()
            var b = other.chunks()
            var ai = 0
            var bi = 0
            var ao = 0
            var bo = 0
            while ai < len(a) and bi < len(b):
                var n = min(len(a[ai]) - ao, len(b[bi]) - bo)
                try:
                    if (
                        not a[ai]
                        .slice(ao, n)
                        .equals(b[bi].slice(bo, n), null_equal=null_equal)
                    ):
                        return False
                except:
                    return False
                ao += n
                bo += n
                if ao == len(a[ai]):
                    ai += 1
                    ao = 0
                if bo == len(b[bi]):
                    bi += 1
                    bo = 0
            return True
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if self._data.isa[Column[Scalar[D]]]():
                ref a = self._data[Column[Scalar[D]]]
                ref b = other._data[Column[Scalar[D]]]
                for i in range(len(a)):
                    if a._valid(i) != b._valid(i):
                        return False
                    if a._valid(i):
                        var x = a._get(i)
                        var y = b._get(i)
                        if x != y and not (x != x and y != y):
                            return False
                return True
        if self._data.isa[BoolColumn]():
            ref a = self._data[BoolColumn]
            ref b = other._data[BoolColumn]
            for i in range(len(a)):
                if a._valid(i) != b._valid(i):
                    return False
                if a._valid(i) and a._get(i) != b._get(i):
                    return False
            return True
        ref a = self._data[StringColumn]
        ref b = other._data[StringColumn]
        for i in range(len(a)):
            if a._valid(i) != b._valid(i):
                return False
            if a._valid(i) and a._get(i) != b._get(i):
                return False
        return True

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
        if self.is_chunked():
            return self.rechunk().int64()
        if not self._data.isa[Column[Int64]]():
            raise Error("Expected int64 column")
        return self._data[Column[Int64]].copy()

    def float64(self) raises -> Column[Float64]:
        """Return an owned typed copy, raising on a dtype mismatch."""
        if self.is_chunked():
            return self.rechunk().float64()
        if not self._data.isa[Column[Float64]]():
            raise Error("Expected float64 column")
        return self._data[Column[Float64]].copy()

    def numeric[D: DType](self) raises -> Column[Scalar[D]]:
        """The (shared, immutable) column as Scalar[D], raising on a dtype
        mismatch. Temporal columns read as their Int64 storage."""
        if self.is_chunked():
            return self.rechunk().numeric[D]()
        if not self._data.isa[Column[Scalar[D]]]():
            raise Error("Expected " + String(D) + " column")
        return self._data[Column[Scalar[D]]].copy()

    def int8(self) raises -> Column[Int8]:
        return self.numeric[DType.int8]()

    def int16(self) raises -> Column[Int16]:
        return self.numeric[DType.int16]()

    def int32(self) raises -> Column[Int32]:
        return self.numeric[DType.int32]()

    def uint8(self) raises -> Column[UInt8]:
        return self.numeric[DType.uint8]()

    def uint16(self) raises -> Column[UInt16]:
        return self.numeric[DType.uint16]()

    def uint32(self) raises -> Column[UInt32]:
        return self.numeric[DType.uint32]()

    def uint64(self) raises -> Column[UInt64]:
        return self.numeric[DType.uint64]()

    def float32(self) raises -> Column[Float32]:
        return self.numeric[DType.float32]()

    def bool(self) raises -> BoolColumn:
        """Return an owned typed copy, raising on a dtype mismatch."""
        if self.is_chunked():
            return self.rechunk().bool()
        if not self._data.isa[BoolColumn]():
            raise Error("Expected bool column")
        return self._data[BoolColumn].copy()

    def string(self) raises -> StringColumn:
        """Return the (shared, immutable) column, raising on a dtype mismatch."""
        if self.is_chunked():
            return self.rechunk().string()
        if not self._data.isa[StringColumn]():
            raise Error("Expected string column")
        return self._data[StringColumn].copy()

    def take(self, indices: List[Int]) raises -> Self:
        if self.is_chunked():
            return self.rechunk().take(indices)
        var result = self._take_storage(indices)
        result._dtype = self._dtype
        return result^

    def _take_storage(self, indices: List[Int]) raises -> Self:
        comptime for i in range(len(FixedElements.Ts)):
            comptime E: Copyable & Deinitable = FixedElements.Ts[i]
            if self._data.isa[Column[E]]():
                return Self._wrap(
                    self._name, self._data[Column[E]].take(indices)
                )
        if self._data.isa[BoolColumn]():
            return Self(self._name, self._data[BoolColumn].take(indices))
        return Self(self._name, self._data[StringColumn].take(indices))

    def take_or_null(self, indices: List[Int]) raises -> Self:
        if self.is_chunked():
            return self.rechunk().take_or_null(indices)
        var result = self._take_or_null_storage(indices)
        result._dtype = self._dtype
        return result^

    def _take_or_null_storage(self, indices: List[Int]) raises -> Self:
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if self._data.isa[Column[Scalar[D]]]():
                return Self(
                    self._name,
                    self._data[Column[Scalar[D]]].take_or_null(
                        indices, Scalar[D](0)
                    ),
                )
        if self._data.isa[BoolColumn]():
            return Self(
                self._name,
                self._data[BoolColumn].take_or_null(indices, False),
            )
        return Self(
            self._name,
            self._data[StringColumn].take_or_null(indices, String("")),
        )

    def _less(
        self, a: Int, b: Int, descending: Bool, nulls_last: Bool
    ) raises -> Bool:
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if self._data.isa[Column[Scalar[D]]]():
                ref column = self._data[Column[Scalar[D]]]
                var a_null = column.is_null(a)
                var b_null = column.is_null(b)
                if a_null or b_null:
                    return a_null != b_null and (
                        b_null if nulls_last else a_null
                    )
                var x = column.value(a)
                var y = column.value(b)
                # NaNs follow finite/infinite values in either direction.
                if x != x or y != y:
                    return x == x and y != y
                return y < x if descending else x < y
        if self._data.isa[BoolColumn]():
            var a_null = self._data[BoolColumn].is_null(a)
            var b_null = self._data[BoolColumn].is_null(b)
            if a_null or b_null:
                return a_null != b_null and (b_null if nulls_last else a_null)
            var x = self._data[BoolColumn].value(a)
            var y = self._data[BoolColumn].value(b)
            return Int(y) < Int(x) if descending else Int(x) < Int(y)
        ref column = self._data[StringColumn]
        var a_null = column.is_null(a)
        var b_null = column.is_null(b)
        if a_null or b_null:
            return a_null != b_null and (b_null if nulls_last else a_null)
        var x = column._get(a)
        var y = column._get(b)
        return y < x if descending else x < y

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
        if self.is_chunked():
            return self.rechunk()._sort_ranks(descending, nulls_last)
        var n = len(self)
        var ranks = List[Int](length=n, fill=0)
        var valid = List[Bool](length=n, fill=False)
        var nan = List[Bool](length=n, fill=False)
        var distinct: Int
        distinct = -1
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[k]
            if self._data.isa[Column[Scalar[D]]]():
                ref column = self._data[Column[Scalar[D]]]
                var usable = List[Bool](length=n, fill=False)
                for i in range(n):
                    valid[i] = column._valid(i)
                    var x = column._get(i)
                    nan[i] = valid[i] and x != x
                    usable[i] = valid[i] and not nan[i]
                distinct = _dense_ranks(column.to_list(), usable, ranks)
        if distinct >= 0:
            pass
        elif self._data.isa[BoolColumn]():
            ref column = self._data[BoolColumn]
            for i in range(n):
                valid[i] = column._valid(i)
                ranks[i] = Int(column._get(i))
            distinct = 2
        else:
            ref column = self._data[StringColumn]
            for i in range(n):
                valid[i] = column._valid(i)
            distinct = _dense_string_ranks(column, valid, ranks)
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
        if self.is_chunked():
            return self.rechunk()._argsort_reference(descending, nulls_last)
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
        if self.is_chunked():
            if (
                offset < 0
                or length < 0
                or offset > len(self)
                or length > len(self) - offset
            ):
                raise Error("Invalid column slice")
            ref chunks = self._chunked.value()[]
            if length == 0:
                var empty = Self(
                    self._name, chunks.arrays[0].copy(), self._dtype
                )
                return empty.slice(0, 0)
            # Expression batches are usually within one chunk. Locate the
            # first overlapping array from cumulative ends instead of making
            # a List of every chunk for each batch.
            var lo = 0
            var hi = len(chunks.ends)
            while lo < hi:
                var mid = lo + (hi - lo) // 2
                if offset < chunks.ends[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            var start = 0 if lo == 0 else chunks.ends[lo - 1]
            var stop = chunks.ends[lo]
            var end = offset + length
            var first = Self(self._name, chunks.arrays[lo].copy(), self._dtype)
            if end <= stop:
                return first.slice(offset - start, length)
            var parts = List[Self]()
            parts.append(first.slice(offset - start, stop - offset))
            lo += 1
            while lo < len(chunks.ends) and stop < end:
                start = stop
                stop = chunks.ends[lo]
                var part = Self(
                    self._name, chunks.arrays[lo].copy(), self._dtype
                )
                parts.append(part.slice(0, min(end, stop) - start))
                lo += 1
            return Self._from_chunks(parts)
        var result = self._slice_storage(offset, length)
        result._dtype = self._dtype
        return result^

    def _slice_storage(self, offset: Int, length: Int) raises -> Self:
        comptime for i in range(len(FixedElements.Ts)):
            comptime E: Copyable & Deinitable = FixedElements.Ts[i]
            if self._data.isa[Column[E]]():
                return Self._wrap(
                    self._name, self._data[Column[E]].slice(offset, length)
                )
        if self._data.isa[BoolColumn]():
            return Self(
                self._name, self._data[BoolColumn].slice(offset, length)
            )
        return Self(self._name, self._data[StringColumn].slice(offset, length))

    def _broadcast(self, length: Int) raises -> Self:
        if self.is_chunked():
            return self.rechunk()._broadcast(length)
        var result = self._broadcast_storage(length)
        result._dtype = self._dtype
        return result^

    def _broadcast_storage(self, length: Int) raises -> Self:
        comptime for i in range(len(FixedElements.Ts)):
            comptime E: Copyable & Deinitable = FixedElements.Ts[i]
            if self._data.isa[Column[E]]():
                return Self._wrap(
                    self._name, self._data[Column[E]]._broadcast(length)
                )
        if self._data.isa[BoolColumn]():
            return Self(self._name, self._data[BoolColumn]._broadcast(length))
        return Self(self._name, self._data[StringColumn]._broadcast(length))

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
        comptime for i in range(len(NUMERIC_DTYPES)):
            comptime D = NUMERIC_DTYPES[i]
            if dtype == DataType.of(D):
                return Self(name^, Column[Scalar[D]]._nulls(length, 0))
        if dtype == DataType.BOOL:
            return Self(name^, BoolColumn._nulls(length, False))
        return Self(name^, StringColumn._nulls(length))

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
        return Self._from_chunks([self.copy(), other.copy()])

    def reverse(self) raises -> Self:
        var indices = List[Int](capacity=len(self))
        for i in range(len(self)):
            indices.append(len(self) - 1 - i)
        return self.take(indices)

    def _text_bytes(self) -> Int:
        """Bytes of text this column holds, or 0 when it holds none."""
        if self.is_chunked():
            var total = 0
            for chunk in self.chunks():
                total += chunk._text_bytes()
            return total
        if self._data.isa[StringColumn]():
            return self._data[StringColumn]._value_bytes()
        return 0

    def _reserve_rows(mut self, rows: Int, text_bytes: Int) raises:
        """Size this column for a known final height before appending."""
        if self.is_chunked():
            self = self.rechunk()
        comptime for i in range(len(FixedElements.Ts)):
            comptime E: Copyable & Deinitable = FixedElements.Ts[i]
            if self._data.isa[Column[E]]():
                self._data[Column[E]]._reserve_rows(rows, text_bytes)
        if self._data.isa[BoolColumn]():
            self._data[BoolColumn]._reserve_rows(rows, text_bytes)
        if self._data.isa[StringColumn]():
            self._data[StringColumn]._reserve_rows(rows, text_bytes)

    def _append_series(mut self, other: Self) raises:
        if self.is_chunked():
            self = self.rechunk()
        if other.is_chunked():
            for chunk in other.chunks():
                self._append_series(chunk)
            return
        if self._dtype.physical() != other._dtype.physical():
            raise Error("Cannot append different dtypes")
        comptime for i in range(len(FixedElements.Ts)):
            comptime E: Copyable & Deinitable = FixedElements.Ts[i]
            if self._data.isa[Column[E]]():
                self._data[Column[E]]._append_column(other._data[Column[E]])
        if self._data.isa[BoolColumn]():
            self._data[BoolColumn]._append_column(other._data[BoolColumn])
        if self._data.isa[StringColumn]():
            self._data[StringColumn]._append_column(other._data[StringColumn])


def _equal_columns[
    T: Copyable & Deinitable & Equatable
](a: Column[T], b: Column[T]) -> Bool:
    for i in range(len(a)):
        if a._valid(i) != b._valid(i):
            return False
        if a._valid(i) and not (a._get(i) == b._get(i)):
            return False
    return True


@fieldwise_init
struct _RankPair[T: Copyable & Deinitable & Comparable](
    Copyable, Deinitable, Movable
):
    """One usable value and the row it came from, so sorting the values also
    records where each rank belongs."""

    var value: Self.T
    var row: Int


def _dense_ranks[
    T: Copyable & Deinitable & Comparable
](values: List[T], usable: List[Bool], mut ranks: List[Int]) -> Int:
    """Rank usable values densely by sorted order; returns the rank count.

    Sorting (value, row) pairs and then walking them in order assigns every
    rank in one pass. Sorting the values alone loses the rows, which is why
    this used to reduce them to the distinct values and binary-search each
    row back in -- and that search, not the sort, was the dominant cost of a
    sort: 114 ms of the 182 ms spent ranking two key columns of 1M rows.
    `_dense_string_ranks` already ranked by walking a sorted order; this
    brings the numeric path in line, carrying the value alongside the row so
    that comparisons stay contiguous instead of chasing an index.
    """
    var pairs = List[_RankPair[T]](capacity=len(values))
    for i in range(len(values)):
        if usable[i]:
            pairs.append(_RankPair[T](values[i].copy(), i))
    if len(pairs) == 0:
        return 0

    def by_value(a: _RankPair[T], b: _RankPair[T]) -> Bool:
        return a.value < b.value

    sort(pairs, by_value)
    var distinct = 0
    for k in range(len(pairs)):
        if k > 0 and not (pairs[k].value == pairs[k - 1].value):
            distinct += 1
        ranks[pairs[k].row] = distinct
    return distinct + 1


def _dense_string_ranks(
    column: StringColumn, usable: List[Bool], mut ranks: List[Int]
) -> Int:
    """Dense ranks over borrowed UTF-8 slices (byte order = code point order)."""
    var order = List[Int]()
    for i in range(len(column)):
        if usable[i]:
            order.append(i)

    def less(a: Int, b: Int) {imm column} -> Bool:
        return column._get(a) < column._get(b)

    sort(order, less)
    var distinct = 0
    for k in range(len(order)):
        if k > 0 and column._get(order[k]) != column._get(order[k - 1]):
            distinct += 1
        ranks[order[k]] = distinct
    return distinct + 1 if len(order) > 0 else 0


def _rank_less(ranks: List[List[Int]], a: Int, b: Int) -> Bool:
    """Lexicographic rank order with the row index as the final tie-break."""
    if len(ranks) == 1:
        # Sorting by one key is the common case, and a merge calls this once
        # per output row; going through the outer list costs more than the
        # comparison itself.
        ref key = ranks[0]
        if key[a] != key[b]:
            return key[a] < key[b]
        return a < b
    for key in ranks:
        if key[a] != key[b]:
            return key[a] < key[b]
    return a < b


def _merge_runs(
    ranks: List[List[Int]],
    source: List[Int],
    mut target: List[Int],
    start: Int,
    mid: Int,
    end: Int,
):
    """Merge two adjacent sorted runs, preferring the earlier on ties."""
    var left = start
    var right = mid
    for dest in range(start, end):
        if left < mid and (
            right >= end or not _rank_less(ranks, source[right], source[left])
        ):
            target[dest] = source[left]
            left += 1
        else:
            target[dest] = source[right]
            right += 1


def _co_rank(
    ranks: List[List[Int]],
    source: List[Int],
    start: Int,
    mid: Int,
    end: Int,
    k: Int,
) -> Int:
    """How many of the first `k` merged outputs come from the left run.

    Splitting a merge across workers needs each worker to know where its
    output slice begins in *both* runs. For output position k there is
    exactly one split (i from the left, k - i from the right), because
    `_rank_less` breaks ties by row index and is therefore a total order --
    no two distinct rows compare equal, so no split is ambiguous. That also
    means the slices reproduce the serial merge exactly, including which run
    an equal-keyed row came from, so stability needs no special handling.

    Found by binary search on i, the classic merge-path co-rank.
    """
    var left_len = mid - start
    var right_len = end - mid
    var low = max(0, k - right_len)
    var high = min(k, left_len)
    while low < high:
        var i = (low + high) // 2
        var j = k - i
        # source[mid + j - 1] belongs before source[start + i]: take more
        # from the left run.
        if j > 0 and _rank_less(ranks, source[start + i], source[mid + j - 1]):
            low = i + 1
        else:
            high = i
    return low


def _merge_slice(
    ranks: List[List[Int]],
    source: List[Int],
    mut target: List[Int],
    start: Int,
    mid: Int,
    end: Int,
    first: Int,
    last: Int,
):
    """Merge only outputs [first, last) of merging [start, mid) and
    [mid, end), where both are offsets from `start`."""
    var i = _co_rank(ranks, source, start, mid, end, first)
    var j = first - i
    var left = start + i
    var right = mid + j
    for dest in range(start + first, start + last):
        if left < mid and (
            right >= end or not _rank_less(ranks, source[right], source[left])
        ):
            target[dest] = source[left]
            left += 1
        else:
            target[dest] = source[right]
            right += 1


def _sort_range(ranks: List[List[Int]], start: Int, end: Int) -> List[Int]:
    """Stable bottom-up mergesort of rows [start, end), returned in order."""
    var n = end - start
    var indices = List[Int](capacity=n)
    for i in range(start, end):
        indices.append(i)
    var scratch = indices.copy()
    var width = 1
    while width < n:
        var at = 0
        while at < n:
            var mid = min(at + width, n)
            var stop = min(at + 2 * width, n)
            _merge_runs(ranks, indices, scratch, at, mid, stop)
            at = stop
        var old = indices^
        indices = scratch^
        scratch = old^
        width *= 2
    return indices^


struct _SortRangeJob(Job):
    """Sort one contiguous row range into the shared output."""

    var ranks: ArcPointer[List[List[Int]]]
    var start: Int
    var end: Int
    var rows: List[Int]

    def __init__(
        out self, ranks: ArcPointer[List[List[Int]]], start: Int, end: Int
    ):
        self.ranks = ranks.copy()
        self.start = start
        self.end = end
        self.rows = List[Int]()

    def run(mut self) raises:
        self.rows = _sort_range(self.ranks[], self.start, self.end)


struct _MergeJob(Job):
    """Merge outputs [first, last) of two adjacent sorted runs of `source`
    into `target`, where first and last are offsets from `start`.

    A whole merge is the slice [0, end - start); splitting it lets one merge
    occupy every worker, which matters most in the last round, where the
    pairwise tree has only one merge left and it spans the whole array.
    """

    var ranks: ArcPointer[List[List[Int]]]
    var source: Int
    var target: Int
    var start: Int
    var mid: Int
    var end: Int
    var first: Int
    var last: Int

    def __init__(
        out self,
        ranks: ArcPointer[List[List[Int]]],
        source: Int,
        target: Int,
        start: Int,
        mid: Int,
        end: Int,
        first: Int,
        last: Int,
    ):
        self.ranks = ranks.copy()
        self.source = source
        self.target = target
        self.start = start
        self.mid = mid
        self.end = end
        self.first = first
        self.last = last

    def run(mut self) raises:
        ref out = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.target
        )[]
        ref src = Pointer[List[Int], MutAnyOrigin](
            unsafe_from_address=self.source
        )[]
        _merge_slice(
            self.ranks[],
            src,
            out,
            self.start,
            self.mid,
            self.end,
            self.first,
            self.last,
        )


# Below this many rows a run is not worth its own thread, whatever the core
# count. Sorting is n log n per run, so this is far under MIN_ROWS_PER_WORKER.
comptime _MIN_ROWS_PER_RUN = 8192


def sort_indices(ranks: List[List[Int]]) raises -> List[Int]:
    """Stable bottom-up mergesort of row indices by lexicographic ranks.

    Large inputs sort one contiguous row range per worker and then merge the
    runs in rounds. Ranges are formed in row order and every merge prefers
    the earlier run on ties, so the result is the same stable order the
    serial sort produces, for any worker count.
    """
    if len(ranks) == 0:
        raise Error("Sorting requires at least one key")
    var n = len(ranks[0])
    # One run per thread, not one per MIN_ROWS_PER_WORKER rows: that minimum
    # is sized for a linear scan, and it both caps a 1M-row sort at 15 runs
    # however many cores are free and leaves a 100k-row sort entirely serial.
    # Sorting a run is n log n, so shorter runs still repay their scheduling,
    # and the merge rounds below are themselves split across threads and so
    # do not lengthen as runs are added.
    var workers = configured_workers()
    var target = max(1, min(workers, n // _MIN_ROWS_PER_RUN))
    if target <= 1 or n < 2:
        return _sort_range(ranks, 0, n)

    var shared = ArcPointer(ranks.copy())
    var bounds = partitions(n, target, 1)
    # partitions() can leave empty trailing ranges; keep only real ones.
    var starts = List[Int]()
    for w in range(len(bounds) - 1):
        if bounds[w + 1] > bounds[w]:
            starts.append(bounds[w])
    starts.append(n)
    var runs = len(starts) - 1
    if runs <= 1:
        return _sort_range(ranks, 0, n)

    # One pool for the run pass and every merge round that follows. Creating
    # threads per round cost about 1.27 ms of the sort at 32 threads, against
    # 32 us to wake this pool's, and a sort runs one round plus log2(runs)
    # merge rounds. The pool is released before returning, on every path.
    var pool = Pool(workers)
    var jobs = List[_SortRangeJob](capacity=runs)
    for r in range(runs):
        jobs.append(_SortRangeJob(shared, starts[r], starts[r + 1]))
    pool.run(jobs)
    var indices = List[Int](length=n, fill=0)
    for r in range(runs):
        var at = starts[r]
        for i in range(len(jobs[r].rows)):
            indices[at + i] = jobs[r].rows[i]

    # Merge adjacent runs in rounds, alternating buffers. Each round halves
    # the number of merges, so the later rounds have fewer merges than there
    # are workers -- the last has one, spanning the whole array. Every merge
    # is therefore split into output slices, enough that a round has about
    # one slice per worker however few merges it contains.
    var scratch = List[Int](length=n, fill=0)
    var stride = 1
    while stride < runs:
        var source_address = Int(Pointer(to=indices))
        var merges = List[_MergeJob]()
        var pending = (runs + 2 * stride - 1) // (2 * stride)
        var slices = max(1, (workers + pending - 1) // pending)
        var r = 0
        while r < runs:
            var start = starts[r]
            var mid = starts[min(r + stride, runs)]
            var end = starts[min(r + 2 * stride, runs)]
            if mid < end:
                var width = end - start
                var cuts = min(slices, width)
                for s in range(cuts):
                    var first = (width * s) // cuts
                    var last = (width * (s + 1)) // cuts
                    if last > first:
                        merges.append(
                            _MergeJob(
                                shared,
                                source_address,
                                Int(Pointer(to=scratch)),
                                start,
                                mid,
                                end,
                                first,
                                last,
                            )
                        )
            else:
                for i in range(start, end):
                    scratch[i] = indices[i]
            r += 2 * stride
        pool.run(merges)
        var old = indices^
        indices = scratch^
        scratch = old^
        stride *= 2
    pool.release()
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
