"""Runtime-tagged, named columns without per-element type erasure."""
from std.utils import Variant
from .column import Column
from .value import AnyValue
from .display import render_series

comptime Storage = Variant[
    Column[Int64], Column[Float64], Column[Bool], Column[String]
]


struct Series(Copyable, Sized, Writable):
    var _name: String
    var _data: Storage

    def __init__(out self, var name: String, var column: Column[Int64]):
        self._name = name^
        self._data = Storage(column^)

    def __init__(out self, var name: String, var column: Column[Float64]):
        self._name = name^
        self._data = Storage(column^)

    def __init__(out self, var name: String, var column: Column[Bool]):
        self._name = name^
        self._data = Storage(column^)

    def __init__(out self, var name: String, var column: Column[String]):
        self._name = name^
        self._data = Storage(column^)

    def name(self) -> String:
        return self._name

    def write_to(self, mut writer: Some[Writer]):
        writer.write(render_series(self, 10, 32))

    def to_string(
        self, *, max_rows: Int = 10, max_string_length: Int = 32
    ) -> String:
        """Render at most max_rows values; negative means unlimited."""
        return render_series(self, max_rows, max_string_length)

    def renamed(self, var name: String) -> Self:
        var result = self.copy()
        result._name = name^
        return result^

    def dtype(self) -> String:
        if self._data.isa[Column[Int64]]():
            return "int64"
        if self._data.isa[Column[Float64]]():
            return "float64"
        if self._data.isa[Column[Bool]]():
            return "bool"
        if self._data.isa[Column[String]]():
            return "string"
        return "unreachable"

    def __len__(self) -> Int:
        if self._data.isa[Column[Int64]]():
            return len(self._data[Column[Int64]])
        if self._data.isa[Column[Float64]]():
            return len(self._data[Column[Float64]])
        if self._data.isa[Column[Bool]]():
            return len(self._data[Column[Bool]])
        if self._data.isa[Column[String]]():
            return len(self._data[Column[String]])
        return 0

    def null_count(self) -> Int:
        if self._data.isa[Column[Int64]]():
            return self._data[Column[Int64]].null_count()
        if self._data.isa[Column[Float64]]():
            return self._data[Column[Float64]].null_count()
        if self._data.isa[Column[Bool]]():
            return self._data[Column[Bool]].null_count()
        if self._data.isa[Column[String]]():
            return self._data[Column[String]].null_count()
        return 0

    def get(self, index: Int) raises -> AnyValue:
        """Return one cell as a tagged value; raises when out of bounds."""
        if self._data.isa[Column[Int64]]():
            if self._data[Column[Int64]].is_null(index):
                return AnyValue.null("int64")
            return AnyValue(self._data[Column[Int64]]._values[index])
        if self._data.isa[Column[Float64]]():
            if self._data[Column[Float64]].is_null(index):
                return AnyValue.null("float64")
            return AnyValue(self._data[Column[Float64]]._values[index])
        if self._data.isa[Column[Bool]]():
            if self._data[Column[Bool]].is_null(index):
                return AnyValue.null("bool")
            return AnyValue(self._data[Column[Bool]]._values[index])
        if self._data[Column[String]].is_null(index):
            return AnyValue.null("string")
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
        if self._data.isa[Column[Int64]]():
            return Self(self._name, self._data[Column[Int64]].take(indices))
        if self._data.isa[Column[Float64]]():
            return Self(self._name, self._data[Column[Float64]].take(indices))
        if self._data.isa[Column[Bool]]():
            return Self(self._name, self._data[Column[Bool]].take(indices))
        if self._data.isa[Column[String]]():
            return Self(self._name, self._data[Column[String]].take(indices))
        raise Error("Unknown column type")

    def take_or_null(self, indices: List[Int]) raises -> Self:
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
        """Stable bottom-up mergesort, O(n log n) time and O(n) workspace."""
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
        if self._data.isa[Column[Int64]]():
            return Self(
                self._name, self._data[Column[Int64]].slice(offset, length)
            )
        if self._data.isa[Column[Float64]]():
            return Self(
                self._name, self._data[Column[Float64]].slice(offset, length)
            )
        if self._data.isa[Column[Bool]]():
            return Self(
                self._name, self._data[Column[Bool]].slice(offset, length)
            )
        if self._data.isa[Column[String]]():
            return Self(
                self._name, self._data[Column[String]].slice(offset, length)
            )
        raise Error("Unknown column type")

    def _broadcast(self, length: Int) raises -> Self:
        if self._data.isa[Column[Int64]]():
            return Self(
                self._name, self._data[Column[Int64]]._broadcast(length)
            )
        if self._data.isa[Column[Float64]]():
            return Self(
                self._name, self._data[Column[Float64]]._broadcast(length)
            )
        if self._data.isa[Column[Bool]]():
            return Self(self._name, self._data[Column[Bool]]._broadcast(length))
        if self._data.isa[Column[String]]():
            return Self(
                self._name, self._data[Column[String]]._broadcast(length)
            )
        raise Error("Unknown column type")

    @staticmethod
    def full_null(var name: String, dtype: String, length: Int) raises -> Self:
        """A column of `length` nulls with the requested dtype."""
        if dtype == "int64":
            return Self(name^, Column[Int64]._nulls(length, 0))
        if dtype == "float64":
            return Self(name^, Column[Float64]._nulls(length, 0))
        if dtype == "bool":
            return Self(name^, Column[Bool]._nulls(length, False))
        if dtype == "string":
            return Self(name^, Column[String]._nulls(length, ""))
        raise Error("Unknown dtype: " + dtype)

    def append(self, other: Self) raises -> Self:
        """Return a new series with other's rows after this one's."""
        if self.dtype() != other.dtype():
            raise Error(
                "Cannot append "
                + other.dtype()
                + " to "
                + self.dtype()
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
        if self.dtype() != other.dtype():
            raise Error("Cannot append different dtypes")
        if self._data.isa[Column[Int64]]():
            self._data[Column[Int64]]._append_column(other._data[Column[Int64]])
        if self._data.isa[Column[Float64]]():
            self._data[Column[Float64]]._append_column(
                other._data[Column[Float64]]
            )
        if self._data.isa[Column[Bool]]():
            self._data[Column[Bool]]._append_column(other._data[Column[Bool]])
        if self._data.isa[Column[String]]():
            self._data[Column[String]]._append_column(
                other._data[Column[String]]
            )


def _equal_columns[
    T: Copyable & Deinitable & Equatable
](a: Column[T], b: Column[T]) -> Bool:
    for i in range(len(a)):
        if a._valid(i) != b._valid(i):
            return False
        if a._valid(i) and not (a._values[i] == b._values[i]):
            return False
    return True
