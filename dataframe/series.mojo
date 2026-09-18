"""Runtime-tagged, named columns without per-element type erasure."""
from std.utils import Variant
from .column import Column

comptime Storage = Variant[
    Column[Int64], Column[Float64], Column[Bool], Column[String]
]


struct Series(Copyable, Sized):
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
