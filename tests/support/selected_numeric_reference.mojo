"""Scalar statistics over explicitly ordered source rows, independent of kernels."""
from std.math import sqrt


struct SelectedNumbers(Copyable):
    var rows: List[Int]
    var ordered: List[Float64]
    var sorted: List[Float64]
    var unique: List[Float64]
    var count: Int
    var saw_null: Bool
    var total: Float64
    var mean: Float64
    var centered: Float64

    def __init__(
        out self, values: List[Float64], valid: List[Bool], rows: List[Int]
    ):
        self.rows = rows.copy()
        self.ordered = List[Float64]()
        self.sorted = List[Float64]()
        self.unique = List[Float64]()
        self.count = 0
        self.saw_null = False
        self.total = 0
        self.mean = 0
        self.centered = 0
        for row in rows:
            if not valid[row]:
                self.saw_null = True
                continue
            var value = values[row]
            self.ordered.append(value)
            self.total += value
            self.count += 1
            var position = len(self.sorted)
            self.sorted.append(value)
            while position and self.sorted[position - 1] > value:
                self.sorted[position] = self.sorted[position - 1]
                position -= 1
            self.sorted[position] = value
            var found = False
            for prior in self.unique:
                found = found or prior == value
            if not found:
                self.unique.append(value)
        if self.count:
            self.mean = self.total / Float64(self.count)
            for value in self.ordered:
                self.centered += (value - self.mean) ** 2

    def quantile(self, q: Float64, method: String) -> Float64:
        var position = q * Float64(self.count - 1)
        var low = Int(position)
        var high = min(low + 1, self.count - 1)
        var fraction = position - Float64(low)
        if method == "lower":
            return self.sorted[low]
        if method == "higher":
            return self.sorted[low if fraction == 0 else high]
        if method == "nearest":
            return self.sorted[low if fraction < 0.5 else high]
        if method == "midpoint":
            return (
                self.sorted[low] if fraction
                == 0 else (self.sorted[low] + self.sorted[high]) / 2
            )
        return self.sorted[low] * (1 - fraction) + self.sorted[high] * fraction
