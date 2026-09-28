"""Mergeable reduction states, independent of execution scheduling.

At most Int.MAX Int64 inputs fit exactly in a signed 128-bit accumulator.
This allows arbitrary partitions and merge trees without intermediate overflow.
"""
from std.collections import Optional
from std.math import sqrt

comptime WideInt = SIMD[DType.int128, 1]


@fieldwise_init
struct IntSumState(Copyable):
    var total: WideInt
    var count: Int64

    def __init__(out self):
        self.total = 0
        self.count = 0

    def add(mut self, value: Int64):
        self.total += value.cast[DType.int128]()
        self.count += 1

    def add_wide(mut self, value: WideInt):
        """Add a value outside Int64 (UInt64 inputs)."""
        self.total += value
        self.count += 1

    def merge(mut self, other: Self):
        """Combine states for disjoint partitions of one input."""
        self.total += other.total
        self.count += other.count

    def value(self) raises -> Int64:
        if self.total > WideInt(9223372036854775807) or self.total < WideInt(
            -9223372036854775808
        ):
            raise Error("Int64 expression sum overflow")
        return self.total.cast[DType.int64]()


@fieldwise_init
struct FloatSumState(Copyable):
    var total: Float64
    var count: Int64

    def __init__(out self):
        self.total = 0
        self.count = 0

    def add(mut self, value: Float64):
        self.total += value
        self.count += 1

    def merge(mut self, other: Self):
        """Reassociation is allowed; bitwise reproducibility is not promised."""
        self.total += other.total
        self.count += other.count


@fieldwise_init
struct LogicState(Copyable):
    """Kleene any/all over Bool values; merges are order-independent."""

    var saw_true: Bool
    var saw_false: Bool
    var saw_null: Bool

    def __init__(out self):
        self.saw_true = False
        self.saw_false = False
        self.saw_null = False

    def add(mut self, valid: Bool, value: Bool):
        if not valid:
            self.saw_null = True
        elif value:
            self.saw_true = True
        else:
            self.saw_false = True

    def merge(mut self, other: Self):
        self.saw_true = self.saw_true or other.saw_true
        self.saw_false = self.saw_false or other.saw_false
        self.saw_null = self.saw_null or other.saw_null

    def any(self, ignore_nulls: Bool) -> Optional[Bool]:
        if self.saw_true:
            return True
        if not ignore_nulls and self.saw_null:
            return None
        return False

    def all(self, ignore_nulls: Bool) -> Optional[Bool]:
        if self.saw_false:
            return False
        if not ignore_nulls and self.saw_null:
            return None
        return True


@fieldwise_init
struct VarState(Copyable):
    """Welford running moments; `merge` uses Chan's parallel update.

    Merging partitions in any order gives results equal up to rounding.
    """

    var count: Int64
    var mean: Float64
    var m2: Float64

    def __init__(out self):
        self.count = 0
        self.mean = 0
        self.m2 = 0

    def add(mut self, value: Float64):
        self.count += 1
        var delta = value - self.mean
        self.mean += delta / Float64(self.count)
        self.m2 += delta * (value - self.mean)

    def merge(mut self, other: Self):
        if other.count == 0:
            return
        if self.count == 0:
            self = other.copy()
            return
        var total = self.count + other.count
        var delta = other.mean - self.mean
        var weight = Float64(other.count) / Float64(total)
        self.mean += delta * weight
        self.m2 += other.m2 + delta * delta * Float64(self.count) * Float64(
            other.count
        ) / Float64(total)
        self.count = total

    def variance(self, ddof: Int) -> Optional[Float64]:
        """Null when fewer than ddof + 1 values were seen."""
        if self.count - Int64(ddof) <= 0:
            return None
        return self.m2 / Float64(self.count - Int64(ddof))


struct MomentState(Copyable):
    """Running central moments up to the fourth, for skew and kurtosis.

    `add` is Terriberry's online update and `merge` Pébay's pairwise
    combination, so partitions merge in any order up to rounding.
    """

    var count: Int64
    var mean: Float64
    var m2: Float64
    var m3: Float64
    var m4: Float64

    def __init__(out self):
        self.count = 0
        self.mean = 0
        self.m2 = 0
        self.m3 = 0
        self.m4 = 0

    def add(mut self, value: Float64):
        var before = Float64(self.count)
        self.count += 1
        var n = Float64(self.count)
        var delta = value - self.mean
        var delta_n = delta / n
        var delta_n2 = delta_n * delta_n
        var term = delta * delta_n * before
        self.mean += delta_n
        self.m4 += (
            term * delta_n2 * (n * n - 3 * n + 3)
            + 6 * delta_n2 * self.m2
            - 4 * delta_n * self.m3
        )
        self.m3 += term * delta_n * (n - 2) - 3 * delta_n * self.m2
        self.m2 += term

    def merge(mut self, other: Self):
        if other.count == 0:
            return
        if self.count == 0:
            self = other.copy()
            return
        var a = Float64(self.count)
        var b = Float64(other.count)
        var n = a + b
        var delta = other.mean - self.mean
        var delta2 = delta * delta
        var m2 = self.m2 + other.m2 + delta2 * a * b / n
        var m3 = (
            self.m3
            + other.m3
            + delta2 * delta * a * b * (a - b) / (n * n)
            + 3 * delta * (a * other.m2 - b * self.m2) / n
        )
        var m4 = (
            self.m4
            + other.m4
            + delta2 * delta2 * a * b * (a * a - a * b + b * b) / (n * n * n)
            + 6 * delta2 * (a * a * other.m2 + b * b * self.m2) / (n * n)
            + 4 * delta * (a * other.m3 - b * self.m3) / n
        )
        self.mean += delta * b / n
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.count += other.count

    def skew(self, bias: Bool) -> Optional[Float64]:
        """Sample skewness as scipy and Polars define it: null with no
        values, or with fewer than three when bias is corrected; NaN for
        constant input."""
        if self.count == 0 or (not bias and self.count < 3):
            return None
        var n = Float64(self.count)
        var g1 = sqrt(n) * self.m3 / (self.m2 * sqrt(self.m2))
        if bias:
            return g1
        return g1 * sqrt(n * (n - 1)) / (n - 2)

    def kurtosis(self, fisher: Bool, bias: Bool) -> Optional[Float64]:
        """Sample kurtosis (excess when fisher): null with no values, or with
        fewer than four when bias is corrected; NaN for constant input."""
        if self.count == 0 or (not bias and self.count < 4):
            return None
        var n = Float64(self.count)
        var g2 = n * self.m4 / (self.m2 * self.m2)
        var excess = g2 - 3
        if not bias:
            excess = ((n + 1) * excess + 6) * (n - 1) / ((n - 2) * (n - 3))
        return excess if fisher else excess + 3


struct CoMomentState(Copyable):
    """Running means, co-moment and second moments of (x, y) pairs, for
    Pearson correlation and covariance; `merge` is the pairwise update."""

    var count: Int64
    var mean_x: Float64
    var mean_y: Float64
    var cxy: Float64
    var m2x: Float64
    var m2y: Float64

    def __init__(out self):
        self.count = 0
        self.mean_x = 0
        self.mean_y = 0
        self.cxy = 0
        self.m2x = 0
        self.m2y = 0

    def add(mut self, x: Float64, y: Float64):
        self.count += 1
        var n = Float64(self.count)
        var dx = x - self.mean_x
        self.mean_x += dx / n
        var dy = y - self.mean_y
        self.mean_y += dy / n
        self.cxy += dx * (y - self.mean_y)
        self.m2x += dx * (x - self.mean_x)
        self.m2y += dy * (y - self.mean_y)

    def merge(mut self, other: Self):
        if other.count == 0:
            return
        if self.count == 0:
            self = other.copy()
            return
        var a = Float64(self.count)
        var b = Float64(other.count)
        var n = a + b
        var dx = other.mean_x - self.mean_x
        var dy = other.mean_y - self.mean_y
        self.cxy += other.cxy + dx * dy * a * b / n
        self.m2x += other.m2x + dx * dx * a * b / n
        self.m2y += other.m2y + dy * dy * a * b / n
        self.mean_x += dx * b / n
        self.mean_y += dy * b / n
        self.count += other.count

    def correlation(self) -> Float64:
        """Pearson's r; NaN below two pairs or for a constant side."""
        if self.count < 2:
            return Float64(0) / Float64(0)
        return self.cxy / sqrt(self.m2x * self.m2y)

    def covariance(self, ddof: Int) -> Optional[Float64]:
        """Null with no pairs; 0.0 when pairs do not exceed ddof, as Polars
        returns; otherwise the co-moment over (pairs - ddof)."""
        if self.count == 0:
            return None
        if self.count <= Int64(ddof):
            return Float64(0)
        return self.cxy / Float64(self.count - Int64(ddof))
