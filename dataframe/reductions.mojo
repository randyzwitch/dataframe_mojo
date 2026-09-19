"""Mergeable reduction states, independent of execution scheduling.

At most Int.MAX Int64 inputs fit exactly in a signed 128-bit accumulator.
This allows arbitrary partitions and merge trees without intermediate overflow.
"""
from std.collections import Optional

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
