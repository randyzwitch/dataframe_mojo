"""Mergeable reduction states, independent of execution scheduling.

At most Int.MAX Int64 inputs fit exactly in a signed 128-bit accumulator.
This allows arbitrary partitions and merge trees without intermediate overflow.
"""
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
