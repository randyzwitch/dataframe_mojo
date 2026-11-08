"""Bounded collision-checked group/value tables for exact distinct counts."""
comptime DISTINCT_CAPACITY_FAULT = UInt32(0x40000000)
comptime MAX_DISTINCT_CAPACITY = 2097152
comptime MAX_DISTINCT_VALUES = MAX_DISTINCT_CAPACITY // 2
comptime MAX_DISTINCT_PROBES = 512


@fieldwise_init
struct DistinctLayout(Copyable, Movable):
    var capacity: Int
    var fields: Int
    var unique_capacity: Int

    @always_inline
    def table_offset(self) -> Int:
        return 4 + self.fields

    @always_inline
    def first_offset(self) -> Int:
        return self.table_offset() + self.capacity

    @always_inline
    def counts_offset(self) -> Int:
        return self.first_offset() + self.capacity

    @always_inline
    def nulls_offset(self) -> Int:
        return self.counts_offset() + self.capacity * self.fields

    @always_inline
    def unique_offset(self) -> Int:
        return self.nulls_offset() + self.capacity * self.fields

    @always_inline
    def words(self) -> Int:
        return self.unique_offset() + self.unique_capacity * self.fields

    def check(self) raises:
        if (
            self.capacity < 512
            or self.capacity > 131072
            or self.capacity & (self.capacity - 1)
        ):
            raise Error(
                "Metal distinct group capacity must be a bounded power of two"
            )
        if self.fields < 1 or self.fields > 4:
            raise Error("Metal distinct supports one to four native values")
        if (
            self.unique_capacity < 512
            or self.unique_capacity > MAX_DISTINCT_CAPACITY
            or self.unique_capacity & (self.unique_capacity - 1)
        ):
            raise Error(
                "Metal distinct value capacity must be a bounded power of two"
            )
