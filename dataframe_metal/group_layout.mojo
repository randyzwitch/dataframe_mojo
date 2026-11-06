"""Bounded scratch layout shared by general grouped host/device paths."""
comptime GENERAL_GROUP_LIMIT = 65536
comptime GENERAL_TABLE_LIMIT = GENERAL_GROUP_LIMIT * 2
comptime GROUP_CONTROL_WORDS = 4
comptime GROUP_VALUE_WORDS = 32


@fieldwise_init
struct GroupLayout(Copyable, Movable):
    var capacity: Int
    var fields: Int
    var blocks: Int
    var global_state: Bool

    @always_inline
    def state_blocks(self) -> Int:
        return 1 if self.global_state else self.blocks

    @always_inline
    def table_offset(self) -> Int:
        return GROUP_CONTROL_WORDS

    @always_inline
    def first_offset(self) -> Int:
        return self.table_offset() + self.capacity

    @always_inline
    def rows_offset(self) -> Int:
        return self.first_offset() + self.capacity

    @always_inline
    def states_offset(self) -> Int:
        return self.rows_offset() + self.capacity * self.state_blocks()

    @always_inline
    def entries(self) -> Int:
        return self.capacity * self.state_blocks() * self.fields

    @always_inline
    def words(self) -> Int:
        return self.states_offset() + self.entries() * GROUP_VALUE_WORDS

    @always_inline
    def resolved_values(self) -> Int:
        return self.entries() * 2

    def check(self) raises:
        if (
            self.capacity < 512
            or self.capacity > GENERAL_TABLE_LIMIT
            or (self.capacity & (self.capacity - 1))
        ):
            raise Error(
                "Metal group table capacity must be a bounded power of two"
            )
        if (
            self.fields < 1
            or self.fields > 4
            or self.blocks < 1
            or self.blocks > 128
        ):
            raise Error("Metal group layout has invalid field or launch counts")
        if not self.global_state and self.capacity > 512:
            raise Error(
                "Metal per-workgroup state requires a low-cardinality table"
            )
