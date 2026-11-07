"""Owned host readback from a bounded numeric grouped scan."""


@fieldwise_init
struct GroupedPartials(Movable):
    var first: List[UInt32]
    var rows: List[UInt32]
    var states: List[UInt32]
    var extrema: List[Int64]
    var blocks: Int
