"""Float32 ranges preserve gradual-underflow comparisons on Metal."""
from std.memory import bitcast


@always_inline
def _ordered(raw: UInt32) -> UInt32:
    # Normalize signed zero; map IEEE signs to monotonically ordered keys.
    var value = UInt32(0) if raw & 0x7FFF_FFFF == 0 else raw
    return ~value if value >> 31 else value ^ 0x8000_0000


@always_inline
def float_in_range(
    value: Float32, low: Float32, high: Float32, inclusive: UInt32
) -> Bool:
    var lower = bitcast[DType.uint32](low)
    var upper = bitcast[DType.uint32](high)
    var lm = lower & 0x7FFF_FFFF
    var hm = upper & 0x7FFF_FFFF
    if lm > 0x0080_0000 and hm > 0x0080_0000:
        # Flushing a subnormal to zero cannot cross either normal bound.
        var above = value >= low if inclusive & 1 else value > low
        var below = value <= high if inclusive & 2 else value < high
        return above & below
    var raw = bitcast[DType.uint32](value)
    if raw & 0x7FFF_FFFF > 0x7F80_0000 or lm > 0x7F80_0000 or hm > 0x7F80_0000:
        return False
    var key = _ordered(raw)
    var lo = _ordered(lower)
    var hi = _ordered(upper)
    var above = key >= lo if inclusive & 1 else key > lo
    var below = key <= hi if inclusive & 2 else key < hi
    return above & below
