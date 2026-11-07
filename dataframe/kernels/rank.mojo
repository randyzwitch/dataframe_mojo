"""Order-preserving 64-bit keys of fixed-width values, and their packing
above a row index into one 128-bit integer, for the top-k heap and the
aggregate table's ordered states. NaN orders above every number, and -0.0
equals 0.0.

(Ranks are read off the window layout's peer boundaries: see
`dataframe.window_sort.rank_rows`.)
"""
from std.memory import bitcast

comptime _SIGN = UInt64(1) << 63


@always_inline
def _key[D: DType](value: Scalar[D], descending: Bool) -> UInt64:
    """An unsigned key that orders as the value does."""
    var key: UInt64
    comptime if D.is_floating_point():
        var x = value.cast[DType.float64]()
        if x != x:
            key = UInt64(0x7FF8000000000000) ^ _SIGN  # above +inf
        elif x == 0:
            key = _SIGN  # -0.0 ranks with 0.0
        else:
            var bits = bitcast[DType.uint64](x)
            key = ~bits if (bits & _SIGN) != 0 else bits ^ _SIGN
    elif D.is_unsigned():
        key = value.cast[DType.uint64]()
    else:
        key = value.cast[DType.int64]().cast[DType.uint64]() ^ _SIGN
    return ~key if descending else key


@always_inline
def _pack(key: UInt64, row: Int) -> UInt128:
    return (UInt128(key) << 64) | UInt128(row)
