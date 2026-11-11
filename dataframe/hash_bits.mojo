"""Bit-stable integer mixing, independent of key encoding and RNG state."""


@always_inline
def _splitmix64(value: UInt64) -> UInt64:
    """SplitMix64's finalizer only; the caller owns any state increment."""
    var mixed = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EB
    return mixed ^ (mixed >> 31)
