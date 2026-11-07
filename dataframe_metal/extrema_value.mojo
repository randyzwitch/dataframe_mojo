"""Decode ordered Float32 extrema with the shared CPU-adjudication policy."""
from .accelerator.faults import data_fault


def float_extrema_bits(
    value: Int64, valid: Bool, flags: UInt32
) raises -> UInt32:
    if not valid and flags & 1:
        raise data_fault(
            "all-NaN Float32 extrema require CPU payload adjudication"
        )
    var raw = UInt32(0)
    if valid:
        var key = UInt32(value)
        raw = key ^ 0x8000_0000 if key >> 31 else ~key
        if raw & 0x7FFF_FFFF == 0:
            if flags & 6 == 6:
                raise data_fault(
                    "mixed signed-zero extrema require CPU tie adjudication"
                )
            if flags & 4:
                raw = 0x8000_0000
    return raw
