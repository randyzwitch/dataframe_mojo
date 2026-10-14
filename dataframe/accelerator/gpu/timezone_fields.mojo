"""Fixed-offset local ticks for the shared temporal field evaluator.

CPU temporal.to_local shifts Int64 storage before extracting fields, retaining
fixed-width wrapping at the endpoints. Localize and calendar mutations have
different checked associations and must not use this helper.
"""


def local_ticks(
    value: UInt64, offset_seconds: Int64, ticks_per_second: Int64
) -> UInt64:
    return value + (
        offset_seconds.cast[DType.uint64]()
        * ticks_per_second.cast[DType.uint64]()
    )
