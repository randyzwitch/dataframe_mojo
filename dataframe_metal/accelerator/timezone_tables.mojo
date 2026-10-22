"""Metadata-only transport of the existing CPU timezone tables."""
from dataframe.timezone import TimeZone
from .memory import checked_add, checked_mul

comptime HEADER_WORDS = 16
comptime FORMAT_VERSION = 1


def serialize(zone: TimeZone) raises -> List[UInt64]:
    if len(zone._times) != len(zone._types) or not len(zone._offsets):
        raise Error("Mojo GPU invalid time zone table")
    for kind in zone._types:
        if kind < 0 or kind >= len(zone._offsets):
            raise Error("Mojo GPU time zone transition type is out of range")
    if (
        zone._std < -1
        or zone._dst < -1
        or zone._std >= len(zone._offsets)
        or zone._dst >= len(zone._offsets)
    ):
        raise Error("Mojo GPU time zone footer type is out of range")
    var words = List[UInt64]()
    words.extend(
        [
            UInt64(len(zone._times)),
            UInt64(len(zone._offsets)),
            Int64(zone._std).cast[DType.uint64](),
            Int64(zone._dst).cast[DType.uint64](),
        ]
    )
    words.extend(
        [
            Int64(zone._start.kind).cast[DType.uint64](),
            zone._start.month.cast[DType.uint64](),
            zone._start.week.cast[DType.uint64](),
            zone._start.day.cast[DType.uint64](),
            zone._start.seconds.cast[DType.uint64](),
            Int64(zone._end.kind).cast[DType.uint64](),
            zone._end.month.cast[DType.uint64](),
            zone._end.week.cast[DType.uint64](),
            zone._end.day.cast[DType.uint64](),
            zone._end.seconds.cast[DType.uint64](),
            UInt64(FORMAT_VERSION),
            UInt64(0),
        ]
    )
    for i in range(len(zone._times)):
        words.append(zone._times[i].cast[DType.uint64]())
        words.append(UInt64(zone._types[i]))
    for value in zone._offsets:
        words.append(value.cast[DType.uint64]())
    return words^


def validate(words: List[UInt64], base: Int) raises:
    """Check every device-visible address/type index before context creation."""
    if base < 0 or base > len(words) or len(words) - base < HEADER_WORDS:
        raise Error("Mojo GPU time zone header is out of range")
    if words[base] > UInt64(Int.MAX) or words[base + 1] > UInt64(Int.MAX):
        raise Error("Mojo GPU time zone table size overflows")
    var transitions = Int(words[base])
    var kinds = Int(words[base + 1])
    var size = checked_add(
        HEADER_WORDS, checked_add(checked_mul(transitions, 2), kinds)
    )
    if (
        not kinds
        or size > len(words) - base
        or words[base + 14] != UInt64(FORMAT_VERSION)
        or words[base + 15] != 0
    ):
        raise Error("Mojo GPU time zone table metadata is invalid")
    for at in [2, 3]:
        var kind = words[base + at].cast[DType.int64]()
        if kind < -1 or kind >= Int64(kinds):
            raise Error("Mojo GPU time zone footer type is out of range")
    for row in range(transitions):
        if words[base + HEADER_WORDS + 2 * row + 1] >= UInt64(kinds):
            raise Error("Mojo GPU time zone transition type is out of range")
    if (
        words[base + 2].cast[DType.int64]() >= 0
        and words[base + 3].cast[DType.int64]() >= 0
    ):
        if words[base + 4] > 2 or words[base + 9] > 2:
            raise Error("Mojo GPU time zone recurrence kind is invalid")
