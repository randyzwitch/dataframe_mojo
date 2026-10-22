"""Additional CPU timezone metadata for exact device wall-time resolution.

The existing version1 UTC lookup prefix is unchanged. Separate descriptors
identify the appended daylight-save and original candidate-offset planes.
No input rows or wall times are evaluated while building this metadata.
"""
from dataframe.timezone import TimeZone
from . import timezone_tables
from .memory import checked_add, checked_mul


@fieldwise_init
struct ResolutionTable(Movable):
    var words: List[UInt64]
    var saves: Int
    var candidates: Int
    var candidate_count: Int
    var construction_peak_bytes: Int


def serialize(zone: TimeZone) raises -> ResolutionTable:
    if len(zone._saves) != len(zone._offsets):
        raise Error("Mojo GPU time zone save table size is invalid")
    var prefix = timezone_tables.serialize(zone)
    var saves = len(prefix)
    var candidates = checked_add(saves, len(zone._saves))
    var count = len(zone._candidates)
    var length = checked_add(candidates, count)
    var peak = checked_mul(checked_add(prefix.capacity(), length), 8)
    var words = List[UInt64](capacity=length)
    words.extend(prefix^)
    for value in zone._saves:
        words.append(value.cast[DType.uint64]())
    for value in zone._candidates:
        words.append(value.cast[DType.uint64]())
    return ResolutionTable(words^, saves, candidates, count, peak)


def validate(
    words: List[UInt64],
    base: Int,
    saves: Int,
    candidates: Int,
    candidate_count: Int,
) raises:
    timezone_tables.validate(words, base)
    var prefix_end = checked_add(
        base,
        checked_add(
            timezone_tables.HEADER_WORDS,
            checked_add(checked_mul(Int(words[base]), 2), Int(words[base + 1])),
        ),
    )
    var saves_end = checked_add(saves, Int(words[base + 1]))
    var candidates_end = checked_add(candidates, candidate_count)
    if (
        saves < prefix_end
        or saves_end > candidates
        or candidates_end > len(words)
        or candidate_count > Int(Int32.MAX)
    ):
        raise Error("Mojo GPU time zone resolution planes are out of range")
