"""CPU-compatible wall-time resolution over immutable timezone metadata.

All inputs and results here are seconds. Row adapters must separately preserve
subsecond remainders, checked tick arithmetic and public error messages.
"""
from std.memory import Pointer
from . import iana_fields

comptime OK = 0
comptime COULD_NOT_LOCALIZE = 1
comptime SHIFTED_NONEXISTENT = 2
comptime SHIFTED_AMBIGUOUS = 3


def type_at(
    table: Pointer[UInt64, ImmutAnyOrigin], base: Int, seconds: Int64
) -> Int:
    var count = Int(table[unsafe_offset=base])
    var start = base + iana_fields.HEADER_WORDS
    if count and seconds < table[unsafe_offset=start].cast[DType.int64]():
        return 0
    if (
        not count
        or seconds
        >= table[unsafe_offset=start + 2 * (count - 1)].cast[DType.int64]()
    ):
        if table[unsafe_offset=base + 2].cast[DType.int64]() >= 0:
            return iana_fields._rule_type(table, base, seconds)
        return Int(
            table[unsafe_offset=start + 2 * (count - 1) + 1]
        ) if count else 0
    var low = 0
    var high = count - 1
    while low < high:
        var middle = (low + high + 1) // 2
        if (
            table[unsafe_offset=start + 2 * middle].cast[DType.int64]()
            <= seconds
        ):
            low = middle
        else:
            high = middle - 1
    return Int(table[unsafe_offset=start + 2 * low + 1])


def save_at(
    table: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    saves: Int,
    seconds: Int64,
) -> Int64:
    return table[unsafe_offset=saves + type_at(table, base, seconds)].cast[
        DType.int64
    ]()


def resolve(
    table: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    candidates: Int,
    candidate_count: Int,
    local: Int64,
) -> Tuple[Int, Int64, Int64]:
    var count = 0
    var earliest = Int64(0)
    var latest = Int64(0)
    for index in range(candidate_count):
        var offset = table[unsafe_offset=candidates + index].cast[DType.int64]()
        var instant = local - offset
        if iana_fields.offset_at(table, base, instant) != offset:
            continue
        if not count or instant < earliest:
            earliest = instant
        if not count or instant > latest:
            latest = instant
        count += 1
    return (count, earliest, latest)


def relocalize(
    table: Pointer[UInt64, ImmutAnyOrigin],
    base: Int,
    saves: Int,
    candidates: Int,
    candidate_count: Int,
    local: Int64,
    original: Int64,
) -> Tuple[Int64, Int]:
    var found = resolve(table, base, candidates, candidate_count, local)
    if found[0] == 1:
        return (found[1], OK)
    var save = save_at(table, base, saves, original)
    if found[0] > 1:
        if save_at(table, base, saves, found[1]) == save:
            return (found[1], OK)
        if save_at(table, base, saves, found[2]) == save:
            return (found[2], OK)
        return (Int64(0), COULD_NOT_LOCALIZE)
    var shifted = Int64(0)
    if save != 0:
        shifted = local + save
    else:
        var later = resolve(
            table, base, candidates, candidate_count, local + 3600
        )
        if not later[0]:
            return (Int64(0), COULD_NOT_LOCALIZE)
        shifted = local - save_at(table, base, saves, later[1])
    var result = resolve(table, base, candidates, candidate_count, shifted)
    if result[0] == 1:
        return (result[1], OK)
    return (
        Int64(0),
        SHIFTED_NONEXISTENT if not result[0] else SHIFTED_AMBIGUOUS,
    )
