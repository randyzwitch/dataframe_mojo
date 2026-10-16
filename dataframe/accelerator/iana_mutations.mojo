"""Metadata-only descriptors for private, occurrence-certified zone mutations.

Public binding validates interval syntax unconditionally. Subsequent semantic
interval diagnostics are retained until a valid active device operand occurs.
Zone and policy validation is unconditional.
All table offsets are relative to this immutable capsule's word plane.
"""
from dataframe.binding import BoundExpr
from dataframe.dtype import DataType
from dataframe.expr import DT_REPLACE_TZ, DT_TRUNCATE, DT_OFFSET_BY
from dataframe.temporal import ambiguous_code, non_existent_code
from dataframe.temporal_kernels import Truncation, ZonedOffset, Offset
from dataframe.timezone import TimeZone, canonical_zone_syntax
from . import timezone_resolution_tables as resolution
from . import timezone_tables
from .memory import checked_add, checked_mul

comptime ROW_IANA_MUTATION = Int64(1023)
comptime DESCRIPTOR_WORDS = 16
comptime FORMAT_VERSION = UInt64(1)
comptime INTERVAL_ERROR = 7


struct Capsule(Movable):
    var words: List[UInt64]
    var names: List[String]
    var bases: List[Int]
    var saves: List[Int]
    var candidates: List[Int]
    var counts: List[Int]
    var descriptors: List[Int]
    var diagnostics: List[String]
    var construction_peak_bytes: Int

    def __init__(out self):
        self.words = List[UInt64]()
        self.names = List[String]()
        self.bases = List[Int]()
        self.saves = List[Int]()
        self.candidates = List[Int]()
        self.counts = List[Int]()
        self.descriptors = List[Int]()
        self.diagnostics = List[String]()
        self.construction_peak_bytes = 0

    def zone(mut self, name: String) raises -> Int:
        var canonical = canonical_zone_syntax(
            name
        ) if name.byte_length() else String("UTC")
        for index in range(len(self.names)):
            if self.names[index] == canonical:
                return index
        var table = resolution.serialize(TimeZone.load(canonical))
        # Cache descriptor scalars before moving the independently owned plane.
        var table_saves = table.saves
        var table_candidates = table.candidates
        var table_count = table.candidate_count
        var table_peak = table.construction_peak_bytes
        var previous_bytes = checked_mul(self.words.capacity(), 8)
        var table_bytes = checked_mul(table.words.capacity(), 8)
        var packed_words = table.words.copy()
        var base = len(self.words)
        _ = checked_add(base, len(table.words))
        self.construction_peak_bytes = max(
            self.construction_peak_bytes,
            checked_add(
                checked_mul(self.words.capacity(), 8),
                table_peak,
            ),
        )
        self.names.append(canonical)
        self.bases.append(base)
        self.saves.append(checked_add(base, table_saves))
        self.candidates.append(checked_add(base, table_candidates))
        self.counts.append(table_count)
        self.words.extend(packed_words^)
        self.construction_peak_bytes = max(
            self.construction_peak_bytes,
            checked_add(
                checked_add(
                    previous_bytes, checked_mul(self.words.capacity(), 8)
                ),
                checked_mul(table_bytes, 2),
            ),
        )
        return len(self.names) - 1

    def append(mut self, bound: BoundExpr, index: Int) raises -> Int:
        ref node = bound.expr._nodes[index]
        var op = node.op
        if op != DT_REPLACE_TZ and op != DT_TRUNCATE and op != DT_OFFSET_BY:
            raise Error("invalid IANA mutation operation")
        var dtype = bound.dtypes[node.left]
        if not dtype.is_datetime():
            raise Error("IANA mutation requires Datetime")
        var source: Int
        var target: Int
        var ambiguous = 0
        var non_existent = 0
        var parameter = Int64(0)
        var days = Int64(0)
        var ticks = Int64(0)
        var months = False
        var diagnostic = String("")
        if op == DT_REPLACE_TZ:
            var options = node.text2.split(",")
            ambiguous = ambiguous_code(String(options[0]))
            non_existent = non_existent_code(String(options[1]))
            source = self.zone(dtype.time_zone())
            target = self.zone(node.text)
        elif op == DT_TRUNCATE:
            source = self.zone(dtype.time_zone())
            target = source
            try:
                var interval = Truncation(dtype, node.text)
                months = interval.months != 0
                parameter = interval.months if months else interval.step
            except error:
                diagnostic = String(error)
        else:
            source = self.zone(dtype.time_zone())
            target = source
            try:
                if dtype.time_zone().byte_length():
                    var interval = ZonedOffset(dtype, node.text)
                    parameter = interval.months
                    days = interval.days
                    ticks = interval.ticks
                else:
                    var interval = Offset(dtype, node.text)
                    parameter = interval.months
                    ticks = interval.ticks
            except error:
                diagnostic = String(error)
        var base = len(self.words)
        _ = checked_add(base, DESCRIPTOR_WORDS)
        self.words.extend(
            [
                FORMAT_VERSION,
                UInt64(op),
                UInt64(self.bases[source]),
                UInt64(self.bases[target]),
                UInt64(self.saves[target]),
                UInt64(self.candidates[target]),
                UInt64(self.counts[target]),
                UInt64(
                    1 if dtype.unit()
                    == "ms" else 2 if dtype.unit()
                    == "us" else 3
                ),
                UInt64(ambiguous),
                UInt64(non_existent),
                UInt64(dtype.time_zone().byte_length() != 0),
                UInt64(
                    node.text.byte_length()
                    != 0 if op
                    == DT_REPLACE_TZ else months
                ),
                parameter.cast[DType.uint64](),
                days.cast[DType.uint64](),
                ticks.cast[DType.uint64](),
                UInt64(diagnostic.byte_length() != 0),
            ]
        )
        self.descriptors.append(base)
        self.diagnostics.append(diagnostic)
        self.construction_peak_bytes = max(
            self.construction_peak_bytes, checked_mul(self.words.capacity(), 8)
        )
        return base

    def validate(self) raises:
        for index in range(len(self.names)):
            resolution.validate(
                self.words,
                self.bases[index],
                self.saves[index],
                self.candidates[index],
                self.counts[index],
            )
        for base in self.descriptors:
            validate_descriptor(self.words, base)
            if (
                base < 0
                or len(self.words) - base < DESCRIPTOR_WORDS
                or self.words[base] != FORMAT_VERSION
            ):
                raise Error("invalid IANA mutation descriptor")


def contains(bound: BoundExpr) -> Bool:
    for node in bound.expr._nodes:
        if node.left < 0:
            continue
        var dtype = bound.dtypes[node.left]
        if not dtype.is_datetime():
            continue
        if node.op == DT_REPLACE_TZ and (
            dtype.time_zone() not in List[String](["", "UTC"])
            or node.text not in List[String](["", "UTC"])
        ):
            return True
        if (
            node.op == DT_TRUNCATE or node.op == DT_OFFSET_BY
        ) and dtype.time_zone() not in List[String](["", "UTC"]):
            return True
    return False


def mutation(bound: BoundExpr, index: Int) -> Bool:
    ref node = bound.expr._nodes[index]
    return (
        node.left >= 0
        and bound.dtypes[node.left].is_datetime()
        and node.op in List[Int]([DT_REPLACE_TZ, DT_TRUNCATE, DT_OFFSET_BY])
    )


def validate_descriptor(words: List[UInt64], base: Int) raises:
    if base < 0 or base > len(words) or len(words) - base < DESCRIPTOR_WORDS:
        raise Error("IANA mutation descriptor is out of range")
    if words[base] != FORMAT_VERSION or words[base + 1] not in List[UInt64](
        [UInt64(DT_REPLACE_TZ), UInt64(DT_TRUNCATE), UInt64(DT_OFFSET_BY)]
    ):
        raise Error("IANA mutation descriptor version or operation is invalid")
    for field in [2, 3, 4, 5, 6]:
        if words[base + field] > UInt64(Int.MAX):
            raise Error("IANA mutation metadata overflows address range")
    timezone_tables.validate(words, Int(words[base + 2]))
    resolution.validate(
        words,
        Int(words[base + 3]),
        Int(words[base + 4]),
        Int(words[base + 5]),
        Int(words[base + 6]),
    )
    if (
        words[base + 7] < 1
        or words[base + 7] > 3
        or words[base + 8] > 3
        or words[base + 9] > 1
        or words[base + 10] > 1
        or words[base + 11] > 1
        or words[base + 15] > 1
    ):
        raise Error("IANA mutation unit or policy is invalid")
    if (
        words[base + 1] == UInt64(DT_TRUNCATE)
        and not words[base + 15]
        and words[base + 12].cast[DType.int64]() <= 0
    ):
        raise Error("IANA truncation descriptor step is invalid")
