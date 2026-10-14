"""Private logical text provenance and final dynamic dictionary ownership.

Carrier UInt32 is never itself evidence of text. Each field retains its CPU
logical dtype; byte/geometry fields and numeric UInt32 fields remain unrelated.
This module contains host metadata only and never imports a GPU owner/kernel.
"""
from std.memory import Pointer
from dataframe.dtype import DataType
from .memory import checked_add, checked_mul

comptime TEXT_LITERAL = -2147483647


@fieldwise_init
struct TextField(Copyable):
    var name: String
    var dtype: DataType


struct TextQuery(Movable):
    var base_domain: DataType
    var final_domains: List[Optional[DataType]]
    var final_owned_bytes: Int

    def __init__(out self, base_domain: DataType):
        self.base_domain = base_domain
        self.final_domains = List[Optional[DataType]]()
        self.final_owned_bytes = 0

    def record_final(mut self, domains: List[Optional[DataType]]) raises:
        # Query execution is serial; the outer immutable Arc keeps metadata
        # alive while the one executing DAG publishes its final owner list.
        self.final_domains = domains.copy()
        self.final_owned_bytes = 0
        var seen = List[DataType]()
        for dtype in domains:
            if not dtype or same_dictionary(dtype.value(), self.base_domain):
                continue
            var duplicate = False
            for old in seen:
                duplicate |= same_dictionary(old, dtype.value())
            if duplicate:
                continue
            seen.append(dtype.value())
            self.final_owned_bytes = checked_add(
                self.final_owned_bytes, dictionary_bytes(dtype.value())
            )


def dictionary_bytes(dtype: DataType) raises -> Int:
    if not dtype.is_categorical():
        raise Error("Private text domain requires dictionary metadata")
    var owner = dtype.dictionary()
    return checked_add(
        owner[].bytes.capacity(), checked_mul(owner[].offsets.capacity(), 8)
    )


def same_dictionary(left: DataType, right: DataType) -> Bool:
    if not left.is_categorical() or not right.is_categorical():
        return False
    return Int(Pointer(to=left.dictionary()[])) == Int(
        Pointer(to=right.dictionary()[])
    )


def text_less(left: StringSlice, right: StringSlice) -> Bool:
    var a = left.as_bytes()
    var b = right.as_bytes()
    for at in range(min(len(a), len(b))):
        if a[at] != b[at]:
            return a[at] < b[at]
    return len(a) < len(b)


def code(dtype: DataType, value: StringSlice) raises -> UInt32:
    var owner = dtype.dictionary()
    var lo = 0
    var hi = len(owner[])
    while lo < hi:
        var middle = lo + (hi - lo) // 2
        if text_less(owner[].get(middle), value):
            lo = middle + 1
        else:
            hi = middle
    if lo >= len(owner[]) or owner[].get(lo) != value:
        raise Error(
            "Private text domain is missing a required dictionary value"
        )
    return UInt32(lo)
