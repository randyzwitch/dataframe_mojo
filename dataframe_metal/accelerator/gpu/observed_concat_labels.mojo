"""Build a sorted dictionary from compact observed source-code pairs.

Labels are copied directly into one exact-capacity UTF8/offset pool. This
avoids a host String allocation for every pair and never reads source rows.
"""
from std.memory import ArcPointer
from dataframe.dtype import CategoricalDictionary, DataType
from dataframe_metal.accelerator.code_domains import text_less
from .residency import Scope, TextDomain, checked_add, checked_mul


def _append_label(
    mut target: List[UInt8],
    left: StringSlice,
    separator: StringSlice,
    right: StringSlice,
):
    for byte in left.as_bytes():
        target.append(byte)
    for byte in separator.as_bytes():
        target.append(byte)
    for byte in right.as_bytes():
        target.append(byte)


def _sift(
    pool: CategoricalDictionary, mut order: List[UInt32], root: Int, end: Int
):
    var at = root
    while at * 2 + 1 < end:
        var chosen = at * 2 + 1
        if chosen + 1 < end and text_less(
            pool.get(Int(order[chosen])), pool.get(Int(order[chosen + 1]))
        ):
            chosen += 1
        if not text_less(
            pool.get(Int(order[at])), pool.get(Int(order[chosen]))
        ):
            return
        var held = order[at]
        order[at] = order[chosen]
        order[chosen] = held
        at = chosen


def build(
    scope: Scope,
    left_codes: List[UInt32],
    right_codes: List[UInt32],
    source: DataType,
    separator: String,
    reservation: Int,
) raises -> Tuple[ArcPointer[TextDomain], List[UInt32], Int]:
    """Return the ordered result domain and output code for every pair ID."""
    var count = len(left_codes)
    if count != len(right_codes) or count >= Int(UInt32.MAX):
        raise Error("Observed concat compact pair shape is invalid")
    if not source.is_categorical():
        raise Error("Observed concat source dictionary is missing")
    var dictionary = source.dictionary()
    var separator_slice = StringSlice(separator)
    var total_bytes = 0
    for index in range(count):
        if left_codes[index] >= UInt32(len(dictionary[])) or right_codes[
            index
        ] >= UInt32(len(dictionary[])):
            raise Error(
                "Observed concat pair code is outside its source dictionary"
            )
        total_bytes = checked_add(
            total_bytes,
            checked_add(
                dictionary[].get(Int(left_codes[index])).byte_length(),
                checked_add(
                    separator.byte_length(),
                    dictionary[].get(Int(right_codes[index])).byte_length(),
                ),
            ),
        )
    var pool_payload = checked_add(total_bytes, checked_mul(count + 1, 8))
    var result_payload = checked_add(total_bytes, checked_mul(count + 1, 8))
    var working = checked_add(
        pool_payload,
        checked_add(checked_mul(count, 8), checked_mul(count, 4)),
    )
    working = checked_add(working, checked_mul(count, 4))
    working = checked_add(working, checked_add(checked_mul(count, 48), 128))
    var peak = scope.check(
        checked_add(reservation, checked_add(working, result_payload))
    )
    var pool_bytes = List[UInt8](capacity=total_bytes)
    var pool_offsets = List[Int64](capacity=count + 1)
    pool_offsets.append(0)
    var pool = CategoricalDictionary(pool_bytes^, pool_offsets^)
    for index in range(count):
        _append_label(
            pool.bytes,
            dictionary[].get(Int(left_codes[index])),
            separator_slice,
            dictionary[].get(Int(right_codes[index])),
        )
        pool.offsets.append(Int64(len(pool.bytes)))
    var order = List[UInt32](capacity=count)
    for index in range(count):
        order.append(UInt32(index))
    for root in range(count // 2 - 1, -1, -1):
        _sift(pool, order, root, count)
    for end in range(count - 1, 0, -1):
        var held = order[0]
        order[0] = order[end]
        order[end] = held
        _sift(pool, order, 0, end)
    var result_bytes = List[UInt8](capacity=total_bytes)
    var result_offsets = List[Int64](capacity=count + 1)
    result_offsets.append(0)
    var result = CategoricalDictionary(result_bytes^, result_offsets^)
    var pair_codes = List[UInt32](length=count, fill=0)
    for index in range(count):
        var pair_id = Int(order[index])
        var value = pool.get(pair_id)
        if index == 0 or value != pool.get(Int(order[index - 1])):
            result.append(value)
        pair_codes[pair_id] = UInt32(len(result) - 1)
    var domain = ArcPointer(
        TextDomain(scope, DataType.categorical(result^), True)
    )
    return (domain^, pair_codes^, peak)
