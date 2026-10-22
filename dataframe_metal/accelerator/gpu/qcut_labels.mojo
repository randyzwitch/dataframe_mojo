"""Default QCUT label metadata built from O(K) GPU-produced edge controls.

The CPU formatter defines public spelling and closure. This helper never
receives source values or row codes; classification remains on the device.
Labels live in two exact-capacity UTF8/offset buffers while an index heapsort
orders them. There is no pool of independently allocated String owners.
"""
from std.memory import ArcPointer, bitcast
from dataframe.dtype import CategoricalDictionary, DataType
from dataframe.window import _bin_label
from dataframe_metal.accelerator.code_domains import text_less
from .residency import Scope, TextDomain, checked_add, checked_mul


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
    words: List[UInt64],
    used: Int,
    left_closed: Bool,
    reservation: Int,
) raises -> Tuple[ArcPointer[TextDomain], List[UInt32], Int]:
    if used < 0 or used > len(words) or used >= Int(UInt32.MAX):
        raise Error("QCUT dynamic edge control shape is invalid")
    _ = scope.check(checked_add(reservation, checked_mul(used, 8)))
    var edges = List[Float64](capacity=used)
    for index in range(used):
        edges.append(bitcast[DType.float64](words[index]))
    var text_bytes = 0
    var largest = 0
    for index in range(used + 1):
        var length = _bin_label(edges, index, left_closed).byte_length()
        text_bytes = checked_add(text_bytes, length)
        largest = max(largest, length)
    var bins = used + 1
    var payload = checked_add(text_bytes, checked_mul(bins + 1, 8))
    # Unsorted UTF8/offset pool + sorted dictionary + index/code arrays +
    # Float64 controls coexist. The one-label formatter's temporary Strings
    # have a separately reserved constant envelope, rather than K owners.
    var transient = checked_add(
        payload, checked_add(checked_mul(bins, 12), checked_mul(used, 8))
    )
    transient = checked_add(transient, checked_mul(largest + 32, 16))
    var peak = scope.check(
        checked_add(reservation, checked_add(transient, payload))
    )
    var pool_bytes = List[UInt8](capacity=text_bytes)
    var pool_offsets = List[Int64](capacity=bins + 1)
    pool_offsets.append(0)
    var pool = CategoricalDictionary(pool_bytes^, pool_offsets^)
    var order = List[UInt32](capacity=bins)
    for index in range(bins):
        pool.append(_bin_label(edges, index, left_closed))
        order.append(UInt32(index))
    for root in range(bins // 2 - 1, -1, -1):
        _sift(pool, order, root, bins)
    for end in range(bins - 1, 0, -1):
        var held = order[0]
        order[0] = order[end]
        order[end] = held
        _sift(pool, order, 0, end)
    var bytes = List[UInt8](capacity=text_bytes)
    var offsets = List[Int64](capacity=bins + 1)
    offsets.append(0)
    var dictionary = CategoricalDictionary(bytes^, offsets^)
    var codes = List[UInt32](length=bins, fill=0)
    for index in range(bins):
        var value = pool.get(Int(order[index]))
        if index == 0 or value != pool.get(Int(order[index - 1])):
            dictionary.append(value)
        codes[Int(order[index])] = UInt32(len(dictionary) - 1)
    var domain = ArcPointer(
        TextDomain(scope, DataType.categorical(dictionary^), True)
    )
    return (domain^, codes^, peak)
