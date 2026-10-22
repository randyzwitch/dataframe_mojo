"""One immutable text domain for O(G*K) GPU-produced partition edge controls.

Only produced controls reach this formatter. No source values/codes or CPU
expression evaluation are accepted. The shared CPU _bin_label defines spelling;
a global index heapsort deduplicates labels without a quadratic union per group.
"""
from std.memory import ArcPointer, bitcast
from dataframe.dtype import CategoricalDictionary, DataType
from dataframe.window import _bin_label
from dataframe_metal.accelerator.code_domains import text_less
from .residency import Scope, TextDomain, checked_add, checked_mul


struct Labels(Movable):
    var scope: Scope
    var domain: ArcPointer[TextDomain]
    var codes: List[UInt32]
    var peak: Int
    var retained: Int

    def __init__(
        out self,
        scope: Scope,
        domain: ArcPointer[TextDomain],
        var codes: List[UInt32],
        peak: Int,
    ) raises:
        if not domain[].scope.same_query(scope):
            raise Error("Partition label domain belongs to a different scope")
        var bytes = checked_mul(codes.capacity(), 4)
        scope._retain(bytes)
        self.scope = scope.copy()
        self.domain = domain.copy()
        self.codes = codes^
        self.peak = peak
        self.retained = bytes

    def __deinit__(deinit self):
        self.scope._release(self.retained)


def _sift(
    pool: CategoricalDictionary, mut order: List[Int], root: Int, end: Int
):
    var at = root
    while at * 2 + 1 < end:
        var chosen = at * 2 + 1
        if chosen + 1 < end and text_less(
            pool.get(order[chosen]), pool.get(order[chosen + 1])
        ):
            chosen += 1
        if not text_less(pool.get(order[at]), pool.get(order[chosen])):
            return
        var held = order[at]
        order[at] = order[chosen]
        order[chosen] = held
        at = chosen


def _edges(
    mut values: List[Float64],
    words: List[UInt64],
    group: Int,
    used: Int,
    count: Int,
):
    values.clear()
    for at in range(used):
        values.append(bitcast[DType.float64](words[group * max(1, count) + at]))


def build(
    scope: Scope,
    words: List[UInt64],
    used: List[UInt32],
    count: Int,
    left_closed: Bool,
    reservation: Int = 0,
) raises -> Labels:
    var groups = len(used)
    if (
        count < 0
        or count > Int(Int32.MAX)
        or groups <= 0
        or groups > Int(Int32.MAX)
        or reservation < 0
        or (count and len(words) < checked_mul(groups, count))
    ):
        raise Error("Partition label control geometry is invalid")
    var entries = 0
    for size in used:
        if UInt64(size) > UInt64(count):
            raise Error("Partition label used-edge count exceeds controls")
        entries = checked_add(entries, Int(size) + 1)
    var mapping = checked_mul(groups, count + 1)
    var minimum = checked_add(checked_mul(count, 8), checked_mul(mapping, 4))
    _ = scope.check(checked_add(reservation, minimum))
    var edges = List[Float64](capacity=count)
    var text_bytes = 0
    var largest = 0
    for group in range(groups):
        _edges(edges, words, group, Int(used[group]), count)
        for bin in range(Int(used[group]) + 1):
            var length = _bin_label(edges, bin, left_closed).byte_length()
            text_bytes = checked_add(text_bytes, length)
            largest = max(largest, length)
    var payload = checked_add(text_bytes, checked_mul(entries + 1, 8))
    # Two exact-capacity UTF8/offset pools; Int64 permutation+UInt32 code by
    # pool index; fixed-stride UInt32 output map; one reusable Float64 edge
    # vector; one shared formatter workspace. No G separately owned domains.
    var transient = checked_add(
        checked_mul(payload, 2),
        checked_add(
            checked_mul(entries, 12),
            checked_add(
                checked_mul(mapping, 4),
                checked_add(
                    checked_mul(count, 8), checked_mul(largest + 32, 16)
                ),
            ),
        ),
    )
    var peak = scope.check(checked_add(reservation, transient))
    var pool_bytes = List[UInt8](capacity=text_bytes)
    var pool_offsets = List[Int64](capacity=entries + 1)
    pool_offsets.append(0)
    var pool = CategoricalDictionary(pool_bytes^, pool_offsets^)
    var order = List[Int](capacity=entries)
    for group in range(groups):
        _edges(edges, words, group, Int(used[group]), count)
        for bin in range(Int(used[group]) + 1):
            pool.append(_bin_label(edges, bin, left_closed))
            order.append(len(order))
    for root in range(entries // 2 - 1, -1, -1):
        _sift(pool, order, root, entries)
    for end in range(entries - 1, 0, -1):
        var held = order[0]
        order[0] = order[end]
        order[end] = held
        _sift(pool, order, 0, end)
    var bytes = List[UInt8](capacity=text_bytes)
    var offsets = List[Int64](capacity=entries + 1)
    offsets.append(0)
    var dictionary = CategoricalDictionary(bytes^, offsets^)
    var codes_by_pool = List[UInt32](length=entries, fill=0)
    for at in range(entries):
        var label = pool.get(order[at])
        if at == 0 or label != pool.get(order[at - 1]):
            if len(dictionary) >= Int(UInt32.MAX):
                raise Error(
                    "Partition label domain exceeds UInt32 code capacity"
                )
            dictionary.append(label)
        codes_by_pool[order[at]] = UInt32(len(dictionary) - 1)
    var codes = List[UInt32](length=mapping, fill=0)
    var position = 0
    for group in range(groups):
        for bin in range(Int(used[group]) + 1):
            codes[group * (count + 1) + bin] = codes_by_pool[position]
            position += 1
    var domain = ArcPointer(
        TextDomain(scope, DataType.categorical(dictionary^), True)
    )
    return Labels(scope, domain, codes^, peak)
