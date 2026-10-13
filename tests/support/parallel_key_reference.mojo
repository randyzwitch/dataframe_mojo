"""Independent typed ordering and partition membership for parallel kernels."""
from std.math import isnan
from dataframe.comparison_sort import KeyOrder


def parallel_value(row: Int) -> Float64:
    if row % 97 == 11:
        return Float64(0) / Float64(0)
    if row % 101 == 0:
        return -Float64(0)
    return Float64((row * 19) % 31 - 15)


def parallel_present(row: Int) -> Bool:
    return row % 13 != 2


def parallel_group(row: Int, groups: Int) -> Int:
    return groups if row % 97 == 3 else row % groups


def rank_before(a: Int, b: Int, offset: Int, descending: Bool) -> Bool:
    var x = parallel_value(a + offset)
    var y = parallel_value(b + offset)
    if isnan(x) != isnan(y):
        return isnan(x) if descending else not isnan(x)
    if not isnan(x) and x != y:
        return x > y if descending else x < y
    return a < b


def expected_partition_mask(
    n: Int,
    offset: Int,
    groups: Int,
    limit: Int64,
    strict: Bool,
    descending: Bool,
) -> List[Bool]:
    var k = max(Int64(0), limit)
    if strict and k > 0:
        k -= 1
    var members = List[List[Int]]()
    for _ in range(groups + 1):
        members.append(List[Int]())
    for row in range(n):
        if parallel_present(row + offset):
            members[parallel_group(row + offset, groups)].append(row)
    var mask = List[Bool](length=n, fill=False)
    for group in members:
        var rows = group.copy()

        def before(a: Int, b: Int) {offset, descending} -> Bool:
            return rank_before(a, b, offset, descending)

        sort(rows, before)
        for i in range(min(len(rows), Int(k))):
            mask[rows[i]] = True
    return mask^


def parallel_primary(row: Int, distinct: Int, skew: Bool) -> Int:
    return 0 if skew and row % 10 != 0 else (row * 37) % distinct


def parallel_secondary(row: Int) -> Int:
    return (row * 17 // 3) % 7


def parallel_string(row: Int) -> String:
    var tails: List[String] = ["", "\0", "a", "a\0", "b", "é", "z"]
    return "abcdefghijklmnopqrstuvwx" + tails[parallel_secondary(row)]


def prefix_remainder_string(row: Int) -> String:
    # Distinct first-byte families prevent packed sort from stripping the
    # common 24-byte prefix that equal-family pairs must fully compare.
    return ("a" if row % 2 == 0 else "z") + parallel_string(row)


struct ScalarParallelOrder(KeyOrder):
    var distinct: Int
    var skew: Bool
    var descending: Bool
    var strings: Bool

    def __init__(
        out self,
        distinct: Int,
        skew: Bool,
        descending: Bool,
        strings: Bool = False,
    ):
        self.distinct = distinct
        self.skew = skew
        self.descending = descending
        self.strings = strings

    def less(self, a: Int, b: Int) -> Bool:
        var x = parallel_primary(a, self.distinct, self.skew)
        var y = parallel_primary(b, self.distinct, self.skew)
        if x != y:
            return x > y if self.descending else x < y
        if self.strings:
            var xs = parallel_string(a)
            var ys = parallel_string(b)
            if xs != ys:
                return xs < ys
        else:
            var xs = parallel_secondary(a)
            var ys = parallel_secondary(b)
            if xs != ys:
                return xs < ys
        return a < b


def scalar_parallel_rows(n: Int, order: ScalarParallelOrder) -> List[Int]:
    var rows = List[Int]()
    for row in range(n):
        rows.append(row)

    def before(a: Int, b: Int) {imm order} -> Bool:
        return order.less(a, b)

    sort(rows, before)
    return rows^


def expected_parallel_ranks(
    n: Int, ids: List[Int], method: String, descending: Bool
) -> List[Float64]:
    var rows = List[Int]()
    for row in range(n):
        if parallel_present(row + 7):
            rows.append(row)

    def before(a: Int, b: Int) {imm ids, descending} -> Bool:
        var ga = ids[a] if len(ids) > 0 else 0
        var gb = ids[b] if len(ids) > 0 else 0
        return ga < gb if ga != gb else rank_before(a, b, 7, descending)

    sort(rows, before)
    var out = List[Float64](length=n, fill=Float64(0) / Float64(0))
    var start = 0
    var partition_start = 0
    var previous_group = -1
    var dense = 0
    while start < len(rows):
        var group = ids[rows[start]] if len(ids) > 0 else 0
        if group != previous_group:
            previous_group = group
            partition_start = start
            dense = 0
        var finish = start + 1
        var value = parallel_value(rows[start] + 7)
        while finish < len(rows):
            var next_group = ids[rows[finish]] if len(ids) > 0 else 0
            var next_value = parallel_value(rows[finish] + 7)
            if next_group != group or not (
                (isnan(value) and isnan(next_value)) or value == next_value
            ):
                break
            finish += 1
        dense += 1
        for i in range(start, finish):
            out[rows[i]] = (
                Float64(i - partition_start + 1) if method
                == "ordinal" else Float64(start - partition_start + 1) if method
                == "min" else Float64(finish - partition_start) if method
                == "max" else Float64(dense) if method
                == "dense" else Float64(
                    start + 1 + finish - 2 * partition_start
                )
                / 2
            )
        start = finish
    return out^
