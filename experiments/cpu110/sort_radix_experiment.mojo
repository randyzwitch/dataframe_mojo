"""Isolated #108 experiment: stable radix sort versus merge sort on ranks.

This deliberately excludes rank creation. `sort_indices` receives the same
already-encoded nonnegative rank values, so the measurement answers only
whether a radix replacement can repay its four histogram/scatter passes.
"""
from std.time import monotonic

from dataframe.series import sort_indices


comptime ROWS = 1_000_000
comptime REPETITIONS = 3
comptime RADIX = 1 << 16


def radix_indices(keys: List[Int]) -> List[Int]:
    """Four-pass stable LSD radix order for nonnegative Int ranks."""
    var source = List[Int](capacity=len(keys))
    for row in range(len(keys)):
        source.append(row)
    var target = List[Int](length=len(keys), fill=0)
    for shift in [0, 16, 32, 48]:
        var starts = List[Int](length=RADIX, fill=0)
        for row in source:
            starts[(keys[row] >> shift) & (RADIX - 1)] += 1
        var total = 0
        for bucket in range(RADIX):
            var count = starts[bucket]
            starts[bucket] = total
            total += count
        for row in source:
            var bucket = (keys[row] >> shift) & (RADIX - 1)
            target[starts[bucket]] = row
            starts[bucket] += 1
        var old = source^
        source = target^
        target = old^
    return source^


def _keys() -> List[Int]:
    var keys = List[Int](capacity=ROWS)
    var state = UInt64(0x9E3779B97F4A7C15)
    for _ in range(ROWS):
        state = state * 6364136223846793005 + 1442695040888963407
        # Dense repeated ranks make the required stable tie ordering visible.
        keys.append(Int((state >> 24) % 250_003))
    return keys^


def _best_merge(keys: List[Int]) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var start = monotonic()
        var order = sort_indices([keys.copy()])
        best = min(best, monotonic() - start)
        if len(order) != len(keys):
            raise Error("merge sort returned wrong length")
    return best


def _best_radix(keys: List[Int]) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var start = monotonic()
        var order = radix_indices(keys)
        best = min(best, monotonic() - start)
        if len(order) != len(keys):
            raise Error("radix sort returned wrong length")
    return best


def main() raises:
    var keys = _keys()
    var expected = sort_indices([keys.copy()])
    var actual = radix_indices(keys)
    if actual != expected:
        raise Error("radix order differs from stable merge order")
    print("workload,rows,merge_ns,radix_ns")
    print(
        "encoded_rank_only_repeated_int64,",
        ROWS,
        ",",
        _best_merge(keys),
        ",",
        _best_radix(keys),
        sep="",
    )
