"""Comparator mergesort versus rank-resolved sorting; setup is untimed."""
from std.time import monotonic

from dataframe import Column, DataFrame, Series

comptime ROWS = 200000
comptime ITERATIONS = 3


def _data() raises -> DataFrame:
    var ints = List[Int64](capacity=ROWS)
    var strings = List[String](capacity=ROWS)
    var valid = List[Bool](capacity=ROWS)
    var state = UInt64(12345)
    for i in range(ROWS):
        state = state * 6364136223846793005 + 1442695040888963407
        var r = Int((state >> 33) % 100000)
        ints.append(Int64(r))
        strings.append("key-" + String(r % 5000))
        valid.append(i % 17 != 0)
    return DataFrame(
        [
            Series("i", Column[Int64](ints^, valid)),
            Series("s", Column[String](strings^, valid)),
        ]
    )


def _best_reference(series: Series) raises -> Int:
    var best = Int(9223372036854775807)
    for _ in range(ITERATIONS):
        var start = monotonic()
        var order = series._argsort_reference(False, True)
        best = min(best, monotonic() - start)
        if len(order) != ROWS:
            raise Error("wrong length")
    return best


def _best_ranked(series: Series) raises -> Int:
    var best = Int(9223372036854775807)
    for _ in range(ITERATIONS):
        var start = monotonic()
        var order = series.argsort(False, True)
        best = min(best, monotonic() - start)
        if len(order) != ROWS:
            raise Error("wrong length")
    return best


def main() raises:
    var frame = _data()
    print("workload,rows,reference_ns,ranked_ns")
    for name in ["i", "s"]:
        var series = frame.column(name)
        if series.argsort() != series._argsort_reference():
            raise Error("sort results differ")
        print(
            "single_" + series.dtype(),
            ",",
            ROWS,
            ",",
            _best_reference(series),
            ",",
            _best_ranked(series),
            sep="",
        )
    var best = Int(9223372036854775807)
    for _ in range(ITERATIONS):
        var start = monotonic()
        var order = frame.arg_sort(["s", "i"])
        best = min(best, monotonic() - start)
        if len(order) != ROWS:
            raise Error("wrong length")
    print("multi_string_int64,", ROWS, ",,", best, sep="")
    best = Int(9223372036854775807)
    for _ in range(ITERATIONS):
        var start = monotonic()
        var top = frame.top_k(10, ["i"])
        best = min(best, monotonic() - start)
        if top.height() != 10:
            raise Error("wrong top_k height")
    print("top_k_10_int64,", ROWS, ",,", best, sep="")
