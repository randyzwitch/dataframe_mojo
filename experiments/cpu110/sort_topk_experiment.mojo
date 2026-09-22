"""#108 full-pipeline bounded selection versus sorting, including nulls."""
from std.time import monotonic

from dataframe import Column, DataFrame, Series


comptime ROWS = 1_000_000
comptime K = 10
comptime REPETITIONS = 3


def _frame() raises -> DataFrame:
    var values = List[Int64](capacity=ROWS)
    var valid = List[Bool](capacity=ROWS)
    var state = UInt64(0xD1B54A32D192ED03)
    for _ in range(ROWS):
        state = state * 6364136223846793005 + 1442695040888963407
        # Repeats check stable tie selection; nulls exercise rank conversion.
        values.append(Int64((state >> 24) % 50_003))
        valid.append((state & 31) != 0)
    return DataFrame([Series("k", Column[Int64](values^, valid^))])


def _best_top(frame: DataFrame) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var start = monotonic()
        var result = frame.top_k(K, "k")
        best = min(best, monotonic() - start)
        if result.height() != K:
            raise Error("top_k returned wrong height")
    return best


def _best_full(frame: DataFrame) raises -> Int:
    var best = Int.MAX
    for _ in range(REPETITIONS):
        var start = monotonic()
        var result = frame.sort("k", descending=True).head(K)
        best = min(best, monotonic() - start)
        if result.height() != K:
            raise Error("sort head returned wrong height")
    return best


def main() raises:
    var frame = _frame()
    var expected = frame.sort("k", descending=True).head(K)
    var actual = frame.top_k(K, "k")
    if not actual.equals(expected):
        raise Error("top_k differs from stable full-sort head")
    print("workload,rows,k,full_sort_ns,top_k_ns")
    print(
        "nullable_duplicate_int64,",
        ROWS,
        ",",
        K,
        ",",
        _best_full(frame),
        ",",
        _best_top(frame),
        sep="",
    )
