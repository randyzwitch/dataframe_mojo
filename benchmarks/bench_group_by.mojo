"""Grouping cost by key count and cardinality; setup is untimed."""
from std.time import monotonic

from dataframe import Column, DataFrame, Series, col

comptime ROWS = 200000
comptime ITERATIONS = 3


def _frame(cardinality: Int) raises -> DataFrame:
    var a = List[Int64](capacity=ROWS)
    var b = List[String](capacity=ROWS)
    var c = List[Float64](capacity=ROWS)
    var v = List[Float64](capacity=ROWS)
    var state = UInt64(99)
    for i in range(ROWS):
        state = state * 6364136223846793005 + 1442695040888963407
        var r = Int((state >> 33) % UInt64(cardinality))
        a.append(Int64(r))
        b.append("k" + String(r % 97))
        c.append(Float64(r % 13) / 4)
        v.append(Float64(i % 1000) / 10)
    return DataFrame(
        [
            Series("a", Column[Int64](a^)),
            Series("b", Column[String](b^)),
            Series("c", Column[Float64](c^)),
            Series("v", Column[Float64](v^)),
        ]
    )


def main() raises:
    print("keys,cardinality,groups,best_ns,ns_per_row")
    var key_sets: List[List[String]] = [["a"], ["a", "b"], ["a", "b", "c"]]
    for cardinality in [16, 10000, 150000]:
        var frame = _frame(cardinality)
        for keys in key_sets:
            var best = Int(9223372036854775807)
            var groups = 0
            for _ in range(ITERATIONS):
                var start = monotonic()
                var result = frame.group_by(keys).agg(
                    [col("v").sum(), col("v").count().alias("n")]
                )
                best = min(best, monotonic() - start)
                groups = result.height()
            print(
                len(keys),
                ",",
                cardinality,
                ",",
                groups,
                ",",
                best,
                ",",
                Float64(best) / Float64(ROWS),
                sep="",
            )
