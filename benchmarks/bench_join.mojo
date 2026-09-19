"""Join cost by side sizes and key skew; input construction is untimed."""
from std.time import monotonic

from dataframe import Column, DataFrame, Series

comptime ITERATIONS = 3


def _side(rows: Int, keys: Int, skew: Bool, seed: UInt64) raises -> DataFrame:
    var k = List[Int64](capacity=rows)
    var v = List[Float64](capacity=rows)
    var state = seed
    for i in range(rows):
        state = state * 6364136223846793005 + 1442695040888963407
        var r = Int((state >> 33) % UInt64(keys))
        # Skewed inputs send half the rows to one hot key.
        k.append(Int64(0 if skew and i % 2 == 0 else r))
        v.append(Float64(i))
    return DataFrame(
        [
            Series("k", Column[Int64](k^)),
            Series("v", Column[Float64](v^)),
        ]
    )


def main() raises:
    print("how,left_rows,right_rows,skew,output_rows,best_ns")
    var shapes: List[List[Int]] = [
        [200000, 2000],
        [2000, 200000],
        [100000, 100000],
    ]
    for shape in shapes:
        for skew in [False, True]:
            var left = _side(shape[0], 50000, skew, 1)
            var right = _side(shape[1], 50000, False, 2)
            for how in ["inner", "left", "semi"]:
                var best = Int(9223372036854775807)
                var rows = 0
                for _ in range(ITERATIONS):
                    var start = monotonic()
                    var joined = left.join(right, "k", how)
                    best = min(best, monotonic() - start)
                    rows = joined.height()
                print(
                    how,
                    ",",
                    shape[0],
                    ",",
                    shape[1],
                    ",",
                    skew,
                    ",",
                    rows,
                    ",",
                    best,
                    sep="",
                )
