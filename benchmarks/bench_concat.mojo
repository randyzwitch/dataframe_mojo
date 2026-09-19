"""Vertical concatenation scaling; input construction is untimed."""
from std.time import monotonic

from dataframe import Column, DataFrame, Series, concat

comptime PARTS = 8
comptime ITERATIONS = 5


def _part(rows: Int, offset: Int) raises -> DataFrame:
    var ints = List[Int64](capacity=rows)
    var floats = List[Float64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for i in range(rows):
        ints.append(Int64(offset + i))
        floats.append(Float64(i) / 3)
        valid.append(i % 5 != 0)
    return DataFrame(
        [
            Series("i", Column[Int64](ints^)),
            Series("f", Column[Float64](floats^, valid)),
        ]
    )


def main() raises:
    print("rows,parts,best_ns,ns_per_row")
    # Odd part sizes force the shifted validity-merge path at every boundary.
    var sizes: List[Int] = [12501, 125001, 1250001]
    for size in sizes:
        var frames = List[DataFrame]()
        for p in range(PARTS):
            frames.append(_part(size, p * size))
        var expected = size * PARTS
        var best = Int(9223372036854775807)
        for _ in range(ITERATIONS):
            var start = monotonic()
            var result = concat(frames)
            var elapsed = monotonic() - start
            if result.height() != expected:
                raise Error("concat benchmark produced the wrong height")
            best = min(best, elapsed)
        print(
            expected,
            ",",
            PARTS,
            ",",
            best,
            ",",
            Float64(best) / Float64(expected),
            sep="",
        )
