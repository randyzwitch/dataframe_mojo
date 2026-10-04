from std.time import monotonic
from std.os import getenv
from dataframe import DataFrame, Series, Column, Expr, col
from dataframe.hashing import encode_rows
from dataframe.partition import low_cardinality, small_key_product
from dataframe.frame import _StreamReduction


def main() raises:
    var n = Int(getenv("PROFILE_ROWS", "1000000"))
    for count in [1, 2, 3]:
        for shape in [0, 1, 2]:
            if count == 1 and shape == 2:
                continue
            for groups in [10, 1000, 500000]:
                var columns = List[Series]()
                var names = List[String]()
                var keys = List[Series]()
                for k in range(count):
                    var name = "k" + String(k)
                    names.append(name)
                    if shape == 1 or (shape == 2 and k % 2 == 1):
                        var values = List[String](capacity=n)
                        for i in range(n):
                            values.append(
                                "value_" + String((i * 37 % groups) + k)
                            )
                        columns.append(Series(name, Column[String](values^)))
                    else:
                        var values = List[Int64](capacity=n)
                        for i in range(n):
                            values.append(Int64((i * 37 % groups) + k))
                        columns.append(Series(name, Column[Int64](values^)))
                    keys.append(columns[len(columns) - 1].copy())
                var values = List[Int64](capacity=n)
                for i in range(n):
                    values.append(Int64(i % 101 - 50))
                columns.append(Series("v", Column[Int64](values^)))
                var frame = DataFrame(columns^)
                var times = List[Float64]()
                for rep in range(9):
                    var start = monotonic()
                    var result = frame.group_by(names).agg(col("v").sum())
                    var elapsed = Float64(monotonic() - start) / 1e6
                    if rep > 0:
                        times.append(elapsed)
                    if result.height() != min(n, groups):
                        raise Error("group count")
                    if (
                        n == 1_000_000
                        and result.select(col("v").sum().alias("total"))
                        .item(0, "total")
                        .int64()
                        != -50
                    ):
                        raise Error("sum conservation")
                var ordered = times.copy()
                sort(ordered)
                print(
                    count,
                    shape,
                    groups,
                    "aggregate",
                    (ordered[3] + ordered[4]) / 2,
                    "samples",
                    times,
                )
