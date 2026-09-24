"""Large grouped multi-aggregate workloads with input construction untimed."""
from std.time import monotonic
from dataframe import Column, DataFrame, Series, Expr, col, scan_csv


def frame(rows: Int, groups: Int) raises -> DataFrame:
    var keys = List[Int64](capacity=rows)
    var x = List[Float64](capacity=rows)
    var y = List[Float64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for i in range(rows):
        keys.append(Int64(i % groups))
        x.append(Float64(i % 1000) * 0.125)
        y.append(Float64((i * 37) % 1000) * 0.25)
        valid.append(i % 10 != 0)
    return DataFrame(
        [
            Series("key", Column[Int64](keys^)),
            Series("x", Column[Float64](x^, valid^)),
            Series("y", Column[Float64](y^)),
        ]
    )


def bench(data: DataFrame, key: String, name: String) raises:
    var exprs: List[Expr] = [
        col("x").sum().alias("sx"),
        col("y").sum().alias("sy"),
        col("x").count().alias("cx"),
        col("y").count().alias("cy"),
    ]
    var reference_exprs = exprs.copy()
    reference_exprs.append(col("x").min().alias("minimum"))
    var expected = (
        data.group_by(key, maintain_order=True)
        .agg(reference_exprs)
        .drop(["minimum"])
    )
    var actual = data.group_by(key, maintain_order=True).agg(exprs)
    if actual.height() != expected.height():
        raise Error("fused aggregate changed group count")
    for row in range(actual.height()):
        if actual.item(row, key).int64() != expected.item(row, key).int64():
            raise Error("fused aggregate changed group key")
        for name in [String("cx"), "cy"]:
            if (
                actual.item(row, name).int64()
                != expected.item(row, name).int64()
            ):
                raise Error("fused aggregate changed count")
        for name in [String("sx"), "sy"]:
            var left = actual.item(row, name).float64()
            var right = expected.item(row, name).float64()
            if abs(left - right) > 1e-12 * max(1.0, abs(right)):
                raise Error("fused aggregate changed sum beyond rounding")
    for _ in range(7):
        var start = monotonic()
        var result = data.group_by(key).agg(exprs)
        print(name, " ns=", monotonic() - start, " groups=", result.height())


def main() raises:
    bench(frame(10_000_000, 1024), "key", "contiguous")
    bench(
        scan_csv("build/bench_polars/left_10000000.csv")
        .select(["key_low", "x", "y"])
        .collect(),
        "key_low",
        "csv_chunks",
    )
