"""Native expression pipeline with SIMD Float64 arithmetic."""
from dataframe import Column, Series, DataFrame, col, lit


def main() raises:
    var sales = DataFrame(
        [
            Series(
                "region",
                Column[String](["east", "west", "east", "west", "north"]),
            ),
            Series(
                "amount",
                Column[Float64](
                    [100, 80, -20, 120, 999], [True, True, True, True, False]
                ),
            ),
        ]
    )
    var result = (
        sales.filter(col("amount") > lit(Float64(0)))
        .with_columns((col("amount") * lit(Float64(0.9))).alias("net"))
        .group_by("region", maintain_order=True)
        .agg(
            [
                col("net").sum().alias("revenue"),
                col("net").count().alias("sales"),
            ]
        )
    )
    var regions = result.column("region").string()
    var revenue = result.column("revenue").float64()
    var counts = result.column("sales").int64()
    for i in range(result.height()):
        print(regions.value(i), revenue.value(i), counts.value(i))
