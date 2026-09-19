"""Native expression pipeline with SIMD Float64 arithmetic."""
from dataframe import Column, Series, DataFrame, col


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
        sales.filter(col("amount") > 0)
        .with_columns((col("amount") * 0.9).alias("net"))
        .group_by("region", maintain_order=True)
        .agg(
            [
                col("net").sum().alias("revenue"),
                col("net").count().alias("sales"),
            ]
        )
    )
    print(result)
