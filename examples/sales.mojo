"""Filter sales, derive net revenue, and sum by region without Python."""
from dataframe import Column, DataFrame, Series, col


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
        .agg(col("net").sum().alias("revenue"))
    )
    var region = result.column("region").string()
    var revenue = result.column("revenue").float64()
    for i in range(result.height()):
        print(region.value(i), revenue.value(i))
