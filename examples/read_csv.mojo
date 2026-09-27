"""Read typed CSV data and execute a native expression pipeline."""
from dataframe import DataType, col, read_csv


def main() raises:
    var sales = read_csv(
        "examples/sales.csv",
        schema=[("region", DataType.STRING), ("amount", DataType.FLOAT64)],
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
