"""Read typed CSV data and execute a native expression pipeline."""
from dataframe import CsvField, CsvSchema, col, lit, read_csv


def main() raises:
    var sales = read_csv(
        "examples/sales.csv",
        CsvSchema(
            [
                CsvField.string("region", False),
                CsvField.float64("amount"),
            ]
        ),
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
    print(result)
