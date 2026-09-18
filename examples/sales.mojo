"""Filter sales, derive net revenue, and sum by region without Python."""
from dataframe import Column, Series, DataFrame, greater_than, multiply


def main() raises:
    var regions: List[String] = ["east", "west", "east", "west", "north"]
    var amounts: List[Float64] = [100, 80, -20, 120, 999]
    var valid: List[Bool] = [True, True, True, True, False]
    var columns: List[Series] = [
        Series("region", Column[String](regions^)),
        Series("amount", Column[Float64](amounts^, valid)),
    ]
    var sales = DataFrame(columns^)
    var positive = sales.filter(
        greater_than(sales.column("amount").float64(), Float64(0))
    )
    var net = multiply(positive.column("amount").float64(), 0.9)
    var result = positive.with_column(Series("net", net^)).group_by_sum(
        "region", "net", "revenue"
    )
    var region = result.column("region").string()
    var revenue = result.column("revenue").float64()
    for i in range(result.height()):
        print(region.value(i), revenue.value(i))
