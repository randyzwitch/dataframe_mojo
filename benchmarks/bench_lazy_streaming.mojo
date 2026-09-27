"""CSV filter/project/group pipeline; args path streaming batch_size rows."""
from std.sys import argv
from std.time import monotonic
from dataframe import scan_csv, col


def main() raises:
    var args = argv()
    var plan = (
        scan_csv(String(args[1]))
        .filter(col("v") >= 0)
        .select_exprs([col("k"), (col("v") * 2).alias("value")])
        .group_by("k", maintain_order=True)
        .agg(
            [
                col("value").sum().alias("sum"),
                col("value").count().alias("count"),
            ]
        )
    )
    var start = monotonic()
    var result = plan.collect(
        streaming=Int(String(args[2])) != 0, batch_size=Int(String(args[3]))
    )
    var elapsed = monotonic() - start
    var count = result.select(col("count").sum()).item().int64()
    var total = result.select(col("sum").sum()).item().int64()
    var rows = Int64(Int(String(args[4])))
    if count != rows or total != rows * (rows - 1):
        raise Error("wrong pipeline result")
    print(elapsed, count, total)
