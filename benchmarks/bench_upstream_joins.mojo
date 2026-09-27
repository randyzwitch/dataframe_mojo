"""Pinned DuckDB join workloads, with loading and checks outside timing.

Args: CASE LEFT_CSV RIGHT_CSV REPS. See docs/upstream-join-baseline.md.
"""
from std.sys import argv
from std.time import monotonic
from dataframe import DataFrame, DataType, col, read_csv


def query(left: DataFrame, right: DataFrame, kind: String) raises -> DataFrame:
    if kind == "highcardinality":
        return (
            left.lazy()
            .join(right.lazy(), "k")
            .group_by(["a", "b"])
            .agg(col("k").len().alias("n"))
            .sort("a")
            .head(5)
            .collect()
        )
    return (
        left.lazy()
        .join(right.lazy(), "k")
        .select(col("k").len().alias("n"))
        .collect()
    )


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error("CASE LEFT_CSV RIGHT_CSV REPS")
    var kind = String(args[1])
    if kind != "highcardinality" and kind != "duplicate_strings":
        raise Error("unknown case")
    var left_schema: List[Tuple[String, DataType]] = [
        (String("k"), DataType.STRING)
    ]
    var right_schema: List[Tuple[String, DataType]] = [
        (String("k"), DataType.STRING)
    ]
    if kind == "highcardinality":
        left_schema = [
            (String("k"), DataType.INT64),
            (String("a"), DataType.INT64),
        ]
        right_schema = [
            (String("k"), DataType.INT64),
            (String("b"), DataType.INT64),
        ]
    var left = read_csv(String(args[2]), schema=left_schema)
    var right = read_csv(String(args[3]), schema=right_schema)
    var reps = Int(String(args[4]))
    for rep in range(reps + 1):
        var start = monotonic()
        var result = query(left, right, kind)
        var elapsed = monotonic() - start
        if kind == "highcardinality":
            if result.height() != 5:
                raise Error("incorrect result height")
            for i in range(5):
                if (
                    result.item(i, "a").int64() != Int64(i)
                    or result.item(i, "b").int64() != Int64(i)
                    or result.item(i, "n").int64() != 1
                ):
                    raise Error("incorrect grouped result")
        elif result.item().int64() != Int64(4 * right.height()):
            raise Error("incorrect duplicate join count")
        if rep > 0:
            print(elapsed)
