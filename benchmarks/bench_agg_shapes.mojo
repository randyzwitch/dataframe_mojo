"""Group-by with several aggregation lists. Args: CSV REPS. Prints shape, key, best ns."""
from std.sys import argv
from std.time import monotonic
from dataframe import CsvField, CsvSchema, DataFrame, Expr, col, read_csv


def main() raises:
    var args = argv()
    var frame = read_csv(
        String(args[1]),
        CsvSchema(
            [
                CsvField.int64("key_low"),
                CsvField.int64("key_high"),
                CsvField.int64("key_skew"),
                CsvField.string("key_str"),
                CsvField.int64("jk"),
                CsvField.float64("x"),
                CsvField.float64("y"),
                CsvField.int64("n"),
            ]
        ),
    )
    var reps = Int(String(args[2]))
    var shapes: List[String] = [
        "sum_count",
        "sum_count_mean",
        "sum",
        "min_max_int",
    ]
    var keys: List[String] = ["key_low", "key_high", "key_skew"]
    for shape in shapes:
        var exprs = List[Expr]()
        if shape == "sum_count":
            exprs = [col("x").sum().alias("s"), col("n").count().alias("c")]
        elif shape == "sum_count_mean":
            exprs = [
                col("x").sum().alias("s"),
                col("n").count().alias("c"),
                col("y").mean().alias("m"),
            ]
        elif shape == "sum":
            exprs = [col("x").sum().alias("s")]
        else:
            exprs = [col("n").min().alias("lo"), col("n").max().alias("hi")]
        for key in keys:
            var warm = frame.group_by(key).agg(exprs)
            var best = Int.MAX
            for _ in range(reps):
                var start = monotonic()
                var result = frame.group_by(key).agg(exprs)
                best = min(best, monotonic() - start)
                if result.height() != warm.height():
                    raise Error("height changed")
            print(shape, key, best, warm.height(), sep="\t")
