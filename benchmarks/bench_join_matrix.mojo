"""Head-to-head join shapes; input construction and result checks are untimed.

Usage: bench_join_matrix DATA_DIR ROWS REPETITIONS [CASE]

CASE runs one workload for profiling without constructing the other key shapes.
"""
from std.sys import argv
from std.time import monotonic

from dataframe import (
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Expr,
    col,
    lit,
    read_csv,
)


def left_schema() raises -> CsvSchema:
    return CsvSchema(
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
    )


def right_schema() raises -> CsvSchema:
    return CsvSchema([CsvField.int64("jk"), CsvField.float64("r")])


def checksum(frame: DataFrame, column: String) raises -> Float64:
    var value = frame.select(col(column).sum()).item()
    return 0.0 if value.is_null() else value.float64()


def emit(name: String, warm: DataFrame, best: Int, column: String) raises:
    print(name, best, warm.height(), checksum(warm, column), sep="\t")


def timed(
    left: DataFrame,
    right: DataFrame,
    name: String,
    keys: List[String],
    how: String,
    repetitions: Int,
    column: String,
) raises:
    var warm = left.join(right, keys, how)
    var best = Int.MAX
    for _ in range(repetitions):
        var start = monotonic()
        var result = left.join(right, keys, how)
        best = min(best, monotonic() - start)
        if result.height() != warm.height():
            raise Error(name + " row count changed")
    emit(name, warm, best, column)


def timed_lazy_narrow(
    left: DataFrame, right: DataFrame, repetitions: Int
) raises:
    var warm = left.lazy().join(right.lazy(), "jk").select(["x", "r"]).collect()
    var best = Int.MAX
    for _ in range(repetitions):
        var start = monotonic()
        var result = (
            left.lazy().join(right.lazy(), "jk").select(["x", "r"]).collect()
        )
        best = min(best, monotonic() - start)
        if result.height() != warm.height():
            raise Error("lazy_narrow row count changed")
    emit("lazy_narrow", warm, best, "r")


def main() raises:
    var args = argv()
    if len(args) != 4 and len(args) != 5:
        raise Error("usage: bench_join_matrix DATA_DIR ROWS REPETITIONS [CASE]")
    var only = String(args[4]) if len(args) == 5 else String()
    var rows = Int(String(args[2]))
    var repetitions = Int(String(args[3]))
    var base = String(args[1])
    var left = read_csv(base + "/left_" + String(rows) + ".csv", left_schema())
    var right = read_csv(
        base + "/right_" + String(rows) + ".csv", right_schema()
    )

    if only == "" or only == "inner_dense":
        timed(left, right, "inner_dense", ["jk"], "inner", repetitions, "r")
    if only == "" or only == "lazy_narrow":
        timed_lazy_narrow(left, right, repetitions)

    if only == "" or only == "inner_sparse":
        var sparse_left = left.with_columns(
            (col("jk") * lit(Int64(17))).alias("key")
        )
        var sparse_right = right.with_columns(
            (col("jk") * lit(Int64(17))).alias("key")
        )
        timed(
            sparse_left,
            sparse_right,
            "inner_sparse",
            ["key"],
            "inner",
            repetitions,
            "r",
        )

    if only == "" or only == "inner_string":
        var string_left = left.with_columns(
            col("jk").cast(DataType.STRING).alias("key")
        )
        var string_right = right.with_columns(
            col("jk").cast(DataType.STRING).alias("key")
        )
        timed(
            string_left,
            string_right,
            "inner_string",
            ["key"],
            "inner",
            repetitions,
            "r",
        )

    if only == "" or only == "inner_multi":
        var compound: List[Expr] = [
            (col("jk") // lit(Int64(4))).alias("a"),
            (col("jk") % lit(Int64(4))).alias("b"),
        ]
        var multi_left = left.with_columns(compound)
        var multi_right = right.with_columns(compound)
        timed(
            multi_left,
            multi_right,
            "inner_multi",
            ["a", "b"],
            "inner",
            repetitions,
            "r",
        )

    if (
        only == ""
        or only == "left_unmatched"
        or only == "right_unmatched"
        or only == "full_unmatched"
        or only == "semi_unmatched"
        or only == "anti_unmatched"
    ):
        var shifted = right.with_columns(
            (col("jk") + lit(Int64(rows // 4))).alias("jk")
        )
        for how in [String("left"), "right", "full", "semi", "anti"]:
            if only != "" and only != how + "_unmatched":
                continue
            var column = "x" if how == "semi" or how == "anti" else "r"
            timed(
                left,
                shifted,
                how + "_unmatched",
                ["jk"],
                how,
                repetitions,
                column,
            )

    if only == "" or only == "left_sparse":
        var sparse_left = left.with_columns(
            (col("jk") * lit(Int64(17))).alias("jk")
        )
        var sparse_right = right.with_columns(
            ((col("jk") + lit(Int64(rows // 4))) * lit(Int64(17))).alias("jk")
        )
        timed(
            sparse_left,
            sparse_right,
            "left_sparse",
            ["jk"],
            "left",
            repetitions,
            "r",
        )

    if only == "" or only == "inner_duplicate":
        var duplicates = right.with_columns(
            (col("jk") // lit(Int64(2))).alias("jk")
        )
        timed(
            left,
            duplicates,
            "inner_duplicate",
            ["jk"],
            "inner",
            repetitions,
            "r",
        )
