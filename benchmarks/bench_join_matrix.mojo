"""Head-to-head join shapes; input construction and result checks are untimed.

Usage: bench_join_matrix DATA_DIR ROWS REPETITIONS [CASE] [--variant NAME]

CASE runs one workload for profiling without constructing the other key shapes.

NAME picks the key layout the inputs arrive in. Every Int64 case derives its
right key from a sorted `range(n)`, which the ordered-key and direct-address
join paths recognise, so `base` alone measures those paths:

- `base`: the files as generated; right keys sorted and consecutive.
- `shuffled`: the same right rows in random order. The join and its result
  are identical, but no ordered-key path applies.
- `wide`: every key mapped one-to-one onto a 2^40 range, so no bounded-range
  path applies either. `right_ROWS_wide.csv` carries the unmatched and
  duplicate keys precomputed (`jk_shift`, `jk_dup`), because deriving them
  from the mapped key would change which rows match.
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


def right_wide_schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("jk"),
            CsvField.int64("jk_shift"),
            CsvField.int64("jk_dup"),
            CsvField.float64("r"),
        ]
    )


def shifted_right(source: DataFrame, rows: Int, wide: Bool) raises -> DataFrame:
    """Right keys offset so a quarter of the left keys go unmatched."""
    if wide:
        return source.with_columns(col("jk_shift").alias("jk")).select(
            ["jk", "r"]
        )
    return source.with_columns((col("jk") + lit(Int64(rows // 4))).alias("jk"))


def duplicated_right(source: DataFrame, wide: Bool) raises -> DataFrame:
    """Right keys halved so each left key matches two right rows."""
    if wide:
        return source.with_columns(col("jk_dup").alias("jk")).select(
            ["jk", "r"]
        )
    return source.with_columns((col("jk") // lit(Int64(2))).alias("jk"))


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
    if len(args) < 4:
        raise Error(
            "usage: bench_join_matrix DATA_DIR ROWS REPETITIONS [CASE]"
            " [--variant NAME]"
        )
    var only = String()
    var variant = String("base")
    var i = 4
    while i < len(args):
        var arg = String(args[i])
        if arg == "--variant":
            if i + 1 >= len(args):
                raise Error("--variant needs a name")
            variant = String(args[i + 1])
            i += 2
        else:
            only = arg
            i += 1
    if variant != "base" and variant != "shuffled" and variant != "wide":
        raise Error("variant must be base, shuffled, or wide")
    var wide = variant == "wide"
    var rows = Int(String(args[2]))
    var repetitions = Int(String(args[3]))
    var base = String(args[1])
    var left_suffix = String("_wide") if wide else String()
    var right_suffix = String("_" + variant) if variant != "base" else String()
    var left = read_csv(
        base + "/left_" + String(rows) + left_suffix + ".csv", left_schema()
    )
    var source = read_csv(
        base + "/right_" + String(rows) + right_suffix + ".csv",
        right_wide_schema() if wide else right_schema(),
    )
    var right = source.select(["jk", "r"]) if wide else source.copy()

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
        var shifted = shifted_right(source, rows, wide)
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

    if only == "" or only == "left_sparse" or only == "right_sparse":
        var sparse_left = left.with_columns(
            (col("jk") * lit(Int64(17))).alias("jk")
        )
        var sparse_right = shifted_right(source, rows, wide).with_columns(
            (col("jk") * lit(Int64(17))).alias("jk")
        )
        if only == "" or only == "left_sparse":
            timed(
                sparse_left,
                sparse_right,
                "left_sparse",
                ["jk"],
                "left",
                repetitions,
                "r",
            )
        if only == "" or only == "right_sparse":
            timed(
                sparse_left,
                sparse_right,
                "right_sparse",
                ["jk"],
                "right",
                repetitions,
                "r",
            )

    if only == "" or only == "inner_duplicate":
        var duplicates = duplicated_right(source, wide)
        timed(
            left,
            duplicates,
            "inner_duplicate",
            ["jk"],
            "inner",
            repetitions,
            "r",
        )
