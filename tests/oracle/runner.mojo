"""Execute one differential-test case and write the result as CSV.

Usage: runner LEFT_CSV RIGHT_CSV OUTPUT_CSV SPEC...

SPEC is one operation (see scripts/oracle.py for the generator):
  filter COL OP VALUE          OP in gt, lt, ge, le, eq, ne
  arith OP A B                 OP in add, sub, mul, div; output column "out"
  agg FN COL                   group_by("k", maintain_order=True)
  sort COLS DESCS              comma-separated names and 0/1 flags
  join HOW                     left.join(right, "k", HOW)
  unique COLS                  keep="first", maintain_order=True
  cum_sum COL                  output column "out"
  cast COL DTYPE               non-strict
  stat CONTEXT FN COL          CONTEXT in global, group, over (see oracle.py)
  prep OP ...                  interpolate, cut, or qcut

Setting DATAFRAME_ORACLE_INJECT=1 drops the last result row, so the harness
can prove it detects a wrong answer.
"""
from std.os import getenv
from std.sys import argv

from dataframe import (
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Expr,
    col,
    corr,
    cov,
    lit,
    read_csv,
    write_csv,
)


def left_schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.string("k"),
            CsvField.int64("g"),
            CsvField.float64("x"),
            CsvField.float64("y"),
            CsvField.int64("n"),
            CsvField.bool("b"),
        ]
    )


def right_schema() raises -> CsvSchema:
    return CsvSchema([CsvField.string("k"), CsvField.int64("r")])


def literal(frame: DataFrame, name: String, text: String) raises -> Expr:
    var dtype = frame.column(name).dtype()
    if dtype == DataType.INT64:
        return lit(Int64(Int(text)))
    if dtype == DataType.FLOAT64:
        return lit(Float64(text))
    if dtype == DataType.BOOL:
        return lit(text == "true")
    return lit(text)


def split(text: String) -> List[String]:
    var out = List[String]()
    for part in text.split(","):
        out.append(String(part))
    return out^


def run(
    left: DataFrame, right: DataFrame, spec: List[String]
) raises -> DataFrame:
    var op = spec[0]
    if op == "prep":
        var operation = spec[1]
        var e: Expr
        if operation == "interpolate":
            e = col("x").interpolate(spec[3])
            if spec[2] == "group":
                e = e.over("k")
        elif operation == "cut":
            e = col("x").cut([1.0, 5.0], left_closed=spec[2] == "1")
        else:
            e = col("x").qcut([0.25, 0.5, 0.75], left_closed=spec[2] == "1")
        return left.with_columns(e.alias("out"))
    if op == "filter":
        var c = col(spec[1])
        var v = literal(left, spec[1], spec[3])
        var p: Expr
        if spec[2] == "gt":
            p = c > v
        elif spec[2] == "lt":
            p = c < v
        elif spec[2] == "ge":
            p = c >= v
        elif spec[2] == "le":
            p = c <= v
        elif spec[2] == "eq":
            p = c.eq(v)
        else:
            p = c.ne(v)
        return left.filter(p)
    if op == "decimal":
        var dtype = DataType.decimal(15, 2)
        var x = col("x").cast(dtype)
        var y = col("y").cast(dtype)
        var e: Expr
        if spec[1] == "cast":
            e = x.copy()
        elif spec[1] == "sum":
            e = x.sum()
        elif spec[1] == "mean":
            e = x.mean()
        elif spec[1] == "add":
            e = x + y
        elif spec[1] == "sub":
            e = x - y
        else:
            e = x * y
        return left.select(e.cast("string").alias("out"))
    if op == "arith":
        var a = col(spec[2])
        var b = col(spec[3])
        var e: Expr
        if spec[1] == "add":
            e = a + b
        elif spec[1] == "sub":
            e = a - b
        elif spec[1] == "mul":
            e = a * b
        else:
            e = a / b
        return left.with_columns(e.alias("out"))
    if op == "agg":
        var c = col(spec[2])
        var e: Expr
        if spec[1] == "sum":
            e = c.sum()
        elif spec[1] == "count":
            e = c.count()
        elif spec[1] == "min":
            e = c.min()
        elif spec[1] == "max":
            e = c.max()
        elif spec[1] == "mean":
            e = c.mean()
        elif spec[1] == "n_unique":
            e = c.n_unique()
        elif spec[1] == "first":
            e = c.first()
        else:
            e = c.last()
        return left.group_by("k", maintain_order=True).agg(e.alias("out"))
    if op == "sort":
        var names = split(spec[1])
        var flags = List[Bool]()
        for f in split(spec[2]):
            flags.append(f == "1")
        return left.sort(
            names,
            descending=flags,
            nulls_last=List[Bool](length=len(names), fill=True),
        )
    if op == "join":
        return left.join(right, "k", spec[1])
    if op == "unique":
        return left.unique(split(spec[1]), keep="first", maintain_order=True)
    if op == "cum_sum":
        return left.with_columns(col(spec[1]).cum_sum().alias("out"))
    if op == "stat":
        var context = spec[1]
        var name = spec[2]
        var column = spec[3]
        var frame = left.copy()
        var e: Expr
        if name.startswith("corr") or name.startswith("cov"):
            var names = split(column)
            var a = col(names[0])
            var b = col(names[1])
            frame = left.filter(a.is_not_null() & b.is_not_null())
            if name == "corr":
                e = corr(a, b)
            elif name == "corr_spearman":
                e = corr(a, b, method="spearman")
            else:
                e = cov(a, b, ddof=0 if name == "cov_ddof0" else 1)
        else:
            var c = col(column)
            if column == "xn":
                c = col("x") * (col("x") / col("x"))
            if name == "arg_min":
                e = c.arg_min()
            elif name == "arg_max":
                e = c.arg_max()
            elif name == "mode":
                e = c.mode()
            elif name == "skew":
                e = c.skew()
            elif name == "skew_unbiased":
                e = c.skew(bias=False)
            elif name == "kurtosis":
                e = c.kurtosis()
            elif name == "kurtosis_unbiased":
                e = c.kurtosis(bias=False)
            else:
                e = c.kurtosis(fisher=False)
        e = e.alias("out")
        if context == "global":
            return frame.select(e)
        if context == "over":
            return frame.with_columns(e.over("k"))
        var grouped = frame.group_by("k", maintain_order=True).agg(e)
        return grouped.explode("out") if name == "mode" else grouped^
    if op == "describe":
        var frame = left.with_columns(
            (col("x") * (col("x") / col("x"))).alias("xn")
        ).select(split(spec[1]))
        var qs = List[Float64]()
        if spec[2] != "none":
            for part in split(spec[2]):
                qs.append(Float64(part))
        return frame.describe(percentiles=qs)
    if op == "value_counts":
        var context = spec[1]
        var column = spec[2]
        var e = (
            col(column)
            .value_counts(sort=spec[3] == "1", normalize=spec[4] == "1")
            .alias("out")
        )
        if context == "global":
            return left.select_exprs([e^]).unnest("out")
        return (
            left.group_by("k", maintain_order=True)
            .agg([e^])
            .explode("out")
            .unnest("out")
        )
    if op == "cast":
        return left.with_columns(
            col(spec[1]).cast(spec[2], strict=False).alias("out")
        )
    raise Error("unknown oracle operation: " + op)


def main() raises:
    var args = argv()
    var spec = List[String]()
    for i in range(4, len(args)):
        spec.append(String(args[i]))
    var left = read_csv(String(args[1]), left_schema())
    var right = read_csv(String(args[2]), right_schema())
    var result = run(left, right, spec)
    if getenv("DATAFRAME_ORACLE_INJECT", "0") == "1" and result.height() > 0:
        result = result.head(-1)
    write_csv(result, String(args[3]))
