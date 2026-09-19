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
