"""This library's side of the operator parity harness.
See benchmarks/tools/bench_operators.py. One process runs one cell:

    runner REPS ANSWER_PATH key=value ... @table=path ...

It loads every `@table` Parquet file into memory (untimed), builds the
cell's lazy plan from the `key=value` arguments, runs it once to warm up,
then runs it REPS times through `LazyFrame.profile` and prints, per run,

    wall<TAB>REP<TAB>NANOSECONDS
    op<TAB>REP<TAB>NODE<TAB>OPERATOR<TAB>EXECUTOR<TAB>INPUT_ROWS
        <TAB>BUILD_ROWS<TAB>OUTPUT_ROWS<TAB>BUSY_MS<TAB>WALL_MS

one `op` line per executed plan node (the execution report, #439). The
last run's result is written to ANSWER_PATH as Parquet after its timer
stops. Errors print `failed<TAB>MESSAGE` (or `unsupported<TAB>...` for an
error that starts with "unsupported:").

Plans, by `op=`:

- join: `left`, `right` tables joined on `lk`/`rk` with `how`, then
  `select` (comma-separated output columns).
- agg: `t` grouped by `keys`, with `aggs` as `fn:column:alias` items
  separated by `;` (fn: len, sum, mean, min, max, first, last, n_unique).
- filter_sum: `t` filtered by `f < lt`, then sum(`a` * `b`).
- sort: `t` sorted by `by` with `desc` flags (0/1), nulls last, keeping
  `select`; `limit` > 0 keeps that many rows.
- window: `t` with one window column `w`: `fn` (row_number, rank,
  cum_sum, sum, rolling) of `value`, partitioned by `part` (empty: one
  partition) and ordered by `order`, then `select`.
- sched: `t` probed against `builds` (comma-separated tables), joining
  `lk` (one key per build, comma-separated) with `rk`, then `select`.
"""
from std.collections import Dict
from std.sys import argv
from std.time import monotonic

from dataframe.row_expressions import len as row_count
from dataframe import (
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    as_struct,
    col,
    int_range,
    lit,
    read_parquet,
    write_parquet,
)


def names(text: String) -> List[String]:
    var out = List[String]()
    if text == "":
        return out^
    for part in text.split(","):
        out.append(String(part))
    return out^


def aggregate(kind: String, name: String, alias: String) raises -> Expr:
    var c = col(name)
    if kind == "len":
        return row_count().alias(alias)
    if kind == "sum":
        return c.sum().alias(alias)
    if kind == "mean":
        return c.mean().alias(alias)
    if kind == "min":
        return c.min().alias(alias)
    if kind == "max":
        return c.max().alias(alias)
    if kind == "first":
        return c.first().alias(alias)
    if kind == "last":
        return c.last().alias(alias)
    if kind == "n_unique":
        return c.n_unique().alias(alias)
    raise Error("unknown aggregate " + kind)


def window(
    kind: String, value: String, part: List[String], order: List[String]
) raises -> Expr:
    """The window column, matching the DuckDB SQL in bench_operators.py:
    ROWS frames over unique order keys, so every engine's answer is
    determined."""
    var e: Expr
    if kind == "row_number":
        e = int_range(row_count()) + lit(Int64(1))
    elif kind == "rank":
        var fields = List[Expr]()
        for name in order:
            fields.append(col(name))
        e = (col(order[0]) if len(order) == 1 else as_struct(fields)).rank(
            "min"
        )
        if len(part) == 0:
            return e.alias("w")
        return e.over(part).alias("w")
    elif kind == "cum_sum":
        e = col(value).cum_sum()
    elif kind == "sum":
        e = col(value).sum()
        if len(part) == 0:
            return e.alias("w")
        return e.over(part).alias("w")
    elif kind == "rolling":
        e = col(value).rolling_sum(11, min_samples=1)
    else:
        raise Error("unknown window function " + kind)
    if len(part) == 0:
        return e.over(order_by=order).alias("w")
    return e.over(part, order).alias("w")


def plan(
    spec: Dict[String, String], tables: Dict[String, DataFrame]
) raises -> LazyFrame:
    var op = spec["op"]
    if op == "join":
        return (
            tables[spec["left"]]
            .lazy()
            .join(
                tables[spec["right"]].lazy(),
                left_on=names(spec["lk"]),
                right_on=names(spec["rk"]),
                how=spec["how"],
                coalesce=spec["how"] != "full",
            )
            .select(names(spec["select"]))
        )
    if op == "agg":
        var exprs = List[Expr]()
        for item in spec["aggs"].split(";"):
            var parts = String(item).split(":")
            exprs.append(
                aggregate(String(parts[0]), String(parts[1]), String(parts[2]))
            )
        return tables[spec["t"]].lazy().group_by(names(spec["keys"])).agg(exprs)
    if op == "filter_sum":
        return (
            tables[spec["t"]]
            .lazy()
            .filter(col(spec["f"]) < lit(Int64(Int(spec["lt"]))))
            .select((col(spec["a"]) * col(spec["b"])).sum().alias("s"))
        )
    if op == "sort":
        var by = names(spec["by"])
        var desc = List[Bool]()
        for flag in names(spec["desc"]):
            desc.append(flag == "1")
        var out = (
            tables[spec["t"]]
            .lazy()
            .select(names(spec["select"]))
            .sort(
                by,
                descending=desc,
                nulls_last=List[Bool](length=len(by), fill=True),
            )
        )
        var limit = Int(spec["limit"])
        if limit > 0:
            return out.head(limit)
        return out^
    if op == "window":
        return (
            tables[spec["t"]]
            .lazy()
            .with_columns(
                window(
                    spec["fn"],
                    spec["value"],
                    names(spec["part"]),
                    names(spec["order"]),
                )
            )
            .select(names(spec["select"]))
        )
    if op == "sched":
        var out = tables[spec["t"]].lazy()
        var builds = names(spec["builds"])
        var lk = names(spec["lk"])
        var rk = names(spec["rk"])
        for i in range(len(builds)):
            out = out.join(
                tables[builds[i]].lazy(), left_on=[lk[i]], right_on=[rk[i]]
            )
        return out.select(names(spec["select"]))
    raise Error("unknown op " + op)


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error("usage: runner REPS ANSWER_PATH key=value ... @t=path ...")
    var reps = Int(String(args[1]))
    var answer = String(args[2])
    var spec = Dict[String, String]()
    var tables = Dict[String, DataFrame]()
    for i in range(3, len(args)):
        var arg = String(args[i])
        var parts = arg.split("=", 1)
        if len(parts) != 2:
            raise Error("expected key=value, found " + arg)
        var key = String(parts[0])
        if key.startswith("@"):
            tables[String(key[byte=1:])] = read_parquet(String(parts[1]))
        else:
            spec[key] = String(parts[1])
    try:
        var query = plan(spec, tables)
        var result = query.profile(engine="cpu")[0].copy()
        for rep in range(reps):
            var start = monotonic()
            var run = query.profile(engine="cpu")
            var elapsed = monotonic() - start
            _ = result^
            result = run[0].copy()
            var report = run[1].copy()
            print("wall\t" + String(rep) + "\t" + String(elapsed))
            for r in range(report.height()):
                var line = "op\t" + String(rep)
                for name in [
                    "node",
                    "operator",
                    "executor",
                    "input_rows",
                    "build_rows",
                    "output_rows",
                    "busy_ms",
                    "wall_ms",
                ]:
                    var cell = report.item(r, name)
                    if cell.dtype() == DataType.INT64:
                        line += "\t" + String(cell.int64())
                    elif cell.dtype() == DataType.FLOAT64:
                        line += "\t" + String(cell.float64())
                    else:
                        line += "\t" + cell.string().replace("\t", " ")
                print(line)
        write_parquet(result, answer)
        print("rows\t" + String(result.height()))
    except e:
        var message = String(e).replace("\n", " ")
        if message.startswith("unsupported:"):
            print("unsupported\t" + message)
        else:
            print("failed\t" + message)
