"""DuckDB and Polars sides of the operator parity harness.

`python3 operator_engines.py ENGINE CELL_JSON REPS ANSWER_PATH` loads the cell's
Parquet tables into memory (untimed: DuckDB tables via CREATE TABLE AS,
Polars DataFrames), runs the cell's query once to warm up, then REPS times,
and prints one JSON line per run:

    {"rep": i, "wall_ms": ..., "nodes": [{"name", "busy_ms", "rows"}, ...]}

- DuckDB: `PRAGMA enable_profiling='json'` with profiling_mode detailed;
  each node is an operator of the physical plan with its `timing` (summed
  over threads) and cardinality. wall_ms is the profiler's query latency.
- Polars: POLARS_LOG_METRICS=1 (Polars 2.0 removed LazyFrame.profile, and
  this is its streaming engine's per-node account): each node's total poll
  plus state-update time, summed over threads, and rows sent. wall_ms is
  the time around `collect`.

The last run's result is written to ANSWER_PATH as Parquet afterwards.
"""

import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

THREADS = int(os.environ.get("BENCH_THREADS", "8"))


# --- DuckDB -----------------------------------------------------------------


def _order_sql(names, desc):
    return ", ".join(
        f"{n} {'DESC' if d else 'ASC'} NULLS LAST" for n, d in zip(names, desc)
    )


AGG_SQL = {
    "len": "count(*)",
    # Polars sums an all-null group to 0; SQL gives NULL.
    "sum": "coalesce(sum({c}), 0)",
    "mean": "avg({c})",
    "min": "min({c})",
    "max": "max({c})",
    "first": "first({c})",
    "last": "last({c})",
    "n_unique": "count(DISTINCT {c})",
}

JOIN_SQL = {
    "inner": "JOIN",
    "left": "LEFT JOIN",
    "right": "RIGHT JOIN",
    "full": "FULL OUTER JOIN",
    "semi": "SEMI JOIN",
    "anti": "ANTI JOIN",
}


def window_sql(q):
    part = q["part"]
    over = []
    if part:
        over.append("PARTITION BY " + ", ".join(part))
    if q["order"]:
        over.append("ORDER BY " + _order_sql(q["order"], [False] * len(q["order"])))
    frame = {
        "cum_sum": " ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
        "rolling": " ROWS BETWEEN 10 PRECEDING AND CURRENT ROW",
    }.get(q["fn"], "")
    spec = " ".join(over) + frame
    fn = {
        "row_number": "row_number()",
        "rank": "rank()",
        "cum_sum": f"sum({q['value']})",
        "sum": f"sum({q['value']})",
        "rolling": f"sum({q['value']})",
    }[q["fn"]]
    return f"{fn} OVER ({spec})"


def duckdb_sql(q):
    op = q["op"]
    if op == "join":
        on = " AND ".join(
            f"{l} = {r}" for l, r in zip(q["lk"], q["rk"])
        )
        return (
            f"SELECT {', '.join(q['select'])} FROM {q['left']} "
            f"{JOIN_SQL[q['how']]} {q['right']} ON {on}"
        )
    if op == "agg":
        items = list(q["keys"]) + [
            AGG_SQL[fn].format(c=c) + f" AS {alias}"
            for fn, c, alias in q["aggs"]
        ]
        return (
            f"SELECT {', '.join(items)} FROM {q['t']} "
            f"GROUP BY {', '.join(q['keys'])}"
        )
    if op == "filter_sum":
        return (
            f"SELECT sum({q['a']} * {q['b']}) AS s FROM {q['t']} "
            f"WHERE {q['f']} < {q['lt']}"
        )
    if op == "sort":
        sql = (
            f"SELECT {', '.join(q['select'])} FROM {q['t']} "
            f"ORDER BY {_order_sql(q['by'], q['desc'])}"
        )
        if q["limit"]:
            sql += f" LIMIT {q['limit']}"
        return sql
    if op == "window":
        items = [c if c != "w" else window_sql(q) + " AS w" for c in q["select"]]
        return f"SELECT {', '.join(items)} FROM {q['t']}"
    if op == "sched":
        sql = f"SELECT {', '.join(q['select'])} FROM {q['t']}"
        for b, l, r in zip(q["builds"], q["lk"], q["rk"]):
            sql += f" JOIN {b} ON {l} = {r}"
        return sql
    raise ValueError(op)


def duckdb_nodes(profile):
    """(type, timing ms, rows) for every operator in DuckDB's JSON profile."""
    out = []

    def walk(node):
        out.append(
            {
                "name": node.get("type") or node.get("operator_type", ""),
                "busy_ms": float(node.get("timing", node.get("operator_timing", 0.0)))
                * 1000.0,
                "rows": int(
                    node.get("intermediate_rows", node.get("operator_cardinality", 0))
                ),
            }
        )
        for child in node.get("children", []):
            walk(child)

    for root in profile.get("operator", profile.get("children", [])):
        walk(root)
    return out


def run_duckdb(cell, paths, reps, answer):
    import duckdb
    import pyarrow.parquet as pq

    con = duckdb.connect()
    con.execute(f"SET threads={THREADS}")
    for name, path in paths.items():
        con.execute(
            f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{path}')"
        )
    sql = duckdb_sql(cell["query"])
    profile_path = Path(tempfile.mkdtemp()) / "profile.json"
    con.execute("PRAGMA enable_profiling='json'")
    con.execute("PRAGMA profiling_mode='detailed'")
    con.execute(f"PRAGMA profiling_output='{profile_path}'")
    # The relation API writes the profile when each query ends; results
    # are collected as Arrow by DuckDB's own (parallel) result collector.
    result = con.sql(sql).to_arrow_table()
    for rep in range(reps):
        result = None
        result = con.sql(sql).to_arrow_table()
        profile = json.loads(profile_path.read_text())
        query = profile.get("query", profile)
        wall = float(query.get("total_time", query.get("latency", 0.0))) * 1000
        emit(rep, wall, duckdb_nodes(profile))
    con.execute("PRAGMA disable_profiling")
    pq.write_table(result, answer, store_decimal_as_integer=True)
    print(json.dumps({"rows": result.num_rows}), flush=True)


# --- Polars -----------------------------------------------------------------


def polars_plan(q, frames):
    import polars as pl

    op = q["op"]
    if op == "join":
        how = q["how"]
        out = frames[q["left"]].join(
            frames[q["right"]],
            left_on=q["lk"],
            right_on=q["rk"],
            how=how,
            coalesce=False if how == "full" else None,
        )
        return out.select(q["select"])
    if op == "agg":
        exprs = []
        for fn, c, alias in q["aggs"]:
            if fn == "len":
                e = pl.len()
            else:
                e = getattr(pl.col(c), fn)()
            exprs.append(e.alias(alias))
        return frames[q["t"]].group_by(q["keys"]).agg(exprs)
    if op == "filter_sum":
        return (
            frames[q["t"]]
            .filter(pl.col(q["f"]) < q["lt"])
            .select((pl.col(q["a"]) * pl.col(q["b"])).sum().alias("s"))
        )
    if op == "sort":
        out = frames[q["t"]].select(q["select"]).sort(
            q["by"], descending=q["desc"], nulls_last=True
        )
        return out.head(q["limit"]) if q["limit"] else out
    if op == "window":
        part, order, fn = q["part"], q["order"], q["fn"]
        v = pl.col(q["value"])
        if fn == "sum":
            e = v.sum()
            if part:
                e = e.over(part)
        elif fn == "rank":
            target = pl.col(order[0]) if len(order) == 1 else pl.struct(order)
            e = target.rank("min")
            if part:
                e = e.over(part)
        else:
            e = {
                "row_number": pl.int_range(pl.len(), dtype=pl.Int64) + 1,
                "cum_sum": v.cum_sum(),
                "rolling": v.rolling_sum(11, min_samples=1),
            }[fn]
            e = e.over(part or None, order_by=order)
        return frames[q["t"]].with_columns(e.alias("w")).select(q["select"])
    if op == "sched":
        out = frames[q["t"]]
        for b, l, r in zip(q["builds"], q["lk"], q["rk"]):
            out = out.join(frames[b], left_on=l, right_on=r)
        return out.select(q["select"])
    raise ValueError(op)


_DURATION = re.compile(r"([0-9.]+)(ns|µs|us|ms|s)$")
_SCALE = {"ns": 1e-6, "µs": 1e-3, "us": 1e-3, "ms": 1.0, "s": 1000.0}


def _ms(text):
    match = _DURATION.match(text.strip())
    if not match:
        raise ValueError(f"unreadable duration {text!r}")
    return float(match.group(1)) * _SCALE[match.group(2)]


_NODE = re.compile(r"^([A-Za-z0-9_\-]+): tot\(([^)]*)\).*sent\(row=(\d+)")


def polars_nodes(stderr):
    """Per-run node lists from POLARS_LOG_METRICS output; runs are
    separated by the `polars-rep` markers run_polars prints."""
    runs, current = {}, None
    for line in stderr.splitlines():
        if line.startswith("polars-rep "):
            tag = line.split()[1]
            current = int(tag) if tag.isdigit() else None
            if current is not None:
                runs[current] = []
            continue
        match = _NODE.match(line)
        if match and current is not None:
            runs[current].append(
                {
                    "name": match.group(1),
                    "busy_ms": _ms(match.group(2)),
                    "rows": int(match.group(3)),
                }
            )
    return runs


def run_polars(cell, paths, reps, answer):
    import polars as pl

    frames = {
        name: pl.read_parquet(path).lazy() for name, path in paths.items()
    }
    plan = polars_plan(cell["query"], frames)
    plan.collect()
    result = None
    for rep in range(reps):
        print(f"polars-rep {rep}", file=sys.stderr, flush=True)
        start = time.perf_counter()
        result = plan.collect()
        wall = (time.perf_counter() - start) * 1000
        sys.stderr.flush()
        print(json.dumps({"rep": rep, "wall_ms": wall}), flush=True)
    print("polars-rep end", file=sys.stderr, flush=True)
    result.write_parquet(answer)
    print(json.dumps({"rows": result.height}), flush=True)


def emit(rep, wall_ms, nodes):
    print(json.dumps({"rep": rep, "wall_ms": wall_ms, "nodes": nodes}), flush=True)


def main():
    engine, cell_path, reps, answer = sys.argv[1:5]
    cell = json.loads(Path(cell_path).read_text())
    paths = cell["paths"]
    try:
        if engine == "duckdb":
            run_duckdb(cell, paths, int(reps), answer)
        else:
            run_polars(cell, paths, int(reps), answer)
    except Exception as error:  # reported per cell, never fatal
        message = f"{type(error).__name__}: {error}".replace("\n", " ")
        print(json.dumps({"failed": message[:2000]}), flush=True)


if __name__ == "__main__":
    main()
