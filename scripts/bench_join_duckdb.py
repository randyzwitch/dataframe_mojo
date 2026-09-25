"""Compare join result materialization in Mojo, Polars, and DuckDB.

Inputs and derived key tables are prepared outside timing. DuckDB executes
CREATE TEMP TABLE AS SELECT for each join, which consumes every output row and
stores the result in its native format. This avoids timing Arrow export and
avoids timing only the first chunk returned by execute(). Each engine warms up
once; the best of the requested timed runs is reported. Row counts and numeric
totals are checked across engines before any case is printed.

    pixi run -e oracle bench-joins-duckdb --sizes 1000000,10000000 --reps 5 --threads 32
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

from bench_join_polars import CASES, build_runner, check, run_mojo, run_polars
from bench_polars import ROOT, generate, physical_cores


LEFT_COLUMNS = ["key_low", "key_high", "key_skew", "key_str", "jk", "x", "y", "n"]
BASIC_COLUMNS = LEFT_COLUMNS + ["r"]
RIGHT_COLUMNS = [name for name in LEFT_COLUMNS if name != "jk"] + ["jk", "r"]


def run_duckdb(data_dir: Path, rows: int, reps: int, threads: int) -> dict:
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET threads = {threads}")
    left_types = (
        "{'key_low': 'BIGINT', 'key_high': 'BIGINT', "
        "'key_skew': 'BIGINT', 'key_str': 'VARCHAR', "
        "'jk': 'BIGINT', 'x': 'DOUBLE', 'y': 'DOUBLE', 'n': 'BIGINT'}"
    )
    right_types = "{'jk': 'BIGINT', 'r': 'DOUBLE'}"
    con.execute(
        f"CREATE TABLE l AS SELECT * FROM read_csv(?, types={left_types})",
        [str(data_dir / f"left_{rows}.csv")],
    )
    con.execute(
        f"CREATE TABLE r AS SELECT * FROM read_csv(?, types={right_types})",
        [str(data_dir / f"right_{rows}.csv")],
    )
    results = {}

    def prepare(name: str, query: str) -> None:
        con.execute(f"CREATE TABLE {name} AS {query}")

    def drop(*names: str) -> None:
        for name in names:
            con.execute(f"DROP TABLE {name}")

    def time_case(name: str, query: str, columns: list[str]) -> None:
        statement = "CREATE TEMP TABLE bench_result AS " + query
        con.execute(statement)  # Warmup; input and key preparation are untimed.
        con.execute("DROP TABLE bench_result")
        best_ns = float("inf")
        for rep in range(reps):
            start = time.perf_counter_ns()
            con.execute(statement)
            best_ns = min(best_ns, time.perf_counter_ns() - start)
            if rep + 1 < reps:
                con.execute("DROP TABLE bench_result")

        actual = [
            row[1]
            for row in con.execute("PRAGMA table_info('bench_result')").fetchall()
        ]
        if actual != columns:
            raise RuntimeError(f"{name} columns differ: {actual} != {columns}")
        total_column = "x" if name in ("semi_unmatched", "anti_unmatched") else "r"
        height, total = con.execute(
            f"SELECT COUNT(*), SUM({total_column}) FROM bench_result"
        ).fetchone()
        results[name] = (
            best_ns / 1e6,
            height,
            0.0 if total is None else float(total),
        )
        con.execute("DROP TABLE bench_result")

    time_case("inner_dense", "SELECT l.*, r.r FROM l JOIN r USING (jk)", BASIC_COLUMNS)
    time_case("lazy_narrow", "SELECT l.x, r.r FROM l JOIN r USING (jk)", ["x", "r"])

    prepare("ls", "SELECT *, jk * 17 AS key FROM l")
    prepare("rs", "SELECT *, jk * 17 AS key FROM r")
    time_case(
        "inner_sparse",
        "SELECT ls.*, rs.jk AS jk_right, rs.r FROM ls JOIN rs USING (key)",
        LEFT_COLUMNS + ["key", "jk_right", "r"],
    )
    drop("ls", "rs")

    prepare("lt", "SELECT *, CAST(jk AS VARCHAR) AS key FROM l")
    prepare("rt", "SELECT *, CAST(jk AS VARCHAR) AS key FROM r")
    time_case(
        "inner_string",
        "SELECT lt.*, rt.jk AS jk_right, rt.r FROM lt JOIN rt USING (key)",
        LEFT_COLUMNS + ["key", "jk_right", "r"],
    )
    drop("lt", "rt")

    prepare("lm", "SELECT *, jk // 4 AS a, jk % 4 AS b FROM l")
    prepare("rm", "SELECT *, jk // 4 AS a, jk % 4 AS b FROM r")
    time_case(
        "inner_multi",
        "SELECT lm.*, rm.jk AS jk_right, rm.r FROM lm JOIN rm USING (a, b)",
        LEFT_COLUMNS + ["a", "b", "jk_right", "r"],
    )
    drop("lm", "rm")

    shift = rows // 4
    prepare("shifted", f"SELECT jk + {shift} AS jk, r FROM r")
    time_case(
        "left_unmatched",
        "SELECT l.*, shifted.r FROM l LEFT JOIN shifted USING (jk)",
        BASIC_COLUMNS,
    )
    right_projection = (
        "SELECT l.key_low, l.key_high, l.key_skew, l.key_str, "
        "l.x, l.y, l.n, shifted.jk, shifted.r FROM l"
    )
    time_case(
        "right_unmatched",
        right_projection + " RIGHT JOIN shifted USING (jk)",
        RIGHT_COLUMNS,
    )
    full_projection = (
        "SELECT l.* REPLACE (COALESCE(l.jk, shifted.jk) AS jk), "
        "shifted.r FROM l"
    )
    time_case(
        "full_unmatched",
        full_projection + " FULL OUTER JOIN shifted USING (jk)",
        BASIC_COLUMNS,
    )
    time_case(
        "semi_unmatched",
        "SELECT l.* FROM l SEMI JOIN shifted USING (jk)",
        LEFT_COLUMNS,
    )
    time_case(
        "anti_unmatched",
        "SELECT l.* FROM l ANTI JOIN shifted USING (jk)",
        LEFT_COLUMNS,
    )
    drop("shifted")

    prepare("sl", "SELECT * REPLACE (jk * 17 AS jk) FROM l")
    prepare("sr", f"SELECT (jk + {shift}) * 17 AS jk, r FROM r")
    time_case(
        "left_sparse",
        "SELECT sl.*, sr.r FROM sl LEFT JOIN sr USING (jk)",
        BASIC_COLUMNS,
    )
    sparse_right_projection = (
        "SELECT sl.key_low, sl.key_high, sl.key_skew, sl.key_str, "
        "sl.x, sl.y, sl.n, sr.jk, sr.r FROM sl"
    )
    time_case(
        "right_sparse",
        sparse_right_projection + " RIGHT JOIN sr USING (jk)",
        RIGHT_COLUMNS,
    )
    drop("sl", "sr")

    prepare("duplicates", "SELECT jk // 2 AS jk, r FROM r")
    time_case(
        "inner_duplicate",
        "SELECT l.*, duplicates.r FROM l JOIN duplicates USING (jk)",
        BASIC_COLUMNS,
    )
    drop("duplicates")
    con.close()
    if set(results) != set(CASES):
        raise RuntimeError(f"DuckDB benchmark missing {set(CASES) - set(results)}")
    return results


def main() -> int:
    import duckdb
    import polars as pl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000000,10000000")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=physical_cores())
    parser.add_argument("--data-dir", type=Path, default=ROOT / "build" / "bench_polars")
    parser.add_argument("--runner", type=Path)
    args = parser.parse_args()
    if args.reps < 1 or args.threads < 1:
        parser.error("reps and threads must be positive")
    sizes = [int(value) for value in args.sizes.split(",")]
    if any(rows < 1_000_000 for rows in sizes):
        parser.error("join comparison requires at least 1,000,000 rows")

    os.environ["POLARS_MAX_THREADS"] = str(args.threads)
    runner = args.runner.resolve() if args.runner else build_runner()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"# duckdb={duckdb.__version__} polars={pl.__version__} "
        f"threads={args.threads} reps={args.reps} "
        f"machine={platform.machine()} {platform.system()} "
        "DuckDB=native temporary table",
        flush=True,
    )
    print(
        "| join case | input rows | Mojo ms | Polars ms | DuckDB ms | "
        "Mojo / DuckDB |",
        flush=True,
    )
    print("|---|---:|---:|---:|---:|---:|", flush=True)
    for rows in sizes:
        generate(args.data_dir, rows)
        mojo = run_mojo(runner, args.data_dir, rows, args.reps, args.threads)
        polars = run_polars(args.data_dir, rows, args.reps)
        duckdb_results = run_duckdb(args.data_dir, rows, args.reps, args.threads)
        check(rows, mojo, polars)
        check(rows, mojo, duckdb_results, "DuckDB")
        for case in CASES:
            m, p, d = mojo[case][0], polars[case][0], duckdb_results[case][0]
            print(
                f"| {case} | {rows:,} | {m:.2f} | {p:.2f} | {d:.2f} | "
                f"{m/d:.2f}x |",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
