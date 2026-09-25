"""Compare representative joins with Polars at one million rows or more.

Inputs are constructed once outside timing. The two engines run the same
cardinality, key type, join mode, and projected-output cases. Each result is
checked by row count and a numeric column total before times are compared.

    pixi run -e oracle bench-joins-polars --sizes 1000000
    pixi run -e oracle bench-joins-polars --sizes 10000000 --threads 32
"""
import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

from bench_polars import ROOT, best_of, generate, physical_cores

CASES = [
    "inner_dense", "lazy_narrow", "inner_sparse", "inner_string",
    "inner_multi", "left_unmatched", "right_unmatched", "full_unmatched",
    "semi_unmatched", "anti_unmatched", "left_sparse", "right_sparse", "inner_duplicate",
]


def build_runner() -> Path:
    source = ROOT / "benchmarks" / "bench_join_matrix.mojo"
    runner = ROOT / "build" / "bench_join_matrix"
    newest = max(p.stat().st_mtime for p in [source, *(ROOT / "dataframe").rglob("*.mojo")])
    if not runner.exists() or runner.stat().st_mtime < newest:
        subprocess.run(
            ["mojo", "build", "-I", str(ROOT), str(source), "-o", str(runner)],
            check=True,
        )
    return runner


def run_mojo(runner: Path, data_dir: Path, rows: int, reps: int, threads: int) -> dict:
    env = dict(os.environ, DATAFRAME_THREADS=str(threads))
    output = subprocess.run(
        [str(runner), str(data_dir), str(rows), str(reps)],
        capture_output=True, text=True, env=env, check=True,
    ).stdout
    results = {}
    for line in output.splitlines():
        name, ns, height, value = line.split("\t")
        results[name] = (int(ns) / 1e6, int(height), float(value))
    missing = set(CASES) - results.keys()
    if missing:
        raise RuntimeError(f"Mojo benchmark missing cases {sorted(missing)}")
    return results


def run_polars(data_dir: Path, rows: int, reps: int) -> dict:
    import polars as pl

    left = pl.read_csv(
        data_dir / f"left_{rows}.csv",
        schema={
            "key_low": pl.Int64, "key_high": pl.Int64,
            "key_skew": pl.Int64, "key_str": pl.String,
            "jk": pl.Int64, "x": pl.Float64, "y": pl.Float64,
            "n": pl.Int64,
        },
    )
    right = pl.read_csv(
        data_dir / f"right_{rows}.csv",
        schema={"jk": pl.Int64, "r": pl.Float64},
    )
    results = {}

    def time_case(name, fn, column):
        ms, result = best_of(fn, reps)
        total = result[column].sum()
        results[name] = (ms, result.height, 0.0 if total is None else float(total))

    time_case("inner_dense", lambda: left.join(right, on="jk", how="inner"), "r")
    time_case(
        "lazy_narrow",
        lambda: left.lazy().join(right.lazy(), on="jk").select("x", "r").collect(),
        "r",
    )
    sparse_left = left.with_columns((pl.col("jk") * 17).alias("key"))
    sparse_right = right.with_columns((pl.col("jk") * 17).alias("key"))
    time_case(
        "inner_sparse",
        lambda: sparse_left.join(sparse_right, on="key", how="inner"),
        "r",
    )
    del sparse_left, sparse_right

    string_left = left.with_columns(pl.col("jk").cast(pl.String).alias("key"))
    string_right = right.with_columns(pl.col("jk").cast(pl.String).alias("key"))
    time_case(
        "inner_string",
        lambda: string_left.join(string_right, on="key", how="inner"),
        "r",
    )
    del string_left, string_right

    compound = [
        (pl.col("jk") // 4).alias("a"),
        (pl.col("jk") % 4).alias("b"),
    ]
    multi_left = left.with_columns(compound)
    multi_right = right.with_columns(compound)
    time_case(
        "inner_multi",
        lambda: multi_left.join(multi_right, on=["a", "b"], how="inner"),
        "r",
    )
    del multi_left, multi_right

    shifted = right.with_columns((pl.col("jk") + rows // 4).alias("jk"))
    for how in ("left", "right", "full", "semi", "anti"):
        column = "x" if how in ("semi", "anti") else "r"
        time_case(
            f"{how}_unmatched",
            lambda how=how: left.join(shifted, on="jk", how=how, coalesce=True),
            column,
        )
    del shifted

    sparse_left = left.with_columns((pl.col("jk") * 17).alias("jk"))
    sparse_shifted = right.with_columns(
        ((pl.col("jk") + rows // 4) * 17).alias("jk")
    )
    time_case(
        "left_sparse",
        lambda: sparse_left.join(sparse_shifted, on="jk", how="left"),
        "r",
    )
    time_case(
        "right_sparse",
        lambda: sparse_left.join(sparse_shifted, on="jk", how="right", coalesce=True),
        "r",
    )
    del sparse_left, sparse_shifted

    duplicate_right = right.with_columns((pl.col("jk") // 2).alias("jk"))
    time_case(
        "inner_duplicate",
        lambda: left.join(duplicate_right, on="jk", how="inner"),
        "r",
    )
    return results


def check(rows: int, mojo: dict, polars: dict, other_name: str = "Polars") -> None:
    for case in CASES:
        m_time, m_height, m_total = mojo[case]
        p_time, p_height, p_total = polars[case]
        if m_height != p_height:
            raise RuntimeError(
                f"{case} @ {rows}: row count differs: Mojo {m_height}, "
                f"{other_name} {p_height}"
            )
        tolerance = 1e-8 * max(abs(m_total), abs(p_total), 1.0)
        if abs(m_total - p_total) > tolerance:
            raise RuntimeError(
                f"{case} @ {rows}: total differs: Mojo {m_total}, "
                f"{other_name} {p_total}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000000,10000000")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=physical_cores())
    parser.add_argument("--data-dir", type=Path, default=ROOT / "build" / "bench_polars")
    parser.add_argument("--runner", type=Path)
    args = parser.parse_args()
    os.environ["POLARS_MAX_THREADS"] = str(args.threads)
    import polars as pl

    runner = args.runner.resolve() if args.runner else build_runner()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"# polars={pl.__version__} threads={args.threads} reps={args.reps} "
        f"machine={platform.machine()} {platform.system()}",
        flush=True,
    )
    print("| join case | input rows | Mojo ms | Polars ms | Mojo / Polars |", flush=True)
    print("|---|---:|---:|---:|---:|", flush=True)
    for rows in (int(s) for s in args.sizes.split(",")):
        if rows < 1_000_000:
            raise ValueError("join comparison requires at least 1,000,000 rows")
        generate(args.data_dir, rows)
        mojo = run_mojo(runner, args.data_dir, rows, args.reps, args.threads)
        polars = run_polars(args.data_dir, rows, args.reps)
        check(rows, mojo, polars)
        for case in CASES:
            m, p = mojo[case][0], polars[case][0]
            print(
                f"| {case} | {rows:,} | {m:.2f} | {p:.2f} | {m/p:.2f}x |",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
