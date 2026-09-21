"""Head-to-head benchmark against Polars (a development-only comparison).

Generates deterministic CSV inputs once, builds benchmarks/bench_vs_polars.mojo
as an optimized binary, and runs the same workloads through both engines with
their thread counts pinned equal. Each engine reads the same bytes, warms up
once, and reports the best of REPETITIONS timed runs. Before times are
compared, each workload's row count and an order-insensitive column total
must agree between the engines, so the table cannot compare different work.

    pixi run -e oracle bench-polars                 # 100k and 1M rows
    pixi run -e oracle bench-polars --sizes 10000000
    pixi run -e oracle bench-polars --smoke         # tiny, for CI

The ratio column is dataframe_mojo time / Polars time: below 1 means this
engine is faster. Measure with an optimized build only; the Mojo binary is
built with `mojo build`, and `mojo run` would understate its speed.
"""
import argparse
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260918

WORKLOADS = [
    "csv_read",
    "arithmetic_chain",
    "nullable_compare",
    "filter",
    "global_sum",
    "grouped_low",
    "grouped_high",
    "grouped_skew",
    "grouped_str",
    "join_inner",
    "sort_multi",
]


def physical_cores() -> int:
    """Physical cores, matching Mojo's num_physical_cores() as closely as
    the platform allows, so both engines get the same thread count."""
    try:
        if platform.system() == "Linux":
            out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
            per_socket = sockets = None
            for line in out.splitlines():
                if line.startswith("Core(s) per socket:"):
                    per_socket = int(line.split(":")[1])
                elif line.startswith("Socket(s):"):
                    sockets = int(line.split(":")[1])
            if per_socket and sockets:
                return per_socket * sockets
        elif platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True).stdout
            return int(out.strip())
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def generate(data_dir: Path, rows: int) -> None:
    """Write left_ROWS.csv and right_ROWS.csv if they do not already exist."""
    left = data_dir / f"left_{rows}.csv"
    right = data_dir / f"right_{rows}.csv"
    if left.exists() and right.exists():
        return
    import polars as pl

    rng = random.Random(SEED + rows)
    high = max(rows // 10, 1)
    half = max(rows // 2, 1)
    strings = [f"region_{i:03d}" for i in range(100)]
    key_low = [rng.randrange(16) for _ in range(rows)]
    key_high = [rng.randrange(high) for _ in range(rows)]
    # Skewed keys send half the rows to group 0, as bench_suite does.
    key_skew = [0 if i % 2 == 0 else rng.randrange(1000) for i in range(rows)]
    key_str = [strings[rng.randrange(100)] for _ in range(rows)]
    jk = [rng.randrange(half) for _ in range(rows)]
    # x is 10% null (every tenth row), written as an empty field.
    x = [None if i % 10 == 3 else rng.randrange(10000) / 100 - 50 for i in range(rows)]
    y = [rng.randrange(1000) / 10 for _ in range(rows)]
    n = [rng.randrange(1000) - 500 for _ in range(rows)]
    pl.DataFrame(
        {
            "key_low": key_low,
            "key_high": key_high,
            "key_skew": key_skew,
            "key_str": key_str,
            "jk": jk,
            "x": x,
            "y": y,
            "n": n,
        },
        schema={
            "key_low": pl.Int64,
            "key_high": pl.Int64,
            "key_skew": pl.Int64,
            "key_str": pl.String,
            "jk": pl.Int64,
            "x": pl.Float64,
            "y": pl.Float64,
            "n": pl.Int64,
        },
    ).write_csv(left)
    pl.DataFrame(
        {"jk": list(range(half)), "r": [rng.randrange(1000) / 10 for _ in range(half)]},
        schema={"jk": pl.Int64, "r": pl.Float64},
    ).write_csv(right)


def build_runner() -> Path:
    out = ROOT / "build" / "bench_vs_polars"
    out.parent.mkdir(exist_ok=True)
    source = ROOT / "benchmarks" / "bench_vs_polars.mojo"
    # The binary bakes in the library, so any library change must rebuild it.
    # rglob, not glob: a future subpackage must not go unnoticed here, because
    # the failure is silently benchmarking code that is no longer the source.
    newest = max(p.stat().st_mtime for p in [source, *(ROOT / "dataframe").rglob("*.mojo")])
    if not out.exists() or out.stat().st_mtime < newest:
        subprocess.run(["mojo", "build", "-I", str(ROOT), str(source), "-o", str(out)], check=True)
    return out


def run_mojo(runner: Path, data_dir: Path, rows: int, reps: int, threads: int) -> dict:
    env = dict(os.environ, DATAFRAME_THREADS=str(threads))
    proc = subprocess.run(
        [str(runner), str(data_dir), str(rows), str(reps)],
        capture_output=True, text=True, env=env, check=True,
    )
    results = {}
    for line in proc.stdout.splitlines():
        if line.startswith("#"):
            continue
        workload, best_ns, height, value = line.split("\t")
        results[workload] = (int(best_ns) / 1e6, int(height), float(value))
    missing = [w for w in WORKLOADS if w not in results]
    if missing:
        raise SystemExit(f"mojo runner produced no result for: {missing}\n{proc.stderr}")
    return results


def best_of(fn, reps: int):
    fn()  # warm up
    best = float("inf")
    result = None
    for _ in range(reps):
        start = time.perf_counter_ns()
        result = fn()
        best = min(best, time.perf_counter_ns() - start)
    return best / 1e6, result


def run_polars(data_dir: Path, rows: int, reps: int) -> dict:
    import polars as pl

    left_path = data_dir / f"left_{rows}.csv"
    left_schema = {
        "key_low": pl.Int64, "key_high": pl.Int64, "key_skew": pl.Int64,
        "key_str": pl.String, "jk": pl.Int64, "x": pl.Float64, "y": pl.Float64, "n": pl.Int64,
    }
    right = pl.read_csv(data_dir / f"right_{rows}.csv", schema={"jk": pl.Int64, "r": pl.Float64})

    def total(frame: pl.DataFrame, name: str) -> float:
        value = frame[name].sum()
        return 0.0 if value is None else float(value)

    results = {}

    ms, left = best_of(lambda: pl.read_csv(left_path, schema=left_schema), reps)
    results["csv_read"] = (ms, left.height, total(left, "x"))

    arithmetic = ((pl.col("x") + 3.0) * (pl.col("y") - 2.0) / 4.0).alias("out")
    ms, out = best_of(lambda: left.with_columns(arithmetic), reps)
    results["arithmetic_chain"] = (ms, out.height, total(out, "out"))

    compare = (pl.col("x") > pl.col("y")).alias("out")
    ms, out = best_of(lambda: left.with_columns(compare), reps)
    results["nullable_compare"] = (ms, out.height, total(out, "out"))

    ms, out = best_of(lambda: left.filter(pl.col("x") > 0.0), reps)
    results["filter"] = (ms, out.height, total(out, "y"))

    ms, out = best_of(lambda: left.select(pl.col("x").sum()), reps)
    results["global_sum"] = (ms, 1, float(out.item()))

    for workload, key in [("grouped_low", "key_low"), ("grouped_high", "key_high"),
                          ("grouped_skew", "key_skew"), ("grouped_str", "key_str")]:
        aggs = [pl.col("x").sum().alias("s"), pl.col("n").count().alias("c")]
        ms, out = best_of(lambda: left.group_by(key).agg(aggs), reps)
        results[workload] = (ms, out.height, total(out, "s") + total(out, "c"))

    ms, out = best_of(lambda: left.join(right, on="jk", how="inner"), reps)
    results["join_inner"] = (ms, out.height, total(out, "r"))

    # dataframe_mojo's sort is always stable; Polars' default is not, which
    # is the faster of its two modes, so this favors Polars. nulls_last
    # matches dataframe_mojo's default.
    ms, out = best_of(lambda: left.sort(["key_low", "x"], nulls_last=True), reps)
    results["sort_multi"] = (ms, out.height, total(out.head(1000), "x"))
    return results


def check_agreement(rows: int, mojo: dict, polars: dict) -> None:
    for workload in WORKLOADS:
        _, mh, mv = mojo[workload]
        _, ph, pv = polars[workload]
        if mh != ph:
            raise SystemExit(f"{workload} @ {rows}: heights differ (mojo {mh}, polars {ph})")
        tolerance = 1e-9 * max(abs(mv), abs(pv), 1.0)
        if abs(mv - pv) > tolerance:
            raise SystemExit(f"{workload} @ {rows}: values differ (mojo {mv!r}, polars {pv!r})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="100000,1000000", help="comma-separated row counts")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=physical_cores())
    parser.add_argument("--data-dir", default=str(ROOT / "build" / "bench_polars"))
    parser.add_argument("--smoke", action="store_true", help="5,000 rows, one repetition")
    args = parser.parse_args()

    # Polars reads its thread cap at import, so it must be set before then.
    os.environ["POLARS_MAX_THREADS"] = str(args.threads)
    import polars as pl

    sizes = [5000] if args.smoke else [int(s) for s in args.sizes.split(",")]
    reps = 1 if args.smoke else args.reps
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    runner = build_runner()

    print(f"# polars={pl.__version__} threads={args.threads} reps={reps} "
          f"machine={platform.machine()} {platform.system()}")
    print("| workload | rows | dataframe_mojo ms | polars ms | mojo / polars |")
    print("|---|---|---|---|---|")
    for rows in sizes:
        generate(data_dir, rows)
        mojo = run_mojo(runner, data_dir, rows, reps, args.threads)
        polars = run_polars(data_dir, rows, reps)
        check_agreement(rows, mojo, polars)
        for workload in WORKLOADS:
            m, p = mojo[workload][0], polars[workload][0]
            ratio = m / p if p > 0 else float("inf")
            print(f"| {workload} | {rows:,} | {m:.2f} | {p:.2f} | {ratio:.1f}x |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
