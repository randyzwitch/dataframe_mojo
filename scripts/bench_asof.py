"""Issue #224: 10M-left/1M-right as-of join versus Polars.

Development mechanism measurement, not an external-suite headline. Both
engines read the same Parquet inputs before timing, use the same thread
setting, and materialize complete outputs. Compare every output cell.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def generate(directory, left_rows, right_rows, seed, variant):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    for side, size, step in (
        ("left", left_rows, 20),
        ("right", right_rows, 200),
    ):
        keys = np.cumsum(rng.integers(0, step, size, dtype=np.int64))
        mask = rng.random(size) < 0.05 if variant == "nulls" else None
        columns = {"k": pa.array(keys, mask=mask)}
        if variant == "grouped":
            columns["g"] = pa.array(rng.integers(0, 64, size, dtype=np.int64))
        if side == "right":
            columns["quote"] = pa.array(np.arange(size, dtype=np.int64))
        pq.write_table(
            pa.table(columns),
            directory / f"{side}.parquet",
            row_group_size=1_000_000,
        )


def polars_run(path, strategy, variant, reps):
    import polars as pl

    left = pl.read_parquet(path / "left.parquet")
    right = pl.read_parquet(path / "right.parquet")
    if variant == "nulls":
        assert left["k"].drop_nulls().is_sorted()
        assert right["k"].drop_nulls().is_sorted()
    values = []
    for i in range(reps + 1):
        start = time.perf_counter_ns()
        result = left.join_asof(
            right,
            on="k",
            by="g" if variant == "grouped" else None,
            strategy=strategy,
            check_sortedness=variant != "nulls",
        )
        elapsed = time.perf_counter_ns() - start
        if i:
            values.append(elapsed)
    result.write_parquet(path / "polars-output.parquet")
    print(json.dumps({"ns": values, "polars": pl.__version__}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=int, default=10_000_000)
    parser.add_argument("--right", type=int, default=1_000_000)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=224)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/asof-benchmark"
    )
    parser.add_argument(
        "--polars-worker", nargs=3, metavar=("PATH", "STRATEGY", "VARIANT")
    )
    args = parser.parse_args()
    if args.polars_worker:
        path, strategy, variant = args.polars_worker
        polars_run(Path(path), strategy, variant, args.reps)
        return
    if min(args.left, args.right, args.reps, args.threads) <= 0:
        parser.error("sizes, repetitions and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    executable = args.output / "bench-asof"
    subprocess.run(
        [
            "mojo",
            "build",
            "-I",
            ".",
            "benchmarks/bench_asof.mojo",
            "-o",
            str(executable),
        ],
        cwd=ROOT,
        check=True,
    )
    env = dict(
        os.environ,
        DATAFRAME_THREADS=str(args.threads),
        POLARS_MAX_THREADS=str(args.threads),
    )
    results = []
    report = dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        machine=platform.platform(),
        cpu_count=os.cpu_count(),
        load_average=os.getloadavg(),
        left_rows=args.left,
        right_rows=args.right,
        threads=args.threads,
        seed=args.seed,
        warmups=1,
        reps=args.reps,
        results=results,
    )
    for variant in ("base", "nulls", "grouped"):
        path = args.output / variant
        generate(path, args.left, args.right, args.seed, variant)
        for strategy in ("backward", "forward", "nearest"):
            mojo = subprocess.check_output(
                [str(executable), str(path), strategy, variant, str(args.reps)],
                cwd=ROOT,
                env=env,
                text=True,
            )
            times = [int(line) for line in mojo.splitlines()]
            assert len(times) == args.reps
            other = json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        __file__,
                        "--polars-worker",
                        str(path),
                        strategy,
                        variant,
                        "--reps",
                        str(args.reps),
                    ],
                    env=env,
                    text=True,
                )
            )
            import polars as pl
            from polars.testing import assert_frame_equal

            assert_frame_equal(
                pl.read_parquet(path / "mojo-output.parquet"),
                pl.read_parquet(path / "polars-output.parquet"),
                check_exact=True,
            )
            row = dict(
                variant=variant,
                strategy=strategy,
            check_sortedness=variant != "nulls",
                mojo_ns=times,
                polars_ns=other["ns"],
                polars_version=other["polars"],
                ratio=statistics.median(times) / statistics.median(other["ns"]),
                output_equal=True,
            )
            results.append(row)
            (args.output / "results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps(row), flush=True)
            # Full outputs are checked above; avoid retaining nine large copies.
            (path / "mojo-output.parquet").unlink()
            (path / "polars-output.parquet").unlink()


if __name__ == "__main__":
    main()
