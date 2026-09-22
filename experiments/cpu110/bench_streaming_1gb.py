#!/usr/bin/env python3
"""Benchmark the #109 CSV aggregate in Polars' eager and streaming engines.

Use one process per mode when collecting RSS, for example:

  /usr/bin/time -v pixi run -e oracle python experiments/cpu110/bench_streaming_1gb.py \\
    build/cpu110_streaming_1gb.csv --engine streaming --schema 10 --rss \\
    --expected 449890723.99999994

The corresponding dataframe_mojo binaries are:

  /tmp/df_streaming_experiment/bench_streaming_109       (8 columns)
  /tmp/df_streaming_experiment/bench_streaming_109_10col (10 columns)

They accept the same CSV and pipeline modes: eager, stream, or parallel.
"""

from __future__ import annotations

import argparse
import math
import resource
import time
from pathlib import Path


def schema_for(columns: int, pl):
    schema = {
        "key_low": pl.Int64,
        "key_high": pl.Int64,
        "key_skew": pl.Int64,
        "key_str": pl.String,
        "jk": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "n": pl.Int64,
    }
    if columns == 10:
        schema.update(extra_a=pl.Int64, extra_b=pl.Float64)
    return schema


def aggregate(frame, pl) -> float:
    result = (
        frame.filter(pl.col("x") > 0.0)
        .select(((pl.col("x") * 2.0).alias("double")), pl.col("y"))
        .select(pl.col("double").sum().alias("sum"))
    )
    if hasattr(result, "collect"):
        result = result.collect(engine="streaming")
    value = result.item(0, 0)
    return 0.0 if value is None else float(value)


def close_enough(actual: float, expected: float) -> bool:
    # Same documented allowance used by the Mojo streaming harness.  Parallel
    # reduction may reassociate Float64 additions, but this rejects data loss.
    return abs(actual - expected) <= 1e-6 + 1e-12 * max(abs(actual), abs(expected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--engine", choices=("eager", "streaming"), required=True)
    parser.add_argument("--schema", type=int, choices=(8, 10), required=True)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--rss", action="store_true", help="one un-warmed pass")
    parser.add_argument("--expected", type=float)
    args = parser.parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(args.csv)
    if args.reps < 1:
        raise ValueError("--reps must be positive")
    if args.rss and args.reps != 8:
        raise ValueError("--rss performs one pass; omit --reps")

    import polars as pl

    schema = schema_for(args.schema, pl)

    def run() -> float:
        if args.engine == "eager":
            return aggregate(pl.read_csv(args.csv, schema=schema), pl)
        return aggregate(pl.scan_csv(args.csv, schema=schema), pl)

    repetitions = 1 if args.rss else args.reps
    if not args.rss:
        warm = run()
        if args.expected is not None and not close_enough(warm, args.expected):
            raise AssertionError(f"warmup sum {warm!r} differs from {args.expected!r}")

    best_ns = None
    actual = 0.0
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        actual = run()
        elapsed = time.perf_counter_ns() - started
        best_ns = elapsed if best_ns is None else min(best_ns, elapsed)
    if args.expected is not None and not close_enough(actual, args.expected):
        raise AssertionError(f"sum {actual!r} differs from {args.expected!r}")

    print(f"engine={args.engine}")
    print(f"schema_columns={args.schema}")
    print(f"sum={actual:.17g}")
    print(f"best_ns={best_ns}")
    print(f"self_maxrss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")


if __name__ == "__main__":
    main()
