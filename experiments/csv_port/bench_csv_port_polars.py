#!/usr/bin/env python3
"""Emit raw Polars 1.44.2 CSV-ingestion timing samples for one input file."""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def schema(pl):
    return {
        "id": pl.Int64,
        "value": pl.Float64,
        "active": pl.Boolean,
        "label": pl.String,
    }


def columns_for(scenario: str) -> list[str] | None:
    if scenario == "full":
        return None
    if scenario == "projected":
        return ["id", "label"]
    raise ValueError("scenario must be 'full' or 'projected'")


def total_nulls(frame) -> int:
    return sum(frame.get_column(name).null_count() for name in frame.columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("iterations", type=int)
    parser.add_argument("scenario", choices=("full", "projected"))
    args = parser.parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(args.csv)
    if args.iterations < 1:
        raise ValueError("iterations must be positive")

    import polars as pl

    if pl.__version__ != "1.44.2":
        raise RuntimeError(f"Expected Polars 1.44.2, got {pl.__version__}")

    use_columns = columns_for(args.scenario)

    def read():
        return pl.read_csv(args.csv, schema=schema(pl), columns=use_columns)

    reference = read()
    expected_shape = reference.shape
    expected_nulls = total_nulls(reference)
    warm = read()
    del warm

    workers = pl.thread_pool_size()
    print("engine,scenario,workers,iteration,read_ns,rows,width,nulls")
    for iteration in range(args.iterations):
        started = time.perf_counter_ns()
        frame = read()
        elapsed = time.perf_counter_ns() - started
        if (
            frame.shape != expected_shape
            or total_nulls(frame) != expected_nulls
            or not frame.equals(reference)
        ):
            raise AssertionError("CSV result changed from the untimed reference")
        print(
            f"polars,{args.scenario},{workers},{iteration},{elapsed},"
            f"{frame.height},{frame.width},{total_nulls(frame)}"
        )
        # Keep destruction of the preceding output outside the next read.
        del frame


if __name__ == "__main__":
    main()
