#!/usr/bin/env python3
"""Compute medians from raw CSV-port benchmark samples; never discard samples."""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: summarize.py RAW_SAMPLE.csv [RAW_SAMPLE.csv ...]")
    print("file,engine,scenario,workers,samples,median_read_ns")
    for name in sys.argv[1:]:
        with Path(name).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"{name}: no samples")
        keys = {(row["engine"], row["scenario"], row["workers"]) for row in rows}
        if len(keys) != 1:
            raise ValueError(f"{name}: mixed benchmark identities")
        engine, scenario, workers = keys.pop()
        values = [int(row["read_ns"]) for row in rows]
        print(
            f"{name},{engine},{scenario},{workers},{len(values)},"
            f"{statistics.median(values)}"
        )


if __name__ == "__main__":
    main()
