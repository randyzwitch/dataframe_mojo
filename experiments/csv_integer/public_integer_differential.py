#!/usr/bin/env python3
"""Differential-check public Mojo CSV integer conversion against Polars 1.44.2.

Build first:
  pixi run mojo build -I . experiments/csv_integer/public_integer_roundtrip.mojo \
    -o /tmp/csv-public-integer-roundtrip
Run:
  pixi run -e oracle python experiments/csv_integer/public_integer_differential.py \
    /tmp/csv-public-integer-roundtrip
"""
import csv
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import polars as pl
from polars.testing import assert_frame_equal

ROWS = 100_000
COLUMNS = {
    "i8": pl.Int8,
    "u8": pl.UInt8,
    "i16": pl.Int16,
    "u16": pl.UInt16,
    "i32": pl.Int32,
    "u32": pl.UInt32,
    "i64": pl.Int64,
    "u64": pl.UInt64,
}
LIMITS = {
    "i8": (-(1 << 7), (1 << 7) - 1),
    "u8": (0, (1 << 8) - 1),
    "i16": (-(1 << 15), (1 << 15) - 1),
    "u16": (0, (1 << 16) - 1),
    "i32": (-(1 << 31), (1 << 31) - 1),
    "u32": (0, (1 << 32) - 1),
    "i64": (-(1 << 63), (1 << 63) - 1),
    "u64": (0, (1 << 64) - 1),
}


def encode(value: int, row: int) -> str:
    """Exercise signs and leading-zero routes without changing the value."""
    if value < 0:
        return "-" + ("0" * (row % 7)) + str(-value)
    prefix = "+" if row % 11 == 0 else ""
    return prefix + ("0" * (row % 13)) + str(value)


def generate(path: Path) -> None:
    rng = random.Random(149)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in range(ROWS):
            fields = []
            for index, (name, (lo, hi)) in enumerate(LIMITS.items()):
                edge = (row + index * 3) % 29
                if edge == 0:
                    value = lo
                elif edge == 1:
                    value = hi
                elif edge == 2:
                    value = 0
                else:
                    value = rng.randint(lo, hi)
                fields.append("" if (row + index * 17) % 31 == 0 else encode(value, row))
            writer.writerow(fields)


def parsed(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, schema_overrides=COLUMNS)


def main() -> None:
    if pl.__version__ != "1.44.2":
        raise SystemExit(f"expected Polars 1.44.2, got {pl.__version__}")
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ROUNDTRIP_BINARY")
    binary = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="csv_integer_diff_") as directory:
        source = Path(directory) / "source.csv"
        output = Path(directory) / "mojo.csv"
        generate(source)
        expected = parsed(source)
        for threads in ("1", "32"):
            environment = os.environ | {"DATAFRAME_THREADS": threads}
            subprocess.run([binary, source, output], check=True, env=environment)
            assert_frame_equal(expected, parsed(output), check_dtypes=True)
            print(f"public CSV differential passed: {ROWS} rows, 8 widths, threads={threads}")


if __name__ == "__main__":
    main()
