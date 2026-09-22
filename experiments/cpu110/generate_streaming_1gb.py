#!/usr/bin/env python3
"""Create a repeatable >=1 GiB, 10-column CSV fixture for streaming tests.

The input is the established one-million-row, eight-column fixture.  The
script materializes one transformed body, then copies that body in large
blocks, so only the first pass parses individual lines.  The CSV header is
written exactly once.
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path


DEFAULT_SOURCE = Path("build/bench_polars/left_1000000.csv")
DEFAULT_OUTPUT = Path("build/cpu110_streaming_1gb.csv")
GIB = 1024**3
COPY_BUFFER = 16 * 1024 * 1024


def transformed_body(source: Path, scratch: Path) -> tuple[int, int]:
    """Append two typed fields to each source record and return rows/bytes."""
    rows = 0
    with source.open("rb") as source_file, scratch.open("wb", buffering=COPY_BUFFER) as body:
        header = source_file.readline()
        if not header:
            raise ValueError(f"empty input: {source}")
        for line in source_file:
            if not line.endswith(b"\n"):
                raise ValueError("input must be newline terminated")
            # Constants keep the added columns valid Int64 and Float64 values
            # while preserving x/y exactly for the reference aggregation.
            body.write(line[:-1])
            body.write(b",17,3.25\n")
            rows += 1
    return rows, scratch.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--min-bytes", type=int, default=GIB)
    args = parser.parse_args()

    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if not args.source.is_file():
        raise FileNotFoundError(args.source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp_dir:
        body_path = Path(temp_dir) / "body.csv"
        rows_per_copy, body_bytes = transformed_body(args.source, body_path)
        repetitions = max(args.repetitions, math.ceil(args.min_bytes / body_bytes))

        with args.source.open("rb") as source_file, args.output.open(
            "wb", buffering=COPY_BUFFER
        ) as output:
            source_header = source_file.readline().rstrip(b"\r\n")
            output.write(source_header + b",extra_a,extra_b\n")
            for _ in range(repetitions):
                with body_path.open("rb", buffering=COPY_BUFFER) as body:
                    shutil.copyfileobj(body, output, length=COPY_BUFFER)

    expected_min = args.min_bytes
    actual_bytes = args.output.stat().st_size
    if actual_bytes < expected_min:
        raise AssertionError(f"fixture is only {actual_bytes} bytes")
    print(
        f"wrote {args.output}: {rows_per_copy * repetitions} rows, "
        f"{actual_bytes} bytes, {repetitions} copies"
    )


if __name__ == "__main__":
    main()
