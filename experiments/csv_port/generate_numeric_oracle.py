#!/usr/bin/env python3
"""Verify CSV float fixture bits with the pinned Polars 1.44.2 oracle.

Run with:
    pixi run -e oracle python experiments/csv_port/generate_numeric_oracle.py

The static Mojo fixtures stay hermetic at test time. This script extracts their
literal corpus and checks every expected IEEE bit pattern by casting Polars
Utf8 values directly to Float32/Float64.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "tests" / "test_csv_numeric.mojo"


def bits32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits64(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def cases(body: str, function: str, width: int):
    start = body.index(f"def {function}")
    rest = body[start:]
    next_test = re.search(r"\n\ndef test_", rest[1:])
    if next_test:
        rest = rest[: next_test.start() + 1]
    name = "_assert_bits" if width == 32 else "_assert_bits64"
    pattern = re.compile(
        rf"{name}\(\s*\"([^\"]+)\"\s*,\s*UInt{width}\(0x([0-9A-Fa-f]+)\)",
        re.S,
    )
    return [(text, int(expected, 16)) for text, expected in pattern.findall(rest)]


def main() -> None:
    if pl.__version__ != "1.44.2":
        raise SystemExit(f"expected polars 1.44.2, got {pl.__version__}")
    body = TEST.read_text()
    checks = [
        (32, pl.Float32, bits32, "test_direct_float32_matches_polars_oracle_bits"),
        (64, pl.Float64, bits64, "test_direct_float64_matches_polars_oracle_bits"),
    ]
    count = 0
    for width, dtype, pack, function in checks:
        for text, expected in cases(body, function, width):
            value = pl.Series("value", [text], dtype=pl.String).cast(dtype)[0]
            actual = pack(value)
            if actual != expected:
                raise SystemExit(
                    f"Float{width} {text!r}: fixture 0x{expected:0{width // 4}X}, "
                    f"Polars 0x{actual:0{width // 4}X}"
                )
            count += 1
    print(f"verified {count} Float32/Float64 fixture bit patterns with Polars 1.44.2")


if __name__ == "__main__":
    main()
