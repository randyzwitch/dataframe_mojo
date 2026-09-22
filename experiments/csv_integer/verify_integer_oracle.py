#!/usr/bin/env python3
"""Compare public CSV integer coercion with pinned Polars 1.44.2.

Run: pixi run -e oracle python experiments/csv_integer/verify_integer_oracle.py
"""
import io

import polars as pl

CASES = [
    (pl.Int8, "-128", -128), (pl.Int8, "127", 127),
    (pl.UInt8, "255", 255), (pl.Int16, "-32768", -32768),
    (pl.UInt16, "65535", 65535), (pl.Int32, "-2147483648", -2147483648),
    (pl.UInt32, "4294967295", 4294967295),
    (pl.Int64, "-9223372036854775808", -9223372036854775808),
    (pl.Int64, "+000000000000000000001234", 1234),
    (pl.UInt64, "18446744073709551615", 18446744073709551615),
    (pl.UInt64, "0000000000000000000000000000000000", 0),
    # Source-route coverage: every valid magnitude length from 1 through 20.
    (pl.Int64, "1", 1), (pl.Int64, "12", 12), (pl.Int64, "123", 123),
    (pl.Int64, "1234", 1234), (pl.Int64, "12345", 12345),
    (pl.Int64, "123456", 123456), (pl.Int64, "1234567", 1234567),
    (pl.Int64, "12345678", 12345678), (pl.Int64, "123456789", 123456789),
    (pl.Int64, "1234567890", 1234567890),
    (pl.Int64, "12345678901", 12345678901),
    (pl.Int64, "123456789012", 123456789012),
    (pl.Int64, "1234567890123", 1234567890123),
    (pl.Int64, "12345678901234", 12345678901234),
    (pl.Int64, "123456789012345", 123456789012345),
    (pl.Int64, "1234567890123456", 1234567890123456),
    (pl.Int64, "12345678901234567", 12345678901234567),
    (pl.Int64, "123456789012345678", 123456789012345678),
    (pl.Int64, "1234567890123456789", 1234567890123456789),
    (pl.UInt64, "12345678901234567890", 12345678901234567890),
]


def read_csv_value(dtype: pl.DataType, text: str) -> object:
    return pl.read_csv(
        io.StringIO("value\n" + text + "\n"),
        schema_overrides={"value": dtype},
    )[0, 0]


def main() -> None:
    if pl.__version__ != "1.44.2":
        raise SystemExit(f"expected Polars 1.44.2, got {pl.__version__}")
    for dtype, text, expected in CASES:
        actual = read_csv_value(dtype, text)
        if actual != expected:
            raise SystemExit(f"{dtype} {text!r}: expected {expected}, got {actual}")
    for text in ("-0", "-000"):
        try:
            read_csv_value(pl.UInt8, text)
        except Exception:
            pass
        else:
            raise SystemExit(f"Polars CSV UInt8 accepted {text!r}")
    print(
        f"verified {len(CASES)} public CSV fixtures and unsigned -0 rejection "
        "with Polars 1.44.2"
    )


if __name__ == "__main__":
    main()
