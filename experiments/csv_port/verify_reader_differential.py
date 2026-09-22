"""Deterministic, untimed clean-reader versus Polars CSV differential suite.

This stays intentionally independent of the reader implementation.  It emits
real CSV bytes, invokes the small Mojo driver, then compares typed frames with
the pinned Polars 1.44.2 reader.  It exercises the clean reader's supported
four inferred types and documented options, rather than silently testing
unsupported Polars configuration.
"""
import argparse
import csv
import os
from pathlib import Path
import random
import subprocess
import tempfile

import polars as pl
from polars.testing import assert_frame_equal


BASE_SCHEMA = {
    "id": pl.Int64,
    "value": pl.Float64,
    "active": pl.Boolean,
    "label": pl.String,
}
WIDE_SCHEMA = {
    "i0": pl.Int64,
    "f0": pl.Float64,
    "b0": pl.Boolean,
    "s0": pl.String,
    "i1": pl.Int64,
    "f1": pl.Float64,
    "b1": pl.Boolean,
    "s1": pl.String,
}


def write_csv(path: Path, header, rows, *, bom=False, quote_all=False):
    with path.open("w", newline="", encoding="utf-8-sig" if bom else "utf8") as out:
        writer = csv.writer(
            out,
            lineterminator="\n",
            quoting=csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL,
        )
        writer.writerow(header)
        writer.writerows(rows)


def base_rows(seed: int, count: int, *, null_tokens=False):
    rng = random.Random(seed)
    rows = []
    unicode_values = ["café", "日本語", "Δelta", "plain", "emoji 😀"]
    for i in range(count):
        ident = str(rng.randrange(-2_000_000, 2_000_000))
        value = [
            f"{rng.randrange(-100_000, 100_000)}.{rng.randrange(0, 1_000_000):06d}",
            f"{rng.randrange(-999, 999)}.{rng.randrange(1, 99)}e{rng.randrange(-4, 4):+d}",
            "0.0",
        ][i % 3]
        active = "TrUe" if i % 2 else "FALSE"
        label = f'{unicode_values[i % len(unicode_values)]}, row={i}, quote="{i % 17}"'
        if i % 41 == 0:
            label += "\nsecond line"
        if i % 29 == 0:
            value = ""
        if i % 37 == 0:
            label = ""
        if null_tokens:
            if i % 43 == 0:
                ident = "NULL"
            if i % 47 == 0:
                value = "NA"
            if i % 53 == 0:
                active = "NULL"
            if i % 59 == 0:
                label = "NA"
        rows.append([ident, value, active, label])
    return rows


def wide_rows(seed: int, count: int):
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        s0 = f'left,{i}, "{["café", "日本", "x"][i % 3]}"'
        s1 = f"right={i}"
        if i % 67 == 0:
            s0 += "\nwrapped"
        rows.append([
            str(rng.randrange(-10_000_000, 10_000_000)),
            f"{rng.randrange(-9999, 9999)}.{i % 1000:03d}",
            "true" if i % 2 else "false",
            s0,
            str(rng.randrange(-10_000_000, 10_000_000)),
            f"{rng.randrange(-9999, 9999)}.{(i * 7) % 1000:03d}",
            "FALSE" if i % 3 else "TRUE",
            s1,
        ])
    return rows


def comments_fixture(path: Path):
    # `skip_rows=1` discards the quoted record before the header.  Comments
    # before and between records test that they do not consume the skip count
    # or produce output rows.  The header is hand-written so the first record
    # is intentionally valid CSV with a physical embedded LF.
    with path.open("w", newline="", encoding="utf8") as out:
        out.write("# prelude comment\n")
        out.write('"skip, id",0,true,"quoted\nprelude"\n')
        out.write("# comment before header\n")
        out.write("id,value,active,label\n")
        writer = csv.writer(out, lineterminator="\n")
        for i, row in enumerate(base_rows(91, 101)):
            if i % 17 == 0:
                out.write("# data comment\n")
            writer.writerow(row)


def invoke(binary: Path, source: Path, output: Path, case: str, expected: pl.DataFrame, threads: int):
    subprocess.run(
        [str(binary.resolve()), str(source), str(output), case],
        check=True,
        env={**os.environ, "DATAFRAME_THREADS": str(threads)},
    )
    # Re-read with the expected schema so the serialization layer does not
    # become a type-inference test of either engine.
    actual = pl.read_csv(output, schema=expected.schema)
    assert_frame_equal(actual, expected)
    print(f"PASS {case} threads={threads}: {actual.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    assert pl.__version__ == "1.44.2", pl.__version__
    with tempfile.TemporaryDirectory(prefix="csv-reader-differential-") as directory:
        directory = Path(directory)
        output = directory / "output.csv"

        base = directory / "base.csv"
        write_csv(base, list(BASE_SCHEMA), base_rows(17, 211), bom=True)
        nulls = directory / "nulls.csv"
        # Polars removes quotes before matching configured null tokens.  Make
        # every token quoted so this is not merely a bare-field test.
        write_csv(
            nulls,
            list(BASE_SCHEMA),
            base_rows(31, 193, null_tokens=True),
            quote_all=True,
        )
        wide = directory / "wide.csv"
        write_csv(wide, list(WIDE_SCHEMA), wide_rows(47, 257))
        comments = directory / "comments.csv"
        comments_fixture(comments)
        unterminated = directory / "unterminated.csv"
        write_csv(unterminated, list(BASE_SCHEMA), base_rows(73, 79))
        with unterminated.open("rb+") as out:
            out.seek(-1, os.SEEK_END)
            out.truncate()

        cases = [
            (base, "base_explicit", dict(schema=BASE_SCHEMA)),
            (base, "base_inferred", dict(schema=BASE_SCHEMA)),
            (base, "base_projected_explicit", dict(schema=BASE_SCHEMA, columns=["id", "label"])),
            (base, "base_projected_inferred", dict(schema=BASE_SCHEMA, columns=["id", "label"])),
            (unterminated, "base_explicit", dict(schema=BASE_SCHEMA)),
            (unterminated, "base_inferred", dict(schema=BASE_SCHEMA)),
            (nulls, "nulls_explicit", dict(schema=BASE_SCHEMA, null_values=["NULL", "NA"])),
            (nulls, "nulls_inferred", dict(schema=BASE_SCHEMA, null_values=["NULL", "NA"])),
            (comments, "comments_limit_explicit", dict(schema=BASE_SCHEMA, comment_prefix="#", skip_rows=1, n_rows=53)),
            (comments, "comments_limit_inferred", dict(schema=BASE_SCHEMA, comment_prefix="#", skip_rows=1, n_rows=53)),
            (wide, "wide_explicit", dict(schema=WIDE_SCHEMA)),
            (wide, "wide_inferred", dict(schema=WIDE_SCHEMA)),
        ]
        for threads in (1, 4):
            for source, case, options in cases:
                expected = pl.read_csv(source, **options)
                invoke(args.binary, source, output, case, expected, threads)


if __name__ == "__main__":
    main()
