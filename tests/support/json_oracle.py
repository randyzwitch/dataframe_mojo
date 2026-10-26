"""Independent Polars/Python JSON fixtures and actual Mojo writer consumers."""
import argparse
import base64
import io
import json
import struct
from decimal import Decimal, localcontext
import datetime as dt
import re
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

FIXTURES = Path(__file__).parents[1] / "fixtures/json_tables"


def generate():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    strings = ["", 'é😀\n\r\t\b\f\x00"\\/', "e\u0301", "汉字", "\u2028\u2029", "Greek αβ", "\x01\x1f", "long UTF-8 value beyond twelve bytes"]
    lines = []
    for row in range(257):
        record = {
            "row": row,
            "text": strings[row % len(strings)] if row % 7 else None,
            "integer": row - 128 if row % 11 else None,
            "number": -0.0 if row % 13 == 0 else row / 8,
            "boolean": bool(row % 2) if row % 5 else None,
            "list": [row, None, row + 0.5] if row % 3 else [],
            "struct": {"left": row, "right": strings[row % len(strings)]},
            "none": None,
        }
        if row % 17 == 1:
            del record["text"]
        if row % 19 == 2:
            record["list"] = None
        if row % 23 == 3:
            record["struct"] = None
        elif row % 29 == 4:
            del record["struct"]["right"]
        lines.append(json.dumps(record, ensure_ascii=bool(row % 2), separators=(",", ":")))
        if row % 31 == 0:
            lines.append("")
    source = "\n".join(lines)
    (FIXTURES / "mixed.ndjson").write_text(source, encoding="utf-8")
    expected = pl.read_ndjson(io.StringIO(source))
    expected.write_ndjson(FIXTURES / "mixed.expected.ndjson")
    expected.write_json(FIXTURES / "mixed.expected.json")
    (FIXTURES / "manifest.json").write_text(json.dumps({"polars": pl.__version__, "rows": expected.height, "schema": {k: str(v) for k, v in expected.schema.items()}}, indent=2) + "\n")


def equal_values(actual, expected):
    if isinstance(expected, float):
        assert isinstance(actual, (float, int))
        assert struct.pack("!d", float(actual)) == struct.pack("!d", expected), (actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (actual, expected)
        for key in expected:
            equal_values(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            equal_values(left, right)
    else:
        assert actual == expected, (actual, expected)


def check_written(root):
    root = Path(root)
    expected = pl.read_ndjson(FIXTURES / "mixed.ndjson")
    actual = pl.read_ndjson(root / "json-written-oracle.ndjson")
    assert_frame_equal(actual, expected)
    array = json.loads((root / "json-written-oracle.json").read_text())
    equal_values(array, expected.to_dicts())
    lines = [json.loads(line) for line in (root / "json-written-oracle.ndjson").read_text().split("\n") if line]
    equal_values(lines, expected.to_dicts())
    # Polars cannot read its own Time/Duration/Binary NDJSON in the pinned
    # version. Check those documented wire forms independently with Python.
    logical = root / "json-logical.ndjson"
    if logical.exists():
        rows = [json.loads(line) for line in logical.read_text().split("\n") if line]
        assert [base64.b64decode(row["binary"], validate=True) if row["binary"] is not None else None for row in rows] == [bytes([0, 255, 128, 1]), b"", b"\x01", b"\x01\x02", None]
        assert [row["binary"] for row in rows] == [row["geometry"] for row in rows]
        assert Decimal(rows[0]["d128"]) == Decimal((1, tuple(map(int, str(1 << 100))), -10))
        assert rows[0]["duration-ns"] == "-PT9223372036.854775808S"
        assert rows[-1]["duration-ns"] == "PT9223372036.854775807S"
        with localcontext() as context:
            context.prec = 60
            ticks = [-123456789, 0, 1, 123456789, None]
            durations = [-(1 << 63), -1, 0, 1, (1 << 63) - 1]
            for unit, rate in [("ns", 10**9), ("us", 10**6), ("ms", 1000)]:
                for row, value in zip(rows, durations):
                    text = row["duration-" + unit]
                    sign = -1 if text.startswith("-") else 1
                    assert Decimal(text.removeprefix("-")[2:-1]) * rate * sign == value
                for zone in ["", "UTC", "America/New_York"]:
                    for row, value in zip(rows, ticks):
                        text = row["datetime-" + unit + zone]
                        if value is None:
                            assert text is None
                            continue
                        match = re.fullmatch(r"(.{19})(?:\.(\d+))?(.*)", text)
                        assert match, text
                        whole, fraction, offset = match.groups()
                        instant = dt.datetime.fromisoformat(whole).replace(tzinfo=dt.timezone.utc)
                        seconds = int((instant - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)).total_seconds())
                        assert (Decimal(seconds) + Decimal("0." + (fraction or "0"))) * rate == value
                        assert offset in ("", "Z", "+00:00", "+0000")
            for name, scale, values in [
                ("d32", 3, [-12345, 0, 1, 999999999, None]),
                ("d64", 9, [-12345, 0, 1, 999999999999999999, None]),
                ("d128", 10, [-(1 << 100), 0, 1, 1 << 100, None]),
            ]:
                for row, value in zip(rows, values):
                    assert (Decimal(row[name]) * 10**scale if value is not None else row[name]) == value
        assert [row["cat"] for row in rows] == ["é", "", "a", "é", None]
    print(f"Independent JSON consumers passed: {expected.height} rows, {pl.__version__=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-written", type=Path)
    args = parser.parse_args()
    if args.check_written:
        check_written(args.check_written)
    else:
        generate()
