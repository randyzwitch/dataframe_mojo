"""Pinned Polars 1.44.2 oracle for clean CSV schema inference.

Run from the repository root with:

    .pixi/envs/oracle/bin/python \
        /tmp/dataframe-mojo-csv-stack/experiments/csv_port/verify_inference_oracle.py

The cases deliberately cover only the source-port's supported Bool/Int64/
Float64/String inference surface.  It records the actual Python Polars 1.44.2
reader behavior used by ``dataframe/csv_infer.mojo`` tests; it is not a
benchmark and does not exercise the Mojo reader.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl


def observed(payload: bytes, **kwargs: object) -> tuple[list[str], list[str]]:
    with tempfile.TemporaryDirectory(prefix="csv-infer-oracle-") as directory:
        path = Path(directory) / "case.csv"
        path.write_bytes(payload)
        frame = pl.read_csv(path, **kwargs)
    return frame.columns, [str(dtype) for dtype in frame.dtypes]


def check(
    name: str,
    payload: bytes,
    expected: tuple[list[str], list[str]],
    **kwargs: object,
) -> None:
    actual = observed(payload, **kwargs)
    assert actual == expected, f"{name}: {actual!r} != {expected!r}"
    print(f"PASS {name}: {actual}")


def main() -> None:
    assert pl.__version__ == "1.44.2", pl.__version__

    check(
        "regex_signs_zeroes_specials",
        b"i,f,plus,space,upper,nan,bad\n"
        b"007,-7e-05,+1, 1,INF,NaN,1.e1\n"
        b"-3,2.,2,1 ,inf,-NaN,.5e2\n",
        (
            ["i", "f", "plus", "space", "upper", "nan", "bad"],
            ["Int64", "Float64", "String", "String", "String", "Float64", "String"],
        ),
    )
    check(
        "duplicates_quoted_headers",
        b'a,a,"b,b",a\n1,2,1.5,text\n',
        (
            ["a", "a_duplicated_0", "b,b", "a_duplicated_1"],
            ["Int64", "Int64", "Float64", "String"],
        ),
    )
    check(
        "null_leading_and_all_null",
        b"i,b,only\nNA,NA,NA\n1,true,NA\n",
        (["i", "b", "only"], ["Int64", "Boolean", "String"]),
        null_values="NA",
    )
    check(
        "zero_sample_is_string",
        b"i,b\n1,true\n",
        (["i", "b"], ["String", "String"]),
        infer_schema_length=0,
    )
    check(
        "sample_limit",
        b"v\n1\n2\nlate\n",
        (["v"], ["Int64"]),
        infer_schema_length=2,
        # Observe schema rather than fail while decoding the excluded row.
        ignore_errors=True,
    )
    check(
        "all_rows_limit_none",
        b"v\n1\n2\nlate\n",
        (["v"], ["String"]),
        infer_schema_length=None,
    )
    check(
        "headerless",
        b'"1",x\n2,y\n',
        (["column_1", "column_2"], ["Int64", "String"]),
        has_header=False,
    )
    check(
        "bom_comments_skip_rows",
        b"\xef\xbb\xbf\n# before\n"
        b'skip;"quoted\nrecord";x\n'
        b"# after\nid;flag;value\n1;NA;1\n2;true;2.5\n",
        (["id", "flag", "value"], ["Int64", "Boolean", "Float64"]),
        separator=";",
        comment_prefix="#",
        skip_rows=1,
        null_values="NA",
    )
    check(
        "unknown_override_ignored",
        b"v\n1\n",
        (["v"], ["Int64"]),
        schema_overrides={"missing": pl.String},
    )
    check(
        "known_override_any_dtype",
        b"v\n1\n",
        (["v"], ["Float32"]),
        schema_overrides={"v": pl.Float32},
    )
    check(
        "lossy_header",
        b"\xff,b\n1,2\n",
        (["�", "b"], ["Int64", "Int64"]),
    )


if __name__ == "__main__":
    main()
