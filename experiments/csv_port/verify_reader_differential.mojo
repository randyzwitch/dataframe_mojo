"""Untimed clean-reader differential driver for deterministic CSV fixtures.

The Python companion creates bytes and compares this serialized result to the
pinned Polars result.  Cases deliberately select only the clean reader's
documented option/type surface.
"""
from std.sys import argv
from dataframe import CsvField, CsvSchema, write_csv
from dataframe import read_csv


def _base_schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("id"),
            CsvField.float64("value"),
            CsvField.bool("active"),
            CsvField.string("label"),
        ]
    )


def _wide_schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField.int64("i0"),
            CsvField.float64("f0"),
            CsvField.bool("b0"),
            CsvField.string("s0"),
            CsvField.int64("i1"),
            CsvField.float64("f1"),
            CsvField.bool("b1"),
            CsvField.string("s1"),
        ]
    )


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error("usage: verify_reader_differential INPUT OUTPUT CASE")
    var input = String(args[1])
    var output = String(args[2])
    var scenario = String(args[3])

    if scenario == "base_explicit":
        write_csv(read_csv(input, _base_schema()), output)
    elif scenario == "base_inferred":
        write_csv(read_csv(input), output)
    elif scenario == "base_projected_explicit":
        write_csv(
            read_csv(input, _base_schema(), columns=["id", "label"]),
            output,
        )
    elif scenario == "base_projected_inferred":
        write_csv(read_csv(input, columns=["id", "label"]), output)
    elif scenario == "nulls_explicit":
        write_csv(
            read_csv(input, _base_schema(), null_values=["NULL", "NA"]),
            output,
        )
    elif scenario == "nulls_inferred":
        write_csv(read_csv(input, null_values=["NULL", "NA"]), output)
    elif scenario == "comments_limit_explicit":
        write_csv(
            read_csv(
                input,
                _base_schema(),
                comment_prefix="#",
                skip_rows=1,
                n_rows=53,
            ),
            output,
        )
    elif scenario == "comments_limit_inferred":
        write_csv(
            read_csv(
                input,
                comment_prefix="#",
                skip_rows=1,
                n_rows=53,
            ),
            output,
        )
    elif scenario == "wide_explicit":
        write_csv(read_csv(input, _wide_schema()), output)
    elif scenario == "wide_inferred":
        write_csv(read_csv(input), output)
    else:
        raise Error("unknown reader differential case: " + scenario)
