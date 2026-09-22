"""Untimed differential driver; Python checks serialized output against Polars."""
from std.sys import argv
from dataframe import CsvField, CsvSchema, write_csv
from dataframe.csv_reader import read_csv_explicit, read_csv_inferred


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error(
            "usage: verify_reader INPUT OUTPUT explicit|inferred full|projected"
        )
    var columns = List[String]()
    if String(args[4]) == "projected":
        columns = ["id", "label"]
    if String(args[3]) == "inferred":
        var frame = read_csv_inferred(String(args[1]), columns=columns)
        write_csv(frame, String(args[2]))
    else:
        var schema = CsvSchema(
            [
                CsvField.int64("id"),
                CsvField.float64("value"),
                CsvField.bool("active"),
                CsvField.string("label"),
            ]
        )
        var frame = read_csv_explicit(String(args[1]), schema, columns=columns)
        write_csv(frame, String(args[2]))
