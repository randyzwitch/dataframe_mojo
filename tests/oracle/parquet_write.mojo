"""Produce Parquet files for the separate-process PyArrow oracle."""
from std.sys import argv
from dataframe import (
    DataFrame,
    DataType,
    Series,
    Column,
    read_parquet,
    write_parquet,
)


def main() raises:
    var directory = String(argv()[1])
    for codec in ["zstd", "snappy", "uncompressed"]:
        for fixture in [
            "types.parquet",
            "list_int.parquet",
            "struct.parquet",
            "empty.parquet",
        ]:
            var source = read_parquet("tests/fixtures/" + fixture)
            write_parquet(
                source,
                directory + "/" + codec + "-" + fixture,
                compression=codec,
                row_group_size=2,
            )
        var columns = List[Series]()
        for unit in ["ns", "us", "ms"]:
            columns.append(
                Series(
                    "duration_" + unit,
                    Column[Int64]([0, -123, 456], [True, False, True]),
                ).with_dtype(DataType.duration(unit))
            )
            columns.append(
                Series(
                    "timestamp_" + unit, Column[Int64]([0, -123, 456])
                ).with_dtype(DataType.datetime(unit))
            )
            columns.append(
                Series(
                    "zoned_" + unit, Column[Int64]([0, -123, 456])
                ).with_dtype(DataType.datetime(unit, "America/New_York"))
            )
        columns.append(
            Series.binary(
                "binary", [[0xFF, 0x00], [], [0x61]], [True, False, True]
            )
        )
        write_parquet(
            DataFrame(columns^),
            directory + "/" + codec + "-temporal.parquet",
            compression=codec,
            row_group_size=2,
        )
