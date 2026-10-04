"""Parquet string columns stored dictionary-encoded read as String columns
that keep their codes (`StringCodes`), and grouping by them groups the
codes. Values, nulls and group results must match plain strings, across
row groups whose dictionaries differ, through slices, and with another key.
"""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    read_parquet,
    write_parquet,
)

comptime PATH = "/tmp/dataframe_parquet_dictionary_strings.parquet"
comptime ROWS = 12_000


def source() raises -> DataFrame:
    var key = List[String](capacity=ROWS)
    var valid = List[Bool](capacity=ROWS)
    var other = List[Int64](capacity=ROWS)
    var value = List[Int64](capacity=ROWS)
    for i in range(ROWS):
        # Later row groups bring values earlier ones lack, so the running
        # dictionary grows from one batch to the next.
        key.append("k" + String((i * 7919) % (50 + i // 400)))
        valid.append(i % 41 != 5)
        other.append(Int64(i % 3))
        value.append(Int64(i % 101))
    return DataFrame(
        [
            Series("key", Column[String](key^, valid^)),
            Series("other", Column[Int64](other^)),
            Series("value", Column[Int64](value^)),
        ]
    )


def same(got: DataFrame, want: DataFrame, by: List[String]) raises:
    assert_equal(got.columns(), want.columns())
    assert_equal(got.height(), want.height())
    for name in got.columns():
        assert_equal(got.column(name).dtype(), want.column(name).dtype())
    assert_true(got.sort(by).equals(want.sort(by)), "grouped results differ")


def test_codes_kept_and_grouping_matches_strings() raises:
    var plain = source()
    write_parquet(plain, PATH, row_group_size=2_000)
    var read = read_parquet(PATH)
    assert_equal(read.column("key").dtype(), DataType.STRING)
    assert_true(read.column("key").equals(plain.column("key")))
    assert_true(
        Bool(read.column("key")._dictionary_codes()), "codes were not kept"
    )
    var aggs: List[Expr] = [
        col("value").sum().alias("s"),
        col("value").len().alias("n"),
    ]
    var key_sets: List[List[String]] = [["key"], ["key", "other"]]
    for keys in key_sets:
        var by = keys.copy()
        same(
            read.group_by(keys).agg(aggs),
            plain.group_by(keys).agg(aggs),
            by,
        )
        var ordered = read.group_by(keys, maintain_order=True).agg(aggs)
        assert_true(
            ordered.equals(plain.group_by(keys, maintain_order=True).agg(aggs))
        )
    # A slice keeps its codes in step with its rows.
    var part = read.slice(3_333, 5_000)
    assert_true(Bool(part.column("key")._dictionary_codes()))
    same(
        part.group_by("key").agg(aggs),
        plain.slice(3_333, 5_000).group_by("key").agg(aggs),
        ["key"],
    )


def test_operations_drop_codes_and_selection_reads_them() raises:
    write_parquet(source(), PATH, row_group_size=2_000)
    var read = read_parquet(PATH)
    # A new column from the strings carries no codes.
    var upper = read.select_exprs([col("key").str().to_uppercase()])
    assert_true(not upper.column("key")._dictionary_codes())
    var taken = read.filter(col("other") == 1)
    assert_true(not taken.column("key")._dictionary_codes())
    # Selecting only the string column still finds it dictionary-encoded.
    var only = read_parquet(PATH, columns=["key"])
    assert_true(Bool(only.column("key")._dictionary_codes()))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
