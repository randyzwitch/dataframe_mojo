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
    lit,
    read_parquet,
    when,
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


def coded(frame: DataFrame) raises -> Bool:
    return Bool(frame.column("key")._dictionary_codes())


def test_gathers_keep_codes_and_group_like_strings() raises:
    """A filter, take, sort or join output gathers rows, and the codes go
    with them: each result still groups by codes and gives what the same
    operation on plain strings gives."""
    var plain = source()
    write_parquet(plain, PATH, row_group_size=2_000)
    var read = read_parquet(PATH)
    var aggs: List[Expr] = [
        col("value").sum().alias("s"),
        col("key").count().alias("n"),
    ]
    var predicates: List[Expr] = [
        col("other") == 1,
        col("value") > 90,
        (col("other") == 0) & (col("value") < 50),
        col("key") == "k7",
        col("key").is_null() | (col("value") == 3),
        col("value") < 0,
    ]
    for predicate in predicates:
        var kept = read.filter(predicate)
        assert_true(coded(kept), "a filter dropped the codes")
        var want = plain.filter(predicate)
        assert_true(kept.equals(want), "filtered rows differ")
        same(
            kept.group_by("key").agg(aggs),
            want.group_by("key").agg(aggs),
            ["key"],
        )
        same(
            kept.group_by(["key", "other"]).agg(aggs),
            want.group_by(["key", "other"]).agg(aggs),
            ["key", "other"],
        )
        # A second gather reads the first one's codes.
        var again = kept.filter(col("value") % 2 == 0)
        assert_true(coded(again))
        same(
            again.group_by("key").agg(aggs),
            want.filter(col("value") % 2 == 0).group_by("key").agg(aggs),
            ["key"],
        )
    var rows: List[Int] = [11_999, 0, 5, 5, 7_001, 46]
    var taken = read.take(rows)
    assert_true(coded(taken))
    same(
        taken.group_by("key").agg(aggs),
        plain.take(rows).group_by("key").agg(aggs),
        ["key"],
    )
    var ordered = read.sort(["value", "other"])
    assert_true(coded(ordered))
    same(
        ordered.group_by("key").agg(aggs),
        plain.group_by("key").agg(aggs),
        ["key"],
    )
    # A slice of a filtered frame keeps codes in step with its rows.
    var part = read.filter(col("other") == 1).slice(100, 1_500)
    assert_true(coded(part))
    same(
        part.group_by("key").agg(aggs),
        plain.filter(col("other") == 1)
        .slice(100, 1_500)
        .group_by("key")
        .agg(aggs),
        ["key"],
    )


def test_join_outputs_keep_codes_and_null_extended_rows_group_as_null() raises:
    var plain = source()
    write_parquet(plain, PATH, row_group_size=2_000)
    var read = read_parquet(PATH)
    var ids = List[Int64]()
    var weights = List[Int64]()
    for i in range(140):
        ids.append(Int64(i))
        weights.append(Int64(i * 3))
    var left = DataFrame(
        [
            Series("value", Column[Int64](ids^)),
            Series("w", Column[Int64](weights^)),
        ]
    )
    var aggs: List[Expr] = [
        col("w").sum().alias("s"),
        col("key").count().alias("n"),
    ]
    for how in ["inner", "left"]:
        # Values 101..139 have no row on the Parquet side: a left join
        # null-extends them, and they must group as null, not as code 0.
        var joined = left.join(read, on="value", how=how)
        assert_true(coded(joined), "a join dropped the codes")
        same(
            joined.group_by("key").agg(aggs),
            left.join(plain, on="value", how=how).group_by("key").agg(aggs),
            ["key"],
        )
    var probe = read.join(left, on="value", how="inner")
    assert_true(coded(probe))
    same(
        probe.group_by("key").agg(aggs),
        plain.join(left, on="value", how="inner").group_by("key").agg(aggs),
        ["key"],
    )


def test_new_strings_drop_codes_and_selection_reads_them() raises:
    write_parquet(source(), PATH, row_group_size=2_000)
    var read = read_parquet(PATH)
    # A new column from the strings carries no codes.
    var upper = read.select_exprs([col("key").str().to_uppercase()])
    assert_true(not upper.column("key")._dictionary_codes())
    # Selecting only the string column still finds it dictionary-encoded.
    var only = read_parquet(PATH, columns=["key"])
    assert_true(Bool(only.column("key")._dictionary_codes()))


def test_row_wise_expressions_evaluate_on_the_codes() raises:
    """A row-wise expression over a coded String column runs once per
    dictionary value and is spread by code (#489), with the same answer
    as over the plain strings: comparisons with literals, is_in, string
    operations, conditionals, and casts, across row-group chunks, slices
    and gathered rows with nulls."""
    var plain = source()
    write_parquet(plain, PATH, row_group_size=2_000)
    var read = read_parquet(PATH)
    assert_true(Bool(read.column("key")._dictionary_codes()))
    var expressions = List[Expr]()
    expressions.append((col("key") == lit(String("k7"))).alias("eq"))
    expressions.append((col("key") != lit(String("k7"))).alias("ne"))
    expressions.append((col("key") > lit(String("k3"))).alias("gt"))
    var members: List[String] = ["k1", "k12", "k40", "missing"]
    var member_literals = List[Expr]()
    for member in members:
        member_literals.append(lit(member))
    expressions.append(col("key").is_in(member_literals).alias("member"))
    expressions.append(col("key").str().len_chars().alias("length"))
    expressions.append(col("key").str().to_uppercase().alias("upper"))
    expressions.append(col("key").str().contains("1").alias("has1"))
    expressions.append(col("key").is_null().alias("missing"))
    expressions.append(
        when(col("key") == lit(String("k2")))
        .then(lit(String("two")))
        .otherwise(col("key"))
        .alias("picked")
    )
    # Two-column expressions reach the comparison node by itself, inside
    # the per-node evaluation of a batch.
    expressions.append(
        (
            (col("key") == lit(String("k7"))) | (col("other") == lit(Int64(2)))
        ).alias("either")
    )
    expressions.append(
        (
            (lit(String("k3")) < col("key")) & (col("other") >= lit(Int64(0)))
        ).alias("literal_first")
    )
    expressions.append(
        when(col("key") <= lit(String("k1")))
        .then(col("value"))
        .otherwise(lit(Int64(-1)))
        .alias("picked_value")
    )
    for frame in [
        read.copy(),
        read.slice(1_001, 7_777),
        read.filter(col("other") == lit(Int64(1))),
    ]:
        # A new column from the strings carries no codes.
        var stripped = frame.select_exprs(
            [
                col("key")
                .str()
                .to_uppercase()
                .str()
                .to_lowercase()
                .alias("key"),
                col("other"),
                col("value"),
            ]
        )
        # The stripped column has no codes: the strings themselves.
        assert_true(not stripped.column("key")._dictionary_codes())
        var got = frame.select_exprs(expressions)
        var want = stripped.select_exprs(expressions)
        assert_true(got.equals(want), "coded evaluation differs")
        assert_true(got.height() == frame.height())
        # The same comparison inside a streamed grouped sum (TPC-DS q59's
        # shape: the condition is evaluated per batch of the stream).
        var conditional = (
            when(col("key") == lit(String("k7")))
            .then(col("value"))
            .otherwise(lit(Int64(0)))
            .sum()
            .alias("k7_sum")
        )
        var got_sum = (
            frame.lazy()
            .group_by(["other"])
            .agg([conditional.copy()])
            .sort("other")
            .collect()
        )
        var want_sum = (
            stripped.lazy()
            .group_by(["other"])
            .agg([conditional.copy()])
            .sort("other")
            .collect()
        )
        assert_true(got_sum.equals(want_sum), "coded grouped sum differs")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
