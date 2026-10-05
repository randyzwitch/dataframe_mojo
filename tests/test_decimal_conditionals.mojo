"""Conditionals and null filling on decimals of every storage width.

`when/then/otherwise`, `fill_null` and `coalesce` take a value from one of
two inputs. On decimals they keep the inputs' type when it is shared and
widen to decimal(38) at the same scale otherwise; unequal scales are an
error, since choosing a value must not round it (#464, #465).
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    coalesce,
    col,
    lit,
    when,
)


def prices() raises -> DataFrame:
    var cents: List[Int64] = [Int64(1999), -250, 100000, 1999, 0, 75]
    var valid: List[Bool] = [True, True, True, True, False, True]
    var wide = List[Int128]()
    for value in cents:
        wide.append(Int128(value) * 3)
    var small: List[Int32] = [Int32(15), 7, 3, 15, 1, 9]
    return DataFrame(
        [
            Series("p64", Column[Int64](cents.copy(), valid.copy())).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("p128", Column[Int128](wide^, valid.copy())).with_dtype(
                DataType.decimal(20, 2)
            ),
            Series("p32", Column[Int32](small^, valid.copy())).with_dtype(
                DataType.decimal(9, 2, 32)
            ),
            Series("day", Column[String](["a", "b", "a", "b", "a", "b"])),
            Series("k", Column[Int64]([Int64(1), 2, 1, 1, 2, 2])),
        ]
    )


def texts(series: Series) raises -> List[String]:
    var out = List[String]()
    for i in range(len(series)):
        var cell = series.get(i)
        out.append(String("null") if cell.is_null() else String(cell))
    return out^


def test_when_without_otherwise_gives_nulls_at_every_width() raises:
    var data = prices()
    var picked = data.select_exprs(
        [
            when(col("day") == "a").then(col("p64")).end().alias("a"),
            when(col("day") == "a").then(col("p128")).end().alias("b"),
            when(col("day") == "a").then(col("p32")).end().alias("c"),
        ]
    )
    assert_true(picked.column("a").dtype() == DataType.decimal(15, 2, 64))
    assert_true(picked.column("b").dtype() == DataType.decimal(20, 2))
    assert_true(picked.column("c").dtype() == DataType.decimal(9, 2, 32))
    assert_equal(
        texts(picked.column("a")),
        ["19.99", "null", "1000.00", "null", "null", "null"],
    )
    assert_equal(
        texts(picked.column("b")),
        ["59.97", "null", "3000.00", "null", "null", "null"],
    )
    assert_equal(
        texts(picked.column("c")),
        ["0.15", "null", "0.03", "null", "null", "null"],
    )


def test_otherwise_of_the_same_type_keeps_it() raises:
    var data = prices()
    var zero = lit("0").cast(DataType.decimal(15, 2, 64))
    var picked = data.select(
        when(col("day") == "a").then(col("p64")).otherwise(zero).alias("v")
    )
    assert_true(picked.column("v").dtype() == DataType.decimal(15, 2, 64))
    assert_equal(
        texts(picked.column("v")),
        ["19.99", "0.00", "1000.00", "0.00", "null", "0.00"],
    )


def test_branches_of_different_decimal_types_widen_at_one_scale() raises:
    var data = prices()
    # A sum of two columns is decimal(38, 2); the other branch is a narrow
    # column, as in `CASE WHEN ... THEN a - b ELSE 0 END`.
    var picked = data.select(
        when(col("day") == "a")
        .then(col("p64") + col("p64"))
        .otherwise(col("p32"))
        .alias("v")
    )
    assert_true(picked.column("v").dtype() == DataType.decimal(38, 2))
    assert_equal(
        texts(picked.column("v")),
        ["39.98", "0.07", "2000.00", "0.15", "null", "0.09"],
    )
    var mixed = data.select(
        when(col("day") == "a")
        .then(col("p64"))
        .otherwise(col("p128"))
        .alias("v")
    )
    assert_true(mixed.column("v").dtype() == DataType.decimal(38, 2))
    assert_equal(
        texts(mixed.column("v")),
        ["19.99", "-7.50", "1000.00", "59.97", "null", "2.25"],
    )
    var other_scale = lit("0").cast(DataType.decimal(15, 3))
    with assert_raises(contains="requires decimals of one scale"):
        _ = data.select(
            when(col("day") == "a")
            .then(col("p64"))
            .otherwise(other_scale)
            .alias("v")
        )


def test_conditional_sums_by_group() raises:
    var data = prices()
    var sums: List[Expr] = [
        when(col("day") == "a").then(col("p64")).end().sum().alias("a"),
        when(col("day") == "b").then(col("p64")).end().sum().alias("b"),
        when(col("day") == "a")
        .then(col("p64") + col("p64"))
        .otherwise(lit("0").cast(DataType.decimal(15, 2, 64)))
        .sum()
        .alias("twice"),
    ]
    var grouped = data.group_by("k", maintain_order=True).agg(sums)
    assert_equal(texts(grouped.column("a")), ["1019.99", "0.00"])
    assert_equal(texts(grouped.column("b")), ["19.99", "-1.75"])
    assert_equal(texts(grouped.column("twice")), ["2039.98", "0.00"])
    var lazy = data.lazy().group_by(["k"]).agg(sums).sort(["k"]).collect()
    assert_equal(texts(lazy.column("a")), ["1019.99", "0.00"])
    assert_equal(texts(lazy.column("twice")), ["2039.98", "0.00"])


def test_fill_null_and_coalesce_at_every_width() raises:
    var data = prices()
    var filled = data.select_exprs(
        [
            col("p64")
            .fill_null(lit("-1").cast(DataType.decimal(15, 2, 64)))
            .alias("a"),
            col("p128")
            .fill_null(lit("-1").cast(DataType.decimal(20, 2)))
            .alias("b"),
            col("p32").fill_null(col("p32")).alias("c"),
            coalesce([col("p64"), col("p128")]).alias("d"),
            coalesce(
                [
                    when(col("day") == "b").then(col("p64")).end(),
                    lit("0").cast(DataType.decimal(15, 2, 64)),
                ]
            ).alias("e"),
        ]
    )
    assert_true(filled.column("a").dtype() == DataType.decimal(15, 2, 64))
    assert_equal(
        texts(filled.column("a")),
        ["19.99", "-2.50", "1000.00", "19.99", "-1.00", "0.75"],
    )
    assert_true(filled.column("b").dtype() == DataType.decimal(20, 2))
    assert_equal(texts(filled.column("b"))[4], "-1.00")
    assert_equal(texts(filled.column("c"))[4], "null")
    # Different decimal types of one scale widen; both null stays null.
    assert_true(filled.column("d").dtype() == DataType.decimal(38, 2))
    assert_equal(
        texts(filled.column("d")),
        ["19.99", "-2.50", "1000.00", "19.99", "null", "0.75"],
    )
    assert_equal(
        texts(filled.column("e")),
        ["0.00", "-2.50", "0.00", "19.99", "0.00", "0.75"],
    )
    with assert_raises(contains="requires decimals of one scale"):
        _ = data.select(
            col("p64").fill_null(lit("0").cast(DataType.decimal(15, 3)))
        )
    with assert_raises(contains="requires two decimal operands"):
        _ = data.select(col("p64").fill_null(lit(Float64(0))))


def test_filled_decimals_keep_working_in_arithmetic() raises:
    var data = prices()
    # TPC-DS q40's shape: a price less a refund that may be missing.
    var net = col("p128") - coalesce(
        [
            when(col("day") == "b").then(col("p64")).end(),
            lit("0").cast(DataType.decimal(15, 2, 64)),
        ]
    )
    var out = data.select(net.alias("net"))
    assert_equal(
        texts(out.column("net")),
        ["59.97", "-5.00", "3000.00", "39.98", "null", "1.50"],
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
