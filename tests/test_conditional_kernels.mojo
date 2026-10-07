"""Typed conditional kernels (#488): the SIMD select over numeric branches
and the masked reduction of `sum(when(c).then(x))` match the general
evaluator over large, null-bearing, chunked inputs."""
from std.math import isnan
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    lit,
    null,
    when,
)


def frame(rows: Int) raises -> DataFrame:
    var ints = List[Int64](capacity=rows)
    var floats = List[Float64](capacity=rows)
    var narrow = List[Int16](capacity=rows)
    var keys = List[Int64](capacity=rows)
    var valid_i = List[Bool](capacity=rows)
    var valid_f = List[Bool](capacity=rows)
    var nan = Float64(0) / Float64(0)
    for i in range(rows):
        ints.append(Int64((i * 7919) % 1001) - 500)
        floats.append(nan if i % 97 == 0 else Float64(i % 113) / 3)
        narrow.append(Int16(i % 300) - 150)
        keys.append(Int64(i % 37))
        valid_i.append(i % 11 != 0)
        valid_f.append(i % 13 != 0)
    return DataFrame(
        [
            Series("i", Column[Int64](ints^, valid_i^)),
            Series("f", Column[Float64](floats^, valid_f^)),
            Series("n", Column[Int16](narrow^)),
            Series("k", Column[Int64](keys^)),
        ]
    )


def reference(frame: DataFrame, chosen: Expr) raises -> DataFrame:
    """The same selection through the general route: a string branch is
    chosen row by row, so cast the result back."""
    return frame.select_exprs([chosen.copy()])


def close(got: DataFrame, want: DataFrame, message: String) raises:
    """Equal frames, with float sums equal to one part in 1e12: the masked
    reduction adds the same values in a different order."""
    assert_equal(got.height(), want.height(), message)
    for name in want.columns():
        var a = got.column(name)
        var b = want.column(name)
        if a.dtype() == DataType.FLOAT64:
            for i in range(len(a)):
                var x = a.get(i)
                var y = b.get(i)
                assert_equal(x.is_null(), y.is_null(), message + ": " + name)
                if not x.is_null():
                    var scale = max(abs(y.float64()), 1.0)
                    assert_true(
                        abs(x.float64() - y.float64()) <= 1e-12 * scale,
                        message + ": " + name,
                    )
        else:
            assert_true(a.equals(b), message + ": " + name)


def test_numeric_select_matches_for_every_shape() raises:
    var df = frame(20_011)
    var chunked = DataFrame(
        [
            Series._from_chunks(
                [
                    df.column("i").slice(0, 1000),
                    df.column("i").slice(1000, 19_011),
                ]
            ),
            df.column("f"),
            df.column("n"),
            df.column("k"),
        ]
    )
    var cases = List[Expr]()
    # column against column, scalar branches, null literal, narrow type,
    # NaN in a branch, nested conditionals.
    cases.append(
        when(col("i") > lit(Int64(0)))
        .then(col("i"))
        .otherwise(col("k"))
        .alias("a")
    )
    cases.append(
        when(col("f") > lit(10.0))
        .then(col("f"))
        .otherwise(lit(-1.0))
        .alias("b")
    )
    cases.append(
        when(col("i") % lit(Int64(2)) == lit(Int64(0)))
        .then(lit(Int64(1)))
        .otherwise(lit(Int64(0)))
        .alias("c")
    )
    cases.append(when(col("n") > lit(Int16(0))).then(col("n")).end().alias("d"))
    cases.append(
        when(col("f").is_null())
        .then(null(DataType.FLOAT64))
        .otherwise(col("f"))
        .alias("e")
    )
    cases.append(
        when(col("i") > lit(Int64(100)))
        .then(
            when(col("f") > lit(20.0))
            .then(col("f"))
            .otherwise(col("i").cast(DataType.FLOAT64))
        )
        .otherwise(lit(0.0))
        .alias("g")
    )
    cases.append(when(lit(True)).then(col("i")).otherwise(col("k")).alias("h"))
    cases.append(when(lit(False)).then(col("i")).end().alias("j"))
    for source in [df.copy(), chunked.copy()]:
        for expression in cases:
            var got = source.select_exprs([expression.copy()])
            var want = source.select_exprs([expression.cast(DataType.STRING)])
            assert_true(
                got.column(got.columns()[0])
                .cast(DataType.STRING)
                .equals(want.column(want.columns()[0])),
                "select differs for " + got.columns()[0],
            )


def test_masked_sums_match_the_chosen_column() raises:
    """Reductions over `when(c).then(x)` without an otherwise, ungrouped
    and grouped, equal the same reductions over the materialized choice."""
    var df = frame(50_003)
    var materialized = df.with_columns(
        [
            when(col("i") > lit(Int64(0))).then(col("i")).end().alias("ci"),
            when(col("f") > lit(5.0))
            .then(col("f"))
            .otherwise(null(DataType.FLOAT64))
            .alias("cf"),
            when(col("n") < lit(Int16(0))).then(col("n")).end().alias("cn"),
        ]
    )
    var fused: List[Expr] = [
        when(col("i") > lit(Int64(0))).then(col("i")).end().sum().alias("si"),
        when(col("i") > lit(Int64(0))).then(col("i")).end().mean().alias("mi"),
        when(col("i") > lit(Int64(0))).then(col("i")).end().count().alias("ki"),
        when(col("f") > lit(5.0))
        .then(col("f"))
        .otherwise(null(DataType.FLOAT64))
        .sum()
        .alias("sf"),
        when(col("f") > lit(5.0))
        .then(col("f"))
        .otherwise(null(DataType.FLOAT64))
        .max()
        .alias("xf"),
        when(col("f") > lit(5.0))
        .then(col("f"))
        .otherwise(null(DataType.FLOAT64))
        .min()
        .alias("nf"),
        when(col("n") < lit(Int16(0))).then(col("n")).end().sum().alias("sn"),
        when(col("n") < lit(Int16(0))).then(col("n")).end().mean().alias("mn"),
    ]
    var plain: List[Expr] = [
        col("ci").sum().alias("si"),
        col("ci").mean().alias("mi"),
        col("ci").count().alias("ki"),
        col("cf").sum().alias("sf"),
        col("cf").max().alias("xf"),
        col("cf").min().alias("nf"),
        col("cn").sum().alias("sn"),
        col("cn").mean().alias("mn"),
    ]
    var got = df.select_exprs(fused)
    var want = materialized.select_exprs(plain)
    close(got, want, "ungrouped masked reductions differ")
    var grouped = df.group_by("k", maintain_order=True).agg(fused)
    var grouped_want = materialized.group_by("k", maintain_order=True).agg(
        plain
    )
    close(grouped, grouped_want, "grouped masked reductions differ")
    var lazy = df.lazy().group_by("k", maintain_order=True).agg(fused).collect()
    close(lazy, grouped_want, "lazy masked reductions differ")
    # A predicate that selects nothing reduces as an all-null column does.
    var none = df.select_exprs(
        [
            when(col("i") > lit(Int64(10_000)))
            .then(col("i"))
            .end()
            .sum()
            .alias("s"),
            when(col("i") > lit(Int64(10_000)))
            .then(col("i"))
            .end()
            .mean()
            .alias("m"),
        ]
    )
    var none_want = df.with_columns(
        [when(col("i") > lit(Int64(10_000))).then(col("i")).end().alias("c")]
    ).select_exprs([col("c").sum().alias("s"), col("c").mean().alias("m")])
    assert_true(none.equals(none_want))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
