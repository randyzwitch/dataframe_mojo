"""A filter that keeps at least half of each morsel feeds the aggregate
table its rows in place: dropped rows join no group and update a discarded
one, and computed inputs run over every row, retried on the kept rows when
that raises. Every result equals grouping the filtered frame eagerly: string
and integer keys, nulls, float and decimal computed inputs, a key that only
dropped rows hold, and a computed input that overflows only on dropped
rows; one that overflows on a kept row still raises."""
# contract-group: reduce.grouped
from std.testing import assert_equal, assert_raises, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 200_000


def frame() raises -> DataFrame:
    var flags = List[String]()
    var groups = List[Int64]()
    var group_valid = List[Bool]()
    var price = List[Float64]()
    var cents = List[String]()
    var day = List[Int64]()
    for row in range(ROWS):
        flags.append(String(row % 3))
        groups.append(Int64((row * 7919) % 37))
        group_valid.append(row % 41 != 3)
        price.append(Float64(row % 1000) / 4.0)
        cents.append(String(row % 997) + "." + String(row % 10) + "5")
        day.append(Int64(row % 100))
    # Rows past day 95 (dropped by the filter below) hold group 99 only.
    for row in range(0, ROWS, 101):
        groups[row] = 99 if day[row] > 95 else groups[row]
    return DataFrame(
        [
            Series("flag", Column[String](flags^)),
            Series("g", Column[Int64](groups^, group_valid^)),
            Series("price", Column[Float64](price^)),
            Series("amount", Column[String](cents^)).cast(
                DataType.decimal(12, 2, 64)
            ),
            Series("day", Column[Int64](day^)),
        ]
    )


def test_kept_rows_group_as_the_filtered_frame_does() raises:
    var source = frame()
    var aggregations: List[Expr] = [
        col("price").sum().alias("price"),
        (col("price") * (lit(1.0) - col("price") / lit(1000.0)))
        .sum()
        .alias("net"),
        (col("amount") * col("amount")).sum().alias("squares"),
        col("amount").sum().alias("amount"),
        col("day").len().alias("rows"),
    ]
    var kept = col("day") <= lit(Int64(95))
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var piped = (
        source.lazy()
        .filter(kept.copy())
        .group_by(["flag", "g"])
        .agg(aggregations.copy())
        .sort(["flag", "g"])
        .collect()
    )
    var took = "dataframe-path: pipeline.table_in_place\n" in observed.text()
    _ = observed^
    _ = trace^
    assert_true(took, "the filtered rows were compacted, not read in place")
    var eager = (
        source.filter(kept)
        .group_by(["flag", "g"])
        .agg(aggregations)
        .sort(["flag", "g"])
    )
    assert_frame(piped, eager)
    # Group 99 lives only in dropped rows: no group is made for it.
    assert_equal(piped.filter(col("g") == lit(Int64(99))).height(), 0)


def test_only_a_kept_rows_overflow_raises() raises:
    var source = frame()
    var big = List[Int64]()
    for row in range(ROWS):
        # Past the 18 digits a decimal(18) product may hold, on dropped
        # rows only (day 99).
        big.append(Int64(999_999_999) if row % 100 == 99 else Int64(row % 7))
    var wide = source.with_column(
        Series("big", Column[Int64](big^)).cast(DataType.decimal(9, 0, 32))
    )
    var product = (col("big") * col("big") * col("big")).sum().alias("p")
    var fine = (
        wide.lazy()
        .filter(col("day") <= lit(Int64(95)))
        .group_by(["flag"])
        .agg([product.copy()])
        .collect()
    )
    assert_equal(fine.height(), 3)
    with assert_raises():
        _ = (
            wide.lazy()
            .filter(col("day") >= lit(Int64(1)))
            .group_by(["flag"])
            .agg([product^])
            .collect()
        )
