"""A conditional whose branches cannot raise (columns, literals, numeric
widening casts) takes each row's value from the branch its condition picks:
a null condition picks the otherwise, a missing otherwise gives null, and
each branch's own nulls stay null. Lazy chunks equal the eager evaluation
and a row-by-row reference, for integers, floats, decimals and Booleans,
in every layout."""
# contract-group: expr.null_logic
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    lit,
    when,
)
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import chunk_layout, offset_layout


comptime _ROWS = 4500


def _frame() raises -> DataFrame:
    var picks = List[Bool]()
    var picks_valid = List[Bool]()
    var ints = List[Int64]()
    var ints_valid = List[Bool]()
    var floats = List[Float64]()
    var flags = List[Bool]()
    var flags_valid = List[Bool]()
    var cents = List[Int64]()
    for row in range(_ROWS):
        picks.append(row % 3 == 0)
        picks_valid.append(row % 7 != 0)
        ints.append(Int64(row * 11 % 101) - 50)
        ints_valid.append(row % 5 != 2)
        floats.append(Float64(row % 13) * 0.5)
        flags.append(row % 4 == 1)
        flags_valid.append(row % 9 != 4)
        cents.append(Int64(row % 777) - 300)
    return DataFrame(
        [
            Series("pick", Column[Bool](picks^, picks_valid)),
            Series("i", Column[Int64](ints^, ints_valid)),
            Series("f", Column[Float64](floats^)),
            Series("flag", Column[Bool](flags^, flags_valid)),
            Series("d", Column[Int64](cents^)).cast(DataType.decimal(9, 2, 64)),
        ]
    )


def _exprs() raises -> List[Expr]:
    return [
        when(col("pick")).then(col("i")).otherwise(lit(Int64(0))).alias("io"),
        when(col("pick")).then(col("i")).end().alias("in"),
        when(col("pick"))
        .then(lit(Int64(1)))
        .otherwise(lit(Int64(0)))
        .alias("one"),
        when(col("pick"))
        .then(col("f"))
        .otherwise(col("i").cast(DataType.FLOAT64))
        .alias("fo"),
        when(col("pick")).then(lit(True)).otherwise(col("flag")).alias("bo"),
        when(col("pick")).then(col("flag")).end().alias("bn"),
        when(col("pick")).then(col("d")).end().alias("dn"),
    ]


def test_blended_conditionals_match_eager_and_a_row_reference() raises:
    var source = _frame()
    for frame in [source.copy(), offset_layout(source), chunk_layout(source)]:
        var lazy = frame.lazy().select_exprs(_exprs()).collect()
        assert_frame(lazy, frame.select_exprs(_exprs()))
        # Row reference for the integer forms.
        var io = lazy.column("io").int64()
        var inn = lazy.column("in").int64()
        var one = lazy.column("one").int64()
        var i = frame.column("i").int64()
        var pick = frame.column("pick").bool()
        for row in range(_ROWS):
            var picked = pick.is_valid(row) and pick._get(row)
            if picked:
                assert_equal(io.is_valid(row), i.is_valid(row))
                assert_equal(inn.is_valid(row), i.is_valid(row))
                if i.is_valid(row):
                    assert_equal(io._get(row), i._get(row))
                assert_equal(one._get(row), 1)
            else:
                assert_true(io.is_valid(row))
                assert_equal(io._get(row), 0)
                assert_true(not inn.is_valid(row))
                assert_equal(one._get(row), 0)


def test_blended_conditionals_feed_reductions() raises:
    var source = _frame()
    var reductions: List[Expr] = [
        when(col("pick"))
        .then(lit(Int64(1)))
        .otherwise(lit(Int64(0)))
        .sum()
        .alias("hits"),
        when(col("pick")).then(col("f")).otherwise(lit(0.0)).sum().alias("f"),
        when(col("pick"))
        .then(col("i"))
        .otherwise(lit(Int64(0)))
        .max()
        .alias("i"),
    ]
    for frame in [source.copy(), offset_layout(source), chunk_layout(source)]:
        var plan = frame.lazy().select_exprs(reductions)
        assert_frame(plan.collect(), frame.select_exprs(reductions))
        assert_frame(plan.collect(), plan.collect(optimize=False))
