"""Common subexpression elimination: a row-local subexpression that an
aggregation or a projection reads more than once is computed once, in a
with_columns below it, and every result equals the unoptimized plan's
(TPC-DS q99's delay buckets over one date difference). A subexpression
read once, a column alone and an aggregate stay where they are."""
# contract-group: planner.relational
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Expr, LazyFrame, Series, col, lit, when
from dataframe.lazy import WITH_COLUMNS
from tests.support.coverage_compare import assert_frame


def _one_if(condition: Expr) -> Expr:
    return when(condition).then(lit(Int64(1))).otherwise(lit(Int64(0)))


def _frame() raises -> DataFrame:
    var keys = List[Int64]()
    var shipped = List[Int64]()
    var sold = List[Int64]()
    var valid = List[Bool]()
    for row in range(5000):
        keys.append(Int64(row % 7))
        shipped.append(Int64((row * 37) % 150))
        sold.append(Int64(row % 11))
        valid.append(row % 13 != 0)
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("shipped", Column[Int64](shipped^, valid^)),
            Series("sold", Column[Int64](sold^)),
        ]
    )


def _buckets() -> List[Expr]:
    var lag = col("shipped") - col("sold")
    return [
        _one_if(lag <= lit(Int64(30))).sum().alias("early"),
        _one_if((lag > lit(Int64(30))) & (lag <= lit(Int64(90))))
        .sum()
        .alias("middle"),
        _one_if(lag > lit(Int64(90))).sum().alias("late"),
        (col("sold") * lit(Int64(2))).max().alias("once"),
    ]


def _shared(plan: LazyFrame) -> List[String]:
    """The names each with_columns of the optimized plan computes."""
    var names = List[String]()
    for node in plan._nodes:
        if node.kind == WITH_COLUMNS:
            for e in node.exprs:
                names.append(e._name)
    return names^


def test_a_grouped_aggregation_computes_a_repeated_difference_once() raises:
    var query = _frame().lazy().group_by("k").agg(_buckets()).sort("k")
    var shared = _shared(query._optimized())
    # The difference alone: the doubled column is read once.
    assert_equal(len(shared), 1)
    assert_true(shared[0].startswith("__cse_"))
    var result = query.collect()
    assert_equal(result.columns(), ["k", "early", "middle", "late", "once"])
    assert_frame(result, query.collect(optimize=False))


def test_a_projection_of_reductions_computes_it_once() raises:
    var query = _frame().lazy().select_exprs(_buckets())
    assert_equal(len(_shared(query._optimized())), 1)
    var result = query.collect()
    assert_equal(result.columns(), ["early", "middle", "late", "once"])
    assert_frame(result, query.collect(optimize=False))
    # A row-local projection keeps only its own columns.
    var lag = col("shipped") - col("sold")
    var rows = (
        _frame()
        .lazy()
        .select_exprs([(lag * lag).alias("square"), (lag + lag).alias("twice")])
    )
    var collected = rows.collect()
    assert_equal(collected.columns(), ["square", "twice"])
    assert_frame(collected, rows.collect(optimize=False))
