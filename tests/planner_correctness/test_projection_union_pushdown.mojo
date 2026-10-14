"""Filters move through renaming projections and constants, into every
input of a union, and into a subplan several readers filter differently,
without changing any result: each plan is compared with itself run
unoptimized, with nulls in the filtered and aggregated columns. Where a
filter lands is checked through the rows the operators below it read."""
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    lit,
    null,
)
from tests.support.coverage_compare import assert_frame

comptime ROWS = 400


def sales(salt: Int) raises -> DataFrame:
    """Rows over years 2000-2003 and 13 customers, with null amounts and
    null years."""
    var years = List[Int64]()
    var year_valid = List[Bool]()
    var customers = List[Int64]()
    var amounts = List[Int64]()
    var amount_valid = List[Bool]()
    for row in range(ROWS):
        years.append(Int64(2000 + (row * salt) % 4))
        year_valid.append((row * salt) % 37 != 5)
        customers.append(Int64((row * 7 + salt) % 13))
        amounts.append(Int64((row * salt) % 9) - 3)
        amount_valid.append((row + salt) % 11 != 2)
    return DataFrame(
        [
            Series("year", Column[Int64](years^, year_valid^)),
            Series("customer", Column[Int64](customers^)),
            Series("amount", Column[Int64](amounts^, amount_valid^)),
        ]
    )


def channel(source: DataFrame, tag: String) raises -> LazyFrame:
    """One channel's yearly totals per customer, renamed and tagged with
    a constant, as TPC-DS q4's year_total is written."""
    return (
        source.lazy()
        .group_by(["customer", "year"])
        .agg([col("amount").sum().alias("total")])
        .select_exprs(
            [
                col("customer").alias("id"),
                col("year").alias("dyear"),
                col("total"),
                lit(tag).alias("kind"),
            ]
        )
    )


def ordered(frame: DataFrame) raises -> DataFrame:
    return frame.sort(frame.columns())


def assert_same(plan: LazyFrame) raises:
    assert_frame(
        ordered(plan.collect(optimize=True, engine="cpu")),
        ordered(plan.collect(optimize=False, engine="cpu")),
    )


def rows_into(plan: LazyFrame, operator: String) raises -> List[Int]:
    """The input rows of each node whose operator label starts with
    `operator`, in the profile's order."""
    var report = plan.profile(engine="cpu")[1].copy()
    var out = List[Int]()
    for r in range(report.height()):
        if report.item(r, "operator").string().startswith(operator):
            out.append(Int(report.item(r, "input_rows").int64()))
    return out^


def rows_from(plan: LazyFrame, operator: String) raises -> List[Int]:
    """The output rows of each node whose operator label starts with
    `operator`."""
    var report = plan.profile(engine="cpu")[1].copy()
    var out = List[Int]()
    for r in range(report.height()):
        if report.item(r, "operator").string().startswith(operator):
            out.append(Int(report.item(r, "output_rows").int64()))
    return out^


def test_filter_moves_through_a_rename_to_the_grouped_rows() raises:
    var source = sales(3)
    var plan = channel(source, "s").filter(col("dyear") == lit(Int64(2001)))
    assert_same(plan)
    # The year filter reached the rows before grouping: the group_by
    # reads only 2001's rows.
    var kept = source.filter(col("year") == lit(Int64(2001))).height()
    assert_true(kept < ROWS)
    assert_equal(rows_into(plan, "GROUP_BY"), [kept])


def test_a_constant_tag_prunes_union_inputs() raises:
    var a = sales(3)
    var b = sales(5)
    var c = sales(7)
    var union = channel(a, "s").concat(channel(b, "w")).concat(channel(c, "c"))
    for tag in ["s", "w", "c", "x"]:
        for negate in [False, True]:
            var keep = (col("kind") != tag) if negate else (col("kind") == tag)
            assert_same(union.filter(keep & (col("dyear") >= lit(Int64(2001)))))
    # Only the "w" input is grouped; the other two become empty.
    var grouped = rows_into(union.filter(col("kind") == "w"), "GROUP_BY")
    var read = 0
    for rows in grouped:
        read += rows
    assert_equal(read, ROWS)


def test_readers_of_one_subplan_keep_their_own_filters() raises:
    var union = channel(sales(3), "s").concat(channel(sales(5), "w"))
    var readers = List[LazyFrame]()
    for tag in ["s", "w"]:
        for year in [2001, 2002]:
            var keep = (col("kind") == tag) & (col("dyear") == lit(Int64(year)))
            if year == 2001:
                # Over the aggregate: a null total fails it.
                keep = keep & (col("total") > lit(Int64(0)))
            readers.append(
                union.filter(keep).select_exprs(
                    [
                        col("id").alias("id_" + tag + String(year)),
                        col("total").alias("total_" + tag + String(year)),
                    ]
                )
            )
    var plan = (
        readers[0]
        .join(readers[1], left_on=["id_s2001"], right_on=["id_s2002"])
        .join(readers[2], left_on=["id_s2001"], right_on=["id_w2001"])
        .join(readers[3], left_on=["id_s2001"], right_on=["id_w2002"])
    )
    assert_same(plan)
    # The readers' filters reached the shared subplan as their OR: the
    # result every reader scans holds 2002's totals and 2001's positive
    # ones, and no other year's.
    var wanted = 0
    for salt in [3, 5]:
        var totals = (
            sales(salt)
            .group_by(["customer", "year"])
            .agg([col("amount").sum().alias("total")])
        )
        wanted += totals.filter(col("year") == lit(Int64(2002))).height()
        wanted += totals.filter(
            (col("year") == lit(Int64(2001))) & (col("total") > lit(Int64(0)))
        ).height()
    var largest = 0
    for rows in rows_from(plan, "SCAN"):
        largest = max(largest, rows)
    assert_equal(largest, wanted)


def test_constant_predicates_fold_under_three_valued_logic() raises:
    var union = channel(sales(3), "s").concat(channel(sales(5), "w"))
    var predicates: List[Expr] = [
        (col("kind") == "s") | (col("total") > lit(Int64(2))),
        ~(col("kind") == "s") & (col("total") > lit(Int64(0))),
        (col("kind") == "w") | (col("dyear") == lit(Int64(2003))),
        (col("kind") != "s") | col("total").is_null(),
        (col("kind") == "s") & (col("kind") == "w"),
        (col("kind") == null(DataType.STRING)) | (col("kind") == "s"),
    ]
    for predicate in predicates:
        assert_same(union.filter(predicate))
