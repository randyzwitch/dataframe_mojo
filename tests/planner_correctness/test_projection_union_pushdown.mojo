"""Filters move through renaming projections and constants, into every
input of a union, and into a subplan several readers filter differently,
without changing any result: each plan is compared with itself run
unoptimized, with nulls in the filtered and aggregated columns. Where a
filter lands is checked through the rows the operators below it read."""
# contract-group: planner.relational
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
            Series("year", Column[Int64](years^, year_valid)),
            Series("customer", Column[Int64](customers^)),
            Series("amount", Column[Int64](amounts^, amount_valid)),
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


def year_rows(salt: Int, year: Int) raises -> Int:
    var years = sales(salt).column("year")
    var count = 0
    for row in range(ROWS):
        var y = years.get(row)
        if not y.is_null() and y.int64() == Int64(year):
            count += 1
    return count


def test_a_filter_on_join_keys_also_filters_the_other_input() raises:
    var left = channel(sales(3), "s")
    var right = channel(sales(5), "w").select_exprs(
        [
            col("id").alias("w_id"),
            col("dyear").alias("w_year"),
            col("total").alias("w_total"),
        ]
    )
    for how in ["inner", "left", "semi", "anti"]:
        var joined = left.join(
            right,
            left_on=["dyear", "id"],
            right_on=["w_year", "w_id"],
            how=how,
        )
        for keep in [
            col("dyear") == lit(Int64(2001)),
            (col("dyear") >= lit(Int64(2002))) | col("dyear").is_null(),
            (col("dyear") == lit(Int64(2003))) & (col("id") < lit(Int64(6))),
        ]:
            assert_same(joined.filter(keep))
        # The year reaches both inputs: each groups only 2001's rows.
        var plan = joined.filter(col("dyear") == lit(Int64(2001)))
        var read = rows_into(plan, "GROUP_BY")
        assert_equal(len(read), 2)
        var total = read[0] + read[1]
        assert_equal(total, year_rows(3, 2001) + year_rows(5, 2001))


def test_unique_over_every_column_runs_as_a_group_by() raises:
    """A unique keeping any or the first row over all its columns runs as
    an aggregation, with the same rows (and the same order when it keeps
    order); other uniques are left alone. Nulls in a key and a string key
    included."""
    var source = (
        sales(3)
        .lazy()
        .with_columns(
            [
                (col("customer") % lit(Int64(4))).alias("customer"),
                col("amount").cast(DataType.STRING).alias("label"),
            ]
        )
        .select(["year", "customer", "label"])
    )
    var everything: List[String] = ["label", "year", "customer"]
    var subsets: List[List[String]] = [List[String](), everything^, ["year"]]
    for subset in subsets:
        for keep in ["any", "first", "last", "none"]:
            for maintain in [False, True]:
                var plan = source.unique(
                    subset, keep=keep, maintain_order=maintain
                )
                var fast = plan.collect(optimize=True, engine="cpu")
                var plain = plan.collect(optimize=False, engine="cpu")
                if maintain:
                    assert_frame(fast, plain)
                else:
                    assert_frame(ordered(fast), ordered(plain))
                var grouped = len(rows_into(plan, "GROUP_BY")) > 0
                var eligible = len(subset) != 1 and (
                    keep == "any" or keep == "first"
                )
                assert_equal(grouped, eligible)
