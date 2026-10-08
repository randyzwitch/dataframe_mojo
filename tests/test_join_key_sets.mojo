"""Join key sets passed sideways (DuckDB's join filter pushdown), the
single-pass integer `is_in`, and semi joins that hash a small left side.

Every plan is checked against the unoptimized plan, which runs none of
these rewrites."""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import Column, DataFrame, Expr, LazyFrame, Series, col, lit


def ints(name: String, n: Int, modulus: Int, step: Int = 7919) -> Series:
    var values = List[Int64](capacity=n)
    for i in range(n):
        values.append(Int64((i * step) % modulus))
    return Series(name, Column[Int64](values^))


def same(plan: LazyFrame, label: String) raises:
    var got = plan.collect().sort(plan.collect().columns())
    var want = plan.collect(optimize=False).sort(got.columns())
    assert_true(got.equals(want), label)


def facts() raises -> DataFrame:
    """A large table joined to a small one before it meets the key set."""
    return DataFrame(
        [
            ints("f_item", 200_000, 5_000),
            ints("f_day", 200_000, 365, 31),
            ints("f_qty", 200_000, 1_000, 13),
        ]
    )


def days() raises -> DataFrame:
    return DataFrame([ints("d_day", 365, 365, 1), ints("d_month", 365, 12, 1)])


def items(n: Int, step: Int) raises -> DataFrame:
    """`n` items with keys spaced `step` apart, one of them null."""
    var keys = List[Int64](capacity=n)
    var valid = List[Bool](capacity=n)
    var labels = List[Int64](capacity=n)
    for i in range(n):
        keys.append(Int64(i * step))
        valid.append(i != 3)
        labels.append(Int64(i))
    return DataFrame(
        [
            Series("i_item", Column[Int64](keys^, valid^)),
            Series("i_label", Column[Int64](labels^)),
        ]
    )


def test_key_sets_reach_the_right_side_and_keep_the_answer() raises:
    """A filtered small table joined to a chain over a large one: its keys
    filter the large table (a list up to 50 keys, a range beyond), with
    nulls, duplicates and keys the right side lacks, for inner and semi
    joins."""
    var stock = (
        facts()
        .lazy()
        .filter(col("f_qty") > lit(Int64(100)))
        .join(days().lazy(), left_on=["f_day"], right_on=["d_day"])
        .select(["f_item", "d_month", "f_qty"])
    )
    for shape in range(4):
        var n = 6 if shape < 2 else 400
        var step = 37 if shape % 2 == 0 else 1
        var small = (
            items(n, step).lazy().filter(col("i_label") >= lit(Int64(0)))
        )
        same(
            small.join(stock, left_on=["i_item"], right_on=["f_item"]),
            "inner " + String(shape),
        )
        same(
            small.join(
                stock, left_on=["i_item"], right_on=["f_item"], how="semi"
            ),
            "semi " + String(shape),
        )
    var small = items(6, 37).lazy().filter(col("i_label") >= lit(Int64(0)))
    var report = (
        small.join(stock, left_on=["i_item"], right_on=["f_item"])
        .profile()[1]
        .copy()
    )
    # The right side runs, filtered, before the join, which then builds on
    # the few stock rows of those items instead of the 180K that pass the
    # quantity filter.
    var narrowed = False
    for r in range(report.height()):
        if report.item(r, "operator").string().startswith("JOIN inner on i_"):
            narrowed = report.item(r, "build_rows").int64() < 1_000
    assert_true(narrowed, String(report))


def test_two_key_sets_through_a_rename_from_a_join_chain() raises:
    """TPC-DS q72's shape: a left input made by joins, two join keys, and
    one right key that is a renamed column of a table under the right
    side's own join. Each key filters on its own where it is selective,
    the right side runs first, and the answer is the unoptimized one."""
    var orders = DataFrame(
        [
            ints("o_item", 40_000, 5_000),
            ints("o_day", 40_000, 365, 31),
            ints("o_kind", 40_000, 9, 1),
        ]
    )
    var kinds = DataFrame([ints("k_kind", 9, 9, 1), ints("k_flag", 9, 2, 1)])
    var weeks = DataFrame(
        [ints("w_day", 365, 365, 1), ints("w_week", 365, 365, 1)]
    )
    var sold = (
        orders.lazy()
        .join(
            kinds.lazy().filter(col("k_flag") == lit(Int64(1))),
            left_on=["o_kind"],
            right_on=["k_kind"],
        )
        .join(
            weeks.lazy()
            .filter(col("w_day") < lit(Int64(20)))
            .select_exprs([col("w_day"), (col("w_week") // 7).alias("week")]),
            left_on=["o_day"],
            right_on=["w_day"],
        )
    )
    var stock = (
        facts()
        .lazy()
        .join(
            weeks.lazy().select_exprs(
                [col("w_day").alias("s_day"), col("w_week").alias("s_week")]
            ),
            left_on=["f_day"],
            right_on=["s_day"],
        )
    )
    same(
        sold.join(
            stock, left_on=["o_item", "o_day"], right_on=["f_item", "s_week"]
        ),
        "two keys through a rename",
    )
    same(
        sold.join(
            stock,
            left_on=["o_item", "o_day"],
            right_on=["f_item", "s_week"],
            how="semi",
        ),
        "two keys, semi",
    )


def test_integer_is_in_matches_the_equalities() raises:
    """`is_in` over typed integer literals is one node; its answer equals
    the OR of equalities, nulls stay null, a null literal never matches,
    and a literal of another width is the same error `==` gives."""
    var n = 1_003
    var values = List[Int32](capacity=n)
    var valid = List[Bool](capacity=n)
    for i in range(n):
        values.append(Int32((i * 17) % 101 - 50))
        valid.append(i % 13 != 2)
    var frame = DataFrame([Series("x", Column[Int32](values^, valid^))])
    var wanted: List[Int32] = [-50, 0, 7, 49, 1000]
    var literals = List[Expr]()
    for w in wanted:
        literals.append(lit(w))
    var listed = frame.select(col("x").is_in(literals).alias("r")).column("r")
    var ored = (col("x") == lit(wanted[0])) | (col("x") == lit(wanted[1]))
    for k in range(2, len(wanted)):
        ored = ored | (col("x") == lit(wanted[k]))
    var expected = frame.select(ored.alias("r")).column("r")
    assert_equal(listed.null_count(), expected.null_count())
    for i in range(n):
        assert_equal(listed.get(i).is_null(), expected.get(i).is_null())
        if not listed.get(i).is_null():
            assert_equal(listed.get(i).bool(), expected.get(i).bool())
    assert_equal(
        frame.filter(col("x").is_in(literals)).height(),
        frame.filter(ored).height(),
    )
    with assert_raises(contains="eq requires matching dtypes"):
        _ = frame.select(col("x").is_in([lit(Int64(7))]))


def test_semi_join_hashes_a_small_left_side() raises:
    """A small left side against a large right one keeps the left rows
    that have a match, in order, with duplicate and null keys; anti keeps
    the others."""
    var left_keys = List[Int64]()
    var left_valid = List[Bool]()
    for i in range(40):
        left_keys.append(Int64((i * 97) % 700_000))
        left_valid.append(i % 9 != 4)
    left_keys.append(left_keys[0])
    left_valid.append(True)
    var left = DataFrame(
        [Series("k", Column[Int64](left_keys^, left_valid^))]
    ).with_column(ints("row", 41, 1_000_000, 1))
    var right = DataFrame([ints("rk", 600_000, 350_000, 2)])
    var semi = left.join(right, left_on=["k"], right_on=["rk"], how="semi")
    var anti = left.join(right, left_on=["k"], right_on=["rk"], how="anti")
    var present = List[Bool](length=700_000, fill=False)
    for i in range(right.height()):
        present[Int(right.column("rk").get(i).int64())] = True
    var want_semi = List[Int64]()
    var want_anti = List[Int64]()
    for i in range(left.height()):
        var cell = left.column("k").get(i)
        var row = left.column("row").get(i).int64()
        if not cell.is_null() and present[Int(cell.int64())]:
            want_semi.append(row)
        else:
            want_anti.append(row)
    assert_equal(semi.height(), len(want_semi))
    assert_equal(anti.height(), len(want_anti))
    for i in range(len(want_semi)):
        assert_equal(semi.column("row").get(i).int64(), want_semi[i])
    for i in range(len(want_anti)):
        assert_equal(anti.column("row").get(i).int64(), want_anti[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
