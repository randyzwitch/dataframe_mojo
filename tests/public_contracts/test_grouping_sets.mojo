"""Test ROLLUP, CUBE and GROUPING SETS in eager and lazy execution. Each
expands into its grouping sets; `grouping()` distinguishes subtotal nulls
from null key values. Duplicate sets are preserved, the empty set yields
one row even for empty input, and each set matches a plain `group_by`."""
# contract-group: reduce.grouping_sets
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    Expr,
    LazyFrame,
    Series,
    col,
    grouping,
    grouping_id,
    null,
)
from tests.support.coverage_compare import assert_frame

# Stands for a null in literal expectations.
comptime NULL = Int64.MIN


def sales() raises -> DataFrame:
    """Six rows; `a` and `b` hold a null key value each."""
    return DataFrame(
        [
            Series(
                "a",
                Column[String](
                    ["x", "x", "y", "", "x", "y"],
                    [True, True, True, False, True, True],
                ),
            ),
            Series(
                "b",
                Column[Int64](
                    [1, 2, 1, 2, 0, 1], [True, True, True, True, False, True]
                ),
            ),
            Series("m", Column[Int64]([10, 20, 30, 40, 50, 60])),
        ]
    )


def strings(values: List[String], nulls: List[Bool]) raises -> Series:
    return Series("a", Column[String](values, nulls))


def expected(
    a: List[String], b: List[Int64], total: List[Int64], gid: List[Int64]
) raises -> DataFrame:
    """Rows of (a, b, total, gid); the string "null" and NULL are nulls."""
    var a_valid = List[Bool]()
    for value in a:
        a_valid.append(value != "null")
    var b_valid = List[Bool]()
    for value in b:
        b_valid.append(value != NULL)
    return DataFrame(
        [
            Series("a", Column[String](a.copy(), a_valid.copy())),
            Series("b", Column[Int64](b.copy(), b_valid.copy())),
            Series("total", Column[Int64](total.copy())),
            Series("gid", Column[Int64](gid.copy())),
        ]
    )


def aggs() -> List[Expr]:
    return [
        col("m").sum().alias("total"),
        grouping_id(["a", "b"]).alias("gid"),
    ]


def test_rollup_matches_duckdb() raises:
    var want = expected(
        ["x", "x", "x", "y", "null", "x", "y", "null", "null"],
        [1, 2, NULL, 1, 2, NULL, NULL, NULL, NULL],
        [10, 20, 50, 90, 40, 80, 90, 40, 210],
        [0, 0, 0, 0, 0, 1, 1, 1, 3],
    )
    var lazy = sales().lazy().group_by_rollup(["a", "b"]).agg(aggs()).collect()
    var eager = sales().group_by_rollup(["a", "b"]).agg(aggs())
    for frame in [lazy.copy(), eager.copy()]:
        assert_frame(frame.sort(["gid", "a", "b"]), want)


def test_grouping_columns_and_names() raises:
    var frame = (
        sales()
        .group_by_rollup(["a", "b"])
        .agg(
            [
                col("m").sum(),
                col("m").len().alias("n"),
                grouping("a"),
                grouping("b"),
                grouping_id(["a", "b"]),
                grouping_id(["b", "a"]).alias("swapped"),
            ]
        )
        .sort(["grouping_id(a, b)", "a", "b"])
    )
    assert_equal(
        frame.columns(),
        [
            "a",
            "b",
            "m",
            "n",
            "grouping(a)",
            "grouping(b)",
            "grouping_id(a, b)",
            "swapped",
        ],
    )
    # The last row is the grand total, 3 for (a, b) and 3 for (b, a); the
    # row before it is the null-key subtotal, where only b is rolled up.
    var last = frame.height() - 1
    assert_equal(frame.item(last, "m").int64(), 210)
    assert_equal(frame.item(last, "n").int64(), 6)
    assert_equal(frame.item(last, "grouping(a)").int64(), 1)
    assert_equal(frame.item(last, "grouping(b)").int64(), 1)
    assert_equal(frame.item(last, "grouping_id(a, b)").int64(), 3)
    assert_equal(frame.item(last - 1, "grouping_id(a, b)").int64(), 1)
    assert_equal(frame.item(last - 1, "swapped").int64(), 2)
    assert_true(frame.item(last - 1, "a").is_null())
    assert_equal(frame.item(last - 1, "m").int64(), 40)


def test_cube_matches_duckdb() raises:
    var want = expected(
        [
            "x",
            "x",
            "x",
            "y",
            "null",
            "x",
            "y",
            "null",
            "null",
            "null",
            "null",
            "null",
        ],
        [1, 2, NULL, 1, 2, NULL, NULL, NULL, 1, 2, NULL, NULL],
        [10, 20, 50, 90, 40, 80, 90, 40, 100, 60, 50, 210],
        [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
    )
    assert_frame(
        sales().group_by_cube(["a", "b"]).agg(aggs()).sort(["gid", "a", "b"]),
        want,
    )
    assert_frame(
        sales()
        .lazy()
        .group_by_cube(["a", "b"])
        .agg(aggs())
        .sort(["gid", "a", "b"])
        .collect(),
        want,
    )


def test_grouping_sets_keep_duplicates() raises:
    var sets: List[List[String]] = [["a"], ["a"], ["b"]]
    var want = expected(
        ["x", "x", "y", "y", "null", "null", "null", "null", "null"],
        [NULL, NULL, NULL, NULL, NULL, NULL, 1, 2, NULL],
        [80, 80, 90, 90, 40, 40, 100, 60, 50],
        [1, 1, 1, 1, 1, 1, 2, 2, 2],
    )
    assert_frame(
        sales()
        .group_by_grouping_sets(sets)
        .agg(aggs())
        .sort(["gid", "a", "b"]),
        want,
    )
    # A key listed twice in one set is one key; the set order is the
    # expansion order.
    var doubled: List[List[String]] = [["a", "a"], []]
    var frame = sales().group_by_grouping_sets(doubled).agg(aggs())
    assert_equal(frame.height(), 4)
    assert_equal(frame.columns(), ["a", "total", "gid"])
    assert_equal(frame.item(3, "total").int64(), 210)
    # b is no key here, so it counts as rolled up: the total row has gid 3.
    assert_equal(frame.item(3, "gid").int64(), 3)


def test_rollup_with_repeated_key_matches_duckdb() raises:
    # DuckDB: rollup(a, a) is (a, a), (a), (): the a rows twice, one total.
    var frame = sales().group_by_rollup(["a", "a"]).agg(col("m").sum())
    assert_equal(frame.height(), 3 + 3 + 1)
    var cube = sales().group_by_cube(["a", "a"]).agg(col("m").sum())
    assert_equal(cube.height(), 3 + 3 + 3 + 1)


def test_empty_input() raises:
    var empty = sales().filter(col("m") > 1000)
    var frame = empty.group_by_rollup(["a"]).agg(
        [col("m").sum().alias("total"), grouping("a").alias("g")]
    )
    # DuckDB: only the grand total, which is a row over no rows. Its sum is
    # this library's empty sum (0; DuckDB's is null).
    assert_equal(frame.height(), 1)
    assert_true(frame.item(0, "a").is_null())
    assert_equal(frame.item(0, "total").int64(), 0)
    assert_equal(frame.item(0, "g").int64(), 1)
    var lazy = empty.lazy().group_by_cube(["a", "b"]).agg(col("m").len())
    assert_equal(lazy.collect().height(), 1)


def test_errors() raises:
    with assert_raises():
        _ = sales().group_by_rollup(["missing"]).agg(col("m").sum())
    with assert_raises():
        _ = sales().group_by_rollup(List[String]()).agg(col("m").sum())
    with assert_raises():
        _ = (
            sales()
            .group_by_grouping_sets(List[List[String]]())
            .agg(col("m").sum())
        )


def big() raises -> DataFrame:
    var a = List[Int64]()
    var a_valid = List[Bool]()
    var b = List[String]()
    var m = List[Int64]()
    for i in range(20000):
        a.append(Int64((i * 7) % 13))
        a_valid.append(i % 11 != 0)
        b.append("s" + String(i % 5))
        m.append(Int64(i % 1000))
    return DataFrame(
        [
            Series("a", Column[Int64](a^, a_valid.copy())),
            Series("b", Column[String](b^)),
            Series("m", Column[Int64](m^)),
        ]
    )


def test_every_set_equals_group_by_on_20k_rows() raises:
    var frame = big()
    var plan = (
        frame.lazy()
        .group_by_cube(["a", "b"])
        .agg(
            [
                col("m").sum().alias("total"),
                col("m").len().alias("n"),
                grouping_id(["a", "b"]).alias("gid"),
            ]
        )
    )
    var result = plan.collect()
    for mask in range(4):
        # Bit 1 set: a is rolled up. Bit 0 set: b is rolled up.
        var members = List[String]()
        if mask & 2 == 0:
            members.append("a")
        if mask & 1 == 0:
            members.append("b")
        var got = result.filter(col("gid") == mask)
        var want: LazyFrame
        var plain: List[Expr] = [
            col("m").sum().alias("total"),
            col("m").len().alias("n"),
        ]
        if len(members) == 0:
            want = frame.lazy().select_exprs(plain)
        else:
            want = frame.lazy().group_by(members).agg(plain)
        var fill = List[Expr]()
        if mask & 2 != 0:
            fill.append(null("int64").alias("a"))
        if mask & 1 != 0:
            fill.append(null("string").alias("b"))
        if len(fill) > 0:
            want = want.with_columns(fill)
        var order: List[String] = ["a", "b", "total", "n"]
        assert_equal(got.height(), want.collect().height())
        assert_frame(
            got.select(order).sort(["a", "b"]),
            want.select(order).collect().sort(["a", "b"]),
        )


def facts() raises -> DataFrame:
    """Enough rows for several morsels, with nulls in a measure and a key."""
    var item = List[Int64]()
    var m = List[Int64]()
    var m_valid = List[Bool]()
    var x = List[Float64]()
    for i in range(300000):
        item.append(Int64((i * 31) % 40))
        m.append(Int64(i % 977))
        m_valid.append(i % 13 != 0)
        x.append(Float64(i % 101) / 4.0)
    return DataFrame(
        [
            Series("item", Column[Int64](item^)),
            Series("m", Column[Int64](m^, m_valid^)),
            Series("x", Column[Float64](x^)),
        ]
    )


def items() raises -> DataFrame:
    var item = List[Int64]()
    var category = List[String]()
    var class_valid = List[Bool]()
    var class_name = List[String]()
    for i in range(40):
        item.append(Int64(i))
        category.append("a category name past twelve bytes " + String(i % 3))
        class_name.append("class" + String(i % 7))
        class_valid.append(i % 9 != 0)
    return DataFrame(
        [
            Series("item", Column[Int64](item^)),
            Series("category", Column[String](category^)),
            Series("class", Column[String](class_name^, class_valid^)),
        ]
    )


def check_sets_against_group_by(
    source: LazyFrame,
    keys: List[String],
    sets: List[List[String]],
    plain: List[Expr],
) raises:
    """Every set's rows of one grouping-sets aggregation (found by its
    grouping id) equal the same `group_by` of that set, the empty set a
    plain reduction, with the keys it leaves out null."""
    var aggregates = plain.copy()
    aggregates.append(grouping_id(keys).alias("gid"))
    var result = source.group_by_grouping_sets(sets).agg(aggregates).collect()
    var rows = 0
    for one in sets:
        # The first key is the id's highest bit; a bit is set where the set
        # leaves the key out.
        var mask = 0
        for name in keys:
            mask = 2 * mask + (0 if name in one else 1)
        var want: LazyFrame
        if len(one) == 0:
            want = source.select_exprs(plain)
        else:
            want = source.group_by(one).agg(plain)
        var fill = List[Expr]()
        for name in keys:
            if name not in one:
                fill.append(null("string").alias(name))
        if len(fill) > 0:
            want = want.with_columns(fill)
        var columns = keys.copy()
        for e in plain:
            columns.append(e._name)
        var got = result.filter(col("gid") == mask).select(columns)
        var expected = want.select(columns).collect()
        rows += expected.height()
        assert_equal(got.height(), expected.height())
        assert_frame(got.sort(keys), expected.sort(keys))
    assert_equal(result.height(), rows)


def test_sets_sunk_together_over_a_join_match_group_by() raises:
    """The aggregations the aggregate table keeps inline run in one pass
    over the join's morsels, with a state per set."""
    var source = (
        facts().lazy().filter(col("x") > 3.0).join(items().lazy(), on="item")
    )
    var plain: List[Expr] = [
        col("m").sum().alias("total"),
        col("m").len().alias("n"),
        col("m").count().alias("valid"),
        col("m").mean().alias("avg"),
        col("m").min().alias("low"),
        col("m").max().alias("high"),
        col("x").sum().alias("xs"),
    ]
    var keys: List[String] = ["category", "class"]
    var sets: List[List[String]] = [
        ["category", "class"],
        ["category"],
        ["class"],
        [],
    ]
    check_sets_against_group_by(source, keys, sets, plain)


def test_sets_over_a_left_join_keep_missing_build_rows_apart() raises:
    """A build's string keys group by their codes in every set: the facts
    no item row matches (a null category and class), the items' own null
    classes and the nulls a set leaves for a key it rolls up each land
    in their own groups."""
    var source = (
        facts()
        .lazy()
        .join(
            items().lazy().filter(col("item") < 30),
            on="item",
            how="left",
        )
    )
    var plain: List[Expr] = [
        col("m").sum().alias("total"),
        col("m").len().alias("n"),
        col("x").mean().alias("avg"),
    ]
    var keys: List[String] = ["category", "class"]
    var sets: List[List[String]] = [["category", "class"], ["category"], []]
    check_sets_against_group_by(source, keys, sets, plain)


def test_sets_of_one_builds_packed_keys_match_group_by() raises:
    """Three string keys of one build are coded and packed into one word,
    a field each: a set keeps its own keys' fields and masks the others,
    and sets that hold no other's keys (brand alone, category with brand)
    still come from the finest set's groups. The items' null classes are
    a field of zero, one null group in every set that keeps the class. A
    left join, whose missing items would make a null packed key, codes
    the keys apart."""
    var brands = List[String]()
    for i in range(40):
        brands.append("brand " + String(i % 5))
    var dimension = items().with_column(
        Series("brand", Column[String](brands^))
    )
    var plain: List[Expr] = [
        col("m").sum().alias("total"),
        col("m").len().alias("n"),
        col("x").max().alias("high"),
    ]
    var keys: List[String] = ["category", "class", "brand"]
    var sets: List[List[String]] = [
        ["category", "class", "brand"],
        ["category", "class"],
        ["category", "brand"],
        ["class"],
        ["brand"],
        [],
    ]
    for how in ["inner", "left"]:
        var source = (
            facts()
            .lazy()
            .join(
                dimension.lazy().filter(col("item") < 34),
                on="item",
                how=how,
            )
        )
        check_sets_against_group_by(source, keys, sets, plain)
        # A plain group-by packs the keys too; the joined rows grouped
        # once collected are the reference.
        var got = source.group_by(keys).agg(plain).collect()
        var expected = source.collect().group_by(keys).agg(plain)
        assert_frame(got.sort(keys), expected.sort(keys))
        # With a median, every set groups the input rows by its masked key.
        var middle: List[Expr] = [col("m").median().alias("mid")]
        check_sets_against_group_by(source, keys, sets, middle)


def test_sets_with_reductions_the_table_lacks_match_group_by() raises:
    """A median is not an inline state: the sets group the collected input
    one after another, to the same rows."""
    var source = facts().lazy().join(items().lazy(), on="item")
    var plain: List[Expr] = [
        col("m").median().alias("mid"),
        col("m").n_unique().alias("distinct"),
        col("x").sum().alias("xs"),
    ]
    var keys: List[String] = ["category", "class"]
    var sets: List[List[String]] = [["category", "class"], ["category"], []]
    check_sets_against_group_by(source, keys, sets, plain)


def test_rolled_up_total_row_counts_every_row() raises:
    var plan = (
        facts()
        .lazy()
        .join(items().lazy(), on="item")
        .group_by_rollup(["category", "class"])
        .agg(
            [
                col("m").len().alias("n"),
                grouping_id(["category", "class"]).alias("gid"),
            ]
        )
    )
    var result = plan.collect()
    var total = result.filter(col("gid") == 3)
    assert_equal(total.height(), 1)
    assert_equal(total.item(0, "n").int64(), 300000)
    var by_category = result.filter(col("gid") == 1)
    assert_equal(by_category.height(), 3)
    assert_equal(by_category.select(col("n").sum()).item().int64(), 300000)
