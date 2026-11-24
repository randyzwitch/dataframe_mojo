"""A reduction of filtered values in a group-by, `x.filter(p).sum()` or
`x.drop_nulls().first()`, runs in the aggregate table: every row finds its
group, and only the rows the filter keeps update that reduction's state.
The lazy pipeline and the eager group-by both take the table, on several
workers, with string keys, integer keys with nulls (a perfect hash table),
null values and null predicates. Each answer equals a reduction of nulls
where the filter drops (for reductions that skip nulls), or grouping the
filtered frame; a group whose rows are all dropped keeps its row with an
empty reduction's answer; a value that would overflow a sum only on dropped
rows never reaches it. Distinct counts, and correlations or covariances of
two inputs filtered by one predicate, take the table too. A group-by with no expressions is the table's groups
alone."""
# contract-group: reduce.grouped
from std.math import isnan
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from dataframe import (
    Column,
    DataFrame,
    Expr,
    Series,
    col,
    corr,
    cov,
    lit,
    when,
)
from tests.support.aggregate_table_reference import assert_close
from tests.support.coverage_compare import assert_frame
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime ROWS = 100_000


def frame(rows: Int = ROWS) raises -> DataFrame:
    """String key k (group "dead" holds rows the filter always drops),
    integer key g with nulls, values x with nulls, predicate p with nulls,
    and big, whose dropped rows would overflow an Int64 sum."""
    var k = List[String]()
    var g = List[Int64]()
    var g_valid = List[Bool]()
    var x = List[Int64]()
    var x_valid = List[Bool]()
    var p = List[Bool]()
    var p_valid = List[Bool]()
    var big = List[Int64]()
    for row in range(rows):
        var dead = row % 1000 == 999
        k.append("dead" if dead else "k" + String(row % 7))
        g.append(Int64(row % 13))
        g_valid.append(row % 101 != 5)
        x.append(Int64(row % 1000 - 300))
        x_valid.append(row % 17 != 3)
        var keep = row % 5 < 3 and not dead
        p.append(keep)
        p_valid.append(dead or row % 23 != 7)
        var kept = keep and (dead or row % 23 != 7)
        big.append(Int64(row % 10) if kept else Int64.MAX)
    return DataFrame(
        [
            Series("k", Column[String](k^)),
            Series("g", Column[Int64](g^, g_valid^)),
            Series("x", Column[Int64](x^, x_valid^)),
            Series("p", Column[Bool](p^, p_valid^)),
            Series("big", Column[Int64](big^)),
        ]
    )


def filtered() -> Expr:
    return col("x").filter(col("p"))


def nulled() -> Expr:
    return when(col("p")).then(col("x")).end()


def skipping(value: Expr, big: Expr) -> List[Expr]:
    """Reductions that skip nulls, of `value`, and a sum of `big`."""
    return [
        value.sum().alias("s"),
        value.count().alias("c"),
        value.min().alias("lo"),
        value.max().alias("hi"),
        value.mean().alias("m"),
        value.median().alias("md"),
        big.sum().alias("bs"),
    ]


def positional() -> List[Expr]:
    """Reductions that see every kept row, null or not."""
    var f = filtered()
    return [
        f.len().alias("n"),
        f.first().alias("f"),
        f.last().alias("l"),
        f.null_count().alias("nc"),
    ]


def every() -> List[Expr]:
    var all = skipping(filtered(), col("big").filter(col("p")))
    all.extend(positional())
    return all^


def test_pipeline_filters_per_reduction() raises:
    var source = frame()
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var piped = source.lazy().group_by(["k"]).agg(every()).sort(["k"]).collect()
    var text = observed.text()
    _ = observed^
    _ = trace^
    assert_true("dataframe-path: pipeline.group_table\n" in text)
    assert_false("group_by.row_lists" in text)
    assert_equal(piped.height(), 8)
    # Reductions that skip nulls: the filter is nulls where it drops.
    var nulls = (
        source.lazy()
        .group_by(["k"])
        .agg(skipping(nulled(), when(col("p")).then(col("big")).end()))
        .sort(["k"])
        .collect()
    )
    assert_frame(
        piped.select(["k", "s", "c", "lo", "hi", "m", "md", "bs"]), nulls
    )
    # The others: grouping the filtered frame, whose groups are the ones
    # with a kept row.
    var kept = (
        source.filter(col("p"))
        .group_by(["k"])
        .agg(
            [
                col("x").len().alias("n"),
                col("x").first().alias("f"),
                col("x").last().alias("l"),
                col("x").null_count().alias("nc"),
            ]
        )
        .sort(["k"])
    )
    assert_frame(
        piped.filter(col("k") != lit("dead")).select(
            ["k", "n", "f", "l", "nc"]
        ),
        kept,
    )
    # The group whose rows are all dropped: an empty reduction's answers.
    var dead = piped.filter(col("k") == lit("dead"))
    assert_equal(dead.height(), 1)
    assert_equal(String(dead.item(0, "s")), "0")
    assert_equal(String(dead.item(0, "c")), "0")
    assert_equal(String(dead.item(0, "n")), "0")
    assert_equal(String(dead.item(0, "nc")), "0")
    assert_equal(String(dead.item(0, "bs")), "0")
    for name in ["lo", "hi", "m", "md", "f", "l"]:
        assert_true(dead.item(0, name).is_null(), name)
    # The eager group-by takes the table too, with the same answers.
    var trace_eager = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var seen = StrategyTrace()
    var eager = source.group_by(["k"]).agg(every()).sort(["k"])
    var eager_text = seen.text()
    _ = seen^
    _ = trace_eager^
    _ = workers^
    assert_true("pipeline.group_table" in eager_text)
    assert_false("group_by.row_lists" in eager_text)
    assert_frame(eager, piped)


def test_integer_keys_with_nulls() raises:
    var source = frame()
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var piped = (
        source.lazy()
        .group_by(["g"])
        .agg(skipping(filtered(), col("big").filter(col("p"))))
        .sort(["g"])
        .collect()
    )
    var nulls = (
        source.lazy()
        .group_by(["g"])
        .agg(skipping(nulled(), when(col("p")).then(col("big")).end()))
        .sort(["g"])
        .collect()
    )
    _ = workers^
    assert_equal(piped.height(), 14)
    assert_frame(piped, nulls)
    var eager = (
        source.group_by(["g"])
        .agg(skipping(filtered(), col("big").filter(col("p"))))
        .sort(["g"])
    )
    assert_frame(eager, nulls)


def test_every_group_filtered_out() raises:
    var source = frame()
    var never = col("x").filter(col("x") > lit(Int64(10_000)))
    var exprs: List[Expr] = [
        never.sum().alias("s"),
        never.len().alias("n"),
        never.first().alias("f"),
        never.mean().alias("m"),
        never.median().alias("md"),
        never.max().alias("hi"),
    ]
    var piped = (
        source.lazy().group_by(["k"]).agg(exprs.copy()).sort(["k"]).collect()
    )
    var eager = source.group_by(["k"]).agg(exprs).sort(["k"])
    assert_frame(eager, piped)
    assert_equal(piped.height(), 8)
    for row in range(piped.height()):
        assert_equal(String(piped.item(row, "s")), "0")
        assert_equal(String(piped.item(row, "n")), "0")
        for name in ["f", "m", "md", "hi"]:
            assert_true(piped.item(row, name).is_null(), name)


def test_empty_input() raises:
    var source = frame(1000).clear()
    var piped = source.lazy().group_by(["k"]).agg(every()).collect()
    assert_equal(piped.height(), 0)
    assert_equal(piped.width(), 12)
    var eager = source.group_by(["k"]).agg(every())
    assert_equal(eager.height(), 0)
    assert_equal(eager.columns(), piped.columns())


def test_computed_values_and_dropped_nulls() raises:
    var source = frame()
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var positive = col("x") > lit(Int64(0))
    var doubled = col("x") * lit(Int64(2))
    var exprs: List[Expr] = [
        doubled.filter(positive.copy()).sum().alias("s"),
        col("x").drop_nulls().first().alias("f"),
        col("x").drop_nulls().len().alias("n"),
    ]
    var piped = (
        source.lazy().group_by(["k"]).agg(exprs.copy()).sort(["k"]).collect()
    )
    var eager = source.group_by(["k"]).agg(exprs).sort(["k"])
    assert_frame(eager, piped)
    var sums = (
        source.lazy()
        .group_by(["k"])
        .agg([when(positive).then(doubled).end().sum().alias("s")])
        .sort(["k"])
        .collect()
    )
    assert_frame(piped.select(["k", "s"]), sums)
    var present = (
        source.filter(col("x").is_not_null())
        .group_by(["k"])
        .agg([col("x").first().alias("f"), col("x").len().alias("n")])
        .sort(["k"])
    )
    _ = workers^
    assert_frame(piped.select(["k", "f", "n"]), present)


def test_groups_alone() raises:
    var source = frame()
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var eager = source.group_by(["k", "g"]).agg(List[Expr]()).sort(["k", "g"])
    var piped = (
        source.lazy()
        .group_by(["k", "g"])
        .agg(List[Expr]())
        .sort(["k", "g"])
        .collect()
    )
    var text = observed.text()
    _ = observed^
    _ = trace^
    _ = workers^
    assert_false("group_by.row_lists" in text)
    var counted = (
        source.group_by(["k", "g"])
        .agg([col("x").len().alias("n")])
        .sort(["k", "g"])
        .select(["k", "g"])
    )
    var names: List[String] = ["k", "g"]
    assert_equal(eager.columns(), names)
    assert_frame(eager, counted)
    assert_frame(piped, counted)
    var empty = source.clear().group_by(["k"]).agg(List[Expr]())
    assert_equal(empty.height(), 0)
    var key: List[String] = ["k"]
    assert_equal(empty.columns(), key)


def _same_float(actual: Float64, expected: Float64, what: String) raises:
    if isnan(expected):
        assert_true(isnan(actual), what)
    else:
        assert_close(actual, expected, what)


def test_filtered_distinct_counts_and_pairs() raises:
    """A filtered distinct count and a correlation or covariance of two
    inputs filtered by one predicate run in the table too: a distinct count
    counts the values of the kept rows (a null once), a pair reduction pairs
    the kept rows, and a group with no kept row keeps its row (a count of
    0, a NaN correlation, a null covariance)."""
    var source = frame()
    var y = (col("x") * 3 + col("g")).cast("float64")
    var exprs: List[Expr] = [
        filtered().n_unique().alias("u"),
        col("g").filter(col("p")).n_unique().alias("ug"),
        col("x").drop_nulls().n_unique().alias("ud"),
        corr(filtered(), y.filter(col("p"))).alias("r"),
        cov(filtered(), y.filter(col("p"))).alias("v"),
    ]
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var piped = source.lazy().group_by(["k"]).agg(exprs).sort(["k"]).collect()
    var eager = source.group_by(["k"]).agg(exprs).sort(["k"])
    var text = observed.text()
    _ = observed^
    _ = trace^
    _ = workers^
    assert_true("dataframe-path: pipeline.group_table\n" in text)
    assert_false("group_by.row_lists" in text)
    # Parallel float states merge in no fixed order: those agree closely.
    var counts: List[String] = ["k", "u", "ug", "ud"]
    assert_frame(eager.select(counts), piped.select(counts))
    for name in ["r", "v"]:
        for row in range(piped.height()):
            if piped.item(row, name).is_null():
                assert_true(eager.item(row, name).is_null(), name)
                continue
            _same_float(
                eager.item(row, name).float64(),
                piped.item(row, name).float64(),
                name,
            )
    # Grouping the filtered frame gives the groups with a kept row.
    var kept = source.filter(col("p"))
    var expected = (
        kept.group_by(["k"])
        .agg(
            [
                col("x").n_unique().alias("u"),
                col("g").n_unique().alias("ug"),
                corr(col("x"), y.copy()).alias("r"),
                cov(col("x"), y.copy()).alias("v"),
            ]
        )
        .sort(["k"])
    )
    var live = piped.filter(col("k") != lit("dead"))
    assert_frame(
        live.select(["k", "u", "ug"]), expected.select(["k", "u", "ug"])
    )
    for name in ["r", "v"]:
        for row in range(expected.height()):
            _same_float(
                live.item(row, name).float64(),
                expected.item(row, name).float64(),
                name,
            )
    var dropped = (
        source.filter(col("x").is_not_null())
        .group_by(["k"])
        .agg([col("x").n_unique().alias("ud")])
        .sort(["k"])
    )
    assert_frame(piped.select(["k", "ud"]), dropped)
    var dead = piped.filter(col("k") == lit("dead"))
    assert_equal(String(dead.item(0, "u")), "0")
    assert_equal(String(dead.item(0, "ug")), "0")
    # No pairs: a correlation is NaN, a covariance null.
    assert_true(isnan(dead.item(0, "r").float64()))
    assert_true(dead.item(0, "v").is_null())


def test_filtered_distinct_counts_alone_and_ordered() raises:
    """Every reduction a filtered distinct count: the groups come from the
    distinct tables alone, a group with no kept row among them, in first
    row order when asked."""
    var source = frame()
    var exprs: List[Expr] = [filtered().n_unique().alias("u")]
    var nulled_count = (
        source.lazy().group_by(["k"], maintain_order=True).agg(exprs).collect()
    )
    var eager = source.group_by(["k"], maintain_order=True).agg(exprs)
    assert_frame(eager, nulled_count)
    var order = source.unique(["k"], keep="first", maintain_order=True)
    assert_frame(nulled_count.select(["k"]), order.select(["k"]))
    var one_thread = ScopedEnvironment("DATAFRAME_THREADS", "1")
    var serial = source.group_by(["k"], maintain_order=True).agg(exprs)
    _ = one_thread^
    assert_frame(serial, nulled_count)


def test_pairs_filtered_differently_keep_their_error() raises:
    """A correlation whose inputs keep rows by different predicates pairs
    nothing: the grouped evaluator keeps its error."""
    var source = frame(1000)
    with assert_raises(contains="different lengths"):
        _ = source.group_by(["k"]).agg(
            [corr(filtered(), col("g").filter(col("x") > 0)).alias("r")]
        )
