"""Ungrouped first, last, arg_min and arg_max run in the morsel pipeline and
give the source-order answer on any number of workers."""
# contract-group: planner.routing
# contract-group: reduce.global
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Expr, LazyFrame, Series, col
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime _ROWS = 4096


def _source(nulls: Bool) raises -> DataFrame:
    # x repeats every 101 rows, so its minimum and maximum tie in every
    # morsel; with `nulls`, the first and last rows and every 13th are null.
    var keys = List[Int64]()
    var values = List[Int64]()
    var x = List[Int64]()
    var f = List[Float64]()
    var text = List[String]()
    var valid = List[Bool]()
    for row in range(_ROWS):
        keys.append(Int64(row % 97))
        values.append(Int64(row))
        valid.append(
            not (nulls and (row == 0 or row == _ROWS - 1 or row % 13 == 6))
        )
        x.append(Int64((row * 37) % 101))
        f.append(Float64((row * 53) % 89) / 4.0)
        text.append("r" + String(row))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^)),
            Series("x", Column[Int64](x^, valid.copy())),
            Series("f", Column[Float64](f^, valid.copy())),
            Series("s", Column[String](text^, valid.copy())),
        ]
    )


def _dimension() raises -> DataFrame:
    # Every key but a few, once each: the inner join drops some rows.
    var keys = List[Int64]()
    var names = List[String]()
    for key in range(97):
        if key % 10 == 3:
            continue
        keys.append(Int64(key))
        names.append("d" + String(key))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("d", Column[String](names^)),
        ]
    )


def _reductions() -> List[Expr]:
    return [
        col("x").first().alias("x_first"),
        col("x").last().alias("x_last"),
        col("s").first().alias("s_first"),
        col("s").last().alias("s_last"),
        col("f").first().alias("f_first"),
        col("f").last().alias("f_last"),
        col("x").arg_min().alias("x_arg_min"),
        col("x").arg_max().alias("x_arg_max"),
        col("f").arg_min().alias("f_arg_min"),
        col("f").arg_max().alias("f_arg_max"),
        col("v").sum().alias("v_sum"),
    ]


def _same(actual: DataFrame, expected: DataFrame) raises:
    assert_equal(actual.height(), expected.height())
    assert_equal(actual.columns(), expected.columns())
    assert_true(
        actual.equals(expected),
        String(actual) + "\n!=\n" + String(expected),
    )


def _check(query: LazyFrame, expected: DataFrame) raises:
    for workers in [1, 3, 8]:
        var threads = ScopedEnvironment("DATAFRAME_THREADS", String(workers))
        for optimize in [False, True]:
            _same(
                query.collect(engine="cpu", streaming=True, optimize=optimize),
                expected,
            )
        _ = threads^


def _assert_pipelined(query: LazyFrame) raises:
    """The reductions run in the morsel pipeline, not over ordered
    batches."""
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    _ = query.collect(engine="cpu", streaming=True, optimize=False)
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        paths.find("dataframe-strategy: morsel_pipeline") >= 0,
        paths,
    )
    assert_true(paths.find("batch_stream") < 0, paths)


def test_positional_reductions_keep_source_order() raises:
    for nulls in [False, True]:
        var frame = _source(nulls)
        var query = frame.lazy().select_exprs(_reductions())
        _assert_pipelined(query)
        _check(query, frame.select_exprs(_reductions()))


def test_positional_reductions_over_filters() raises:
    for nulls in [False, True]:
        var frame = _source(nulls)
        # Sparse rows in every morsel; rows only in the last morsel, so the
        # first comes from the morsel some worker took last; no row at all.
        var filters: List[Expr] = [
            col("k") == 5,
            col("v") >= _ROWS - 40,
            col("v") < 0,
        ]
        for predicate in filters:
            var query = (
                frame.lazy().filter(predicate).select_exprs(_reductions())
            )
            _assert_pipelined(query)
            _check(query, frame.filter(predicate).select_exprs(_reductions()))


def test_positional_reductions_over_joins() raises:
    var dimension = _dimension()
    var reductions = _reductions()
    reductions.append(col("d").first().alias("d_first"))
    reductions.append(col("d").last().alias("d_last"))
    for nulls in [False, True]:
        var frame = _source(nulls)
        var query = (
            frame.lazy()
            .join(dimension.lazy(), on="k", how="inner")
            .select_exprs(reductions)
        )
        _assert_pipelined(query)
        _check(
            query,
            frame.join(dimension, on="k", how="inner").select_exprs(reductions),
        )


def test_positional_reductions_of_empty_input() raises:
    var frame = _source(True).clear()
    var query = frame.lazy().select_exprs(_reductions())
    _check(query, frame.select_exprs(_reductions()))
    var expected = frame.select_exprs(_reductions())
    assert_equal(expected.height(), 1)
    assert_true(expected.item(0, "x_first").is_null())
    assert_true(expected.item(0, "x_arg_min").is_null())


def test_positional_reductions_over_cross_products() raises:
    # The product's pairs come probe row by probe row, each with every
    # build row: ties in x repeat across build rows, so the position of an
    # extreme counts them in that order. Builds smaller and larger than a
    # chunk of probe rows.
    var reductions = _reductions()
    reductions.append(col("b").first().alias("b_first"))
    reductions.append(col("b").last().alias("b_last"))
    reductions.append(col("b").arg_max().alias("b_arg_max"))
    for size in [3, 2500]:
        var values = List[Int64]()
        for row in range(size):
            values.append(Int64((row * 5) % 7))
        var build = DataFrame([Series("b", Column[Int64](values^))])
        var frame = _source(True)
        if size > 3:
            frame = frame.slice(0, 7)
        var filters: List[Expr] = [
            col("v") >= 0,
            (col("b") == 6) & (col("k") > 40),
        ]
        for predicate in filters:
            var query = (
                frame.lazy()
                .join(build.lazy(), how="cross")
                .filter(predicate)
                .select_exprs(reductions)
            )
            _assert_pipelined(query)
            _check(
                query,
                frame.join(build, how="cross")
                .filter(predicate)
                .select_exprs(reductions),
            )
