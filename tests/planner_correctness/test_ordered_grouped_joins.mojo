"""A group-by that keeps its groups in first-appearance order runs over a
join in the aggregate table pipeline and gives the groups in the order the
join's rows come, on any number of workers."""
# contract-group: planner.routing
# contract-group: reduce.grouped
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Expr, LazyFrame, Series, col
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)

comptime _ROWS = 4096


def _source(nulls: Bool) raises -> DataFrame:
    # g takes its 31 values in a scrambled order, and the value 31 only in
    # the last rows, so that group first appears in the last morsel; with
    # `nulls`, every 17th key and every 13th value is null.
    var keys = List[Int64]()
    var groups = List[Int64]()
    var group_valid = List[Bool]()
    var values = List[Int64]()
    var value_valid = List[Bool]()
    var text = List[String]()
    for row in range(_ROWS):
        keys.append(Int64((row * 7) % 97))
        if row >= _ROWS - 8:
            groups.append(Int64(31))
        else:
            groups.append(Int64((row * 19 + 5) % 31))
        group_valid.append(not (nulls and row % 17 == 9))
        values.append(Int64(row))
        value_valid.append(not (nulls and row % 13 == 6))
        text.append("r" + String(row))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("g", Column[Int64](groups^, group_valid)),
            Series("v", Column[Int64](values^, value_valid)),
            Series("s", Column[String](text^)),
        ]
    )


def _dimension() raises -> DataFrame:
    # Every key but a few, once each: the inner join drops some rows, the
    # left join keeps them with a null d.
    var keys = List[Int64]()
    var names = List[String]()
    for key in range(97):
        if key % 10 == 3:
            continue
        keys.append(Int64(key))
        names.append("d" + String(key % 7))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("d", Column[String](names^)),
        ]
    )


def _reductions() -> List[Expr]:
    return [
        col("v").sum().alias("v_sum"),
        col("v").first().alias("v_first"),
        col("v").last().alias("v_last"),
        col("s").first().alias("s_first"),
        col("v").arg_min().alias("v_arg_min"),
        col("v").len().alias("rows"),
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


def _assert_table(query: LazyFrame) raises:
    """The grouping runs in the aggregate table pipeline, not over ordered
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
        paths.find("dataframe-strategy: table_pipeline") >= 0,
        paths,
    )


def test_ordered_groups_over_joins_keep_first_appearance() raises:
    var dimension = _dimension()
    for nulls in [False, True]:
        var frame = _source(nulls)
        for how in ["inner", "left", "semi", "anti"]:
            var keys: List[String] = ["g"]
            if how == "inner" or how == "left":
                keys = ["d", "g"]
            var bys = List[List[String]]()
            bys.append(["g"])
            bys.append(keys.copy())
            for by in bys:
                var query = (
                    frame.lazy()
                    .join(dimension.lazy(), on="k", how=how)
                    .group_by(by, maintain_order=True)
                    .agg(_reductions())
                )
                _assert_table(query)
                _check(
                    query,
                    frame.join(dimension, on="k", how=how)
                    .group_by(by, maintain_order=True)
                    .agg(_reductions()),
                )


def test_ordered_groups_over_filtered_joins() raises:
    var dimension = _dimension()
    var frame = _source(True)
    # Rows dropped from every morsel; rows only in the last morsel; no
    # row. The probe side stays larger than the dimension: an inner join's
    # row order is unspecified, and a smaller one is probed from the other
    # side, in the dimension's order.
    var filters: List[Expr] = [
        col("k") != 14,
        col("v") >= _ROWS - 400,
        col("k") < 0,
    ]
    for predicate in filters:
        var query = (
            frame.lazy()
            .filter(predicate)
            .join(dimension.lazy(), on="k", how="inner")
            .group_by("g", maintain_order=True)
            .agg(_reductions())
        )
        _check(
            query,
            frame.filter(predicate)
            .join(dimension, on="k", how="inner")
            .group_by("g", maintain_order=True)
            .agg(_reductions()),
        )


def test_ordered_groups_over_cross_products() raises:
    # The product's pairs come probe row by probe row, each with every
    # build row, so a group keyed on the build column first appears where
    # its build row first pairs.
    var values = List[Int64]()
    for row in range(5):
        values.append(Int64((row * 3) % 5))
    var build = DataFrame([Series("b", Column[Int64](values^))])
    var frame = _source(True)
    var bys = List[List[String]]()
    bys.append(["b"])
    bys.append(["g", "b"])
    for by in bys:
        var query = (
            frame.lazy()
            .join(build.lazy(), how="cross")
            .filter(col("v") >= 10)
            .group_by(by, maintain_order=True)
            .agg(_reductions())
        )
        _assert_table(query)
        _check(
            query,
            frame.join(build, how="cross")
            .filter(col("v") >= 10)
            .group_by(by, maintain_order=True)
            .agg(_reductions()),
        )


def test_ordered_groups_over_build_row_joins_stay_eager() raises:
    # A right or full join's rows come in its build side's order, which the
    # pipeline does not keep: the grouping runs over the joined frame.
    var dimension = _dimension()
    var frame = _source(True)
    for how in ["right", "full"]:
        var query = (
            frame.lazy()
            .join(dimension.lazy(), on="k", how=how)
            .group_by("d", maintain_order=True)
            .agg(_reductions())
        )
        _check(
            query,
            frame.join(dimension, on="k", how=how)
            .group_by("d", maintain_order=True)
            .agg(_reductions()),
        )
