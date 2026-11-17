"""Cross products, explodes, unnests and joins on keys no hash table holds
run as steps of the morsel pipeline on several workers, with the eager
operators' rows."""
# contract-group: planner.routing
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    LazyFrame,
    ListColumn,
    Series,
    as_struct,
    col,
    lit,
)
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def _probe(rows: Int) raises -> DataFrame:
    """Keys in 13 groups, values with every seventh one null, and text."""
    var keys = List[Int64]()
    var values = List[Int64]()
    var valid = List[Bool]()
    var text = List[String]()
    for row in range(rows):
        keys.append(Int64(row % 13))
        values.append(Int64(row))
        valid.append(row % 7 != 3)
        text.append("p" + String(row))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^, valid)),
            Series("s", Column[String](text^)),
        ]
    )


def _build(rows: Int) raises -> DataFrame:
    """A build side sharing the name `v` with the probe (suffixed in the
    output), every fifth value null."""
    var values = List[Int64]()
    var valid = List[Bool]()
    var labels = List[String]()
    for row in range(rows):
        values.append(Int64(row * 3))
        valid.append(row % 5 != 2)
        labels.append("b" + String(row % 11))
    return DataFrame(
        [
            Series("v", Column[Int64](values^, valid)),
            Series("b", Column[String](labels^)),
        ]
    )


def _sorted(frame: DataFrame) raises -> DataFrame:
    return frame.sort(frame.columns())


def _same_rows(actual: DataFrame, expected: DataFrame) raises:
    """The same rows in any order: a join's output order is unspecified."""
    assert_equal(actual.height(), expected.height())
    assert_equal(actual.columns(), expected.columns())
    for c in range(actual.width()):
        assert_true(
            actual._columns[c].dtype() == expected._columns[c].dtype(),
            actual.columns()[c],
        )
    assert_true(_sorted(actual).equals(_sorted(expected)))


def _collect(query: LazyFrame, optimize: Bool) raises -> DataFrame:
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var result = query.collect(engine="cpu", streaming=True, optimize=optimize)
    _ = threads^
    return result^


def _check(query: LazyFrame, expected: DataFrame) raises:
    _same_rows(_collect(query, False), expected)
    _same_rows(_collect(query, True), expected)


def _assert_pipelined(query: LazyFrame) raises:
    """The query's steps run in a morsel pipeline: no stream evaluates them
    batch by batch."""
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    _ = query.collect(engine="cpu", streaming=True, optimize=False)
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(paths.find("dataframe-path: pipeline.run") >= 0, paths)
    assert_true(paths.find("batch_stream") < 0, paths)


def test_cross_products_materialize_every_pair() raises:
    # Probe and build sizes on both sides of a vector, so each side is in
    # turn the one read a row at a time.
    var sizes: List[Tuple[Int, Int]] = [
        (0, 3),
        (5, 0),
        (1, 1),
        (2500, 3),
        (7, 2100),
        (300, 2100),
    ]
    for size in sizes:
        var probe = _probe(size[0])
        var build = _build(size[1])
        var predicate = col("k") != lit(Int64(4))
        var eager = probe.filter(predicate).join(build, how="cross")
        _check(
            probe.lazy().filter(predicate).join(build.lazy(), how="cross"),
            eager,
        )
        # Steps after the product, reading both sides.
        var after = col("v").is_null() | (col("v_right") > col("v"))
        _check(
            probe.lazy()
            .join(build.lazy(), how="cross")
            .filter(after)
            .select(["s", "b", "v_right"]),
            probe.join(build, how="cross")
            .filter(after)
            .select(["s", "b", "v_right"]),
        )
    _assert_pipelined(_probe(2500).lazy().join(_build(3).lazy(), how="cross"))


def test_cross_products_into_aggregates() raises:
    var probe = _probe(2500)
    var build = _build(2100)
    var grouped = (
        probe.join(build, how="cross")
        .group_by(["k", "b"])
        .agg(
            [
                col("v").sum().alias("sum_v"),
                col("v_right").max().alias("max_r"),
                col("s").count().alias("n"),
            ]
        )
    )
    var query = (
        probe.lazy()
        .join(build.lazy(), how="cross")
        .group_by(["k", "b"])
        .agg(
            [
                col("v").sum().alias("sum_v"),
                col("v_right").max().alias("max_r"),
                col("s").count().alias("n"),
            ]
        )
    )
    _check(query, grouped)
    _assert_pipelined(query)
    # Ungrouped, and over an empty build.
    _check(
        probe.lazy()
        .join(build.lazy(), how="cross")
        .select_exprs([col("v_right").sum().alias("total"), col("s").count()]),
        probe.join(build, how="cross").select_exprs(
            [col("v_right").sum().alias("total"), col("s").count()]
        ),
    )
    _check(
        probe.lazy()
        .join(build.clear().lazy(), how="cross")
        .group_by("k")
        .agg([col("v").sum()]),
        probe.join(build.clear(), how="cross")
        .group_by("k")
        .agg([col("v").sum()]),
    )


def test_cross_products_under_top_n_and_joins() raises:
    var probe = _probe(2500)
    var build = _build(40)
    # Unique sort keys: a top-N's ties are unspecified.
    var top = (
        probe.lazy()
        .join(build.lazy(), how="cross")
        .with_columns(
            [(col("v") * lit(Int64(1000)) + col("v_right")).alias("o")]
        )
        .filter(col("o").is_not_null())
        .sort("o", descending=True)
        .head(25)
    )
    var expected = (
        probe.join(build, how="cross")
        .with_columns(
            [(col("v") * lit(Int64(1000)) + col("v_right")).alias("o")]
        )
        .filter(col("o").is_not_null())
        .sort("o", descending=True)
        .head(25)
    )
    _check(top, expected)
    _assert_pipelined(top)
    # A hash join after the product probes each piece.
    var labels = DataFrame(
        [
            Series("b", Column[String](["b1", "b4", "b9", "zz"])),
            Series("w", Column[Int64]([1, 4, 9, 0])),
        ]
    )
    for how in ["inner", "left"]:
        _check(
            probe.lazy()
            .join(build.lazy(), how="cross")
            .join(labels.lazy(), on="b", how=how),
            probe.join(build, how="cross").join(labels, on="b", how=how),
        )
    # A slice keeps as many rows.
    var sliced = _collect(
        probe.lazy().join(build.lazy(), how="cross").head(4500), False
    )
    assert_equal(sliced.height(), 4500)


def _lists(rows: Int) raises -> DataFrame:
    """Lists of 0 to 3 elements, every ninth list null, some elements
    null."""
    var items = List[Series]()
    var valid = List[Bool]()
    var keys = List[Int64]()
    for row in range(rows):
        var values = List[Int64]()
        var present = List[Bool]()
        for e in range(row % 4):
            values.append(Int64(row + e))
            present.append((row + e) % 5 != 0)
        items.append(Series("item", Column[Int64](values^, present)))
        valid.append(row % 9 != 4)
        keys.append(Int64(row % 6))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("xs", ListColumn.from_lists(items, DataType.INT64, valid)),
        ]
    )


def test_explodes_and_unnests_run_in_the_pipeline() raises:
    for rows in [0, 1, 6000]:
        var frame = _lists(rows)
        var eager = frame.explode("xs").filter(col("k") != lit(Int64(2)))
        _check(
            frame.lazy().explode("xs").filter(col("k") != lit(Int64(2))),
            eager,
        )
        var query = (
            frame.lazy()
            .explode("xs")
            .group_by("k")
            .agg([col("xs").sum().alias("t"), col("xs").count().alias("n")])
        )
        _check(
            query,
            frame.explode("xs")
            .group_by("k")
            .agg([col("xs").sum().alias("t"), col("xs").count().alias("n")]),
        )
        if rows > 0:
            _assert_pipelined(query)
        var records = frame.with_columns(
            [as_struct([col("k"), col("xs")], "r")]
        ).select(["r"])
        _check(
            records.lazy()
            .unnest("r")
            .filter(col("k") > lit(Int64(1)))
            .explode("xs"),
            records.unnest("r").filter(col("k") > lit(Int64(1))).explode("xs"),
        )


def test_struct_key_joins_probe_each_morsel() raises:
    var probe = _probe(5000).with_columns(
        [as_struct([col("k"), (col("k") % lit(Int64(2))).alias("p")], "key")]
    )
    var dimension = (
        _probe(13)
        .select(["k"])
        .filter(col("k") != lit(Int64(5)))
        .with_columns(
            [
                as_struct(
                    [col("k"), (col("k") % lit(Int64(2))).alias("p")], "key"
                ),
                (col("k") * lit(Int64(10))).alias("d"),
            ]
        )
        .select(["key", "d"])
    )
    for how in ["inner", "left", "semi", "anti"]:
        var query = probe.lazy().join(dimension.lazy(), on="key", how=how)
        var expected = probe.join(dimension, on="key", how=how)
        _check(query.drop(["key"]), expected.drop(["key"]))
        _check(
            query.group_by("k").agg([col("v").sum()]),
            expected.group_by("k").agg([col("v").sum()]),
        )
