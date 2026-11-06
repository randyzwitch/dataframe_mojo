"""Physical choices report reasons while preserving independently counted rows."""
# contract-group: planner.routing
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, LazyFrame, Series, col
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def _source(unique: Bool = False) raises -> DataFrame:
    var keys = List[Int64]()
    var text = List[String]()
    var values = List[Int64]()
    for row in range(512):
        var key = row if unique else row % 4
        keys.append(Int64(key))
        text.append("key:" + String(key))
        values.append(Int64(row % 7))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("s", Column[String](text^)),
            Series("v", Column[Int64](values^)),
        ]
    )


def _run(query: LazyFrame, expected_reason: String) raises -> DataFrame:
    var threads = ScopedEnvironment("DATAFRAME_THREADS", "4")
    var setting = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var trace = StrategyTrace()
    var result = query.collect(
        engine="cpu", streaming=True, optimize=False, batch_size=31
    )
    var paths = trace.text()
    _ = trace^
    _ = setting^
    _ = threads^
    assert_true(
        paths.find("dataframe-strategy: " + expected_reason) >= 0, paths
    )
    return result^


def _assert_sums(result: DataFrame, key: String, unique: Bool = False) raises:
    var count = 512 if unique else 4
    var expected = List[Int64](length=count, fill=0)
    for row in range(512):
        expected[row if unique else row % 4] += Int64(row % 7)
    assert_equal(result.height(), count)
    # Count independently by the reported key, avoiding an ordering assumption.
    for row in range(result.height()):
        var group: Int
        if key == "k":
            group = Int(result.column(key).int64().value(row).value())
        else:
            var label = result.column(key).string().value(row)
            group = Int(String(label[byte=4:]))
        assert_equal(result.column("v").int64().value(row), expected[group])


def test_inline_table_eligibility_precedes_cardinality_preferences() raises:
    var frame = _source(True)
    var result = _run(
        frame.lazy().group_by("k").agg(col("v").sum()),
        "table_pipeline: inline_aggregate_states",
    )
    _assert_sums(result, "k", True)


def test_ordered_groupings_take_the_inline_table() raises:
    # maintain_order: the table orders its groups by first row in the
    # finish, so key density no longer picks an insert or collect route.
    for unique in [False, True]:
        var frame = _source(unique)
        var result = _run(
            frame.lazy().group_by("k", maintain_order=True).agg(col("v").sum()),
            "table_pipeline: inline_aggregate_states",
        )
        _assert_sums(result, "k", unique)
        for row in range(result.height()):
            assert_equal(
                result.column("k").int64().value(row).value(), Int64(row)
            )


def test_positional_global_reduction_uses_ordered_batches() raises:
    var frame = _source()
    var result = _run(
        frame.lazy().select(col("v").first()),
        "batch_stream: positional_global_reduction",
    )
    assert_equal(result.height(), 1)
    assert_equal(result.column("v").int64().value(0), Int64(0))


def test_filtered_ordered_string_keys_take_the_inline_table() raises:
    for unique in [False, True]:
        var frame = _source(unique)
        var query = (
            frame.lazy()
            .filter(col("v") >= 0)
            .group_by("s", maintain_order=True)
            .agg(col("v").sum())
        )
        var result = _run(query, "table_pipeline: inline_aggregate_states")
        _assert_sums(result, "s", unique)


def test_ordered_quantiles_take_the_inline_table() raises:
    var frame = _source()
    var result = _run(
        frame.lazy().group_by("k", maintain_order=True).agg(col("v").median()),
        "table_pipeline: inline_aggregate_states",
    )
    assert_equal(result.height(), 4)
    for row in range(result.height()):
        assert_true(result.column("v").float64().value(row) == Float64(3))
