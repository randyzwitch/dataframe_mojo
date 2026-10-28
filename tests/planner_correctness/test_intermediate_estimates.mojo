"""Intermediate cost estimates are independent from exact heights and bounds."""
# contract-group: planner.estimates
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, StringColumn, col
from dataframe.planning.cost.intermediate import (
    IntermediateEstimates,
    _cost_rows,
    bounded_distinct,
    frame_column_width,
)
from dataframe.planning.graph import plan_height_bound, plan_known_height
from dataframe.kernels.distinct_statistics import TableStatistics
from tests.support.coverage_compare import assert_frame


def test_nested_join_rows_and_alias_domains_propagate_without_execution() raises:
    var fact = DataFrame(
        [
            Series("k", Column[Int64]([1, 1, 2, 2])),
            Series("v", Column[Int64]([10, 20, 30, 40])),
        ]
    )
    var dimension = DataFrame([Series("id", Column[Int64]([1, 2]))])
    var query = fact.lazy().join(
        dimension.lazy(), left_on=["k"], right_on=["id"]
    )
    query = query.select_exprs([col("k").alias("key"), col("v")]).join(
        dimension.lazy(), left_on=["key"], right_on=["id"]
    )
    var index = len(query._plan._nodes) - 1
    var estimates = IntermediateEstimates(len(query._plan._nodes))
    assert_equal(estimates.rows(query._plan, index), 4)
    assert_equal(estimates.column(query._plan, index, "key")[0], 2)
    assert_true(not estimates.facts[index].exact)
    assert_equal(plan_known_height(query._plan, index), -1)
    assert_equal(plan_height_bound(query._plan, index), -1)
    assert_equal(estimates.rows(query._plan, index), 4)


def test_group_filter_slice_and_union_estimates_keep_their_provenance() raises:
    var frame = DataFrame([Series("k", Column[Int64]([1, 1, 2, 2]))])
    var grouped = frame.lazy().group_by(["k"]).agg([col("k").len().alias("n")])
    var index = len(grouped._plan._nodes) - 1
    var estimates = IntermediateEstimates(len(grouped._plan._nodes))
    assert_equal(estimates.rows(grouped._plan, index), 2)
    assert_true(not estimates.facts[index].exact)
    var selected = frame.lazy().filter(col("k") == 1).slice(1, 1)
    index = len(selected._plan._nodes) - 1
    var filtered = IntermediateEstimates(len(selected._plan._nodes))
    assert_equal(filtered.rows(selected._plan, index), 1)
    assert_true(filtered.facts[index].exact)
    var union = grouped.concat(grouped)
    assert_equal(union._estimated_rows(len(union._plan._nodes) - 1), 4)


def test_correlated_join_keys_do_not_multiply_domain_estimates() raises:
    var frame = DataFrame(
        [
            Series("a", Column[Int64]([1, 1, 2, 2])),
            Series("b", Column[Int64]([1, 1, 2, 2])),
        ]
    )
    var query = frame.lazy().join(frame.lazy(), on=["a", "b"])
    # Two correlated predicates describe the same two-value domain.
    assert_equal(query._estimated_rows(len(query._plan._nodes) - 1), 8)


def test_payload_width_retains_source_string_lengths_and_projection_aliases() raises:
    var frame = DataFrame(
        [
            Series("k", Column[Int64]([1, 2])),
            Series("text", StringColumn(["a", "abcdefghijk"])),
        ]
    )
    var query = frame.lazy().select_exprs([col("text").alias("label")])
    var index = len(query._plan._nodes) - 1
    var estimates = IntermediateEstimates(len(query._plan._nodes))
    assert_equal(estimates.payload_width(query._plan, index, ["label"]), 22.0)
    assert_equal(estimates.payload_width(query._plan, index, ["missing"]), 16.0)
    assert_true(_cost_rows(Float64(Int.MAX) * Float64(Int.MAX)) > 0)


def test_unknown_computed_domains_do_not_execute_expressions_for_statistics() raises:
    var frame = DataFrame([Series("k", Column[Int64]([0, 1]))])
    var query = frame.lazy().select_exprs([(col("k") // 0).alias("bad")])
    var index = len(query._plan._nodes) - 1
    var estimates = IntermediateEstimates(len(query._plan._nodes))
    assert_equal(estimates.column(query._plan, index, "bad")[0], -1)
    assert_equal(estimates.payload_width(query._plan, index, ["bad"]), 16.0)


def test_nested_join_choices_preserve_nulls_duplicates_and_output_order() raises:
    var fact = DataFrame(
        [
            Series("k", Column[Int64]([1, 1, 2, 0], [True, True, True, False])),
            Series("v", Column[Int64]([10, 20, 30, 40])),
        ]
    )
    var first = DataFrame([Series("id", Column[Int64]([1, 1, 2]))])
    var second = DataFrame([Series("id2", Column[Int64]([1, 2]))])
    for how in ["inner", "left", "semi", "anti"]:
        var eager = fact.join(first, left_on=["k"], right_on=["id"], how=how)
        var query = fact.lazy().join(
            first.lazy(), left_on=["k"], right_on=["id"], how=how
        )
        var expected = eager.join(
            second, left_on=["k"], right_on=["id2"], how="left"
        )
        query = query.join(
            second.lazy(), left_on=["k"], right_on=["id2"], how="left"
        )
        assert_frame(
            query.collect(engine="cpu", optimize=False, batch_size=2), expected
        )
        assert_frame(query.collect(engine="cpu", batch_size=2), expected)


def test_periodic_input_is_not_sampled_at_one_fixed_phase() raises:
    var values = List[Int64]()
    for row in range(65_536):
        values.append(Int64(0 if row % 16 == 0 else 1))
    assert_equal(bounded_distinct(Series("k", Column[Int64](values^))), 2)


def test_zero_sampled_rows_is_not_used_as_an_empty_relation_proof() raises:
    var values = List[Int64](length=65_536, fill=0)
    var valid = List[Bool](length=65_536, fill=False)
    # The first stratum's selected offset is 10, so this rare valid key
    # is deliberately absent from the cost sample.
    values[0] = 7
    valid[0] = True
    var source = DataFrame([Series("k", Column[Int64](values^, valid^))])
    var build = DataFrame([Series("id", Column[Int64]([7]))])
    var query = source.lazy().join(build.lazy(), left_on=["k"], right_on=["id"])
    var index = len(query._plan._nodes) - 1
    assert_equal(query._estimated_rows(index), 0)
    assert_equal(plan_known_height(query._plan, index), -1)
    for optimize in [False, True]:
        var result = query.collect(engine="cpu", optimize=optimize)
        assert_equal(result.height(), 1)
        assert_equal(result.item(0, "k").int64(), Int64(7))


def test_observed_intermediate_replaces_only_its_cost_facts() raises:
    var frame = DataFrame([Series("k", Column[Int64]([1, 2]))])
    var query = frame.lazy().join(frame.lazy().filter(col("k") > 1), on=["k"])
    var index = len(query._plan._nodes) - 1
    var right = query._plan._nodes[index].right
    var estimates = IntermediateEstimates(len(query._plan._nodes))
    estimates.observe(right, frame.filter(col("k") > 1))
    assert_equal(estimates.rows(query._plan, index), 1)
    assert_true(estimates.facts[right].exact)
    assert_equal(estimates.facts[right].reason, "observed_intermediate")
    assert_equal(plan_known_height(query._plan, index), -1)
    var rejected = False
    try:
        estimates.observe(right, frame)
    except:
        rejected = True
    assert_true(rejected)


def test_extreme_negative_slice_offset_does_not_overflow_estimation() raises:
    var frame = DataFrame([Series("k", Column[Int64]([1, 2]))])
    var query = frame.lazy().slice(Int.MIN, 1)
    assert_equal(query._estimated_rows(len(query._plan._nodes) - 1), 1)


def test_bounded_domain_estimates_do_not_grow_with_repeated_population_rows() raises:
    # A fixed domain remains a fixed domain when more rows repeat it. The
    # former distinct-fraction extrapolation estimated 498K for 8,192 keys.
    for n in [100_000, 1_000_000]:
        for domain in [32, 512, 8192]:
            var values = List[Int64](capacity=n)
            for row in range(n):
                values.append(Int64(row % domain))
            var estimate = bounded_distinct(Series("k", Column[Int64](values^)))
            assert_true(estimate >= domain // 2 and estimate <= domain * 2)


def test_bounded_sampling_preserves_fragmented_and_sliced_input_domains() raises:
    var chunks = List[Series]()
    for part in range(17):
        var values = List[Int64]()
        for row in range(5000):
            values.append(Int64((part * 5000 + row) % 512))
        chunks.append(Series("k", Column[Int64](values^)))
    var series = Series._from_chunks(chunks)
    var sliced = series.slice(13, 80_003)
    var estimate = bounded_distinct(sliced)
    assert_true(estimate >= 256 and estimate <= 1024)
    assert_true(series.is_chunked())
    assert_true(sliced.is_chunked())
    assert_equal(len(series), 85_000)


def _unknown_estimate(series: Series) raises -> Int:
    return -1


def _failed_estimate(series: Series) raises -> Int:
    raise Error("statistics preparation failed")


def test_cost_statistics_cache_distinguishes_equal_length_views_and_validity() raises:
    var column = Column[Int64]([1, 1, 1, 2, 3, 4])
    var table = DataFrame([Series("k", column.slice(0, 3))])._with_statistics()
    assert_equal(
        table._cost_distinct_count("k", "test-v1", bounded_distinct), 1
    )
    var other = table.copy()
    other._columns[0] = Series("k", column.slice(3, 3))
    assert_equal(
        other._cost_distinct_count("k", "test-v1", bounded_distinct), 3
    )
    assert_equal(
        table._cost_distinct_count("k", "test-v1", bounded_distinct), 1
    )
    assert_equal(
        other._cost_distinct_count("k", "test-v1", bounded_distinct), 3
    )
    # The bitmap uses the same offset convention as the shared data.
    var mask_owner = Column[Int64](
        [0, 0, 0, 0, 0, 0], [False, False, False, True, False, False]
    )
    var masked = Column[Int64]._from_buffers(
        column._data, mask_owner._bits, 3, 3
    )
    other._columns[0] = Series("k", masked^)
    assert_equal(
        other._cost_distinct_count("k", "test-v1", bounded_distinct), 1
    )
    assert_equal(len(table._statistics.value()[].cost_counts), 1)


def test_cost_statistics_policy_errors_and_entry_limit_remain_explicit() raises:
    var stats = TableStatistics()
    var source = Series("k", Column[Int64]([1, 2, 3]))
    assert_equal(stats.cost_distinct(source, "known", bounded_distinct), 3)
    assert_equal(stats.cost_distinct(source, "unknown", _unknown_estimate), -1)
    var raised = False
    try:
        _ = stats.cost_distinct(source, "failed", _failed_estimate)
    except:
        raised = True
    assert_true(raised)
    assert_equal(stats.cost_distinct(source, "failed", bounded_distinct), 3)
    for key in range(150):
        var next = Series("k" + String(key), Column[Int64]([1, 2]))
        assert_equal(stats.cost_distinct(next, "known", bounded_distinct), 2)
    assert_true(len(stats.cost_counts) <= 128)


def test_failed_filter_sample_is_unknown_and_execution_still_reports_errors() raises:
    var query = (
        DataFrame([Series("s", StringColumn(["1", "bad"]))])
        .lazy()
        .filter(col("s").cast(DataType.INT64) > 0)
    )
    var index = len(query._plan._nodes) - 1
    var estimates = IntermediateEstimates(len(query._plan._nodes))
    assert_equal(estimates.rows(query._plan, index), 2)
    assert_equal(estimates.facts[index].reason, "unavailable_filter_sample")
    assert_true(not estimates.facts[index].exact)
    var failed = False
    try:
        _ = query.collect(engine="cpu")
    except:
        failed = True
    assert_true(failed)


def test_width_estimates_preserve_sliced_string_views_and_fragmented_storage() raises:
    var short = StringColumn(
        List[String](length=6000, fill="abcdefghij")
    )._as_views()
    var long = StringColumn(
        List[String](length=6000, fill="abcdefghijklmnopqrstuvwxyz1234")
    )._as_views()
    var series = Series._from_chunks(
        [Series("s", short.slice(7, 5000)), Series("s", long.slice(11, 5000))]
    ).slice(17, 9000)
    var expected = 16.0 + (4983.0 * 10.0 + 4017.0 * 30.0) / 9000.0
    var width = frame_column_width(DataFrame([series.copy()]), "s")
    assert_true(abs(width - expected) < 0.02)
    assert_equal(series.n_chunks(), 2)
    assert_equal(series._chunked.value()[].cache[].state.load(), 0)
    assert_equal(frame_column_width(DataFrame([series.slice(0, 0)]), "s"), 16.0)
    # A contiguous offsets column needs only its two endpoint offsets.
    assert_equal(
        frame_column_width(
            DataFrame([Series("s", StringColumn(["abc", "defgh"]))]), "s"
        ),
        20.0,
    )
