"""Grouped floating reductions checked against independent exact references."""
# contract-group: execution.backends
from std.testing import assert_true, assert_equal
from dataframe import DataFrame
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from tests.support.floating_reduction_literals import (
    check_result,
    dyadic_fixture,
    exact_expressions,
    special_fixture,
    special_kind,
    special_expressions,
    check_special,
    null_payload_fixture,
    edge_fixture,
    edge_expressions,
    check_edge,
)


def _collect(query: LazyFrame, gpu: Bool) raises -> DataFrame:
    return query.collect(
        accelerator=MetalRuntime(), engine="accel"
    ) if gpu else query.collect(engine="cpu")


# requires: mojo_gpu
def test_mojo_gpu_grouped_floating_exact_values_layouts_and_nulls() raises:
    for n in [0, 1, 257, 65_537, 131_073]:
        for chunked in [False, True]:
            for narrow in [False, True]:
                var fixture = dyadic_fixture(n, chunked, narrow)
                var query = (
                    fixture[0]
                    .lazy()
                    .group_by("k")
                    .agg(exact_expressions(n + 1))
                )
                for gpu in [False, True]:
                    var result = _collect(query, gpu)
                    var observed = List[Bool](length=6, fill=False)
                    for row in range(result.height()):
                        var key = result.item(row, "k")
                        var group = 5 if key.is_null() else Int(
                            key.numeric[DType.int32]()
                        )
                        assert_true(fixture[3][group] and not observed[group])
                        observed[group] = True
                        var expected_units = Int64(fixture[1][group])
                        var expected_count = Int64(fixture[2][group])
                        check_result(
                            result, row, expected_units, expected_count, narrow
                        )
                    assert_equal(observed, fixture[3])


# requires: mojo_gpu
def test_mojo_gpu_grouped_floating_special_values_and_association() raises:
    for case_index in range(6):
        var frame = special_fixture(case_index)
        var query = frame.lazy().group_by("k").agg(special_expressions())
        for gpu in [False, True]:
            check_special(_collect(query, gpu), special_kind(case_index))


# requires: mojo_gpu
def test_mojo_gpu_grouped_floating_null_payloads_and_min_count() raises:
    for narrow in [False, True]:
        var frame = null_payload_fixture(narrow)
        var query = frame.lazy().group_by("k").agg(exact_expressions(1))
        for gpu in [False, True]:
            check_result(_collect(query, gpu), 0, 0, 0, narrow)


# requires: mojo_gpu
def test_mojo_gpu_grouped_floating_accumulator_width_overflow_and_underflow() raises:
    for case_index in range(5):
        var fixture = edge_fixture(case_index)
        var query = fixture[0].lazy().group_by("k").agg(edge_expressions())
        for gpu in [False, True]:
            check_edge(_collect(query, gpu), fixture[1], fixture[2], fixture[3])
