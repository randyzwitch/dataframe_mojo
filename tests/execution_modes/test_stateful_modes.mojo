"""Fixed plans retain independent semantics across CPU execution modes."""
from std.testing import assert_equal, assert_true
from dataframe import DataFrame, DataType, col
from dataframe.parallel import _helpers
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.reference_model import (
    ReferenceTable,
    generated_model,
    reference_groups,
)
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import (
    assert_model,
    assert_join_model,
    assert_reference_cell,
)
from tests.support.reduction_contract import (
    integer_reductions,
    assert_integer_reductions,
)


def _assert_groups(actual: DataFrame, model: ReferenceTable) raises:
    var groups = reference_groups(model, ["s"])
    assert_equal(
        actual.columns(),
        [
            "s",
            "sum",
            "count",
            "len",
            "min",
            "max",
            "mean",
            "var",
            "std",
            "median",
            "first",
            "last",
            "distinct",
        ],
    )
    assert_equal(actual.height(), len(groups))
    assert_true(actual.column("s").dtype() == DataType.STRING)
    for group in range(len(groups)):
        assert_reference_cell(
            actual.item(group, "s"), model.rows[groups[group][0]], "s"
        )
        assert_integer_reductions(actual, group, model.take(groups[group]))


def test_stateful_groups_and_joins_preserve_complete_rows_across_batches_workers_and_layouts() raises:
    for n in [0, 1, 65]:
        for mode in range(3):
            var left_model = generated_model(n, 3, mode)
            var right_model = generated_model(9, 5, mode)
            for layout in range(3):
                var left = frame_from_model(left_model, 7 if layout == 1 else 0)
                var right = frame_from_model(
                    right_model, 7 if layout == 1 else 0
                )
                if layout == 2:
                    left = chunk_layout(left)
                    right = chunk_layout(right)
                var grouped = (
                    left.lazy()
                    .group_by("s", maintain_order=True)
                    .agg(integer_reductions())
                )
                for workers in [1, 2, 4]:
                    var setting = WorkerSetting(workers)
                    _assert_groups(
                        grouped.collect(
                            optimize=False, streaming=False, engine="cpu"
                        ),
                        left_model,
                    )
                    for batch in [1, 7, 64, 257]:
                        _assert_groups(
                            grouped.collect(
                                optimize=False,
                                streaming=True,
                                batch_size=batch,
                                engine="cpu",
                            ),
                            left_model,
                        )
                        for how in ["inner", "left", "semi", "anti"]:
                            var plan = left.lazy().join(
                                right.lazy(), "s", how=how
                            )
                            assert_join_model(
                                plan.collect(
                                    optimize=False,
                                    streaming=True,
                                    batch_size=batch,
                                    engine="cpu",
                                ),
                                left_model,
                                right_model,
                                ["s"],
                                how,
                            )
                            if batch == 1:
                                assert_join_model(
                                    plan.collect(
                                        optimize=False,
                                        streaming=False,
                                        engine="cpu",
                                    ),
                                    left_model,
                                    right_model,
                                    ["s"],
                                    how,
                                )
                        assert_equal(_helpers().load(), Int64(0))
                    _ = setting^


def test_dispatch_sized_row_pipeline_preserves_order_values_and_observed_streaming_route() raises:
    comptime ROWS = 196609
    var model = generated_model(ROWS, 3, 1)
    var expected = model.double_i().filter_i_gt(0).project(["s", "i"])
    for layout in range(3):
        var source = frame_from_model(model, 7 if layout == 1 else 0)
        if layout == 2:
            source = chunk_layout(source)
        var plan = (
            source.lazy()
            .with_columns((col("i") * 2).alias("i"))
            .filter(col("i") > 0)
            .select(["s", "i"])
        )
        for workers in [1, 2, 4]:
            var setting = WorkerSetting(workers)
            assert_model(
                plan.collect(optimize=False, streaming=False, engine="cpu"),
                expected,
            )
            for batch in [65536, 65537]:
                var result = plan.profile(
                    optimize=False,
                    streaming=True,
                    batch_size=batch,
                    engine="cpu",
                )
                assert_model(result[0], expected)
                var observed = False
                var routes = String()
                for row in range(result[1].height()):
                    var operator = result[1].item(row, "operator").string()
                    var executor = result[1].item(row, "executor").string()
                    routes += operator + ":" + executor + ";"
                    if operator.startswith("FILTER") or operator.startswith(
                        "SELECT"
                    ):
                        observed = (
                            observed
                            or executor == "streaming"
                            or executor == "pipeline"
                        )
                assert_true(
                    observed,
                    "dispatch-sized row plan must report actual streaming route: "
                    + routes,
                )
                assert_equal(_helpers().load(), Int64(0))
            _ = setting^
