"""Batch boundaries and worker policy may not change deterministic results."""
# contract-group: execution.modes
from dataframe import col
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.coverage_generators import generated_frame, chunk_layout
from tests.support.reference_model import generated_model
from tests.support.coverage_compare import assert_model


def test_batches_and_workers_preserve_ordered_transforms() raises:
    for workers in [1, 2, 4]:
        var setting = WorkerSetting(workers)
        for n in [0, 1, 9, 65, 4097]:
            for mode in range(3):
                var source = chunk_layout(generated_frame(n, 3, mode))
                var expected = (
                    generated_model(n, 3, mode)
                    .filter_i_gt(0)
                    .project(["s", "i"])
                )
                var plan = source.lazy().filter(col("i") > 0).select(["s", "i"])
                for batch in [1, 7, 64, 257]:
                    assert_model(
                        plan.collect(
                            streaming=True, batch_size=batch, engine="cpu"
                        ),
                        expected,
                    )
                assert_model(
                    plan.collect(streaming=False, engine="cpu"), expected
                )
        _ = setting^
