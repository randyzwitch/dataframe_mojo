"""Rewrite correctness uses the same executor with optimization disabled."""
from dataframe import col
from tests.support.coverage_generators import generated_frame, chunk_layout
from tests.support.reference_model import generated_model
from tests.support.coverage_compare import assert_frame, assert_model


def test_projection_filter_slice_rewrites_preserve_results() raises:
    for n in [0, 1, 9, 65, 257]:
        for mode in range(3):
            var source = chunk_layout(generated_frame(n, 3, mode))
            var plan = (
                source.lazy()
                .with_columns((col("i") * 2).alias("i"))
                .filter(col("i") > 0)
                .select(["s", "i"])
                .slice(1, 7)
            )
            assert_frame(
                plan.collect(optimize=True, streaming=False, engine="cpu"),
                plan.collect(optimize=False, streaming=False, engine="cpu"),
            )
            var expected = (
                generated_model(n, 3, mode)
                .double_i()
                .filter_i_gt(0)
                .project(["s", "i"])
                .slice(1, 7)
            )
            assert_model(
                plan.collect(optimize=True, streaming=False, engine="cpu"),
                expected,
            )
            assert_model(
                plan.collect(optimize=False, streaming=False, engine="cpu"),
                expected,
            )
