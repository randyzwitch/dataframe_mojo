"""Resident movement and ASOF exports retain exact Arrow field owners."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, Series, col
from dataframe_metal.metal import MetalRuntime
from tests.support.coverage_compare import assert_frame
from tests.public_contracts.test_accelerator_field_metadata import (
    field_source,
    field_row_query,
    field_group_query,
    field_text_source,
    assert_field_owner,
)


# requires: metal
def test_mojo_gpu_field_metadata_rows_groups_empty_and_computed_outputs() raises:
    var original = field_source()
    var runtime = MetalRuntime()
    for empty in [False, True]:
        var frame = original.clear() if empty else original.copy()
        var query = field_row_query(frame)
        var actual = query.collect(accelerator=runtime)
        assert_frame(actual, query.collect(engine="cpu", streaming=False))
        assert_field_owner(actual.column("alias"), original.column("x"))
        assert_field_owner(actual.column("k"), original.column("k"))
        assert_false(Bool(actual.column("computed")._field_metadata))
        query = field_group_query(frame)
        actual = query.collect(accelerator=runtime)
        assert_frame(actual, query.collect(engine="cpu", streaming=False))
        assert_field_owner(actual.column("k"), original.column("k"))
        for name in ["total", "first", "last", "low"]:
            assert_false(Bool(actual.column(name)._field_metadata))
    # A replacement expression deliberately creates a new field, even when
    # it reuses the original public name or computes the same numeric value.
    var replaced = (
        original.lazy().with_columns((col("x") + 0).alias("x")).select(col("x"))
    )
    var output = replaced.collect(accelerator=runtime)
    assert_frame(output, replaced.collect(engine="cpu", streaming=False))
    assert_false(Bool(output.column("x")._field_metadata))


# requires: metal
def test_mojo_gpu_field_metadata_asof_resident_aliases_and_public_export() raises:
    var left = field_source().sort("k")
    var right = (
        field_source()
        .sort("k")
        .select_exprs([col("k"), col("x").alias("right_value")])
    )
    var runtime = MetalRuntime()
    for empty in [False, True]:
        var source = left.clear() if empty else left.copy()
        var query = source.lazy().join_asof(right.lazy(), on="k")
        var actual = query.collect(accelerator=runtime)
        assert_frame(actual, query.collect(engine="cpu", streaming=False))
        assert_equal(actual.height(), 0 if empty else 4)
        assert_field_owner(actual.column("k"), left.column("k"))
        assert_field_owner(actual.column("x"), left.column("x"))
        assert_field_owner(
            actual.column("right_value"), right.column("right_value")
        )
        var consumed = query.select_exprs(
            [col("x").alias("alias"), col("right_value")]
        ).slice(0, 2)
        var measured = consumed.profile(accelerator=runtime)
        assert_frame(
            measured[0], consumed.collect(engine="cpu", streaming=False)
        )
        assert_field_owner(measured[0].column("alias"), left.column("x"))
        assert_field_owner(
            measured[0].column("right_value"), right.column("right_value")
        )
        if not empty:
            assert_true(
                measured[1].item(0, "query_kernel_launches").int64() > 0
            )


# requires: metal
def test_mojo_gpu_field_metadata_dictionary_aliases_asof_and_empty_owners() raises:
    var runtime = MetalRuntime()
    for categorical in [False, True]:
        var original = field_text_source(categorical)
        for empty in [False, True]:
            var frame = original.clear() if empty else original.copy()
            var query = (
                frame.lazy()
                .filter(col("k") >= 0)
                .select_exprs(
                    [
                        col("label").alias("alias"),
                        col("label").str().to_uppercase().alias("computed"),
                    ]
                )
                .sort("alias")
                .slice(0, 3)
            )
            var actual = query.collect(accelerator=runtime)
            assert_frame(actual, query.collect(engine="cpu", streaming=False))
            assert_field_owner(actual.column("alias"), original.column("label"))
            assert_false(Bool(actual.column("computed")._field_metadata))
            var sorted = frame.sort("k")
            var right = original.sort("k").select_exprs(
                [col("k"), col("label").alias("right_label")]
            )
            var joined = (
                sorted.lazy()
                .join_asof(right.lazy(), on="k")
                .select_exprs([col("label").alias("alias"), col("right_label")])
            )
            actual = joined.collect(accelerator=runtime)
            assert_frame(actual, joined.collect(engine="cpu", streaming=False))
            assert_field_owner(actual.column("alias"), original.column("label"))
            assert_field_owner(
                actual.column("right_label"), original.column("label")
            )
