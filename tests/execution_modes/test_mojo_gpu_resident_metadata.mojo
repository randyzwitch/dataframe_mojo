"""Resident cardinality overrides are validated before creating a device."""
from std.testing import assert_raises, assert_equal, assert_true
from dataframe import DataFrame, DataType, Column, Series, col
from dataframe.accelerator.gpu.runtime import (
    _preflight,
    capabilities,
    estimate_resources,
)
from dataframe.accelerator.joins import lower_join_dag, DAG_JOIN
from dataframe.accelerator.gpu.join_runtime import preflight as join_preflight


# requires: mojo_gpu
def test_mojo_gpu_resident_cardinality_metadata_bounds() raises:
    var metadata = DataFrame([Series("x", Column[Int64](List[Int64]()))])
    var query = metadata.lazy().select_exprs([col("x")])
    with assert_raises():
        _ = _preflight(query, source_rows=Int(Int32.MAX) + 1)
    with assert_raises():
        _ = _preflight(query, source_rows=-2)
    var empty_plan = _preflight(query, source_rows=0)
    var empty_resident = estimate_resources(empty_plan, resident_input=True)
    var empty_standalone = estimate_resources(empty_plan)
    assert_equal(empty_resident.peak_bytes, 0)
    assert_equal(empty_standalone.peak_bytes, 0)
    assert_equal(empty_resident.upload_bytes, 0)
    assert_equal(empty_resident.download_bytes, 0)
    assert_equal(empty_resident.kernel_launches, 0)
    assert_equal(empty_resident.synchronizations, 0)
    # Positive row counts still reserve both mapping descriptors: three
    # UInt64 words per column on host and device, 48 bytes in total.
    var one_standalone = estimate_resources(empty_plan, rows_override=1)
    var one_resident = estimate_resources(
        empty_plan, rows_override=1, resident_input=True
    )
    assert_equal(one_resident.peak_bytes, one_standalone.peak_bytes + 48)
    var plan = _preflight(query, source_rows=257)
    assert_equal(plan.source.height(), 0)
    var wide_schema = DataFrame(
        [
            Series.full_null("key", DataType.decimal(38, 2), 0),
            Series("value", Column[Int64](List[Int64]())),
        ]
    )
    var wide_query = wide_schema.lazy().join(wide_schema.lazy(), on="key")
    var dag = lower_join_dag(wide_query, capabilities())
    for task in dag.tasks:
        if task.kind == DAG_JOIN:
            var layout = join_preflight(
                task, wide_schema, wide_schema, Int(Int32.MAX), 0
            )
            assert_equal(layout.capacity, Int(Int32.MAX))
            assert_equal(layout.cells, Int(Int32.MAX) * 4)
            assert_true(layout.cells > Int(UInt32.MAX))
