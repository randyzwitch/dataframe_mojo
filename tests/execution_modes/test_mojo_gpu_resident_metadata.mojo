"""Resident cardinality overrides are validated before creating a device."""
from std.testing import assert_raises, assert_equal, assert_true
from dataframe import DataFrame, DataType, Column, Series, col
from dataframe.accelerator.gpu.runtime import _preflight, capabilities
from dataframe.accelerator.joins import lower_join_dag, DAG_JOIN
from dataframe.accelerator.gpu.join_runtime import preflight as join_preflight


def test_mojo_gpu_resident_cardinality_metadata_bounds() raises:
    var metadata = DataFrame([Series("x", Column[Int64](List[Int64]()))])
    var query = metadata.lazy().select_exprs([col("x")])
    with assert_raises():
        _ = _preflight(query, source_rows=Int(Int32.MAX) + 1)
    with assert_raises():
        _ = _preflight(query, source_rows=-2)
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
