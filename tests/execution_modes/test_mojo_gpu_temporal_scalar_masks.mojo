"""Checked scalar temporal branches observe ANY selected row, including row1."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.testing import assert_equal, assert_raises, assert_true, assert_false
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe.metal import MetalRuntime
from dataframe.accelerator.gpu.runtime import _preflight
from dataframe.accelerator.constants import (
    has_literal_text_parsing,
    fold_literal_text_parsing,
)
from dataframe.accelerator.gpu.temporal_descriptors import (
    ROW_TEMPORAL_CAST,
    ROW_TEMPORAL_BINARY,
)


# requires: metal
def test_mojo_gpu_temporal_scalar_faults_any_active_and_empty_rows() raises:
    var dtype = DataType.duration("ns")
    var checked_scalars: List[Expr] = [
        lit(Int64.MAX).cast(DataType.duration("ms")).cast(dtype),
        lit(Int64.MAX).cast(dtype) + lit(Int64(1)).cast(dtype),
    ]
    var inactive = DataFrame([Series("keep", Column[Bool]([False, False]))])
    var second = DataFrame([Series("keep", Column[Bool]([False, True]))])
    for operation_index in range(len(checked_scalars)):
        var expression = (
            when(col("keep"))
            .then(checked_scalars[operation_index].copy())
            .otherwise(lit(Int64(0)).cast(dtype))
            .alias("v")
        )
        var safe_query = inactive.lazy().select(expression)
        # Prove folding reaches a fixed point without evaluating the checked
        # scalar before its row mask. Actual public collection takes this hook.
        var folded = fold_literal_text_parsing(safe_query)
        assert_false(has_literal_text_parsing(folded))
        var safe_plan = _preflight(folded)
        assert_equal(safe_plan.reductions, False)
        var found = False
        for node in range(len(safe_plan.node_dtypes)):
            var op = safe_plan.code[4 * node]
            if operation_index == 0:
                found = found or op == ROW_TEMPORAL_CAST
            else:
                found = found or op == ROW_TEMPORAL_BINARY
        assert_true(found)
        var cpu = safe_query.collect(engine="cpu", streaming=False)
        var gpu = safe_query.collect(accelerator=MetalRuntime(), engine="accel")
        for result in [cpu.copy(), gpu.copy()]:
            assert_equal(result.height(), 2)
            assert_true(result.column("v").dtype() == dtype)
            for row in range(2):
                assert_equal(result.item(row, "v").is_null(), False)
                assert_equal(result.item(row, "v").to_physical(), Int64(0))
        var active_query = second.lazy().select(expression)
        _ = _preflight(active_query)
        with assert_raises(contains="temporal value overflows Int64"):
            _ = active_query.collect(engine="cpu", streaming=False)
        with assert_raises(contains="temporal value overflows Int64"):
            _ = active_query.collect(accelerator=MetalRuntime(), engine="accel")
        # Empty ROWS projections evaluate no branch payloads. This differs
        # from a scalar-only projection, which still produces one row.
        var empty_query = second.clear().lazy().select(expression)
        _ = _preflight(empty_query)
        var empty_cpu = empty_query.collect(engine="cpu", streaming=False)
        var empty_gpu = empty_query.collect(
            accelerator=MetalRuntime(), engine="accel"
        )
        for result in [empty_cpu.copy(), empty_gpu.copy()]:
            assert_equal(result.height(), 0)
            assert_equal(result.width(), 1)
            assert_true(result.column("v").dtype() == dtype)


# requires: metal
def test_mojo_gpu_temporal_safe_literal_cast_folding_preserves_dt_fields() raises:
    var source = DataFrame([Series("keep", Column[Bool]([False, True]))])
    var query = source.lazy().select(
        when(col("keep"))
        .then(
            lit(Int64(0))
            .cast(DataType.DATE)
            .cast(DataType.datetime("ns"))
            .dt()
            .year()
        )
        .otherwise(lit(Int64(7)))
        .alias("year")
    )
    assert_true(has_literal_text_parsing(query))
    var folded = fold_literal_text_parsing(query)
    assert_false(has_literal_text_parsing(folded))
    # Retaining an unnecessary temporal CAST here would make the new bounded
    # validator reject this already-supported DT_YEAR composition.
    _ = _preflight(folded)
    var cpu = query.collect(engine="cpu", streaming=False)
    var gpu = query.collect(accelerator=MetalRuntime(), engine="accel")
    for result in [cpu.copy(), gpu.copy()]:
        assert_true(result.column("year").dtype() == DataType.INT64)
        assert_equal(result.height(), 2)
        assert_equal(result.item(0, "year").int64(), Int64(7))
        assert_equal(result.item(1, "year").int64(), Int64(1970))
