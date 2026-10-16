"""Primitive struct fields execute on GPU before terminal host owner assembly."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.testing import assert_equal, assert_false, assert_raises
from dataframe import Column, DataFrame, Series, as_struct, col, lit
from dataframe.metal import MetalRuntime
from dataframe.accelerator.structs import lower_terminal_structs, assemble
from tests.public_contracts.test_terminal_struct_transport import (
    source_frame,
    expressions,
    check_values,
)
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


# requires: metal
def test_mojo_gpu_terminal_struct_pack_primitive_exports_nullable_children_and_budget() raises:
    var raw = source_frame()
    for layout in range(3):
        var frame = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        for empty in [False, True]:
            var source = frame.clear() if empty else frame.copy()
            var query = source.lazy().select_exprs(expressions())
            var plan = lower_terminal_structs(query)
            var flat = plan.query.collect(
                accelerator=MetalRuntime(), engine="accel"
            )
            var actual = assemble(flat, plan)
            check_values(actual, 0 if empty else 3)
            assert_frame(actual, query.collect(engine="cpu", streaming=False))
            if layout == 0 and not empty:
                var measured = plan.query.profile(accelerator=MetalRuntime())
                var peak = Int(
                    measured[1].item(0, "query_peak_requested_bytes").int64()
                )
                var bounded = plan.query.collect(
                    accelerator=MetalRuntime(memory_limit_bytes=peak),
                    engine="accel",
                )
                assert_frame(assemble(bounded, plan), actual)
                with assert_raises(contains="budget"):
                    _ = plan.query.collect(
                        accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
                        engine="accel",
                    )


# requires: metal
def test_mojo_gpu_terminal_struct_scalar_only_empty_input_keeps_valid_parent() raises:
    var frame = DataFrame([Series("x", Column[Int64]([]))])
    var query = frame.lazy().select(
        as_struct([lit(Int64(9)).alias("n"), lit("z").alias("s")], "record")
    )
    var plan = lower_terminal_structs(query)
    var result = assemble(
        plan.query.collect(accelerator=MetalRuntime(), engine="accel"), plan
    )
    assert_equal(result.height(), 1)
    assert_false(result.item().is_null())
    assert_equal(result.item().struct_field("n").int64(), Int64(9))
    assert_equal(result.item().struct_field("s").string(), "z")
