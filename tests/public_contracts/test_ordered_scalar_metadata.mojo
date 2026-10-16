"""CUT scalar ranks are metadata-only and charge their real device protocol."""
# contract-group: expr.cut
# contract-group: public.apple_parity
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col, lit
from dataframe.accelerator.cuts import lower_cuts
from dataframe.accelerator.gpu import runtime


def test_cut_ordered_scalar_rank_transport_and_resource_metadata() raises:
    var frame = DataFrame([Series("x", Column[Int64]([1, 2]))])
    var scalar = lit(Int64(128)).cast(DataType.INT8).cast(DataType.INT64) + lit(
        Int64(2)
    ).pow(lit(Int64(-1)))
    var query = lower_cuts(
        frame.lazy().select_exprs(
            [
                scalar.alias("checked"),
                col("x").cut([0], ["lower", "upper"]).alias("bin"),
            ]
        )
    )
    var found = 0
    for operation in query._nodes:
        if (
            len(operation.names2) < 3
            or operation.names2[1] != "ordered_scalar_v1"
        ):
            continue
        var region = (
            DataFrame([Series(operation.names2[2], Column[Int8]([0]))])
            .lazy()
            .select_exprs(operation.exprs.copy())
        )
        region._nodes[len(region._nodes) - 1].text = operation.text
        region._nodes[len(region._nodes) - 1].names2 = operation.names2.copy()
        var plan = runtime._preflight(region, terminal_boundary=True)
        var ranks = len(plan.ordered_ranks[])
        if not ranks:
            continue
        found += 1
        assert_true(len(plan.ordered_ranks[]) != 0)
        assert_equal(len(plan.steps), 1)
        assert_equal(len(plan.outputs), 1)
        assert_equal(ranks, len(plan.literals))
        var owner = plan.ordered_ranks.copy()
        assert_equal(
            Int(owner[].unsafe_ptr()), Int(plan.ordered_ranks[].unsafe_ptr())
        )
        with assert_raises(contains="private one-row CUT producer"):
            _ = runtime._preflight(
                region, terminal_boundary=True, source_rows=2
            )
        var ranked = runtime.estimate_resources(plan, 1, True, True, False)
        plan.ordered_ranks = ArcPointer(List[UInt32]())
        var ordinary = runtime.estimate_resources(plan, 1, True, True, False)
        assert_equal(ranked.program_bytes - ordinary.program_bytes, 4 * ranks)
        assert_equal(ranked.peak_bytes - ordinary.peak_bytes, 8 * ranks + 16)
        assert_equal(ranked.upload_bytes - ordinary.upload_bytes, 4 * ranks)
        assert_equal(ranked.download_bytes - ordinary.download_bytes, 16)
        assert_equal(ranked.kernel_launches - ordinary.kernel_launches, 2)
        assert_equal(ranked.synchronizations - ordinary.synchronizations, 2)
    assert_equal(found, 1)
