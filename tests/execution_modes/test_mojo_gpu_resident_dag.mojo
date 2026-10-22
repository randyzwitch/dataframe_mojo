"""Computed DAG regions retain device values until their final CPU boundary."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col
from dataframe_metal.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_float64_emulation import same


# requires: mojo_gpu
def test_mojo_gpu_resident_dag_branches_final_export_profile_and_budget() raises:
    var source = DataFrame(
        [
            Series("k", Column[Int64]([1, 2, 2, 3])),
            Series("x", Column[Float64]([10, 20, 21, 30])),
        ]
    )
    var original = source.copy()
    var branch = source.lazy().with_columns((col("x") + 1).alias("x"))
    var query = branch.join(branch, on="k").with_columns(
        (col("x") + col("x_right")).alias("sum")
    )
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    same(measured[0], query.collect(engine="cpu", streaming=False))
    same(source, original)
    assert_equal(measured[0].height(), 6)
    var resident_regions = 0
    var final_regions = 0
    var error_only_regions = 0
    for row in range(measured[1].height()):
        var boundary = (
            measured[1].item(row, "materialization_boundary").string()
        )
        var downloaded = measured[1].item(row, "download_bytes").int64()
        var output_rows = measured[1].item(row, "output_rows").int64()
        if boundary == "Mojo GPU resident table":
            resident_regions += 1
            assert_equal(
                measured[1].item(row, "host_result_bytes").int64(), Int64(0)
            )
            if (
                measured[1].item(row, "algorithm").string()
                == "mojo_gpu_row_program"
            ):
                assert_equal(downloaded, output_rows * 2 * 4)
                error_only_regions += 1
        else:
            assert_equal(boundary, "Mojo GPU -> CPU columns")
            final_regions += 1
            assert_equal(
                downloaded, output_rows * Int64(measured[0].width()) * 21
            )
    assert_true(resident_regions >= 2)
    assert_true(error_only_regions >= 1)
    assert_equal(final_regions, 1)
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    with assert_raises(contains="memory budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    same(
        measured[0],
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ),
    )


# requires: mojo_gpu
def test_mojo_gpu_resident_dag_empty_scalar_identity_and_zero_column_consumers() raises:
    var source = DataFrame([Series("x", Column[Float64](List[Float64]()))])
    var query = (
        source.lazy()
        .select(col("x").sum())
        .with_columns((col("x") + 5).alias("x"))
    )
    var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
    same(measured[0], query.collect(engine="cpu", streaming=False))
    assert_equal(measured[0].height(), 1)
    assert_equal(measured[0].item().float64(), Float64(5))
    assert_equal(measured[1].item(0, "query_download_bytes").int64(), Int64(21))
    assert_equal(measured[1].item(0, "upload_bytes").int64(), Int64(0))
    assert_equal(measured[1].item(0, "download_bytes").int64(), Int64(0))
    assert_equal(
        measured[1].item(0, "resident_output_bytes").int64(), Int64(21)
    )
    var nonempty = DataFrame([Series("k", Column[Int64]([1, 2, 3]))])
    for count in [0, 1, 3]:
        var empty_columns = (
            nonempty.lazy()
            .join(nonempty.lazy(), on="k")
            .select(List[String]())
            .head(count)
        )
        var actual = empty_columns.collect(
            accelerator=MetalRuntime(), engine="accel"
        )
        assert_equal(actual.height(), count)
        assert_equal(actual.width(), 0)
