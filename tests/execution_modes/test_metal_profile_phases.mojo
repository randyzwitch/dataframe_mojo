"""Host phase attribution survives mixed resident regions and final export."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, col, lit
from dataframe.metal import MetalRuntime
from dataframe.accelerator.gpu.dag_runtime import (
    _append_profile_region,
    _profile_phases,
)


def test_metal_profile_phase_schema_keeps_unknown_measurements_null() raises:
    var measured = DataFrame(
        [
            Series("algorithm", Column[String](["row"])),
            Series("staging_copy_ms", Column[Float64]([1.25])),
        ]
    )
    var unknown = DataFrame([Series("algorithm", Column[String](["join"]))])
    var first = _profile_phases(measured)
    var second = _profile_phases(unknown)
    var rows = first.vstack(second.select(first.columns()))
    assert_equal(rows.item(0, "staging_copy_ms").float64(), Float64(1.25))
    assert_true(rows.item(1, "staging_copy_ms").is_null())
    assert_true(rows.item(0, "readback_ms").is_null())
    assert_equal(rows.column("readback_ms").dtype(), DataType.FLOAT64)
    assert_true(_profile_phases(first).equals(first))


def test_metal_profile_regions_union_optional_metrics_in_first_seen_order() raises:
    var join = DataFrame(
        [
            Series("algorithm", Column[String](["mojo_gpu_native_hash_join"])),
            Series("native_hash_phase_launches", Column[Int64]([7])),
            Series("readback_ms", Column[Float64]([1.5])),
        ]
    )
    var reduction = DataFrame(
        [
            Series("algorithm", Column[String](["mojo_gpu_reductions"])),
            Series("readback_ms", Column[Float64]([0.25])),
        ]
    )
    var empty = DataFrame(List[Series]())
    var first_region = _append_profile_region(empty, join)
    assert_true(first_region.equals(join))

    # Native join first: its phase counter must stay populated on that row
    # and become null on the later reduction region.
    var join_first = _append_profile_region(join, reduction)
    assert_equal(
        join_first.columns(),
        ["algorithm", "native_hash_phase_launches", "readback_ms"],
    )
    assert_equal(
        join_first.item(0, "native_hash_phase_launches").int64(), Int64(7)
    )
    assert_true(join_first.item(1, "native_hash_phase_launches").is_null())
    assert_equal(
        join_first.item(0, "algorithm").string(), "mojo_gpu_native_hash_join"
    )
    assert_equal(
        join_first.item(1, "algorithm").string(), "mojo_gpu_reductions"
    )
    assert_equal(join_first.item(0, "readback_ms").float64(), Float64(1.5))
    assert_equal(join_first.item(1, "readback_ms").float64(), Float64(0.25))
    assert_equal(
        join_first.column("native_hash_phase_launches").dtype(),
        DataType.INT64,
    )

    # A phase-specific field discovered later must be appended in stable
    # first-seen order and leave the earlier row null.
    var reduction_first = _append_profile_region(reduction, join)
    assert_equal(
        reduction_first.columns(),
        ["algorithm", "readback_ms", "native_hash_phase_launches"],
    )
    assert_true(reduction_first.item(0, "native_hash_phase_launches").is_null())
    assert_equal(
        reduction_first.item(1, "native_hash_phase_launches").int64(), Int64(7)
    )
    assert_equal(
        reduction_first.item(0, "algorithm").string(), "mojo_gpu_reductions"
    )
    assert_equal(
        reduction_first.item(1, "algorithm").string(),
        "mojo_gpu_native_hash_join",
    )
    assert_equal(
        reduction_first.item(0, "readback_ms").float64(), Float64(0.25)
    )
    assert_equal(reduction_first.item(1, "readback_ms").float64(), Float64(1.5))
    assert_equal(
        reduction_first.column("native_hash_phase_launches").dtype(),
        DataType.INT64,
    )
    print("PASS profile region schema union and optional metrics")


# requires: mojo_gpu
def test_metal_profile_phases_preserve_results_and_empty_execution() raises:
    for rows in [0, 1, 513]:
        var source = DataFrame(
            [Series("x", Column[Float32](List[Float32](length=rows, fill=2)))]
        )
        var query = source.lazy().select(
            (col("x") * lit(Float32(3)) + lit(Float32(1))).alias("y")
        )
        var runtime = MetalRuntime()
        var profile = query.profile(engine="accel", accelerator=runtime)
        assert_true(profile[0].equals(query.collect(engine="cpu")))
        assert_true(
            profile[0].equals(
                query.collect(engine="accel", accelerator=runtime)
            )
        )
        var wall = profile[1].item(0, "query_wall_ms").float64()
        for name in [
            "preflight_ms",
            "host_prepare_ms",
            "staging_copy_ms",
            "initialization_ms",
            "submit_wait_ms",
            "synchronization_ms",
            "readback_ms",
            "result_copy_ms",
        ]:
            var duration = profile[1].item(0, name).float64()
            assert_true(duration >= 0)
            assert_true(duration <= wall)
        if rows == 0:
            assert_equal(
                profile[1].item(0, "initialization_ms").float64(), Float64(0)
            )
            assert_equal(
                profile[1].item(0, "synchronizations").int64(), Int64(0)
            )


# requires: mojo_gpu
def test_metal_profile_phases_survive_join_between_row_regions() raises:
    var source = DataFrame(
        [
            Series("k", Column[Int32]([1, 2, 2, 3])),
            Series("x", Column[Float32]([1, 2, 3, 4])),
        ]
    )
    var branch = source.lazy().with_columns(
        (col("x") + lit(Float32(1))).alias("x")
    )
    var query = branch.join(branch, "k").with_columns(
        (col("x") + col("x_right")).alias("total")
    )
    var measured = query.profile(engine="accel", accelerator=MetalRuntime())
    assert_true(measured[0].equals(query.collect(engine="cpu")))
    var observed_rows = 0
    var unknown_rows = 0
    var exports = 0
    for row in range(measured[1].height()):
        var staging = measured[1].item(row, "staging_copy_ms")
        if staging.is_null():
            unknown_rows += 1
        else:
            observed_rows += 1
            assert_true(staging.float64() >= 0)
        var readback = measured[1].item(row, "readback_ms")
        if not readback.is_null():
            exports += 1
            assert_true(readback.float64() >= 0)
            assert_true(measured[1].item(row, "result_copy_ms").float64() >= 0)
        assert_true(measured[1].item(row, "query_preflight_ms").float64() >= 0)
    assert_true(observed_rows >= 2)
    assert_true(unknown_rows >= 1)
    assert_equal(exports, 1)


# requires: mojo_gpu
def test_metal_profile_native_join_followed_by_scalar_sum() raises:
    var source = DataFrame(
        [
            Series("k", Column[Int32]([1, 2, 2, 3])),
            Series("x", Column[Int64]([10, 20, 30, 40])),
        ]
    )
    var query = (
        source.lazy()
        .join(source.lazy(), on="k")
        .select_exprs([col("x_right").sum().alias("total")])
    )
    var measured = query.profile(engine="accel", accelerator=MetalRuntime())
    assert_true(
        measured[0].equals(query.collect(engine="cpu", streaming=False))
    )
    assert_equal(measured[0].height(), 1)
    assert_equal(measured[0].item(0, "total").int64(), Int64(150))

    var join_row = -1
    var sum_row = -1
    for row in range(measured[1].height()):
        var operator = measured[1].item(row, "operator").string()
        if operator == "JOIN":
            join_row = row
        if operator == "ROW EXPRESSIONS":
            sum_row = row
    assert_true(join_row >= 0 and sum_row > join_row)
    assert_equal(
        measured[1].item(join_row, "algorithm").string(),
        "mojo_gpu_native_hash_join",
    )
    assert_true(
        measured[1].item(join_row, "native_hash_phase_launches").int64() > 0
    )
    assert_true(
        measured[1].item(sum_row, "native_hash_phase_launches").is_null()
    )
    assert_equal(
        measured[1].item(sum_row, "algorithm").string(), "mojo_gpu_reductions"
    )
    assert_equal(measured[1].item(join_row, "output_rows").int64(), Int64(6))
    assert_equal(measured[1].item(sum_row, "input_rows").int64(), Int64(6))
    assert_equal(
        measured[1].item(join_row, "resident_output_bytes").int64(),
        measured[1].item(sum_row, "resident_input_bytes").int64(),
    )
    assert_equal(
        measured[1].item(sum_row, "materialization_boundary").string(),
        "Mojo GPU -> CPU columns",
    )
    assert_equal(
        measured[1].item(sum_row, "host_result_bytes").int64(), Int64(9)
    )
    assert_equal(
        measured[1].item(0, "query_host_result_bytes").int64(), Int64(9)
    )
    print("PASS native join followed by scalar SUM profile")
