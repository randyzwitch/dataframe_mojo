"""Both-empty joins validate metadata and need no device or retained bytes."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, DataType, Series
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime


def _empty_inputs() raises -> Tuple[DataFrame, DataFrame]:
    return (
        DataFrame(
            [
                Series.full_null("k", DataType.INT32, 0),
                Series.full_null("a", DataType.decimal(20, 2, 128), 0),
            ]
        ),
        DataFrame(
            [
                Series.full_null("k", DataType.INT32, 0),
                Series.full_null("b", DataType.FLOAT64, 0),
            ]
        ),
    )


def _empty_query(
    left: DataFrame, right: DataFrame, how: String, coalesce: Bool
) raises -> LazyFrame:
    if how == "cross":
        return left.lazy().join(right.lazy(), how="cross")
    return left.lazy().join(right.lazy(), on="k", how=how, coalesce=coalesce)


def test_mojo_gpu_both_empty_join_styles_schema_zero_budget_without_device() raises:
    var inputs = _empty_inputs()
    # An invalid device ID proves that neither staging nor a GPU context is
    # needed; zero is the exact requested budget, with no retained payload.
    var runtime = MetalRuntime(999999, memory_limit_bytes=0)
    for how in ["inner", "left", "right", "full", "semi", "anti", "cross"]:
        for coalesce in [False, True]:
            for optimize in [False, True]:
                for limit in [-1, 0, 3]:
                    var query = _empty_query(
                        inputs[0], inputs[1], how, coalesce
                    )
                    if limit >= 0:
                        query = query.head(limit)
                    var measured = query.profile(
                        accelerator=runtime,
                        engine="accel",
                        optimize=optimize,
                        streaming=False,
                    )
                    var expected_names: List[String] = ["k", "a"]
                    var expected_types: List[DataType] = [
                        DataType.INT32,
                        DataType.decimal(20, 2, 128),
                    ]
                    if how != "semi" and how != "anti":
                        if not coalesce or how == "cross":
                            expected_names.append("k_right")
                            expected_types.append(DataType.INT32)
                        expected_names.append("b")
                        expected_types.append(DataType.FLOAT64)
                    assert_equal(measured[0].height(), 0)
                    assert_equal(measured[0].columns(), expected_names)
                    for index in range(len(expected_names)):
                        assert_equal(
                            measured[0].column(expected_names[index]).dtype(),
                            expected_types[index],
                        )
                    var cpu = query.collect(
                        engine="cpu", optimize=optimize, streaming=False
                    )
                    assert_true(measured[0].equals(cpu))
                    assert_true(measured[1].height() > 0)
                    for row in range(measured[1].height()):
                        for counter in [
                            "input_rows",
                            "output_rows",
                            "build_rows",
                            "builds",
                            "peak_requested_bytes",
                            "query_peak_requested_bytes",
                            "upload_bytes",
                            "download_bytes",
                            "kernel_launches",
                            "synchronizations",
                            "query_upload_bytes",
                            "query_download_bytes",
                            "query_kernel_launches",
                            "query_synchronizations",
                            "host_result_bytes",
                            "resident_input_bytes",
                            "resident_output_bytes",
                            "retained_intermediate_bytes",
                            "intermediate_output_bytes",
                        ]:
                            assert_equal(
                                measured[1].item(row, counter).int64(),
                                Int64(0),
                            )


def test_mojo_gpu_both_empty_join_invalid_metadata_precedes_device_setup() raises:
    var inputs = _empty_inputs()
    var runtime = MetalRuntime(999999, memory_limit_bytes=0)
    var wrong_type = DataFrame(
        [
            Series.full_null("k", DataType.INT64, 0),
            Series.full_null("b", DataType.FLOAT64, 0),
        ]
    )
    for optimize in [False, True]:
        with assert_raises(contains="Join key dtypes differ"):
            _ = (
                inputs[0]
                .lazy()
                .join(wrong_type.lazy(), on="k")
                .collect(
                    accelerator=runtime,
                    engine="accel",
                    optimize=optimize,
                    streaming=False,
                )
            )
        with assert_raises(contains="Unknown column: missing"):
            _ = (
                inputs[0]
                .lazy()
                .join(inputs[1].lazy(), left_on=["missing"], right_on=["k"])
                .collect(
                    accelerator=runtime,
                    engine="accel",
                    optimize=optimize,
                    streaming=False,
                )
            )
        with assert_raises(contains="Duplicate join key"):
            _ = (
                inputs[0]
                .lazy()
                .join(inputs[1].lazy(), left_on=["k", "k"], right_on=["k", "k"])
                .collect(
                    accelerator=runtime,
                    engine="accel",
                    optimize=optimize,
                    streaming=False,
                )
            )
        with assert_raises(contains="A cross join takes no keys"):
            _ = (
                inputs[0]
                .lazy()
                .join(
                    inputs[1].lazy(), left_on=["k"], right_on=["k"], how="cross"
                )
                .collect(
                    accelerator=runtime,
                    engine="accel",
                    optimize=optimize,
                    streaming=False,
                )
            )
