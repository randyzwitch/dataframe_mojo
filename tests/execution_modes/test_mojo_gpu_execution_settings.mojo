"""Metal receives requested settings while preserving operator batch contracts."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    CsvField,
    CsvSchema,
    col,
    scan_csv,
)
from dataframe.metal import MetalRuntime
from dataframe.accelerator.sources import prepare_file_sources
from tests.execution_modes.test_mojo_gpu_file_sources import (
    _source_order_left,
    _source_order_error,
)
from tests.tooling.contract_runner import (
    temporary_directory,
    write_text,
    run_command,
)


def test_mojo_gpu_settings_control_file_planning_and_preserve_eager_fault_batches() raises:
    var work = temporary_directory("/tmp/mojo-gpu-execution-settings")
    var missing = work + "/missing.csv"
    var query = (
        _source_order_left(True, True)
        .join(
            scan_csv(missing, CsvSchema([CsvField("right", DataType.INT64)])),
            how="cross",
        )
        .select(col("total"))
    )
    # Default optimizer metadata failure precedes the left data error; explicit
    # optimize=False must evaluate the original left producer first.
    var optimized = _source_order_error(query, False)
    assert_true(optimized != "")
    assert_true("decimal precision" not in optimized)
    assert_equal(_source_order_error(query, True), optimized)
    assert_true("decimal precision" in _source_order_error(query, False, False))
    assert_true("decimal precision" in _source_order_error(query, True, False))
    with assert_raises(contains="decimal precision"):
        _ = query.collect(
            engine="accel", optimize=False, streaming=False, batch_size=1
        )
    var path = work + "/fields.csv"
    write_text(path, "x,unused\n7,broken\n")
    var projected = scan_csv(
        path,
        CsvSchema(
            [CsvField("x", DataType.INT64), CsvField("unused", DataType.INT64)]
        ),
    ).select(col("x"))
    assert_equal(
        projected.collect(
            accelerator=MetalRuntime(), optimize=True, streaming=False
        )
        .item()
        .int64(),
        Int64(7),
    )
    for gpu in [False, True]:
        with assert_raises(contains="CSV integer"):
            if gpu:
                _ = projected.collect(
                    accelerator=MetalRuntime(), optimize=False, streaming=False
                )
            else:
                _ = projected.collect(
                    engine="cpu", optimize=False, streaming=False
                )
    # Explicit helper callers retain the old optimized default, and can request
    # the same no-optimization behavior without any GPU initialization.
    var prepared = prepare_file_sources(projected)
    assert_equal(
        prepared.query._frames[len(prepared.query._frames) - 1].item().int64(),
        Int64(7),
    )
    var failed = prepare_file_sources(projected, optimize=False)
    assert_true("CSV integer" in failed.query._nodes[0].source_failure)
    run_command(["rm", "-r", work])
    var frame = DataFrame(
        [
            Series("x", Column[Int64]([128, 0])),
            Series("t", Column[Int64]([0, Int64.MAX])).with_dtype(
                DataType.duration("ms")
            ),
        ]
    )
    var expression = col("t").cast(DataType.duration("ns")).cast(
        DataType.INT64
    ) + col("x").cast(DataType.INT8).cast(DataType.INT64)
    for implode in [False, True]:
        var output = expression.implode() if implode else expression.copy()
        for streaming in [False, True]:
            for batch_size in [1, 2, 65536]:
                with assert_raises(contains="temporal value overflows Int64"):
                    _ = (
                        frame.lazy()
                        .select(output)
                        .collect(
                            accelerator=MetalRuntime(),
                            optimize=False,
                            streaming=streaming,
                            batch_size=batch_size,
                        )
                    )
