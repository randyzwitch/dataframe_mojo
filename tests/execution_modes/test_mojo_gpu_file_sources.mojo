"""File decoders feed real GPU operators, with observed staging and budgets."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
    scan_csv,
    scan_parquet,
    write_csv,
)
from dataframe.lazy import LazyFrame
from dataframe.decimal import pow10
from dataframe.metal import MetalRuntime
from dataframe.accelerator.sources import file_source_nodes, stage_file_sources
from tests.support.coverage_compare import assert_frame
from tests.tooling.contract_runner import (
    temporary_directory,
    write_text,
    run_command,
)


def _check_profile(query: LazyFrame, scans: Int) raises -> Int:
    var measured = query.profile(engine="accel", accelerator=MetalRuntime())
    assert_frame(measured[0], query.collect(engine="cpu"))
    var report = measured[1].copy()
    var launches = 0
    for row in range(report.height()):
        assert_equal(report.item(row, "executor").string(), "metal")
        assert_equal(report.item(row, "file_scan_count").int64(), Int64(scans))
        assert_true(report.item(row, "source_decode_ms").float64() >= 0)
        assert_true(report.item(row, "source_staging_bytes").int64() > 0)
        launches += Int(report.item(row, "kernel_launches").int64())
    assert_true(launches > 0)
    return Int(report.item(0, "query_peak_requested_bytes").int64())


def test_mojo_gpu_csv_sources_projection_nullable_rows_budget_and_recovery() raises:
    var work = temporary_directory("/tmp/mojo-gpu-file-sources")
    var path = work + "/input.csv"
    write_text(
        path,
        "id,x,bad\n0,10,broken\n1,,broken\n2,30,broken\n3,40,broken\n4,-5,broken\n",
    )
    var schema = CsvSchema(
        [
            CsvField("id", DataType.INT64),
            CsvField("x", DataType.INT64),
            CsvField("bad", DataType.INT64),
        ]
    )
    var query = (
        scan_csv(path, schema)
        .select_exprs([col("id"), col("x")])
        .filter(col("x") >= 0)
        .with_columns((col("x") * 2).alias("y"))
        .select_exprs([col("id"), col("y")])
        .sort("id", descending=True)
        .head(2)
    )
    var peak = _check_profile(query, 1)
    var result = query.collect(
        engine="accel", accelerator=MetalRuntime(memory_limit_bytes=peak)
    )
    assert_equal(result.height(), 2)
    assert_equal(result.item(0, "id").int64(), Int64(3))
    assert_equal(result.item(1, "id").int64(), Int64(2))
    assert_equal(result.item(0, "y").int64(), Int64(80))
    assert_equal(result.item(1, "y").int64(), Int64(60))
    with assert_raises(contains="budget"):
        _ = query.collect(
            engine="accel",
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
        )
    with assert_raises(contains="budget"):
        _ = query.collect(
            engine="accel", accelerator=MetalRuntime(memory_limit_bytes=0)
        )
    assert_frame(
        query.collect(engine="accel", accelerator=MetalRuntime()), result
    )
    var bad = scan_csv(path, schema).select(col("bad"))
    with assert_raises():
        _ = bad.collect(engine="accel", accelerator=MetalRuntime())
    assert_frame(
        query.collect(engine="accel", accelerator=MetalRuntime()), result
    )
    run_command(["rm", "-r", work])


def test_mojo_gpu_csv_sources_dictionary_join_inference_and_metadata() raises:
    var work = temporary_directory("/tmp/mojo-gpu-file-sources")
    var path = work + "/text.csv"
    var input = DataFrame(
        [
            Series("id", Column[Int64]([0, 1, 2, 3])),
            Series("tag", StringColumn(["red", "green", "red", "blue"])),
        ]
    )
    write_csv(input, path)
    var scanned = scan_csv(path)
    var query = (
        scanned.join(input.lazy(), on="id")
        .filter(col("id") > 0)
        .select_exprs([col("id"), col("tag")])
        .sort("id")
    )
    assert_equal(len(file_source_nodes(query)), 1)
    var metadata = stage_file_sources(query, empty=True)
    assert_equal(metadata[2], 1)
    assert_equal(len(file_source_nodes(metadata[0])), 0)
    assert_true(
        "file source decoding"
        in query.explain(engine="accel", accelerator=MetalRuntime())
    )
    _ = _check_profile(query, 1)
    var actual = query.collect(engine="accel", accelerator=MetalRuntime())
    assert_equal(actual.item(0, "tag").string(), "green")
    assert_equal(actual.item(1, "tag").string(), "red")
    assert_equal(actual.item(2, "tag").string(), "blue")
    var both = scan_csv(path).join(scan_csv(path), on="id").select(col("id"))
    _ = _check_profile(both, 2)
    var empty = scan_csv(path).filter(col("id") < 0).select(col("id"))
    assert_equal(
        empty.collect(engine="accel", accelerator=MetalRuntime()).height(), 0
    )
    run_command(["rm", "-r", work])


def test_mojo_gpu_parquet_sources_typed_projection_groups_and_join() raises:
    var source = scan_parquet("tests/fixtures/parquet/plain.parquet")
    var query = (
        source.filter(col("x") >= 2)
        .select_exprs([col("x"), col("s")])
        .sort("x")
    )
    _ = _check_profile(query, 1)
    var frame = query.collect(engine="cpu")
    var joined = source.join(frame.lazy(), on="x").select(col("x")).sort("x")
    _ = _check_profile(joined, 1)
    var grouped = source.group_by("s", maintain_order=True).agg(
        col("x").sum().alias("total")
    )
    _ = _check_profile(grouped, 1)
    var typed = scan_parquet("tests/fixtures/parquet/typed.parquet")
    var narrowed = typed.select(col("i")).filter(col("i") >= 0)
    _ = _check_profile(narrowed, 1)


def test_mojo_gpu_file_sources_observe_left_decimal_failure_before_right_csv_decode() raises:
    var work = temporary_directory("/tmp/mojo-gpu-source-errors")
    var path = work + "/right.csv"
    var schema = CsvSchema([CsvField("right", DataType.INT64)])
    var runtime = MetalRuntime()
    var amount = Int128(6) * pow10(37)
    for dictionary in [False, True]:
        write_text(path, "right\nbroken\n")
        var columns: List[Series] = [
            Series("v", Column[Int128]([amount, amount])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("k", Column[Int64]([0, 0])),
        ]
        var expressions: List[Expr] = [col("v").sum().over("k").alias("total")]
        if dictionary:
            columns.append(Series("label", StringColumn(["z", "a"])))
            expressions.append(col("label"))
        var failing_left = DataFrame(columns^).lazy().select_exprs(expressions)
        var query = failing_left.join(scan_csv(path, schema), how="cross")
        # Metadata-only probes may succeed, but the original left OVER
        # producer must observe its checked precision error before visiting
        # the malformed right source. The dictionary case also runs ordered
        # source encoding before that producer, without changing precedence.
        with assert_raises(contains="decimal precision"):
            _ = query.collect(engine="cpu", streaming=False)
        with assert_raises(contains="decimal precision"):
            _ = query.collect(engine="accel", accelerator=runtime)

        var safe_columns: List[Series] = [
            Series("v", Column[Int128]([amount, -amount])).with_dtype(
                DataType.decimal(38, 0)
            ),
            Series("k", Column[Int64]([0, 0])),
        ]
        if dictionary:
            safe_columns.append(Series("label", StringColumn(["z", "a"])))
        var safe_left = (
            DataFrame(safe_columns^).lazy().select_exprs(expressions)
        )
        var reader_failure = safe_left.join(scan_csv(path, schema), how="cross")
        with assert_raises(contains="CSV integer"):
            _ = reader_failure.collect(engine="cpu", streaming=False)
        with assert_raises(contains="CSV integer"):
            _ = reader_failure.collect(engine="accel", accelerator=runtime)

        # Failure records belong to one execution, so rewriting the input
        # and rerunning the same LazyFrame/runtime must recover normally.
        write_text(path, "right\n7\n")
        var actual = reader_failure.collect(engine="accel", accelerator=runtime)
        var expected = reader_failure.collect(engine="cpu", streaming=False)
        assert_frame(actual, expected)
        assert_equal(actual.height(), 2)
        for row in range(2):
            assert_equal(actual.column("total")._decimal_raw(row), Int128(0))
            assert_equal(actual.item(row, "right").int64(), Int64(7))
    run_command(["rm", "-r", work])


def _source_order_left(failing: Bool, dictionary: Bool) raises -> LazyFrame:
    var amount = Int128(6) * pow10(37)
    var last = amount if failing else -amount
    var columns: List[Series] = [
        Series("v", Column[Int128]([amount, last])).with_dtype(
            DataType.decimal(38, 0)
        ),
        Series("k", Column[Int64]([0, 0])),
    ]
    var expressions: List[Expr] = [
        col("v").sum().over("k").alias("total"),
        col("k"),
    ]
    if dictionary:
        columns.append(Series("label", StringColumn(["z", "a"])))
        expressions.append(col("label"))
    return DataFrame(columns^).lazy().select_exprs(expressions)


def _source_order_error(
    query: LazyFrame, accelerator: Bool, optimize: Bool = True
) raises -> String:
    try:
        if accelerator:
            _ = query.collect(engine="accel", accelerator=MetalRuntime())
        else:
            _ = query.collect(engine="cpu", optimize=optimize, streaming=False)
    except error:
        return String(error)
    raise Error("File error-order fixture unexpectedly succeeded")


def test_mojo_gpu_file_sources_unavailable_schema_preserves_cpu_error_order() raises:
    var work = temporary_directory("/tmp/mojo-gpu-missing-source-errors")
    var path = work + "/missing.csv"
    var schema = CsvSchema([CsvField("right", DataType.INT64)])
    for dictionary in [False, True]:
        var missing = scan_csv(path, schema)
        var failing = _source_order_left(True, dictionary).join(
            missing, how="cross"
        )
        # This root has no planning-time dependency on the right schema.
        # The failed source remains unavailable; evaluating only the earlier
        # complete prefix must still observe its checked decimal producer.
        assert_true("decimal precision" in _source_order_error(failing, False))
        assert_true("decimal precision" in _source_order_error(failing, True))

        var safe = _source_order_left(False, dictionary).join(
            missing, how="cross"
        )
        var original = _source_order_error(safe, False)
        assert_true(original != "")
        assert_true("decimal precision" not in original)
        assert_equal(_source_order_error(safe, True), original)

        # Reusing the unavailable source plan after creating the file must
        # decode it normally, rather than retaining a previous failure.
        write_text(path, "right\n7\n")
        var actual = safe.collect(engine="accel", accelerator=MetalRuntime())
        assert_frame(actual, safe.collect(engine="cpu", streaming=False))
        assert_equal(actual.height(), 2)
        for row in range(2):
            assert_equal(actual.column("total")._decimal_raw(row), Int128(0))
            assert_equal(actual.item(row, "right").int64(), Int64(7))
        run_command(["rm", path])
    run_command(["rm", "-r", work])


def test_mojo_gpu_file_sources_planning_metadata_failure_precedes_gpu_prefix() raises:
    var work = temporary_directory("/tmp/mojo-gpu-source-metadata-errors")
    var path = work + "/missing.csv"
    var raw = _source_order_left(True, True).join(
        scan_csv(path, CsvSchema([CsvField("right", DataType.INT64)])),
        how="cross",
    )
    var projected = raw.select(col("total"))
    var original = _source_order_error(projected, False)
    assert_true(original != "")
    assert_true("decimal precision" not in original)
    assert_equal(_source_order_error(projected, True), original)
    # CPU optimization asks which join input owns the projected name before
    # it evaluates rows. Without optimization its earlier decimal producer
    # fails first; the backend currently receives no optimize flag, so the
    # forced GPU contract here explicitly targets optimize=True.
    assert_true(
        "decimal precision" in _source_order_error(projected, False, False)
    )

    var filtered = (
        _source_order_left(True, True)
        .join(
            scan_csv(path, CsvSchema([CsvField("k", DataType.INT64)])), on="k"
        )
        .filter(col("total") > 0)
    )
    original = _source_order_error(filtered, False)
    assert_true(original != "")
    assert_true("decimal precision" not in original)
    assert_equal(_source_order_error(filtered, True), original)
    assert_true(
        "decimal precision" in _source_order_error(filtered, False, False)
    )
    run_command(["rm", "-r", work])


def test_mojo_gpu_file_sources_prefix_observer_downloads_only_error_words() raises:
    for dictionary in [False, True]:
        var columns: List[Series] = [Series("x", Column[Int64]([1, 2, 3]))]
        var expressions: List[Expr] = [(col("x") + 1).alias("y")]
        if dictionary:
            columns.append(Series("label", StringColumn(["z", "a", "z"])))
            expressions.append(col("label"))
        var query = DataFrame(columns^).lazy().select_exprs(expressions)
        var runtime = MetalRuntime()
        runtime._observation_only = True
        var measured = query.profile(engine="accel", accelerator=runtime)
        assert_equal(measured[0].height(), 0)
        assert_equal(measured[0].width(), len(expressions))
        var report = measured[1].copy()
        assert_equal(report.height(), 1)
        assert_equal(report.item(0, "host_result_bytes").int64(), Int64(0))
        # Three rows, one UInt32 error per public output: neither low/high
        # value planes nor validity/decoded string buffers cross to the CPU.
        assert_equal(
            report.item(0, "download_bytes").int64(),
            Int64(3 * len(expressions) * 4),
        )
        assert_true(report.item(0, "kernel_launches").int64() > 0)
        var expected = query.collect(engine="cpu", streaming=False)
        assert_frame(
            query.collect(engine="accel", accelerator=MetalRuntime()), expected
        )


def test_mojo_gpu_file_prefix_unvisited_dictionary_does_not_hide_earlier_error() raises:
    from tests.public_contracts.test_file_source_prefixes import (
        _prefix_budget_left,
        _prefix_budget_query,
    )

    # Measure the exact successful version of the earlier producer first;
    # error-only observation needs no more resources than its public export.
    var safe_prefix = _prefix_budget_left(False)
    var measured = safe_prefix.profile(accelerator=MetalRuntime())
    var peak = Int(measured[1].item(0, "query_peak_requested_bytes").int64())
    assert_true(peak > 0)
    assert_true(peak < 1048576)
    var runtime = MetalRuntime(memory_limit_bytes=peak)
    var failing = _prefix_budget_query(True)
    with assert_raises(contains="decimal precision"):
        _ = failing.collect(engine="cpu", streaming=False)
    with assert_raises(contains="decimal precision"):
        _ = failing.collect(accelerator=runtime)
    var safe = _prefix_budget_query(False)
    var original = _source_order_error(safe, False)
    var actual = String()
    try:
        _ = safe.collect(accelerator=runtime)
    except error:
        actual = String(error)
    assert_equal(actual, original)
    # Both failure paths must leave the same small runtime reusable.
    assert_frame(safe_prefix.collect(accelerator=runtime), measured[0])
