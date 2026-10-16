"""Isolated public fixed-width source Struct query contracts."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    BoolColumn,
    Column,
    CsvField,
    CsvSchema,
    DataFrame,
    DataType,
    Expr,
    Series,
    StructColumn,
    StringColumn,
    col,
    lit,
    when,
    scan_csv,
)
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from dataframe.accelerator.source_structs import lower_source_structs
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.tooling.contract_runner import temporary_directory, run_command


def _source(visible_poison: Bool = False) raises -> DataFrame:
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    var number = Series(
        "n",
        Column[Int64](
            [
                5,
                Int64.MAX,
                Int64.MAX,
                Int64.MAX if visible_poison else Int64(7),
            ],
            [True, True, False, True],
        ),
    )
    number._field_metadata = ArcPointer(metadata^)
    var fields: List[Series] = [
        number^,
        Series("flag", BoolColumn([True, True, False, False])),
        Series(
            "wide", Column[Int128]([1234, Int128.MIN, -5678, 9012])
        ).with_dtype(DataType.decimal(38, 2)),
        Series("time", Column[Int64]([0, Int64.MAX, -1, 1])).with_dtype(
            DataType.datetime("ns", "UTC")
        ),
    ]
    var bits: List[UInt8] = [13]
    var record = Series("record", StructColumn(fields^, bits^))
    var parent_metadata: List[UInt8] = [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        107,
        1,
        0,
        0,
        0,
        112,
    ]
    record._field_metadata = ArcPointer(parent_metadata^)
    return DataFrame(
        [
            Series("id", Column[Int64]([0, 1, 2, 3])),
            record^,
            Series("use", BoolColumn([True, True, False, False])),
        ]
    )


def _expressions() raises -> List[Expr]:
    return [
        col("record").field("n").alias("n"),
        (col("record").field("n") + 1).alias("plus"),
        when(col("use"))
        .then(col("record").field("n") + 1)
        .otherwise(lit(Int64(0)))
        .alias("chosen"),
        col("record").field("wide").alias("wide"),
        col("record").field("time").alias("time"),
        col("record").field("flag").alias("flag"),
    ]


def _check_values(actual: DataFrame, offset: Int) raises:
    assert_equal(actual.height(), 4 - offset)
    assert_equal(
        actual.columns(), ["n", "plus", "chosen", "wide", "time", "flag"]
    )
    assert_equal(actual.column("wide").dtype(), DataType.decimal(38, 2))
    assert_equal(actual.column("time").dtype(), DataType.datetime("ns", "UTC"))
    for local in range(actual.height()):
        var row = local + offset
        assert_equal(actual.item(local, "n").is_null(), row == 1 or row == 2)
        assert_equal(actual.item(local, "plus").is_null(), row == 1 or row == 2)
        assert_equal(actual.item(local, "chosen").is_null(), row == 1)
        if row == 0 or row == 3:
            assert_equal(
                actual.item(local, "n").int64(), Int64(5 if row == 0 else 7)
            )
            assert_equal(
                actual.item(local, "plus").int64(), Int64(6 if row == 0 else 8)
            )
        if row != 1:
            assert_equal(
                actual.item(local, "chosen").int64(),
                Int64(6 if row == 0 else 0),
            )
            assert_equal(
                actual.column("wide")._decimal_raw(local),
                Int128(1234 if row == 0 else -5678 if row == 2 else 9012),
            )
            ref ticks = actual._columns[4]._data[Column[Int64]]
            assert_equal(
                ticks._get(local),
                Int64(0 if row == 0 else -1 if row == 2 else 1),
            )
            assert_equal(actual.column("flag").bool()._get(local), row == 0)
        assert_equal(actual.item(local, "wide").is_null(), row == 1)
        assert_equal(actual.item(local, "time").is_null(), row == 1)
        assert_equal(actual.item(local, "flag").is_null(), row == 1)


# requires: metal
def test_mojo_gpu_source_struct_fields_nullable_layouts_literal_values_and_metadata() raises:
    var original = _source()
    var child = original.column("record").struct_column().field("n")
    var runtime = MetalRuntime()
    for sliced in [False, True]:
        var source = original.slice(1, 3) if sliced else original.copy()
        for variant in range(3):
            var frame = (
                source.copy() if variant
                == 0 else offset_layout(source) if variant
                == 1 else chunk_layout(source)
            )
            var query = frame.lazy().select_exprs(_expressions())
            var actual = query.collect(accelerator=runtime, engine="accel")
            _check_values(actual, 1 if sliced else 0)
            var cpu = query.collect(engine="cpu", streaming=False, batch_size=1)
            _check_values(cpu, 1 if sliced else 0)
            assert_true(actual.equals(cpu))
            assert_true(
                actual.column("n")._field_metadata.value().ptr()
                == child._field_metadata.value().ptr()
            )
            assert_false(Bool(actual.column("plus")._field_metadata))
            assert_false(Bool(actual.column("chosen")._field_metadata))
            var empty_query = frame.clear().lazy().select_exprs(_expressions())
            var empty = empty_query.collect(accelerator=runtime, engine="accel")
            assert_equal(empty.height(), 0)
            assert_equal(empty.dtypes(), actual.dtypes())
            assert_true(
                empty.column("n")._field_metadata.value().ptr()
                == child._field_metadata.value().ptr()
            )
            assert_true(
                empty.equals(
                    empty_query.collect(
                        engine="cpu", streaming=False, batch_size=1
                    )
                )
            )
    # The source's raw hidden poison/metadata remains immutable.
    assert_equal(child.int64()._get(1), Int64.MAX)
    assert_false(child.int64()._valid(2))


# requires: metal
def test_mojo_gpu_source_struct_errors_masks_profile_budget_and_recovery() raises:
    _check_missing_file_prefix_and_error_only_observation()
    var frame = _source(True)
    var bad = frame.lazy().select((col("record").field("n") + 1).alias("bad"))
    for gpu in [False, True]:
        with assert_raises(contains="overflow"):
            if gpu:
                _ = bad.collect(accelerator=MetalRuntime(), engine="accel")
            else:
                _ = bad.collect(engine="cpu", streaming=False, batch_size=1)
    var safe = frame.lazy().select(
        when(col("use"))
        .then(col("record").field("n") + 1)
        .otherwise(lit(Int64(0)))
        .alias("safe")
    )
    var actual = safe.collect(accelerator=MetalRuntime(), engine="accel")
    assert_equal(actual.item(0, "safe").int64(), Int64(6))
    assert_true(actual.item(1, "safe").is_null())
    assert_equal(actual.item(2, "safe").int64(), Int64(0))
    assert_equal(actual.item(3, "safe").int64(), Int64(0))
    assert_true(
        actual.equals(safe.collect(engine="cpu", streaming=False, batch_size=1))
    )
    # Chunk transport owns compact staging; all-valid parents own a new bitmap.
    for all_valid in [False, True]:
        var source = _source()
        if all_valid:
            var structure = source.column("record").struct_column()
            var fields = List[Series]()
            for at in range(structure.field_count()):
                fields.append(structure.field(at))
            source = source.with_column(Series("record", StructColumn(fields^)))
        else:
            source = chunk_layout(source)
        var query = source.lazy().select(
            col("record").field("wide").alias("wide")
        )
        var plan = lower_source_structs(query)
        assert_true(plan.staging_bytes > 0)
        var measured = query.profile(accelerator=MetalRuntime(), engine="accel")
        assert_true(
            measured[1].item(0, "source_struct_staging_bytes").int64()
            == Int64(plan.staging_bytes)
        )
        var peak = Int(
            measured[1].item(0, "query_peak_requested_bytes").int64()
        )
        assert_true(peak > plan.staging_bytes)
        assert_true(
            query.collect(
                accelerator=MetalRuntime(memory_limit_bytes=peak),
                engine="accel",
            ).equals(measured[0])
        )
        var limited = MetalRuntime(memory_limit_bytes=peak - 1)
        with assert_raises(contains="budget"):
            _ = query.collect(accelerator=limited, engine="accel")
        assert_equal(
            DataFrame([Series("x", Column[Int64]([7]))])
            .lazy()
            .select(col("x"))
            .collect(accelerator=limited, engine="accel")
            .item()
            .int64(),
            Int64(7),
        )
        with assert_raises(contains="staging exceeds memory budget"):
            _ = lower_source_structs(
                query, memory_limit_bytes=plan.staging_bytes - 1
            )
    var failed = _source().lazy().select(col("record").field("n"))
    failed._nodes[0].source_failure = "source struct decode fault"
    var rewritten = lower_source_structs(failed)
    assert_equal(
        rewritten.query._nodes[0].source_failure, "source struct decode fault"
    )
    with assert_raises(contains="source struct decode fault"):
        _ = failed.collect(accelerator=MetalRuntime(), engine="accel")


def _source_error(query: LazyFrame, gpu: Bool) raises -> String:
    try:
        if gpu:
            _ = query.collect(accelerator=MetalRuntime(), engine="accel")
        else:
            _ = query.collect(engine="cpu", streaming=False)
    except error:
        return String(error)
    raise Error("Missing-file Struct source fixture unexpectedly succeeded")


def _check_missing_file_prefix_and_error_only_observation() raises:
    var source = _source()
    var expressions: List[Expr] = [
        (col("record").field("n") + 1).alias("plus"),
        col("record").field("wide").alias("wide"),
    ]
    var safe = source.lazy().select_exprs(expressions)
    var observer = MetalRuntime()
    observer._observation_only = True
    var observed = safe.profile(accelerator=observer, engine="accel")
    assert_equal(observed[0].height(), 0)
    assert_equal(observed[0].width(), 2)
    assert_equal(observed[1].item(0, "host_result_bytes").int64(), Int64(0))
    # Only public error words cross to the host. Parent-null MAX and hidden
    # Decimal128 poison are masked on the GPU before this observation.
    assert_equal(
        observed[1].item(0, "download_bytes").int64(), Int64(4 * 2 * 4)
    )
    assert_true(observed[1].item(0, "kernel_launches").int64() > 0)
    var work = temporary_directory("/tmp/mojo-source-struct-missing-prefix")
    var missing = scan_csv(
        work + "/missing.csv", CsvSchema([CsvField("right", DataType.INT64)])
    )
    var good = safe.join(missing, how="cross")
    var original = _source_error(good, False)
    assert_true(original != "")
    assert_false("overflow" in original)
    assert_equal(_source_error(good, True), original)
    var bad = (
        _source(True)
        .lazy()
        .select((col("record").field("n") + 1).alias("plus"))
        .join(missing, how="cross")
    )
    assert_true("overflow" in _source_error(bad, False))
    assert_true("overflow" in _source_error(bad, True))
    run_command(["rm", "-r", work])


# requires: metal
def test_mojo_gpu_source_struct_refusals_and_explanation_remain_metadata_only() raises:
    var frame = _source()
    var runtime = MetalRuntime(device_id=999999)
    var description = runtime.describe(
        frame.lazy().select(col("record").field("n"))
    )
    assert_true("nullable source struct fields" in description)
    assert_false("invalid device" in description)
    var identity = lower_source_structs(
        frame.lazy().select(
            col("record").field("n").cast(DataType.INT64).cast(DataType.INT64)
        ),
        metadata_only=True,
    )
    assert_true(
        identity.prototypes[0]._field_metadata.value().ptr()
        == frame.column("record")
        .struct_column()
        .field("n")
        ._field_metadata.value()
        .ptr()
    )
    for query in [
        frame.lazy().select(col("record")),
        frame.lazy()
        .with_columns(col("record").field("n").alias("n"))
        .select(col("n")),
    ]:
        with assert_raises(contains="source struct"):
            _ = lower_source_structs(query)
    var text = DataFrame(
        [Series("record", StructColumn([Series("s", StringColumn(["x"]))]))]
    )
    with assert_raises(contains="variable-byte"):
        _ = lower_source_structs(text.lazy().select(col("record").field("s")))
    var nested = DataFrame(
        [
            Series(
                "record",
                StructColumn(
                    [
                        Series(
                            "inner",
                            StructColumn([Series("n", Column[Int64]([1]))]),
                        )
                    ]
                ),
            )
        ]
    )
    with assert_raises(contains="primitive outputs"):
        _ = lower_source_structs(
            nested.lazy().select(col("record").field("inner"))
        )


# requires: metal
def test_mojo_gpu_source_struct_identity_cast_field_metadata_and_masks() raises:
    var original = _source()
    var child = original.column("record").struct_column().field("n")
    for sliced in [False, True]:
        var source = original.slice(1, 3) if sliced else original.copy()
        for variant in range(3):
            var frame = (
                source.copy() if variant
                == 0 else offset_layout(source) if variant
                == 1 else chunk_layout(source)
            )
            var field = (
                col("record")
                .field("n")
                .cast(DataType.INT64)
                .cast(DataType.INT64)
            )
            var expressions: List[Expr] = [
                field.copy().alias("identity"),
                (field.copy() + 1).alias("computed"),
                col("record")
                .field("n")
                .cast(DataType.INT32)
                .cast(DataType.INT32)
                .alias("changed"),
                when(col("use"))
                .then(field.copy() + 1)
                .otherwise(lit(Int64(0)))
                .alias("masked"),
            ]
            for empty in [False, True]:
                var input = frame.clear() if empty else frame.copy()
                var query = input.lazy().select_exprs(expressions)
                var prepared = lower_source_structs(query)
                assert_true(
                    prepared.prototypes[0]._field_metadata.value().ptr()
                    == child._field_metadata.value().ptr()
                )
                var gpu = query.collect(
                    accelerator=MetalRuntime(), engine="accel"
                )
                for batch in [1, 64]:
                    var cpu = input.select_exprs(expressions, batch_size=batch)
                    assert_true(gpu.equals(cpu))
                    for actual in [cpu.copy(), gpu.copy()]:
                        assert_equal(
                            actual.height(), 0 if empty else 3 if sliced else 4
                        )
                        assert_true(
                            actual.column("identity")
                            ._field_metadata.value()
                            .ptr()
                            == child._field_metadata.value().ptr()
                        )
                        assert_false(
                            Bool(actual.column("computed")._field_metadata)
                        )
                        assert_false(
                            Bool(actual.column("changed")._field_metadata)
                        )
                        assert_false(
                            Bool(actual.column("masked")._field_metadata)
                        )
                        assert_equal(
                            actual.column("changed").dtype(), DataType.INT32
                        )
                        for row in range(actual.height()):
                            var source_row = row + (1 if sliced else 0)
                            var missing = source_row == 1 or source_row == 2
                            assert_equal(
                                actual.item(row, "identity").is_null(), missing
                            )
                            assert_equal(
                                actual.item(row, "computed").is_null(), missing
                            )
                            assert_equal(
                                actual.item(row, "changed").is_null(), missing
                            )
                            assert_equal(
                                actual.item(row, "masked").is_null(),
                                source_row == 1,
                            )
                            if not missing:
                                var value = Int64(5 if source_row == 0 else 7)
                                assert_equal(
                                    actual.item(row, "identity").int64(), value
                                )
                                assert_equal(
                                    actual.item(row, "computed").int64(),
                                    value + 1,
                                )
                                assert_equal(
                                    actual.item(row, "changed").int32(),
                                    Int32(value),
                                )
                            if source_row != 1:
                                assert_equal(
                                    actual.item(row, "masked").int64(),
                                    Int64(6 if source_row == 0 else 0),
                                )
    # A visible MAX value under an unselected outer mask must not overflow;
    # parent-null and child-null poison rows remain unobserved as well.
    var masked = (
        _source(True)
        .lazy()
        .select(
            when(col("use"))
            .then(
                col("record")
                .field("n")
                .cast(DataType.INT64)
                .cast(DataType.INT64)
                + 1
            )
            .otherwise(lit(Int64(0)))
            .alias("out")
        )
    )
    var actual = masked.collect(accelerator=MetalRuntime(), engine="accel")
    assert_false(Bool(actual.column("out")._field_metadata))
    assert_equal(actual.item(0, "out").int64(), Int64(6))
    assert_true(actual.item(1, "out").is_null())
    assert_equal(actual.item(2, "out").int64(), Int64(0))
    assert_equal(actual.item(3, "out").int64(), Int64(0))
    assert_true(actual.equals(masked.collect(engine="cpu", streaming=False)))
    assert_equal(child.int64()._get(1), Int64.MAX)
    assert_false(child.int64()._valid(2))
