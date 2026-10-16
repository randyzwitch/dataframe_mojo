"""Actual Mojo GPU witnesses: catalog capability `metal` is mandatory.

These tests require Apple Silicon macOS and a working MAX Metal device. Reports
must identify the Mojo implementation and actual kernel submissions.
"""
# contract-group: execution.backends
from dataframe.accelerator.gpu.runtime import _preflight, estimate_resources
from std.memory import bitcast
from std.sys import CompilationTarget
from std.testing import assert_equal, assert_true
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    DataType,
    Series,
    col,
    import_arrow_series,
)
from dataframe.io.arrow import export_arrow_series
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime, metal_installed
from tests.support.lifecycle_stream import array_release, schema_release

comptime NATIVE_NUMERIC = (
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float32,
)


def require_metal() raises -> MetalRuntime:
    assert_true(CompilationTarget.is_macos(), "metal capability requires macOS")
    assert_true(
        metal_installed(), "metal capability requires the Mojo Metal target"
    )
    assert_true(
        MetalRuntime.device_count() > 0,
        "metal capability requires a working Mojo GPU device",
    )
    return MetalRuntime()


def native_value(row: Int) -> Int:
    return row % 9 + 1


def native_valid(row: Int) -> Bool:
    return row % 7 != 2


def native_source[D: DType](n: Int, offset: Int = 7) raises -> DataFrame:
    var values = List[Scalar[D]]()
    var bits = List[Bool]()
    for row in range(offset + n):
        values.append(Scalar[D](native_value(row) if native_valid(row) else 99))
        bits.append(native_valid(row))
    return DataFrame([Series("x", Column[Scalar[D]](values^, bits^))]).slice(
        offset, n
    )


def assert_native_rows[
    D: DType
](actual: DataFrame, n: Int, offset: Int = 7) raises:
    var expected = List[Int]()
    for row in range(offset, offset + n):
        if native_valid(row) and native_value(row) >= 3:
            expected.append(native_value(row) + 2)
    var names: List[String] = ["result"]
    assert_equal(actual.columns(), names)
    assert_equal(actual.height(), len(expected))
    assert_true(actual.column("result").dtype() == DataType.of(D))
    for i in range(len(expected)):
        assert_equal(
            actual.item(i, "result").numeric[D](), Scalar[D](expected[i])
        )


def assert_native_source[D: DType](actual: DataFrame, n: Int) raises:
    assert_equal(actual.height(), n)
    assert_true(actual.column("x").dtype() == DataType.of(D))
    for row in range(n):
        var cell = actual.item(row, "x")
        assert_equal(cell.is_null(), not native_valid(row + 7))
        if native_valid(row + 7):
            assert_equal(cell.numeric[D](), Scalar[D](native_value(row + 7)))


def assert_native_report(report: DataFrame, nonempty: Bool = True) raises:
    assert_true(report.height() > 0)
    var launches = Int64(0)
    var waits = Int64(0)
    for row in range(report.height()):
        assert_equal(report.item(row, "executor").string(), "metal")
        assert_equal(
            report.item(row, "runtime_implementation").string(), "mojo"
        )
        launches += report.item(row, "kernel_launches").int64()
        waits += report.item(row, "synchronizations").int64()
        assert_true(report.item(row, "host_result_bytes").int64() >= 0)
    if nonempty:
        assert_true(
            launches > 0,
            "parity must include actual Mojo GPU kernel submission",
        )
        assert_true(
            waits > 0, "CPU-visible results require Mojo GPU synchronization"
        )


# requires: metal
def test_native_metal_common_numeric_and_boolean_domain_matches_literal_rows_and_cpu_modes() raises:
    var runtime = require_metal()
    comptime for index in range(len(NATIVE_NUMERIC)):
        comptime D = NATIVE_NUMERIC[index]
        for n in [0, 1, 9, 65]:
            var source = native_source[D](n)
            var plan = (
                source.lazy()
                .filter(col("x") >= 3)
                .select((col("x") + 2).alias("result"))
            )
            var native = plan.profile(accelerator=runtime, engine="accel")
            assert_native_rows[D](native[0], n)
            assert_native_report(native[1], n > 1)
            assert_native_rows[D](plan.collect(engine="accel"), n)
            for optimize in [False, True]:
                for streaming in [False, True]:
                    for batch in [1, 7, 64, 257]:
                        assert_native_rows[D](
                            plan.collect(
                                engine="cpu",
                                optimize=optimize,
                                streaming=streaming,
                                batch_size=batch,
                            ),
                            n,
                        )
            assert_native_source[D](source, n)
    var booleans: List[Bool] = [
        False,
        True,
        True,
        False,
        True,
        False,
        True,
        True,
        False,
    ]
    var bits: List[Bool] = [
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    var plan = (
        DataFrame([Series("b", Column[Bool](booleans^, bits^))])
        .lazy()
        .filter(col("b"))
    )
    var native = plan.profile(accelerator=runtime)
    assert_native_report(native[1])
    assert_equal(native[0].height(), 4)
    assert_true(native[0].column("b").dtype() == DataType.BOOL)
    for row in range(4):
        assert_equal(native[0].item(row, "b").bool(), True)


# requires: metal
def test_native_metal_payload_budget_accepts_exact_estimate_and_rejects_one_byte_less() raises:
    var runtime = require_metal()
    var plan = (
        native_source[DType.int64](65)
        .lazy()
        .filter(col("x") >= 3)
        .select((col("x") + 2).alias("result"))
    )
    var estimate = estimate_resources(_preflight(plan))
    assert_true(estimate.peak_bytes > 0)
    var exact = MetalRuntime(memory_limit_bytes=estimate.peak_bytes)
    var executed = plan.profile(accelerator=exact)
    assert_native_rows[DType.int64](executed[0], 65)
    assert_native_report(executed[1])
    assert_equal(
        executed[1].item(0, "peak_requested_bytes").int64(),
        Int64(estimate.peak_bytes),
    )
    for budget in [0, estimate.peak_bytes - 1]:
        var rejected = MetalRuntime(memory_limit_bytes=budget)
        var message = String()
        try:
            _ = plan.collect(accelerator=rejected)
        except error:
            message = String(error)
        assert_true("exceeds memory budget before submission" in message)
    # A failed request must release native resources and leave the runtime usable.
    assert_native_rows[DType.int64](plan.collect(accelerator=exact), 65)


def resident_join_plan() raises -> LazyFrame:
    var left = DataFrame(
        [
            Series("key", Column[Int64]([2, 1, 2, 3])),
            Series("a", Column[Int64]([20, 10, 21, 30])),
        ]
    )
    var right = DataFrame(
        [
            Series("key", Column[Int64]([2, 2, 1, 4])),
            Series("b", Column[Int64]([200, 201, 100, 400])),
        ]
    )
    return (
        left.lazy()
        .filter(col("a") >= 10)
        .join(right.lazy(), "key")
        .sort(["a", "b"])
        .select_exprs([col("key"), (col("a") + col("b")).alias("sum")])
    )


def assert_resident_rows(actual: DataFrame) raises:
    var names: List[String] = ["key", "sum"]
    assert_equal(actual.columns(), names)
    assert_equal(actual.height(), 5)
    var keys: List[Int64] = [1, 2, 2, 2, 2]
    var sums: List[Int64] = [110, 220, 221, 221, 222]
    for row in range(5):
        assert_equal(actual.item(row, "key").int64(), keys[row])
        assert_equal(actual.item(row, "sum").int64(), sums[row])


# requires: metal
def test_native_metal_resident_join_regions_export_complete_owned_typed_results() raises:
    var runtime = require_metal()
    var plan = resident_join_plan()
    var result = plan.profile(accelerator=runtime)
    assert_resident_rows(result[0])
    assert_native_report(result[1])
    var resident_inputs = Int64(0)
    var resident_outputs = Int64(0)
    var joins = 0
    for row in range(result[1].height()):
        resident_inputs += result[1].item(row, "resident_input_bytes").int64()
        resident_outputs += result[1].item(row, "resident_output_bytes").int64()
        if "join" in result[1].item(row, "algorithm").string():
            joins += 1
    assert_equal(joins, 1)
    assert_true(resident_inputs > 0)
    assert_true(resident_outputs > 0)
    for optimize in [False, True]:
        for streaming in [False, True]:
            assert_resident_rows(
                plan.collect(
                    engine="cpu",
                    optimize=optimize,
                    streaming=streaming,
                    batch_size=1,
                )
            )
    var exported = List[Series]()
    for name in result[0].columns():
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_arrow_series(result[0].column(name), array, schema)
        var imported = import_arrow_series(array, schema).renamed(name)
        array_release(array)
        schema_release(schema)
        exported.append(imported^)
    # Destroy the profiled source/result owners before checking exported copies.
    _ = result^
    assert_resident_rows(DataFrame(exported^))


def opaque_word(row: Int) -> UInt64:
    var words: List[UInt64] = [
        0,
        UInt64(1) << 63,
        0x7FF8000000000001,
        0x7FF80000000000AB,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x3FF4000000000000,
    ]
    return words[row % len(words)]


# requires: metal
def test_native_metal_opaque_float64_preserves_ieee_words_through_filter_sort_and_export() raises:
    var runtime = require_metal()
    var values = List[Float64]()
    var keys = List[Int32]()
    var valid = List[Bool]()
    for row in range(72):
        values.append(bitcast[DType.float64](opaque_word(row)))
        keys.append(Int32(row % 3))
        valid.append(row % 5 != 2)
    var source = DataFrame(
        [
            Series("key", Column[Int32](keys^)),
            Series("opaque", Column[Float64](values^, valid^)),
        ]
    ).slice(7, 65)
    var plan = (
        source.lazy().filter(col("key") >= 1).sort("key", descending=True)
    )
    var expected = List[Int]()
    for key in [2, 1]:
        for row in range(7, 72):
            if row % 3 == key:
                expected.append(row)
    var result = plan.profile(accelerator=runtime)
    assert_native_report(result[1])
    var frames: List[DataFrame] = [result[0].copy(), plan.collect(engine="cpu")]
    for frame in frames:
        assert_equal(frame.height(), len(expected))
        assert_true(frame.column("opaque").dtype() == DataType.FLOAT64)
        for i in range(len(expected)):
            var row = expected[i]
            assert_equal(frame.item(i, "key").int32(), Int32(row % 3))
            var cell = frame.item(i, "opaque")
            assert_equal(cell.is_null(), row % 5 == 2)
            if row % 5 != 2:
                assert_equal(
                    bitcast[DType.uint64](cell.float64()), opaque_word(row)
                )
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow_series(result[0].column("opaque"), array, schema)
    var imported = import_arrow_series(array, schema)
    array_release(array)
    schema_release(schema)
    _ = result^
    _ = frames^
    for i in range(len(expected)):
        var row = expected[i]
        assert_equal(imported.get(i).is_null(), row % 5 == 2)
        if row % 5 != 2:
            assert_equal(
                bitcast[DType.uint64](imported.get(i).float64()),
                opaque_word(row),
            )
    # Float64 is opaque supported storage here; arithmetic is rejected by the
    # separate portable preflight contract, without any implicit downcast.
