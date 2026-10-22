"""Mojo cost metadata, native-curve isolation and complete measured execution."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Series, col, lit
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.cost import (
    CalibrationKey,
    select_auto as select_cost,
)
from dataframe_metal.accelerator.gpu.calibration import estimate, measure
from dataframe_metal.accelerator.gpu.identity import key
from dataframe_metal.accelerator.gpu.runtime import (
    _preflight,
    estimate_resources,
)
from tests.support.calibration_fixture import (
    calibration_record,
    calibration_table,
)


def _source(rows: Int, other_values: Bool = False) raises -> DataFrame:
    var values = List[Float32]()
    var valid = List[Bool]()
    for row in range(rows + 6):
        values.append(Float32(-50 if other_values else row % 37))
        valid.append(False if other_values else row % 11 != 0)
    var column = Series("x", Column[Float32](values^, valid^)).slice(3, rows)
    return DataFrame(
        [
            Series._from_chunks(
                [
                    column.slice(0, rows // 2),
                    column.slice(rows // 2, 0),
                    column.slice(rows // 2, rows - rows // 2),
                ]
            )
        ]
    )


# requires: mojo_gpu
def test_mojo_gpu_calibration_metadata() raises:
    var source = _source(257)
    var query = (
        source.lazy()
        .filter(col("x") > lit(Float32(0)))
        .select((col("x") * lit(Float32(2))).alias("x"))
    )
    var alternative = (
        _source(257, True)
        .lazy()
        .filter(col("x") > lit(Float32(0)))
        .select((col("x") * lit(Float32(2))).alias("x"))
    )
    var facts = estimate(query)
    var other = estimate(alternative)
    assert_equal(facts.peak_bytes, other.peak_bytes)
    assert_equal(facts.family, other.family)
    assert_equal(facts.placement, "accel")
    assert_true(";mojo-metal-v1;ops=" in facts.family)
    for i in range(len(facts.cpu)):
        assert_equal(facts.cpu[i], other.cpu[i])
        assert_equal(facts.accelerator[i], other.accelerator[i])
    var division = estimate(
        source.lazy().select((col("x") / lit(Float32(2))).alias("x"))
    )
    var multiplication = estimate(
        source.lazy().select((col("x") * lit(Float32(2))).alias("x"))
    )
    assert_true(division.family != multiplication.family)
    var plan = _preflight(query)
    var standalone = estimate_resources(plan)
    var resident = estimate_resources(
        plan, resident_input=True, terminal_boundary=False
    )
    assert_equal(standalone.peak_bytes, facts.peak_bytes)
    assert_true(resident.peak_bytes >= standalone.peak_bytes)
    assert_true(resident.upload_bytes < standalone.upload_bytes)
    assert_true(resident.download_bytes < standalone.download_bytes)
    assert_true(resident.synchronizations >= 2)
    var override = estimate_resources(plan, rows_override=513)
    assert_true(override.peak_bytes > standalone.peak_bytes)
    with assert_raises(contains="override"):
        _ = estimate_resources(plan, rows_override=-2)
    with assert_raises(contains="device index"):
        _ = estimate_resources(plan, rows_override=Int(Int32.MAX) + 1)
    var profiled = query.profile(accelerator=MetalRuntime(), engine="accel")
    assert_true(profiled[0].equals(query.collect(engine="cpu")))
    var report = profiled[1].copy()
    # Metadata retains the generic allocation envelope; native typed routes
    # can use less storage without weakening the preflight budget bound.
    assert_true(
        report.item(0, "peak_requested_bytes").int64()
        <= Int64(facts.peak_bytes)
    )
    var names = List[String](
        [
            "kernel_launches",
            "synchronizations",
            "upload_bytes",
            "download_bytes",
        ]
    )
    for i in range(len(names)):
        assert_true(
            Float64(report.item(0, names[i]).int64())
            <= facts.accelerator[8 + i]
        )
    # An unfiltered native computation has a distinct owner-lifetime drain.
    # Check both cold and reused contexts against the metadata-only bound.
    var projection = source.lazy().select(
        (col("x") * lit(Float32(2))).alias("x")
    )
    var runtime = MetalRuntime()
    for repeat in range(2):
        var projected = projection.profile(accelerator=runtime, engine="accel")
        assert_true(projected[0].equals(projection.collect(engine="cpu")))
        for i in range(len(names)):
            assert_true(
                Float64(projected[1].item(0, names[i]).int64())
                <= multiplication.accelerator[8 + i]
            )
    var reductions = estimate(
        source.lazy().select_exprs(
            [col("x").sum().alias("sum"), col("x").count().alias("count")]
        )
    )
    assert_true(reductions.accelerator[11] >= 42)
    assert_true(reductions.accelerator[11] < facts.accelerator[11])


# requires: mojo_gpu
def test_mojo_gpu_calibration_isolation() raises:
    var facts = estimate(
        _source(17).lazy().select((col("x") * lit(Float32(2))).alias("x"))
    )
    var identity = CalibrationKey(
        "metal",
        "mojo-metal;ordinal=0;test",
        "mojo-metal-v1",
        "cpu-v1",
        "host",
        1,
        True,
        False,
        7,
    )
    var record = calibration_record(cpu=10000, device=1, family=facts.family)
    record = record.replace('"provider":"probe"', '"provider":"metal"')
    record = record.replace(
        '"device":"device"', '"device":"mojo-metal;ordinal=0;test"'
    )
    record = record.replace(
        '"runtime_build_id":"runtime-v1"', '"runtime_build_id":"mojo-metal-v1"'
    )
    var table = calibration_table(record)
    assert_equal(
        select_cost(facts, table, identity, facts.peak_bytes, 100).engine,
        "accel",
    )
    var native = identity.copy()
    native.runtime_build_id = "native-runtime-v1"
    assert_equal(
        select_cost(facts, table, native, facts.peak_bytes, 100).engine, "cpu"
    )
    native = identity.copy()
    native.device = "registry=123;unified=1"
    assert_equal(
        select_cost(facts, table, native, facts.peak_bytes, 100).engine, "cpu"
    )
    var different = facts.copy()
    different.family += ";other-operation"
    assert_equal(
        select_cost(different, table, identity, facts.peak_bytes, 100).engine,
        "cpu",
    )
    assert_equal(
        select_cost(facts, table, identity, facts.peak_bytes - 1, 100).engine,
        "cpu",
    )
    assert_equal(
        select_cost(facts, table, identity, facts.peak_bytes, 201).engine, "cpu"
    )
    native = identity.copy()
    native.runtime_build_id = "development-source-no-calibration"
    assert_equal(
        select_cost(facts, table, native, facts.peak_bytes, 100).engine, "cpu"
    )


# requires: mojo_gpu
def test_mojo_gpu_calibration_measurements() raises:
    var query = (
        _source(257).lazy().select((col("x") * lit(Float32(2))).alias("x"))
    )
    var identity = key(0, True, False, 7)
    var measured = measure(query, MetalRuntime(), identity, 3)
    assert_true(measured.cold_ns > 0)
    assert_equal(len(measured.cpu_ns), 3)
    assert_equal(len(measured.accelerator_ns), 3)
    for i in range(3):
        assert_true(measured.cpu_ns[i] > 0)
        assert_true(measured.accelerator_ns[i] > 0)
    assert_equal(measured.identity.device, identity.device)
    assert_equal(measured.facts.family, estimate(query).family)
    with assert_raises(contains="repetitions"):
        _ = measure(query, MetalRuntime(), identity, 0)
