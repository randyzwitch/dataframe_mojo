"""Opt-in measurements of caller-supplied plans and independent answers.

Build/import the same identified packages as the application. No placement
profile or row cutoff is inferred from a particular benchmark name. Every
Metal sample uses direct execution, so CPU adjudication cannot be recorded
as a GPU win. Failed/unsupported cases remain explicit capture records.
"""
from std.math import isfinite, isinf
from std.time import perf_counter_ns
from dataframe import DataFrame, DataType, LazyFrame
from dataframe.json.value import json_quote
from .calibration import identity_json, pipeline_key
from .accelerator.gpu.query_scope import shared_service
from .handoff import execute_timed, layout_keys, supports


def _check(actual: DataFrame, expected: DataFrame) raises:
    if (
        actual.columns() != expected.columns()
        or actual.height() != expected.height()
    ):
        raise Error(
            "Metal calibration answer shape differs from independent reference"
        )
    for name in expected.columns():
        if actual.column(name).dtype() != expected.column(name).dtype():
            raise Error(
                "Metal calibration answer dtype differs from independent reference"
            )
        for row in range(expected.height()):
            var a = actual.item(row, name)
            var b = expected.item(row, name)
            if a.is_null() != b.is_null():
                raise Error(
                    "Metal calibration answer null differs from independent reference"
                )
            if b.is_null():
                continue
            if b.dtype() == DataType.FLOAT32 or b.dtype() == DataType.FLOAT64:
                var x = (
                    Float64(a.float32()) if b.dtype()
                    == DataType.FLOAT32 else a.float64()
                )
                var y = (
                    Float64(b.float32()) if b.dtype()
                    == DataType.FLOAT32 else b.float64()
                )
                if x != x and y != y:
                    continue
                if x == y:
                    continue
                var tolerance = 1e-5 if b.dtype() == DataType.FLOAT32 else 1e-12
                if (
                    not isfinite(x)
                    or not isfinite(y)
                    or abs(x - y) > tolerance * (1 + abs(y))
                ):
                    raise Error(
                        "Metal calibration floating answer differs from independent reference"
                    )
        # Structural equality scans the column. Checking it inside the row
        # loop makes verification quadratic for grouped integer outputs.
        if (
            expected.column(name).dtype() != DataType.FLOAT32
            and expected.column(name).dtype() != DataType.FLOAT64
            and not actual.column(name).equals(expected.column(name))
        ):
            raise Error(
                "Metal calibration integer answer differs from independent reference"
            )


def _cpu_sample(
    region: LazyFrame, expected: DataFrame, streaming: Bool
) raises -> Int:
    var started = Int(perf_counter_ns())
    var result = region.collect(engine="cpu", streaming=streaming)
    var elapsed = Int(perf_counter_ns()) - started
    _check(result, expected)
    return elapsed


def capture(
    region: LazyFrame,
    expected: DataFrame,
    repetitions: Int = 5,
    reverse: Bool = False,
) raises -> String:
    """One JSONL record; caller supplies independently computed answers."""
    if repetitions < 3:
        raise Error("Metal calibration requires at least three repetitions")
    var identity = identity_json()
    var prefix = '{"schema_version":2,"identity":' + identity
    if not supports(region):
        return prefix + ',"status":"skipped","reason":"unsupported region"}'
    var keys = layout_keys(region)
    var rows = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].height()
    prefix += (
        ',"pipeline_key":'
        + json_quote(pipeline_key(region))
        + ',"key":'
        + json_quote(keys[0])
        + ',"cpu_layout_family":'
        + json_quote(keys[1])
        + ',"rows":'
        + String(rows)
    )
    if rows == 0:
        return (
            prefix
            + ',"status":"skipped","reason":"empty input has no GPU timing"}'
        )
    _ = _cpu_sample(region, expected, True)
    var materialized_error = Optional[String]()
    try:
        _ = _cpu_sample(region, expected, False)
    except error:
        materialized_error = String(error)
    var warm: DataFrame
    var first_use = shared_service().cached_generations() == 0
    var started_first = Int(perf_counter_ns())
    try:
        var measured = execute_timed(region)
        warm = measured[0].copy()
    except error:
        return (
            prefix
            + ',"status":"skipped","reason":'
            + json_quote(String(error))
            + "}"
        )
    var first_ns = Int(perf_counter_ns()) - started_first
    _check(warm, expected)
    if first_use:
        prefix += ',"first_accel_ns":' + String(first_ns)
    var cpu = String("[")
    var materialized = String("[")
    var accel = String("[")
    for rep in range(repetitions):
        for trial in range(3):
            var engine = (rep + (2 - trial if reverse else trial)) % 3
            var separator = "," if rep else ""
            if engine == 0:
                cpu += separator + String(_cpu_sample(region, expected, True))
            elif engine == 1:
                if not materialized_error:
                    try:
                        var elapsed = _cpu_sample(region, expected, False)
                        materialized += separator + String(elapsed)
                    except error:
                        materialized_error = String(error)
            else:
                var started = Int(perf_counter_ns())
                var result: DataFrame
                try:
                    var measured = execute_timed(region)
                    result = measured[0].copy()
                except error:
                    return (
                        prefix
                        + ',"status":"skipped","reason":'
                        + json_quote(String(error))
                        + "}"
                    )
                var elapsed = Int(perf_counter_ns()) - started
                _check(result, expected)
                accel += separator + String(elapsed)
    if identity_json() != identity:
        return (
            prefix
            + ',"status":"skipped","reason":"hardware or power identity changed during capture"}'
        )
    var alternative = (
        ',"cpu_materialized_error":'
        + json_quote(
            materialized_error.value()
        ) if materialized_error else ',"cpu_materialized_ns":'
        + materialized
        + "]"
    )
    return (
        prefix
        + ',"status":"measured","verified":true,"direct_gpu":true,"cpu_ns":'
        + cpu
        + "]"
        + alternative
        + ',"accel_ns":'
        + accel
        + "]}"
    )
