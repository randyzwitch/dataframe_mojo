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
from .calibration import identity_json, shape_key, pipeline_key
from .handoff import execute_timed, region_key, supports
from .region_plan import normalize_region


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
            elif not actual.column(name).equals(expected.column(name)):
                raise Error(
                    "Metal calibration integer answer differs from independent reference"
                )


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
    var normalized = normalize_region(region)
    var selected = normalized.take()
    var key = shape_key(selected, region_key(selected))
    var rows = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].height()
    prefix += (
        ',"pipeline_key":'
        + json_quote(pipeline_key(region))
        + ',"key":'
        + json_quote(key)
        + ',"rows":'
        + String(rows)
    )
    if rows == 0:
        return (
            prefix
            + ',"status":"skipped","reason":"empty input has no GPU timing"}'
        )
    _check(region.collect(engine="cpu"), expected)
    var warm: DataFrame
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
    _check(warm, expected)
    var cpu = String("[")
    var accel = String("[")
    for rep in range(repetitions):
        for engine in range(2):
            var gpu = (engine + rep + Int(reverse)) % 2 == 1
            var started = Int(perf_counter_ns())
            var result: DataFrame
            if gpu:
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
            else:
                result = region.collect(engine="cpu")
            var elapsed = Int(perf_counter_ns()) - started
            _check(result, expected)
            if gpu:
                accel += ("," if rep else "") + String(elapsed)
            else:
                cpu += ("," if rep else "") + String(elapsed)
    if identity_json() != identity:
        return (
            prefix
            + ',"status":"skipped","reason":"hardware or power identity changed during capture"}'
        )
    return (
        prefix
        + ',"status":"measured","verified":true,"direct_gpu":true,"cpu_ns":'
        + cpu
        + '],"accel_ns":'
        + accel
        + "]}"
    )
