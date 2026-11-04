"""Versioned, context-free validation of measured automatic placement costs.

Unidentified source builds and legacy profiles deliberately have no measured
cost. CPU and Metal models must also match the CPU's path cache: overwriting
an existing filename can never combine a new GPU model with an old CPU one.
"""
from std.math import isfinite
from std.memory import bitcast
from dataframe import DataType, LazyFrame, Series
from dataframe.expr import COL, LIT_INT, LIT_FLOAT
from dataframe.json.value import JsonValue, json_quote
from dataframe.parallel import configured_workers
from dataframe.placement import calibration_path, load_costs, LinearCost
from .accelerator.gpu.identity import hardware, runtime_build_id

comptime SCHEMA_VERSION = 1


def identity_json() raises -> String:
    var facts = hardware()
    return (
        '{"runtime_build_id":'
        + json_quote(runtime_build_id())
        + ',"device":'
        + json_quote(facts.device)
        + ',"host":'
        + json_quote(facts.host)
        + ',"workers":'
        + String(configured_workers())
        + "}"
    )


def _chunk_shape[D: DType](series: Series) raises -> String:
    var column = series.numeric[D]()
    return String(Int(len(column._bits[]) != 0))


def shape_key(region: LazyFrame, operator: String) raises -> String:
    """Physical layout and output work, independent of user column names."""
    var names = List[String]()
    for node in region._plan._nodes:
        for expression in node.exprs:
            for value in expression._nodes:
                if value.op == COL and value.text not in names:
                    names.append(value.text.copy())
    var frame = region._plan._frames[
        region._plan._nodes[0].source_slot()
    ].copy()
    var key = operator + ":layout_v1"
    # Literal bounds and arithmetic constants influence selectivity, overflow
    # and exact-bin retries. Do not borrow costs from a different query shape.
    for node in region._plan._nodes:
        for expression in node.exprs:
            for value in expression._nodes:
                if value.op == LIT_INT:
                    key += ":int=" + String(value.integer)
                elif value.op == LIT_FLOAT:
                    key += ":float=" + String(
                        bitcast[DType.uint64](value.floating)
                    )

    for name in names:
        var series = frame.column(name)
        key += ":" + String(series.n_chunks()) + "["
        for chunk in series.chunks():
            if chunk.dtype() == DataType.INT32:
                key += _chunk_shape[DType.int32](chunk)
            elif chunk.dtype() == DataType.INT64:
                key += _chunk_shape[DType.int64](chunk)
            elif chunk.dtype() == DataType.FLOAT32:
                key += _chunk_shape[DType.float32](chunk)
            else:
                raise Error("Unsupported Metal calibration layout")
            key += ","
        key += "]"
    return (
        key
        + ":outputs="
        + String(len(region._plan._nodes[len(region._plan._nodes) - 1].exprs))
    )


def _number(value: JsonValue) raises -> Float64:
    return Float64(value.canonical())


def _model(entry: JsonValue, engine: String) raises -> LinearCost:
    var model = entry.get(engine)
    var fixed = _number(model.get("fixed_ns"))
    var row = _number(model.get("row_ns"))
    if not isfinite(fixed) or not isfinite(row) or fixed < 0 or row < 0:
        raise Error("Calibration costs must be finite and nonnegative")
    return LinearCost(fixed, row)


def validate_identity(profile: JsonValue, current: JsonValue) raises -> String:
    if profile.get("schema_version").canonical() != String(SCHEMA_VERSION):
        return "unsupported or legacy calibration schema"
    var build = current.get("runtime_build_id").string()
    if (
        not build.startswith("dataframe-mojo-metal-v2;cpu=")
        or "development" in build
    ):
        return "CPU and Metal builds have no calibration identity"
    var saved = profile.get("identity")
    for field in ["runtime_build_id", "device", "host", "workers"]:
        if saved.get(field).canonical() != current.get(field).canonical():
            return "calibration identity mismatch: " + field
    return String()


def measured_cost(key: String, rows: Int) raises -> Float64:
    var path = calibration_path()
    if not path:
        return -1
    var text = String()
    with open(path, "r") as source:
        text = source.read()
    var profile = JsonValue(text)
    if validate_identity(profile, JsonValue(identity_json())):
        return -1
    var operators = profile.get("operators")
    if not operators.has(key):
        return -1
    var entry = operators.get(key)
    var first = _number(entry.get("min_rows"))
    var last = _number(entry.get("max_rows"))
    if (
        not isfinite(first)
        or not isfinite(last)
        or first < 1
        or last < first
        or Float64(rows) < first
        or Float64(rows) > last
    ):
        return -1
    var cpu = _model(entry, "cpu")
    var accel = _model(entry, "accel")
    var cached_cpu = load_costs("cpu").get(key)
    var cached_accel = load_costs("accel").get(key)
    if not cached_cpu or not cached_accel:
        return -1
    if (
        cached_cpu.value().fixed_ns != cpu.fixed_ns
        or cached_cpu.value().row_ns != cpu.row_ns
        or cached_accel.value().fixed_ns != accel.fixed_ns
        or cached_accel.value().row_ns != accel.row_ns
    ):
        return -1
    var value = accel.ns(rows)
    return value if isfinite(value) and isfinite(cpu.ns(rows)) else Float64(-1)


def calibration_status() -> String:
    """Explain a missing or mismatched profile without creating a GPU context."""
    try:
        var path = calibration_path()
        if not path:
            return "no Metal calibration configured"
        var text = String()
        with open(path, "r") as source:
            text = source.read()
        var reason = validate_identity(
            JsonValue(text), JsonValue(identity_json())
        )
        return reason if reason else String(
            "Metal calibration identity matches"
        )
    except error:
        return "invalid Metal calibration: " + String(error)
