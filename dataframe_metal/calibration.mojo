"""Versioned, context-free validation of measured automatic placement costs.

Unidentified source builds and legacy profiles deliberately have no measured
cost. CPU and Metal models must also match the CPU's path cache: overwriting
an existing filename can never combine a new GPU model with an old CPU one.
"""
from std.math import isfinite
from std.atomic import Atomic
from std.collections import Dict
from std.ffi import _get_global
from std.memory import Pointer
from std.memory.alloc import Layout, alloc
from std.memory import bitcast
from dataframe import DataType, LazyFrame, Series
from dataframe.expr import COL, LIT_INT, LIT_FLOAT
from dataframe.json.value import JsonValue, json_quote
from dataframe.parallel import configured_workers
from dataframe.planning.device_placement import plan_pipeline_cuts
from dataframe.placement import calibration_path, load_costs, LinearCost
from .accelerator.gpu.identity import hardware, runtime_build_id
from .accelerator.gpu.erased_input import single_native_span

comptime SCHEMA_VERSION = 2


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
    var key = operator + ":layout_v2"
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
        key += (
            ":"
            + String(series.n_chunks())
            + ":span="
            + String(Int(single_native_span(series)))
            + "["
        )
        for chunk in series.chunks():
            # Ratios generalize across calibrated row scales, while uneven
            # chunk geometry must not borrow a balanced layout's timings.
            key += (
                String(
                    (len(chunk) * 16 + len(series) // 2) // max(1, len(series))
                )
                + "/"
            )
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


def pipeline_key(region: LazyFrame) raises -> String:
    var optimized = region._optimized()
    var units = plan_pipeline_cuts(optimized._plan)
    if len(units) != 1:
        raise Error("Metal calibration requires one supported pipeline")
    return units[0].key.copy()


@fieldwise_init
struct RegionCost(Copyable, Movable):
    var cpu: LinearCost
    var accel: LinearCost
    var pipeline: String
    var cpu_key: String
    var first: Float64
    var last: Float64


struct _ProfileCache(Movable):
    var lock: Atomic[Int64]
    var path: String
    var text: String
    var identity: String
    var regions: Dict[String, RegionCost]
    var valid: Bool

    def __init__(out self):
        self.lock = Atomic[Int64](0)
        self.path = String()
        self.text = String()
        self.identity = String()
        self.regions = Dict[String, RegionCost]()
        self.valid = False


def _profile_init() -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    var slot = alloc(Layout[_ProfileCache].single()).unsafe_leak()
    slot.unsafe_write(_ProfileCache())
    return Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=Int(slot))


def _profile_keep(address: Optional[Pointer[NoneType, MutUntrackedOrigin]]):
    pass


def _snapshot(
    path: String, text: String, key: String
) raises -> Tuple[String, Optional[RegionCost]]:
    var address = _get_global[
        "DATAFRAME_METAL_CALIBRATION_PROFILE_V2", _profile_init, _profile_keep
    ]()
    ref cache = Pointer[_ProfileCache, MutAnyOrigin](
        unsafe_from_address=Int(address.value())
    )[]
    var unlocked = Int64(0)
    while not cache.lock.compare_exchange(unlocked, 1):
        unlocked = 0
    try:
        if cache.path != path:
            cache.path = path.copy()
            cache.text = text.copy()
            cache.valid = False
            cache.regions = Dict[String, RegionCost]()
            var profile = JsonValue(text)
            if profile.get("schema_version").canonical() == String(
                SCHEMA_VERSION
            ):
                cache.identity = profile.get("identity").canonical()
                var regions = profile.get("regions")
                var operators = profile.get("operators")
                for name in regions.keys():
                    var entry = regions.get(name)
                    var first = _number(entry.get("min_rows"))
                    var last = _number(entry.get("max_rows"))
                    if (
                        not isfinite(first)
                        or not isfinite(last)
                        or first < 1
                        or last < first
                    ):
                        raise Error("Invalid measured Metal row range")
                    var cpu = _model(entry, "cpu")
                    var accel = _model(entry, "accel")
                    var pipeline = entry.get("pipeline_key").string()
                    var cpu_key = entry.get("cpu_key").string() if entry.has(
                        "cpu_key"
                    ) else pipeline.copy()
                    if cpu_key != name and cpu_key != pipeline:
                        raise Error(
                            "Metal CPU model has a different region key"
                        )
                    var reference = _model(operators.get(cpu_key), "cpu")
                    if (
                        not pipeline.startswith("pipeline:")
                        or reference.fixed_ns != cpu.fixed_ns
                        or reference.row_ns != cpu.row_ns
                    ):
                        raise Error(
                            "Metal region does not match its CPU pipeline model"
                        )
                    cache.regions[name] = RegionCost(
                        cpu^, accel^, pipeline^, cpu_key^, first, last
                    )
                cache.valid = True
        # File bytes are checked on every call, while JSON is parsed once.
        # An overwritten cached path cannot mix new GPU and old CPU models.
        var result: Tuple[String, Optional[RegionCost]]
        if cache.valid and cache.text == text:
            result = (cache.identity.copy(), cache.regions.get(key))
        else:
            result = (String(), Optional[RegionCost]())
        cache.lock.store(0)
        return result^
    except error:
        cache.lock.store(0)
        raise error^


def measured_costs(key: String, rows: Int) raises -> Tuple[Float64, Float64]:
    var path = calibration_path()
    if not path:
        return (Float64(-1), Float64(-1))
    var text: String
    with open(path, "r") as source:
        text = source.read()
    var snapshot = _snapshot(path, text, key)
    if not snapshot[1]:
        return (Float64(-1), Float64(-1))
    var current = JsonValue(identity_json())
    var saved = JsonValue('{"schema_version":2,"identity":' + snapshot[0] + "}")
    if validate_identity(saved, current):
        return (Float64(-1), Float64(-1))
    var entry = snapshot[1].value().copy()
    if Float64(rows) < entry.first or Float64(rows) > entry.last:
        return (Float64(-1), Float64(-1))
    var cpu_models = load_costs("cpu")
    var cached_cpu = cpu_models.get(entry.cpu_key)
    if not cached_cpu:
        return (Float64(-1), Float64(-1))
    if (
        cached_cpu.value().fixed_ns != entry.cpu.fixed_ns
        or cached_cpu.value().row_ns != entry.cpu.row_ns
    ):
        return (Float64(-1), Float64(-1))
    var cpu_ns = entry.cpu.ns(rows)
    var accel_ns = entry.accel.ns(rows)
    if not isfinite(cpu_ns) or not isfinite(accel_ns):
        return (Float64(-1), Float64(-1))
    return (cpu_ns, accel_ns)


def measured_cost(key: String, rows: Int) raises -> Float64:
    return measured_costs(key, rows)[1]


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
