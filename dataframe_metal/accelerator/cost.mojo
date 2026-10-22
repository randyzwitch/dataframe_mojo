"""Shared analytical automatic placement; providers supply facts, never policy.

Features are bounded plan metadata. No driver, execution context, source values,
or provider implementation is imported here. Calibration uses complete caller
wall times and independently validated lower/upper residual envelopes.
"""
from std.math import isfinite
from dataframe.json_value import JsonValue
from dataframe.plan_types import PlanEstimate

comptime COST_FORMAT = "dataframe-analytical-cost-v1"
comptime COST_FEATURE_COUNT = 15


def cost_feature_names() -> List[String]:
    return List[String](
        [
            "fixed",
            "rows",
            "source_bytes",
            "projection_node_rows",
            "filter_node_rows",
            "reduction_rows",
            "integer_node_rows",
            "cast_node_rows",
            "launches",
            "synchronizations",
            "upload_bytes",
            "download_bytes",
            "allocation_bytes",
            "cold",
            "materializations",
        ]
    )


@fieldwise_init
struct CalibrationKey(Copyable):
    var provider: String
    var device: String
    var runtime_build_id: String
    var cpu_build_id: String
    var host: String
    var threads: Int
    var optimize: Bool
    var streaming: Bool
    var batch_size: Int


@fieldwise_init
struct CostDecision(Copyable):
    var engine: String
    var reason: String
    var cpu_ns: Float64
    var accelerator_ns: Float64


def cpu_decision(reason: String) -> CostDecision:
    return CostDecision("cpu", reason, 0, 0)


def _number(
    record: JsonValue, key: String, positive: Bool = False
) raises -> Float64:
    var value = Float64(record.get(key).text)
    if not isfinite(value) or value < 0 or (positive and value == 0):
        raise Error("invalid analytical coefficient " + key)
    return value


def _matches(
    record: JsonValue, key: CalibrationKey, family: String
) raises -> Bool:
    return (
        record.get("provider").string() == key.provider
        and record.get("device").string() == key.device
        and record.get("runtime_build_id").string() == key.runtime_build_id
        and record.get("cpu_build_id").string() == key.cpu_build_id
        and record.get("host").string() == key.host
        and Int(record.get("threads").text) == key.threads
        and record.get("optimize").text == ("true" if key.optimize else "false")
        and record.get("streaming").text
        == ("true" if key.streaming else "false")
        and Int(record.get("batch_size").text) == key.batch_size
        and record.get("family").string() == family
    )


def _predict(
    model: JsonValue, features: List[Float64]
) raises -> Tuple[Float64, String]:
    if len(features) != COST_FEATURE_COUNT:
        raise Error("analytical feature vector has the wrong width")
    var total = Float64(0)
    var largest = Float64(-1)
    var term = String()
    var coefficients = model.get("coefficients")
    var minimum = model.get("minimum")
    var maximum = model.get("maximum")
    var feature_names = cost_feature_names()
    for i in range(len(feature_names)):
        var name = feature_names[i]
        var value = features[i]
        var low = _number(minimum, name)
        var high = _number(maximum, name)
        if not isfinite(value) or value < low or value > high or low > high:
            raise Error("plan is outside analytical coverage for " + name)
        var contribution = value * _number(coefficients, name)
        total += contribution
        if contribution > largest:
            largest = contribution
            term = name
    var factor = _number(model, "residual_factor", True)
    total *= factor
    if not isfinite(total) or total <= 0:
        raise Error("analytical estimate is not finite and positive")
    return (total, term^)


def select_auto(
    facts: PlanEstimate,
    table: JsonValue,
    key: CalibrationKey,
    budget_bytes: Int,
    now: Int64,
) raises -> CostDecision:
    """Choose once before submission; reject absent or stale evidence to CPU."""
    if table.get("format").string() != COST_FORMAT:
        return cpu_decision("unsupported analytical calibration format")
    if facts.placement != "accel":
        return cpu_decision("analytical CPU tail costs are not calibrated")
    if facts.peak_bytes < 0 or budget_bytes < facts.peak_bytes:
        return cpu_decision(
            "estimated accelerator payload exceeds memory budget"
        )
    if (
        key.cpu_build_id == "development-source-no-calibration"
        or key.runtime_build_id == "development-source-no-calibration"
    ):
        return cpu_decision(
            "analytical calibration requires identified CPU and provider builds"
        )
    var found = False
    var decision = cpu_decision(
        "no matching analytical hardware/build/settings calibration"
    )
    for record in table.get("records").items():
        if not _matches(record, key, facts.family):
            continue
        if found:
            raise Error("ambiguous analytical calibration records")
        found = True
        if Int64(Int(record.get("expires_at").text)) <= now:
            decision = cpu_decision("analytical calibration has expired")
            continue
        if (
            record.get("validated").text != "true"
            or Int(record.get("validation_failures").text) != 0
            or Int(record.get("training_queries").text) < 5
            or Int(record.get("validation_queries").text) < 5
            or record.get("distribution_coverage").string()
            != "conservative-null-selectivity-envelope"
        ):
            decision = cpu_decision(
                "analytical calibration lacks independent query/distribution validation"
            )
            continue
        var margin = _number(record, "safety_factor", True)
        if margin < 1.5:
            raise Error("analytical safety_factor must be at least 1.5")
        var cpu = _predict(record.get("cpu"), facts.cpu)
        var device = _predict(record.get("accelerator"), facts.accelerator)
        if (
            _number(record.get("cpu"), "residual_factor", True) > 1
            or _number(record.get("accelerator"), "residual_factor", True) < 1
        ):
            raise Error(
                "analytical residual envelopes point in the wrong direction"
            )
        var upper = device[0] * margin
        if not isfinite(upper):
            raise Error("analytical safety comparison overflow")
        var engine = "accel" if cpu[0] > upper else "cpu"
        decision = CostDecision(
            engine,
            (
                "accelerator wins conservative analytical comparison" if engine
                == "accel" else "CPU wins conservative analytical comparison"
            )
            + "; cpu_lower_ns="
            + String(cpu[0])
            + "; accelerator_upper_ns="
            + String(device[0])
            + "; safety_factor="
            + String(margin)
            + "; deciding_term="
            + (device[1] if engine == "cpu" else cpu[1]),
            cpu[0],
            device[0],
        )
    return decision^
