"""Literal calibration-format fixtures; this encoder is not a cost oracle."""
from dataframe.json_value import JsonValue, json_quote


def calibration_features() -> List[String]:
    return [
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


def calibration_vector() -> List[Float64]:
    var vector = List[Float64](length=15, fill=0)
    vector[0] = 1
    return vector^


def calibration_model(fixed: Float64, residual: Float64 = 1) -> String:
    var coefficients = String("{")
    var minimum = String("{")
    var maximum = String("{")
    var names = calibration_features()
    for i in range(len(names)):
        var comma = String(",") if i else String()
        var key = '"' + names[i] + '":'
        coefficients += comma + key + String(fixed if i == 0 else Float64(0))
        minimum += comma + key + "0"
        maximum += comma + key + "1000000"
    return (
        '{"coefficients":'
        + coefficients
        + '},"minimum":'
        + minimum
        + '},"maximum":'
        + maximum
        + '},"residual_factor":'
        + String(residual)
        + "}"
    )


def calibration_record(
    cpu: Float64 = 1000,
    device: Float64 = 100,
    *,
    family: String = "integer",
    expiry: Int = 200,
    validated: Bool = True,
    safety: Float64 = 1.5,
    cpu_residual: Float64 = 1,
    device_residual: Float64 = 1,
) -> String:
    return (
        '{"provider":"probe","device":"device","runtime_build_id":"runtime-v1","cpu_build_id":"cpu-v1","host":"host","threads":1,"optimize":true,"streaming":false,"batch_size":7,"family":'
        + json_quote(family)
        + ',"expires_at":'
        + String(expiry)
        + ',"validated":'
        + ("true" if validated else "false")
        + ',"validation_failures":0,"training_queries":5,"validation_queries":5,"distribution_coverage":"conservative-null-selectivity-envelope","safety_factor":'
        + String(safety)
        + ',"cpu":'
        + calibration_model(cpu, cpu_residual)
        + ',"accelerator":'
        + calibration_model(device, device_residual)
        + "}"
    )


def calibration_table(
    record: String, duplicate: Bool = False
) raises -> JsonValue:
    return JsonValue(
        '{"format":"dataframe-analytical-cost-v1","records":['
        + record
        + ("," + record if duplicate else "")
        + "]}"
    )
