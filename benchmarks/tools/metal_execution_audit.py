"""Validate observed pipeline evidence independently of placement labels."""
import math


FORMAT = "dataframe-pipeline-observation-v1"
PHASES = (
    "context_wall_ms",
    "input_wall_ms",
    "device_work_wall_ms",
    "finish_wall_ms",
)


def validate_pipeline(record):
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        raise ValueError("Unknown pipeline observation format")
    for name in (
        "node",
        "pipeline",
        "plan_id",
        "query_id",
        "execution_attempt",
        "executions",
        "gpu_launches",
        "gpu_completed_launches",
        "upload_bytes",
        "download_bytes",
        "device_waits",
    ):
        if type(record.get(name)) is not int or record[name] < 0:
            raise ValueError(f"Invalid pipeline counter: {name}")
    if record["node"] != record["pipeline"] or record["executions"] == 0:
        raise ValueError("Pipeline record must describe an executed terminal")
    if record["plan_id"] > record["node"]:
        raise ValueError("Pipeline node precedes its plan's ID range")
    if type(record.get("result_used")) is not bool:
        raise ValueError("Missing pipeline result disposition")
    for name in (
        "operator",
        "result_device",
        "device_status",
        "fallback_reason",
    ):
        if not isinstance(record.get(name), str):
            raise ValueError(f"Missing pipeline field: {name}")
    launches, completed = (
        record["gpu_launches"],
        record["gpu_completed_launches"],
    )
    if completed > launches:
        raise ValueError("More completed kernels than submitted kernels")
    status = record["device_status"]
    if status not in {
        "cpu",
        "cpu_fallback",
        "no_device_work",
        "completed",
        "failed",
        "unverified",
    }:
        raise ValueError("Unknown device execution status")
    if status == "completed" and not (launches > 0 and launches == completed):
        raise ValueError("Device completion lacks completed kernel evidence")
    if record["result_device"] == "metal" and status != "completed":
        raise ValueError("Metal result lacks verified device completion")
    if status == "completed" and record["result_device"] != "metal":
        raise ValueError("Completed Metal work lacks its device result")
    if status == "cpu" and record["result_device"] != "cpu":
        raise ValueError("CPU status contradicts result device")
    if status == "failed" and record["result_used"]:
        raise ValueError("Failed pipeline cannot supply a used result")
    if status in {"cpu", "no_device_work"} and launches != 0:
        raise ValueError("No-device status contradicts submitted kernels")
    if status == "no_device_work" and record["result_device"] != "host":
        raise ValueError("No-device result must be identified as host work")
    if status == "cpu_fallback" and (
        record["result_device"] != "cpu" or not record["fallback_reason"]
    ):
        raise ValueError("CPU fallback lacks its outcome or reason")
    timing = record.get("phase_timing_kind")
    if timing not in {"host_wall", "unavailable"}:
        raise ValueError(
            "Pipeline phases must be host wall times or unavailable"
        )
    for name in PHASES:
        value = record.get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Invalid pipeline phase: {name}")
        if (timing == "host_wall" and value < 0) or (
            timing == "unavailable" and value != -1
        ):
            raise ValueError("Pipeline phase availability is inconsistent")
    wait = record.get("synchronize_wall_ms")
    if type(wait) not in (int, float) or not math.isfinite(wait) or wait < 0:
        raise ValueError("Invalid host synchronization time")
    return record


def summarize(calls, collections):
    verified = len(calls) >= collections and collections > 0
    counts = dict(
        gpu_completed_calls=0,
        gpu_result_calls=0,
        cpu_fallback_calls=0,
        cpu_compute_calls=0,
        no_device_work_calls=0,
        pipeline_count=0,
    )
    for call in calls:
        records = [validate_pipeline(r) for r in call["pipelines"]]
        ids = {
            (r["query_id"], r["execution_attempt"], r["pipeline"])
            for r in records
        }
        if len(ids) != len(records):
            raise ValueError(
                "Duplicate pipeline observations in one collection"
            )
        if len({r["query_id"] for r in records}) > 1:
            raise ValueError("One collection has multiple query identities")
        verified &= (
            bool(records)
            and call["status"]
            in {"gpu_observed", "cpu_only", "no_device_work", "cpu"}
            and all(r["device_status"] != "unverified" for r in records)
        )
        used = [r for r in records if r["result_used"]]
        verified &= bool(used)
        gpu = any(
            r["result_device"] == "metal" and r["gpu_completed_launches"] > 0
            for r in used
        )
        if records and call["status"] == "gpu_observed" and not gpu:
            raise ValueError(
                "GPU collection summary lacks a used device result"
            )
        if (
            records
            and call["status"] in {"cpu_only", "no_device_work", "cpu"}
            and gpu
        ):
            raise ValueError(
                "CPU/host collection summary contradicts a used device result"
            )
        counts["gpu_completed_calls"] += any(
            r["gpu_completed_launches"] > 0 for r in records
        )
        counts["gpu_result_calls"] += gpu
        counts["cpu_fallback_calls"] += any(
            r["device_status"] == "cpu_fallback" for r in used
        )
        counts["cpu_compute_calls"] += any(
            r["result_device"] == "cpu" and r["operator"] != "SCAN frame"
            for r in used
        )
        counts["no_device_work_calls"] += any(
            r["device_status"] == "no_device_work" for r in used
        )
        counts["pipeline_count"] += len(records)
    return dict(
        format="dataframe-runtime-execution-v1",
        status="complete" if verified else "incomplete",
        observed_calls=len(calls),
        whole_query_gpu_certificate=False,
        calls=calls,
        **counts,
    )


def verified_summary(evidence, collections):
    """Recompute admission counters from the retained pipeline records."""
    if (
        not isinstance(evidence, dict)
        or type(collections) is not int
        or collections <= 0
    ):
        return False
    try:
        actual = summarize(evidence["calls"], collections)
    except (KeyError, TypeError, ValueError):
        return False
    return actual["status"] == "complete" and actual == evidence
