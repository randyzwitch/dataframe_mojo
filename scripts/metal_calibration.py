#!/usr/bin/env python3
"""Fit conservative, versioned Metal placement costs from JSONL captures.

Input records come from dataframe_metal.calibration.capture, which verifies
CPU and direct GPU answers before timing. Publish to a new content-addressed
filename; the CPU placement cache deliberately retains a file's first models.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

SCHEMA_VERSION = 1
IDENTITY_FIELDS = {"runtime_build_id", "device", "host", "workers"}


def _positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _envelope(points, upper):
    slopes = [
        (y2 - y1) / (x2 - x1)
        for i, (x1, y1) in enumerate(points)
        for x2, y2 in points[i + 1 :]
        if x2 != x1
    ]
    slope = max(0.0, median(slopes))
    if not upper:
        slope = min(slope, min(y / x for x, y in points))
    residuals = [y - slope * x for x, y in points]
    fixed = max(0.0, max(residuals) if upper else min(residuals))
    # Reserve extra headroom beyond the observed envelopes.
    margin = 1.1 if upper else 0.9
    return {"fixed_ns": fixed * margin, "row_ns": slope * margin}


def fit(records):
    identity = None
    grouped = defaultdict(list)
    skipped = []
    for record in records:
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported capture schema")
        current = record.get("identity", {})
        if set(current) != IDENTITY_FIELDS or not all(
            current.get(k) for k in IDENTITY_FIELDS
        ):
            raise ValueError("missing calibration identity")
        runtime = current["runtime_build_id"]
        if (
            not isinstance(runtime, str)
            or not runtime.startswith("dataframe-mojo-metal-v2;cpu=")
            or ";metal=" not in runtime
            or "development" in runtime
        ):
            raise ValueError("capture requires identified CPU and Metal builds")
        if (
            not isinstance(current["workers"], int)
            or isinstance(current["workers"], bool)
            or current["workers"] < 1
        ):
            raise ValueError("invalid CPU worker budget")
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError(
                "mixed build, hardware, power state or thread identities"
            )
        if record.get("status") != "measured":
            skipped.append(record)
            continue
        if (
            record.get("verified") is not True
            or record.get("direct_gpu") is not True
        ):
            raise ValueError(
                "timing requires independently checked direct GPU completion"
            )
        rows = record.get("rows")
        key = record.get("key")
        if (
            not isinstance(rows, int)
            or isinstance(rows, bool)
            or rows < 1
            or not isinstance(key, str)
            or not key
        ):
            raise ValueError("invalid row count or operator key")
        for engine in ("cpu_ns", "accel_ns"):
            samples = record.get(engine, [])
            if len(samples) < 3 or not all(_positive(n) for n in samples):
                raise ValueError(
                    "at least three positive finite samples per engine required"
                )
        grouped[key].append(record)
    if identity is None or not grouped:
        raise ValueError("no measured operators")
    operators = {}
    for key, entries in sorted(grouped.items()):
        rows = sorted({e["rows"] for e in entries})
        if len(rows) < 3:
            raise ValueError(f"{key}: measure at least three distinct scales")
        cpu, accel = [], []
        for n in rows:
            same = [e for e in entries if e["rows"] == n]
            cpu.append((n, min(t for e in same for t in e["cpu_ns"])))
            accel.append((n, max(t for e in same for t in e["accel_ns"])))
        operators[key] = {
            "cpu": _envelope(cpu, False),
            "accel": _envelope(accel, True),
            "min_rows": rows[0],
            "max_rows": rows[-1],
            "measured_rows": rows,
            "samples_per_engine": sum(len(e["cpu_ns"]) for e in entries),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "operators": operators,
        "skipped": skipped,
        "capture_records": records,
    }


def publish(profile, directory):
    payload = (
        json.dumps(profile, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    name = "metal-calibration-" + hashlib.sha256(payload).hexdigest() + ".json"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    try:
        with path.open("xb") as target:
            target.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("calibration filename collision")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.capture.read_text().splitlines()
        if line.strip()
    ]
    print(publish(fit(records), args.output_dir))
