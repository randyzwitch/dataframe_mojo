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
import os
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile

SCHEMA_VERSION = 2
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


def _typical(points):
    """Robust affine fit with nonnegative coefficients, including boundary refits."""
    slopes = [
        (y2 - y1) / (x2 - x1)
        for i, (x1, y1) in enumerate(points)
        for x2, y2 in points[i + 1 :]
        if x2 != x1
    ]
    slope = max(0.0, median(slopes))
    fixed = median(y - slope * x for x, y in points)
    if fixed < 0:
        fixed = 0.0
        slope = median(y / x for x, y in points)
    return {"fixed_ns": fixed, "row_ns": slope}


def _relative_upper_bound(points):
    """Least average relative excess for a nonnegative affine upper bound.

    Sum((fixed + slope * x) / y) is proportional to the line's height at
    the mean x weighted by 1/y. A supporting line of the upper hull at
    that point minimizes the objective while covering every training cost.
    Including (0, 0) enforces a nonnegative intercept.
    """
    hull = [(0, 0.0)]
    for x, y in points:
        while len(hull) >= 2:
            x0, y0 = hull[-2]
            x1, y1 = hull[-1]
            if (y1 - y0) * (x - x1) > (y - y1) * (x1 - x0):
                break
            hull.pop()
        hull.append((x, y))
    pivot = sum(x / y for x, y in points) / sum(1 / y for _, y in points)
    for (x0, y0), (x1, y1) in zip(hull, hull[1:]):
        if x1 >= pivot:
            slope = max(0.0, (y1 - y0) / (x1 - x0))
            break
    fixed = max(0.0, max(y - slope * x for x, y in points))
    return {"fixed_ns": fixed, "row_ns": slope}


def _ns(model, rows):
    return model["fixed_ns"] + model["row_ns"] * rows


def _materialized_cpu_ranges(pipelines):
    """An executor preference, valid only where every contributing layout was measured."""
    ranges = {}
    for pipeline, entries in pipelines.items():
        # A legacy capture could be contrary evidence for any family.
        if any(
            not isinstance(e.get("cpu_layout_family"), str) or not e["cpu_layout_family"]
            for e in entries
        ):
            continue
        families = defaultdict(list)
        for entry in entries:
            families[entry["cpu_layout_family"]].append(entry)
        for family, members in families.items():
            layouts = defaultdict(list)
            for entry in members:
                layouts[entry["key"]].append(entry["rows"])
            if len(layouts) < 2 or not all(
                "cpu_materialized_ns" in e
                and all(m < s for m, s in zip(e["cpu_materialized_ns"], e["cpu_ns"]))
                for e in members
            ):
                continue
            first = max(min(scales) for scales in layouts.values())
            last = min(max(scales) for scales in layouts.values())
            if first <= last:
                ranges[pipeline + "|" + family] = {
                    "min_rows": first, "max_rows": last
                }
    return ranges


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
        pipeline = record.get("pipeline_key")
        if not isinstance(pipeline, str) or not pipeline.startswith(
            "pipeline:"
        ):
            raise ValueError("capture requires a CPU pipeline key")
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
        if "cpu_materialized_ns" in record:
            samples = record["cpu_materialized_ns"]
            if (
                not isinstance(samples, list)
                or len(samples) != len(record["cpu_ns"])
                or len(samples) != len(record["accel_ns"])
                or not all(_positive(n) for n in samples)
            ):
                raise ValueError("invalid paired materialized CPU samples")
        if "first_accel_ns" in record and not _positive(record["first_accel_ns"]):
            raise ValueError("invalid first Metal execution time")
        grouped[key].append(record)
    if identity is None or not grouped:
        raise ValueError("no measured operators")
    operators = {}
    pipelines = defaultdict(list)
    for entries in grouped.values():
        for entry in entries:
            pipelines[entry["pipeline_key"]].append(entry)
    for key, entries in sorted(pipelines.items()):
        points = []
        for n in sorted({e["rows"] for e in entries}):
            points.append(
                (
                    n,
                    min(
                        t
                        for e in entries
                        if e["rows"] == n
                        for t in e["cpu_ns"]
                    ),
                )
            )
        operators[key] = {"cpu": _envelope(points, False)}
    regions = {}
    for key, entries in sorted(grouped.items()):
        rows = sorted({e["rows"] for e in entries})
        if len(rows) < 3:
            raise ValueError(f"{key}: measure at least three distinct scales")
        cpu, accel, materialized = [], [], []
        paired_cpu = all("cpu_materialized_ns" in e for e in entries)
        for n in rows:
            same = [e for e in entries if e["rows"] == n]
            cpu.append((n, median(t for e in same for t in e["cpu_ns"])))
            accel.append((n, median(t for e in same for t in e["accel_ns"])))
            if paired_cpu:
                times = [t for e in same for t in e["cpu_materialized_ns"]]
                materialized.append((n, median(times)))
        pipeline_keys = {e["pipeline_key"] for e in entries}
        if len(pipeline_keys) != 1:
            raise ValueError(
                "one region cannot have multiple pipeline identities"
            )
        pipeline = next(iter(pipeline_keys))
        cpu_model = _typical(cpu)
        accel_model = _typical(accel)
        materialized_model = _typical(materialized) if paired_cpu else None
        # Separate within-scale timing noise from systematic fit optimism.
        # Robust scale medians protect against individual timing outliers;
        # no consistently slower measured scale may be fitted as a GPU win.
        errors = []
        required = []
        for n in rows:
            sample_key = "cpu_ns"
            cpu_ns = _ns(cpu_model, n)
            if materialized_model and _ns(materialized_model, n) < cpu_ns:
                cpu_ns = _ns(materialized_model, n)
                sample_key = "cpu_materialized_ns"
            paired = [
                c / a
                for e in entries
                if e["rows"] == n
                for c, a in zip(e[sample_key], e["accel_ns"])
            ]
            observed = median(paired)
            required.append((n, cpu_ns / observed))
            errors.extend(abs(r / observed - 1) for r in paired)
        # A localized residual need not multiply both startup and per-row
        # cost. Bound every observed paired ratio, then apply timing noise.
        guarded = _relative_upper_bound(required)
        margin = max(0.05, 2 * median(errors))
        if key.startswith("pipeline:"):
            raise ValueError(
                "region key uses the reserved CPU pipeline namespace"
            )
        operators[key] = {"cpu": cpu_model}
        regions[key] = {
            "pipeline_key": pipeline,
            "cpu_key": key,
            "cpu": cpu_model,
            "accel_typical": accel_model,
            "switch_margin": margin,
            "accel": {k: v * (1 + margin) for k, v in guarded.items()},
            "min_rows": rows[0],
            "max_rows": rows[-1],
            "measured_rows": rows,
            "samples_per_engine": sum(len(e["cpu_ns"]) for e in entries),
        }
        if paired_cpu:
            regions[key]["cpu_materialized"] = materialized_model
        # Compare within the same capture so differences in row count or
        # layout do not get charged as initialization. Older captures contain
        # only warm samples and keep their previous placement behavior.
        startup = [
            e["first_accel_ns"] - median(e["accel_ns"])
            for e in entries
            if "first_accel_ns" in e
        ]
        if startup:
            regions[key]["startup_ns"] = max(0.0, median(startup))
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "operators": operators,
        "regions": regions,
        "materialized_cpu_ranges": _materialized_cpu_ranges(pipelines),
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
    with NamedTemporaryFile(dir=directory, delete=False) as target:
        target.write(payload)
    try:
        try:
            os.link(target.name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("calibration filename collision")
    finally:
        Path(target.name).unlink(missing_ok=True)
    return path


def install(profile, cache):
    identity = profile["identity"]
    cpu = identity["runtime_build_id"].split(";cpu=", 1)[1].split(";", 1)[0]
    if len(cpu) != 64 or any(c not in "0123456789abcdef" for c in cpu):
        raise ValueError("installation requires a packaged CPU build identity")
    directory = cache / "dataframe-mojo" / "calibration" / cpu / str(identity["workers"])
    path = publish(profile, directory)
    with NamedTemporaryFile(mode="w", dir=directory, delete=False) as tmp:
        tmp.write(path.name + "\n")
    try:
        os.replace(tmp.name, directory / "active")
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    dest = parser.add_mutually_exclusive_group(required=True)
    dest.add_argument("--output-dir", type=Path)
    dest.add_argument("--install", action="store_true", help="use this profile for local auto placement")
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.capture.read_text().splitlines()
        if line.strip()
    ]
    profile = fit(records)
    if args.install:
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        print(install(profile, cache))
    else:
        print(publish(profile, args.output_dir))
