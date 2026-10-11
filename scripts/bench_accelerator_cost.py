#!/usr/bin/env python3
"""Fit shared analytical terms on generated queries; evaluate held-out regret.

No benchmark executes without --collect. Use a provider-enabled identified
package directory and a prebuilt bench_accelerator_cost binary. The same runner
and fitting code serve any provider implementing the estimate/identity hooks.
Named mechanisms have prior development exposure and are query-held-out from
this fit; they are not an external-suite holdout claim. TPC-DS and ClickBench
remain validation-only under bench_policy.py. Failed and unsupported cases are
retained. No held-out observation changes coefficients or residual envelopes.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("fixed", "rows", "source_bytes", "projection_node_rows", "filter_node_rows", "reduction_rows", "integer_node_rows", "cast_node_rows", "launches", "synchronizations", "upload_bytes", "download_bytes", "allocation_bytes", "cold", "materializations")
FORMAT = "dataframe-analytical-cost-v1"
DTYPES = ("Float32", "Float64", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64")


def case_id(case):
    # Canonical query operations, excluding data shape and display names.
    workload = case["workload"]
    if workload == "chain32":
        operations = [("project", 2)] * 32
    elif workload == "count":
        operations = [("filter", 8), ("reduce", "count")]
    elif workload in ("sum", "extrema"):
        operations = [("reduce", workload)]
    else:
        adds = (case["nodes"] - 1) // 2
        position = case["filter"]
        operations = []
        if position == "before":
            operations.append(("filter", case["threshold"]))
        if position == "middle":
            operations += [("project", adds // 2), ("filter", case["threshold"]), ("project", adds - adds // 2)]
        else:
            operations += [("project", adds)] * case["stages"]
        if position == "after":
            operations.append(("filter", case["threshold"]))
        if case["reduction"] != "rows":
            operations.append(("reduce", case["reduction"]))
    return hashlib.sha256(json.dumps((case["dtype"], operations)).encode()).hexdigest()


def matrix(samples=192, seed=583):
    rng = random.Random(seed)
    generated = list(itertools.product((1, 3, 7, 15, 31, 61), ("none", "before", "middle", "after"), (-1, 8, 16), (1, 2, 4, 8, 16, 32), ("rows", "count", "sum", "mean", "extrema")))
    rng.shuffle(generated)
    generated = generated[:samples]
    cases = []
    rows = (4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 33554432)
    nulls = (0, 5, 50, 100)
    for dtype in DTYPES:
        for i, (nodes, position, threshold, stages, reduction) in enumerate(generated):
            cases.append(dict(role="train", workload="generated", dtype=dtype, rows=rows[i % len(rows)], null_percent=nulls[(i // len(rows)) % len(nulls)], nodes=nodes, filter=position, threshold=threshold, stages=stages, reduction=reduction))
        heldout = (("chain32", 5, "none", 8, 32), ("count", 1, "before", 8, 1), ("sum", 1, "none", 8, 1), ("extrema", 1, "none", 8, 1), ("projection", 3, "none", 8, 1), ("expression15", 61, "none", 8, 1), ("filter_half", 3, "before", 8, 1), ("filter_sparse", 3, "after", 15, 1))
        for workload, nodes, position, threshold, stages in heldout:
            for n in (65536, 1048576, 16777216):
                for null in nulls:
                    cases.append(dict(role="heldout", workload=workload, dtype=dtype, rows=n, null_percent=null, nodes=nodes, filter=position, threshold=threshold, stages=stages, reduction="rows"))
    heldout_ids = {case_id(c) for c in cases if c["role"] == "heldout"}
    return [c for c in cases if c["role"] == "heldout" or case_id(c) not in heldout_ids]


def parse_run(output, require_complete=True):
    record = {"cpu": [], "accelerator": [], "samples": [], "profile": {}}
    cpu_features, accelerator_features = {}, {}
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        kind, body = parts
        if kind == "KEY":
            record["key"] = json.loads(body)
        elif kind == "FEATURE":
            name, cpu, accelerator = body.split()
            cpu_features[name], accelerator_features[name] = float(cpu), float(accelerator)
        elif kind == "PEAK":
            record["peak_bytes"] = int(body)
        elif kind == "COLD":
            cold, planning = map(int, body.split())
            record["cold_ns"], record["cold_planning_ns"] = cold, planning
        elif kind == "SAMPLE":
            cpu, accelerator, planning = map(int, body.split())
            record["samples"].append(dict(cpu_ns=cpu, accelerator_ns=accelerator, planning_ns=planning))
        elif kind == "PROFILE":
            name, value = body.split()
            record["profile"][name] = int(value)
        elif kind == "VALIDATED":
            record["parity"] = body.lower() == "true"
    record["cpu_features"] = cpu_features
    record["accelerator_features"] = accelerator_features
    if not require_complete:
        return record
    if not record.get("parity") or len(record["samples"]) < 5 or set(cpu_features) != set(FEATURES) or set(accelerator_features) != set(FEATURES):
        raise ValueError("incomplete parity, timing or feature output")
    if "key" not in record or "cold_ns" not in record or "cold_planning_ns" not in record or "peak_bytes" not in record:
        raise ValueError("incomplete calibration identity or cold observation")
    if record["peak_bytes"] < 0 or any(not math.isfinite(value) or value < 0 for features in (cpu_features, accelerator_features) for value in features.values()):
        raise ValueError("invalid calibration memory or feature values")
    if record["cold_ns"] <= 0 or record["cold_planning_ns"] <= 0 or any(sample[name] <= 0 for sample in record["samples"] for name in ("cpu_ns", "accelerator_ns", "planning_ns")):
        raise ValueError("calibration timings must be positive")
    required_identity = ("provider", "device", "runtime_build_id", "cpu_build_id", "host", "threads", "optimize", "streaming", "batch_size", "family")
    if not isinstance(record["key"], dict) or any(name not in record["key"] for name in required_identity):
        raise ValueError("incomplete calibration identity fields")
    for key in ("cpu_build_id", "runtime_build_id"):
        if not record["key"][key] or record["key"][key] == "development-source-no-calibration":
            raise ValueError("calibration requires identified packages")
    return record


def collect(binary, cases, output, table, reps, timeout):
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output.open("w") as file:
        for number, case in enumerate(cases):
            env = os.environ.copy()
            env.update({"COST_" + k.upper(): str(v) for k, v in case.items() if k != "role"})
            env.update(COST_REPS=str(reps), DATAFRAME_ACCEL_COST_TABLE=str(table.resolve()))
            record = {"case": case, "query_id": case_id(case)}
            try:
                result = subprocess.run([str(binary.resolve())], env=env, text=True, capture_output=True, timeout=timeout)
                record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
                if result.returncode:
                    record["status"] = "failed_or_unsupported"
                else:
                    record.update(parse_run(result.stdout), status="ok")
            except (subprocess.TimeoutExpired, ValueError) as error:
                record.update(status="error", error=str(error))
                if isinstance(error, subprocess.TimeoutExpired):
                    record["stdout"] = (error.stdout or b"").decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
                    record["stderr"] = (error.stderr or b"").decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
            if record["status"] != "ok" and record.get("stdout"):
                try:
                    record.update(parse_run(record["stdout"], require_complete=False))
                except (ValueError, KeyError) as error:
                    record["partial_metadata_error"] = str(error)
            file.write(json.dumps(record, sort_keys=True) + "\n")
            file.flush()
            records.append(record)
            print(f"{number + 1}/{len(cases)} {case['role']} {case['dtype']} {record['status']}", flush=True)
    return records


def nnls(x, y, iterations=2000):
    """Nonnegative coordinate descent with scaled columns; no extra dependency."""
    scales = [max(1.0, max(row[j] for row in x)) for j in range(len(FEATURES))]
    scaled = [[v / scales[j] for j, v in enumerate(row)] for row in x]
    coefficients = [0.0] * len(FEATURES)
    residual = list(y)
    for _ in range(iterations):
        changed = 0.0
        for j in range(len(FEATURES)):
            denominator = sum(row[j] ** 2 for row in scaled)
            if not denominator:
                continue
            old = coefficients[j]
            new = max(0.0, old + sum(row[j] * r for row, r in zip(scaled, residual)) / denominator)
            delta = new - old
            if delta:
                for i, row in enumerate(scaled):
                    residual[i] -= row[j] * delta
                coefficients[j] = new
                changed = max(changed, abs(delta) / max(1.0, abs(new)))
        if changed < 1e-9:
            break
    return {name: coefficients[j] / scales[j] for j, name in enumerate(FEATURES)}


def observations(records, engine):
    x, y = [], []
    for record in records:
        features = record[engine + "_features"].copy()
        if engine == "accelerator":
            features["cold"] = 0
        for sample in record["samples"]:
            x.append([features[name] for name in FEATURES])
            y.append(sample["cpu_ns"] if engine == "cpu" else sample["accelerator_ns"] + sample["planning_ns"])
        if engine == "accelerator":
            features = features.copy()
            features["cold"] = 1
            x.append([features[name] for name in FEATURES])
            y.append(record["cold_ns"] + record["cold_planning_ns"])
    return x, y


def fit_model(records, engine):
    x, y = observations(records, engine)
    coefficients = nnls(x, y)
    predictions = [sum(coefficients[name] * row[j] for j, name in enumerate(FEATURES)) for row in x]
    if any(not math.isfinite(p) or p <= 0 for p in predictions):
        raise ValueError("nonpositive or nonfinite fitted prediction")
    ratios = [actual / predicted for actual, predicted in zip(y, predictions)]
    factor = min(1.0, min(ratios) * .9) if engine == "cpu" else max(1.0, max(ratios) * 1.1)
    return dict(coefficients=coefficients, residual_factor=factor, minimum={name: min(row[j] for row in x) for j, name in enumerate(FEATURES)}, maximum={name: max(row[j] for row in x) for j, name in enumerate(FEATURES)})


def fit(records, ttl=86400):
    groups = {}
    for record in records:
        if record["status"] == "ok" and record["case"]["role"] == "train":
            groups.setdefault(json.dumps(record["key"], sort_keys=True), []).append(record)
    result = dict(format=FORMAT, memory_fraction=.8, records=[], rejected_keys=[])
    for identity, group in groups.items():
        key = json.loads(identity)
        failures = [r for r in records if r["status"] != "ok" and r["case"]["role"] == "train" and r.get("key") == key]
        if failures:
            result["rejected_keys"].append(dict(key=key, reason="training execution or parity failed", failures=len(failures)))
            continue
        queries = {r["query_id"] for r in group}
        if len(queries) < 5:
            continue
        result["records"].append(dict(json.loads(identity), expires_at=int(time.time()) + ttl, validated=False, training_queries=len(queries), validation_queries=0, validation_failures=0, distribution_coverage="conservative-null-selectivity-envelope", safety_factor=1.5, cpu=fit_model(group, "cpu"), accelerator=fit_model(group, "accelerator")))
    return result


def predict(model, features):
    if any(not math.isfinite(features[k]) or features[k] < model["minimum"][k] or features[k] > model["maximum"][k] for k in FEATURES):
        return None
    return sum(model["coefficients"][k] * features[k] for k in FEATURES) * model["residual_factor"]


def evaluate(table, records, training_ids):
    """Freeze fits, validate envelopes, retain CPU fallbacks and decision regret."""
    evaluation = []
    for calibration in table["records"]:
        key = {k: calibration[k] for k in ("provider", "device", "runtime_build_id", "cpu_build_id", "host", "threads", "optimize", "streaming", "batch_size", "family")}
        heldout = [r for r in records if r["status"] == "ok" and r["case"]["role"] == "heldout" and r["key"] == key]
        failures = [r for r in records if r["status"] != "ok" and r["case"]["role"] == "heldout" and r.get("key") == key]
        calibration["validation_failures"] = len(failures)
        covered_queries, good = set(), not failures
        calibration_cases = []
        for record in heldout:
            if record["query_id"] in training_ids:
                raise ValueError("training/held-out query leakage")
            for cold in (False, True):
                features = record["accelerator_features"].copy()
                features["cold"] = float(cold)
                cpu = predict(calibration["cpu"], record["cpu_features"])
                accelerator = predict(calibration["accelerator"], features)
                observed_cpu = min(s["cpu_ns"] for s in record["samples"])
                planning = record["cold_planning_ns"] if cold else max(s["planning_ns"] for s in record["samples"])
                raw_accel = record["cold_ns"] if cold else max(s["accelerator_ns"] for s in record["samples"])
                observed_accel = raw_accel + planning
                covered = cpu is not None and accelerator is not None
                if covered:
                    covered_queries.add(record["query_id"])
                    good &= cpu <= observed_cpu and accelerator >= observed_accel
                choose = covered and cpu > accelerator * calibration["safety_factor"]
                chosen = observed_accel if choose else observed_cpu + planning
                best = min(observed_cpu, raw_accel)
                calibration_cases.append(dict(query_id=record["query_id"], case=record["case"], cold=cold, covered=covered, engine="accel" if choose else "cpu", cpu_lower_ns=cpu, accelerator_upper_ns=accelerator, observed_cpu_ns=observed_cpu, observed_accelerator_ns=observed_accel, regret_ns=chosen-best, regret_ratio=chosen/best, planning_ns=planning, engine_regret_ns=(raw_accel if choose else observed_cpu)-best, wrong_choice=((raw_accel if choose else observed_cpu) > best), raw_accelerator_ns=raw_accel, oracle_ns=best, candidate_engine="accel" if choose else "cpu", candidate_regret_ns=chosen-best))
        calibration["validation_queries"] = len(covered_queries)
        calibration["validated"] = bool(good and len(covered_queries) >= 5)
        for case in calibration_cases:
            case["calibration_validated"] = calibration["validated"]
            # A rejected calibration never enables a device in the core policy.
            # Preserve the proposed model's choice separately for diagnosis.
            if not calibration["validated"]:
                case["engine"] = "cpu"
                case["regret_ns"] = case["observed_cpu_ns"] + case["planning_ns"] - case["oracle_ns"]
                case["regret_ratio"] = (case["observed_cpu_ns"] + case["planning_ns"]) / case["oracle_ns"]
                case["engine_regret_ns"] = case["observed_cpu_ns"] - case["oracle_ns"]
                case["wrong_choice"] = case["observed_cpu_ns"] > case["oracle_ns"]
            evaluation.append(case)
    return dict(method="policy replay against explicit-engine wall samples plus measured preflight overhead; automatic-execution latency is not directly measured", cases=evaluation, maximum_regret_ratio=max((r["regret_ratio"] for r in evaluation), default=None), total_regret_ns=sum(r["regret_ns"] for r in evaluation), wrong_choices=sum(r["wrong_choice"] for r in evaluation), covered=sum(r["covered"] for r in evaluation), note="Named mechanisms were used in prior development. Query-held-out from this fit; no external-suite generalization claim. Fits and residual envelopes are frozen before validation.")


def artifact_manifest(binary, package):
    metadata = json.loads((package / "provider.json").read_text())
    if metadata.get("engine_dirty") or metadata.get("core_dirty"):
        raise ValueError("calibration requires clean identified core and provider sources")
    for name, expected in metadata["packages"].items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError("provider package content does not match its identity")
    return dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), source_sha256=hashlib.sha256((ROOT / "benchmarks/bench_accelerator_cost.mojo").read_bytes()).hexdigest(), packages=metadata, flags=["-O3", "--fp-mode=contract=off"])


def build_runner(binary, package):
    binary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="accelerator-cost-consumer-") as directory:
        source = Path(directory) / "bench_accelerator_cost.mojo"
        source.write_bytes((ROOT / "benchmarks/bench_accelerator_cost.mojo").read_bytes())
        subprocess.run(["mojo", "build", "-O3", "--fp-mode=contract=off", "-I", str(package.resolve()), str(source), "-o", str(binary.resolve())], cwd=directory, check=True)
    binary.with_suffix(".manifest.json").write_text(json.dumps(artifact_manifest(binary, package), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.build:
        if args.binary is None or args.package_dir is None:
            parser.error("--build requires --binary and --package-dir")
        build_runner(args.binary, args.package_dir)
    cases = matrix(args.samples)
    provenance = dict(recorded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), runner_sha256=hashlib.sha256((ROOT / "benchmarks/bench_accelerator_cost.mojo").read_bytes()).hexdigest(), exposure="generated training; named mechanisms held out from this fit but previously exposed during development", external_holdout_policy="scripts/bench_policy.py: TPC-DS and ClickBench validation only", method="fresh process per case; one explicit runtime; cold context cost retained; alternating warm CPU/accelerator pairs; parity outside timings; full preflight overhead measured separately and added to accelerator cost; CPU lower and accelerator upper residual envelopes fitted on training only")
    (args.output / "manifest.json").write_text(json.dumps(dict(provenance=provenance, cases=cases), indent=2) + "\n")
    if not args.collect:
        print("Wrote case manifest only; pass --collect with an identified provider binary to measure.")
        return
    if args.binary is None or args.package_dir is None or args.reps < 5:
        parser.error("--collect requires --binary, --package-dir and at least five repetitions")
    manifest = json.loads(args.binary.with_suffix(".manifest.json").read_text())
    if artifact_manifest(args.binary, args.package_dir) != manifest:
        raise ValueError("runner binary/source/package identity changed; rebuild before collecting")
    provenance["artifacts"] = manifest
    (args.output / "manifest.json").write_text(json.dumps(dict(provenance=provenance, cases=cases), indent=2) + "\n")
    bootstrap = args.output / "bootstrap.json"
    bootstrap.write_text(json.dumps(dict(format=FORMAT, memory_fraction=.8, records=[])))
    train = collect(args.binary, [c for c in cases if c["role"] == "train"], args.output / "train.jsonl", bootstrap, args.reps, args.timeout)
    table = fit(train)
    # This temporary table is used only by read-only select_auto calls to time
    # complete coefficient lookup. Collection always explicitly selects engines.
    timing_table = json.loads(json.dumps(table))
    for record in timing_table["records"]:
        record.update(validated=True, validation_queries=5)
    temporary = args.output / "timing-only-do-not-deploy.json"
    temporary.write_text(json.dumps(timing_table))
    heldout = collect(args.binary, [c for c in cases if c["role"] == "heldout"], args.output / "heldout.jsonl", temporary, args.reps, args.timeout)
    regret = evaluate(table, heldout, {r["query_id"] for r in train})
    (args.output / "calibration.json").write_text(json.dumps(table, indent=2) + "\n")
    (args.output / "regret.json").write_text(json.dumps(dict(provenance=provenance, **regret), indent=2) + "\n")
    temporary.unlink()
    print("Calibration written; only records with validated=true are eligible for auto.")


if __name__ == "__main__":
    main()
