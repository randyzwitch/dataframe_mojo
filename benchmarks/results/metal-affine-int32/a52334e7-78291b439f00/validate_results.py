#!/usr/bin/env python3
"""Read-only structural check for a completed paired Metal benchmark result."""
from __future__ import annotations

import itertools
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RESULTS.json", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    data = json.loads(path.read_text())
    selection = data["selection"]
    expected = set(
        itertools.product(
            range(selection["rounds"]),
            ("baseline", "mojo"),
            selection["rows"],
            selection["workloads"],
            selection["variants"],
        )
    )
    keyed = {}
    errors = []
    for run in data["runs"]:
        key = (run["round"], run["runtime"], run["rows"], run["workload"], run["variant"])
        if key in keyed:
            errors.append(f"duplicate run key: {key}")
        keyed[key] = run
        if run.get("status") != "ok" or run.get("validated") is not True:
            errors.append(f"run did not complete and validate: {key}")
        if not isinstance(run.get("output_rows"), int) or not 0 <= run["output_rows"] <= run["rows"]:
            errors.append(f"invalid output row count: {key}")
        if len(run.get("samples", [])) != selection["reps"]:
            errors.append(f"wrong timed repetition count: {key}")
        if not run.get("first_use_ns", {}).get("cpu") or not run.get("first_use_ns", {}).get("metal"):
            errors.append(f"missing first-use CPU/Metal samples: {key}")
        profile = run.get("profile", {})
        for field in ("upload_bytes", "download_bytes", "peak_requested_bytes", "kernel_launches"):
            if not isinstance(profile.get(field), int) or profile[field] < 0:
                errors.append(f"missing/invalid {field}: {key}")
        for sample in run.get("samples", []):
            if sample.get("cpu_ns", 0) <= 0 or sample.get("metal_ns", 0) <= 0:
                errors.append(f"nonpositive timed CPU/Metal sample: {key}")
    missing, extra = expected - keyed.keys(), keyed.keys() - expected
    if missing:
        errors.append(f"missing run keys: {len(missing)}")
    if extra:
        errors.append(f"unexpected run keys: {len(extra)}")
    receipts = data["metadata"].get("binary_receipts", {})
    for name in ("baseline", "mojo"):
        if receipts.get(name, {}).get("status") != "verified":
            errors.append(f"{name} build receipt is not verified")
    if data["metadata"].get("engine_status") != "":
        errors.append("engine working tree was not clean")
    if data["metadata"].get("source_tree", {}).get("dataframe_status") != "":
        errors.append("dataframe working tree was not clean")
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"VALIDATION OK: {len(keyed)}/{len(expected)} unique run keys")
    print(f"All runs: status=ok, validated=true, complete output, {selection['reps']} timed repetitions, first-use CPU+Metal present")
    print("Build receipts: baseline=verified, mojo=verified; engine/dataframe source trees clean")
    print("Runtime counts:", dict(Counter(r["runtime"] for r in data["runs"])))
    print("Rows/workloads/variants/rounds:", selection["rows"], selection["workloads"], selection["variants"], selection["rounds"])

    by_cell = {}
    for run in data["runs"]:
        for sample in run["samples"]:
            by_cell.setdefault((run["rows"], run["workload"], run["variant"], run["runtime"]), {"cpu": [], "metal": []})
            cell = by_cell[(run["rows"], run["workload"], run["variant"], run["runtime"])]
            cell["cpu"].append(sample["cpu_ns"])
            cell["metal"].append(sample["metal_ns"])
    candidate_cpu_wins = 0
    for rows, workload, variant in itertools.product(selection["rows"], selection["workloads"], selection["variants"]):
        cand = by_cell[(rows, workload, variant, "mojo")]
        if statistics.median(cand["cpu"]) < statistics.median(cand["metal"]):
            candidate_cpu_wins += 1
    print(f"Candidate warm median CPU faster than Metal: {candidate_cpu_wins}/{len(selection['rows']) * len(selection['workloads']) * len(selection['variants'])} case cells")
    output_counts = {}
    for run in data["runs"]:
        output_counts.setdefault((run["rows"], run["workload"], run["variant"]), set()).add(run["output_rows"])
    inconsistent = [key for key, values in output_counts.items() if len(values) != 1]
    if inconsistent:
        print("WARNING: output row counts varied across runs:", inconsistent)
        return 1
    print(f"Output cardinalities stable across runtimes/rounds: {len(output_counts)} case cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
