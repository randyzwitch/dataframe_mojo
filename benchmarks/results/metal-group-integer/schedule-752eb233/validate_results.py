#!/usr/bin/env python3
"""Validate an archived paired group-schedule benchmark without hardware."""

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


class ValidationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    require(path.is_file(), f"Missing archive file: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"Invalid JSON {path}: {error}") from error


def load_harness(path):
    require(path.is_file(), f"Missing archived benchmark harness: {path}")
    spec = importlib.util.spec_from_file_location("archived_bench_metal", path)
    require(spec is not None and spec.loader is not None, "Cannot load harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_route(rows, workload, variant):
    parallel = rows == 1_048_576 and (
        workload == "group_skew"
        or (workload == "group_hash" and variant == "nulls")
    )
    return "mojo_gpu_grouped_int_partials" if parallel else "legacy"


def validate_plan(plan):
    require(plan.get("candidate_engine")
            == "752eb23322ef9d18e7f2d6b950f733d5e0d8a340",
            "Unexpected candidate engine pin")
    require(plan.get("candidate_pin")
            == "a61e8dd98cf4550b2f4988a6ab09139a00469388",
            "Unexpected candidate measurement pin")
    for field, expected in {
        "rows": [4097, 1048576],
        "workloads": ["group_hash", "group_high", "group_skew"],
        "variants": ["base", "sorted", "nulls"],
        "rounds": 3,
        "reps": 3,
        "warmups": 1,
        "timeout_seconds": 300,
    }.items():
        require(plan.get(field) == expected,
                f"Unexpected plan.{field}: {plan.get(field)!r}")
    expected_harness = {
        "benchmarks/bench_metal.mojo":
            "704598dab6810b1197074b35400839e3ca298ba1a04755cf35c8671fd3552e97",
        "benchmarks/tools/bench_metal.py":
            "09bcbbb472ef308a945b4d4d33783c6b832c34cce9359ac1e749f3ac98618fc7",
    }
    require(plan.get("benchmark_file_sha256") == expected_harness,
            "Unexpected shared benchmark harness pins")
    expected_references = {
        "frozen": (
            "5a07d32fd8c2e7fbcc65ae717a4949dedbe07bad",
            "fde59b3869eeb50785cc37b8b2e92234f531d320d6cf5b5b42920bc7aa937633",
            "579b19cbd598cf423107a282daccf7ecc154fdae820700d6290b91fb917f4c32",
        ),
        "intermediate": (
            "f758ce9a067cd3ddf67b0c3393f5d26d684ddeb0",
            "ea9edee7a88d12d6e94d84fa3aafb9a6e78dad72308474d4c857c0f44a4b9185",
            "f10aba9f86f6e194c817bdd8d33bf4bce2ee81f4c577a696d672e9d7bb535b1f",
        ),
        "previous": (
            "b3e231a652d5e64571a2ab416e93d037aa06afbb",
            "8e6a646f1aa5579ba454b6795fa0c9bc6ed38c2f3e3e6869923048342bcbee44",
            "11b0037108c776e23b6bb9d6dd3389510982f92f28439cf4374ebbc2c1362261",
        ),
    }
    refs = plan.get("references")
    require(isinstance(refs, list) and len(refs) == len(expected_references),
            "Wrong reference set in plan")
    for ref in refs:
        expected = expected_references.get(ref.get("label"))
        require(expected is not None, f"Unexpected reference: {ref.get('label')}")
        actual = (ref.get("measurement_pin"), ref.get("binary_sha256"),
                  ref.get("engine_content_sha256"))
        require(actual == expected,
                f"Reference pin mismatch for {ref.get('label')}")


def safe_log_path(results_json, relative):
    require(isinstance(relative, str) and relative, "Missing raw log path")
    root = (results_json.parent / "results").resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"Raw log escapes results directory: {relative}")
    require(path.is_file(), f"Missing raw worker log: {path}")
    return path


def verify_selection(data, plan, smoke):
    selection = data.get("selection")
    require(isinstance(selection, dict), "Missing benchmark selection metadata")
    expected = {
        "rows": [4097] if smoke else plan["rows"],
        "workloads": plan["workloads"],
        "variants": plan["variants"],
        "rounds": 1 if smoke else plan["rounds"],
        "reps": 1 if smoke else plan["reps"],
        "warmups": 0 if smoke else plan["warmups"],
        "threads": 8,
        "timeout_seconds": plan["timeout_seconds"],
    }
    for name, value in expected.items():
        require(selection.get(name) == value,
                f"Unexpected selection.{name}: {selection.get(name)!r}")


def check_receipt(receipt, revision, engine_content, benchmark_hashes,
                  binary_hash=None):
    require(isinstance(receipt, dict), "Missing binary build receipt")
    require(receipt.get("status") == "verified", "Unverified binary receipt")
    require(receipt.get("git_commit") == revision,
            f"Wrong binary source commit: {receipt.get('git_commit')}")
    require(receipt.get("engine_content_sha256") == engine_content,
            "Wrong binary engine content fingerprint")
    require(isinstance(engine_content, str) and len(engine_content) == 64
            and all(char in "0123456789abcdef" for char in engine_content),
            "Invalid binary engine content fingerprint")
    require(receipt.get("git_status") == ""
            and receipt.get("engine_status") == "",
            "Binary receipt is dirty or missing clean-source status")
    require(
        receipt.get("benchmark_source_sha256")
        == benchmark_hashes["benchmarks/bench_metal.mojo"],
        "Binary receipt has wrong benchmark source hash",
    )
    require(
        receipt.get("driver_source_sha256")
        == benchmark_hashes["benchmarks/tools/bench_metal.py"],
        "Binary receipt has wrong benchmark driver hash",
    )
    require(isinstance(receipt.get("binary_sha256"), str)
            and len(receipt["binary_sha256"]) == 64
            and all(char in "0123456789abcdef"
                    for char in receipt["binary_sha256"]),
            "Missing binary SHA256")
    if binary_hash is not None:
        require(receipt["binary_sha256"] == binary_hash,
                "Binary receipt SHA does not match the plan/result")


def validate_result(root, folder, plan, harness, *, smoke, reference=None):
    result_path = root / folder / "results.json"
    data = read_json(result_path)
    verify_selection(data, plan, smoke)

    rounds = 1 if smoke else plan["rounds"]
    reps = 1 if smoke else plan["reps"]
    rows = [4097] if smoke else plan["rows"]
    runtimes = ["mojo"] if smoke else ["mojo", "baseline"]
    expected = set(itertools.product(
        range(rounds), runtimes, rows, plan["workloads"], plan["variants"]
    ))
    runs = data.get("runs")
    require(isinstance(runs, list), f"Missing worker list in {folder}")
    observed = set()
    for run in runs:
        require(isinstance(run, dict), f"Invalid worker entry in {folder}")
        key = (run.get("round"), run.get("runtime"), run.get("rows"),
               run.get("workload"), run.get("variant"))
        require(key in expected, f"Unexpected worker key in {folder}: {key}")
        require(key not in observed, f"Duplicate worker key in {folder}: {key}")
        observed.add(key)
        require(run.get("status") == "ok" and run.get("validated") is True,
                f"Unsuccessful/unvalidated worker: {key}")
        samples = run.get("samples")
        require(isinstance(samples, list) and len(samples) == reps,
                f"Wrong sample count for {key}")
        for sample in samples:
            require(isinstance(sample.get("metal_ns"), int)
                    and sample["metal_ns"] > 0
                    and isinstance(sample.get("cpu_ns"), int)
                    and sample["cpu_ns"] > 0,
                    f"Invalid timing sample for {key}")
        output_rows = run.get("output_rows")
        require(isinstance(output_rows, int) and not isinstance(output_rows, bool)
                and 0 <= output_rows <= run["rows"],
                f"Invalid group output row count for {key}")
        profile = run.get("profile")
        require(isinstance(profile, dict), f"Missing profile for {key}")
        for field in ("peak_requested_bytes", "kernel_launches", "upload_bytes",
                      "download_bytes", "synchronizations"):
            value = profile.get(field)
            require(isinstance(value, int) and not isinstance(value, bool)
                    and value >= 0, f"Missing/invalid profile.{field} for {key}")
        if run["runtime"] == "mojo":
            route = profile.get("region_0_grouped_int_algorithm")
            require(route == expect_route(run["rows"], run["workload"],
                                          run["variant"]),
                    f"Wrong native/legacy route {route!r} for {key}")

        raw_log = safe_log_path(result_path, run.get("log"))
        parsed = harness.parse_run(raw_log.read_text(), reps)
        for name, value in parsed.items():
            require(run.get(name) == value,
                    f"Raw log mismatch for {key} field {name}")

    require(observed == expected,
            f"Incomplete coverage in {folder}: {len(observed)}/{len(expected)}")
    expected_count = len(expected)
    require(len(observed) == expected_count,
            f"Unexpected worker count in {folder}: {len(observed)}")

    metadata = data.get("metadata")
    require(isinstance(metadata, dict), f"Missing metadata in {folder}")
    receipts = metadata.get("binary_receipts")
    binary_hashes = metadata.get("binaries_sha256")
    require(isinstance(receipts, dict) and isinstance(binary_hashes, dict),
            f"Missing binary receipt metadata in {folder}")
    required_receipts = {"mojo"} if smoke else {"mojo", "baseline"}
    require(set(receipts) == required_receipts,
            f"Unexpected binary receipt set in {folder}: {sorted(receipts)}")
    require(set(binary_hashes) == required_receipts,
            f"Unexpected binary hash set in {folder}: {sorted(binary_hashes)}")

    candidate_receipt = receipts["mojo"]
    # Pin the engine fingerprint independently computed from qualified revision
    # 752eb233, then require the same receipt across every comparison.
    candidate_engine_content = "5bae65bb1f5d7fe8dac2df79a8e5a5b1767f9d069030d751eb12a0317af72247"
    check_receipt(candidate_receipt, plan["candidate_pin"],
                  candidate_engine_content,
                  plan["benchmark_file_sha256"], binary_hashes["mojo"])
    require(candidate_receipt.get("binary_sha256") == binary_hashes["mojo"],
            f"Candidate binary hash mismatch in {folder}")

    baseline_receipt = None
    if reference is not None:
        baseline_receipt = receipts["baseline"]
        check_receipt(baseline_receipt, reference["measurement_pin"],
                      reference["engine_content_sha256"],
                      plan["benchmark_file_sha256"],
                      reference["binary_sha256"])
        require(binary_hashes["baseline"] == reference["binary_sha256"],
                f"Wrong reused baseline binary in {folder}")
        harness.validate_paired_build_receipts(baseline_receipt,
                                               candidate_receipt)
    return {
        "candidate_receipt": candidate_receipt,
        "candidate_engine_content": candidate_engine_content,
        "baseline_receipt": baseline_receipt,
        "candidate_hash": binary_hashes["mojo"],
        "workers": len(observed),
    }


def validate_passed_gate(root, plan):
    validation = root / "validation"
    receipt_path = validation / "metal-oct20-group-schedule-v2-receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = read_json(receipt_path)
    qualification = read_json(root / "qualification.json")
    require(sha(receipt_path) == qualification.get("gate_receipt_sha256"),
            "Qualification gate receipt hash mismatch")
    gates = receipt.get("gates")
    require(isinstance(gates, list) and len(gates) == 1,
            "Passed receipt must contain exactly one gate")
    gate = gates[0]
    require(qualification.get("gate") == gate,
            "Qualification gate object differs from archived receipt")
    require(gate.get("label") == "group-schedule",
            "Unexpected passed gate label")
    require(gate.get("revision") == plan["candidate_engine"],
            "Passed gate engine revision differs from plan")
    tree_id = gate.get("engine_tree")
    require(isinstance(tree_id, str) and len(tree_id) == 40
            and all(char in "0123456789abcdef" for char in tree_id),
            "Passed gate has an invalid engine tree id")
    require(gate.get("clean") is True and gate.get("build_exit") == 0
            and gate.get("test_exit") == 0,
            "Qualification receipt does not record a passing clean gate")
    files = gate.get("files")
    require(isinstance(files, dict), "Passed receipt has no artifact hashes")
    seen_build = seen_test = False
    for path, digest in files.items():
        name = Path(path).name
        if name.endswith("-build.log"):
            seen_build = True
        elif name.endswith("-test.log"):
            seen_test = True
        else:
            continue  # The gate executable hash is retained in the receipt.
        archived = validation / name
        require(archived.is_file() and sha(archived) == digest,
                f"Passed gate log mismatch/missing: {name}")
    require(seen_build and seen_test,
            "Passed gate receipt lacks build/test log hashes")
    require(receipt_bytes,
            "Empty passed gate receipt")
    return sha(receipt_path)


def validate_failed_gate(root):
    folder = root / "validation" / "failed"
    path = folder / "metal-oct20-group-schedule-failed-receipt.json"
    receipt = read_json(path)
    require(sha(path)
            == "07a4a62f61a7332e52ad4d3f2e8620a685b665dcb4e66ed73100f3da3b07a3ac",
            "Retained failed v1 receipt hash changed")
    gates = receipt.get("gates")
    require(isinstance(gates, list) and len(gates) == 1,
            "Failed receipt must contain exactly one gate")
    gate = gates[0]
    require(gate.get("label") == "group-schedule-failed",
            "Retained failed gate has unexpected label")
    require(gate.get("revision") == "e610ca2cd8220dad36b822b4ef7650855cdee95f",
            "Retained failure is not the recorded v1 candidate")
    require(gate.get("clean") is True and gate.get("build_exit") == 0
            and gate.get("test_exit") == 1,
            "Retained failed gate must show build 0 / test 1")
    files = gate.get("files")
    require(isinstance(files, dict), "Failed receipt has no artifact hashes")
    seen_build = seen_test = False
    for original, digest in files.items():
        name = Path(original).name
        if name.endswith("-build.log"):
            seen_build = True
        elif name.endswith("-test.log"):
            seen_test = True
        else:
            continue
        archived = folder / name
        require(archived.is_file() and sha(archived) == digest,
                f"Failed gate log mismatch/missing: {name}")
    require(seen_build and seen_test,
            "Failed gate receipt lacks build/test log hashes")
    return gate, sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path,
                        help="Root containing plan.json and archived results")
    root = parser.parse_args().archive_root.resolve()
    plan = read_json(root / "plan.json")
    validate_plan(plan)
    for name, digest in plan["benchmark_file_sha256"].items():
        harness_path = root / "harness" / Path(name).name
        require(sha(harness_path) == digest,
                f"Archived harness hash mismatch: {name}")
    harness = load_harness(root / "harness" / "bench_metal.py")

    passed_receipt_hash = validate_passed_gate(root, plan)
    failed_gate, failed_receipt_hash = validate_failed_gate(root)

    smoke = validate_result(root, "smoke", plan, harness, smoke=True)
    pair_results = {}
    pair_folders = {
        "frozen": "frozen-pair",
        "intermediate": "intermediate-pair",
        "previous": "previous-pair",
    }
    references = {reference["label"]: reference
                  for reference in plan["references"]}
    require(set(pair_folders) == set(references),
            "Pair folders do not match plan references")
    for label, folder in pair_folders.items():
        pair_results[label] = validate_result(
            root, folder, plan, harness, smoke=False,
            reference=references[label],
        )

    # Every pair reuses the candidate binary built by the candidate smoke.
    candidate_receipt = smoke["candidate_receipt"]
    candidate_hash = smoke["candidate_hash"]
    total_workers = smoke["workers"]
    for label, result in pair_results.items():
        require(result["candidate_receipt"] == candidate_receipt,
                f"Candidate receipt changed in {label} pair")
        require(result["candidate_hash"] == candidate_hash,
                f"Candidate binary changed in {label} pair")
        require(result["candidate_engine_content"]
                == smoke["candidate_engine_content"],
                f"Candidate engine content changed in {label} pair")
        total_workers += result["workers"]

    require(total_workers == 333,
            f"Expected 333 workers, found {total_workers}")
    followup = read_json(root / "skew-followup-plan.json")
    followup_plan = dict(plan)
    followup_plan.update(rows=[4097], workloads=["group_skew"], variants=["sorted"], rounds=10, reps=5, warmups=1)
    require(followup.get("plan") == followup_plan, "Followup plan differs from declared longer check")
    require(followup.get("reference") == references["previous"], "Followup reference changed")
    repeated = validate_result(root, "skew-followup", followup_plan, harness, smoke=False, reference=references["previous"])
    require(repeated["workers"] == 20, "Expected 20 followup workers")
    require(repeated["candidate_receipt"] == candidate_receipt, "Followup candidate binary or receipt changed")
    require(repeated["baseline_receipt"] == pair_results["previous"]["baseline_receipt"], "Followup previous binary or receipt changed")
    print("PASS: 20 followup workers; original comparison remains separately retained")
    gate_hash = passed_receipt_hash
    print(f"PASS: 333 workers; passed gate receipt SHA256 {gate_hash}")
    print(
        "RETAINED FAILED GATE: "
        f"revision={failed_gate['revision']} test_exit={failed_gate['test_exit']}; "
        "recorded as a failure, not a pass"
    )
    print(f"FAILED RECEIPT SHA256 {failed_receipt_hash}")


if __name__ == "__main__":
    main()
