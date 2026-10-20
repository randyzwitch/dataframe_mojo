"""Guarded paired Int32 single-key Metal join benchmark.

This script is prepared for a later serialized Mac run. It refuses to start
while the declared correctness jobs are active, if gate receipts are missing,
if either pinned tree or shared harness changed, or if output paths already
exist. --self-test exercises receipt auditing without hardware.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "plan.json").read_text())
TREES = Path(PLAN["worktree_root"])
BASE = TREES / PLAN["baseline_remote"]
CANDIDATE = TREES / PLAN["candidate_remote"]
PIXI = PLAN["pixi"]
MANIFEST = PLAN["manifest"]
LOCK = ROOT / ".run_pairs.lock"


def command(args, cwd=None):
    return subprocess.check_output(
        args, cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint(root):
    """Use the benchmark harness's stable tree-content fingerprint."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def expect(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_tree(tree, revision_key, engine_revision_key, engine_tree_key,
               engine_content_key):
    tree = Path(tree)
    expected_revision = PLAN[revision_key]
    expected_engine_revision = PLAN[engine_revision_key]
    expect(tree.is_dir(), f"Missing pinned worktree: {tree}")
    expect(
        command(["git", "rev-parse", "HEAD"], cwd=tree) == expected_revision,
        f"Unexpected HEAD in {tree}",
    )
    expect(
        not command(["git", "status", "--porcelain"], cwd=tree),
        f"Dirty worktree: {tree}",
    )
    current_tree = command(["git", "rev-parse", "HEAD:dataframe"], cwd=tree)
    pinned_tree = command(
        ["git", "rev-parse", f"{expected_engine_revision}:dataframe"],
        cwd=tree,
    )
    expect(
        current_tree == PLAN[engine_tree_key] == pinned_tree,
        f"Engine source tree changed in {tree}",
    )
    expect(
        not command(
            ["git", "diff", "--binary", expected_engine_revision, "HEAD", "--", "dataframe"],
            cwd=tree,
        ),
        f"Engine source differs from its pin in {tree}",
    )
    expect(
        source_fingerprint(tree / "dataframe") == PLAN[engine_content_key],
        f"Engine source content hash changed in {tree}",
    )
    for file, expected in PLAN["harness_file_sha256"].items():
        expect(
            sha(tree / file) == expected,
            f"Benchmark harness changed: {tree / file}",
        )


def check_shared_harness():
    for file in PLAN["harness_file_sha256"]:
        expect(
            sha(BASE / file) == sha(CANDIDATE / file),
            f"Baseline/candidate harness differs: {file}",
        )


def check_gate_receipt():
    path = Path(PLAN["gate_receipt"])
    expect(path.is_file(), f"Missing passed-gate receipt: {path}")
    expect(
        sha(path) == PLAN["gate_receipt_sha256"],
        "Passed-gate receipt hash changed; refresh and review the plan",
    )
    receipt = json.loads(path.read_text())
    for required in PLAN["required_gates"]:
        matches = [
            gate
            for gate in receipt.get("gates", [])
            if gate.get("label") == required["label"]
        ]
        expect(len(matches) == 1, f"Missing/duplicate gate: {required['label']}")
        gate = matches[0]
        for field in (
            "revision",
            "engine_tree",
            "build_exit",
            "test_exit",
            "clean",
        ):
            expect(
                gate.get(field) == required[field],
                f"Gate {required['label']} has wrong {field}",
            )
        for original_path, expected_hash in required["artifacts"].items():
            artifact = Path(original_path)
            if not artifact.is_file():
                artifact = Path(PLAN["gate_artifact_fallback_dir"]) / artifact.name
            expect(artifact.is_file(), f"Missing gate artifact: {original_path}")
            expect(
                sha(artifact) == expected_hash,
                f"Gate artifact hash changed: {original_path}",
            )


def check_guard_pids():
    live = []
    for pid in PLAN["guard_pids"]:
        try:
            os.kill(pid, 0)
            live.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            live.append(pid)
    expect(not live, "Correctness jobs still live: " + ", ".join(map(str, live)))


def check_busy_mac():
    lines = command(["ps", "-Ao", "pid=,command="]).splitlines()
    patterns = (
        re.compile(r"\bmojo\s+(?:test|build)\b"),
        re.compile(r"(?:^|/)bench_metal\.py(?:\s|$)"),
        re.compile(r"metal-[^\s]*(?:queue|gate)\.py"),
    )
    busy = []
    for line in lines:
        match = re.match(r"\s*(\d+)\s+(.*)", line)
        if not match or int(match.group(1)) == os.getpid():
            continue
        if any(pattern.search(match.group(2)) for pattern in patterns):
            busy.append(line.strip())
    expect(not busy, "Mac has an active compiler/test/benchmark job: " + repr(busy))


def output_rows_limit(workload, rows):
    # A self-join with N rows and floor(N/2) keys can have one three-row
    # key at 4097 rows. Its output is 8197, slightly above 2N. Three N is a
    # simple conservative bound; join_fanout is exactly bounded by 4N.
    multiplier = 3 if workload == "join_hash" else 4
    return multiplier * rows


def check_binary_receipt(receipt, expected_revision, expected_engine_hash):
    expect(isinstance(receipt, dict), "Missing or invalid binary receipt")
    expect(receipt.get("status") == "verified", "Binary receipt is unverified")
    expect(
        receipt.get("git_commit") == expected_revision,
        f"Binary was built from an unexpected commit: {receipt.get('git_commit')}",
    )
    expect(
        receipt.get("engine_content_sha256") == expected_engine_hash,
        "Binary receipt has an unexpected engine source hash",
    )
    for field, plan_field in (
        ("benchmark_source_sha256", "benchmarks/bench_metal.mojo"),
        ("driver_source_sha256", "benchmarks/tools/bench_metal.py"),
    ):
        expect(
            receipt.get(field) == PLAN["harness_file_sha256"][plan_field],
            f"Binary receipt has an unexpected {field}",
        )
    expect(
        receipt.get("pixi_lock_sha256") == PLAN["pixi_lock_sha256"],
        "Binary receipt has an unexpected pixi.lock hash",
    )


def audit_payload(data, rows, rounds, reps, runtimes, native,
                  workloads=None, variants=None, mojo_role="candidate"):
    workloads = PLAN["workloads"] if workloads is None else workloads
    variants = PLAN["variants"] if variants is None else variants
    expected = set(
        itertools.product(
            range(rounds), runtimes, rows, workloads, variants
        )
    )
    runs = data.get("runs", [])
    keys = [
        (r["round"], r["runtime"], r["rows"], r["workload"], r["variant"])
        for r in runs
    ]
    expect(
        len(keys) == len(expected) and set(keys) == expected,
        "Missing or duplicate benchmark workers",
    )
    failures = [
        run
        for run in runs
        if run.get("status") != "ok"
        or not run.get("validated")
        or len(run.get("samples", [])) != reps
    ]
    expect(not failures, f"Worker validation/sample failure: {failures[:2]}")
    for run in runs:
        profile = run.get("profile", {})
        peak_bytes = profile.get("peak_requested_bytes")
        output_rows = run.get("output_rows")
        expect(
            isinstance(peak_bytes, (int, float))
            and not isinstance(peak_bytes, bool)
            and math.isfinite(peak_bytes)
            and peak_bytes >= 0,
            f"Missing or invalid peak_requested_bytes: {run}",
        )
        expect(
            isinstance(output_rows, (int, float))
            and not isinstance(output_rows, bool)
            and math.isfinite(output_rows)
            and output_rows >= 0,
            f"Missing or invalid output_rows: {run}",
        )
        expect(
            peak_bytes <= PLAN["reported_peak_ceiling_bytes"],
            f"Reported peak exceeds post-run ceiling: {run}",
        )
        expect(
            output_rows <= output_rows_limit(run["workload"], run["rows"]),
            f"Join fanout exceeds its declared bound: {run}",
        )
        if native and run["runtime"] == "mojo":
            expect(
                profile.get("region_0_algorithm") == PLAN["profile_algorithm"],
                "Candidate did not use the native hash join path",
            )
    metadata = data.get("metadata")
    expect(isinstance(metadata, dict), "Missing or invalid benchmark metadata")
    receipts = metadata.get("binary_receipts")
    expect(isinstance(receipts, dict), "Missing or invalid binary receipts")
    mojo_revision_key = f"{mojo_role}_revision"
    mojo_hash_key = f"{mojo_role}_engine_content_sha256"
    expected_receipts = {
        "mojo": (
            PLAN[mojo_revision_key],
            PLAN[mojo_hash_key],
        )
    }
    if "baseline" in runtimes:
        expected_receipts["baseline"] = (
            PLAN["baseline_revision"],
            PLAN["baseline_engine_content_sha256"],
        )
    expect(
        set(receipts) == set(expected_receipts),
        f"Unexpected binary receipts: {sorted(receipts)}",
    )
    for name, (revision, engine_hash) in expected_receipts.items():
        check_binary_receipt(receipts[name], revision, engine_hash)
    paired = {}
    for run in runs:
        paired.setdefault(
            (run["round"], run["rows"], run["workload"], run["variant"]), {}
        )[run["runtime"]] = run.get("output_rows")
    for key, outputs in paired.items():
        if set(outputs) == {"mojo", "baseline"}:
            expect(
                outputs["mojo"] == outputs["baseline"],
                f"Baseline/candidate output row count differs for {key}",
            )
    return len(keys)


def check_empty_outputs():
    outputs = [
        BASE / "build" / "bench_metal",
        BASE / "build" / "bench_metal.receipt.json",
        CANDIDATE / "build" / "bench_metal",
        CANDIDATE / "build" / "bench_metal.receipt.json",
    ]
    outputs += [
        BASE / "build" / label / "results.json"
        for label in ("join-baseline-smoke", "join-paired")
    ]
    outputs += [
        CANDIDATE / "build" / label / "results.json"
        for label in ("join-candidate-smoke", "join-paired")
    ]
    outputs += [ROOT / f"{label}.log" for label in (
        "join-baseline-smoke", "join-candidate-smoke", "join-paired"
    )]
    existing = [str(path) for path in outputs if path.exists()]
    expect(not existing, "Refusing to overwrite existing artifacts: " + repr(existing))


def preflight():
    expect(platform.system() == "Darwin" and platform.machine() == "arm64",
           "This plan is Mac arm64 only")
    check_guard_pids()
    check_busy_mac()
    check_tree(
        BASE,
        "baseline_revision",
        "baseline_engine_revision",
        "baseline_engine_tree",
        "baseline_engine_content_sha256",
    )
    check_tree(
        CANDIDATE,
        "candidate_revision",
        "candidate_engine_revision",
        "candidate_engine_tree",
        "candidate_engine_content_sha256",
    )
    check_shared_harness()
    check_gate_receipt()
    check_empty_outputs()


def run(tree, label, rows, rounds, reps, warmups, native=False, baseline=None,
        mojo_role="candidate"):
    output = tree / "build" / label / "results.json"
    expect(not output.exists(), f"Refusing to overwrite {output}")
    args = [
        PIXI,
        "run",
        "--manifest-path",
        MANIFEST,
        "--locked",
        "-e",
        "oracle",
        "python3",
        str(tree / "benchmarks/tools/bench_metal.py"),
        "--binary",
        str(tree / "build/bench_metal"),
        "--rows",
        ",".join(map(str, rows)),
        "--workloads",
        ",".join(PLAN["workloads"] if baseline else ["join_hash"]),
        "--variants",
        ",".join(PLAN["variants"] if baseline else ["base"]),
        "--rounds",
        str(rounds),
        "--reps",
        str(reps),
        "--warmups",
        str(warmups),
        "--threads",
        str(PLAN["threads"]),
        "--timeout-seconds",
        str(PLAN["timeout_seconds"]),
        "--output",
        str(output),
    ]
    if baseline:
        args += ["--skip-build", "--baseline-binary", str(baseline / "build/bench_metal")]
    print("START", label, flush=True)
    with (ROOT / f"{label}.log").open("x") as log:
        result = subprocess.run(args, cwd=tree, stdout=log, stderr=subprocess.STDOUT)
    print("STATUS", label, result.returncode, flush=True)
    expect(result.returncode == 0 and output.is_file(),
           f"Benchmark failed; retain logs and do not restart: {label}")
    data = json.loads(output.read_text())
    if baseline:
        audit_payload(
            data,
            PLAN["rows"],
            PLAN["rounds"],
            PLAN["reps"],
            ["mojo", "baseline"],
            native,
            mojo_role=mojo_role,
        )
    else:
        selected_rows = [4097]
        audit_payload(
            data,
            selected_rows,
            1,
            1,
            ["mojo"],
            native,
            workloads=["join_hash"],
            variants=["base"],
            mojo_role=mojo_role,
        )
    return data


def self_test():
    expect(
        PLAN["expected_process_workers"]
        == PLAN["rounds"] * len(PLAN["rows"]) * len(PLAN["workloads"])
        * len(PLAN["variants"]) * 2,
        "Plan worker count is inconsistent",
    )
    def receipt(revision, engine_hash):
        return {
            "status": "verified",
            "git_commit": revision,
            "engine_content_sha256": engine_hash,
            "benchmark_source_sha256": PLAN["harness_file_sha256"][
                "benchmarks/bench_metal.mojo"
            ],
            "driver_source_sha256": PLAN["harness_file_sha256"][
                "benchmarks/tools/bench_metal.py"
            ],
            "pixi_lock_sha256": PLAN["pixi_lock_sha256"],
        }

    synthetic = {
        "runs": [],
        "metadata": {
            "binary_receipts": {
                "mojo": receipt(
                    PLAN["candidate_revision"],
                    PLAN["candidate_engine_content_sha256"],
                ),
                "baseline": receipt(
                    PLAN["baseline_revision"],
                    PLAN["baseline_engine_content_sha256"],
                ),
            }
        },
    }
    for round_id, runtime, rows, workload, variant in itertools.product(
        range(1), ["mojo", "baseline"], [4097], PLAN["workloads"],
        PLAN["variants"]
    ):
        synthetic["runs"].append(
            {
                "round": round_id,
                "runtime": runtime,
                "rows": rows,
                "workload": workload,
                "variant": variant,
                "status": "ok",
                "validated": True,
                "samples": [{"index": i} for i in range(3)],
                "output_rows": min(rows * (3 if workload == "join_hash" else 4), 10_000),
                "profile": {
                    "peak_requested_bytes": 1234,
                    "region_0_algorithm": (
                        PLAN["profile_algorithm"] if runtime == "mojo" else "generic"
                    ),
                },
            }
        )
    expect(
        audit_payload(
            synthetic, [4097], 1, 3, ["mojo", "baseline"], True,
            workloads=PLAN["workloads"], variants=PLAN["variants"],
        )
        == len(synthetic["runs"]),
        "Valid synthetic receipts failed",
    )
    smoke = {
        "metadata": {"binary_receipts": {"mojo": synthetic["metadata"][
            "binary_receipts"]["mojo"]}},
        "runs": [
            {
                "round": 0,
                "runtime": "mojo",
                "rows": 4097,
                "workload": "join_hash",
                "variant": "base",
                "status": "ok",
                "validated": True,
                "samples": [{"index": 0}],
                "output_rows": 8197,
                "profile": {
                    "peak_requested_bytes": 1234,
                    "region_0_algorithm": PLAN["profile_algorithm"],
                },
            }
        ],
    }
    expect(
        audit_payload(
            smoke,
            [4097],
            1,
            1,
            ["mojo"],
            True,
            workloads=["join_hash"],
            variants=["base"],
        ) == 1,
        "Valid native smoke receipt failed",
    )
    baseline_smoke = json.loads(json.dumps(smoke))
    baseline_smoke["metadata"]["binary_receipts"]["mojo"] = receipt(
        PLAN["baseline_revision"], PLAN["baseline_engine_content_sha256"]
    )
    expect(
        audit_payload(
            baseline_smoke,
            [4097],
            1,
            1,
            ["mojo"],
            False,
            workloads=["join_hash"],
            variants=["base"],
            mojo_role="baseline",
        ) == 1,
        "Valid baseline smoke receipt failed",
    )
    invalid = json.loads(json.dumps(synthetic))
    invalid["runs"][0]["profile"]["region_0_algorithm"] = "wrong"
    try:
        audit_payload(
            invalid, [4097], 1, 3, ["mojo", "baseline"], True,
            workloads=PLAN["workloads"], variants=PLAN["variants"],
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Native algorithm refusal self-test did not fail")
    invalid_receipt = json.loads(json.dumps(synthetic))
    invalid_receipt["metadata"]["binary_receipts"]["mojo"][
        "git_commit"
    ] = PLAN["baseline_revision"]
    try:
        audit_payload(
            invalid_receipt, [4097], 1, 3, ["mojo", "baseline"], True,
            workloads=PLAN["workloads"], variants=PLAN["variants"],
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Unexpected binary provenance self-test did not fail")
    missing_metadata = json.loads(json.dumps(smoke))
    missing_metadata.pop("metadata")
    try:
        audit_payload(
            missing_metadata, [4097], 1, 1, ["mojo"], True,
            workloads=["join_hash"], variants=["base"],
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Missing metadata self-test did not fail")
    missing_peak = json.loads(json.dumps(smoke))
    missing_peak["runs"][0]["profile"].pop("peak_requested_bytes")
    try:
        audit_payload(
            missing_peak, [4097], 1, 1, ["mojo"], True,
            workloads=["join_hash"], variants=["base"],
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Missing peak measurement self-test did not fail")
    missing_output_rows = json.loads(json.dumps(smoke))
    missing_output_rows["runs"][0].pop("output_rows")
    try:
        audit_payload(
            missing_output_rows, [4097], 1, 1, ["mojo"], True,
            workloads=["join_hash"], variants=["base"],
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Missing output row count self-test did not fail")
    negative_peak = json.loads(json.dumps(smoke))
    negative_peak["runs"][0]["profile"]["peak_requested_bytes"] = -1
    try:
        audit_payload(
            negative_peak, [4097], 1, 1, ["mojo"], True,
            workloads=["join_hash"], variants=["base"],
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Negative peak measurement self-test did not fail")
    print("PASS non-hardware pair audit self-test", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    fd = None
    lock_created = False
    try:
        preflight()
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        lock_created = True
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.close(fd)
        fd = None
        # Smoke/build receipts first, then the paired matrix. Every log is
        # exclusive-create and all result JSON files are checked before use.
        run(
            BASE,
            "join-baseline-smoke",
            [4097],
            1,
            1,
            0,
            mojo_role="baseline",
        )
        run(CANDIDATE, "join-candidate-smoke", [4097], 1, 1, 0, native=True)
        run(
            CANDIDATE,
            "join-paired",
            PLAN["rows"],
            PLAN["rounds"],
            PLAN["reps"],
            PLAN["warmups"],
            native=True,
            baseline=BASE,
        )
    finally:
        if fd is not None:
            os.close(fd)
        if lock_created and LOCK.exists():
            LOCK.unlink()


if __name__ == "__main__":
    main()
