"""Guarded paired top-k/radix Metal benchmark for a later Mac run.

Observed profile algorithm IDs remain generic (mojo_gpu_ordered_rows). Exact
sorter attribution is an inference from pinned planner admission, query shape,
and expected whole-query launch counts, not a direct profile label.
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
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "plan.json").read_text())
GATES = ROOT / "gates"
LOCK = ROOT / ".run_pairs.lock"


def expect(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def content_fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(rel).to_bytes(8, "big")); digest.update(rel)
        digest.update(len(data).to_bytes(8, "big")); digest.update(data)
    return digest.hexdigest()


def tree(role):
    item = PLAN["source_roles"][role]
    path = Path(PLAN["worktree_root"]) / item["remote"]
    expect(path.is_dir(), f"Missing staged {role} worktree: {path}")
    expect(command(["git", "rev-parse", "HEAD"], path) == item["measurement"], f"Unexpected {role} measurement commit")
    expect(not command(["git", "status", "--porcelain"], path), f"Dirty {role} worktree")
    current_tree = command(["git", "rev-parse", "HEAD:dataframe"], path)
    engine_tree = command(["git", "rev-parse", f"{item['engine']}:dataframe"], path)
    expect(current_tree == engine_tree == item["tree"], f"{role} engine tree changed")
    expect(not command(["git", "diff", "--binary", item["engine"], "HEAD", "--", "dataframe"], path), f"{role} engine source differs from pin")
    expect(content_fingerprint(path / "dataframe") == item["fingerprint"], f"{role} source fingerprint changed")
    for rel, expected in PLAN["harness_file_sha256"].items():
        expect(sha(path / rel) == expected, f"{role} shared harness changed: {rel}")
    expect(sha(path / "pixi.lock") == PLAN["pixi_lock_sha256"], f"{role} lockfile changed")
    return path


def check_bundles():
    for role, item in PLAN["source_roles"].items():
        p = ROOT / item["bundle"]
        expect(p.is_file() and sha(p) == item["bundle_sha256"], f"Missing/changed {role} bundle")
        command(["git", "bundle", "verify", str(p)])


def check_gates():
    expected = {
        "candidate": ("topk-restack-receipt.json", "topk-restack", "candidate"),
        "intermediate": ("radix-restack-receipt.json", "radix-restack", "intermediate"),
    }
    for role, (name, label, source_role) in expected.items():
        path = GATES / name
        receipt_plan = PLAN["gate_receipts"][role]
        expect(path.is_file() and sha(path) == receipt_plan["sha256"], f"Missing/changed {role} gate receipt")
        receipt = json.loads(path.read_text())
        matches = [g for g in receipt.get("gates", []) if g.get("label") == label]
        expect(len(matches) == 1, f"Expected exactly one {label} gate")
        gate = matches[0]
        source = PLAN["source_roles"][source_role]
        for key, value in (("revision", source["engine"]), ("engine_tree", source["tree"]), ("clean", True), ("build_exit", 0), ("test_exit", 0)):
            expect(gate.get(key) == value, f"{label} gate has invalid {key}")
        files = gate.get("files", {})
        expect(len(files) == 3, f"{label} gate artifact set is incomplete")
        for artifact, digest in files.items():
            p = Path(artifact)
            expect(p.is_file() and sha(p) == digest, f"Missing/changed gate artifact: {artifact}")


def check_pids():
    live = []
    for pid in PLAN["guard_pids"]:
        try:
            os.kill(pid, 0); live.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            live.append(pid)
    expect(not live, "Correctness jobs still live: " + ", ".join(map(str, live)))


def check_busy_mac():
    lines = command(["ps", "-Ao", "pid=,command="]).splitlines()
    patterns = (re.compile(r"\bmojo\s+(?:test|build)\b"), re.compile(r"\bbench_metal\.py\b"), re.compile(r"metal-[^\s]*(?:queue|gate)\.py"))
    busy = []
    for line in lines:
        m = re.match(r"\s*(\d+)\s+(.*)", line)
        if m and int(m.group(1)) != os.getpid() and any(p.search(m.group(2)) for p in patterns):
            busy.append(line.strip())
    expect(not busy, "Active compiler/test/benchmark job on Mac: " + repr(busy))


def expected_rows(workload, n):
    if workload == "sort_radix":
        return n
    if workload == "sort_topk31":
        return min(n, 31)
    if workload == "sort_topk1024":
        return min(n, 1024)
    raise RuntimeError(f"Unknown ordering workload {workload}")


def expected_launches(workload, n):
    table = PLAN["expected_candidate_launches"]
    return table[str(n)][workload]


def check_binary_receipt(receipt, role):
    item = PLAN["source_roles"][role]
    expect(isinstance(receipt, dict) and receipt.get("status") == "verified", f"Unverified {role} binary receipt")
    expect(receipt.get("git_commit") == item["measurement"], f"Wrong {role} binary commit")
    expect(receipt.get("engine_content_sha256") == item["fingerprint"], f"Wrong {role} binary engine hash")
    for key, file in (("benchmark_source_sha256", "benchmarks/bench_metal.mojo"), ("driver_source_sha256", "benchmarks/tools/bench_metal.py")):
        expect(receipt.get(key) == PLAN["harness_file_sha256"][file], f"Wrong {role} binary {key}")
    expect(receipt.get("pixi_lock_sha256") == PLAN["pixi_lock_sha256"], f"Wrong {role} binary lock hash")


def audit(data, role, workloads, variants, rows, rounds, reps, runtimes, baseline_role=None):
    expected = set(itertools.product(range(rounds), runtimes, rows, workloads, variants))
    runs = data.get("runs", [])
    keys = [(r.get("round"), r.get("runtime"), r.get("rows"), r.get("workload"), r.get("variant")) for r in runs]
    expect(len(keys) == len(expected) and set(keys) == expected, "Missing or duplicate workers")
    for run in runs:
        expect(run.get("status") == "ok" and run.get("validated") is True and len(run.get("samples", [])) == reps, f"Failed worker: {run}")
        n, workload, runtime = run["rows"], run["workload"], run["runtime"]
        out = run.get("output_rows")
        expect(isinstance(out, (int,float)) and not isinstance(out,bool) and math.isfinite(out) and out == expected_rows(workload,n), f"Wrong output shape: {run}")
        profile = run.get("profile", {})
        peak = profile.get("peak_requested_bytes")
        expect(isinstance(peak,(int,float)) and not isinstance(peak,bool) and math.isfinite(peak) and 0 <= peak <= PLAN["reported_peak_ceiling_bytes"], f"Missing/invalid peak profile: {run}")
        if runtime == "mojo":
            expect(profile.get("region_0_algorithm") == PLAN["generic_profile_algorithm"], "Expected generic GPU ordering region missing")
            # Direct algorithm attribution is unavailable; this counter and the
            # pinned planner admission jointly provide only inference.
            if role == "candidate":
                launches = profile.get("query_kernel_launches", profile.get("kernel_launches"))
                expect(launches == expected_launches(workload,n), f"Unexpected candidate kernel launch count (route inference changed): {run}")
    metadata = data.get("metadata", {})
    receipts = metadata.get("binary_receipts", {})
    expect(isinstance(receipts, dict), "Missing binary receipts")
    expected_roles = {"mojo": role}
    if "baseline" in runtimes:
        expected_roles["baseline"] = baseline_role
    expect(set(receipts) == set(expected_roles), "Unexpected binary receipt roles")
    for key, source_role in expected_roles.items():
        check_binary_receipt(receipts[key], source_role)
    paired = {}
    for run in runs:
        paired.setdefault((run["round"],run["rows"],run["workload"],run["variant"]),{})[run["runtime"]] = run["output_rows"]
    for key, values in paired.items():
        if set(values) == {"mojo","baseline"}:
            expect(values["mojo"] == values["baseline"], f"Pair output shape differs for {key}")
    return len(runs)


def workload_set(name):
    if name == "frozen-pair": return PLAN["candidate_workloads"]
    if name == "radix-pair": return PLAN["intermediate_workloads"]
    if name == "smoke-frozen": return PLAN["candidate_workloads"]
    if name == "smoke-intermediate": return PLAN["intermediate_workloads"]
    if name == "smoke-candidate": return PLAN["candidate_workloads"]
    raise RuntimeError(f"Unknown run label {name}")


def runner(role, label, workloads, variants, rows, rounds, reps, warmups, baseline_role=None):
    t = tree(role)
    output = t / "build" / label / "results.json"
    expect(not output.parent.exists(), f"Refusing to overwrite output directory {output.parent}")
    args = [PLAN["pixi"], "run", "--manifest-path", PLAN["manifest"], "--locked", "-e", "oracle", "python3", str(t/"benchmarks/tools/bench_metal.py"), "--binary", str(t/"build/bench_metal"), "--rows", ",".join(map(str,rows)), "--workloads", ",".join(workloads), "--variants", ",".join(variants), "--rounds", str(rounds), "--reps", str(reps), "--warmups", str(warmups), "--threads", str(PLAN["threads"]), "--timeout-seconds", str(PLAN["timeout_seconds"]), "--output", str(output)]
    if baseline_role:
        base = tree(baseline_role)
        args += ["--baseline-binary", str(base/"build/bench_metal")]
    if (t / "build/bench_metal").is_file():
        args += ["--skip-build"]
    log = ROOT / f"{label}.log"
    expect(not log.exists(), f"Refusing to overwrite {log}")
    print("START", label, flush=True)
    with log.open("x") as f:
        result = subprocess.run(args, cwd=t, stdout=f, stderr=subprocess.STDOUT)
    print("STATUS", label, result.returncode, flush=True)
    expect(result.returncode == 0 and output.is_file(), f"Benchmark failed; retain {log}")
    data = json.loads(output.read_text())
    if label.startswith("smoke-"):
        audit(data, role, workloads, variants, rows, rounds, reps, ["mojo"])
    else:
        audit(data, role, workloads, variants, rows, rounds, reps, ["mojo","baseline"], baseline_role=baseline_role)
    return data


def outputs_check():
    labels = ["smoke-frozen","smoke-intermediate","smoke-candidate","frozen-pair","radix-pair"]
    paths = [ROOT / f"{x}.log" for x in labels]
    for role in PLAN["source_roles"]:
        t = Path(PLAN["worktree_root"])/PLAN["source_roles"][role]["remote"]
        paths += [t/"build/bench_metal", t/"build/bench_metal.receipt.json"]
        paths += [t/"build"/f"smoke-{role}"]
    candidate = Path(PLAN["worktree_root"])/PLAN["source_roles"]["candidate"]["remote"]
    paths += [candidate/"build/frozen-pair", candidate/"build/radix-pair"]
    exist = [str(p) for p in paths if p.exists()]
    expect(not exist, "Refusing to overwrite existing artifacts: " + repr(exist))


def preflight():
    expect(platform.system()=="Darwin" and platform.machine()=="arm64", "This plan is for Mac arm64")
    check_pids(); check_busy_mac(); check_bundles(); check_gates()
    trees = {role: tree(role) for role in PLAN["source_roles"]}
    for file in PLAN["harness_file_sha256"]:
        expect(len({sha(t/file) for t in trees.values()}) == 1, f"Worktree harness mismatch: {file}")
    outputs_check()


def self_test():
    expect(PLAN["workers"]["frozen_pair"] == PLAN["rounds"]*2*len(PLAN["candidate_workloads"])*len(PLAN["rows"])*len(PLAN["variants"]), "Frozen pair worker count mismatch")
    expect(PLAN["workers"]["radix_pair"] == PLAN["rounds"]*2*len(PLAN["intermediate_workloads"])*len(PLAN["rows"])*len(PLAN["variants"]), "Radix pair worker count mismatch")
    expect(expected_launches("sort_radix",4097)==39 and expected_launches("sort_topk31",1048576)==24, "Launch expectation drift")
    for n in PLAN["rows"]:
        for w in PLAN["candidate_workloads"]:
            expect(expected_rows(w,n) == (n if w=="sort_radix" else min(n,int(w.removeprefix("sort_topk")))), "Output expectation drift")
    print("PASS non-hardware ordering package self-test", flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--self-test",action="store_true"); args=parser.parse_args()
    if args.self_test: self_test(); return
    fd=None; lock_created=False
    try:
        preflight()
        fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); lock_created=True; os.write(fd,f"pid={os.getpid()}\n".encode()); os.close(fd); fd=None
        smoke=PLAN["smoke"]
        for role, label in (("frozen","smoke-frozen"),("intermediate","smoke-intermediate"),("candidate","smoke-candidate")):
            works=PLAN["intermediate_workloads"] if role=="intermediate" else PLAN["candidate_workloads"]
            runner(role,label,works,smoke["variants"],smoke["rows"],smoke["rounds"],smoke["reps"],smoke["warmups"])
        runner("candidate","frozen-pair",PLAN["candidate_workloads"],PLAN["variants"],PLAN["rows"],PLAN["rounds"],PLAN["reps"],PLAN["warmups"],"frozen")
        runner("candidate","radix-pair",PLAN["intermediate_workloads"],PLAN["variants"],PLAN["rows"],PLAN["rounds"],PLAN["reps"],PLAN["warmups"],"intermediate")
    finally:
        if fd is not None: os.close(fd)
        if lock_created and LOCK.exists(): LOCK.unlink()

if __name__ == "__main__": main()
