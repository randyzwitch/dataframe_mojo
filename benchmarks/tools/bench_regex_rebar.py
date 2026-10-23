#!/usr/bin/env python3
"""Compare frozen engine commits on complete, pinned upstream rebar groups.

Run with the oracle Pixi environment. Only a local clone of rebar is required;
this driver never downloads or modifies upstream data. Results stay in build/.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import tomllib
from pathlib import Path

import polars as pl

from bench_host import compiler_processes

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = "463d00f31887e84c38467805b9e3122c314b9521"
GROUPS = (
    "curated/01-literal",
    "curated/02-literal-alternate",
    "imported/sherlock",
    "opt/onepass",
    "captures",
)
VARIANTS = ("base", "reverse-lines", "double")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True
    ).strip()


def compilers():
    # macOS truncates `comm`; inspect the full executable/argument column.
    found = []
    for line in subprocess.check_output(["ps", "-axo", "pid=,args="], text=True).splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 2:
            continue
        name = Path(fields[1]).name
        args = " " + (fields[2] if len(fields) > 2 else "") + " "
        if name in {"clang", "clang++", "cc1", "cc1plus", "rustc"} or (
            name == "mojo" and any(" " + verb + " " in args for verb in
                                   ("build", "run", "doc", "test", "package", "precompile"))):
            found.append((fields[0], name))
    return found


def prepare(upstream, output):
    if git(upstream, "rev-parse", "HEAD") != UPSTREAM:
        raise ValueError(f"rebar must be checked out at {UPSTREAM}")
    if git(upstream, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("rebar has modified tracked files")
    output.mkdir(parents=True, exist_ok=True)
    cases, rows = [], []
    for group in GROUPS:
        definition = upstream / "benchmarks/definitions" / f"{group}.toml"
        for bench in tomllib.loads(definition.read_text())["bench"]:
            name = group + "/" + bench["name"]
            original = bench["regex"]
            assert isinstance(original, str)
            pattern = (
                ("(?i)" if bench.get("case-insensitive", False) else "")
                + ("(?-u)" if not bench.get("unicode", False) else "")
                + original
            )
            index = len(cases)
            pattern_path = output / f"pattern-{index}.txt"
            pattern_path.write_text(pattern)
            data_path = upstream / "benchmarks/haystacks" / bench["haystack"]["path"]
            data = data_path.read_bytes().decode("utf-8")
            expected = bench["count"]
            if isinstance(expected, list):
                expected = next(
                    item["count"] for item in expected
                    if re.fullmatch(item["engine"], "rust/regex")
                )
            for variant in VARIANTS:
                text = data
                if variant == "reverse-lines":
                    text = "".join(data.splitlines(keepends=True)[::-1])
                elif variant == "double":
                    text = data + "\n" + data
                haystack_path = output / f"hay-{index}-{variant}.txt"
                haystack_path.write_bytes(text.encode("utf-8"))
                oracle_error = None
                try:
                    if bench["model"] == "grep-captures":
                        inputs = pl.Series(text.split("\n")).str.strip_suffix("\r")
                        groups = inputs.str.extract_groups(pattern).struct.unnest()
                        oracle = inputs.str.contains(pattern).sum() + sum(
                            groups.height - value for value in groups.null_count().row(0)
                        )
                    elif bench["model"] == "count-captures":
                        matches = pl.Series([text]).str.extract_all(pattern).explode().drop_nulls()
                        groups = matches.str.extract_groups(pattern).struct.unnest()
                        oracle = len(matches) + sum(
                            groups.height - value for value in groups.null_count().row(0)
                        )
                    else:
                        found = pl.Series([text]).str.extract_all(pattern)
                        oracle = (
                            found.list.eval(pl.element().str.len_bytes()).list.sum().item()
                            if bench["model"] == "count-spans"
                            else found.list.len().item()
                        )
                    if variant == "base" and oracle != expected:
                        raise ValueError(f"upstream/Polars mismatch: {name}: {oracle} != {expected}")
                except pl.exceptions.ComputeError as error:
                    oracle, oracle_error = -1, str(error)
                key = name + "/" + variant
                rows.append("\t".join(map(str, (
                    key, pattern_path, haystack_path, bench["model"], oracle,
                ))))
                cases.append(dict(
                    key=key, name=name, variant=variant, pattern=pattern,
                    model=bench["model"], expected=oracle,
                    upstream_expected=expected if variant == "base" else None,
                    oracle_error=oracle_error,
                    definition_sha256=sha256(definition),
                    haystack_sha256=sha256(haystack_path), bytes=len(text.encode()),
                ))
    manifest = output / "manifest.tsv"
    manifest.write_text("\n".join(rows) + "\n")
    return cases, manifest


def build(ref, directory):
    commit = git(ROOT, "rev-parse", f"{ref}^{{commit}}")
    tree = directory / commit
    tree.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "-C", str(ROOT), "archive", commit, "dataframe"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(tree, filter="data")
    runner = ROOT / "benchmarks/regex_rebar_runner.mojo"
    binary = tree / "runner"
    subprocess.run([
        "mojo", "build", "-O3", "-I", str(tree), str(runner), "-o", str(binary),
    ], check=True)
    return dict(commit=commit, binary=str(binary), binary_sha256=sha256(binary))


def save(path, result):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def report(result):
    lines = [
        "# Frozen upstream regex comparison", "",
        f"rebar `{UPSTREAM}`; {len(GROUPS)} complete groups, "
        f"{len(result['cases']) // len(VARIANTS)} workloads, {len(VARIANTS)} variants; "
        "three rounds of three samples per build (20 full scans per sample).", "",
        "Speedup = parent / candidate; below 1 is a slowdown. "
        "Worst variant includes base, reversed lines, and doubled input. "
        "Counts/spans are checked against upstream expectations and Polars. "
        "Compile time and input loading are untimed.", "",
        "| Workload | Parent base ms | Candidate base ms | Base speedup | Worst variant |",
        "|---|---:|---:|---:|---:|",
    ]
    round_samples = {}
    failures = []
    for record in result["records"]:
        if record.get("rejected"):
            continue
        for line in record["stdout"].splitlines():
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            key = (record["build"], fields[0])
            if fields[1] == "OK":
                round_samples.setdefault((*key, record["round"]), []).append(int(fields[3]))
            elif fields[1] in ("WRONG", "ERROR"):
                failures.append((record["build"], fields[0], fields[1], "\t".join(fields[2:])))
    samples = {}
    for (build_name, key, round_number), values in round_samples.items():
        if len(values) == 3:
            samples.setdefault((build_name, key), []).append(statistics.median(values))
    for case in result["cases"]:
        if case["expected"] >= 0:
            for build_name in ("parent", "candidate"):
                if len(samples.get((build_name, case["key"]), [])) != 3:
                    failures.append((build_name, case["key"], "MISSING", "incomplete timing samples"))
    for name in dict.fromkeys(case["name"] for case in result["cases"]):
        ratios = []
        base = None
        for variant in VARIANTS:
            key = name + "/" + variant
            parent = samples.get(("parent", key), [])
            candidate = samples.get(("candidate", key), [])
            if len(parent) == len(candidate) == 3:
                p, c = statistics.median(parent), statistics.median(candidate)
                ratios.append(p / c)
                if variant == "base":
                    base = (p / 1e6, c / 1e6, p / c)
        if base is None:
            lines.append(f"| {name} | unsupported/failing; see raw records | | | |")
        else:
            p, c, ratio = base
            lines.append(f"| {name} | {p:.4f} | {c:.4f} | {ratio:.3f}× | {min(ratios):.3f}× |")
    unique = set((build, key, status) for build, key, status, _ in failures)
    lines += ["", f"Unsupported/failing build/cases retained: {len(unique)}.", ""]
    return "\n".join(lines), failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebar-dir", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", default="HEAD")
    parser.add_argument("--output", type=Path, default=ROOT / "build/regex-rebar/results.json")
    parser.add_argument("--wait-seconds", type=int, default=600)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cases, manifest = prepare(args.rebar_dir.resolve(), output.parent / "inputs")
    builds = {
        name: build(ref, output.parent / "engines")
        for name, ref in (("parent", args.baseline), ("candidate", args.candidate))
    }
    cpu = max(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    result = dict(
        upstream=UPSTREAM, groups=GROUPS, variants=VARIANTS, cases=cases,
        builds=builds, records=[], cpu=cpu, platform=platform.platform(),
        compiler=subprocess.check_output(["mojo", "--version"], text=True).strip(),
        polars=pl.__version__, iterations_per_sample=20, driver_sha256=sha256(Path(__file__)),
        runner_sha256=sha256(ROOT / "benchmarks/regex_rebar_runner.mojo"),
        started=time.time(),
    )
    save(output, result)
    for round_number in range(3):
        order = ("parent", "candidate") if round_number % 2 == 0 else ("candidate", "parent")
        for name in order:
            for attempt in range(3):
                deadline = time.monotonic() + args.wait_seconds
                if compilers():
                    print("Waiting for compiler activity to stop", flush=True)
                while compilers():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("compiler activity persisted; partial results retained")
                    time.sleep(5)
                command = [builds[name]["binary"], str(manifest)]
                if cpu is not None and shutil.which("taskset"):
                    command = ["taskset", "-c", str(cpu), *command]
                started = time.time()
                activity = []
                worker = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                while True:
                    try:
                        stdout, stderr = worker.communicate(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        activity.extend(compilers())
                        if time.time() - started >= 300:
                            worker.kill()
                            stdout, stderr = worker.communicate()
                            break
                activity.extend(compilers())
                record = dict(build=name, round=round_number, attempt=attempt,
                              started=started, stdout=stdout, stderr=stderr,
                              returncode=worker.returncode, load=os.getloadavg(),
                              rejected=bool(activity), compiler_activity=activity)
                result["records"].append(record)
                save(output, result)
                if worker.returncode:
                    raise RuntimeError(f"{name} worker failed; raw logs retained")
                if not activity:
                    print(f"Completed {name} round {round_number + 1}", flush=True)
                    break
                print("Discarding compiler-overlapped worker", flush=True)
            else:
                raise RuntimeError("could not obtain a compiler-free sample")
    text, failures = report(result)
    output.with_suffix(".md").write_text(text)
    print(text)
    supported_keys = {case["key"] for case in cases if case["expected"] >= 0}
    if any(status in ("WRONG", "MISSING") or key in supported_keys for _, key, status, _ in failures):
        raise SystemExit("wrong results retained in report")


if __name__ == "__main__":
    main()
