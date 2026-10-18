"""Collect CPU/forced-Apple complete-query samples; no HTML or coverage timings.

Preparation/building and measurement are separate invocations. Run only after
feature work on a quiet Mac. Strict CPU/GPU coverage and the external DuckDB
summary policy are separate evidence; neither changes the other's outcome.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
sys.path.insert(0, str(TOOLS))
import bench_host
import bench_metal_suites as metal
import bench_policy
import bench_suites as suites
import metal_query_report as coverage

ENGINES = ("cpu", "metal")
FORMAT = "dataframe-apple-query-performance-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, cwd=ROOT):
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def engine_identity(tree):
    paths = command(["git", "ls-files", "dataframe"], tree).splitlines()
    return {
        "revision": command(["git", "rev-parse", "HEAD"], tree),
        "status": command(["git", "status", "--porcelain", "--", "dataframe"], tree),
        "package_sha256": {path: sha(tree / path) for path in paths},
    }


def stage(destination):
    receipt = metal.stage_sources(destination)
    helper = ROOT / "benchmarks/metal_suite_timing.mojo"
    (destination / helper.name).write_bytes(helper.read_bytes())
    for name in ("h2o", "pdsh", "tpcds", "clickbench"):
        if name == "h2o":
            imports = """from cpu_h2o import groupby as cpu_groupby, join as cpu_join
from metal_h2o import groupby as metal_groupby, join as metal_join"""
            dispatch = """    if String(argv()[1]) == "h2o_groupby":
        if String(argv()[3]) == "cpu":
            run[cpu_groupby]()
        else:
            run[metal_groupby]()
    elif String(argv()[3]) == "cpu":
        run[cpu_join]()
    else:
        run[metal_join]()"""
        else:
            function = "query" if name == "clickbench" else "plan"
            imports = f"from cpu_{name} import {function} as cpu_query\nfrom metal_{name} import {function} as metal_query"
            dispatch = """    if String(argv()[3]) == "cpu":
        run[cpu_query]()
    else:
        run[metal_query]()"""
        (destination / f"performance_{name}.mojo").write_text(
            "from std.sys import argv\nfrom metal_suite_timing import run\n"
            + imports
            + "\ndef main() raises:\n"
            + dispatch
            + "\n"
        )
    receipt["original_source_sha256"][str(helper.relative_to(ROOT))] = sha(helper)
    receipt["derived_source_sha256"] = {
        path.name: sha(path) for path in sorted(destination.glob("*.mojo"))
    }
    return receipt


def schedule(cases, rounds):
    """Alternate first engine by round and case, retaining a reproducible order."""
    result = []
    for round_index in range(rounds):
        for index, case in enumerate(cases):
            order = ENGINES if (index + round_index) % 2 == 0 else ENGINES[::-1]
            for engine in order:
                result.append(dict(case, engine=engine, round=round_index))
    return result


def parse(output, repetitions, warmups, allow_incomplete=False):
    samples = {}
    outcomes = []
    for line in output.splitlines():
        fields = line.split("\t")
        if fields[0] == "outcome":
            if len(fields) != 3 or fields[1] not in {
                "ok",
                "unsupported",
                "execution_failed",
                "input_failed",
                "answer_failed",
            }:
                raise ValueError("Malformed performance outcome")
            outcomes.append(dict(status=fields[1], reason=fields[2]))
        elif fields[0] in {"sample", "answer"}:
            if len(fields) != (4 if fields[0] == "sample" else 6):
                raise ValueError("Malformed performance sample")
            phase, index = fields[1], int(fields[2])
            limits = {"cold": 1, "warmup": warmups, "warm": repetitions}
            if phase not in limits or not 0 <= index < limits[phase]:
                raise ValueError("Unexpected sample phase or index")
            sample = samples.setdefault((phase, index), dict(phase=phase, index=index))
            name = "ns" if fields[0] == "sample" else "summary"
            if name in sample:
                raise ValueError("Duplicate performance sample")
            if name == "ns":
                value = int(fields[3])
                if value < 0:
                    raise ValueError("Negative query duration")
                sample[name] = value
            else:
                values = [float(v) for v in fields[4].split(",")] if fields[4] else []
                names = fields[5].split(",") if fields[5] else []
                height = int(fields[3])
                if height < 0 or len(values) != len(names):
                    raise ValueError("Malformed answer summary")
                # Numeric strings preserve NaN/Inf without nonstandard JSON numbers.
                sample[name] = dict(
                    height=height,
                    values=fields[4].split(",") if fields[4] else [],
                    names=names,
                )
    if not outcomes and allow_incomplete:
        return dict(status="uncompleted_protocol", samples=list(samples.values()))
    if len(outcomes) != 1:
        raise ValueError("Missing or ambiguous performance outcome")
    outcome = outcomes[0]
    if outcome["status"] == "ok" and (
        len(samples) != 1 + warmups + repetitions
        or any("ns" not in s or "summary" not in s for s in samples.values())
    ):
        raise ValueError("Successful worker has incomplete samples")
    return dict(outcome, samples=list(samples.values()))


def strict_basis(case, receipts, engine, source_hashes, mojo, threads, metadata):
    """Require a matching exact receipt; never derive correctness from timings."""
    key = coverage.key(case)
    for receipt in receipts:
        provenance = receipt.get("provenance", {})
        if (
            provenance.get("engine_revision") != engine["revision"]
            or provenance.get("engine_status")
            or engine["status"]
            or provenance.get("mojo") != mojo
            or provenance.get("threads") != threads
        ):
            continue
        originals = provenance.get("original_source_sha256", {})
        workload = {
            p: h for p, h in source_hashes.items() if p.startswith("benchmarks/suites/")
        }
        if any(originals.get(p) != h for p, h in workload.items()):
            continue
        matches = [c for c in receipt.get("cases", []) if coverage.key(c) == key]
        if len(matches) != 1:
            continue
        measured = matches[0]
        inputs = lambda c: {n: v["sha256"] for n, v in c.get("inputs", {}).items()}
        if inputs(measured) != inputs(case):
            continue
        audit = coverage.execution_audit(
            measured, originals, metadata, provenance.get("derived_source_sha256", {})
        )
        return dict(
            receipt_sha256=receipt.get("receipt_sha256"),
            status=measured["status"],
            execution_boundary=audit["execution_boundary"],
            execution_audit=audit,
            whole_query_gpu_certificate=audit.get("whole_query_gpu_certificate", False),
            verified=measured["status"] == "ok"
            and measured.get("collections", 0) > 0
            and isinstance(measured.get("ordered_equal"), bool),
            note="Separate exact full-schema/duplicate-row receipt; not per-timed-sample exact comparison.",
        )
    return dict(
        status="unmeasured",
        execution_boundary="unknown",
        verified=False,
        note="No matching engine/SDK/thread/workload/input exact receipt.",
    )


def oracle_checks(run, reference):
    if reference.get("status") != "ok":
        return dict(status=reference.get("status", "unavailable"), samples=[])
    checks = []
    for sample in run.get("samples", []):
        if "summary" not in sample:
            continue
        got = dict(
            sample["summary"], values=[float(v) for v in sample["summary"]["values"]]
        )
        checks.append(
            dict(
                phase=sample["phase"],
                index=sample["index"],
                equal=suites.same_answer(
                    run["suite"], run["query"], got, reference["summary"]
                ),
            )
        )
    return dict(
        status="matches" if checks and all(c["equal"] for c in checks) else "differs",
        samples=checks,
    )


def ratio_basis(runs, strict, rounds, repetitions):
    """Conservative admission only; HTML and aggregate ratios are a later step."""
    if not strict.get("verified") or strict.get("execution_boundary") != "gpu_collects":
        return "unverified_or_hybrid_exact_basis"
    for engine in ENGINES:
        values = [r for r in runs if r["engine"] == engine]
        if len(values) != rounds or {r["round"] for r in values} != set(range(rounds)):
            return "incomplete_rounds"
        for run in values:
            if (
                run["status"] != "ok"
                or run.get("compiler_overlap")
                or run.get("busy_host")
            ):
                return "failed_or_contaminated_run"
            if run.get("oracle", {}).get("status") != "matches":
                return "external_answer_policy_not_verified"
            if (
                len(
                    [
                        s
                        for s in run["samples"]
                        if s["phase"] == "warm" and s.get("ns", 0) > 0
                    ]
                )
                != repetitions
            ):
                return "incomplete_warm_samples"
    return "eligible_separate_exact_and_external_summary_evidence"


def input_plan(selected, wanted, scale):
    cases = []
    for suite in selected:
        queries = [
            q for q in suites.SUITES[suite]["queries"] if wanted is None or q in wanted
        ]
        if not queries:
            continue
        for variant in suites.VARIANTS[suite]:
            paths = suites.tables(suite, variant, scale)
            inputs = {
                n: dict(
                    path=str(p.resolve()), sha256=sha(p), file_bytes=p.stat().st_size
                )
                for n, p in paths.items()
            }
            for query in queries:
                cases.append(
                    dict(suite=suite, variant=variant, query=query, inputs=inputs)
                )
    return dict(
        scale=scale,
        suites=selected,
        queries=sorted(wanted) if wanted else None,
        cases=cases,
    )


def validate_input_plan(plan, selected, wanted, scale):
    expected = {
        (s, v, q)
        for s in selected
        for v in suites.VARIANTS[s]
        for q in suites.SUITES[s]["queries"]
        if wanted is None or q in wanted
    }
    observed = [coverage.key(c) for c in plan["cases"]]
    if (
        plan["scale"] != scale
        or plan["suites"] != selected
        or plan["queries"] != (sorted(wanted) if wanted else None)
        or len(observed) != len(set(observed))
        or set(observed) != expected
        or any(not c.get("inputs") for c in plan["cases"])
    ):
        raise ValueError(
            "Input manifest does not cover the selected catalog/configuration"
        )


def inputs_unchanged(cases):
    hashes = {v["path"]: v["sha256"] for c in cases for v in c["inputs"].values()}
    return all(
        Path(path).is_file() and sha(path) == value for path, value in hashes.items()
    )


def host():
    return dict(
        load_average=list(os.getloadavg()), compilers=bench_host.compiler_processes()
    )


def worker_environment(threads, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    env.update(
        DATAFRAME_THREADS=str(threads),
        BENCH_THREADS=str(threads),
        POLARS_MAX_THREADS=str(threads),
        OMP_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
    )
    for name in (
        "DATAFRAME_TRACE_PATHS",
        "DATAFRAME_EXECUTION_REPORT",
        "DATAFRAME_BENCH_ANSWER_EXPORT_PREFIX",
    ):
        env.pop(name, None)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suites", default=",".join(suites.SUITES))
    parser.add_argument("--scale", choices=suites.SCALES, default="default")
    parser.add_argument("--queries", default="")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--prepare-inputs", action="store_true")
    parser.add_argument("--strict-coverage", type=Path, action="append", default=[])
    parser.add_argument(
        "--case-metadata", type=Path, default=TOOLS / "metal_case_metadata.json"
    )
    args = parser.parse_args()
    selected = args.suites.split(",")
    if not set(selected) <= set(suites.SUITES) or len(set(selected)) != len(selected):
        parser.error("Unknown or duplicate suite")
    if min(args.rounds, args.reps, args.warmups, args.threads, args.timeout) < 1:
        parser.error("Execution settings must be positive")
    if sum((args.prepare_only, args.build_only, args.prepare_inputs)) > 1:
        parser.error(
            "Source preparation, input preparation and building are separate modes"
        )
    wanted = set(args.queries.split(",")) if args.queries else None
    available = {q for name in selected for q in suites.SUITES[name]["queries"]}
    if wanted is not None and not wanted <= available:
        parser.error("Query selection is outside the selected catalog")
    output = args.output.resolve()
    if output.exists():
        parser.error("Refusing to overwrite a performance receipt")
    tree = args.engine_tree.resolve()
    directory = output.parent / (output.stem + ".work")
    directory.mkdir(parents=True, exist_ok=True)
    sources = directory / "performance-sources"
    staged = stage(sources)
    engine = engine_identity(tree)
    if engine["status"]:
        parser.error("Engine dataframe working tree must be clean")
    runners = sorted({suites.SUITES[s]["runner"] for s in selected})
    binaries = {name: directory / ("performance_" + name) for name in runners}
    if args.prepare_inputs:
        (directory / "performance-inputs.json").write_text(
            json.dumps(input_plan(selected, wanted, args.scale), indent=2) + "\n"
        )
        return
    if args.prepare_only:
        print(json.dumps(dict(engine=engine, staged=staged), indent=2))
        return
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Execution/building requires Apple Silicon")
    if args.build_only:
        for name, binary in binaries.items():
            subprocess.run(
                [
                    "mojo",
                    "build",
                    "-O3",
                    "-I",
                    str(tree),
                    "-I",
                    str(sources),
                    str(sources / ("performance_" + name + ".mojo")),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
        (directory / "performance-build.json").write_text(
            json.dumps(
                dict(
                    engine=engine,
                    staged=staged,
                    mojo=command(["mojo", "--version"]),
                    binaries={n: sha(p) for n, p in binaries.items()},
                ),
                indent=2,
            )
            + "\n"
        )
        return
    build = json.loads((directory / "performance-build.json").read_text())
    mojo = command(["mojo", "--version"])
    if (
        build["engine"] != engine
        or build["staged"] != staged
        or build["mojo"] != mojo
        or build["binaries"] != {n: sha(p) for n, p in binaries.items()}
    ):
        parser.error(
            "Build receipt does not match engine, SDK, harness sources or binaries"
        )
    metadata = json.loads(args.case_metadata.read_text())
    coverage.validate_metadata(metadata)
    if args.strict_coverage:
        coverage.load_receipts(args.strict_coverage)
    strict_receipts = [
        dict(json.loads(p.read_text()), receipt_sha256=sha(p))
        for p in args.strict_coverage
    ]
    plan = json.loads((directory / "performance-inputs.json").read_text())
    validate_input_plan(plan, selected, wanted, args.scale)
    if not inputs_unchanged(plan["cases"]):
        parser.error("Prepared input hashes no longer match")
    cases = plan["cases"]
    is_headline = (
        args.scale == "default"
        and set(selected) == set(suites.SUITES)
        and wanted is None
        and args.rounds == args.reps == 3
        and args.threads == 8
    )
    result = dict(
        format=FORMAT,
        provenance=dict(
            recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            engine=engine,
            harness_revision=command(["git", "rev-parse", "HEAD"]),
            harness_status=command(
                ["git", "status", "--porcelain", "--", "benchmarks"]
            ),
            harness_sha256={
                str(p.relative_to(ROOT)): sha(p)
                for p in [
                    Path(__file__),
                    ROOT / "benchmarks/metal_suite_timing.mojo",
                    TOOLS / "bench_metal_suites.py",
                    TOOLS / "bench_suites.py",
                    TOOLS / "bench_host.py",
                    TOOLS / "bench_policy.py",
                    TOOLS / "metal_query_report.py",
                ]
            },
            staged=staged,
            mojo=mojo,
            platform=platform.platform(),
            physical_memory_bytes=int(command(["sysctl", "-n", "hw.memsize"])),
            binaries_sha256={n: sha(p) for n, p in binaries.items()},
            threads=args.threads,
            scale=args.scale,
            scale_parameters=suites.SCALES[args.scale],
            rounds=args.rounds,
            repetitions=args.reps,
            warmups=args.warmups,
            headline_full_tier_attempt=is_headline,
            quiet_host_policy="Observed compiler processes and load before/after each worker; no operator attestation required. Load above the configured thread budget flags busy-host ratios; process snapshots do not prove absence of all competing work.",
            timing_scope="Full query call returning materialized host DataFrame; table loading, answer summary and destruction excluded. Cold is first invocation in fresh worker after loading, not cold filesystem/cache/compilation.",
            metal_execution_policy="End-to-end requested Metal backend. Engines with CPU data-fault adjudication may rerun a query on CPU; any rerun is included in elapsed time. Collect-boundary audits do not certify GPU-only execution.",
            strict_coverage_receipts=[
                dict(path=str(p), sha256=sha(p)) for p in args.strict_coverage
            ],
            case_metadata_sha256=sha(args.case_metadata),
            input_manifest_sha256=sha(directory / "performance-inputs.json"),
            evaluation=bench_policy.record(ROOT, selected, suites.SUITES),
            oracle_policy="Independent DuckDB upstream SQL summary via existing bench_suites.same_answer/CHECKS. Separate from exact CPU/GPU multiset comparison; cannot excuse strict differences.",
            oracle_worker_sha256=sha(ROOT / "benchmarks/suites/engines.py"),
            oracle_runtime=oracle_runtime_identity(),
            exposure_note="Existing development/heldout policy retained. These runs measure correctness and complete collection; heldout timing results are not a tuning license.",
        ),
        cases=cases,
        runs=[],
        oracle={},
    )
    logs = directory / "performance-logs"
    logs.mkdir(exist_ok=True)
    env = worker_environment(args.threads)

    def save():
        pending = output.with_name(output.name + ".tmp")
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        pending.replace(output)

    result["provenance"]["query_environment"] = {
        k: v
        for k, v in env.items()
        if k.startswith("DATAFRAME_")
        or k
        in {
            "BENCH_THREADS",
            "POLARS_MAX_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        }
    }
    for case in cases:
        case["strict_basis"] = strict_basis(
            case,
            strict_receipts,
            engine,
            staged["original_source_sha256"],
            mojo,
            args.threads,
            metadata,
        )
    save()
    for planned in schedule(cases, args.rounds):
        before = host()
        if before["compilers"]:
            result["aborted"] = "Compiler active; remaining workers unmeasured"
            save()
            break
        key = "/".join(coverage.key(planned))
        if key not in result["oracle"]:
            argv = [
                sys.executable,
                str(ROOT / "benchmarks/suites/engines.py"),
                "duckdb",
                planned["suite"],
                planned["query"],
                "0",
            ] + [n + "=" + v["path"] for n, v in planned["inputs"].items()]
            reference = execute(argv, env, args.timeout)
            if reference["status"] == "completed":
                parsed = suites.parse(reference["stdout"], [planned["query"]])[
                    planned["query"]
                ]
                # Store finite/nonfinite summary values as strings in strict JSON.
                if "summary" in parsed:
                    parsed["summary"]["values"] = [
                        str(v) for v in parsed["summary"]["values"]
                    ]
                reference["answer"] = parsed
            result["oracle"][key] = reference
        before = host()
        if before["compilers"]:
            result[
                "aborted"
            ] = "Compiler active after oracle; remaining workers unmeasured"
            save()
            break
        argv = [
            str(binaries[suites.SUITES[planned["suite"]]["runner"]]),
            planned["suite"],
            planned["query"],
            planned["engine"],
            str(args.reps),
            str(args.warmups),
        ] + [n + "=" + v["path"] for n, v in planned["inputs"].items()]
        raw = execute(argv, env, args.timeout)
        label = f"{planned['suite']}-{planned['variant']}-{planned['query']}-{planned['engine']}-{planned['round']}"
        log = logs / (label + ".txt")
        log.write_text(raw["stdout"] + "\nSTDERR\n" + raw["stderr"])
        run = dict(
            planned,
            status=raw["status"],
            reason=raw.get("reason", ""),
            returncode=raw.get("returncode"),
            peak_rss_bytes=raw.get("peak_rss_bytes"),
            log=str(log.relative_to(output.parent)),
            log_sha256=sha(log),
            host_before=before,
            host_after=host(),
        )
        run["compiler_overlap"] = bool(run["host_after"]["compilers"])
        run["busy_host"] = before["load_average"][0] > args.threads
        try:
            parsed = parse(
                raw["stdout"],
                args.reps,
                args.warmups,
                allow_incomplete=raw["status"] != "completed",
            )
            if raw["status"] == "completed":
                run.update(parsed)
            else:
                run["samples"] = parsed["samples"]
        except ValueError as error:
            run["protocol_error"] = str(error)
            if raw["status"] == "completed":
                run.update(status="protocol_failed", reason=str(error))
        reference = result["oracle"][key].get("answer", dict(status="unavailable"))
        if "summary" in reference:
            reference = dict(
                reference,
                summary=dict(
                    reference["summary"],
                    values=[float(v) for v in reference["summary"]["values"]],
                ),
            )
        run["oracle"] = oracle_checks(run, reference)
        result["runs"].append(run)
        save()
        print(label, run["status"], flush=True)
    for case in cases:
        runs = [r for r in result["runs"] if coverage.key(r) == coverage.key(case)]
        case["ratio_basis"] = ratio_basis(
            runs, case["strict_basis"], args.rounds, args.reps
        )
    result["all_scheduled_workers_attempted"] = (
        len(result["runs"]) == len(cases) * args.rounds * 2
    )
    result["engine_unchanged_after"] = engine_identity(tree) == engine
    result["inputs_unchanged_after"] = inputs_unchanged(cases)
    if not result["engine_unchanged_after"] or not result["inputs_unchanged_after"]:
        for case in cases:
            case["ratio_basis"] = "engine_or_inputs_changed_during_measurement"
    save()


def oracle_runtime_identity():
    code = "import duckdb,json,platform; print(json.dumps({'duckdb_version':duckdb.__version__,'python_version':platform.python_version()}))"
    try:
        return dict(available=True, **json.loads(command([sys.executable, "-c", code])))
    except (subprocess.CalledProcessError, ValueError) as error:
        return dict(available=False, reason=str(error))


def peak_rss(stderr):
    matches = re.findall(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", stderr)
    return int(matches[0]) if len(matches) == 1 else None


def execute(argv, env, timeout):
    # A private process group ensures a timed-out time(1) wrapper cannot leave
    # its benchmark child running and contaminate the following measurements.
    measured_argv = [
        "/usr/bin/time",
        "-l",
    ] + argv if platform.system() == "Darwin" else argv
    try:
        proc = subprocess.Popen(
            measured_argv,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as error:
        return dict(status="process_failed", reason=str(error), stdout="", stderr="")
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        return dict(
            status="timeout",
            reason=f"Exceeded {timeout}s; owned worker process group terminated",
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    return dict(
        status="completed" if proc.returncode == 0 else "process_failed",
        returncode=proc.returncode,
        peak_rss_bytes=peak_rss(stderr) if platform.system() == "Darwin" else None,
        stdout=stdout,
        stderr=stderr,
    )


if __name__ == "__main__":
    main()
