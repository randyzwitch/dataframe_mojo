#!/usr/bin/env python3
"""Run focused Metal development cases on one host; retain every outcome.

No external suite is represented here. Each case/build/round has a fresh process,
shares the same generated input with its CPU comparison, and materializes results.
"""
import argparse
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = [
    "projection",
    "expression16",
    "chain32",
    "filter_half",
    "filter_sparse",
    "count",
    "int_sum",
]
VARIANTS = ["base", "sorted", "nulls"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args):
    return subprocess.check_output(
        args, cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def metadata(binary, libraries, threads):
    return {
        "recorded_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "physical_cores": command(["sysctl", "-n", "hw.physicalcpu"]),
        "memory_bytes": command(["sysctl", "-n", "hw.memsize"]),
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "engine_status": command(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "dataframe",
                "native/dfmetal",
            ]
        ),
        "benchmark_status": command(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "benchmarks/bench_metal.mojo",
                "scripts/bench_metal.py",
            ]
        ),
        "mojo_version": command(["mojo", "--version"]),
        "xcode": command(["xcodebuild", "-version"]),
        "sdk": command(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
        "compile_flags": ["-O3", "-I", "."],
        "dataframe_threads": threads,
        "binary_sha256": sha(binary),
        "source_sha256": sha(ROOT / "benchmarks/bench_metal.mojo"),
        "driver_sha256": sha(__file__),
        "library_sha256": {k: sha(v) for k, v in libraries.items()},
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("DATAFRAME_", "PLANNER_"))
        },
        "method": "Fresh process per case/build/round; first-use collections retained separately; two extra warmups per engine; alternating CPU/Metal timed order; process/build order rotated per round; inputs and exact CPU-reference validation untimed; complete result materialization timed; result release untimed.",
        "scope": "Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.",
        "data": "Deterministic 1024-value numeric distribution. Base cycles; sorted uses monotone Float32 bins (Int32 sum uses mapped bins); nulls permutes bins with fixed LCG constants and marks exactly every twentieth row invalid. Float32 values are exact multiples of 1/1024; Int32 values are bin modulo 17 minus 8.",
        "limitations": [
            "No external-suite coverage claimed; joins, group-by, Float64 and floating reductions are outside these cases.",
            "First use includes context/shader initialization but is not a cold machine or cold filesystem measurement.",
            "CPU and GPU share host memory and power limits; short timings can vary with scheduler and thermal state.",
            "Native profile is a separate warm collection after timed samples; its GPU interval is not a caller-wall substitute.",
            "Workload exposure: chain32 was used during fusion development; other cases broaden mechanism coverage without a holdout claim.",
        ],
    }


def parse_run(output, reps):
    result = {"samples": [], "profile": {}}
    for line in output.splitlines():
        parts = line.split()
        if not parts:
            continue
        kind = parts[0]
        if kind == "FIRST":
            result["first_use_ns"] = {
                "cpu": int(parts[1]),
                "metal": int(parts[2]),
            }
        elif kind == "SAMPLE":
            result["samples"].append(
                {
                    "index": int(parts[1]),
                    "cpu_ns": int(parts[2]),
                    "metal_ns": int(parts[3]),
                }
            )
        elif kind in ("WORKERS", "OUTPUT_ROWS"):
            result[kind.lower()] = int(parts[1])
        elif kind in ("PROFILE_I", "PROFILE_F"):
            result["profile"][parts[1]] = (
                int if kind == "PROFILE_I" else float
            )(parts[2])
        elif kind == "BUILD_ID":
            result["runtime_build_id"] = parts[1]
        elif kind == "VALIDATED":
            result["validated"] = parts[1] == "True"
    if (
        not result.get("validated")
        or len(result["samples"]) != reps
        or "first_use_ns" not in result
    ):
        raise ValueError("Missing correctness marker or timed samples")
    if result["profile"].get("synchronizations") != 1:
        raise ValueError("Expected one native synchronization")
    return result


def render(data, target):
    lines = [
        "# Apple Metal development measurements",
        "",
        data["metadata"]["scope"],
        "",
        "Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.",
        "",
        "| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Shared MiB |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    groups = {}
    for run in data["runs"]:
        key = (run["runtime"], run["rows"], run["workload"], run["variant"])
        groups.setdefault(key, []).append(run)
    ratios = {}
    for (runtime, rows, workload, variant), runs in sorted(groups.items()):
        successful = [r for r in runs if r["status"] == "ok"]
        if len(successful) != len(runs):
            states = ", ".join(r["status"] for r in runs)
            lines.append(
                f"| {runtime} | {rows} | {workload} | {variant} | {states} | — | — | — | — |"
            )
            continue
        cpu = (
            statistics.median(s["cpu_ns"] for r in runs for s in r["samples"])
            / 1e6
        )
        metal = (
            statistics.median(s["metal_ns"] for r in runs for s in r["samples"])
            / 1e6
        )
        p = runs[-1]["profile"]
        ratio = cpu / metal
        ratios.setdefault((runtime, rows, workload), []).append(
            (ratio, variant)
        )
        lines.append(
            f"| {runtime} | {rows} | {workload} | {variant} | {cpu:.3f} | {metal:.3f} | {ratio:.2f} | {p['kernel_launches']} | {p['shared_buffer_bytes'] / 1048576:.2f} |"
        )
    lines += [
        "",
        "## Worst measured variant per case",
        "",
        "Lowest CPU/Metal ratio among the selected variants; ratios below one favor CPU. These are per-mechanism results, not an aggregate suite score.",
        "",
        "| Runtime | Rows | Workload | Worst variant | CPU/Metal |",
        "|---|---:|---|---|---:|",
    ]
    for (runtime, rows, workload), variants in sorted(ratios.items()):
        ratio, variant = min(variants)
        lines.append(
            f"| {runtime} | {rows} | {workload} | {variant} | {ratio:.2f} |"
        )
    counts = {
        status: sum(r["status"] == status for r in data["runs"])
        for status in ("ok", "unsupported", "failed")
    }
    lines += [
        "",
        f"Process outcomes: {counts}.",
        "",
        "## Provenance and limits",
        "",
        "```json",
        json.dumps(data["metadata"], indent=2),
        "```",
        "",
    ]
    Path(target).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build/bench_metal"
    )
    parser.add_argument(
        "--library", type=Path, default=ROOT / "build/dfmetal/libdfmetal.dylib"
    )
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument("--rows", default="65536,1048576,8388608")
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/metal/results.json"
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--report-from", type=Path)
    args = parser.parse_args()
    if args.report_from:
        data = json.loads(args.report_from.read_text())
        render(data, args.report_from.with_suffix(".md"))
        return
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("This benchmark requires an Apple Silicon Mac")
    if min(args.reps, args.rounds, args.threads) < 1:
        parser.error("Repetitions, rounds and threads must be positive")
    rows = [int(x) for x in args.rows.split(",")]
    workloads, variants = args.workloads.split(","), args.variants.split(",")
    if (
        min(rows) < 1
        or set(workloads) - set(WORKLOADS)
        or set(variants) - set(VARIANTS)
    ):
        parser.error("Invalid rows, workload or variant")
    binary = args.binary.resolve()
    if not args.skip_build:
        binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "mojo",
                "build",
                "-O3",
                "-I",
                ".",
                "benchmarks/bench_metal.mojo",
                "-o",
                str(binary),
            ],
            cwd=ROOT,
            check=True,
        )
    libraries = {"metal": args.library.resolve()}
    if args.baseline_library:
        libraries["baseline"] = args.baseline_library.resolve()
    data = {
        "schema_version": 1,
        "metadata": metadata(binary, libraries, args.threads),
        "selection": {
            "rows": rows,
            "workloads": workloads,
            "variants": variants,
            "rounds": args.rounds,
            "reps": args.reps,
        },
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.output.with_suffix("")
    log_dir.mkdir(exist_ok=True)
    cases = list(itertools.product(rows, workloads, variants))
    for round_id in range(args.rounds):
        ordered = cases if round_id % 2 == 0 else list(reversed(cases))
        builds = list(libraries.items())
        if round_id % 2:
            builds.reverse()
        for case_id, (n, workload, variant) in enumerate(ordered):
            for runtime, library in builds:
                label = f"{round_id}-{runtime}-{n}-{workload}-{variant}"
                env = os.environ.copy()
                env.update(
                    DATAFRAME_METAL_LIBRARY=str(library),
                    DATAFRAME_THREADS=str(args.threads),
                    BENCH_ROWS=str(n),
                    BENCH_WORKLOAD=workload,
                    BENCH_VARIANT=variant,
                    BENCH_REPS=str(args.reps),
                    BENCH_GPU_FIRST=str((case_id + round_id) % 2),
                )
                run = {
                    "round": round_id,
                    "runtime": runtime,
                    "rows": n,
                    "workload": workload,
                    "variant": variant,
                    "gpu_first": env["BENCH_GPU_FIRST"] == "1",
                    "load_before": list(os.getloadavg()),
                    "log": label + ".txt",
                }
                try:
                    process = subprocess.run(
                        [str(binary)],
                        cwd=ROOT,
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=300,
                    )
                    (log_dir / run["log"]).write_text(process.stdout)
                    if process.returncode:
                        run.update(
                            status="unsupported" if "unsupported"
                            in process.stdout else "failed",
                            error=process.stdout,
                            exit_code=process.returncode,
                        )
                    else:
                        run.update(
                            parse_run(process.stdout, args.reps), status="ok"
                        )
                except (subprocess.TimeoutExpired, ValueError) as error:
                    run.update(status="failed", error=str(error))
                data["runs"].append(run)
                args.output.write_text(json.dumps(data, indent=2) + "\n")
                print(label, run["status"], flush=True)
    render(data, args.output.with_suffix(".md"))
    if any(r["status"] != "ok" for r in data["runs"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
