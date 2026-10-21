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
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKLOADS = [
    "sort_radix",
    "group_hash",
    "group_high",
    "group_skew",
    "join_hash",
    "join_fanout",
    "projection",
    "expression15",
    "cumulative_sum",
    "rolling_sum",
    "float64_arithmetic",
    "decimal_sum",
    "wide_float32",
    "chain32",
    "filter_half",
    "filter_sparse",
    "count",
    "int_sum",
    "filtered_int_sum",
    "filter_unused",
]
OPTIONAL_WORKLOADS = ["int32_affine_sum", "filtered_int32_affine_sum"]
OPTIONAL_NATIVE64_SUM_WORKLOADS = [
    "int64_sum",
    "uint64_sum",
    "int64_affine_sum",
    "int64_wide_sum16",
]
OPTIONAL_OPERATOR_WORKLOADS = [
    "sort_topk31",
    "sort_topk1024",
    "integer_cumulative_sum",
    "integer_rolling_min257",
    "integer_rolling_min4097",
    "integer_rolling_max4097",
]
WORKLOADS = (
    DEFAULT_WORKLOADS
    + OPTIONAL_WORKLOADS
    + OPTIONAL_NATIVE64_SUM_WORKLOADS
    + OPTIONAL_OPERATOR_WORKLOADS
)
VARIANTS = ["base", "sorted", "nulls"]
NATIVE64_SUM_CASES = {
    "int64_sum": {
        "input_dtype": "Int64",
        "input_values": "i % 17 - 8",
        "expression": "sum(x)",
        "maximum_absolute_result_per_input_row": 8,
    },
    "uint64_sum": {
        "input_dtype": "UInt64",
        "input_values": "i % 17",
        "expression": "sum(x)",
        "maximum_result_per_input_row": 16,
    },
    "int64_affine_sum": {
        "input_dtype": "Int64",
        "input_values": "i % 17 - 8",
        "expression": "sum((x + 3) * 2)",
        "maximum_absolute_result_per_input_row": 22,
    },
    "int64_wide_sum16": {
        "input_dtype": "Int64",
        "input_values": "i % 17 - 8",
        "expression": "16 separately aliased sum(x) outputs",
        "maximum_absolute_result_per_input_row": 8,
    },
}
NATIVE64_SUM_INPUT_VARIANTS = {
    "base": "Signed values i % 17 - 8; unsigned values i % 17",
    "sorted": (
        "Monotone bins over the same respective signed and unsigned ranges"
    ),
    "nulls": (
        "LCG values use UInt32-wrapped (1664525*i + 1013904223) before % 17; "
        "subtract 8 for signed inputs; every twentieth row null"
    ),
}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint(root):
    """Hash source paths and contents, including untracked files."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def build_receipt_path(binary):
    return Path(str(Path(binary).resolve()) + ".receipt.json")


def powerconditions():
    """Best-effort read-only macOS power and thermal snapshot."""
    snapshot = {}
    for name, args in (("battery", ["pmset", "-g", "batt"]),
                       ("thermal", ["pmset", "-g", "therm"])):
        try:
            result = subprocess.run(
                args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=5, check=False,
            )
            snapshot[name] = {
                "status": "available" if result.returncode == 0 else "unavailable",
                "returncode": result.returncode,
                "output": result.stdout.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            output = captured_text(error.stdout or error.output) if isinstance(
                error, subprocess.TimeoutExpired
            ) else str(error)
            snapshot[name] = {
                "status": "unavailable",
                "error": str(error),
                "output": output,
            }
    return snapshot


def capture_build_identity(compile_flags):
    compiler = command(["mojo", "--version"])
    compiler_path = shutil.which("mojo")
    source_diff = command(
        ["git", "diff", "--binary", "HEAD", "--", "dataframe"]
    )
    return {
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--porcelain"]),
        "engine_status": command(
            ["git", "status", "--porcelain", "--", "dataframe"]
        ),
        "engine_content_sha256": source_fingerprint(ROOT / "dataframe"),
        "engine_diff_sha256": hashlib.sha256(source_diff.encode()).hexdigest(),
        "compiler_identity": compiler,
        "build_platform": platform.platform(),
        "build_sdk": command(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
        "build_xcode": command(["xcodebuild", "-version"]),
        "compiler_executable": compiler_path,
        "compiler_executable_sha256": (
            sha(compiler_path)
            if compiler_path and Path(compiler_path).is_file()
            else None
        ),
        "pixi_lock_sha256": (
            sha(ROOT / "pixi.lock") if (ROOT / "pixi.lock").is_file() else None
        ),
        "benchmark_source_sha256": sha(ROOT / "benchmarks/bench_metal.mojo"),
        "driver_source_sha256": sha(__file__),
        "compile_flags": list(compile_flags),
        "powerconditions": powerconditions(),
    }


def write_build_receipt(binary, compile_flags, build_identity=None):
    binary = Path(binary).resolve()
    if build_identity is None:
        build_identity = capture_build_identity(compile_flags)
    receipt = {
        "schema_version": 1,
        "built_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "binary_path": str(binary),
        "binary_sha256": sha(binary),
        "build_command": [
            "mojo",
            "build",
            *compile_flags,
            "benchmarks/bench_metal.mojo",
            "-o",
            str(binary),
        ],
        **build_identity,
    }
    path = build_receipt_path(binary)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def load_build_receipt(binary):
    binary = Path(binary).resolve()
    binary_sha256 = sha(binary)
    path = build_receipt_path(binary)
    if not path.is_file():
        return {
            "status": "unknown",
            "reason": "No build receipt; binary source attribution is unknown",
            "binary_path": str(binary),
            "binary_sha256": binary_sha256,
        }
    receipt = json.loads(path.read_text())
    if receipt.get("binary_sha256") != binary_sha256:
        raise ValueError(f"Stale build receipt for {binary}")
    if receipt.get("schema_version") != 1:
        raise ValueError(f"Unsupported build receipt schema for {binary}")
    return {"status": "verified", **receipt}


def validate_paired_build_receipts(baseline, current):
    """Check equal harness, lock, compiler, SDK/Xcode and build-flag identity.

    Engine commit and source hashes intentionally differ between the pair. This
    guard does not prove full environment equivalence beyond the listed receipt
    fields; host hardware and runtime conditions are recorded separately.
    """
    identity_fields = (
        "benchmark_source_sha256",
        "driver_source_sha256",
        "pixi_lock_sha256",
        "compile_flags",
        "compiler_identity",
        "compiler_executable_sha256",
        "build_platform",
        "build_sdk",
        "build_xcode",
    )
    for name, receipt in (("baseline", baseline), ("current", current)):
        if receipt.get("status") != "verified":
            raise ValueError(
                f"The {name} binary needs a verified fresh build receipt"
            )
        missing = [
            field for field in identity_fields
            if receipt.get(field) is None
        ]
        if missing:
            raise ValueError(
                f"The {name} build receipt lacks paired identity fields: "
                + ", ".join(missing)
            )
    mismatched = [
        field for field in identity_fields
        if baseline[field] != current[field]
    ]
    if mismatched:
        raise ValueError(
            "Baseline and current binary must match on benchmark source, "
            "driver, lockfile, compiler, SDK/Xcode and compile flags; differing "
            "receipt fields: " + ", ".join(mismatched)
        )


def invocation_metadata(args, argv=None):
    return {
        "argv": list(sys.argv if argv is None else argv),
        "cwd": str(Path.cwd()),
        "build_cwd": str(ROOT),
        "python_executable": sys.executable,
        "timeout_seconds": args.timeout_seconds,
        "warmups": args.warmups,
        "binary": str(args.binary.resolve()),
        "baseline_binary": (
            str(args.baseline_binary.resolve())
            if args.baseline_binary
            else None
        ),
        "output": str(args.output.resolve()),
        "skip_build": args.skip_build,
    }


def command(args):
    return subprocess.check_output(
        args, cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def metadata(binaries, threads, warmups, binary_receipts):
    compiler = command(["mojo", "--version"])
    compiler_path = shutil.which("mojo")
    source_diff = command(
        ["git", "diff", "--binary", "HEAD", "--", "dataframe"]
    )
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
        "source_tree": {
            "status": command(["git", "status", "--porcelain"]),
            "dataframe_status": command(
                ["git", "status", "--porcelain", "--", "dataframe"]
            ),
            "dataframe_diff_sha256": hashlib.sha256(
                source_diff.encode()
            ).hexdigest(),
            "dataframe_content_sha256": source_fingerprint(ROOT / "dataframe"),
        },
        "engine_status": command(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "dataframe",
            ]
        ),
        "benchmark_status": command(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "benchmarks/bench_metal.mojo",
                "benchmarks/tools/bench_metal.py",
            ]
        ),
        "mojo_version": compiler,
        "compiler_identity": compiler,
        "compiler_executable": compiler_path,
        "compiler_executable_sha256": (
            sha(compiler_path)
            if compiler_path and Path(compiler_path).is_file()
            else None
        ),
        "pixi_lock_sha256": (
            sha(ROOT / "pixi.lock") if (ROOT / "pixi.lock").is_file() else None
        ),
        "xcode": command(["xcodebuild", "-version"]),
        "sdk": command(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
        "compile_flags": ["-O3", "-I", "."],
        "dataframe_threads": threads,
        "warmups_per_engine": warmups,
        "powerconditions": powerconditions(),
        "implementation": "mojo",
        "binaries_sha256": {name: sha(path) for name, path in binaries.items()},
        "binary_receipts": binary_receipts,
        "source_sha256": sha(ROOT / "benchmarks/bench_metal.mojo"),
        "driver_sha256": sha(__file__),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("DATAFRAME_", "PLANNER_"))
        },
        "method": "Fresh process per case/build/round; first-use collections retained separately; configured extra warmups per engine; alternating CPU/Metal timed order; process/build order rotated per round; inputs and exact CPU-reference validation untimed; complete result materialization timed; result release untimed.",
        "scope": "Development mechanisms only, not external-suite evidence or a general GPU speedup. Chain32 includes fusion advantages over the current CPU executor.",
        "data": "Row mechanisms use a deterministic 1024-value numeric distribution. Algorithm cases use Int32 keys: group_hash and group_skew have min(N,1024) classes; group_high, join_hash and sort_radix have max(1,N/2) classes; join_fanout uses ceil(N/4) cyclic or monotone balanced classes, so each key occurs at most four times and a self-left-join emits at most 4N rows. Base non-fanout keys are (37*i) modulo cardinality; sorted keys are monotone bins. group_skew maps three quarters of rows to key zero. Null variants invalidate every twentieth key; grouped values additionally invalidate every thirteenth row. Grouped sums use i modulo 17 minus 8; sort and self-left-join carry original Int64 row IDs. Float32 row values are exact multiples of 1/1024. Float64 arithmetic uses exact binary fractions k/1024 with k in [0,1023] and evaluates x*2+0.25. Decimal input is Decimal(15,2) cents with unscaled values in [-8,8]; the query multiplies by Decimal(15,2) 1.25 then sums. Wide Float32 projects eight columns with binary-exact offsets [0,0.125,...,0.875], at most 8N output values or 256 MiB at the default 8,388,608 rows. Optional int32_affine_sum and filtered_int32_affine_sum workloads use the same Int32 base/sorted/nulls input patterns as int_sum, compute sum(cast_Int64((x+3)*2)), and optionally filter x>0 before the expression. Optional native64 workloads use typed Int64 values i%17-8 and typed UInt64 values i%17; sorted variants use monotone bins over the same ranges, and null variants use an LCG-distributed value with every twentieth row null. int64_affine_sum sums (x+3)*2, whose per-row magnitude is at most 22; int64_wide_sum16 emits 16 differently aliased sums of the same Int64 input. The harness validates every output against its exact CPU result, which is not an independent oracle. The optional integer window workloads reuse the Int32 int_sum input patterns, retain all N output rows, and compute cumulative SUM or rolling MIN/MAX with min_count=1 and widths encoded in their names (257 or 4097). Optional sort_topk31 and sort_topk1024 reuse sort_radix keys and original Int64 row IDs, perform stable ascending nulls-last ordering, and retain min(N,K) rows. These optional workloads are explicitly selectable with --workloads and excluded from the default set.",
        "native64_sum_cases": NATIVE64_SUM_CASES,
        "limitations": [
            "No external-suite coverage claimed. Floating reductions remain outside these cases. Algorithm comparisons preserve complete output order and values against CPU; this CPU check is not an independent oracle.",
            "First use includes context/shader initialization but is not a cold machine or cold filesystem measurement.",
            "CPU and GPU share host memory and power limits; short timings can vary with scheduler and thermal state.",
            "The Mojo profile is a separate warm collection after timed samples; complete caller-wall timing remains the performance measure. Reported host phase timings are retained as measured and may overlap; they are not added. GPU-only time and pipeline compilation attribution are unavailable unless the runtime supplies those fields.",
            "Workload exposure: chain32 was used during fusion development; other cases broaden mechanism coverage without a holdout claim.",
            "Power and thermal snapshots use read-only pmset queries when available. Missing commands, unsupported output, and timeouts are retained as unavailable; the snapshot is context, not a controlled power measurement.",
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
        elif kind == "PROFILE_S":
            result["profile"][parts[1]] = " ".join(parts[2:])
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
    if result["profile"].get("runtime_implementation") != "mojo":
        raise ValueError(
            "Benchmark did not execute the Mojo GPU implementation"
        )
    if not result.get("runtime_build_id"):
        raise ValueError("Benchmark did not record its Mojo runtime build identity")
    for name in (
        "kernel_launches",
        "synchronizations",
        "upload_bytes",
        "download_bytes",
        "peak_requested_bytes",
    ):
        # Explicit whole-query counters take priority over region observations.
        value = result["profile"].get("query_" + name, result["profile"].get(name))
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                "Missing or invalid measured Mojo profile field: " + name
            )
        result["profile"][name] = value
    if (
        result["profile"].get("profile_regions", 1) > 1
        and "query_peak_requested_bytes" not in result["profile"]
    ):
        raise ValueError("Multi-region profile is missing its whole-query peak")
    return result


def captured_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def retain_timeout(error, timeout_seconds, log_path):
    captured = captured_text(error.stdout or error.output)
    Path(log_path).write_text(captured)
    return {
        "status": "failed",
        "error": f"Timed out after {timeout_seconds} seconds",
        "timeout_seconds": timeout_seconds,
        "captured_output": captured,
    }


def render(data, target):
    lines = [
        "# Apple Metal development measurements",
        "",
        data["metadata"]["scope"],
        "",
        "Times below are medians of complete warm collections, including allocation, staging, synchronization, and CPU result construction. First-use and all individual samples remain in the JSON. Every selected failed or unsupported case remains visible.",
        "",
        "| Runtime | Rows | Workload | Variant | CPU ms | Metal ms | CPU/Metal | Launches | Peak payload MiB |",
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
        launches = p.get("query_kernel_launches", p["kernel_launches"])
        peak = p.get("query_peak_requested_bytes", p["peak_requested_bytes"])
        ratio = cpu / metal
        ratios.setdefault((runtime, rows, workload), []).append(
            (ratio, variant)
        )
        lines.append(
            f"| {runtime} | {rows} | {workload} | {variant} | {cpu:.3f} | {metal:.3f} | {ratio:.2f} | {launches} | {peak / 1048576:.2f} |"
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
    slowdowns = [
        (runtime, rows, workload, variant, ratio)
        for (runtime, rows, workload, variant), runs in groups.items()
        if len([r for r in runs if r["status"] == "ok"]) == len(runs)
        for ratio in [
            statistics.median(s["cpu_ns"] for r in runs for s in r["samples"])
            / statistics.median(s["metal_ns"] for r in runs for s in r["samples"])
        ]
        if ratio < 1
    ]
    lines += [
        "",
        "## Measured Metal slowdowns",
        "",
        "Cases with CPU/Metal below one, ordered by the largest Metal slowdown. The slowdown factor is Metal time divided by CPU time.",
        "",
        "| Runtime | Rows | Workload | Variant | CPU/Metal | Metal slower by |",
        "|---|---:|---|---|---:|---:|",
    ]
    for runtime, rows, workload, variant, ratio in sorted(
        slowdowns, key=lambda item: item[-1]
    ):
        lines.append(
            f"| {runtime} | {rows} | {workload} | {variant} | {ratio:.2f} | {1 / ratio:.2f}x |"
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
        "--baseline-binary",
        type=Path,
        help="Optional separately built Mojo baseline executable",
    )
    parser.add_argument("--rows", default="65536,1048576,8388608")
    parser.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--warmups",
        type=int,
        default=2,
        help="Extra CPU and Metal collections per worker before timed samples",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Finite timeout for each benchmark process (default: 300 seconds)",
    )
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
    if min(args.reps, args.rounds, args.threads, args.timeout_seconds) < 1:
        parser.error("Repetitions, rounds, threads and timeout must be positive")
    if args.warmups < 0:
        parser.error("Warmups must be nonnegative")
    rows = [int(x) for x in args.rows.split(",")]
    workloads, variants = args.workloads.split(","), args.variants.split(",")
    if (
        min(rows) < 1
        or set(workloads) - set(WORKLOADS)
        or set(variants) - set(VARIANTS)
    ):
        parser.error("Invalid rows, workload or variant")
    binary = args.binary.resolve()
    compile_flags = ["-O3", "-I", "."]
    if not args.skip_build:
        binary.parent.mkdir(parents=True, exist_ok=True)
        build_identity = capture_build_identity(compile_flags)
        subprocess.run(
            [
                "mojo",
                "build",
                *compile_flags,
                "benchmarks/bench_metal.mojo",
                "-o",
                str(binary),
            ],
            cwd=ROOT,
            check=True,
        )
        write_build_receipt(binary, compile_flags, build_identity)
    binaries = {"mojo": binary}
    if args.baseline_binary:
        binaries["baseline"] = args.baseline_binary.resolve()
    binary_receipts = {
        name: load_build_receipt(executable)
        for name, executable in binaries.items()
    }
    if args.baseline_binary:
        validate_paired_build_receipts(
            binary_receipts["baseline"], binary_receipts["mojo"]
        )
    data = {
        "schema_version": 2,
        "metadata": metadata(
            binaries, args.threads, args.warmups, binary_receipts
        ),
        "invocation": invocation_metadata(args),
        "selection": {
            "rows": rows,
            "workloads": workloads,
            "variants": variants,
            "rounds": args.rounds,
            "reps": args.reps,
            "warmups": args.warmups,
            "threads": args.threads,
            "timeout_seconds": args.timeout_seconds,
        },
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.output.with_suffix("")
    log_dir.mkdir(exist_ok=True)
    cases = list(itertools.product(rows, workloads, variants))
    for round_id in range(args.rounds):
        ordered = cases if round_id % 2 == 0 else list(reversed(cases))
        builds = list(binaries.items())
        if round_id % 2:
            builds.reverse()
        for case_id, (n, workload, variant) in enumerate(ordered):
            for runtime, executable in builds:
                label = f"{round_id}-{runtime}-{n}-{workload}-{variant}"
                env = os.environ.copy()
                env.update(
                    DATAFRAME_THREADS=str(args.threads),
                    BENCH_ROWS=str(n),
                    BENCH_WORKLOAD=workload,
                    BENCH_VARIANT=variant,
                    BENCH_REPS=str(args.reps),
                    BENCH_WARMUPS=str(args.warmups),
                    BENCH_GPU_FIRST=str((case_id + round_id) % 2),
                )
                run = {
                    "round": round_id,
                    "runtime": runtime,
                    "rows": n,
                    "workload": workload,
                    "variant": variant,
                    "warmups": args.warmups,
                    "gpu_first": env["BENCH_GPU_FIRST"] == "1",
                    "load_before": list(os.getloadavg()),
                    "power_before": powerconditions(),
                    "log": label + ".txt",
                }
                captured = ""
                try:
                    process = subprocess.run(
                        [str(executable)],
                        cwd=ROOT,
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout_seconds,
                    )
                    captured = process.stdout
                    (log_dir / run["log"]).write_text(captured)
                    if process.returncode:
                        run.update(
                            status="unsupported" if "unsupported"
                            in process.stdout else "failed",
                            error=process.stdout,
                            captured_output=process.stdout,
                            exit_code=process.returncode,
                        )
                    else:
                        run.update(
                            parse_run(process.stdout, args.reps), status="ok"
                        )
                except subprocess.TimeoutExpired as error:
                    run.update(
                        retain_timeout(
                            error, args.timeout_seconds, log_dir / run["log"]
                        )
                    )
                except (ValueError, OSError) as error:
                    retained = captured or str(error)
                    (log_dir / run["log"]).write_text(retained)
                    run.update(
                        status="failed",
                        error=str(error),
                        captured_output=retained,
                    )
                data["runs"].append(run)
                args.output.write_text(json.dumps(data, indent=2) + "\n")
                print(label, run["status"], flush=True)
    render(data, args.output.with_suffix(".md"))
    if any(r["status"] != "ok" for r in data["runs"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
