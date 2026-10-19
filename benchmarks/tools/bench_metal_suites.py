#!/usr/bin/env python3
"""Check unchanged external-suite inputs and expressions with Metal forced.

This is a correctness/coverage driver, not a performance report. Generated
sources make every materialization explicit and are retained with their hashes.
H2O's eager expressions are wrapped as equivalent lazy queries only in the
generated Metal worker; the original CPU worker remains eager. No unsupported
query retries on CPU. Every selected case has one isolated process and outcome.
"""
import argparse
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tokenize
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_suites as suites
import bench_policy
import metal_query_report
from metal_materialization_adapter import lift_scalar_selects
from metal_answer_exports import parse_export, collect_exports


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive(source, name, engine, *, audit_runtime=False):
    """Only change materialization boundaries, preserving query expressions."""
    if engine not in {"cpu", "metal"}:
        raise ValueError(engine)
    if name not in {"h2o", "pdsh", "tpcds", "clickbench"}:
        raise ValueError(name)
    if re.search(r"\.collect\((?!\))", source):
        raise ValueError(
            "Suite has a collect with arguments; review derivation"
        )
    if engine == "cpu":
        return source.replace(".collect()", '.collect(engine="cpu")')
    source, _ = lift_scalar_selects(source)
    runtime = "MetalRuntime"
    source = source.replace(
        ".collect()", f".collect(accelerator={runtime}())"
    )
    if name == "h2o":
        if source.count('ref x = t["x"]') != 2:
            raise ValueError("H2O scan definitions changed; review derivation")
        source = source.replace('ref x = t["x"]', 'var x = t["x"].lazy()')
        source = re.sub(r't\["(small|medium|big)"\]', r't["\1"].lazy()', source)
        # Wrap each query's returned expression as one explicit Metal collect.
        # Token boundaries respect multiline parentheses, comments and strings.
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))

        def offset(position):
            return offsets[position[0] - 1] + position[1]

        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        replacements = []
        for index, token in enumerate(tokens):
            if token.type == tokenize.NAME and token.string == "return":
                if token.start[1] != 8:
                    raise ValueError("Unexpected H2O return; review derivation")
                end = next(
                    t for t in tokens[index + 1 :] if t.type == tokenize.NEWLINE
                )
                start, stop = offset(token.end), offset(end.start)
                expression = source[start:stop].strip()
                replacements.append(
                    (
                        start,
                        stop,
                        " ("
                        + expression
                        + f").collect(accelerator={runtime}())",
                    )
                )
        if len(replacements) != 15:
            raise ValueError("H2O query count changed; review derivation")
        for start, stop, value in reversed(replacements):
            source = source[:start] + value + source[stop:]
    # Mojo only accepts the module docstring before imports. Keep it first.
    source, imports = re.subn(
        r"(?m)^from ",
        (
            "from metal_coverage_audit import CoverageAuditRuntime as MetalRuntime\nfrom "
            if audit_runtime
            else "from dataframe.metal import MetalRuntime\nfrom "
        ),
        source,
        count=1,
    )
    if imports != 1:
        raise ValueError("Suite has no import boundary; review derivation")
    return source


def stage_sources(destination, *, audit_runtime=True):
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    materialization_proofs = {}
    for name in ("h2o", "pdsh", "tpcds", "clickbench"):
        original = ROOT / "benchmarks/suites" / (name + ".mojo")
        copied[str(original.relative_to(ROOT))] = sha(original)
        _, proof = lift_scalar_selects(original.read_text())
        proof["canonical_source_sha256"] = sha(original)
        proof["metal_derived_source_sha256"] = hashlib.sha256(
            derive(original.read_text(), name, "metal", audit_runtime=audit_runtime).encode()
        ).hexdigest()
        materialization_proofs[str(original.relative_to(ROOT))] = proof
        for engine in ("cpu", "metal"):
            (destination / f"{engine}_{name}.mojo").write_text(
                derive(
                    original.read_text(),
                    name,
                    engine,
                    audit_runtime=audit_runtime and engine == "metal",
                )
            )
        if name == "h2o":
            wrapper = """from std.sys import argv
from metal_suite_common import run
from cpu_h2o import groupby as cpu_groupby, join as cpu_join
from metal_h2o import groupby as metal_groupby, join as metal_join
def main() raises:
    if String(argv()[1]) == "h2o_groupby":
        run[cpu_groupby, metal_groupby]()
    else:
        run[cpu_join, metal_join]()
"""
        else:
            wrapper = f'''from metal_suite_common import run
from cpu_{name} import query as cpu_query
from metal_{name} import query as metal_query
def main() raises:
    run[cpu_query, metal_query]()
'''
        (destination / f"suite_{name}.mojo").write_text(wrapper)
    copied_sources = {
        "suite_common.mojo": ROOT / "benchmarks/suites/suite_common.mojo",
        "metal_suite_common.mojo": ROOT / "benchmarks/metal_suite_common.mojo",
        "metal_suite_answer_exports.mojo": ROOT
        / "benchmarks/metal_suite_answer_exports.mojo",
        "metal_suite_diagnostics.mojo": ROOT
        / "benchmarks/metal_suite_diagnostics.mojo",
    }
    if audit_runtime:
        copied_sources["metal_coverage_audit.mojo"] = (
            ROOT / "benchmarks/metal_coverage_audit.mojo"
        )
    for name, original in copied_sources.items():
        shutil.copyfile(original, destination / name)
        copied[str(original.relative_to(ROOT))] = sha(original)
    return {
        "original_source_sha256": copied,
        "scalar_materialization_proofs": materialization_proofs,
        "scalar_materialization_adapter_sha256": sha(
            Path(__file__).with_name("metal_materialization_adapter.py")
        ),
        "derived_source_sha256": {
            p.name: sha(p) for p in destination.glob("*.mojo")
        },
        "derivation": "Only materialization boundaries changed; token-reviewed scalar SELECT/item after collected DataFrames gains lazy/forced-collect boundaries, with exact expression hashes. H2O eager expressions adapted to lazy scans in the Metal worker. CPU expressions and all input dtypes retained. Every collect in Metal query modules explicitly names the forced runtime; no automatic placement or CPU retry. Coverage-only builds wrap MetalRuntime to record its returned fault_adjudication marker; timing-worker builds use MetalRuntime directly.",
    }


def parse(output, query):
    rows = [
        line.split("\t", 3)
        for line in output.splitlines()
        if line.startswith("coverage\t")
    ]
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][1] != query:
        raise ValueError("Missing or ambiguous per-query coverage outcome")
    status = rows[0][2]
    if status not in {
        "ok",
        "unsupported",
        "execution_failed",
        "wrong_answer",
        "reference_unsupported",
        "reference_failed",
    }:
        raise ValueError("Unknown coverage outcome")
    result = {"status": status, "reason": rows[0][3]}
    diagnostics = {}
    exports = []
    runtime_events = []
    for line in output.splitlines():
        fields = line.split("\t")
        if fields[0] == "runtime_audit":
            if len(fields) != 2 or fields[1] not in {
                "cpu",
                "no_marker",
                "unknown",
                "failed",
            }:
                raise ValueError("Malformed runtime adjudication audit")
            runtime_events.append(fields[1])
        elif fields[0] == "answer_export":
            if len(fields) != 2:
                raise ValueError("Malformed answer export line")
            exports.append(parse_export(fields[1]))
        elif len(fields) == 2 and fields[0] in {
            "output_rows",
            "collections",
            "collection_ns",
            "expected_rows",
            "actual_rows",
        }:
            result[fields[0]] = int(fields[1])
        elif len(fields) == 2 and fields[0] == "ordered_equal":
            if fields[1] not in {"True", "False"}:
                raise ValueError("Invalid ordered equality")
            result["ordered_equal"] = fields[1] == "True"
        elif len(fields) == 2 and fields[0] == "cpu_repeat_equal":
            if fields[1] not in {"True", "False"}:
                raise ValueError("Invalid CPU repeat equality")
            diagnostics["cpu_repeat_ordered_equal"] = fields[1] == "True"
        elif fields[0] == "diagnostic":
            if len(fields) != 3 or fields[1] in diagnostics:
                raise ValueError("Malformed or duplicate diagnostic")
            name, value = fields[1:]
            if not name.startswith(("cpu_metal.", "cpu_repeat.")):
                raise ValueError("Unknown comparison diagnostic")
            if name.endswith(("_equal", "_ties")):
                if value not in {"True", "False"}:
                    raise ValueError("Invalid diagnostic Boolean")
                value = value == "True"
            diagnostics[name] = value
        elif fields[0] == "column_bag":
            if (
                len(fields) != 4
                or fields[1] not in {"cpu_metal", "cpu_repeat"}
                or fields[3] not in {"True", "False"}
            ):
                raise ValueError("Malformed column bag diagnostic")
            values = diagnostics.setdefault(fields[1] + ".column_bags", {})
            if fields[2] in values:
                raise ValueError("Duplicate column bag diagnostic")
            values[fields[2]] = fields[3] == "True"
        elif fields[0] == "column_diagnostic":
            if len(fields) != 11 or fields[1] not in {
                "cpu_metal",
                "cpu_repeat",
            }:
                raise ValueError("Malformed numeric column diagnostic")
            counts = [int(value) for value in fields[4:8]]
            if min(counts) < 0 or sum(counts[1:]) > counts[0]:
                raise ValueError("Invalid column mismatch counts")
            metrics = dict(
                zip(
                    (
                        "mismatches",
                        "null_mismatches",
                        "finite_float_mismatches",
                        "nonfinite_float_mismatches",
                    ),
                    counts,
                )
            )
            metrics["dtype"] = fields[3]
            if counts[2] == 0:
                if fields[8:] != ["unavailable"] * 3:
                    raise ValueError(
                        "Distances require finite floating mismatches"
                    )
                metrics.update(
                    max_ulp=None, max_absolute=None, max_relative=None
                )
            else:
                ulp = int(fields[8])
                if not 0 <= ulp <= (1 << 64) - 1 or any(
                    math.isnan(float(v)) or float(v) < 0 for v in fields[9:]
                ):
                    raise ValueError("Invalid finite floating distance")
                metrics.update(
                    max_ulp=ulp, max_absolute=fields[9], max_relative=fields[10]
                )
            values = diagnostics.setdefault(fields[1] + ".columns", {})
            if fields[2] in values:
                raise ValueError("Duplicate numeric column diagnostic")
            values[fields[2]] = metrics
        elif fields[0] == "diagnostic_error":
            if len(fields) != 3 or fields[1] not in {"cpu_metal", "cpu_repeat"}:
                raise ValueError("Malformed diagnostic error")
            diagnostics.setdefault("errors", []).append(
                {"comparison": fields[1], "reason": fields[2]}
            )
    if runtime_events:
        runtime = {
            "status": "complete",
            "observed_calls": len(runtime_events),
            "cpu_adjudicated_calls": runtime_events.count("cpu"),
            "no_cpu_adjudication_marker_calls": runtime_events.count("no_marker"),
            "unknown_calls": runtime_events.count("unknown"),
            "failed_calls": runtime_events.count("failed"),
            "events": runtime_events,
            "interpretation": (
                "no_cpu_adjudication_marker means only that the returned report "
                "had no CPU data-fault adjudication marker; it is not a GPU-only certificate."
            ),
        }
        if runtime["unknown_calls"] or runtime["failed_calls"]:
            runtime["status"] = "incomplete"
        result["runtime_adjudication"] = runtime
    elif status == "ok":
        raise ValueError("Successful outcome lacks per-collection runtime audit")
    if status == "ok" and (
        result.get("collections", 0) < 1 or "ordered_equal" not in result
    ):
        raise ValueError(
            "Successful outcome lacks correctness/repetition evidence"
        )
    if status == "ok" and (
        result["runtime_adjudication"]["status"] != "complete"
        or result["runtime_adjudication"]["observed_calls"]
        < result.get("collections", 0)
    ):
        raise ValueError(
            "Successful outcome lacks complete per-collection runtime audit"
        )
    if diagnostics:
        result["diagnostics"] = diagnostics
    if exports:
        result["answer_exports"] = exports
    return result


def case_audit(
    metadata, suite, variant, query, source_hashes, derived_source_hashes=None
):
    entry = metadata.get("cases", {}).get(suite + "/" + variant + "/" + query)
    if entry is None:
        entry = metadata.get("cases", {}).get(suite + "/" + query)
    if entry is None:
        module = suites.SUITES[suite]["runner"]
        path = "benchmarks/suites/" + module + ".mojo"
        return {
            "workload_kind": "canonical",
            "execution_boundary": "gpu_collects",
            "source_path": path,
            "source_sha256": source_hashes[path],
            "audit_reason": "Every collect is forced; no separate eager-operation certificate.",
        }
    metal_query_report.validate_metadata(
        {
            "format": "dataframe-metal-case-metadata-v1",
            "cases": {"selected": entry},
        }
    )
    result = dict(entry)
    module = suites.SUITES[suite]["runner"]
    derived_path = "metal_" + module + ".mojo"
    canonical_matches = (
        source_hashes.get(entry["source_path"]) == entry["source_sha256"]
    )
    derived_pin = entry.get("metal_derived_source_sha256")
    if derived_pin is not None and (
        not isinstance(derived_pin, str) or not derived_pin
    ):
        raise ValueError("Invalid derived worker source pin")
    derived_matches = derived_pin is None or (
        derived_source_hashes is not None
        and derived_source_hashes.get(derived_path) == derived_pin
    )
    if not canonical_matches or not derived_matches:
        result.update(
            execution_boundary="unknown",
            alignment_keys=[],
            order_keys=[],
            whole_query_gpu_certificate=False,
            audit_reason="Case audit canonical or derived worker source hash does not match current workload.",
        )
    for field in ("alignment_keys", "order_keys"):
        keys = result.get(field, [])
        if (
            not isinstance(keys, list)
            or any(
                not isinstance(k, str) or not k or "\t" in k or "\n" in k
                for k in keys
            )
            or len(set(keys)) != len(keys)
        ):
            raise ValueError("Invalid declared diagnostic keys")
    return result


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def driver_provenance():
    return {
        "driver_sha256": sha(__file__),
        "answer_export_verifier_sha256": sha(
            Path(__file__).with_name("metal_answer_exports.py")
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", default="h2o_groupby,h2o_join,pdsh")
    parser.add_argument("--all-suites", action="store_true")
    parser.add_argument("--scale", choices=list(suites.SCALES), default="smoke")
    parser.add_argument("--queries", default="")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/metal-suites/results.json"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--export-answers",
        action="store_true",
        help="Export complete typed CPU/Metal answers outside collection timing; CPU repeat when the existing mismatch diagnostic runs.",
    )
    parser.add_argument(
        "--case-metadata",
        type=Path,
        default=ROOT / "benchmarks/tools/metal_case_metadata.json",
    )
    args = parser.parse_args()
    selected = list(suites.SUITES) if args.all_suites else args.suites.split(
        ","
    )
    if (
        not set(selected) <= set(suites.SUITES)
        or min(args.reps, args.threads, args.timeout) < 1
    ):
        parser.error("Unknown suite or nonpositive execution setting")
    wanted = set(args.queries.split(",")) if args.queries else None
    out = args.output.resolve().parent
    sources = out / "sources"
    provenance = stage_sources(sources)
    metadata = json.loads(args.case_metadata.read_text())
    metal_query_report.validate_metadata(metadata)
    provenance["case_metadata_sha256"] = sha(args.case_metadata)
    if args.prepare_only:
        print(json.dumps(provenance, indent=2))
        return
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Execution requires an Apple Silicon Mac")
    binaries = {}
    for name in sorted({suites.SUITES[s]["runner"] for s in selected}):
        binary = out / ("suite_" + name)
        if not args.skip_build:
            subprocess.run(
                [
                    "mojo",
                    "build",
                    "-O3",
                    "-I",
                    str(ROOT),
                    "-I",
                    str(sources),
                    str(sources / ("suite_" + name + ".mojo")),
                    "-o",
                    str(binary),
                ],
                cwd=ROOT,
                check=True,
            )
        binaries[name] = binary
    data = {
        "format": "dataframe-metal-suite-coverage-v1",
        "provenance": {
            **provenance,
            "recorded_at_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "engine_revision": command(["git", "rev-parse", "HEAD"]),
            "engine_status": command(
                ["git", "status", "--porcelain", "--", "dataframe"]
            ),
            "mojo": command(["mojo", "--version"]),
            "platform": platform.platform(),
            **driver_provenance(),
            "answer_exports_requested": args.export_answers,
            "implementation": "mojo",
            "binaries_sha256": {
                name: sha(path) for name, path in binaries.items()
            },
            "threads": args.threads,
            "scale": args.scale,
            "repetitions": args.reps,
            "evaluation": bench_policy.record(ROOT, selected, suites.SUITES),
            "comparison": "Exact full schema and row multiset against original CPU query, retaining duplicate rows. Original-order equality recorded separately. No numerical tolerance, dtype conversion, or query rewriting for support. The requested Metal backend may use CPU data-fault adjudication according to its engine policy; collect-boundary audits do not certify GPU-only execution. Collection durations are diagnostic, not performance evidence.",
        },
        "cases": [],
    }
    (out / "logs").mkdir(exist_ok=True)
    for suite in selected:
        queries = [
            q
            for q in suites.SUITES[suite]["queries"]
            if wanted is None or q in wanted
        ]
        for variant in suites.VARIANTS[suite]:
            paths = suites.tables(suite, variant, args.scale)
            inputs = {
                name: {"path": str(path), "sha256": sha(path)}
                for name, path in paths.items()
            }
            for query in queries:
                label = f"{suite}-{variant}-{query}"
                case = {
                    "suite": suite,
                    "variant": variant,
                    "query": query,
                    "inputs": inputs,
                    "log": f"logs/{label}.txt",
                }
                audit = case_audit(
                    metadata,
                    suite,
                    variant,
                    query,
                    provenance["original_source_sha256"],
                    provenance["derived_source_sha256"],
                )
                case["execution_audit"] = audit
                env = os.environ.copy()
                env.update(DATAFRAME_THREADS=str(args.threads))
                env.pop("DATAFRAME_BENCH_ANSWER_EXPORT_PREFIX", None)
                export_prefix = None
                if args.export_answers:
                    (out / "answers").mkdir(exist_ok=True)
                    export_prefix = (
                        Path(
                            tempfile.mkdtemp(
                                prefix=label + "-", dir=out / "answers"
                            )
                        )
                        / "answer"
                    )
                    env["DATAFRAME_BENCH_ANSWER_EXPORT_PREFIX"] = str(
                        export_prefix
                    )
                env["DATAFRAME_BENCH_ALIGNMENT_KEYS"] = "\t".join(
                    audit.get("alignment_keys", [])
                )
                env["DATAFRAME_BENCH_ORDER_KEYS"] = "\t".join(
                    audit.get("order_keys", [])
                )
                call = [
                    str(binaries[suites.SUITES[suite]["runner"]]),
                    suite,
                    query,
                    str(args.reps),
                ] + [f"{name}={path}" for name, path in paths.items()]
                try:
                    process = subprocess.run(
                        call,
                        cwd=ROOT,
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                    )
                    (out / case["log"]).write_text(process.stdout)
                    if process.returncode:
                        case.update(
                            status="process_failed",
                            reason=process.stdout,
                            exit_code=process.returncode,
                        )
                    else:
                        case.update(parse(process.stdout, query))
                except subprocess.TimeoutExpired as error:
                    raw = error.stdout or b""
                    (out / case["log"]).write_text(
                        raw.decode(errors="replace") if isinstance(
                            raw, bytes
                        ) else raw
                    )
                    case.update(
                        status="timeout",
                        reason=f"Exceeded {args.timeout} seconds",
                    )
                except ValueError as error:
                    case.update(status="protocol_failed", reason=str(error))
                if export_prefix is not None:
                    try:
                        case["answer_exports"] = collect_exports(
                            case.get("answer_exports", []),
                            export_prefix,
                            args.reps,
                        )
                    except ValueError as error:
                        case["answer_exports"] = {
                            "status": "unavailable",
                            "error": str(error),
                        }
                data["cases"].append(case)
                args.output.write_text(json.dumps(data, indent=2) + "\n")
                print(label, case["status"], case.get("reason", ""), flush=True)
    # Unsupported coverage is expected; wrong answers and execution failures
    # still make the validation command fail while retaining the complete grid.
    sys.exit(
        any(
            c["status"] not in {"ok", "unsupported", "reference_unsupported"}
            for c in data["cases"]
        )
    )


if __name__ == "__main__":
    main()
