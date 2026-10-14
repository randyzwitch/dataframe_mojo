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
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tokenize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_suites as suites
import bench_policy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive(source, name, engine):
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
    source = source.replace(
        ".collect()", ".collect(accelerator=MetalRuntime())"
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
                        + ").collect(accelerator=MetalRuntime())",
                    )
                )
        if len(replacements) != 15:
            raise ValueError("H2O query count changed; review derivation")
        for start, stop, value in reversed(replacements):
            source = source[:start] + value + source[stop:]
    # Mojo only accepts the module docstring before imports. Keep it first.
    source, imports = re.subn(
        r"(?m)^from ",
        "from dataframe.metal import MetalRuntime\nfrom ",
        source,
        count=1,
    )
    if imports != 1:
        raise ValueError("Suite has no import boundary; review derivation")
    return source


def stage_sources(destination):
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in ("h2o", "pdsh", "tpcds", "clickbench"):
        original = ROOT / "benchmarks/suites" / (name + ".mojo")
        copied[str(original.relative_to(ROOT))] = sha(original)
        for engine in ("cpu", "metal"):
            (destination / f"{engine}_{name}.mojo").write_text(
                derive(original.read_text(), name, engine)
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
    for name, original in {
        "suite_common.mojo": ROOT / "benchmarks/suites/suite_common.mojo",
        "metal_suite_common.mojo": ROOT / "benchmarks/metal_suite_common.mojo",
    }.items():
        shutil.copyfile(original, destination / name)
        copied[str(original.relative_to(ROOT))] = sha(original)
    return {
        "original_source_sha256": copied,
        "derived_source_sha256": {
            p.name: sha(p) for p in destination.glob("*.mojo")
        },
        "derivation": "Only materialization boundaries changed; H2O eager expressions adapted to lazy scans in the Metal worker. CPU expressions and all input dtypes retained. Every collect in Metal query modules explicitly names MetalRuntime; no automatic placement or CPU retry.",
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
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) == 2 and fields[0] in {
            "output_rows",
            "collections",
            "collection_ns",
        }:
            result[fields[0]] = int(fields[1])
        elif len(fields) == 2 and fields[0] == "ordered_equal":
            result["ordered_equal"] = fields[1] == "True"
    if status == "ok" and (
        result.get("collections", 0) < 1 or "ordered_equal" not in result
    ):
        raise ValueError(
            "Successful outcome lacks correctness/repetition evidence"
        )
    return result


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


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
            "driver_sha256": sha(__file__),
            "implementation": "mojo",
            "binaries_sha256": {
                name: sha(path) for name, path in binaries.items()
            },
            "threads": args.threads,
            "scale": args.scale,
            "repetitions": args.reps,
            "evaluation": bench_policy.record(ROOT, selected, suites.SUITES),
            "comparison": "Exact full schema and row multiset against original CPU query, retaining duplicate rows. Original-order equality recorded separately. No numerical tolerance, dtype conversion, query rewriting for support, or CPU fallback. Collection durations are diagnostic, not performance evidence.",
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
                env = os.environ.copy()
                env.update(DATAFRAME_THREADS=str(args.threads))
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
