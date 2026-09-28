"""Run the external benchmark suites against Polars and DuckDB.

See docs/benchmarks.md for the rules these measurements serve. Run under the
oracle environment, after building libdfparquet:

    pixi run -e native build-dfparquet
    pixi run -e oracle python3 scripts/bench_suites.py                 # dev suites
    pixi run -e oracle python3 scripts/bench_suites.py --heldout       # + held-out
    pixi run -e oracle python3 scripts/bench_suites.py --scale smoke --trace

Each (suite, variant, query, engine) runs in a fresh process that loads its
tables untimed, warms up once and times --reps runs; the per-round figure is
the fastest run and the report shows the median over --rounds rounds, with
engine order rotated between rounds. Every answer is checked against DuckDB
(see engines.py `summary`); a wrong answer is reported as such and excluded
from ratios, never silently timed. Unsupported queries are counted, not
dropped. Data lives outside the repository (benchmarks/suites/datagen.py).
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SUITES_DIR = ROOT / "benchmarks" / "suites"
sys.path.insert(0, str(SUITES_DIR))
sys.path.insert(0, str(ROOT / "scripts"))

import datagen  # noqa: E402
from bench_upstream_joins import compiler_processes  # noqa: E402

ENGINES = ("mojo", "polars", "duckdb")

# Development suites may be used to find and tune optimizations. Held-out
# suites are only for reporting: nobody reads their per-query results to
# decide what to optimize (docs/benchmarks.md).
SUITES = {
    "h2o_groupby": {
        "role": "dev",
        "runner": "h2o",
        "queries": [f"q{i}" for i in range(1, 11)],
        "source": "H2O.ai db-benchmark (duckdblabs/db-benchmark) group-by",
    },
    "h2o_join": {
        "role": "dev",
        "runner": "h2o",
        "queries": [f"q{i}" for i in range(1, 6)],
        "source": "H2O.ai db-benchmark (duckdblabs/db-benchmark) join",
    },
    "pdsh": {
        "role": "heldout",
        "runner": "pdsh",
        "queries": [f"q{i}" for i in range(1, 23)],
        "source": "PDS-H (pola-rs/polars-benchmark), TPC-H derived",
    },
    "clickbench": {
        "role": "heldout",
        "runner": "clickbench",
        "queries": [f"q{i}" for i in range(0, 43)],
        "source": "ClickBench (ClickHouse/ClickBench) hits",
    },
}

# Data sizes. "smoke" checks answers in CI; "default" is for measurement.
SCALES = {
    "smoke": {"h2o_rows": 100_000, "pdsh_sf": 0.01, "clickbench_partitions": 1},
    "default": {
        "h2o_rows": 10_000_000,
        "pdsh_sf": 1,
        "clickbench_partitions": 10,
    },
    "large": {
        "h2o_rows": 100_000_000,
        "pdsh_sf": 10,
        "clickbench_partitions": 100,
    },
}

# Data variants that every measurement runs; the report shows each query's
# worst variant beside its base result (docs/benchmarks.md, rule 4). The
# H2O set follows db-benchmark's own axes: group cardinality, nulls, order.
VARIANTS = {
    "h2o_groupby": {
        "k100": {"k": 100, "nas": 0, "sort": False},
        "k10": {"k": 10, "nas": 0, "sort": False},
        "k2": {"k": 2, "nas": 0, "sort": False},
        "k100_na5": {"k": 100, "nas": 5, "sort": False},
        "k100_sorted": {"k": 100, "nas": 0, "sort": True},
    },
    "h2o_join": {"na0": {"nas": 0}, "na5": {"nas": 5}},
    "pdsh": {"base": {}},
    "clickbench": {"base": {}},
}

# Output columns to compare, by position or name, where a query's full result
# is not unique: rows tied at a LIMIT boundary may differ between engines (and
# between runs of one engine), but the ORDER BY columns may not. An empty
# list compares only the row count (an unordered LIMIT).
CHECKS = {
    ("clickbench", "q8"): [1],
    ("clickbench", "q9"): [2],
    ("clickbench", "q10"): [1],
    ("clickbench", "q11"): [2],
    ("clickbench", "q12"): [1],
    ("clickbench", "q13"): [1],
    ("clickbench", "q14"): [2],
    ("clickbench", "q15"): [1],
    ("clickbench", "q16"): [2],
    ("clickbench", "q17"): [],
    ("clickbench", "q18"): [3],
    ("clickbench", "q21"): [2],
    ("clickbench", "q22"): [3],
    ("clickbench", "q23"): ["EventTime"],
    ("clickbench", "q24"): [],
    ("clickbench", "q27"): [1],
    ("clickbench", "q28"): [1],
    ("clickbench", "q30"): [2],
    ("clickbench", "q31"): [2],
    ("clickbench", "q32"): [2],
    ("clickbench", "q33"): [1],
    ("clickbench", "q34"): [2],
    ("clickbench", "q35"): [4],
    ("clickbench", "q36"): [1],
    ("clickbench", "q37"): [1],
    ("clickbench", "q38"): [1],
    ("clickbench", "q39"): [5],
    ("clickbench", "q40"): [2],
    ("clickbench", "q41"): [2],
}


def tables(suite, variant, scale):
    size = SCALES[scale]
    spec = VARIANTS[suite][variant]
    if suite == "h2o_groupby":
        path = datagen.h2o_groupby(
            size["h2o_rows"], spec["k"], spec["nas"], spec["sort"]
        )
        return {"x": path}
    if suite == "h2o_join":
        return datagen.h2o_join(size["h2o_rows"], spec["nas"])
    if suite == "pdsh":
        root = datagen.pdsh(size["pdsh_sf"])
        return {name: root / f"{name}.parquet" for name in datagen.PDSH_TABLES}
    return {"hits": datagen.clickbench(size["clickbench_partitions"])}


# --- building and running workers ------------------------------------------


def build_runners(names, force=False):
    out = ROOT / "build" / "suites"
    out.mkdir(parents=True, exist_ok=True)
    sources = list((ROOT / "dataframe").glob("*.mojo")) + list(
        SUITES_DIR.glob("*.mojo")
    )
    newest = max(path.stat().st_mtime for path in sources)
    binaries = {}
    for name in sorted(set(names)):
        binary = out / f"suite_{name}"
        if force or not binary.exists() or binary.stat().st_mtime < newest:
            print(f"building {binary.relative_to(ROOT)}", file=sys.stderr)
            subprocess.check_call(
                [
                    "mojo",
                    "build",
                    "-O3",
                    "-I",
                    str(ROOT),
                    "-I",
                    str(SUITES_DIR),
                    str(SUITES_DIR / f"{name}.mojo"),
                    "-o",
                    str(binary),
                ],
                cwd=ROOT,
            )
        binaries[name] = binary
    return binaries


def command(engine, binaries, suite, query, reps, paths):
    args = [suite, query, str(reps)] + [f"{k}={v}" for k, v in paths.items()]
    if engine == "mojo":
        return [str(binaries[SUITES[suite]["runner"]])] + args
    return [sys.executable, str(SUITES_DIR / "engines.py"), engine] + args


def environment(threads, trace=False):
    env = dict(os.environ)
    env["DATAFRAME_THREADS"] = str(threads)
    env["POLARS_MAX_THREADS"] = str(threads)
    env["BENCH_THREADS"] = str(threads)
    if trace:
        env["DATAFRAME_TRACE_PATHS"] = "1"
    else:
        env.pop("DATAFRAME_TRACE_PATHS", None)
    return env


def parse(output):
    times, summary, reason = [], None, None
    for line in output.splitlines():
        fields = line.split("\t")
        if fields[0] == "time":
            times.append(int(fields[1]))
        elif fields[0] == "summary":
            values = [float(v) for v in fields[2].split(",")] if fields[2] else []
            names = fields[3].split(",") if len(fields) > 3 and fields[3] else []
            summary = {"height": int(fields[1]), "values": values, "names": names}
        elif fields[0] == "unsupported":
            reason = fields[1]
    return times, summary, reason


def _wait_quiet(limit=3600):
    """Block until no compiler runs; other work on a shared host must not
    overlap timing (docs/benchmarks.md, and the benchmark-hygiene notes)."""
    waited = 0
    while compiler_processes():
        if waited >= limit:
            raise RuntimeError("compiler activity persisted for an hour")
        if waited == 0:
            print("waiting for compiler activity to stop", file=sys.stderr)
        time.sleep(10)
        waited += 10


def run_worker(cmd, env, timeout, allow_busy):
    for _ in range(3):
        if not allow_busy:
            _wait_quiet()
        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        if allow_busy or not compiler_processes():
            break
        print("compiler activity overlapped a run; repeating it", file=sys.stderr)
    else:
        raise RuntimeError("compiler activity kept overlapping timed runs")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return {"status": "failed", "reason": " | ".join(tail)}
    times, summary, reason = parse(proc.stdout)
    if reason is not None:
        return {"status": "unsupported", "reason": reason}
    paths = sorted(
        set(re.findall(r"^dataframe-path: (\S+)$", proc.stderr, re.M))
    )
    return {"status": "ok", "times": times, "summary": summary, "paths": paths}


# --- answer checks ----------------------------------------------------------


def _close(a, b):
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def _positions(spec, summary):
    out = []
    for item in spec:
        if isinstance(item, int):
            out.append(item)
        elif item in summary["names"]:
            out.append(summary["names"].index(item))
        else:
            return None
    return out


def same_answer(suite, query, got, want):
    if got["height"] != want["height"]:
        return False
    spec = CHECKS.get((suite, query))
    if spec is None:
        if len(got["values"]) != len(want["values"]):
            return False
        pairs = zip(got["values"], want["values"])
    else:
        mine, theirs = _positions(spec, got), _positions(spec, want)
        if mine is None or theirs is None:
            return False
        if max(mine + theirs, default=-1) >= min(
            len(got["values"]), len(want["values"])
        ):
            return False
        pairs = [(got["values"][i], want["values"][j]) for i, j in zip(mine, theirs)]
    return all(_close(a, b) for a, b in pairs)


# --- measurement -------------------------------------------------------------


def provenance(args):
    def run(cmd):
        try:
            return subprocess.check_output(
                cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    import duckdb
    import polars

    cpu = "unknown"
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    else:
        cpu = run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain", "dataframe"])),
        "mojo": run(["mojo", "--version"]),
        "polars": polars.__version__,
        "duckdb": duckdb.__version__,
        "platform": platform.platform(),
        "cpu": cpu,
        "threads": args.threads,
        "scale": args.scale,
        "rounds": args.rounds,
        "reps": args.reps,
        "data_root": str(datagen.data_root()),
    }


def measure(args):
    suites = [s for s in args.suites.split(",") if s]
    engines = [e for e in args.engines.split(",") if e]
    binaries = {}
    if "mojo" in engines or args.trace:
        binaries = build_runners(
            [SUITES[s]["runner"] for s in suites], force=args.rebuild
        )
    result = {"provenance": provenance(args), "runs": [], "trace": {}}
    plan = []
    for suite in suites:
        variants = list(VARIANTS[suite])
        if args.variants == "base":
            variants = variants[:1]
        queries = SUITES[suite]["queries"]
        if args.queries:
            queries = [q for q in queries if q in args.queries.split(",")]
        for variant in variants:
            paths = tables(suite, variant, args.scale)
            for query in queries:
                plan.append((suite, variant, query, paths))

    def save():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=1, default=str))

    for round_number in range(args.rounds):
        for suite, variant, query, paths in plan:
            order = engines[round_number % len(engines):] + engines[
                : round_number % len(engines)
            ]
            for engine in order:
                outcome = run_worker(
                    command(engine, binaries, suite, query, args.reps, paths),
                    environment(args.threads),
                    args.timeout,
                    args.allow_busy,
                )
                outcome.update(
                    suite=suite,
                    variant=variant,
                    query=query,
                    engine=engine,
                    round=round_number,
                )
                result["runs"].append(outcome)
                print(
                    f"round {round_number} {suite}/{variant}/{query} {engine}: "
                    f"{outcome['status']}"
                    + (
                        f" {min(outcome['times']) / 1e6:.1f} ms"
                        if outcome.get("times")
                        else ""
                    ),
                    file=sys.stderr,
                )
            save()
    if args.trace:
        for suite, variant, query, paths in plan:
            outcome = run_worker(
                command("mojo", binaries, suite, query, 1, paths),
                environment(args.threads, trace=True),
                args.timeout,
                True,
            )
            result["trace"][f"{suite}/{variant}/{query}"] = outcome.get("paths", [])
        save()
    return result


# --- reporting -----------------------------------------------------------------


def _median_ms(runs):
    per_round = [min(r["times"]) / 1e6 for r in runs if r.get("times")]
    return statistics.median(per_round) if per_round else None


def evaluate(result):
    """Per (suite, variant, query, engine): status and median time, with
    answers checked against DuckDB (or Polars when DuckDB did not run)."""
    grouped = {}
    for run in result["runs"]:
        key = (run["suite"], run["variant"], run["query"], run["engine"])
        grouped.setdefault(key, []).append(run)
    cells = {}
    for (suite, variant, query, engine), runs in grouped.items():
        statuses = {r["status"] for r in runs}
        if "ok" not in statuses:
            first = runs[0]
            cells[(suite, variant, query, engine)] = {
                "status": first["status"],
                "reason": first.get("reason", ""),
            }
            continue
        cells[(suite, variant, query, engine)] = {
            "status": "ok",
            "ms": _median_ms(runs),
            "summary": next(r["summary"] for r in runs if r["status"] == "ok"),
        }
    for (suite, variant, query, engine), cell in cells.items():
        if cell["status"] != "ok":
            continue
        reference = None
        for other in ("duckdb", "polars"):
            ref = cells.get((suite, variant, query, other))
            if ref and ref["status"] == "ok":
                reference = ref
                break
        if reference is None or reference is cell:
            continue
        if not same_answer(suite, query, cell["summary"], reference["summary"]):
            cell["status"] = "wrong answer"
    return cells


def geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def report(result):
    cells = evaluate(result)
    info = result["provenance"]
    lines = [
        "# Benchmark suites",
        "",
        f"Revision `{info['revision'][:10]}`{' (dirty)' if info['dirty'] else ''}, "
        f"{info['mojo']}, Polars {info['polars']}, DuckDB {info['duckdb']}. "
        f"{info['cpu']}, {info['threads']} threads, scale `{info['scale']}`, "
        f"{info['rounds']} rounds of {info['reps']} timed runs.",
        "",
        "Times are medians over rounds of the fastest run, in milliseconds. "
        "Ratios are dataframe_mojo time divided by the other engine's; below "
        "1 is faster. Geometric means use only queries every engine answered "
        "correctly.",
        "",
    ]
    suites = []
    for suite in SUITES:
        if any(key[0] == suite for key in cells):
            suites.append(suite)
    for role, title, note in [
        (
            "dev",
            "Development suites",
            "Use these to find and tune optimizations.",
        ),
        (
            "heldout",
            "Held-out suites",
            "Report-only. Do not use per-query results here to choose what "
            "to optimize; a change that helps the development suites but not "
            "these probably does not generalize (docs/benchmarks.md).",
        ),
    ]:
        chosen = [s for s in suites if SUITES[s]["role"] == role]
        if not chosen:
            continue
        lines += [f"## {title}", "", note, ""]
        for suite in chosen:
            lines += _suite_table(suite, cells)
    if result.get("trace"):
        lines += _coverage(result["trace"])
    return "\n".join(lines) + "\n"


def _suite_table(suite, cells):
    variants = [v for v in VARIANTS[suite] if any(
        k[0] == suite and k[1] == v for k in cells
    )]
    base = variants[0]
    engines = [e for e in ENGINES if any(
        k[0] == suite and k[3] == e for k in cells
    )]
    others = [e for e in engines if e != "mojo"]
    lines = [f"### {suite} — {SUITES[suite]['source']}", ""]
    header = "| Query | " + " | ".join(f"{e} ms" for e in engines)
    header += "".join(f" | vs {e}" for e in others)
    if len(variants) > 1:
        header += "".join(f" | worst vs {e} (variant)" for e in others)
    header += " | Status |"
    lines.append(header)
    lines.append("|" + "---|" * (header.count("|") - 1))
    ratios = {e: [] for e in others}
    worst_all = {e: [] for e in others}
    counts = {"ok": 0, "unsupported": 0, "wrong answer": 0, "other": 0}
    for query in SUITES[suite]["queries"]:
        if not any(k[0] == suite and k[2] == query for k in cells):
            continue
        row = [query]
        status = []
        mojo = cells.get((suite, base, query, "mojo"))
        for engine in engines:
            cell = cells.get((suite, base, query, engine))
            row.append(
                f"{cell['ms']:.1f}" if cell and cell["status"] == "ok" else "—"
            )
            if cell and cell["status"] != "ok":
                label = cell["status"]
                if cell.get("reason"):
                    label += f": {cell['reason']}"
                status.append(f"{engine} {label}")
        for engine in others:
            other = cells.get((suite, base, query, engine))
            if mojo and other and mojo["status"] == other["status"] == "ok":
                ratio = mojo["ms"] / other["ms"]
                ratios[engine].append(ratio)
                row.append(f"{ratio:.2f}")
            else:
                row.append("—")
        if len(variants) > 1:
            for engine in others:
                worst = None
                for variant in variants:
                    m = cells.get((suite, variant, query, "mojo"))
                    o = cells.get((suite, variant, query, engine))
                    if m and o and m["status"] == o["status"] == "ok":
                        ratio = m["ms"] / o["ms"]
                        if worst is None or ratio > worst[0]:
                            worst = (ratio, variant)
                    elif m and m["status"] != "ok":
                        worst = (m["status"], variant)
                        break
                if worst and isinstance(worst[0], float):
                    worst_all[engine].append(worst[0])
                    row.append(f"{worst[0]:.2f} ({worst[1]})")
                elif worst:
                    row.append(f"{worst[0]} ({worst[1]})")
                else:
                    row.append("—")
        mojo_status = mojo["status"] if mojo else "not run"
        counts[mojo_status if mojo_status in counts else "other"] += 1
        row.append("; ".join(status) if status else "ok")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    summary = []
    for engine in others:
        g = geomean(ratios[engine])
        if g:
            summary.append(
                f"geometric mean vs {engine}: {g:.2f} over {len(ratios[engine])} queries"
            )
        w = geomean(worst_all[engine])
        if len(variants) > 1 and w:
            summary.append(f"worst-variant geometric mean vs {engine}: {w:.2f}")
    summary.append(
        f"dataframe_mojo answered {counts['ok']} correctly, "
        f"{counts['unsupported']} unsupported, {counts['wrong answer']} wrong, "
        f"{counts['other']} failed or timed out"
    )
    lines += ["; ".join(summary) + ".", ""]
    return lines


def instrumented_paths():
    found = set()
    for path in (ROOT / "dataframe").glob("*.mojo"):
        found.update(re.findall(r'"((?:join|group_by|filter|sort)\.[a-z_]+)"', path.read_text()))
    return sorted(found)


def _coverage(trace):
    hits = {}
    for query, paths in trace.items():
        for path in paths:
            hits.setdefault(path, []).append(query)
    lines = [
        "## Fast-path coverage",
        "",
        "Specialized paths each query took (DATAFRAME_TRACE_PATHS). A path "
        "that only one query exercises is a candidate for a benchmark-shaped "
        "fast path; check that its trigger is a data property with real "
        "examples (docs/benchmarks.md, rule 5).",
        "",
        "| Path | Queries | Examples |",
        "|---|---:|---|",
    ]
    for path in sorted(hits, key=lambda p: (len(hits[p]), p)):
        flag = " **(one query)**" if len(hits[path]) == 1 else ""
        examples = ", ".join(sorted(hits[path])[:4])
        lines.append(f"| `{path}`{flag} | {len(hits[path])} | {examples} |")
    never = [p for p in instrumented_paths() if p not in hits]
    lines.append("")
    if never:
        lines.append(
            "Instrumented paths no query reached: "
            + ", ".join(f"`{p}`" for p in never)
            + "."
        )
        lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suites", default="h2o_groupby,h2o_join")
    parser.add_argument(
        "--heldout",
        action="store_true",
        help="also run the held-out suites (pdsh, clickbench)",
    )
    parser.add_argument("--engines", default="mojo,polars,duckdb")
    parser.add_argument("--queries", default="", help="comma-separated subset")
    parser.add_argument("--scale", choices=list(SCALES), default="default")
    parser.add_argument(
        "--variants",
        choices=["all", "base"],
        default="all",
        help="'base' skips the perturbed data variants (not for reporting)",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "suites" / "results.json",
    )
    parser.add_argument(
        "--report-from", type=Path, help="render a saved results file"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if any dataframe_mojo answer is wrong or fails",
    )
    args = parser.parse_args()
    if args.heldout:
        args.suites += ",pdsh,clickbench"
    if args.report_from:
        result = json.loads(args.report_from.read_text())
    else:
        result = measure(args)
    text = report(result)
    print(text)
    if args.output and not args.report_from:
        args.output.with_suffix(".md").write_text(text)
    if args.check:
        bad = [
            key
            for key, cell in evaluate(result).items()
            if key[3] == "mojo" and cell["status"] in ("wrong answer", "failed", "timeout")
        ]
        if bad:
            print("dataframe_mojo failures:", bad, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
