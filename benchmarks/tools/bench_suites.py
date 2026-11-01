"""Run the external benchmark suites against Polars and DuckDB.

See docs/design/benchmarks.md for the rules these measurements serve. Run under the
oracle environment, after building libdfparquet (pixi run -e native
build-dfparquet):

    # A code change, in minutes: development suites (H2O at 1M rows, PDS-H
    # and TPC-DS at scale factor 0.1), this build against main, Polars/DuckDB
    # from the reference cache.
    pixi run -e oracle python3 benchmarks/tools/bench_suites.py --baseline main

    # A report: 10M rows, three rounds, fast-path coverage, development and
    # held-out suites (ClickBench).
    pixi run -e oracle python3 benchmarks/tools/bench_suites.py --tier full --all-suites

One worker process per (suite, variant, engine, round) loads the tables once
(untimed), then warms up and times --reps runs of each query; the per-round
figure is the fastest run, the report shows the median over rounds, and the
order of the engines under test rotates. Every answer is checked against
independent DuckDB full relations (see suite_answers.py); a wrong answer is reported as such and
excluded from ratios. Unsupported queries are counted, not dropped. Data
lives outside the repository (benchmarks/suites/datagen.py).
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
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SUITES_DIR = ROOT / "benchmarks" / "suites"
sys.path.insert(0, str(SUITES_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_html  # noqa: E402
import bench_policy  # noqa: E402
import datagen  # noqa: E402
from bench_host import compiler_processes  # noqa: E402
import suite_answers  # noqa: E402
import suite_schema  # noqa: E402

ENGINES = ("mojo", "polars", "duckdb")

# Keep tuning and validation separate. PDS-H was held out until 2026-10-05,
# when it became a development suite and TPC-DS took its place: nothing in
# development covered multi-join plans, the one area behind Polars. TPC-DS
# was held out until 2026-10-15, when it became a development suite too:
# every development query was within 1.5x of Polars and DuckDB but one, and
# most of the queries still outside were TPC-DS. ClickBench stays held out.
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
        "role": "dev",
        "runner": "pdsh",
        "queries": [f"q{i}" for i in range(1, 23)],
        "source": "PDS-H (pola-rs/polars-benchmark), TPC-H derived",
    },
    "tpcds": {
        "role": "dev",
        "runner": "tpcds",
        "queries": [f"q{i}" for i in range(1, 100)],
        "source": "TPC-DS (DuckDB tpcds extension: dsdgen, tpcds_queries)",
    },
    "clickbench": {
        "role": "heldout",
        "runner": "clickbench",
        "queries": [f"q{i}" for i in range(0, 43)],
        "source": "ClickBench (ClickHouse/ClickBench) hits",
    },
}

# Data sizes. "smoke" checks answers in CI; "dev" is the quick tier's size
# for the edit-measure loop; "default" is for reports.
SCALES = {
    "smoke": {
        "h2o_rows": 100_000,
        "pdsh_sf": 0.01,
        "tpcds_sf": 0.01,
        "clickbench_partitions": 1,
    },
    "dev": {
        "h2o_rows": 1_000_000,
        "pdsh_sf": 0.1,
        "tpcds_sf": 0.1,
        "clickbench_partitions": 1,
    },
    "default": {
        "h2o_rows": 10_000_000,
        "pdsh_sf": 1,
        "tpcds_sf": 1,
        "clickbench_partitions": 10,
    },
    "large": {
        "h2o_rows": 100_000_000,
        "pdsh_sf": 10,
        "tpcds_sf": 10,
        "clickbench_partitions": 100,
    },
}

# Two tiers (docs/design/benchmarks.md). `quick` is for checking a code change in
# minutes: development suites at the "dev" scale (H2O at 1M rows, PDS-H and
# TPC-DS at scale factor 0.1), every data variant, three rounds, and
# Polars/DuckDB outcomes reused from the reference cache, since their code
# does not change when this library does. `full` is for reports.
TIERS = {
    "quick": {
        "suites": ",".join(n for n, s in SUITES.items() if s["role"] == "dev"),
        "scale": "dev",
        "rounds": 3,
        "reps": 3,
        "reuse_references": True,
        "trace": False,
    },
    "full": {
        "suites": ",".join(SUITES),
        "scale": "default",
        "rounds": 3,
        "reps": 3,
        "reuse_references": False,
        "trace": True,
    },
}

# Data variants that every measurement runs; the report shows each query's
# worst variant beside its base result (docs/design/benchmarks.md, rule 4). The
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
    "pdsh": {"base": {}, "decimal": {"decimal": True}},
    "tpcds": {"base": {}, "decimal": {"decimal": True}},
    "clickbench": {"base": {}},
}

# Historical lossy-summary policy, retained only to reproduce its defects.
# Complete-result evaluation never uses these column omissions.
LEGACY_CHECKS = {
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


# The same, for one data variant only: (suite, variant, query).
LEGACY_VARIANT_CHECKS = {
    # Float sums of the same rows in different rollup sets can differ by
    # an ulp, which splits q67's rank ties on more threads; the complete
    # check holds rk to the ties tolerance allows (`tolerant_ranks`). The
    # decimal variant is exact, so it still compares rk.
    ("tpcds", "base", "q67"): [
        "i_category",
        "i_class",
        "i_brand",
        "i_product_name",
        "d_year",
        "d_qoy",
        "d_moy",
        "s_store_id",
        "sumsales",
    ],
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
        root = datagen.pdsh(size["pdsh_sf"], spec.get("decimal", False))
        return {name: root / f"{name}.parquet" for name in datagen.PDSH_TABLES}
    if suite == "tpcds":
        root = datagen.tpcds(size["tpcds_sf"], spec.get("decimal", False))
        return {name: root / f"{name}.parquet" for name in datagen.TPCDS_TABLES}
    return {"hits": datagen.clickbench(size["clickbench_partitions"])}


# --- building and running workers ------------------------------------------


def _build(tree, name, binary):
    """Build suite runner `name` against the dataframe package in `tree`,
    always from this checkout's suite sources."""
    print(f"building {binary}", file=sys.stderr)
    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "mojo",
            "build",
            "-O3",
            "-I",
            str(tree),
            "-I",
            str(SUITES_DIR),
            str(SUITES_DIR / f"{name}.mojo"),
            "-o",
            str(binary),
        ],
        cwd=tree,
    )


def build_runners(names, force=False):
    """Runners for this checkout, rebuilt when a library or suite source is
    newer than the binary."""
    out = ROOT / "build" / "suites"
    sources = list((ROOT / "dataframe").rglob("*.mojo")) + list(
        SUITES_DIR.glob("*.mojo")
    )
    newest = max(path.stat().st_mtime for path in sources)
    binaries = {}
    for name in sorted(set(names)):
        binary = out / f"suite_{name}"
        if force or not binary.exists() or binary.stat().st_mtime < newest:
            _build(ROOT, name, binary)
        binaries[name] = binary
    return binaries


def build_baseline(ref, names):
    """Runners built against the library at `ref`, cached by commit.

    The revision is checked out once as a detached git worktree under
    build/suites/baseline/<sha>; its binaries are rebuilt only when this
    checkout's suite sources change. Returns (label, binaries).
    """
    sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    tree = ROOT / "build" / "suites" / "baseline" / sha
    if not (tree / "dataframe").exists():
        subprocess.check_call(
            ["git", "worktree", "add", "--detach", str(tree), sha], cwd=ROOT
        )
    newest = max(path.stat().st_mtime for path in SUITES_DIR.glob("*.mojo"))
    binaries = {}
    for name in sorted(set(names)):
        binary = tree / "build" / "suites" / f"suite_{name}"
        if not binary.exists() or binary.stat().st_mtime < newest:
            _build(tree, name, binary)
        binaries[name] = binary
    return f"base@{sha[:7]}", binaries


def command(engine, runners, suite, queries, reps, paths):
    args = [suite, ",".join(queries), str(reps)] + [
        f"{k}={v}" for k, v in paths.items()
    ]
    if engine in runners:
        return [str(runners[engine][SUITES[suite]["runner"]])] + args
    return [sys.executable, str(SUITES_DIR / "engines.py"), engine] + args


def environment(threads, trace=False):
    env = dict(os.environ)
    env["DATAFRAME_THREADS"] = str(threads)
    env["POLARS_MAX_THREADS"] = str(threads)
    env["BENCH_THREADS"] = str(threads)
    if trace:
        env["DATAFRAME_TRACE_PATHS"] = "1"
        env["DATAFRAME_EXECUTION_REPORT"] = "1"
    else:
        env.pop("DATAFRAME_TRACE_PATHS", None)
        env.pop("DATAFRAME_EXECUTION_REPORT", None)
    return env


def parse(output, queries, answer_directory=None):
    """Per-query outcomes from a worker's tagged output lines."""
    found = {q: {"status": "failed", "reason": "no output"} for q in queries}
    for line in output.splitlines():
        fields = line.split("\t")
        if len(fields) < 3 or fields[1] not in found:
            continue
        kind, query = fields[0], fields[1]
        entry = found[query]
        if kind == "time":
            entry.setdefault("times", []).append(int(fields[2]))
        elif kind in ("answer", "oracle"):
            try:
                if answer_directory is None:
                    raise ValueError(
                        "answer directory was not supplied by driver"
                    )
                directory = (
                    Path(answer_directory) / "oracle" if kind
                    == "oracle" else answer_directory
                )
                receipt = suite_answers.pin_export(
                    json.loads(fields[2]), directory, query
                )
                answers = entry.setdefault(
                    "oracles" if kind == "oracle" else "answers", []
                )
                if any(
                    a["repetition"] == receipt["repetition"] for a in answers
                ):
                    raise ValueError("duplicate answer repetition")
                answers.append(receipt)
            except (OSError, ValueError, TypeError, KeyError) as error:
                entry["answer_error"] = str(error)
        elif kind == "oracle_error":
            entry["oracle_error"] = fields[2]
        elif kind == "summary":
            values = [float(v) for v in fields[3].split(",")] if fields[
                3
            ] else []
            names = (
                fields[4].split(",") if len(fields) > 4 and fields[4] else []
            )
            summary = {
                "height": int(fields[2]),
                "values": values,
                "names": names,
            }
            if len(fields) > 5:
                summary["pairs"] = [
                    float(v) for v in fields[5].split(",")
                ] if fields[5] else []
            entry.update(status="ok", reason="", summary=summary)
        elif kind in ("unsupported", "failed"):
            entry.update(status=kind, reason=fields[2])
    return found


def trace_paths(stderr):
    """Map each query to the specialized paths printed while it ran."""
    paths, current = {}, None
    for line in stderr.splitlines():
        if line.startswith("dataframe-query: "):
            current = line.split(": ", 1)[1].strip()
            paths.setdefault(current, set())
        elif line.startswith("dataframe-path: ") and current:
            paths[current].add(line.split(": ", 1)[1].strip())
    return {query: sorted(found) for query, found in paths.items()}


OPERATOR_COLUMNS = (
    "node operator executor algorithm build_side input_rows build_rows "
    "output_rows builds executions busy_ms wall_ms"
).split()


def operator_reports(stderr):
    """Map each query to the rows of its lazy execution report
    (`LazyFrame.profile`, printed by DATAFRAME_EXECUTION_REPORT): one dict
    per executed plan node, from the query's last run."""
    reports, current = {}, None
    for line in stderr.splitlines():
        if line.startswith("dataframe-query: "):
            current = line.split(": ", 1)[1].strip()
            reports[current] = []
        elif line.startswith("dataframe-operator:\t") and current:
            cells = line.split("\t")[1:]
            # Reports now lead with the cost table's calibration tag
            # (`development-source-no-calibration`); a baseline built
            # before it has none.
            names = OPERATOR_COLUMNS
            if cells and not cells[0].lstrip("-").isdigit():
                names = ["calibration"] + OPERATOR_COLUMNS
            row = dict(zip(names, cells))
            for name in OPERATOR_COLUMNS:
                if name not in row:
                    # A baseline built before the column existed.
                    continue
                if name.endswith("rows") or name in ("node", "builds", "executions"):
                    row[name] = int(row[name])
                elif name.endswith("_ms"):
                    row[name] = float(row[name])
            if reports[current] and row["node"] <= reports[current][-1]["node"]:
                # Rows come in node order; a lower node starts a new run.
                reports[current] = []
            reports[current].append(row)
    return reports


def _wait_quiet(limit=3600):
    """Block until no compiler runs; other work on a shared host must not
    overlap timing (docs/design/benchmarks.md)."""
    waited = 0
    while compiler_processes():
        if waited >= limit:
            raise RuntimeError("compiler activity persisted for an hour")
        if waited == 0:
            print("waiting for compiler activity to stop", file=sys.stderr)
        time.sleep(10)
        waited += 10


LOAD = []  # (load average, threads) sampled before each timed worker


def run_worker(cmd, env, timeout, allow_busy, queries):
    """Run one worker over `queries`; returns (per-query outcomes, stderr)."""
    if hasattr(os, "getloadavg"):
        LOAD.append(os.getloadavg()[0])
    # Retained files are evidence used by caches and --report-from. Never
    # put generated answers in the checkout or delete them before evaluation.
    answer_directory = tempfile.mkdtemp(prefix="dataframe-suite-answers-")
    env = dict(env, DATAFRAME_SUITE_ANSWER_DIR=answer_directory)
    for _ in range(3):
        if not allow_busy:
            _wait_quiet()
        try:
            proc = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as expired:
            out = expired.stdout or ""
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            found = parse(out, queries, answer_directory)
            for entry in found.values():
                if (
                    entry["status"] == "failed"
                    and entry["reason"] == "no output"
                ):
                    entry.update(status="timeout", reason="")
            return found, ""
        if allow_busy or not compiler_processes():
            break
        print(
            "compiler activity overlapped a run; repeating it", file=sys.stderr
        )
    else:
        raise RuntimeError("compiler activity kept overlapping timed runs")
    found = parse(proc.stdout, queries, answer_directory)
    stderr = proc.stderr
    if proc.returncode != 0:
        tail = " | ".join((proc.stderr or "").strip().splitlines()[-2:])
        silent = [
            query
            for query in queries
            if found[query]["status"] == "failed"
            and found[query]["reason"] == "no output"
        ]
        # The worker died in the first query it reported nothing for. That
        # query alone failed: the ones after it never ran, so run them in a
        # new worker instead of counting one crash as many failures.
        if silent:
            found[silent[0]][
                "reason"
            ] = f"worker exited {proc.returncode}: {tail}"
        rest = silent[1:]
        joined = ",".join(queries)
        if rest and joined in cmd:
            again = list(cmd)
            again[again.index(joined)] = ",".join(rest)
            later, more = run_worker(again, env, timeout, allow_busy, rest)
            found.update(later)
            stderr = (stderr or "") + (more or "")
    return found, stderr


# --- answer checks ----------------------------------------------------------


def _close(a, b):
    """Whether two summary numbers agree: both NaN, the same infinity, or
    finite and within a relative 1e-6. A finite number never matches an
    infinity, nor one infinity the other (#606)."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
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


def legacy_summary_agrees(suite, query, got, want, variant=None):
    """Diagnostic legacy behavior only; never use to admit benchmark timing."""
    if got["height"] != want["height"]:
        return False
    spec = LEGACY_VARIANT_CHECKS.get(
        (suite, variant, query)
    ) or LEGACY_CHECKS.get((suite, query))
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
        pairs = [
            (got["values"][i], want["values"][j]) for i, j in zip(mine, theirs)
        ]
    if not all(_close(a, b) for a, b in pairs):
        return False
    # Rows kept together (#606): the neighbouring-column product sums, when
    # every column is compared and both sides carry them (a reference cached
    # before they existed does not).
    if spec is None and "pairs" in got and "pairs" in want:
        if len(got["pairs"]) != len(want["pairs"]):
            return False
        return all(_close(a, b) for a, b in zip(got["pairs"], want["pairs"]))
    return True


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
    # The CPU frequency governor: a parked thread wakes on a core that
    # `schedutil` has clocked down, so short queries ramp from a lower
    # clock and the small cells move between runs (#527).
    governor = "unknown"
    scaling = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if scaling.exists():
        try:
            governor = scaling.read_text().strip()
        except OSError:
            pass
    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "revision": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain", "dataframe"])),
        "mojo": run(["mojo", "--version"]),
        "polars": polars.__version__,
        "duckdb": duckdb.__version__,
        "platform": platform.platform(),
        "cpu": cpu,
        "governor": governor,
        "tier": args.tier,
        "mode": "correctness" if getattr(args, "correctness_only", False) else "performance",
        "baseline": args.baseline or "",
        "threads": args.threads,
        "scale": args.scale,
        "rounds": args.rounds,
        "reps": args.reps,
        "data_root": str(datagen.data_root()),
        "benchmark_dirty": bool(
            run(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "benchmarks/suites",
                    "benchmarks/tools/bench_suites.py",
                    "benchmarks/tools/bench_html.py",
                    "benchmarks/tools/bench_policy.py",
                ]
            )
        ),
        "evaluation": bench_policy.record(ROOT, args.suites.split(","), SUITES),
    }


# --- reference cache -----------------------------------------------------------


def _cache_file(suite, variant, engine):
    return (
        datagen.data_root()
        / "reference-cache"
        / suite
        / variant
        / f"{engine}.json"
    )


def _cache_key(engine, paths, args):
    import duckdb
    import polars

    version = polars.__version__ if engine == "polars" else duckdb.__version__
    # Cached reference answers/timings must match the query implementation.
    import hashlib

    implementation_hash = hashlib.sha256()
    for source in [
        SUITES_DIR / "engines.py",
        SUITES_DIR / "oracle_sql.py",
        Path(__file__),
        Path(suite_answers.__file__),
        Path(suite_schema.__file__),
        Path(__file__).with_name("suite_numerical_policy.json"),
        Path(__file__).with_name("metal_numerical_compare.py"),
        Path(__file__).with_name("metal_numerical_audit.py"),
    ]:
        implementation_hash.update(source.read_bytes())
    implementation = implementation_hash.hexdigest()
    files = sorted(
        f"{name}:{suite_answers.sha256(p)}" for name, p in paths.items()
    )
    return (
        f"{engine} {version} source={implementation} threads={args.threads} reps={args.reps} rounds={getattr(args, 'rounds', 1)} "
        + " ".join(files)
    )


def load_reference(suite, variant, engine, paths, args):
    """Cached Polars/DuckDB outcomes for this data, engine version and thread
    count, or None. Their code does not change when this library does."""
    path = _cache_file(suite, variant, engine)
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    if cached.get("key") != _cache_key(engine, paths, args):
        return None
    # An evicted/corrupt snapshot is a cache miss, not reusable correctness
    # evidence. Rebuild it rather than publishing unverified cached timings.
    try:
        for run in cached["runs"]:
            if run["status"] != "ok":
                continue
            if not run.get("answers") or (
                engine == "duckdb" and len(run.get("oracles", [])) != 1
            ):
                return None
            for record in run.get("answers", []) + run.get("oracles", []):
                suite_answers.verify_answer(record)
    except (OSError, ValueError, TypeError, KeyError, ImportError):
        return None
    return cached


def save_reference(suite, variant, engine, paths, args, runs):
    path = _cache_file(suite, variant, engine)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "key": _cache_key(engine, paths, args),
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "host": platform.node(),
                "runs": runs,
            },
            indent=1,
        )
    )


# --- measurement -------------------------------------------------------------


def measure(args):
    suites = [s for s in args.suites.split(",") if s]
    engines = [e for e in args.engines.split(",") if e]
    runners = {}
    names = [SUITES[s]["runner"] for s in suites]
    if "mojo" in engines or args.trace:
        runners["mojo"] = build_runners(names, force=args.rebuild)
    if args.baseline:
        label, binaries = build_baseline(args.baseline, names)
        runners[label] = binaries
        if label not in engines:
            engines.insert(1, label)
    measured = [e for e in engines if e in runners or not args.reuse_references]
    references = [e for e in engines if e not in measured]
    result = {
        "provenance": provenance(args),
        "engines": engines,
        "baseline": next((e for e in engines if e.startswith("base@")), None),
        "runs": [],
        "trace": {},
        "operators": {},
    }
    plan = []
    for suite in suites:
        variants = list(VARIANTS[suite])
        if args.variants == "base":
            variants = variants[:1]
        queries = SUITES[suite]["queries"]
        if args.queries:
            queries = [q for q in queries if q in args.queries.split(",")]
        for variant in variants:
            plan.append(
                (suite, variant, queries, tables(suite, variant, args.scale))
            )

    result["query_plan"] = [
        {
            "suite": suite,
            "variant": variant,
            "queries": queries,
            "inputs": {
                name: {"path": str(path), "sha256": suite_answers.sha256(path)}
                for name, path in paths.items()
            },
        }
        for suite, variant, queries, paths in plan
    ]

    def save():
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=1, default=str))

    def record(outcomes, suite, variant, engine, round_number, cached=False):
        runs = []
        for query, outcome in outcomes.items():
            outcome = dict(outcome)
            outcome.update(
                suite=suite,
                variant=variant,
                query=query,
                engine=engine,
                round=round_number,
                cached=cached,
            )
            result["runs"].append(outcome)
            runs.append(outcome)
        return runs

    # Reference engines: reuse cached outcomes, or measure and cache them.
    for suite, variant, queries, paths in plan:
        for engine in references:
            cached = load_reference(suite, variant, engine, paths, args)
            done = {r["query"] for r in cached["runs"]} if cached else set()
            if cached and set(queries) <= done:
                for run in cached["runs"]:
                    if run["query"] in queries:
                        run = dict(run, cached=True)
                        result["runs"].append(run)
                print(f"{suite}/{variant} {engine}: cached", file=sys.stderr)
                continue
            runs = []
            for round_number in range(args.rounds):
                outcomes, _ = run_worker(
                    command(engine, runners, suite, queries, args.reps, paths),
                    environment(args.threads),
                    args.timeout * len(queries),
                    args.allow_busy,
                    queries,
                )
                runs += record(outcomes, suite, variant, engine, round_number)
            save_reference(suite, variant, engine, paths, args, runs)
            print(
                f"{suite}/{variant} {engine}: measured and cached",
                file=sys.stderr,
            )
    save()

    # Engines under test, alternating order so no build always runs first.
    for round_number in range(args.rounds):
        for step, (suite, variant, queries, paths) in enumerate(plan):
            shift = (round_number + step) % max(1, len(measured))
            for engine in measured[shift:] + measured[:shift]:
                outcomes, _ = run_worker(
                    command(engine, runners, suite, queries, args.reps, paths),
                    environment(args.threads),
                    args.timeout * len(queries),
                    args.allow_busy,
                    queries,
                )
                record(outcomes, suite, variant, engine, round_number)
                ok = sum(1 for o in outcomes.values() if o["status"] == "ok")
                print(
                    f"round {round_number} {suite}/{variant} {engine}: "
                    f"{ok}/{len(queries)} ok",
                    file=sys.stderr,
                )
            save()
    result["load"] = LOAD
    save()
    if args.trace:
        for suite, variant, queries, paths in plan:
            _, stderr = run_worker(
                command("mojo", runners, suite, queries, 0, paths),
                environment(args.threads, trace=True),
                args.timeout * len(queries),
                True,
                queries,
            )
            for query, paths_hit in trace_paths(stderr).items():
                result["trace"][f"{suite}/{variant}/{query}"] = paths_hit
            for query, rows in operator_reports(stderr).items():
                result["operators"][f"{suite}/{variant}/{query}"] = rows
        save()
    return result


# --- reporting -----------------------------------------------------------------


def _median_ms(runs):
    per_round = [min(r["times"]) / 1e6 for r in runs if r.get("times")]
    return statistics.median(per_round) if per_round else None


def evaluate(result):
    """Validate every repetition and round against independent complete answers.

    A summary-only historical run remains renderable, but cannot contribute
    a verified timing. Failures in any round remain failures.

    Answers are compared as columns, one at a time. Cells are visited query
    by query so each oracle is prepared once for every engine, and an answer
    whose hash-pinned bytes were already compared with the same oracle reuses
    that verdict (DuckDB's own answers are checked for every engine, and
    repetitions often write identical files).
    """
    grouped = {}
    for run in result["runs"]:
        key = (run["suite"], run["variant"], run["query"], run["engine"])
        grouped.setdefault(key, []).append(run)
    def answers(run):
        if run.get("answer_error"):
            raise ValueError(run["answer_error"])
        records = run.get("answers", [])
        if any(record.get("query") != run["query"] for record in records):
            raise ValueError("answer query identity differs from run")
        requested_reps = result.get("provenance", {}).get("reps")
        if requested_reps is not None and len(run.get("times", [])) != requested_reps:
            raise ValueError("missing timed repetitions")
        expected_count = max(1, requested_reps if requested_reps is not None else len(run.get("times", [])))
        if len(records) != expected_count or sorted(a["repetition"] for a in records) != list(range(expected_count)):
            raise ValueError("missing or duplicate complete answer repetitions")
        yield from records

    def compare(record, suite, query, oracle_key, check):
        # The verdict depends only on the Parquet bytes, which the receipt's
        # hash names once a check confirms it: a reused answer still gets
        # every hash and sidecar check, a new one is loaded (and checked).
        key = (oracle_key, record.get("path_sha256"))
        if key in verdicts:
            suite_answers.verify_answer(record)
        else:
            answer = suite_schema.normalize(
                suite_answers.load_answer(record), suite, query
            )
            verdicts[key] = check.compare(answer)
        return verdicts[key]

    cells = dict.fromkeys(grouped)
    verdicts = {}
    prepared = (None, None)  # (oracle key, OracleCheck): one oracle at a time
    try:
        for key in sorted(grouped, key=lambda k: k[:3]):
            suite, variant, query, engine = key
            runs = grouped[key]
            cell = cells[key] = {"status": "ok", "reason": ""}
            bad = next((r for r in runs if r["status"] != "ok"), None)
            if bad is not None:
                cell.update(status=bad["status"], reason=bad.get("reason", ""))
                continue
            expected_rounds = result.get("provenance", {}).get("rounds")
            if expected_rounds is not None and (
                len(runs) != expected_rounds
                or {r.get("round", 0) for r in runs} != set(range(expected_rounds))
            ):
                cell.update(
                    status="unverified",
                    reason="missing or duplicate measurement rounds",
                )
                continue
            reference_runs = grouped.get((suite, variant, query, "duckdb"), [])
            if not reference_runs or any(
                r["status"] != "ok" for r in reference_runs
            ):
                cell.update(
                    status="unverified",
                    reason="missing successful independent DuckDB reference",
                )
                continue
            try:
                records = reference_runs[0].get("oracles", [])
                if len(records) != 1 or reference_runs[0].get("oracle_error"):
                    cell.update(
                        status="unverified",
                        reason=reference_runs[0].get(
                            "oracle_error", "missing independent full relation"
                        ),
                    )
                    continue
                if records[0].get("query") != query:
                    raise ValueError("oracle query identity differs from run")
                oracle_key = (
                    suite,
                    query,
                    records[0].get("path_sha256"),
                    records[0].get("schema_path_sha256"),
                )
                if prepared[0] == oracle_key:
                    suite_answers.verify_answer(records[0])
                else:
                    if prepared[1] is not None:
                        prepared[1].close()
                    prepared = (None, None)  # Release the preceding oracle.
                    oracle = suite_schema.normalize(
                        suite_answers.load_answer(records[0]), suite, query
                    )
                    tolerant = suite_answers.tolerant_columns(
                        suite, query, oracle
                    )
                    ranks = suite_answers.tolerant_ranks(
                        suite, query, oracle, tolerant
                    )
                    prepared = (
                        oracle_key,
                        suite_answers.OracleCheck(oracle, tolerant, ranks),
                    )
                    oracle = None
                    verdicts.clear()
                check = prepared[1]
                checked_runs = runs if engine == "duckdb" else runs + reference_runs
                for record in (r for run in checked_runs for r in answers(run)):
                    comparison = compare(
                        record, suite, query, (oracle_key, tuple(check.tolerant)), check
                    )
                    if comparison["status"] != "equal":
                        cell.update(
                            status="wrong answer" if comparison["status"]
                            == "different" else "unverified",
                            reason=comparison["reason"],
                        )
                        break
            except (OSError, ValueError, TypeError, KeyError, ImportError) as error:
                cell.update(status="unverified", reason=str(error))
            if cell["status"] == "ok":
                cell.update(
                    ms=_median_ms(runs),
                    rounds=[min(r["times"]) / 1e6 for r in runs if r.get("times")],
                    summary=runs[0].get("summary", {}),
                )
    finally:
        if prepared[1] is not None:
            prepared[1].close()
    return cells


def geomean(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def check_correctness(result):
    """Require every planned Mojo outcome and a reference for each answer."""
    cells = evaluate(result)
    failures = []
    for plan in result["query_plan"]:
        for query in plan["queries"]:
            key = (plan["suite"], plan["variant"], query, "mojo")
            cell = cells.get(key, {"status": "missing"})
            status = cell["status"]
            if status == "ok":
                if not any(
                    cells.get((*key[:3], engine), {}).get("status") == "ok"
                    for engine in ("duckdb", "polars")
                ):
                    status = "missing successful reference"
            print("/".join(key[:3]) + ": " + status)
            # Keep explicitly unsupported queries visible, as in --check.
            if status not in ("ok", "unsupported"):
                failures.append((key, status))
    return failures


def report(result, cells=None):
    if cells is None:
        cells = evaluate(result)
    info = result["provenance"]
    engines = result.get("engines", list(ENGINES))
    lines = [
        "# Benchmark suites",
        "",
        f"Tier `{info.get('tier', 'full')}`. Revision `{info['revision'][:10]}`"
        f"{' (dirty)' if info['dirty'] else ''}, "
        f"{info['mojo']}, Polars {info['polars']}, DuckDB {info['duckdb']}. "
        f"{info['cpu']} ({info.get('governor', 'unknown')} governor), "
        f"{info['threads']} threads, scale `{info['scale']}`, "
        f"{info['rounds']} rounds of {info['reps']} timed runs.",
        "",
        "Times are medians over rounds of the fastest run, in milliseconds. "
        "Ratios are dataframe_mojo time divided by the other engine's; below "
        "1 is faster. Geometric means use only queries every engine answered "
        "correctly.",
        "",
    ]
    lines += ["**" + bench_policy.report_note(result) + "**", ""]
    evaluation = info.get("evaluation")
    if evaluation:
        lines += [
            "Local benchmark manifest: `"
            + evaluation["manifest_sha256"]
            + "`.",
            "",
        ]
    if result.get("baseline"):
        lines += [
            f"`{result['baseline']}` is this library built at the baseline "
            "revision; `vs "
            + result["baseline"]
            + "` above 1 means the change "
            "is slower there.",
            "",
        ]
    load = result.get("load") or []
    threads = info["threads"]
    if load and max(load) > threads + 4:
        lines += [
            f"**Busy host:** the one-minute load average reached "
            f"{max(load):.0f} (median {statistics.median(load):.0f}) against "
            f"{threads} benchmark threads. Other work competed for the CPU, so "
            "treat small differences as noise and rerun on a quiet machine.",
            "",
        ]
    if result.get("baseline"):
        lines += _changes(result["baseline"], cells)
    cached = sorted({r["engine"] for r in result["runs"] if r.get("cached")})
    if cached:
        lines += [
            "Times for " + ", ".join(cached) + " come from the reference cache "
            "(same data, engine version and thread count; possibly measured on "
            "an earlier day, so compare their ratios with care).",
            "",
        ]
    for role, heading, guidance in (
        (
            "dev",
            "Development suites",
            "Use these results to diagnose and tune the engine.",
        ),
        (
            "heldout",
            "Held-out suites",
            "Report-only validation of the completed change. Do not use per-query timings to choose optimizations or tune thresholds.",
        ),
    ):
        suites = [
            suite
            for suite in SUITES
            if SUITES[suite]["role"] == role
            and any(key[0] == suite for key in cells)
        ]
        if suites:
            lines += ["## " + heading, "", guidance, ""]
            for suite in suites:
                lines += _suite_table(suite, cells, engines)
    if result.get("trace"):
        lines += _coverage(result["trace"])
    if result.get("operators"):
        lines += _join_builds(result["operators"])
    return "\n".join(lines) + "\n"


def _join_builds(operators):
    """Joins that indexed more rows than probed them, from the lazy
    execution reports (`LazyFrame.profile`): where the executor hashed the
    larger side. Observed counts, so a cost model can be judged on them."""
    joins, larger = 0, []
    for query, rows in sorted(operators.items()):
        for row in rows:
            if not row["operator"].startswith("JOIN") or row["builds"] == 0:
                continue
            joins += 1
            if row["build_rows"] > row["input_rows"] > 0:
                larger.append((row["build_rows"] - row["input_rows"], query, row))
    lines = [
        "## Join builds larger than their probe",
        "",
        f"Of {joins} joins that built an index (DATAFRAME_EXECUTION_REPORT), "
        f"{len(larger)} indexed more rows than probed them. Observed counts "
        "from the lazy execution report; the largest first.",
        "",
        "| Query | Join | Executor | Index | Probe rows | Build rows | Output rows |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for _, query, row in sorted(larger, key=lambda item: -item[0])[:20]:
        lines.append(
            f"| {query} | {row['operator'][:60]} | {row['executor']} | "
            f"{row['algorithm']} | {row['input_rows']:,} | "
            f"{row['build_rows']:,} | {row['output_rows']:,} |"
        )
    lines.append("")
    return lines


def _separated(mine, theirs):
    """+1 when every round of `mine` is slower than every round of `theirs`,
    -1 when every round is faster, 0 when the rounds overlap."""
    if len(mine) < 2 or len(theirs) < 2:
        return 0
    if min(mine) > max(theirs):
        return 1
    if max(mine) < min(theirs):
        return -1
    return 0


def _changes(baseline, cells):
    """The cells where this build and the baseline differ beyond the spread
    of their rounds; the first thing to read after a code change."""
    slower, faster = [], []
    for (suite, variant, query, engine), cell in sorted(cells.items()):
        if engine != "mojo" or cell["status"] != "ok":
            continue
        base = cells.get((suite, variant, query, baseline))
        if not base or base["status"] != "ok":
            continue
        side = _separated(cell["rounds"], base["rounds"])
        ratio = cell["ms"] / base["ms"]
        entry = f"{suite}/{variant}/{query} {ratio:.2f}x ({base['ms']:.1f} → {cell['ms']:.1f} ms)"
        if side > 0 and ratio > 1.03:
            slower.append(entry)
        elif side < 0 and ratio < 0.97:
            faster.append(entry)
    lines = [f"## Changes vs {baseline}", ""]
    lines.append(
        "Only cells where every round of one build beat every round of the "
        "other, by more than 3%, are listed."
    )
    lines.append("")
    lines.append(
        f"- **Slower ({len(slower)}):** " + ("; ".join(slower) or "none")
    )
    lines.append(
        f"- **Faster ({len(faster)}):** " + ("; ".join(faster) or "none")
    )
    lines.append("")
    return lines


def _suite_table(suite, cells, all_engines):
    variants = [
        v
        for v in VARIANTS[suite]
        if any(k[0] == suite and k[1] == v for k in cells)
    ]
    base = variants[0]
    engines = [
        e
        for e in all_engines
        if any(k[0] == suite and k[3] == e for k in cells)
    ]
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
    for path in (ROOT / "dataframe").rglob("*.mojo"):
        found.update(re.findall(r'trace_path\(\s*"([^"]+)"', path.read_text()))
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
        "examples (docs/design/benchmarks.md, rule 5).",
        "",
        "| Path | Distinct queries | Query/variant cases | Examples |",
        "|---|---:|---:|---|",
    ]
    for path in sorted(hits, key=lambda p: (len(hits[p]), p)):
        distinct = len(bench_policy.distinct_queries(hits[path]))
        flag = " **(one query)**" if distinct == 1 else ""
        examples = ", ".join(sorted(hits[path])[:4])
        lines.append(
            f"| `{path}`{flag} | {distinct} | {len(hits[path])} | {examples} |"
        )
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
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        choices=list(TIERS),
        default="quick",
        help="quick: minutes, for a code change; full: for reports",
    )
    parser.add_argument(
        "--baseline",
        default="",
        help="git revision to build and compare against, e.g. main",
    )
    parser.add_argument("--suites", default=None)
    parser.add_argument(
        "--all-suites",
        action="store_true",
        help="run every development and held-out suite (default for full tier)",
    )
    parser.add_argument(
        "--heldout",
        action="store_true",
        help="also run the held-out suites (ClickBench)",
    )
    parser.add_argument("--engines", default="mojo,polars,duckdb")
    parser.add_argument("--queries", default="", help="comma-separated subset")
    parser.add_argument("--scale", choices=list(SCALES), default=None)
    parser.add_argument(
        "--variants",
        choices=["all", "base"],
        default="all",
        help="'base' skips the perturbed data variants (not for reporting)",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument(
        "--timeout", type=int, default=900, help="seconds per query per worker"
    )
    parser.add_argument(
        "--reuse-references",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="reuse cached Polars/DuckDB outcomes (default: on for quick)",
    )
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="record fast-path coverage (default: on for full)",
    )
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
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="execute each query once, compare fresh answers, and skip performance reports",
    )
    args = parser.parse_args()
    for name, value in TIERS[args.tier].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.correctness_only:
        if args.baseline or args.report_from:
            parser.error(
                "--correctness-only cannot use --baseline or --report-from"
            )
        if not {"mojo", "duckdb"} <= set(args.engines.split(",")):
            parser.error(
                "--correctness-only requires Mojo and the independent DuckDB reference"
            )
        args.rounds, args.reps = 1, 0
        args.trace = args.reuse_references = False
        args.allow_busy = True
        if args.output == ROOT / "build" / "suites" / "results.json":
            args.output = ROOT / "build" / "suites" / "answers.json"
    if args.all_suites:
        args.suites = ",".join(SUITES)
    if args.heldout:
        args.suites += "".join(
            "," + n for n, s in SUITES.items() if s["role"] == "heldout"
        )
    args.suites = ",".join(dict.fromkeys(args.suites.split(",")))
    if args.report_from:
        result = json.loads(args.report_from.read_text())
    else:
        start = time.monotonic()
        result = measure(args)
        print(f"measured in {time.monotonic() - start:.0f} s", file=sys.stderr)
    if args.correctness_only:
        failures = check_correctness(result)
        if failures:
            print("dataframe_mojo failures:", failures, file=sys.stderr)
            sys.exit(1)
        return
    # One complete validation covers Markdown, HTML and the exit check in
    # this invocation. A new command always reads and validates answers again.
    cells = evaluate(result)
    text = report(result, cells)
    print(text)
    page = bench_html.render(
        result, cells, SUITES, VARIANTS, instrumented_paths()
    )
    if args.output and not args.report_from:
        args.output.with_suffix(".md").write_text(text)
        args.output.with_suffix(".html").write_text(page)
    elif args.report_from:
        args.report_from.with_suffix(".md").write_text(text)
        args.report_from.with_suffix(".html").write_text(page)
        print(f"wrote {args.report_from.with_suffix('.html')}", file=sys.stderr)
    if args.check:
        bad = [
            key
            for key, cell in cells.items()
            if (key[3] == "mojo" or key[3].startswith("base@"))
            and cell["status"] in ("wrong answer", "failed", "timeout")
        ]
        if bad:
            print("dataframe_mojo failures:", bad, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
