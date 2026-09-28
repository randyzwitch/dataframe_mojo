"""Run the external benchmark suites against Polars and DuckDB.

See docs/benchmarks.md for the rules these measurements serve. Run under the
oracle environment, after building libdfparquet (pixi run -e native
build-dfparquet):

    # A code change, in minutes: development suites at 1M rows, this build
    # against main, Polars/DuckDB from the reference cache.
    pixi run -e oracle python3 scripts/bench_suites.py --baseline main

    # A report: 10M rows, three rounds, fast-path coverage, held-out suites.
    pixi run -e oracle python3 scripts/bench_suites.py --tier full --heldout

One worker process per (suite, variant, engine, round) loads the tables once
(untimed), then warms up and times --reps runs of each query; the per-round
figure is the fastest run, the report shows the median over rounds, and the
order of the engines under test rotates. Every answer is checked against
DuckDB (see engines.py `summary`); a wrong answer is reported as such and
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
import time

ROOT = Path(__file__).resolve().parents[1]
SUITES_DIR = ROOT / "benchmarks" / "suites"
sys.path.insert(0, str(SUITES_DIR))
sys.path.insert(0, str(ROOT / "scripts"))

import datagen  # noqa: E402
from bench_host import compiler_processes  # noqa: E402

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

# Data sizes. "smoke" checks answers in CI; "dev" is the quick tier's size
# for the edit-measure loop; "default" is for reports.
SCALES = {
    "smoke": {"h2o_rows": 100_000, "pdsh_sf": 0.01, "clickbench_partitions": 1},
    "dev": {"h2o_rows": 1_000_000, "pdsh_sf": 0.1, "clickbench_partitions": 1},
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

# Two tiers (docs/benchmarks.md). `quick` is for checking a code change in
# minutes: development suites at 1M rows, every data variant, one round, and
# Polars/DuckDB outcomes reused from the reference cache, since their code
# does not change when this library does. `full` is for reports.
TIERS = {
    "quick": {
        "suites": "h2o_groupby,h2o_join",
        "scale": "dev",
        "rounds": 3,
        "reps": 3,
        "reuse_references": True,
        "trace": False,
    },
    "full": {
        "suites": "h2o_groupby,h2o_join",
        "scale": "default",
        "rounds": 3,
        "reps": 3,
        "reuse_references": False,
        "trace": True,
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
    sources = list((ROOT / "dataframe").glob("*.mojo")) + list(
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
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=ROOT, text=True
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
    else:
        env.pop("DATAFRAME_TRACE_PATHS", None)
    return env


def parse(output, queries):
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
        elif kind == "summary":
            values = [float(v) for v in fields[3].split(",")] if fields[3] else []
            names = fields[4].split(",") if len(fields) > 4 and fields[4] else []
            entry.update(
                status="ok",
                reason="",
                summary={"height": int(fields[2]), "values": values, "names": names},
            )
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


def _wait_quiet(limit=3600):
    """Block until no compiler runs; other work on a shared host must not
    overlap timing (docs/benchmarks.md)."""
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
            found = parse(out, queries)
            for entry in found.values():
                if entry["status"] == "failed" and entry["reason"] == "no output":
                    entry.update(status="timeout", reason="")
            return found, ""
        if allow_busy or not compiler_processes():
            break
        print("compiler activity overlapped a run; repeating it", file=sys.stderr)
    else:
        raise RuntimeError("compiler activity kept overlapping timed runs")
    found = parse(proc.stdout, queries)
    if proc.returncode != 0:
        tail = " | ".join((proc.stderr or "").strip().splitlines()[-2:])
        for entry in found.values():
            if entry["status"] == "failed" and entry["reason"] == "no output":
                entry["reason"] = f"worker exited {proc.returncode}: {tail}"
    return found, proc.stderr


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
        "tier": args.tier,
        "baseline": args.baseline or "",
        "threads": args.threads,
        "scale": args.scale,
        "rounds": args.rounds,
        "reps": args.reps,
        "data_root": str(datagen.data_root()),
    }


# --- reference cache -----------------------------------------------------------


def _cache_file(suite, variant, engine):
    return datagen.data_root() / "reference-cache" / suite / variant / f"{engine}.json"


def _cache_key(engine, paths, args):
    import duckdb
    import polars

    version = polars.__version__ if engine == "polars" else duckdb.__version__
    files = sorted(
        f"{Path(p).name}:{Path(p).stat().st_size}:{int(Path(p).stat().st_mtime)}"
        for p in paths.values()
    )
    return f"{engine} {version} threads={args.threads} reps={args.reps} " + " ".join(files)


def load_reference(suite, variant, engine, paths, args):
    """Cached Polars/DuckDB outcomes for this data, engine version and thread
    count, or None. Their code does not change when this library does."""
    path = _cache_file(suite, variant, engine)
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    if cached.get("key") != _cache_key(engine, paths, args):
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
            plan.append((suite, variant, queries, tables(suite, variant, args.scale)))

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
            print(f"{suite}/{variant} {engine}: measured and cached", file=sys.stderr)
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
            "rounds": [min(r["times"]) / 1e6 for r in runs if r.get("times")],
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
    engines = result.get("engines", list(ENGINES))
    lines = [
        "# Benchmark suites",
        "",
        f"Tier `{info.get('tier', 'full')}`. Revision `{info['revision'][:10]}`"
        f"{' (dirty)' if info['dirty'] else ''}, "
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
    if result.get("baseline"):
        lines += [
            f"`{result['baseline']}` is this library built at the baseline "
            "revision; `vs " + result["baseline"] + "` above 1 means the change "
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
            lines += _suite_table(suite, cells, engines)
    if result.get("trace"):
        lines += _coverage(result["trace"])
    return "\n".join(lines) + "\n"


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
    lines.append(f"- **Slower ({len(slower)}):** " + ("; ".join(slower) or "none"))
    lines.append(f"- **Faster ({len(faster)}):** " + ("; ".join(faster) or "none"))
    lines.append("")
    return lines


def _suite_table(suite, cells, all_engines):
    variants = [v for v in VARIANTS[suite] if any(
        k[0] == suite and k[1] == v for k in cells
    )]
    base = variants[0]
    engines = [e for e in all_engines if any(
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
        "--heldout",
        action="store_true",
        help="also run the held-out suites (pdsh, clickbench)",
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
    args = parser.parse_args()
    for name, value in TIERS[args.tier].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.heldout:
        args.suites += ",pdsh,clickbench"
    if args.report_from:
        result = json.loads(args.report_from.read_text())
    else:
        start = time.monotonic()
        result = measure(args)
        print(
            f"measured in {time.monotonic() - start:.0f} s", file=sys.stderr
        )
    text = report(result)
    print(text)
    if args.output and not args.report_from:
        args.output.with_suffix(".md").write_text(text)
    if args.check:
        bad = [
            key
            for key, cell in evaluate(result).items()
            if (key[3] == "mojo" or key[3].startswith("base@"))
            and cell["status"] in ("wrong answer", "failed", "timeout")
        ]
        if bad:
            print("dataframe_mojo failures:", bad, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
