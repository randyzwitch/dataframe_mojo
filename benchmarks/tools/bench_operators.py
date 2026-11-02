"""Operator-level parity harness: one operator at a time, three engines.

Where bench_suites.py times whole queries, this times single operators
(hash join, hash aggregate, filter -> project -> sum, sort and top-N,
window, and three join builds feeding one probe) over synthetic inputs
that sweep data properties (benchmarks/operators/matrices.py). Each cell
runs on this library, DuckDB and Polars with the same Parquet tables
loaded into memory first (untimed), and records per run:

- the query's wall time (all three engines);
- the operator's busy time, summed over threads: DuckDB's JSON profile
  (`timing` of HASH_JOIN, HASH_GROUP_BY, ORDER_BY, WINDOW, ...), Polars'
  POLARS_LOG_METRICS node totals, and this library's execution report
  (`LazyFrame.profile` busy_ms, #439), where the operator records it;
- the summed busy time of all operators (CPU time).

Every engine's answer is checked against DuckDB's with suite_answers.py's
columnar comparison (floats within its atol/rtol); sort cells check every
engine, DuckDB included, against the unsorted relation with the order and
limit as rank constraints. Results go to <out>/results.json, and
<out>/report.md has a table per matrix plus the worst cells.

    pixi run -e oracle bench-operators --out /some/dir
    pixi run -e oracle bench-operators --matrix join --cells 0-19 --out DIR
    pixi run -e oracle bench-operators --baseline cp26 --out DIR

Cells already in results.json are skipped, so a long matrix can run in
pieces. On a shared host, pass --wrap-timing (a command that runs the
timed chunk with the machine quiet) and --wrap-build; answer checks and
data generation run outside the wrapper, niced. Data lives in the suites'
data root (benchmarks/suites/datagen.py) under operators/.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
OPERATORS = ROOT / "benchmarks" / "operators"
sys.path.insert(0, str(OPERATORS))
sys.path.insert(0, str(ROOT / "benchmarks" / "suites"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matrices  # noqa: E402

ENGINES = ("mojo", "duckdb", "polars")
FLAG = 1.25

# The operator(s) a matrix measures, by engine: a regular expression over
# DuckDB's operator types, Polars' node names and this library's
# execution-report labels. None means every operator (the filter matrix
# measures the whole filter -> project -> sum pipeline, which DuckDB runs
# inside its scan).
OPERATOR_NAMES = {
    "join": {
        "duckdb": r"JOIN",
        "polars": r"join",
        "mojo": r"^JOIN",
    },
    "sched": {
        "duckdb": r"JOIN",
        "polars": r"join",
        "mojo": r"^JOIN",
    },
    "agg": {
        "duckdb": r"GROUP_BY",
        "polars": r"^group-by",
        "mojo": r"^GROUP_BY",
    },
    "sort": {
        "duckdb": r"^(ORDER_BY|TOP_N)",
        # Polars' top-k feeds a dynamic filter back to its input.
        "polars": r"^(sort|top-k|bottom-k|filter)$",
        "mojo": r"^(SORT|TOP_K|SLICE)",
    },
    "window": {
        "duckdb": r"WINDOW",
        # Polars runs ordered windows as in-memory maps or columnar
        # functions inside its streaming graph: every node but the
        # plumbing.
        "polars": (
            r"^(?!(in-memory-source|in-memory-sink|simple-projection"
            r"|multiplexer|zip-broadcast|select)$)"
        ),
        "mojo": r"^(WITH_COLUMNS|WINDOW|OVER)",
    },
    "filter": {"duckdb": None, "polars": None, "mojo": None},
}


# --- cells and data -------------------------------------------------------


def smoke(cell, rows=20_000):
    """The cell at a tiny size, for tests and a quick end-to-end check."""
    cell = json.loads(json.dumps(cell))
    spec = cell["dataset"]
    for name in ("rows", "build_rows", "probe_rows"):
        if name in spec:
            spec[name] = min(spec[name], rows)
    if "groups" in spec:
        spec["groups"] = max(1, min(spec["groups"], rows // 4))
    return cell


def select_cells(args):
    every = matrices.matrices()
    chosen = []
    for name in args.matrix.split(","):
        cells = every[name]
        if args.cells:
            keep = set()
            for part in args.cells.split(","):
                if "-" in part:
                    low, high = part.split("-")
                    keep.update(range(int(low), int(high) + 1))
                else:
                    keep.add(int(part))
            cells = [c for i, c in enumerate(cells) if i in keep]
        chosen += cells
    if args.scale == "smoke":
        chosen = [smoke(c) for c in chosen]
    return chosen


def data_root():
    import datagen

    return datagen.data_root()


def generate(cell):
    import generate as gen

    return {
        name: str(path)
        for name, path in gen.generate(cell["dataset"], data_root()).items()
    }


# --- our runner -------------------------------------------------------------


def _wrapped(wrap, cmd):
    return shlex.split(wrap) + cmd if wrap else cmd


def _build(tree, binary, wrap):
    print(f"building {binary}", file=sys.stderr)
    binary.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mojo",
        "build",
        "-O3",
        "-I",
        str(tree),
        str(OPERATORS / "runner.mojo"),
        "-o",
        str(binary),
    ]
    subprocess.check_call(_wrapped(wrap, cmd), cwd=tree)


def build_runner(wrap):
    """This checkout's runner, rebuilt when a library source or the runner
    is newer than the binary."""
    binary = ROOT / "build" / "operators" / "runner"
    sources = list((ROOT / "dataframe").rglob("*.mojo")) + [
        OPERATORS / "runner.mojo"
    ]
    newest = max(p.stat().st_mtime for p in sources)
    if not binary.exists() or binary.stat().st_mtime < newest:
        _build(ROOT, binary, wrap)
    return binary


def build_baseline(ref, wrap):
    """The runner built against the library at `ref`, in a detached
    worktree under build/operators/baseline/<sha> (this checkout's runner
    source). Returns (label, binary)."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    tree = ROOT / "build" / "operators" / "baseline" / sha
    if not (tree / "dataframe").exists():
        subprocess.check_call(
            ["git", "worktree", "add", "--detach", str(tree), sha], cwd=ROOT
        )
    binary = tree / "build" / "operators" / "runner"
    if (
        not binary.exists()
        or binary.stat().st_mtime < (OPERATORS / "runner.mojo").stat().st_mtime
    ):
        _build(tree, binary, wrap)
    return f"base@{sha[:7]}", binary


def runner_args(query):
    args = []
    for key, value in query.items():
        if key == "aggs":
            value = ";".join(":".join(a) for a in value)
        elif key == "desc":
            value = ",".join("1" if d else "0" for d in value)
        elif isinstance(value, list):
            value = ",".join(value)
        args.append(f"{key}={value}")
    return args


def parse_runner(stdout):
    """Runs from the runner's tagged lines (benchmarks/operators/runner.mojo)."""
    runs, out = {}, {}
    for line in stdout.splitlines():
        fields = line.split("\t")
        if fields[0] == "wall":
            runs.setdefault(int(fields[1]), {"nodes": []})["wall_ms"] = (
                int(fields[2]) / 1e6
            )
        elif fields[0] == "op":
            runs.setdefault(int(fields[1]), {"nodes": []})["nodes"].append(
                {
                    "name": fields[3],
                    "executor": fields[4],
                    "rows": int(fields[7]),
                    "busy_ms": float(fields[8]),
                    "wall_ms": float(fields[9]),
                }
            )
        elif fields[0] == "rows":
            out["rows"] = int(fields[1])
        elif fields[0] in ("failed", "unsupported"):
            out.update(status=fields[0], reason=fields[1][:2000])
    out["runs"] = [dict(runs[k], rep=k) for k in sorted(runs)]
    return out


# --- timing -----------------------------------------------------------------


def environment(threads):
    env = dict(os.environ)
    env["DATAFRAME_THREADS"] = str(threads)
    env["POLARS_MAX_THREADS"] = str(threads)
    env["BENCH_THREADS"] = str(threads)
    env.pop("DATAFRAME_EXECUTION_REPORT", None)
    env.pop("DATAFRAME_TRACE_PATHS", None)
    return env


def run_engine(engine, binary, cell, paths, reps, answer, threads, timeout):
    """One worker process for one cell; returns its outcome."""
    env = environment(threads)
    if engine == "polars":
        env["POLARS_LOG_METRICS"] = "1"
    if binary is not None:
        cmd = [str(binary), str(reps), str(answer)] + runner_args(
            cell["query"]
        ) + [f"@{n}={p}" for n, p in paths.items()]
    else:
        spec = Path(answer).with_suffix(".cell.json")
        spec.write_text(json.dumps(dict(cell, paths=paths)))
        cmd = [
            sys.executable,
            str(OPERATORS / "operator_engines.py"),
            engine,
            str(spec),
            str(reps),
            str(answer),
        ]
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "reason": f"over {timeout} s", "runs": []}
    if binary is not None:
        out = parse_runner(proc.stdout)
    else:
        out = {"runs": []}
        for line in proc.stdout.splitlines():
            if not line.startswith("{"):
                continue
            record = json.loads(line)
            if "failed" in record:
                out.update(status="failed", reason=record["failed"])
            elif "rows" in record:
                out["rows"] = record["rows"]
            else:
                out["runs"].append(record)
        if engine == "polars":
            import operator_engines as python_engines

            nodes = python_engines.polars_nodes(proc.stderr)
            for run in out["runs"]:
                run["nodes"] = nodes.get(run["rep"], [])
    if "status" not in out:
        if proc.returncode != 0 or not out["runs"]:
            tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
            out.update(
                status="failed",
                reason=f"worker exited {proc.returncode}: {tail}"[:2000],
            )
        else:
            out["status"] = "ok"
    return out


def _matches(pattern, name):
    return pattern is None or re.search(pattern, name) is not None


def summarize(matrix, engine, outcome):
    """Medians over the runs: wall, operator busy, all-operator busy."""
    base = "mojo" if engine.startswith("base@") else engine
    pattern = OPERATOR_NAMES[matrix][base]
    walls, busy, cpu, matched = [], [], [], set()
    for run in outcome.get("runs", []):
        walls.append(run["wall_ms"])
        nodes = run.get("nodes", [])
        cpu.append(sum(n["busy_ms"] for n in nodes))
        mine = [n for n in nodes if _matches(pattern, n["name"])]
        matched.update(n["name"] for n in mine)
        busy.append(sum(n["busy_ms"] for n in mine))
    out = {
        k: v for k, v in outcome.items() if k not in ("runs",)
    }
    if walls:
        out["wall_ms"] = statistics.median(walls)
        out["wall_runs"] = walls
        out["cpu_ms"] = statistics.median(cpu)
        # A zero busy time means the operator ran but its engine did not
        # time it (this library's execution report times streamed steps
        # only); it is reported as missing, not as free.
        op = statistics.median(busy)
        out["op_busy_ms"] = op if op > 0 else None
        out["operators"] = sorted(matched)
        runs = outcome["runs"]
        out["nodes"] = runs[len(runs) // 2].get("nodes", [])
    return out


def time_chunk(plan_path):
    """Run every engine on the chunk's cells (the timed part)."""
    plan = json.loads(Path(plan_path).read_text())
    results = {}
    engines = plan["engines"]
    for index, item in enumerate(plan["cells"]):
        cell, paths = item["cell"], item["paths"]
        directory = Path(plan["answers"]) / cell["id"]
        directory.mkdir(parents=True, exist_ok=True)
        order = engines[index % len(engines):] + engines[: index % len(engines)]
        found = {}
        for engine in order:
            binary = plan["binaries"].get(engine)
            answer = directory / f"{engine}.parquet"
            outcome = run_engine(
                engine,
                binary,
                cell,
                paths,
                plan["reps"],
                answer,
                plan["threads"],
                plan["timeout"],
            )
            found[engine] = summarize(cell["matrix"], engine, outcome)
            found[engine]["answer_path"] = str(answer)
            print(
                f"{cell['id']} {engine}: {found[engine]['status']} "
                f"{found[engine].get('wall_ms', float('nan')):.1f} ms",
                file=sys.stderr,
                flush=True,
            )
        results[cell["id"]] = found
    Path(plan["output"]).write_text(json.dumps(results))


# --- answers ----------------------------------------------------------------


def _canonical_pair(actual, expected):
    """Cast both tables to comparable physical types, column by column;
    returns (actual, expected, tolerant names, type notes)."""
    import pyarrow as pa

    notes, tolerant = [], []
    a_cols, e_cols = [], []
    names = expected.column_names
    for k, name in enumerate(names):
        a, e = actual.column(k), expected.column(k)
        at, et = a.type, e.type
        strings = {"string", "large_string"}
        # Storage widths of one logical type (string offsets, decimal
        # 32/64/128 words) are not differences.
        same = {str(at), str(et)} <= strings or (
            pa.types.is_decimal(at)
            and pa.types.is_decimal(et)
            and (at.precision, at.scale) == (et.precision, et.scale)
        )
        if str(at) != str(et) and not same:
            notes.append(f"{name}: {at} vs {et}")

        def numeric(t):
            return pa.types.is_integer(t) or pa.types.is_floating(t) or (
                pa.types.is_decimal(t)
            )

        if numeric(at) and numeric(et) and str(at) != str(et):
            if pa.types.is_floating(at) or pa.types.is_floating(et):
                a, e = a.cast(pa.float64()), e.cast(pa.float64())
            elif pa.types.is_decimal(at) or pa.types.is_decimal(et):
                scale = max(
                    getattr(at, "scale", 0), getattr(et, "scale", 0)
                )
                target = pa.decimal128(38, scale)
                a, e = a.cast(target), e.cast(target)
            else:
                try:
                    a, e = a.cast(pa.int64()), e.cast(pa.int64())
                except pa.ArrowInvalid:
                    target = pa.decimal128(38, 0)
                    a, e = a.cast(target), e.cast(target)
        if pa.types.is_large_string(a.type):
            a = a.cast(pa.string())
        if pa.types.is_large_string(e.type):
            e = e.cast(pa.string())
        if pa.types.is_dictionary(a.type):
            a = a.cast(a.type.value_type)
        if pa.types.is_dictionary(e.type):
            e = e.cast(e.type.value_type)
        if pa.types.is_temporal(a.type):
            a = a.cast(pa.int32() if pa.types.is_date32(a.type) else pa.int64())
        if pa.types.is_temporal(e.type):
            e = e.cast(pa.int32() if pa.types.is_date32(e.type) else pa.int64())
        a_cols.append(a)
        e_cols.append(e)
    return (
        pa.Table.from_arrays(a_cols, names=names),
        pa.Table.from_arrays(e_cols, names=names),
        notes,
    )


def _answer(table, schema=None):
    import pyarrow as pa
    import suite_answers

    types = [suite_answers._logical_type(f.type, pa) for f in table.schema]
    return suite_answers.Answer(
        list(table.column_names), types, table, schema or {}
    )


def computed_floats(cell):
    """Columns an engine computes in floating point, compared within the
    suites' atol/rtol; copied values stay exact."""
    q = cell["query"]
    if q["op"] == "agg":
        return [alias for _, _, alias in q["aggs"]]
    if q["op"] == "filter_sum":
        return ["s"]
    return []


def sort_oracle(cell, paths):
    """The unordered, unlimited relation with the sort keys appended, for
    suite_answers.OracleCheck's rank-constrained comparison."""
    import duckdb

    q = cell["query"]
    con = duckdb.connect(config={"threads": 8})
    keys = [f"{k} AS __order{i}" for i, k in enumerate(q["by"])]
    table = con.sql(
        f"SELECT {', '.join(q['select'] + keys)} "
        f"FROM read_parquet('{paths[q['t']]}')"
    ).to_arrow_table()
    con.close()
    spec = {
        "width": len(q["select"]),
        "order": [
            {"descending": d, "nulls_first": False} for d in q["desc"]
        ],
        "limit": q["limit"] or None,
        "offset": 0,
    }
    return table, spec


def check_cell(cell, paths, found):
    """Answer status per engine: DuckDB is the oracle, except for sorts,
    where every engine is checked against the independent relation."""
    import pyarrow.parquet as pq
    import suite_answers

    statuses = {}
    oracle_engine = "duckdb"
    expected = None
    sort = cell["query"]["op"] == "sort"
    if sort:
        relation, spec = sort_oracle(cell, paths)
    elif found.get(oracle_engine, {}).get("status") == "ok":
        expected = pq.read_table(found[oracle_engine]["answer_path"])
    for engine, outcome in found.items():
        if outcome.get("status") != "ok":
            statuses[engine] = {"status": "no_answer"}
            continue
        if engine == oracle_engine and not sort:
            statuses[engine] = {"status": "oracle"}
            continue
        if expected is None and not sort:
            statuses[engine] = {"status": "unverified", "reason": "no oracle"}
            continue
        actual = pq.read_table(outcome["answer_path"])
        try:
            if sort:
                width = spec["width"]
                if actual.num_columns != width:
                    statuses[engine] = {
                        "status": "different",
                        "reason": "column_count",
                    }
                    continue
                head = relation.select(list(range(width)))
                a, e, notes = _canonical_pair(actual, head)
                keys = relation.select(list(range(width, relation.num_columns)))
                full = e
                for name, column in zip(keys.column_names, keys.columns):
                    full = full.append_column(name, column)
                oracle = _answer(full, {"oracle": spec})
                result = suite_answers.compare_selection(_answer(a), oracle)
            else:
                if actual.num_columns != expected.num_columns:
                    statuses[engine] = {
                        "status": "different",
                        "reason": "column_count",
                    }
                    continue
                a, e, notes = _canonical_pair(actual, expected)
                floats = [
                    n
                    for n in computed_floats(cell)
                    if n in e.column_names
                    and str(e.column(e.column_names.index(n)).type) == "double"
                ]
                result = suite_answers.compare_answers(
                    _answer(a), _answer(e), tolerant_columns=floats
                )
            result = dict(result)
            if notes:
                result["type_notes"] = notes
            if list(actual.column_names) != list(e.column_names):
                result["name_notes"] = list(actual.column_names)
            statuses[engine] = result
        except Exception as error:  # a broken answer is a finding
            statuses[engine] = {"status": "error", "reason": str(error)[:500]}
    return statuses


# --- driver -----------------------------------------------------------------


def provenance(args, engines):
    def run(cmd):
        try:
            return subprocess.check_output(
                cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    import duckdb
    import polars

    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "revision": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain", "--", "dataframe"])),
        "mojo": run(["mojo", "--version"]),
        "duckdb": duckdb.__version__,
        "polars": polars.__version__,
        "cpu": platform.processor() or platform.machine(),
        "threads": args.threads,
        "reps": args.reps,
        "scale": args.scale,
        "engines": engines,
        "baseline": args.baseline or "",
    }


def load_results(path):
    if path.exists():
        return json.loads(path.read_text())
    return {"cells": {}}


def measure(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    result = load_results(results_path)
    engines = [e for e in args.engines.split(",") if e]
    binaries = {}
    if "mojo" in engines:
        binaries["mojo"] = str(build_runner(args.wrap_build))
    if args.baseline:
        label, binary = build_baseline(args.baseline, args.wrap_build)
        engines.append(label)
        binaries[label] = str(binary)
    result["provenance"] = provenance(args, engines)
    cells = select_cells(args)
    todo = [
        c
        for c in cells
        if args.force
        or c["id"] not in result["cells"]
        or any(e not in result["cells"][c["id"]]["engines"] for e in engines)
    ]
    answers = Path(tempfile.mkdtemp(prefix="operator-answers-"))
    try:
        for start in range(0, len(todo), args.chunk):
            chunk = todo[start: start + args.chunk]
            items = [{"cell": c, "paths": generate(c)} for c in chunk]
            plan = {
                "cells": items,
                "engines": engines,
                "binaries": binaries,
                "reps": args.reps,
                "threads": args.threads,
                "timeout": args.timeout,
                "answers": str(answers),
                "output": str(answers / "chunk.json"),
            }
            plan_path = answers / "plan.json"
            plan_path.write_text(json.dumps(plan))
            cmd = [sys.executable, str(Path(__file__)), "--time-chunk",
                   str(plan_path)]
            subprocess.check_call(_wrapped(args.wrap_timing, cmd))
            timed = json.loads((answers / "chunk.json").read_text())
            for item in items:
                cell = item["cell"]
                found = timed[cell["id"]]
                checks = check_cell(cell, item["paths"], found)
                entry = result["cells"].setdefault(
                    cell["id"],
                    {"engines": {}, "answers": {}},
                )
                entry.update(
                    matrix=cell["matrix"],
                    params=cell["params"],
                    query=cell["query"],
                )
                for engine, outcome in found.items():
                    outcome.pop("answer_path", None)
                    entry["engines"][engine] = outcome
                    entry["answers"][engine] = checks.get(engine, {})
                shutil.rmtree(answers / cell["id"], ignore_errors=True)
            results_path.write_text(json.dumps(result, indent=1))
            (out / "report.md").write_text(report(result))
            print(
                f"{min(start + args.chunk, len(todo))}/{len(todo)} cells",
                file=sys.stderr,
            )
    finally:
        shutil.rmtree(answers, ignore_errors=True)
    results_path.write_text(json.dumps(result, indent=1))
    (out / "report.md").write_text(report(result))
    return result


# --- report -----------------------------------------------------------------


def _fmt(value):
    if value is None:
        return "—"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def ratios(entry):
    """(wall ratio, busy ratio) of this library against the faster of
    DuckDB and Polars, None where a side is missing."""
    e = entry["engines"]

    def ok(name, field):
        found = e.get(name, {})
        if found.get("status") != "ok":
            return None
        return found.get(field)

    out = []
    for field in ("wall_ms", "op_busy_ms"):
        mine = ok("mojo", field)
        theirs = [v for v in (ok("duckdb", field), ok("polars", field)) if v]
        out.append(mine / min(theirs) if mine and theirs else None)
    return tuple(out)


def _answer_text(entry, engine):
    status = entry.get("answers", {}).get(engine, {}).get("status", "")
    return {
        "equal": "ok",
        "oracle": "ok",
        "": "",
    }.get(status, status)


def _overlap(rows):
    """Three independent builds: summed join busy time over wall time
    (how many threads the joins kept busy on average)."""
    lines = [
        "Summed join busy time / wall time (threads kept busy):",
        "",
        "| cell | ours | DuckDB | Polars |",
        "|---|---|---|---|",
    ]
    for cid, entry in rows:
        cells = []
        for name in ("mojo", "duckdb", "polars"):
            found = entry["engines"].get(name, {})
            busy, wall = found.get("op_busy_ms"), found.get("wall_ms")
            cells.append(_fmt(busy / wall) if busy and wall else "—")
        lines.append(f"| {cid} | " + " | ".join(cells) + " |")
    return lines + [""]


def _params(entry):
    return ", ".join(f"{k}={v}" for k, v in entry["params"].items())


def report(result):
    cells = result["cells"]
    prov = result.get("provenance", {})
    base = prov.get("baseline") and next(
        (e for e in prov.get("engines", []) if e.startswith("base@")), None
    )
    lines = [
        "# Operator parity",
        "",
        f"Revision `{prov.get('revision', '')[:12]}`"
        f"{' (dirty)' if prov.get('dirty') else ''}; "
        f"DuckDB {prov.get('duckdb', '')}, Polars {prov.get('polars', '')}; "
        f"{prov.get('threads', '')} threads; median of "
        f"{prov.get('reps', '')} runs; scale {prov.get('scale', '')}.",
        "",
        "Times are milliseconds. *wall*: the query's wall time with inputs "
        "in memory. *op busy*: the measured operator's time summed over "
        "threads (— where the engine does not time it). Ratios are this "
        f"library over the faster of DuckDB and Polars; ≥{FLAG} is flagged. "
        "*ans*: this library's / Polars' answer against DuckDB's.",
        "",
    ]
    flagged = []
    for matrix in matrices.MATRICES:
        rows = sorted(
            (cid, entry)
            for cid, entry in cells.items()
            if entry["matrix"] == matrix
        )
        if not rows:
            continue
        header = (
            "| cell | parameters | ours | DuckDB | Polars | ratio | "
            "ours op | DuckDB op | Polars op | op ratio | ans | flag |"
        )
        if base:
            header = header[:-1] + f" {base} |"
        lines += [f"## {matrix}", "", header,
                  "|" + "---|" * (header.count("|") - 1)]
        for cid, entry in rows:
            e = entry["engines"]
            wall, busy = ratios(entry)

            def get(name, field):
                found = e.get(name, {})
                if found.get("status") != "ok":
                    return found.get("status", "—")
                return _fmt(found.get(field))

            flag = ""
            if (wall and wall >= FLAG) or (busy and busy >= FLAG):
                flag = "**slow**"
                flagged.append((max(wall or 0, busy or 0), cid, entry))
            if e.get("mojo", {}).get("status") not in (None, "ok"):
                flag = f"**{e['mojo']['status']}**"
                flagged.append((float("inf"), cid, entry))
            answers = "/".join(
                _answer_text(entry, n) or "—" for n in ("mojo", "polars")
            )
            if answers not in ("ok/ok",):
                if "different" in answers or "error" in answers:
                    flag += " **wrong**"
            row = (
                f"| {cid} | {_params(entry)} | {get('mojo', 'wall_ms')} | "
                f"{get('duckdb', 'wall_ms')} | {get('polars', 'wall_ms')} | "
                f"{_fmt(wall)} | {get('mojo', 'op_busy_ms')} | "
                f"{get('duckdb', 'op_busy_ms')} | "
                f"{get('polars', 'op_busy_ms')} | {_fmt(busy)} | "
                f"{answers} | {flag} |"
            )
            if base:
                row += f" {get(base, 'wall_ms')} |"
            lines.append(row)
        lines.append("")
        if matrix == "sched":
            lines += _overlap(rows)
    lines += ["## Worst cells", ""]
    flagged.sort(key=lambda item: -item[0])
    lines += [
        "| cell | parameters | ours wall | faster wall | wall ratio | "
        "op ratio | ours status |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, cid, entry in flagged[:40]:
        e = entry["engines"]
        wall, busy = ratios(entry)
        theirs = [
            e[n]["wall_ms"]
            for n in ("duckdb", "polars")
            if e.get(n, {}).get("status") == "ok"
        ]
        lines.append(
            f"| {cid} | {_params(entry)} | "
            f"{_fmt(e.get('mojo', {}).get('wall_ms'))} | "
            f"{_fmt(min(theirs) if theirs else None)} | {_fmt(wall)} | "
            f"{_fmt(busy)} | {e.get('mojo', {}).get('status', '')} |"
        )
    total = len(cells)
    slow = sum(
        1 for entry in cells.values()
        if any(r and r >= FLAG for r in ratios(entry))
    )
    wrong = sum(
        1 for entry in cells.values()
        if entry.get("answers", {}).get("mojo", {}).get("status")
        in ("different", "error")
    )
    lines += [
        "",
        f"{slow} of {total} cells at or over {FLAG}x the faster engine; "
        f"{wrong} with a wrong answer from this library.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--matrix", default=",".join(matrices.MATRICES),
        help="comma-separated: " + ", ".join(matrices.MATRICES),
    )
    parser.add_argument(
        "--cells", default="", help="cell indexes within each matrix, e.g. 0-19,25"
    )
    parser.add_argument("--scale", choices=("full", "smoke"), default="full")
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--baseline", default="", help="git revision of this library to A/B")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=10, help="cells per timed call")
    parser.add_argument("--timeout", type=int, default=300, help="seconds per worker")
    parser.add_argument("--out", default=str(ROOT / "build" / "operators" / "results"))
    parser.add_argument("--force", action="store_true", help="rerun cells already in results.json")
    parser.add_argument("--wrap-timing", default="", help="command prefix for the timed chunks")
    parser.add_argument("--wrap-build", default="", help="command prefix for runner builds")
    parser.add_argument("--report-from", default="", help="rewrite report.md from a results.json")
    parser.add_argument("--list", action="store_true", help="print the selected cells")
    parser.add_argument(
        "--generate-only", action="store_true",
        help="write the selected cells' data and stop",
    )
    parser.add_argument("--time-chunk", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.time_chunk:
        time_chunk(args.time_chunk)
        return
    if args.list:
        for cell in select_cells(args):
            print(cell["id"], json.dumps(cell["params"]))
        return
    if args.generate_only:
        for cell in select_cells(args):
            paths = generate(cell)
            print(cell["id"], Path(next(iter(paths.values()))).parent)
        return
    if args.report_from:
        result = json.loads(Path(args.report_from).read_text())
        print(report(result))
        return
    measure(args)


if __name__ == "__main__":
    main()
