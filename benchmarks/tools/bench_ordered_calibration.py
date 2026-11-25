"""Capture a Metal profile, then audit automatic ordered grouping on unseen row counts."""
import argparse
from itertools import product
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys

import duckdb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from metal_calibration import fit, publish


def invoke(binary, env, mode, rows, chunks, distribution, repetitions, warmup_rows=None):
    command = [str(binary), mode, str(rows), str(chunks), distribution, str(repetitions)]
    if warmup_rows is not None:
        command.append(str(warmup_rows))
    result = subprocess.run(
        command,
        env=env, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{mode} {rows}/{chunks}/{distribution}: {result.stderr}{result.stdout}")
    records = [json.loads(line) for line in result.stdout.splitlines()]
    if mode != "export" and len(records) != (1 if mode == "capture" else repetitions):
        raise ValueError(f"Unexpected {mode} record count: {len(records)}")
    return records


def reference(rows, distribution):
    with duckdb.connect() as db:
        # Window position counts every input row. FILTER excludes null values
        # when choosing extrema, while first()/last() retain null endpoints.
        answer = db.execute("""
            WITH keys AS (
                SELECT r, CASE WHEN $skew THEN
                    CASE WHEN r % 3 = 0 THEN 1 + ((r // 3) % 31) ELSE 0 END
                    ELSE r % 32 END AS slot
                FROM range($rows) t(r)
            ), ranked AS (
                SELECT r, 300 - 7 * slot AS k, CASE WHEN r % 11 <> 0 AND slot <> 31
                    THEN (r * 67 + r // 131) % 503 - 251 END AS v,
                    row_number() OVER (PARTITION BY slot ORDER BY r) - 1 AS pos
                FROM keys
            )
            SELECT k, first(v ORDER BY r), last(v ORDER BY r),
                first(pos ORDER BY v ASC, r) FILTER (WHERE v IS NOT NULL),
                first(pos ORDER BY v DESC, r) FILTER (WHERE v IS NOT NULL)
            FROM ranked GROUP BY k ORDER BY min(r)
        """, {"rows": rows, "skew": distribution == "skew"}).fetchall()
        return [list(row) for row in answer]


def emit(stream, record):
    print(json.dumps(record), file=stream, flush=True)


def measure(args):
    binary = args.binary.resolve(strict=True)
    if set(args.train) & set(args.holdout):
        raise ValueError("Training and held-out row counts must be disjoint")
    if len(set(args.train)) < 3 or args.reps < 3 or args.rounds < 1:
        raise ValueError("Use at least three training scales, three repetitions and one round")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, DATAFRAME_HANDOFF_CALIBRATION="")
    shapes = list(product((1, 8), ("balanced", "skew")))
    if args.reuse_capture:
        training = [json.loads(line) for line in args.reuse_capture.read_text().splitlines()]
        if {record["rows"] for record in training} != set(args.train):
            raise ValueError("Reused capture must match the requested training row counts")
    else:
        training = []
        for rows, (chunks, distribution) in product(args.train, shapes):
            record, = invoke(binary, env, "capture", rows, chunks, distribution, args.reps)
            training.append(record)
    with (directory / "capture.jsonl").open("x") as capture:
        for record in training:
            emit(capture, record)
            if record.get("status") != "measured" or "first_accel_ns" not in record:
                raise ValueError(f"Expected checked warm and first-use samples: {record}")
    profile = publish(fit(training), directory)
    env["DATAFRAME_HANDOFF_CALIBRATION"] = str(profile)
    print(f"Using {profile}", flush=True)
    with (directory / "holdout.jsonl").open("x") as raw, (directory / "summary.jsonl").open("x") as summary:
        for rows, (chunks, distribution) in product(args.holdout, shapes):
            shape = dict(rows=rows, chunks=chunks, distribution=distribution)
            actual = invoke(binary, env, "export", rows, chunks, distribution, 1)
            expected = reference(rows, distribution)
            if actual != expected:
                raise ValueError(f"DuckDB ordered result differs for {shape}: {actual} != {expected}")
            emit(raw, {**shape, "mode": "duckdb", "version": duckdb.__version__, "rows_checked": actual})
            for warmup_rows in ([None, args.warmup_rows] if args.warmup_rows else [None]):
                audit_case(binary, env, shape, args, raw, summary, warmup_rows)


def audit_case(binary, env, shape, args, raw, summary, warmup_rows):
    rows, chunks, distribution = shape["rows"], shape["chunks"], shape["distribution"]
    samples = {}
    shape = {**shape, "prewarm_rows": warmup_rows}
    reps = args.reps
    for mode in ("profile-metal", "profile-auto"):
        records = invoke(binary, env, mode, rows, chunks, distribution, reps, warmup_rows)
        samples[mode] = records
        emit(raw, {**shape, "mode": mode, "records": records})
    modes = ("cpu", "materialized", "metal", "auto")
    for turn in range(args.rounds):
        for mode in (modes if turn % 2 == 0 else reversed(modes)):
            records = invoke(binary, env, mode, rows, chunks, distribution, reps, warmup_rows)
            samples.setdefault(mode, []).append(records)
            emit(raw, {**shape, "mode": mode, "round": turn, "records": records})
    phases = (
        [("cold", 0), ("repeated", -1)] if warmup_rows is None
        else [("prewarmed", 0), ("prewarmed_repeated", -1)]
    )
    for thermal, rep in phases:
        timings = {
            mode: median(trial[rep]["ns"] for trial in samples[mode])
            for mode in modes
        }
        winner_ns = min(timings["cpu"], timings["materialized"])
        fastest = "metal" if timings["metal"] < winner_ns else "cpu"
        observed = samples["profile-auto"][rep]
        record = {
            **shape, "temperature": thermal, "ns": timings,
            "fastest": fastest, "selected": observed["device"],
            "completed_launches": observed["completed_launches"],
            "cpu_reason": observed["cpu_reason"],
            "metal_warm_before_auto": warmup_rows is not None or (
                rep != 0 and any(p["device"] == "metal" for p in samples["profile-auto"][:rep])
            ),
        }
        emit(summary, record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path, help="new directory for capture, profile and held-out evidence")
    parser.add_argument("--train", type=int, nargs="+", default=[256, 8192, 32768, 131072, 1048576])
    parser.add_argument("--holdout", type=int, nargs="+", default=[1024, 49152, 65536, 98304, 524288, 786432])
    parser.add_argument("--reuse-capture", type=Path, help="refit the same training samples")
    parser.add_argument("--warmup-rows", type=int, help="also compare after a verified auto Metal submission of the same layout at this size")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    measure(parser.parse_args())
