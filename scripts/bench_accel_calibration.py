#!/usr/bin/env python3
"""Run and preserve a development-only NVIDIA calibration sweep.

Requires a compiled bench_accel_calibration consumer and DuckDB. Does not change
placement policy. Every case gets a fresh process; failures remain in the output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import statistics
import shutil
import tempfile
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODES = ("cpu", "gpu_fresh_handle", "gpu_reused_handle", "kernel_interval")


def command(args: list[str]) -> str:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return (result.stdout + result.stderr).strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(binary: Path, package_dir: Path) -> None:
    source = ROOT / "benchmarks/bench_accel_calibration.mojo"
    binary = binary.resolve()
    binary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="dataframe-calibration-"
    ) as directory:
        consumer = Path(directory) / source.name
        shutil.copyfile(source, consumer)
        subprocess.run(
            [
                "mojo",
                "build",
                "-O3",
                "-I",
                str(package_dir.resolve()),
                str(consumer),
                "-o",
                str(binary),
            ],
            cwd=directory,
            check=True,
        )
    provenance = {
        "benchmark_sha256": digest(source),
        "binary_sha256": digest(binary),
        "provider": json.loads((package_dir / "provider.json").read_text()),
        "compiler": command(["mojo", "--version"]),
    }
    binary.with_suffix(".manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def verify_provenance(binary: Path, package_dir: Path) -> dict:
    provenance = json.loads(binary.with_suffix(".manifest.json").read_text())
    manifest = json.loads((package_dir / "provider.json").read_text())
    if provenance["benchmark_sha256"] != digest(
        ROOT / "benchmarks/bench_accel_calibration.mojo"
    ):
        raise ValueError("benchmark source changed since compilation")
    if (
        provenance["binary_sha256"] != digest(binary)
        or provenance["provider"] != manifest
    ):
        raise ValueError(
            "binary or provider identity changed since compilation"
        )
    for name, expected in manifest["packages"].items():
        if digest(package_dir / name) != expected:
            raise ValueError(f"package checksum mismatch: {name}")
    return provenance


def cases(quick: bool) -> list[dict]:
    dimensions = (
        [1_000, 100_000, 1_000_000, 10_000_000] if quick else [
            1_000,
            10_000,
            100_000,
            250_000,
            500_000,
            1_000_000,
            5_000_000,
            10_000_000,
        ],
        ["float32", "float64"],
        [0, 50] if quick else [0, 10, 50],
        [1, 99] if quick else [1, 50, 99],
        [1, 4] if quick else [1, 2, 4],
    )
    return [
        dict(
            zip(
                (
                    "rows",
                    "dtype",
                    "null_percent",
                    "select_percent",
                    "aggregates",
                ),
                values,
            )
        )
        for values in itertools.product(*dimensions)
    ]


def parse_output(stdout: str, case: dict, repetitions: int) -> dict:
    timings = {name: {} for name in (*MODES, "context_init", "first_query")}
    answers, metrics = {}, {}
    device = None
    for row in csv.reader(stdout.splitlines()):
        if len(row) != 4:
            raise ValueError(f"invalid output record: {row!r}")
        kind, name, index, value = row
        if kind == "device" and name == "name":
            device = value
            continue
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite measurement")
        if kind == "timing" and name in timings:
            index = int(index)
            if index in timings[name] or number <= 0:
                raise ValueError("duplicate or nonpositive timing")
            timings[name][index] = number
        elif kind == "answer" and name.startswith("a"):
            if name in answers:
                raise ValueError("duplicate answer")
            answers[name] = number
        elif kind == "metric":
            metrics[name] = number
        else:
            raise ValueError(f"unexpected record: {row!r}")
    for name, values in timings.items():
        count = repetitions if name in MODES else 1
        if set(values) != set(range(count)):
            raise ValueError(f"missing repetitions for {name}")
    if (
        set(answers) != {f"a{j}" for j in range(case["aggregates"])}
        or not device
    ):
        raise ValueError("missing answers or device identity")
    for name in (
        "upload_bytes",
        "download_bytes",
        "workspace_bytes",
        "peak_requested_device_bytes",
        "kernel_launches",
        "synchronizations",
        "device_id",
    ):
        if name not in metrics or metrics[name] < 0:
            raise ValueError(f"missing or invalid {name}")
    if metrics["kernel_launches"] != 2 * case["aggregates"]:
        raise ValueError("unexpected kernel launch count")
    return {
        "samples_ns": {
            name: [v[i] for i in sorted(v)] for name, v in timings.items()
        },
        "answers": answers,
        "metrics": metrics,
        "device": device,
    }


def duckdb_answers(connection, case: dict) -> dict:
    dtype = "FLOAT" if case["dtype"] == "float32" else "DOUBLE"
    expressions = []
    for j in range(case["aggregates"]):
        if j % 2:
            expressions.append("count(*)")
        else:
            expressions.append(
                f"CAST(sum(CAST(x * CAST({1.25 + j / 8} AS {dtype}) AS DOUBLE)) AS {dtype})"
            )
    sql = f"""WITH source AS (
        SELECT i, CAST(((i * 9973) % 10000 - 5000) / 8.0 AS {dtype}) AS x
        FROM range(19, {case['rows'] + 19}) t(i)
    ) SELECT {', '.join(expressions)} FROM source
    WHERE (i * 37 + 11) % 101 >= {case['null_percent']}
      AND x < {(case['select_percent'] * 100 - 5000) / 8}"""
    return {
        f"a{j}": float(value or 0)
        for j, value in enumerate(connection.execute(sql).fetchone())
    }


def summarize(records: list[dict], planned: list[dict] | None = None) -> dict:
    complete = planned is None or sorted(
        tuple(sorted(r["case"].items())) for r in records
    ) == sorted(tuple(sorted(c.items())) for c in planned)
    successful = [r for r in records if r["status"] == "ok"]
    buckets = {}
    for record in records:
        case = record["case"]
        key = tuple(
            case[k]
            for k in ("dtype", "null_percent", "select_percent", "aggregates")
        )
        buckets.setdefault(key, []).append(record)
    evidence = []
    for key, bucket in sorted(buckets.items()):
        bucket.sort(key=lambda r: r["case"]["rows"])
        for mode in ("gpu_fresh_handle", "gpu_reused_handle"):
            first = None
            for i, record in enumerate(bucket):
                tail = bucket[i:]
                # Empirical envelope, not a confidence interval or extrapolation.
                if all(
                    r["status"] == "ok"
                    and 1.25 * max(r["samples_ns"][mode])
                    < min(r["samples_ns"]["cpu"])
                    for r in tail
                ):
                    first = record["case"]["rows"]
                    break
            evidence.append(
                {
                    "dtype": key[0],
                    "null_percent": key[1],
                    "select_percent": key[2],
                    "aggregates": key[3],
                    "mode": mode,
                    "first_measured_rows_with_margin_through_larger_sizes": first if complete else None,
                    "measured_sizes": [r["case"]["rows"] for r in bucket],
                }
            )
    ratios = [
        statistics.median(r["samples_ns"]["cpu"])
        / statistics.median(r["samples_ns"]["gpu_fresh_handle"])
        for r in successful
    ]
    return {
        "cases": len(records),
        "complete": complete,
        "planned_cases": len(planned) if planned is not None else len(records),
        "passed": len(successful),
        "failed": len(records) - len(successful),
        "fresh_handle_speedup_geomean": math.exp(
            statistics.mean(map(math.log, ratios))
        ) if ratios else None,
        "fresh_handle_worst_speedup": min(ratios) if ratios else None,
        "placement_evidence": evidence,
    }


def write_results(output: Path, metadata: dict, records: list[dict]) -> None:
    output.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "results": records,
                "summary": summarize(records, metadata.get("planned_cases")),
            },
            indent=2,
        )
        + "\n"
    )
    with output.with_suffix(".csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "rows",
                "dtype",
                "null_percent",
                "select_percent",
                "aggregates",
                "status",
                "mode",
                "median_ns",
                "min_ns",
                "max_ns",
                "stdev_ns",
            ]
        )
        for record in records:
            prefix = list(record["case"].values()) + [record["status"]]
            if record["status"] != "ok":
                writer.writerow(prefix + [record["error"]])
                continue
            for mode, samples in record["samples_ns"].items():
                writer.writerow(
                    prefix
                    + [
                        mode,
                        statistics.median(samples),
                        min(samples),
                        max(samples),
                        statistics.stdev(samples) if len(samples) > 1 else 0,
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--build",
        action="store_true",
        help="compile and record provenance, then exit",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 3 or args.threads < 1:
        parser.error("use at least three repetitions and one CPU thread")
    if args.build:
        build(args.binary, args.package_dir)
        return
    if args.output is None:
        parser.error("--output is required when measuring")
    import duckdb

    connection = duckdb.connect()
    connection.execute("SET threads=1")
    binary = args.binary.resolve()
    provenance = verify_provenance(binary, args.package_dir)
    manifest = provenance["provider"]
    env = os.environ.copy()
    env["DATAFRAME_THREADS"] = str(args.threads)
    if (
        env.get("DATAFRAME_ACCEL_DEVICE", "0") != "0"
        or "DATAFRAME_ACCEL_MEMORY_LIMIT" in env
    ):
        parser.error(
            "calibration requires default device 0 and no provider memory limit"
        )
    planned = cases(args.quick)
    metadata = {
        "schema": 1,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_status": command(["git", "-C", str(ROOT), "status", "--short"]),
        "benchmark_sha256": digest(
            ROOT / "benchmarks/bench_accel_calibration.mojo"
        ),
        "runner_sha256": digest(Path(__file__)),
        "binary_sha256": digest(binary),
        "build": provenance,
        "provider": manifest,
        "duckdb": duckdb.__version__,
        "platform": platform.platform(),
        "cpu": command(["lscpu"]),
        "governors": {
            str(p): p.read_text().strip()
            for p in Path("/sys/devices/system/cpu").glob(
                "cpu*/cpufreq/scaling_governor"
            )
        },
        "gpu": command(["nvidia-smi"]),
        "processes_before": command(["ps", "-eo", "pid,pcpu,comm"]),
        "environment": {
            k: v
            for k, v in env.items()
            if k.startswith(("DATAFRAME_", "CUDA_", "MODULAR_DEVICE_CONTEXT_"))
        },
        "repetitions": args.repetitions,
        "threads": args.threads,
        "quick": args.quick,
        "planned_cases": planned,
        "model": "Empirical envelope: 1.25 * max GPU < min CPU at this and all larger measured sizes in the same bucket; no extrapolation, automatic placement, or confidence claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, case in enumerate(planned):
        argv = [
            str(binary),
            *(str(v) for v in case.values()),
            str(args.repetitions),
        ]
        record = {"case": case, "status": "failed", "argv": argv}
        try:
            result = subprocess.run(
                argv, env=env, text=True, capture_output=True, timeout=180
            )
            record.update(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )
            if result.returncode:
                raise RuntimeError(f"benchmark exited {result.returncode}")
            record.update(parse_output(result.stdout, case, args.repetitions))
            oracle = duckdb_answers(connection, case)
            record["duckdb_answers"] = oracle
            if any(
                not math.isclose(
                    record["answers"][k], v, rel_tol=1e-10, abs_tol=1e-10
                )
                for k, v in oracle.items()
            ):
                raise ValueError("DuckDB aggregate mismatch")
            record["status"] = "ok"
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            record["error"] = str(error)
            if isinstance(error, subprocess.TimeoutExpired):
                for key in ("stdout", "stderr"):
                    value = getattr(error, key) or ""
                    record[key] = value.decode(errors="replace") if isinstance(
                        value, bytes
                    ) else value
        records.append(record)
        write_results(args.output, metadata, records)
        print(
            f"{index + 1}/{len(planned)} {case}: {record['status']}", flush=True
        )
    metadata["finished_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    write_results(args.output, metadata, records)
    if any(r["status"] != "ok" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
