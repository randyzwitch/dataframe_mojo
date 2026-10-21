#!/usr/bin/env python3
"""Summarize an existing paired native Int64/UInt64 Metal benchmark run."""
from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path


def ms(values: list[int]) -> str:
    values_ms = [value / 1_000_000 for value in values]
    return f"{statistics.median(values_ms):.3f}"


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} RESULTS.json OUTPUT.md", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    output = Path(sys.argv[2])
    data = json.loads(path.read_text())
    metadata = data["metadata"]
    selection = data["selection"]
    run_plan = json.loads(path.with_name("run-plan.json").read_text())
    run_groups = {}
    for run in data["runs"]:
        key = (
            run["runtime"],
            run["rows"],
            run["workload"],
            run["variant"],
        )
        group = run_groups.setdefault(key, {"cpu": [], "metal": [], "profile": []})
        for sample in run["samples"]:
            group["cpu"].append(sample["cpu_ns"])
            group["metal"].append(sample["metal_ns"])
        group["profile"].append(run["profile"])

    workload_cells = len(selection["rows"]) * len(selection["workloads"]) * len(
        selection["variants"]
    )
    candidate_cpu_wins = sum(
        statistics.median(run_groups[("mojo", rows, workload, variant)]["cpu"])
        < statistics.median(run_groups[("mojo", rows, workload, variant)]["metal"])
        for rows, workload, variant in itertools.product(
            selection["rows"], selection["workloads"], selection["variants"]
        )
    )
    speedups: dict[str, list[float]] = {name: [] for name in selection["workloads"]}
    lines = [
        "# Paired native Int64/UInt64 SUM benchmark results",
        "",
        (
            "Evidence-only summary of the completed paired run in `results.json`. "
            f"It contains {len(selection['rows'])} row counts × "
            f"{len(selection['workloads'])} workloads × {len(selection['variants'])} "
            f"variants = {workload_cells} case cells, with {selection['rounds']} "
            f"rounds and {selection['reps']} timed samples per round. Warm medians "
            "below combine the nine timed samples for each runtime/case cell."
        ),
        "",
        (
            "The Metal comparison is baseline-binary median divided by candidate-binary "
            "median. It measures implementation change within this harness, not GPU "
            "versus CPU speedup. The candidate CPU warm median is lower than candidate "
            f"Metal in {candidate_cpu_wins}/{workload_cells} cells."
        ),
        "",
        (
            f"Environment: {metadata['platform']}; {metadata['cpu']}, "
            f"{metadata['physical_cores']} physical cores, "
            f"{int(metadata['memory_bytes']) // (1024**3)} GiB RAM; "
            f"{metadata['mojo_version']}; {metadata['dataframe_threads']} "
            f"DataFrame threads; flags `{', '.join(metadata['compile_flags'])}`."
        ),
        "",
        (
            f"Source pins: benchmark/driver contents match commit "
            f"`{run_plan['source_commit']}`; baseline engine "
            f"`{metadata['binary_receipts']['baseline']['git_commit']}`; candidate "
            f"engine ` {metadata['binary_receipts']['mojo']['git_commit']}`. "
            f"Benchmark SHA-256 `{metadata['source_sha256']}`; driver SHA-256 "
            f"`{metadata['driver_sha256']}`; pixi.lock SHA-256 "
            f"`{metadata['pixi_lock_sha256']}`. Engine/dataframe source trees were "
            "clean; benchmark and driver were benchmark-only working-tree changes "
            "whose file contents match the cited source commit. Binary hashes and "
            "build commands are retained in the receipts and `results.json`."
        ).replace("` ", "`"),
        "",
        (
            f"Selection: rows `{selection['rows']}`; workloads "
            f"`{', '.join(selection['workloads'])}`; variants "
            f"`{', '.join(selection['variants'])}`; {selection['rounds']} rounds × "
            f"{selection['reps']} repetitions; {selection['threads']} threads. "
            "Each case ran in a fresh process. Input generation and CPU result "
            "validation were untimed; complete result materialization was timed; "
            "engine/process order was rotated across rounds."
        ),
        "",
        (
            f"All {len(data['runs'])}/{len(selection['rows']) * len(selection['workloads']) * len(selection['variants']) * selection['rounds'] * 2} selected run keys completed and matched the CPU executor. That equality "
            "check validates execution but is not an independent oracle. First-use "
            "timings include context and shader initialization, not a cold-machine "
            "measurement. See `results.md` for per-runtime warm tables and `validation/` "
            "for qualification logs."
        ),
        "",
        "| Rows | Workload | Variant | Baseline CPU ms | Baseline Metal ms | Candidate CPU ms | Candidate Metal ms | Metal change | Launches baseline/candidate |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for rows, workload, variant in itertools.product(
        selection["rows"], selection["workloads"], selection["variants"]
    ):
        baseline = run_groups[("baseline", rows, workload, variant)]
        candidate = run_groups[("mojo", rows, workload, variant)]
        speedup = statistics.median(baseline["metal"]) / statistics.median(
            candidate["metal"]
        )
        speedups[workload].append(speedup)
        base_launches = int(
            statistics.median(profile["kernel_launches"] for profile in baseline["profile"])
        )
        candidate_launches = int(
            statistics.median(profile["kernel_launches"] for profile in candidate["profile"])
        )
        lines.append(
            f"| {rows:,} | {workload} | {variant} | {ms(baseline['cpu'])} | "
            f"{ms(baseline['metal'])} | {ms(candidate['cpu'])} | "
            f"{ms(candidate['metal'])} | {speedup:.2f}× | "
            f"{base_launches}/{candidate_launches} |"
        )

    lines.extend(["", "Metal improvement range by workload across all sizes and variants:"])
    lines.extend(
        f"- `{workload}`: {min(values):.2f}–{max(values):.2f}×"
        for workload, values in speedups.items()
    )
    lines.extend(
        [
            "",
            (
                "Qualification note: the archived prior full-gate log records four "
                "failures. Selector expansion, categorical grouped row count and qcut "
                "whole-DAG synchronization parity were independently reproduced on "
                "baseline. The fourth, an ordered profile/resource-estimate failure, "
                "was followed by focused resource-bound fixes whose tests passed; the "
                "full gate was not rerun as all-green. Logs and the qualification receipt "
                "are retained here."
            ),
            "",
            "Limitations: development mechanism measurements only, not external-suite evidence or a general GPU speedup. CPU and Metal share host and power limits; timing variation is possible. Profile data is a separate warm collection and host phase timings may overlap. GPU-only time and pipeline compilation attribution are unavailable unless the runtime provides them.",
            "",
        ]
    )
    output.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
