#!/usr/bin/env python3
"""Create a review table from an already completed paired result JSON."""
from __future__ import annotations
import itertools, json, statistics, sys
from collections import defaultdict
from pathlib import Path


def ms(values):
    values = [v / 1_000_000 for v in values]
    return f"{statistics.median(values):.3f} [{min(values):.3f}–{max(values):.3f}]"

def pair(cpu, metal):
    return f"{ms(cpu)} / {ms(metal)}"

def prof(runs, key):
    vals = [r["profile"][key] for r in runs]
    return int(statistics.median(vals))

def main():
    path = Path(sys.argv[1])
    d = json.loads(path.read_text())
    m, s = d["metadata"], d["selection"]
    groups = defaultdict(list)
    for run in d["runs"]:
        groups[(run["rows"], run["workload"], run["variant"], run["runtime"])].append(run)
    lines = [
        "# Int32 affine SUM paired benchmark results",
        "",
        "This evidence-only report summarizes the existing paired run in `results.json`. It includes 2 input sizes × 5 workloads × 3 variants = 30 case cells, with three rounds and three timed repetitions per round. Warm values are medians across the nine recorded timed samples, with the min–max range in brackets. First-use values are medians across three rounds. They include context and shader initialization and are not cold-machine measurements. CPU and Metal results are retained for both binaries.",
        "",
        "The speedup column is baseline-binary Metal median divided by candidate-binary Metal median. This is an implementation comparison, not a GPU-versus-CPU speedup. Candidate CPU was faster than candidate Metal in all 30 cells by warm medians. The two Int32 affine workloads show 7.10–14.01× Metal improvement at 1,048,576 rows; the three Float32 workloads show 3.84–4.97×. The CPU remains faster than the candidate Metal path in every measured cell.",
        "",
        f"Environment: {m['platform']}; {m['cpu']}, {m['physical_cores']} physical cores, {int(m['memory_bytes']) // (1024**3)} GiB RAM; {m['mojo_version']}; {m['dataframe_threads']} DataFrame threads; compile flags `{', '.join(m['compile_flags'])}`.",
        "",
        f"Source identity: baseline dataframe commit `{m['binary_receipts']['baseline']['git_commit']}`; candidate run metadata commit `{m['git_commit']}` with benchmark-only working-tree changes. The measured benchmark source and driver SHA-256 values `{m['source_sha256']}` and `{m['driver_sha256']}` exactly match the committed files in harness commit `85e09ced8e306432e96300d4d4dcd4dcbc753cb2` (`Add opt-in Int32 affine sum mechanisms`). Compiler SHA-256 `{m['compiler_executable_sha256']}`; pixi.lock SHA-256 `{m['pixi_lock_sha256']}`. Engine/dataframe working trees were clean. Binary SHA-256 baseline `{m['binaries_sha256']['baseline']}`, candidate `{m['binaries_sha256']['mojo']}`. Full identities and build commands are in the receipts and `results.json`.",
        "",
        f"Selection: rows `{s['rows']}`; workloads `{', '.join(s['workloads'])}`; variants `{', '.join(s['variants'])}`; {s['rounds']} rounds × {s['reps']} repetitions; {s['threads']} threads; fresh process per case/build/round. CPU exact-reference validation and input generation were untimed; full result materialization was timed. Paired order alternated and round/process order rotated.",
        "",
        "The workload distributions and output bounds are recorded in `results.json` metadata. In particular, `int32_affine_sum` computes `sum(cast_Int64((x+3)*2))`; the filtered case applies `x > 0` first. The three Float32 workloads are `projection`, `expression15`, and `filter_half`.",
        "",
        "| Rows | Workload | Variant | Baseline first use CPU / Metal ms | Baseline warm CPU / Metal ms | Candidate first use CPU / Metal ms | Candidate warm CPU / Metal ms | Metal improvement | Baseline profile U/D/peak/launches | Candidate profile U/D/peak/launches |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rows, workload, variant in itertools.product(s["rows"], s["workloads"], s["variants"]):
        b = groups[(rows, workload, variant, "baseline")]
        c = groups[(rows, workload, variant, "mojo")]
        b_cpu = [sample["cpu_ns"] for run in b for sample in run["samples"]]
        b_metal = [sample["metal_ns"] for run in b for sample in run["samples"]]
        c_cpu = [sample["cpu_ns"] for run in c for sample in run["samples"]]
        c_metal = [sample["metal_ns"] for run in c for sample in run["samples"]]
        b_first_cpu = [run["first_use_ns"]["cpu"] for run in b]
        b_first_metal = [run["first_use_ns"]["metal"] for run in b]
        c_first_cpu = [run["first_use_ns"]["cpu"] for run in c]
        c_first_metal = [run["first_use_ns"]["metal"] for run in c]
        speedup = statistics.median(b_metal) / statistics.median(c_metal)
        bp = "/".join(str(prof(b, k)) for k in ("upload_bytes", "download_bytes", "peak_requested_bytes", "kernel_launches"))
        cp = "/".join(str(prof(c, k)) for k in ("upload_bytes", "download_bytes", "peak_requested_bytes", "kernel_launches"))
        lines.append(
            f"| {rows:,} | {workload} | {variant} | {pair(b_first_cpu, b_first_metal)} | {pair(b_cpu, b_metal)} | {pair(c_first_cpu, c_first_metal)} | {pair(c_cpu, c_metal)} | {speedup:.2f}× | {bp} | {cp} |"
        )
    lines += [
        "",
        "Profile values are medians of the three per-round profile collections, shown as uploaded bytes / downloaded bytes / peak requested bytes / kernel launches. Profiling is a separate warm collection after timed samples; phase timings can overlap and are not summed. GPU-only time and pipeline compilation attribution are unavailable unless supplied by the runtime.",
        "",
        "Validation: corrected post-run validator passed all 180/180 unique run keys, all records were `status=ok` and `validated=true`, with 3 timed samples each; output row counts were stable across runtime and round. See `validation.log` and `validation_provenance.md`.",
        "",
        "Limitations: this is development mechanism coverage, not external-suite evidence or a holdout test. The CPU equality check is not an independent oracle. First-use captures context/shader initialization but not a cold machine or cold filesystem. CPU and GPU share host memory and power limits; scheduler and thermal state can affect short runs. Read-only `pmset` state is recorded as context, not a controlled power measurement.",
        "",
    ]
    Path(sys.argv[2]).write_text("\n".join(lines))

if __name__ == "__main__":
    main()
