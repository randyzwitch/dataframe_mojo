"""Pair two prebuilt join-matrix revisions, preserving each timed sample.

The matrix runner warms once and receives repetitions=1, so its reported
minimum is one actual timed query. Output height/checksum must agree across
revisions; semantic tests provide the stronger full-result assertions.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from bench_upstream_joins import compiler_processes


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cases", default="lazy_narrow")
    parser.add_argument("--layouts", default="base,shuffled,wide")
    args = parser.parse_args()
    if min(args.rows, args.rounds, args.threads) < 1:
        parser.error("rows, rounds and threads must be positive")
    runners = {
        "baseline": args.baseline.resolve(),
        "candidate": args.candidate.resolve(),
    }
    inputs = set()
    for layout in args.layouts.split(","):
        if layout not in {"base", "shuffled", "wide"}:
            parser.error("unknown layout")
        inputs.add(
            args.data
            / f"left_{args.rows}{'_wide' if layout == 'wide' else ''}.csv"
        )
        inputs.add(
            args.data
            / f"right_{args.rows}{'_' + layout if layout != 'base' else ''}.csv"
        )
    result = {
        "status": "incomplete",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "candidate_sources": {
            name: digest(Path(__file__).resolve().parents[1] / name)
            for name in [
                "dataframe/frame.mojo",
                "dataframe/join_hash.mojo",
                "dataframe/lazy.mojo",
                "benchmarks/bench_join_matrix.mojo",
                "pixi.lock",
            ]
        },
        "threads": args.threads,
        "rows": args.rows,
        "runners": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in runners.items()
        },
        "inputs": {path.name: digest(path) for path in sorted(inputs)},
        "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    expected = {}
    for round_number in range(args.rounds):
        for case in args.cases.split(","):
            for layout in args.layouts.split(","):
                order = ["baseline", "candidate"]
                if round_number % 2:
                    order.reverse()
                for name in order:
                    if compiler_processes():
                        raise RuntimeError("compiler activity before timing")
                    command = [
                        str(runners[name]),
                        str(args.data.resolve()),
                        str(args.rows),
                        "1",
                        case,
                        "--variant",
                        layout,
                    ]
                    output = subprocess.check_output(
                        command,
                        text=True,
                        timeout=300,
                        env={
                            **os.environ,
                            "DATAFRAME_THREADS": str(args.threads),
                        },
                    )
                    if compiler_processes():
                        raise RuntimeError(
                            "compiler activity overlapped timing"
                        )
                    fields = output.strip().split("\t")
                    if (
                        len(fields) != 4
                        or fields[0] != case
                        or int(fields[1]) <= 0
                    ):
                        raise RuntimeError(
                            f"unexpected runner output: {output!r}"
                        )
                    answer = (int(fields[2]), float(fields[3]))
                    key = (case, layout)
                    if key in expected and answer != expected[key]:
                        raise RuntimeError(
                            f"height/checksum mismatch: {key} {answer} != {expected[key]}"
                        )
                    expected[key] = answer
                    sample = {
                        "round": round_number,
                        "revision": name,
                        "case": case,
                        "layout": layout,
                        "ns": int(fields[1]),
                        "height": answer[0],
                        "checksum": answer[1],
                    }
                    result["samples"].append(sample)
                    save()
                    print(json.dumps(sample), flush=True)
    result["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
