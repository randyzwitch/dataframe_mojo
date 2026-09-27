"""Compare two prebuilt revisions using PR #297's fully checked query runner."""

import argparse
import json
import os
from pathlib import Path
import platform
import time

from bench_join_revision import digest
from bench_upstream_joins import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="highcardinality,duplicate_strings")
    parser.add_argument("--layouts", default="base,shuffled,wide")
    parser.add_argument("--sizes", default="1000000,10000000")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    sizes = [int(size) for size in args.sizes.split(",")]
    if min(*sizes, args.rounds, args.reps, args.threads) < 1:
        parser.error("sizes, rounds, reps and threads must be positive")
    runners = {
        "baseline": args.baseline.resolve(),
        "candidate": args.candidate.resolve(),
    }
    root = Path(__file__).resolve().parents[1]
    result = {
        "status": "incomplete",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": args.threads,
        "rounds": args.rounds,
        "reps": args.reps,
        "runners": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in runners.items()
        },
        "candidate_sources": {
            name: digest(root / name)
            for name in [
                "dataframe/frame.mojo",
                "dataframe/join_hash.mojo",
                "dataframe/lazy.mojo",
                "dataframe/string_column.mojo",
                "dataframe/string_view.mojo",
                "dataframe/partition.mojo",
                "benchmarks/bench_upstream_joins.mojo",
                "pixi.lock",
            ]
        },
        "inputs": {},
        "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    for rows in sizes:
        for case in args.cases.split(","):
            if case not in {"highcardinality", "duplicate_strings"}:
                parser.error("unknown case")
            for layout in args.layouts.split(","):
                if layout not in {"base", "shuffled", "wide"}:
                    parser.error("unknown layout")
                if case == "duplicate_strings" and layout == "wide":
                    continue
                paths = [
                    args.data.resolve() / f"{case}-{rows}-{layout}-{side}.csv"
                    for side in ["left", "right"]
                ]
                for path in paths:
                    result["inputs"][path.name] = digest(path)
                save()
                for round_number in range(args.rounds):
                    order = ["baseline", "candidate"]
                    if round_number % 2:
                        order.reverse()
                    for name in order:
                        command = [
                            str(runners[name]),
                            case,
                            *map(str, paths),
                            str(args.reps),
                        ]
                        samples, rss = measure(
                            command,
                            {
                                **os.environ,
                                "DATAFRAME_THREADS": str(args.threads),
                            },
                            300,
                        )
                        if len(samples) != args.reps or min(samples) <= 0:
                            raise RuntimeError("invalid timed samples")
                        for rep, ns in enumerate(samples):
                            result["samples"].append(
                                {
                                    "case": case,
                                    "layout": layout,
                                    "rows": rows,
                                    "revision": name,
                                    "round": round_number,
                                    "rep": rep,
                                    "ns": ns,
                                    "peak_rss_mib": rss,
                                }
                            )
                        save()
                        print(
                            case,
                            layout,
                            rows,
                            round_number,
                            name,
                            samples,
                            flush=True,
                        )
    result["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
