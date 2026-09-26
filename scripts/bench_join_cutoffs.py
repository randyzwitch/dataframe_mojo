#!/usr/bin/env python3
"""Build forced-path binaries and run alternating cutoff sweeps.

Uses isolated copies of dataframe/ under build/cutoffs; never changes the
library being developed. The forced limits are experimental, not API options.
Run inside pixi: python3 scripts/bench_join_cutoffs.py --build
Then --run basic|ranges|strided|ids --threads 32 --rounds 2 --reps 5.
CSV output keeps every repetition; initialization and correctness are untimed.
"""
import argparse
import csv
import statistics
import io
import os
from pathlib import Path
import tarfile
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/cutoffs"


def build(revision, variants):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in variants.split(","):
        if name not in ("serial", "parallel", "hash", "baseline", "candidate"):
            raise ValueError("unknown build variant: " + name)
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        include = ROOT if name == "candidate" else folder
        if name != "candidate":
            archive = subprocess.check_output(
                ["git", "archive", "--format=tar", revision, "dataframe"], cwd=ROOT
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(folder, filter="data")
        if name in ("serial", "parallel", "hash"):
            frame = folder / "dataframe/frame.mojo"
            text = frame.read_text()
            if (
                text.count("var cap = 64_000_000") != 2
                or "comptime _RANGE_JOIN_MAX_IDS = 8_000_000" not in text
            ):
                raise RuntimeError(
                    "revision no longer has the pre-#268/#272 guards; update the forced-path overrides"
                )
            # Force bounded addressing, including shorter-probe strided cases.
            text = text.replace(
                "if len(left) < len(right) and len(right) >= 2_000_000:", "if False:"
            )
            text = text.replace("2_000_000", "0" if name == "parallel" else "Int.MAX")
            text = text.replace("var cap = 64_000_000", "var cap = 256_000_000")
            text = text.replace(
                "comptime _RANGE_JOIN_MAX_IDS = 8_000_000",
                "comptime _RANGE_JOIN_MAX_IDS = "
                + ("0" if name == "hash" else "256_000_000"),
            )
            text = (
                text.replace("cap // 4", "cap // 16")
                .replace("cap = len(right) * 4", "cap = len(right) * 16")
                .replace("cap = len(right_values) * 4", "cap = len(right_values) * 16")
                .replace("cap = total * 4", "cap = total * 16")
            )
            frame.write_text(text)
        subprocess.run(
            [
                "mojo",
                "build",
                "-I",
                str(include),
                str(ROOT / "benchmarks/bench_join_cutoffs.mojo"),
                "-o",
                str(folder / "bench"),
            ],
            check=True,
        )


def scenarios(group):
    if group == "basic":
        for n in (100000, 250000, 500000, 1000000, 1500000, 2000000, 4000000, 10000000):
            for divisor in (1, 16, 1024):
                yield "csr", n, divisor, 100, 1, [
                    ("serial", "serial"),
                    ("serial", "parallel"),
                ]
            yield "progression", n, 17, 100, 1, [
                ("serial", "serial"),
                ("parallel", "parallel"),
            ]
            for width in (2, 8):
                yield "gather", n, 1, 100, width, [
                    ("serial", "serial"),
                    ("serial", "parallel"),
                ]
    elif group == "compact":
        for divisor in (93750, 11718, 1464):
            yield "csr", 6000000, divisor, 100, 1, [
                ("serial", "serial"),
                ("serial", "parallel"),
            ]
        yield "csr", 3000000, 256, 100, 1, [
            ("serial", "serial"),
            ("serial", "parallel"),
        ]
    elif group == "tails":
        for n in (6000000, 8000000, 10000000):
            for divisor in (64, 256, 1024):
                yield "csr", n, divisor, 100, 1, [
                    ("serial", "serial"),
                    ("serial", "parallel"),
                ]
            yield "progression", n, 17, 100, 32, [
                ("serial", "serial"),
                ("parallel", "parallel"),
            ]
    elif group == "shape":
        for n in (1000000, 2000000, 4000000):
            for divisor in (1, 16):
                yield "csr", n, divisor, 100, 2, [
                    ("serial", "serial"),
                    ("serial", "parallel"),
                ]
            yield "progression", n, 17, 100, 32, [
                ("serial", "serial"),
                ("parallel", "parallel"),
            ]
        for n in (2000000, 4000000):
            for divisor in (125, 250, 500, 1000):
                yield "csr", n, divisor, 100, 1, [
                    ("serial", "serial"),
                    ("serial", "parallel"),
                ]
    elif group == "recheck":
        for n, spacing, how in (
            (2000000, 1, "inner"),
            (2000000, 4, "semi"),
            (1000000, 10, "inner"),
        ):
            yield "join", n, spacing, 100, 1, [("baseline", how), ("candidate", how)]
    elif group == "validation":
        for n in (250000, 1000000, 2000000, 4000000):
            for spacing in (1, 4, 10):
                for how in ("inner", "full", "semi"):
                    yield "join", n, spacing, 100, 1, [
                        ("baseline", how),
                        ("candidate", how),
                    ]
        for n in (1000000, 1250000, 2000000, 4000000):
            yield "ordered_join", n, 1, 100, 1, [
                ("baseline", "full"),
                ("candidate", "full"),
            ]
        for spacing in (1, 4, 10):
            yield "join", 1250000, spacing, 100, 1, [
                ("baseline", "full"),
                ("candidate", "full"),
            ]
        for spacing, how in ((1, "inner"), (2, "semi")):
            yield "join", 10000000, spacing, 100, 1, [
                ("baseline", how),
                ("candidate", how),
            ]
    elif group == "portability":
        for n in (500000, 1000000, 2000000, 4000000):
            for divisor in (1, 16, 1024):
                yield "csr", n, divisor, 100, 1, [
                    ("serial", "serial"),
                    ("serial", "parallel"),
                ]
            yield "progression", n, 17, 100, 1, [
                ("serial", "serial"),
                ("parallel", "parallel"),
            ]
            for width in (2, 8):
                yield "gather", n, 1, 100, width, [
                    ("serial", "serial"),
                    ("serial", "parallel"),
                ]
            for spacing in (1, 4):
                yield "bounded", n, spacing, 100, 1, [
                    ("serial", "bounded"),
                    ("parallel", "bounded"),
                    ("serial", "hash"),
                ]
    elif group == "fine":
        for n in (625000, 750000, 875000, 1250000, 1750000, 3000000, 6000000):
            for divisor in (1, 16, 256):
                yield "csr", n, divisor, 100, 1, [
                    ("serial", "serial"),
                    ("serial", "parallel"),
                ]
            yield "progression", n, 17, 100, 1, [
                ("serial", "serial"),
                ("parallel", "parallel"),
            ]
        for n in (20000, 50000, 75000, 125000, 375000, 750000):
            for width in (2, 8):
                yield "gather", n, 1, 100, width, [
                    ("serial", "serial"),
                    ("serial", "parallel"),
                ]
        for n in (250000, 750000):
            for spacing in (1, 2, 4):
                yield "bounded", n, spacing, 100, 1, [
                    ("serial", "bounded"),
                    ("parallel", "bounded"),
                    ("serial", "hash"),
                ]
        for spacing in (5, 6, 7, 8):
            yield "membership", 2000000, spacing, 100, 1, [
                ("serial", "bounded"),
                ("serial", "hash"),
            ]
    elif group == "ranges":
        for n in (100000, 500000, 1000000, 2000000, 4000000, 10000000):
            for spacing in (1, 2, 4, 10):
                yield "bounded", n, spacing, 100, 1, [
                    ("serial", "bounded"),
                    ("parallel", "bounded"),
                    ("serial", "hash"),
                ]
                yield "membership", n, spacing, 100, 1, [
                    ("serial", "bounded"),
                    ("serial", "hash"),
                ]
    elif group == "strided":
        for n in (100000, 250000, 500000, 1000000, 2000000, 4000000, 10000000):
            for probe in (25, 50, 100, 200):
                yield "strided", n, 17, probe, 1, [
                    ("serial", "bounded"),
                    ("serial", "hash"),
                ]
    elif group == "ids":
        for n in (100000, 500000, 1000000, 2000000, 4000000):
            for spacing in (1, 2, 4, 10):
                yield "ids", n, spacing, 100, 1, [
                    ("serial", "bounded"),
                    ("hash", "hash"),
                ]


def run(args):
    if args.rounds <= 0 or args.reps <= 0 or args.threads <= 0:
        raise ValueError("rounds, repetitions and threads must be positive")
    path = OUT / f"{args.run}-{args.threads}.csv"
    with path.open("w") as output:
        output.write(
            "round,variant,kind,rows,spacing,probe,width,algorithm,threads,ns\n"
        )
        for kind, n, spacing, probe, width, variants in scenarios(args.run):
            for round_ in range(args.rounds):
                for variant, algorithm in variants[:: 1 if round_ % 2 == 0 else -1]:
                    result = subprocess.run(
                        [
                            str(OUT / variant / "bench"),
                            kind,
                            str(n),
                            str(spacing),
                            str(probe),
                            str(width),
                            algorithm,
                            str(args.reps),
                        ],
                        env={**os.environ, "DATAFRAME_THREADS": str(args.threads)},
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    for line in result.stdout.splitlines():
                        output.write(f"{round_},{variant},{line}\n")
                    output.flush()
            print(kind, n, spacing, probe, width, variants, flush=True)
    print(path)


def report(path):
    """Retain each round's samples and summary in a reviewable CSV."""
    fields = (
        "round",
        "variant",
        "kind",
        "rows",
        "spacing",
        "probe",
        "width",
        "algorithm",
        "threads",
    )
    groups = {}
    for source in sorted(OUT.glob("*.csv")):
        with source.open() as stream:
            for row in csv.DictReader(stream):
                key = (source.stem, *(row[field] for field in fields))
                groups.setdefault(key, []).append(int(row["ns"]))
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("suite", *fields, "min_ns", "median_ns", "max_ns", "samples_ns")
        )
        for key, samples in groups.items():
            writer.writerow(
                (
                    *key,
                    min(samples),
                    int(statistics.median(samples)),
                    max(samples),
                    ";".join(map(str, samples)),
                )
            )
    print(f"{len(groups)} round records written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--variants", default="serial,parallel,hash")
    parser.add_argument(
        "--revision", default="b2013a55ae9c19ce447e2a37f692e7a394eeaa9a"
    )
    parser.add_argument(
        "--run",
        choices=(
            "basic",
            "fine",
            "portability",
            "ranges",
            "strided",
            "ids",
            "shape",
            "tails",
            "compact",
            "validation",
            "recheck",
        ),
    )
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    if args.build:
        build(args.revision, args.variants)
    if args.run:
        run(args)
    if args.report:
        report(args.report)
