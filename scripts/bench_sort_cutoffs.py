#!/usr/bin/env python3
"""Build isolated forced algorithms; alternate order and retain all samples."""
import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import tarfile
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/sort-cutoffs"
REV = "b2013a55ae9c19ce447e2a37f692e7a394eeaa9a"


def build(variants):
    for variant in variants.split(","):
        folder = OUT / variant
        folder.mkdir(parents=True, exist_ok=True)
        include = ROOT if variant == "candidate" else folder
        if variant != "candidate":
            archive = subprocess.check_output(
                ["git", "archive", REV, "dataframe"], cwd=ROOT
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(folder, filter="data")
            path = folder / "dataframe/series.mojo"
            text = path.read_text()
            guard = "and (n <= 200_000 or len(ranks) <= 3)"
            assert guard in text
            if variant == "merge":
                text = text.replace(guard, "and False")
            elif variant == "bucket":
                text = (
                    text.replace(guard, "and True")
                    .replace("if span > 63:", "if span > 4095:")
                    .replace("if occupied < 4 or largest > n // 4:", "if False:")
                )
            elif variant != "baseline":
                raise ValueError(variant)
            path.write_text(text)
        subprocess.run(
            [
                "mojo",
                "build",
                "-I",
                str(include),
                str(ROOT / "benchmarks/bench_sort_cutoffs.mojo"),
                "-o",
                str(folder / "bench"),
            ],
            check=True,
        )


def cases(group):
    if group == "rows":
        for n in (
            8192,
            16384,
            32768,
            65536,
            100000,
            150000,
            200000,
            250000,
            500000,
            1000000,
        ):
            for words in (2, 3, 4):
                for bits in (12, 47):
                    yield n, words, 16, 0, bits
    elif group == "cardinality":
        for n in (100000, 250000, 1000000):
            for words in (2, 3, 4):
                for distinct in (8, 16, 32, 64, 65, 128, 256):
                    yield n, words, distinct, 0, 47
    elif group == "wide":
        for n in (100000, 200000, 250000, 1000000):
            for words in (2, 3, 4):
                for distinct in (16, 64):
                    yield n, words, distinct, 0, 63
    elif group == "validation":
        for n, words, distinct, skew in (
            (8192, 2, 64, 0),
            (65536, 4, 256, 0),
            (100000, 2, 65, 0),
            (100000, 4, 16, 25),
            (100000, 4, 256, 0),
            (200000, 4, 8, 0),
            (250000, 4, 16, 0),
            (250000, 4, 64, 0),
            (500000, 4, 16, 0),
            (1000000, 3, 8, 0),
            (1000000, 3, 16, 0),
            (1000000, 2, 65, 0),
            (1000000, 2, 256, 0),
            (1000000, 2, 257, 0),
            (1000000, 4, 16, 50),
        ):
            yield n, words, distinct, skew, 47
    elif group == "edge":
        for n in (65536, 150000, 200000, 250000, 500000):
            for words in (2, 3, 4):
                for distinct in (256, 257, 512, 1024):
                    yield n, words, distinct, 0, 47
    elif group == "skew":
        for n in (100000, 250000, 1000000):
            for words in (2, 4):
                for distinct in (16, 64, 128):
                    for skew in (20, 25, 30, 50, 80):
                        yield n, words, distinct, skew, 47


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{args.run}-{args.threads}.csv'
    with path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "round",
                "variant",
                "rows",
                "words",
                "distinct",
                "skew",
                "bits",
                "threads",
                "ns",
            )
        )
        for case in cases(args.run):
            print(case, flush=True)
            for round_ in range(args.rounds):
                variants = args.variants.split(",")
                if round_ % 2:
                    variants.reverse()
                for variant in variants:
                    result = subprocess.check_output(
                        [str(OUT / variant / "bench"), *map(str, case), str(args.reps)],
                        env={**os.environ, "DATAFRAME_THREADS": str(args.threads)},
                        text=True,
                    )
                    for line in result.splitlines():
                        writer.writerow(
                            (round_, variant, *case, args.threads, int(line))
                        )
                    stream.flush()
    print(path)


def report(path):
    fields = (
        "round",
        "variant",
        "rows",
        "words",
        "distinct",
        "skew",
        "bits",
        "threads",
    )
    groups = {}
    for suite in (
        "rows-32",
        "cardinality-32",
        "skew-32",
        "edge-32",
        "wide-32",
        "mac-validation-8",
    ):
        with (OUT / (suite + ".csv")).open() as source:
            for row in csv.DictReader(source):
                key = (suite, *(row[field] for field in fields))
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", action="store_true")
    p.add_argument("--report", type=Path)
    p.add_argument("--variants", default="merge,bucket")
    p.add_argument(
        "--run", choices=("rows", "cardinality", "skew", "edge", "validation", "wide")
    )
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()
    if args.build:
        build(args.variants)
    if args.run:
        run(args)

    if args.report:
        report(args.report)
