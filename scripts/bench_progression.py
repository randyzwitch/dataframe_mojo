#!/usr/bin/env python3
"""Paired public join measurements against the pre-temporal progression code."""
import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/progression"
REV = "a2b045c"


def build():
    for name in ("baseline", "candidate", "no_runs"):
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        include = ROOT
        if name != "candidate":
            include = folder
            revision = REV
            archive = subprocess.check_output(
                ["git", "archive", revision, "dataframe"], cwd=ROOT
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(folder, filter="data")
            if name == "no_runs":
                path = folder / "dataframe/frame.mojo"
                text = (ROOT / "dataframe/frame.mojo").read_text()
                old = "            elif value == previous:\n                run_length += 1"
                assert old in text
                text = text.replace(
                    old,
                    "            elif value == previous:\n                return (False, List[Int](), List[Int]())\n                run_length += 1",
                    1,
                )
                path.write_text(text)
        subprocess.run(
            [
                "mojo",
                "build",
                "-I",
                str(include),
                str(ROOT / "benchmarks/bench_progression.mojo"),
                "-o",
                str(folder / "bench"),
            ],
            check=True,
        )


def run(threads):
    with (OUT / f'results-{threads}.csv').open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("rows", "shape", "round", "variant", "threads", "ns"))
        for n in (100000, 1000000):
            for shape in ("lookup", "calendar", "panel", "missing", "shuffled"):
                print(n, shape, flush=True)
                for round_ in range(2):
                    names = ["baseline", "candidate", "no_runs"]
                    if round_ % 2:
                        names.reverse()
                    for name in names:
                        result = subprocess.check_output(
                            [str(OUT / name / "bench"), str(n), shape, "5"],
                            env={**os.environ, "DATAFRAME_THREADS": str(threads)},
                            text=True,
                        )
                        for line in result.splitlines():
                            writer.writerow(
                                (n, shape, round_, name, threads, int(line))
                            )
                        stream.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--threads", type=int, default=32)
    args = p.parse_args()
    if args.build:
        build()
    if args.run:
        run(args.threads)
