#!/usr/bin/env python3
"""Compare serial/parallel import and streamed Parquet reads at equal threads."""
import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/import"
BASE = "eafdd9c"


def build():
    for variant in ("baseline", "candidate"):
        folder = OUT / variant
        folder.mkdir(parents=True, exist_ok=True)
        include = ROOT
        if variant == "baseline":
            include = folder
            archive = subprocess.check_output(
                ["git", "archive", BASE, "dataframe"]
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(folder, filter="data")
        for name, bench in [
            ("import", "bench_arrow_import"),
            ("read", "bench_parquet_stream"),
        ]:
            subprocess.run(
                [
                    "mojo",
                    "build",
                    "-I",
                    str(include),
                    str(ROOT / f'benchmarks/{bench}.mojo'),
                    "-o",
                    str(folder / name),
                ],
                check=True,
            )


def compiling():
    return (
        os.environ.get("BENCH_WAIT_FOR_COMPILERS") == "1"
        and subprocess.run(
            ["pgrep", "-x", "mojo"], stdout=subprocess.DEVNULL
        ).returncode
        == 0
    )


def timed(command, env):
    while True:
        while compiling():
            time.sleep(5)
        output = subprocess.check_output(command, env=env, text=True)
        if not compiling():
            return output


def run():
    suffix = "dylib" if os.uname().sysname == "Darwin" else "so"
    env = dict(
        os.environ,
        DATAFRAME_PARQUET_LIBRARY=os.environ.get(
            "DATAFRAME_PARQUET_LIBRARY",
            str(ROOT / f'build/dfparquet/libdfparquet.{suffix}'),
        ),
    )
    with (OUT / "results.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "case",
                "threads",
                "rows",
                "columns",
                "round",
                "variant",
                "sample",
                "ns",
            ]
        )
        for threads in (4, 8, 16):
            env["DATAFRAME_THREADS"] = str(threads)
            for rows in (65536, 125000, 250000, 1000000, 10000000):
                for width in (2, 8):
                    for round_ in range(2):
                        variants = (
                            "baseline",
                            "candidate",
                        ) if round_ == 0 else ("candidate", "baseline")
                        for variant in variants:
                            output = timed(
                                [
                                    str(OUT / variant / "import"),
                                    str(rows),
                                    str(width),
                                    "5",
                                ],
                                env,
                            )
                            for sample, line in enumerate(output.splitlines()):
                                writer.writerow(
                                    [
                                        "import",
                                        threads,
                                        rows,
                                        width,
                                        round_,
                                        variant,
                                        sample,
                                        int(line),
                                    ]
                                )
                            file.flush()
            for rows in (1000000, 10000000):
                fixture = ROOT / f'build/parquet-stream/{rows}.parquet'
                if not fixture.exists():
                    raise RuntimeError(
                        "Generate fixtures with scripts/bench_parquet_stream.py --run first"
                    )
                for round_ in range(3):
                    variants = (
                        "baseline",
                        "candidate",
                    ) if round_ % 2 == 0 else ("candidate", "baseline")
                    for variant in variants:
                        output = timed(
                            [
                                str(OUT / variant / "read"),
                                str(fixture),
                                str(rows),
                            ],
                            env,
                        )
                        writer.writerow(
                            [
                                "read",
                                threads,
                                rows,
                                8,
                                round_,
                                variant,
                                0,
                                int(output.split()[0]),
                            ]
                        )
                        file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.build:
        build()
    if args.run:
        run()
