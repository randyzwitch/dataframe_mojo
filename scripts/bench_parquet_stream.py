#!/usr/bin/env python3
"""Paired whole-file/streaming eager reads with separate-process peak RSS."""
import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/parquet-stream"
REV = "b2013a55ae9c19ce447e2a37f692e7a394eeaa9a"


def build():
    for variant in ("baseline", "candidate"):
        folder = OUT / variant
        folder.mkdir(parents=True, exist_ok=True)
        include = ROOT
        if variant == "baseline":
            include = folder
            archive = subprocess.check_output(
                ["git", "archive", REV, "dataframe"], cwd=ROOT
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(folder, filter="data")
        subprocess.run(
            [
                "mojo",
                "build",
                "-I",
                str(include),
                str(ROOT / "benchmarks/bench_parquet_stream.mojo"),
                "-o",
                str(folder / "bench"),
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


def generate(rows, path):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    schema = pa.schema([(f"c{i}", pa.int64()) for i in range(8)])
    with pq.ParquetWriter(
        path, schema, compression="zstd", use_dictionary=False
    ) as writer:
        for start in range(0, rows, 250000):
            base = pa.array(
                range(start, min(rows, start + 250000)), type=pa.int64()
            )
            table = pa.table({f"c{i}": pc.add(base, i) for i in range(8)})
            writer.write_table(table, row_group_size=250000)


def run():
    suffix = "dylib" if sys.platform == "darwin" else "so"
    library = ROOT / f'build/dfparquet/libdfparquet.{suffix}'
    with (OUT / "results.csv").open("w") as report:
        writer = csv.writer(report)
        writer.writerow(
            ("rows", "round", "variant", "ns", "chunks", "peak_rss_bytes")
        )
        for rows in (1000000, 10000000):
            path = OUT / f'{rows}.parquet'
            # Generate in a separate process, so fixture allocations cannot
            # inflate the measured child's inherited/pre-exec RSS.
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--generate",
                    str(rows),
                    "--path",
                    str(path),
                ],
                check=True,
            )
            for round_ in range(3):
                variants = ["baseline", "candidate"]
                if round_ % 2:
                    variants.reverse()
                for variant in variants:
                    while True:
                        while compiling():
                            time.sleep(5)
                        with (OUT / "process.log").open("w+") as output:
                            child = subprocess.Popen(
                                [
                                    str(OUT / variant / "bench"),
                                    str(path),
                                    str(rows),
                                ],
                                stdout=output,
                                env={
                                    **os.environ,
                                    "DATAFRAME_PARQUET_LIBRARY": str(library),
                                    "DATAFRAME_THREADS": "8",
                                },
                            )
                            _, status, usage = os.wait4(child.pid, 0)
                            child.returncode = os.waitstatus_to_exitcode(status)
                            if child.returncode:
                                raise RuntimeError(
                                    f'{variant} exited {child.returncode}'
                                )
                            output.seek(0)
                            ns, chunks = map(int, output.read().split())
                        rss = int(usage.ru_maxrss) * (
                            1 if sys.platform == "darwin" else 1024
                        )
                        if not compiling():
                            break
                        print(
                            "Discarding read interrupted by a Mojo process",
                            flush=True,
                        )
                    writer.writerow((rows, round_, variant, ns, chunks, rss))
                    report.flush()
                    print(rows, round_, variant, ns, chunks, rss, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--generate", type=int)
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()
    if args.build:
        build()
    if args.run:
        run()

    if args.generate:
        generate(args.generate, args.path)
