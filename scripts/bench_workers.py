#!/usr/bin/env python3
"""Cross-machine worker-policy calibration; experiments never patch production.

The experiment copies read tuning knobs once per worker/stage, outside hot
loops. Compare values within the same binary. Overall/join scaling uses the
unmodified production binaries. All runs alternate order and retain samples.
"""
import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import tarfile
import time
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/workers"
REV = "a2b045c"


def build(targets):
    folder = OUT / "experiment"
    folder.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(
        ["git", "archive", REV, "dataframe"], cwd=ROOT
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(folder, filter="data")
    pristine = OUT / "baseline"
    pristine.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(pristine, filter="data")
    path = folder / "dataframe/parallel.mojo"
    text = path.read_text()
    marker = "\ntrait Job("
    helper = """
def _bench_setting(name: String, default: Int) -> Int:
    try:
        return max(0, Int(getenv(name)))
    except:
        return default

"""
    text = text.replace(marker, helper + marker, 1)
    text = text.replace(
        "    var seen = Int64(0)",
        '    var spin_limit = _bench_setting("BENCH_SPIN", _SPIN_LIMIT)\n    var seen = Int64(0)',
        1,
    )
    text = text.replace("spins < _SPIN_LIMIT", "spins < spin_limit")
    text = text.replace(
        "rows // MIN_ROWS_PER_WORKER",
        'rows // max(1, _bench_setting("BENCH_MIN_ROWS", MIN_ROWS_PER_WORKER))',
    )
    path.write_text(text)
    path = folder / "dataframe/frame.mojo"
    text = path.read_text().replace(
        "from .parallel import Job,",
        "from .parallel import _bench_setting, Job,",
        1,
    )
    old = "return min(16, worker_count(rows))"
    assert old in text
    text = text.replace(
        old,
        'return min(max(1, _bench_setting("BENCH_BUILDERS", 16)), worker_count(rows))',
    )
    path.write_text(text)
    source = (ROOT / "benchmarks/bench_join_cutoffs.mojo").read_text()
    source = source.replace(
        "from std.sys import argv",
        "from std.sys import argv\nfrom std.os import getenv",
    )
    source = source.replace(
        "min(4, max(1, configured_workers() // len(columns)))",
        'min(Int(getenv("BENCH_PARTS", "4")), max(1, configured_workers() // len(columns)))',
    )
    source = source.replace("range(32)", "range(spacing)").replace(
        "// 32", "// spacing"
    )
    source = source.replace(
        "    _SortedChunkTakeJob,",
        "    take_parallel,\n    _SortedChunkTakeJob,",
    )
    old = 'var result = gather_columns(\n                columns, indices, algorithm == "parallel"\n            )'
    new = 'var result = take_parallel(columns.copy(), indices.copy(), worker_count(n), or_null=False) if algorithm == "general" else gather_columns(\n                columns, indices, algorithm == "parallel"\n            )'
    assert old in source
    source = source.replace(old, new)
    (folder / "cutoffs.mojo").write_text(source)
    for label, source_path, include in (
        ("overall", ROOT / "benchmarks/bench_suite.mojo", pristine),
        ("joins", ROOT / "benchmarks/bench_join.mojo", pristine),
        ("cutoffs", folder / "cutoffs.mojo", folder),
        ("sort", ROOT / "benchmarks/bench_worker_sort.mojo", folder),
        ("sort-baseline", ROOT / "benchmarks/bench_worker_sort.mojo", pristine),
        ("sort-candidate", ROOT / "benchmarks/bench_worker_sort.mojo", ROOT),
        ("overall-experiment", ROOT / "benchmarks/bench_suite.mojo", folder),
    ):
        if targets and label not in targets.split(","):
            continue
        target = OUT / label
        # Rebuild all experiment binaries, reuse production binaries if present.
        if label in ("overall", "joins") and target.exists():
            continue
        subprocess.run(
            [
                "mojo",
                "build",
                "-I",
                str(include),
                str(source_path),
                "-o",
                str(target),
            ],
            check=True,
        )


def timed(binary, args, env):
    # Shared development hosts may start a compiler between cases. Waiting
    # and discarding an interrupted case keeps those samples out of the CSV.
    def compiling():
        return (
            os.environ.get("BENCH_WAIT_FOR_COMPILERS") == "1"
            and subprocess.run(
                ["pgrep", "-x", "mojo"], stdout=subprocess.DEVNULL
            ).returncode
            == 0
        )

    while True:
        while compiling():
            time.sleep(5)
        result = subprocess.check_output(
            [str(OUT / binary), *map(str, args)],
            env={**os.environ, **env},
            text=True,
        )
        if not compiling():
            return result
        print("Discarding case interrupted by a Mojo process", flush=True)


def scaling():
    for round_ in range(2):
        counts = [4, 8, 16]
        if round_ % 2:
            counts.reverse()
        for threads in counts:
            for binary in ("overall", "joins"):
                path = OUT / f'{binary}-{threads}-{round_}.csv'
                print(path.name, flush=True)
                path.write_text(
                    timed(binary, [], {"DATAFRAME_THREADS": str(threads)})
                )


def experiments(threads, group):
    path = OUT / f'{group}-{threads}.csv'
    with path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case",
                "rows",
                "width",
                "chunks",
                "round",
                "setting",
                "threads",
                "ns",
            )
        )
        if group == "spin":
            scenarios = [
                ("spin", n, w, 0) for n in (100000, 1000000) for w in (2, 4)
            ]
            values = [0, 200000, 1000000, 2000000]
        elif group == "builders":
            scenarios = [("builders", n, 1, 0) for n in (1000000, 4000000)]
            values = [4, 8, 16]
        elif group == "gather":
            scenarios = [
                ("gather", n, w, c)
                for n in (250000, 1000000)
                for w in (2, 8)
                for c in (4, 8, 16, 32)
            ]
            values = [0, 1, 2, 4, 8]
        else:
            raise ValueError(group)
        for case, n, width, chunks in scenarios:
            for round_ in range(2):
                for value in values[:: (-1 if round_ % 2 else 1)]:
                    env = {"DATAFRAME_THREADS": str(threads)}
                    if group == "spin":
                        env["BENCH_SPIN"] = str(value)
                        output = timed("sort", [n, width, 65536, 0, 47, 5], env)
                    elif group == "builders":
                        env["BENCH_BUILDERS"] = str(value)
                        output = timed(
                            "cutoffs",
                            ["bounded", n, 1, 100, 1, "bounded", 5],
                            env,
                        )
                    else:
                        env["BENCH_PARTS"] = str(value)
                        output = timed(
                            "cutoffs",
                            [
                                "gather",
                                n,
                                chunks,
                                100,
                                width,
                                "general" if value
                                == 0 else "serial" if value
                                == 1 else "parallel",
                                5,
                            ],
                            env,
                        )
                    for line in output.splitlines():
                        writer.writerow(
                            (
                                case,
                                n,
                                width,
                                chunks,
                                round_,
                                value,
                                threads,
                                int(line.rsplit(",", 1)[-1]),
                            )
                        )
                    stream.flush()
            print(case, n, width, chunks, flush=True)


def min_rows(threads):
    for round_ in range(2):
        values = [16384, 65536, 262144]
        if round_ % 2:
            values.reverse()
        for value in values:
            path = OUT / f'minrows-{threads}-{value}-{round_}.csv'
            print(path.name, flush=True)
            path.write_text(
                timed(
                    "overall-experiment",
                    [],
                    {
                        "DATAFRAME_THREADS": str(threads),
                        "BENCH_MIN_ROWS": str(value),
                    },
                )
            )


def sort_policy():
    with (OUT / "sort-policy.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(("rows", "width", "round", "variant", "threads", "ns"))
        for threads in (4, 8, 16):
            for rows in (100000, 1000000):
                for width in (2, 4):
                    for round_ in range(2):
                        variants = ["baseline", "candidate"]
                        if round_ % 2:
                            variants.reverse()
                        for variant in variants:
                            output = timed(
                                "sort-" + variant,
                                [rows, width, 65536, 0, 47, 5],
                                {"DATAFRAME_THREADS": str(threads)},
                            )
                            for line in output.splitlines():
                                writer.writerow(
                                    (
                                        rows,
                                        width,
                                        round_,
                                        variant,
                                        threads,
                                        int(line),
                                    )
                                )
                            stream.flush()
                    print(threads, rows, width, flush=True)


def report(destination):
    records = []
    samples = {}
    for machine, folder in (("linux", OUT), ("mac", OUT / "mac")):
        for path in sorted(folder.glob("*.csv")):
            with path.open() as stream:
                for row in csv.DictReader(
                    line for line in stream if not line.startswith("#")
                ):
                    row = {"machine": machine, "source": path.name, **row}
                    if "ns" in row:
                        value = int(row.pop("ns"))
                        key = tuple(sorted(row.items()))
                        samples.setdefault(key, []).append(value)
                    else:
                        records.append(row)
    for key, values in samples.items():
        records.append(
            {
                **dict(key),
                "best_ns": min(values),
                "median_ns": int(statistics.median(values)),
                "max_ns": max(values),
                "samples_ns": ";".join(map(str, values)),
            }
        )
    fields = ["machine", "source"] + sorted(
        set().union(*(r.keys() for r in records)) - {"machine", "source"}
    )
    with Path(destination).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", action="store_true")
    p.add_argument("--report", type=Path)
    p.add_argument("--targets", default="")
    p.add_argument("--scaling", action="store_true")
    p.add_argument("--sort-policy", action="store_true")
    p.add_argument("--run", choices=("spin", "builders", "gather", "minrows"))
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    if args.build:
        build(args.targets)
    if args.scaling:
        scaling()
    if args.sort_policy:
        sort_policy()
    if args.run == "minrows":
        min_rows(args.threads)
    elif args.run:
        experiments(args.threads, args.run)

    if args.report:
        report(args.report)
