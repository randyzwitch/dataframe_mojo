"""Fresh-process, paired comparisons of pinned upstream join queries.

Run under the oracle environment. Each worker loads shared CSV inputs before
one warmup and timed queries. All result values are checked after every run.
RSS covers the whole worker, including loading, runtime, and the warmup.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "d8a1bd4f4fbccdf3d22d8f1fe27a94c11c805d0a"
CASES = ("highcardinality", "duplicate_strings")
SOURCES = {
    "highcardinality": "hashjoin_highcardinality.benchmark",
    "duplicate_strings": "hashjoin_dups_rhs.benchmark",
}


def generate(directory, case, rows, layout):
    import polars as pl

    directory.mkdir(parents=True, exist_ok=True)
    left_path = directory / f"{case}-{rows}-{layout}-left.csv"
    right_path = directory / f"{case}-{rows}-{layout}-right.csv"
    if left_path.exists() and right_path.exists():
        return left_path, right_path
    if case == "highcardinality":
        left = pl.DataFrame({"a": pl.int_range(0, 1000, eager=True)})
        right = pl.DataFrame({"b": pl.int_range(0, rows, eager=True)})

        def key(name):
            value = pl.col(name)
            if layout == "wide":
                value = (value * 0x9E3779B97F) % (1 << 40)
            return value.alias("k")

        left = left.select(key("a"), "a")
        right = right.select(key("b"), "b")
    else:

        def table(n):
            return pl.DataFrame({"id": pl.int_range(0, n, eager=True)}).select(
                (
                    pl.lit("verylargestring")
                    + (pl.col("id") % 32768).cast(pl.String)
                ).alias("k")
            )

        left, right = table(131072), table(rows)
    if layout != "base":
        right = right.sample(fraction=1, shuffle=True, seed=20260927)
    left.write_csv(left_path)
    right.write_csv(right_path)
    return left_path, right_path


def worker(args):
    high = args.case == "highcardinality"
    if args.worker == "polars":
        import polars as pl

        schema_l = {"k": pl.Int64, "a": pl.Int64} if high else {"k": pl.String}
        schema_r = {"k": pl.Int64, "b": pl.Int64} if high else {"k": pl.String}
        left = pl.read_csv(args.left, schema=schema_l)
        right = pl.read_csv(args.right, schema=schema_r)
        expected = [(i, i, 1) for i in range(5)] if high else [
            (4 * right.height,)
        ]
        assert pl.thread_pool_size() == args.threads

        def run():
            if high:
                plan = (
                    left.lazy()
                    .join(right.lazy(), on="k")
                    .group_by("a", "b")
                    .agg(pl.len().alias("n"))
                    .sort("a")
                    .head(5)
                    .select("a", "b", "n")
                )
            else:
                plan = (
                    left.lazy()
                    .join(right.lazy(), on="k")
                    .select(pl.len().alias("n"))
                )
            return plan.collect()

        def values(result):
            return result.rows()

    else:
        import duckdb

        con = duckdb.connect(config={"threads": args.threads})
        lt = "{'k':'BIGINT','a':'BIGINT'}" if high else "{'k':'VARCHAR'}"
        rt = "{'k':'BIGINT','b':'BIGINT'}" if high else "{'k':'VARCHAR'}"
        con.execute(
            f"CREATE TABLE l AS SELECT * FROM read_csv(?, header=true, columns={lt})",
            [args.left],
        )
        con.execute(
            f"CREATE TABLE r AS SELECT * FROM read_csv(?, header=true, columns={rt})",
            [args.right],
        )
        expected = [(i, i, 1) for i in range(5)] if high else [
            (4 * con.execute("SELECT count(*) FROM r").fetchone()[0],)
        ]
        sql = (
            "SELECT a, b, count(*) AS n FROM l JOIN r USING(k) "
            "GROUP BY a,b ORDER BY a LIMIT 5"
        ) if high else ("SELECT count(*) AS n FROM l JOIN r USING(k)")

        def run():
            return con.execute(sql).fetchall()

        def values(result):
            return result

    for rep in range(args.reps + 1):
        start = time.perf_counter_ns()
        result = run()
        elapsed = time.perf_counter_ns() - start
        assert values(result) == expected, (args.case, values(result), expected)
        del result
        if rep:
            print(elapsed, flush=True)


def compiler_processes():
    output = subprocess.check_output(["ps", "-eo", "pid,comm,args"], text=True)
    found = []
    for line in output.splitlines()[1:]:
        fields = line.split(None, 2)
        if len(fields) < 3:
            continue
        name = Path(fields[1]).name
        # `mojo run`, `test` and `package` compile too, and every one of
        # them competes with a timed run for the same cores.
        if name in {"clang", "clang++", "cc1", "cc1plus", "rustc"} or (
            name == "mojo"
            and any(
                f" {verb} " in fields[2]
                for verb in ("build", "run", "test", "package", "precompile")
            )
        ):
            found.append((fields[0], name))
    return found


def measure(command, env, timeout, allow_busy=False):
    if not allow_busy and compiler_processes():
        raise RuntimeError("compiler activity detected; wait for a quiet host")
    with tempfile.TemporaryFile(
        mode="w+"
    ) as output, tempfile.NamedTemporaryFile(mode="r+") as rss_file:
        # GNU time launches the actual worker from a small fresh executable.
        # This avoids Linux accounting for the large Python parent's RSS
        # between fork and exec when interpreting wait4's high-water mark.
        linux = platform.system() == "Linux"
        launch = [
            "/usr/bin/time",
            "-f",
            "%M",
            "-o",
            rss_file.name,
            *command,
        ] if linux else command
        proc = subprocess.Popen(
            launch,
            env=env,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        while True:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                proc.returncode = os.waitstatus_to_exitcode(status)
                break
            if time.monotonic() >= deadline:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise TimeoutError(f"worker exceeded {timeout}s: {command}")
            time.sleep(0.05)
        output.seek(0)
        text = output.read()
        if proc.returncode:
            raise RuntimeError(f"worker failed ({proc.returncode}): {text}")
        rss_file.seek(0)
        rss = (
            float(rss_file.read().strip())
            / 1024 if linux else usage.ru_maxrss
            / 1024**2
        )
    if not allow_busy and compiler_processes():
        raise RuntimeError(
            "compiler activity detected; discard interrupted run"
        )
    samples = [int(line) for line in text.splitlines()]
    return samples, rss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-busy",
        action="store_true",
        help="correctness smoke only; allow compiler activity",
    )
    parser.add_argument("--sizes", default="1000000,10000000")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--layouts", default="base,shuffled,wide")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/upstream-joins/results.csv"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "build/upstream-joins/data"
    )
    parser.add_argument(
        "--runner", type=Path, default=ROOT / "build/bench_upstream_joins"
    )
    parser.add_argument("--worker", choices=("polars", "duckdb"))
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--left")
    parser.add_argument("--right")
    args = parser.parse_args()
    if min(args.threads, args.reps, args.rounds, args.timeout) < 1:
        parser.error("threads, reps, rounds and timeout must be positive")
    if args.worker:
        worker(args)
        return
    sizes = [int(n) for n in args.sizes.split(",")]
    cases, layouts = args.cases.split(","), args.layouts.split(",")
    if any(n < 1000 or n > 13000000 for n in sizes):
        parser.error("sizes must be between 1000 and 13000000")
    if set(cases) - set(CASES) or set(layouts) - {"base", "shuffled", "wide"}:
        parser.error("unknown case or layout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import polars as pl
    import duckdb

    metadata = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__).resolve(),
                ROOT / "benchmarks/bench_upstream_joins.mojo",
                ROOT / "pixi.lock",
            ]
        },
        "runner_sha256": hashlib.sha256(args.runner.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(
            ["mojo", "--version"], text=True
        ).strip(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu": subprocess.check_output(
            ["lscpu"] if platform.system()
            == "Linux" else [
                "sysctl",
                "hw.model",
                "hw.physicalcpu",
                "hw.logicalcpu",
                "hw.memsize",
            ],
            text=True,
        ),
        "polars": pl.__version__,
        "duckdb": duckdb.__version__,
        "upstream": UPSTREAM,
        "sources": SOURCES,
        "command": sys.argv,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {},
        "status": "incomplete",
    }

    def save_metadata():
        args.output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

    save_metadata()
    with args.output.open("w") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "case",
                "layout",
                "rows",
                "threads",
                "round",
                "engine",
                "sample",
                "ns",
                "peak_rss_mib",
            ]
        )
        for rows in sizes:
            for case in cases:
                for layout in layouts:
                    if case == "duplicate_strings" and layout == "wide":
                        continue
                    left, right = generate(args.data_dir, case, rows, layout)
                    for path in (left, right):
                        metadata["inputs"][path.name] = hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                    save_metadata()
                    for round_no in range(args.rounds):
                        engines = ["mojo", "polars", "duckdb"]
                        # Rotate first engine; alternate direction across triples.
                        offset = round_no % 3
                        engines = engines[offset:] + engines[:offset]
                        if (round_no // 3) % 2:
                            engines.reverse()
                        for engine in engines:
                            env = dict(
                                os.environ,
                                DATAFRAME_THREADS=str(args.threads),
                                POLARS_MAX_THREADS=str(args.threads),
                            )
                            command = [
                                str(args.runner),
                                case,
                                str(left),
                                str(right),
                                str(args.reps),
                            ] if engine == "mojo" else [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--worker",
                                engine,
                                "--case",
                                case,
                                "--left",
                                str(left),
                                "--right",
                                str(right),
                                "--reps",
                                str(args.reps),
                                "--threads",
                                str(args.threads),
                            ]
                            samples, rss = measure(
                                command, env, args.timeout, args.allow_busy
                            )
                            assert len(samples) == args.reps
                            for i, ns in enumerate(samples):
                                writer.writerow(
                                    [
                                        case,
                                        layout,
                                        rows,
                                        args.threads,
                                        round_no,
                                        engine,
                                        i,
                                        ns,
                                        rss,
                                    ]
                                )
                            file.flush()
                            print(
                                case,
                                layout,
                                rows,
                                round_no,
                                engine,
                                f"{statistics.median(samples)/1e6:.3f} ms",
                                f"{rss:.1f} MiB",
                                flush=True,
                            )
    metadata["status"] = "complete"
    save_metadata()


if __name__ == "__main__":
    main()
