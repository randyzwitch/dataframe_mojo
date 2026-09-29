"""Generate or download the inputs for the external benchmark suites.

Every engine reads the same Parquet files, written once into the data root
(default: a `dataframe_mojo_benchdata` folder next to this repository, or
$DATAFRAME_BENCH_DATA). Existing files are reused, so nothing is generated
or downloaded twice. Run under the oracle environment (Polars and DuckDB).

Suites and their provenance:

- h2o_groupby, h2o_join: the H2O.ai / DuckDB "db-benchmark" data, following
  `_data/groupby-datagen.R` and `_data/join-datagen.R` in
  https://github.com/duckdblabs/db-benchmark. The R scripts' `sample` and
  `runif` calls are reproduced with seeded Polars sampling and hashing, so
  the distributions match but the exact values do not.
- pdsh: TPC-H tables from DuckDB's `dbgen`, as used by Polars' PDS-H
  benchmark (https://github.com/pola-rs/polars-benchmark), in two variants:
  `base` stores the DECIMAL(15,2) money columns as DOUBLE, as the suite did
  before this library had decimals (#229), and `decimal` keeps them as
  DECIMAL(15,2). Every engine reads the same files in each variant.
- clickbench: the `hits` table from https://github.com/ClickHouse/ClickBench,
  downloaded in 1M-row partitions. Its Parquet files store text as untyped
  byte arrays and times as integers; like ClickBench's own Polars and DuckDB
  scripts, they are converted once to strings, timestamps and dates.
"""

import argparse
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CLICKBENCH_URL = (
    "https://datasets.clickhouse.com/hits_compatible/athena_partitioned/"
    "hits_{}.parquet"
)


def data_root() -> Path:
    """$DATAFRAME_BENCH_DATA, else `<main checkout>_benchdata` beside the main
    checkout, so every git worktree shares one copy of the data."""
    configured = os.environ.get("DATAFRAME_BENCH_DATA")
    if configured:
        return Path(configured)
    main = ROOT
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        main = Path(common).parent
    except (OSError, subprocess.CalledProcessError):
        pass
    return main.parent / f"{main.name}_benchdata"


def _uniform(n: int, seed: int, scale: int):
    """Deterministic uniform integers in [0, scale) from a hashed row index."""
    import polars as pl

    return (pl.int_range(0, n, eager=True).hash(seed) % scale).cast(pl.Int64)


def _sample(values, n: int, seed: int):
    """R's sample(values, n, replace=TRUE): uniform draws with replacement."""
    import polars as pl

    values = pl.Series(values)
    return values.gather(_uniform(n, seed, len(values)))


def _with_nas(frame, pct: int, seed: int, columns):
    """Set about pct percent of each listed column to null (R's set(NA))."""
    import polars as pl

    if pct == 0:
        return frame
    out = []
    for i, name in enumerate(columns):
        hit = _uniform(frame.height, seed + 1000 + i, 100) < pct
        out.append(pl.when(hit).then(None).otherwise(pl.col(name)).alias(name))
    return frame.with_columns(out)


def h2o_groupby(rows: int, k: int, nas: int, sort: bool) -> Path:
    """groupby-datagen.R: 3 string keys, 3 integer keys, 3 values."""
    import polars as pl

    root = data_root() / "h2o"
    path = root / f"G1_{rows:.0e}_{k:.0e}_{nas}_{int(sort)}.parquet".replace(
        "+", ""
    )
    if path.exists():
        return path
    root.mkdir(parents=True, exist_ok=True)
    groups = max(1, rows // k)
    frame = pl.DataFrame(
        {
            "id1": _sample([f"id{i:03d}" for i in range(1, k + 1)], rows, 1),
            "id2": _sample([f"id{i:03d}" for i in range(1, k + 1)], rows, 2),
            "id3": _sample(
                [f"id{i:010d}" for i in range(1, groups + 1)], rows, 3
            ),
            "id4": _uniform(rows, 4, k) + 1,
            "id5": _uniform(rows, 5, k) + 1,
            "id6": _uniform(rows, 6, groups) + 1,
            "v1": _uniform(rows, 7, 5) + 1,
            "v2": _uniform(rows, 8, 15) + 1,
            # round(runif(N, max=100), 6)
            "v3": _uniform(rows, 9, 100_000_000).cast(pl.Float64) / 1e6,
        }
    )
    frame = _with_nas(frame, nas, 10, frame.columns)
    if sort:
        frame = frame.sort(["id1", "id2", "id3", "id4", "id5", "id6"])
    frame.write_parquet(path, row_group_size=1 << 20)
    return path


def _split_xlr(n: int, seed: int):
    """join-datagen.R's split_xlr: each side gets n distinct keys, of which
    90% are shared (x) and 10% occur only on that side (l or r)."""
    import polars as pl

    shared = int(n * 0.9)
    only = n - shared
    pool = pl.int_range(1, int(n * 2.5) + 1, eager=True).sample(
        shared + 2 * only, seed=seed
    )
    x = pool[:shared]
    left = pool[shared : shared + only]
    right = pool[shared + only :]
    return pl.concat([x, left]), pl.concat([x, right])


def h2o_join(rows: int, nas: int) -> dict:
    """join-datagen.R: a left table and small/medium/big right tables."""
    import polars as pl

    root = data_root() / "h2o"
    tag = f"{rows:.0e}_NA_0_{nas}".replace("+", "")
    names = {
        "x": f"J1_{tag}_0_0.parquet",
        "small": f"J1_{tag}_{max(10, rows // 1_000_000):.0e}_0.parquet".replace(
            "+", ""
        ),
        "medium": f"J1_{tag}_{max(100, rows // 1_000):.0e}_0.parquet".replace(
            "+", ""
        ),
        "big": f"J1_{tag}_{rows:.0e}_0.parquet".replace("+", ""),
    }
    paths = {key: root / name for key, name in names.items()}
    if all(path.exists() for path in paths.values()):
        return paths
    root.mkdir(parents=True, exist_ok=True)
    # The R script targets 1e7+ rows. Floors keep tiny smoke-test inputs
    # valid (a 90/10 key split needs at least ten keys); full sizes are
    # unaffected.
    n1 = max(10, rows // 1_000_000)
    n2 = max(100, rows // 1_000)
    n3 = rows
    left1, right1 = _split_xlr(n1, 11)
    left2, right2 = _split_xlr(n2, 12)
    left3, right3 = _split_xlr(n3, 13)

    def labels(ids):
        return pl.format("id{}", ids)

    x = pl.DataFrame(
        {
            "id1": _sample(left1, rows, 21),
            "id2": _sample(left2, rows, 22),
            "id3": left3.sample(rows, seed=23),
        }
    ).with_columns(
        labels(pl.col("id1")).alias("id4"),
        labels(pl.col("id2")).alias("id5"),
        labels(pl.col("id3")).alias("id6"),
        (_uniform(rows, 24, 100_000_000).cast(pl.Float64) / 1e6).alias("v1"),
    )
    small = pl.DataFrame({"id1": right1.sample(n1, seed=31)}).with_columns(
        labels(pl.col("id1")).alias("id4"),
        (_uniform(n1, 32, 100_000_000).cast(pl.Float64) / 1e6).alias("v2"),
    )
    medium = pl.DataFrame(
        {"id1": _sample(right1, n2, 41), "id2": right2.sample(n2, seed=42)}
    ).with_columns(
        labels(pl.col("id1")).alias("id4"),
        labels(pl.col("id2")).alias("id5"),
        (_uniform(n2, 43, 100_000_000).cast(pl.Float64) / 1e6).alias("v2"),
    )
    big = pl.DataFrame(
        {
            "id1": _sample(right1, n3, 51),
            "id2": _sample(right2, n3, 52),
            "id3": right3.sample(n3, seed=53),
        }
    ).with_columns(
        labels(pl.col("id1")).alias("id4"),
        labels(pl.col("id2")).alias("id5"),
        labels(pl.col("id3")).alias("id6"),
        (_uniform(n3, 54, 100_000_000).cast(pl.Float64) / 1e6).alias("v2"),
    )
    x = _with_nas(x, nas, 60, x.columns)
    for key, frame in [("x", x), ("small", small), ("medium", medium), ("big", big)]:
        frame.write_parquet(paths[key], row_group_size=1 << 20)
    return paths


PDSH_TABLES = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]


def pdsh(scale: float, decimal: bool = False) -> Path:
    """TPC-H tables from DuckDB dbgen, with DECIMAL columns as DOUBLE, or
    kept as DECIMAL(15,2) when `decimal`."""
    import duckdb

    root = data_root() / "pdsh" / (
        f"sf{scale:g}_decimal" if decimal else f"sf{scale:g}"
    )
    if all((root / f"{t}.parquet").exists() for t in PDSH_TABLES):
        return root
    root.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={scale})")
    for table in PDSH_TABLES:
        columns = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
        select = ", ".join(
            f"CAST({name} AS DOUBLE) AS {name}"
            if kind.startswith("DECIMAL") and not decimal
            else f"CAST({name} AS BIGINT) AS {name}"
            if kind == "INTEGER"
            else name
            for name, kind in columns
        )
        con.execute(
            f"COPY (SELECT {select} FROM {table}) TO '{root / table}.parquet' "
            "(FORMAT parquet, ROW_GROUP_SIZE 1048576)"
        )
    return root


def clickbench(partitions: int) -> Path:
    """The first `partitions` 1M-row slices of ClickBench's hits table."""
    import duckdb

    root = data_root() / "clickbench"
    path = root / f"hits_{partitions}.parquet"
    if path.exists():
        return path
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(partitions):
        target = raw / f"hits_{i}.parquet"
        if not target.exists():
            partial = target.with_suffix(".part")
            request = urllib.request.Request(
                CLICKBENCH_URL.format(i),
                headers={"User-Agent": "dataframe_mojo-benchmarks"},
            )
            with urllib.request.urlopen(request) as response, open(
                partial, "wb"
            ) as out:
                while block := response.read(1 << 20):
                    out.write(block)
            partial.rename(target)
        files.append(str(target))
    con = duckdb.connect()
    listing = ", ".join(f"'{f}'" for f in files)
    source = f"read_parquet([{listing}], binary_as_string=true)"
    columns = con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
    select = []
    for name, kind, *_ in columns:
        if name in ("EventTime", "ClientEventTime", "LocalEventTime"):
            select.append(f"CAST(to_timestamp({name}) AS TIMESTAMP) AS {name}")
        elif name == "EventDate":
            select.append(f"CAST(DATE '1970-01-01' + {name} AS DATE) AS {name}")
        else:
            select.append(f'"{name}"')
    con.execute(
        f"COPY (SELECT {', '.join(select)} FROM {source}) TO '{path}' "
        "(FORMAT parquet, ROW_GROUP_SIZE 1048576)"
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=["h2o", "pdsh", "clickbench"])
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--nas", type=int, default=0)
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--partitions", type=int, default=10)
    args = parser.parse_args()
    if args.suite == "h2o":
        print(h2o_groupby(args.rows, args.k, args.nas, args.sort))
        print(h2o_join(args.rows, args.nas))
    elif args.suite == "pdsh":
        print(pdsh(args.scale))
    else:
        print(clickbench(args.partitions))


if __name__ == "__main__":
    main()
