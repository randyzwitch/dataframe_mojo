"""Deterministic inputs for the join matrix and its Polars/DuckDB comparisons.

`generate` writes `left_ROWS.csv` and `right_ROWS.csv` (and the other fixed
workload tables) into a data directory, `build/bench_polars` by default for
compatibility with existing fixtures; `generate_join_variants` adds the
shuffled and wide right-key layouts. Used by scripts/bench_join_polars.py,
scripts/bench_join_duckdb.py and scripts/bench_join_revision.py inputs.
Moved here from the retired scripts/bench_polars.py (docs/benchmarks.md).
"""
import os
import platform
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260918

# Maps a join key onto a 2^40 range. The multiplier is odd, so the mapping is
# one-to-one and every join keeps its match count; the range is far too wide
# for any direct-address path. The product fits Int64 while the largest raw
# key (0.75 * rows) stays below 1.3e7, which caps the join variants at 16M rows.
WIDE_MULTIPLIER = 0x9E3779B97F
WIDE_MODULUS = 1 << 40
WIDE_MAX_ROWS = 16_000_000


def left_schema() -> dict:
    import polars as pl

    return {
        "key_low": pl.Int64, "key_high": pl.Int64, "key_skew": pl.Int64,
        "key_str": pl.String, "jk": pl.Int64, "x": pl.Float64, "y": pl.Float64, "n": pl.Int64,
    }


def right_schema() -> dict:
    import polars as pl

    return {"jk": pl.Int64, "r": pl.Float64}


def right_wide_schema() -> dict:
    import polars as pl

    return {"jk": pl.Int64, "jk_shift": pl.Int64, "jk_dup": pl.Int64, "r": pl.Float64}


def physical_cores() -> int:
    """Physical cores, matching Mojo's num_physical_cores() as closely as
    the platform allows, so both engines get the same thread count."""
    try:
        if platform.system() == "Linux":
            out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
            per_socket = sockets = None
            for line in out.splitlines():
                if line.startswith("Core(s) per socket:"):
                    per_socket = int(line.split(":")[1])
                elif line.startswith("Socket(s):"):
                    sockets = int(line.split(":")[1])
            if per_socket and sockets:
                return per_socket * sockets
        elif platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True).stdout
            return int(out.strip())
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def generate(data_dir: Path, rows: int) -> None:
    """Write left_ROWS.csv and right_ROWS.csv if they do not already exist."""
    left = data_dir / f"left_{rows}.csv"
    right = data_dir / f"right_{rows}.csv"
    if left.exists() and right.exists():
        return
    import polars as pl

    rng = random.Random(SEED + rows)
    high = max(rows // 10, 1)
    half = max(rows // 2, 1)
    strings = [f"region_{i:03d}" for i in range(100)]
    key_low = [rng.randrange(16) for _ in range(rows)]
    key_high = [rng.randrange(high) for _ in range(rows)]
    # Skewed keys send half the rows to group 0, as bench_suite does.
    key_skew = [0 if i % 2 == 0 else rng.randrange(1000) for i in range(rows)]
    key_str = [strings[rng.randrange(100)] for _ in range(rows)]
    jk = [rng.randrange(half) for _ in range(rows)]
    # x is 10% null (every tenth row), written as an empty field.
    x = [None if i % 10 == 3 else rng.randrange(10000) / 100 - 50 for i in range(rows)]
    y = [rng.randrange(1000) / 10 for _ in range(rows)]
    n = [rng.randrange(1000) - 500 for _ in range(rows)]
    pl.DataFrame(
        {
            "key_low": key_low,
            "key_high": key_high,
            "key_skew": key_skew,
            "key_str": key_str,
            "jk": jk,
            "x": x,
            "y": y,
            "n": n,
        },
        schema={
            "key_low": pl.Int64,
            "key_high": pl.Int64,
            "key_skew": pl.Int64,
            "key_str": pl.String,
            "jk": pl.Int64,
            "x": pl.Float64,
            "y": pl.Float64,
            "n": pl.Int64,
        },
    ).write_csv(left)
    pl.DataFrame(
        {"jk": list(range(half)), "r": [rng.randrange(1000) / 10 for _ in range(half)]},
        schema={"jk": pl.Int64, "r": pl.Float64},
    ).write_csv(right)


def generate_join_variants(data_dir: Path, rows: int) -> None:
    """Write the right inputs whose keys miss the ordered and bounded-range
    join paths, derived from the base files so every join keeps its result.

    right_ROWS_shuffled.csv: the base right rows in random order.
    left_ROWS_wide.csv, right_ROWS_wide.csv: every key mapped onto a 2^40
    range. The unmatched (`jk_shift`) and duplicate (`jk_dup`) keys are mapped
    from the raw key too, because deriving them from the mapped key would
    change which rows match.
    """
    shuffled = data_dir / f"right_{rows}_shuffled.csv"
    left_wide = data_dir / f"left_{rows}_wide.csv"
    right_wide = data_dir / f"right_{rows}_wide.csv"
    if shuffled.exists() and left_wide.exists() and right_wide.exists():
        return
    if rows > WIDE_MAX_ROWS:
        raise ValueError(f"join variants support at most {WIDE_MAX_ROWS:,} rows")
    import polars as pl

    generate(data_dir, rows)
    left = pl.read_csv(data_dir / f"left_{rows}.csv", schema=left_schema())
    right = pl.read_csv(data_dir / f"right_{rows}.csv", schema=right_schema())
    right.sample(fraction=1.0, shuffle=True, seed=SEED + rows).write_csv(shuffled)

    def wide(expr):
        return (expr * WIDE_MULTIPLIER) % WIDE_MODULUS

    left.with_columns(wide(pl.col("jk")).alias("jk")).write_csv(left_wide)
    right.select(
        wide(pl.col("jk")).alias("jk"),
        wide(pl.col("jk") + rows // 4).alias("jk_shift"),
        wide(pl.col("jk") // 2).alias("jk_dup"),
        pl.col("r"),
    ).write_csv(right_wide)


def best_of(fn, reps: int):
    fn()  # warm up
    best = float("inf")
    result = None
    for _ in range(reps):
        start = time.perf_counter_ns()
        result = fn()
        best = min(best, time.perf_counter_ns() - start)
    return best / 1e6, result
