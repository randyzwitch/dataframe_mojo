"""Compare representative joins with Polars at one million rows or more.

Inputs are constructed once outside timing. The two engines run the same
cardinality, key type, join mode, and projected-output cases. Each result is
checked by row count and a numeric column total before times are compared.

Every case runs on three key layouts. `base` is the generated data, whose
Int64 right keys are a sorted `range(n)` that the ordered-key and
direct-address join paths recognise. `shuffled` reorders the right rows, and
`wide` maps every key onto a 2^40 range; neither changes a join's result, and
neither qualifies for those paths, so they are the general-case numbers.

    pixi run -e oracle bench-joins-polars --sizes 1000000
    pixi run -e oracle bench-joins-polars --sizes 10000000 --threads 32
    pixi run -e oracle bench-joins-polars --variants base   # ordered keys only
"""
import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

from bench_polars import (
    ROOT,
    best_of,
    generate,
    generate_join_variants,
    left_schema,
    physical_cores,
    right_schema,
    right_wide_schema,
)

CASES = [
    "inner_dense", "lazy_narrow", "inner_sparse", "inner_string",
    "inner_multi", "left_unmatched", "right_unmatched", "full_unmatched",
    "semi_unmatched", "anti_unmatched", "left_sparse", "right_sparse", "inner_duplicate",
]
VARIANTS = ["base", "shuffled", "wide"]


def input_files(rows: int, variant: str) -> tuple[str, str]:
    """The left and right CSV names for one key layout."""
    left = f"left_{rows}_wide.csv" if variant == "wide" else f"left_{rows}.csv"
    right = f"right_{rows}.csv" if variant == "base" else f"right_{rows}_{variant}.csv"
    return left, right


def build_runner() -> Path:
    source = ROOT / "benchmarks" / "bench_join_matrix.mojo"
    runner = ROOT / "build" / "bench_join_matrix"
    newest = max(p.stat().st_mtime for p in [source, *(ROOT / "dataframe").rglob("*.mojo")])
    if not runner.exists() or runner.stat().st_mtime < newest:
        subprocess.run(
            ["mojo", "build", "-I", str(ROOT), str(source), "-o", str(runner)],
            check=True,
        )
    return runner


def run_mojo(
    runner: Path, data_dir: Path, rows: int, reps: int, threads: int, variant: str = "base"
) -> dict:
    env = dict(os.environ, DATAFRAME_THREADS=str(threads))
    output = subprocess.run(
        [str(runner), str(data_dir), str(rows), str(reps), "--variant", variant],
        capture_output=True, text=True, env=env, check=True,
    ).stdout
    results = {}
    for line in output.splitlines():
        name, ns, height, value = line.split("\t")
        results[name] = (int(ns) / 1e6, int(height), float(value))
    missing = set(CASES) - results.keys()
    if missing:
        raise RuntimeError(f"Mojo benchmark missing cases {sorted(missing)}")
    return results


def run_polars(data_dir: Path, rows: int, reps: int, variant: str = "base") -> dict:
    import polars as pl

    wide = variant == "wide"
    left_name, right_name = input_files(rows, variant)
    left = pl.read_csv(data_dir / left_name, schema=left_schema())
    source = pl.read_csv(
        data_dir / right_name, schema=right_wide_schema() if wide else right_schema()
    )
    right = source.select("jk", "r") if wide else source
    # The wide files carry these keys precomputed from the raw key; see
    # generate_join_variants.
    if wide:
        shifted = source.select(pl.col("jk_shift").alias("jk"), "r")
        duplicate_right = source.select(pl.col("jk_dup").alias("jk"), "r")
    else:
        shifted = right.with_columns((pl.col("jk") + rows // 4).alias("jk"))
        duplicate_right = right.with_columns((pl.col("jk") // 2).alias("jk"))
    del source
    results = {}

    def time_case(name, fn, column):
        ms, result = best_of(fn, reps)
        total = result[column].sum()
        results[name] = (ms, result.height, 0.0 if total is None else float(total))

    time_case("inner_dense", lambda: left.join(right, on="jk", how="inner"), "r")
    time_case(
        "lazy_narrow",
        lambda: left.lazy().join(right.lazy(), on="jk").select("x", "r").collect(),
        "r",
    )
    sparse_left = left.with_columns((pl.col("jk") * 17).alias("key"))
    sparse_right = right.with_columns((pl.col("jk") * 17).alias("key"))
    time_case(
        "inner_sparse",
        lambda: sparse_left.join(sparse_right, on="key", how="inner"),
        "r",
    )
    del sparse_left, sparse_right

    string_left = left.with_columns(pl.col("jk").cast(pl.String).alias("key"))
    string_right = right.with_columns(pl.col("jk").cast(pl.String).alias("key"))
    time_case(
        "inner_string",
        lambda: string_left.join(string_right, on="key", how="inner"),
        "r",
    )
    del string_left, string_right

    compound = [
        (pl.col("jk") // 4).alias("a"),
        (pl.col("jk") % 4).alias("b"),
    ]
    multi_left = left.with_columns(compound)
    multi_right = right.with_columns(compound)
    time_case(
        "inner_multi",
        lambda: multi_left.join(multi_right, on=["a", "b"], how="inner"),
        "r",
    )
    del multi_left, multi_right

    for how in ("left", "right", "full", "semi", "anti"):
        column = "x" if how in ("semi", "anti") else "r"
        time_case(
            f"{how}_unmatched",
            lambda how=how: left.join(shifted, on="jk", how=how, coalesce=True),
            column,
        )

    sparse_left = left.with_columns((pl.col("jk") * 17).alias("jk"))
    sparse_shifted = shifted.with_columns((pl.col("jk") * 17).alias("jk"))
    del shifted
    time_case(
        "left_sparse",
        lambda: sparse_left.join(sparse_shifted, on="jk", how="left"),
        "r",
    )
    time_case(
        "right_sparse",
        lambda: sparse_left.join(sparse_shifted, on="jk", how="right", coalesce=True),
        "r",
    )
    del sparse_left, sparse_shifted

    time_case(
        "inner_duplicate",
        lambda: left.join(duplicate_right, on="jk", how="inner"),
        "r",
    )
    return results


def check(rows: int, mojo: dict, polars: dict, other_name: str = "Polars") -> None:
    for case in CASES:
        m_time, m_height, m_total = mojo[case]
        p_time, p_height, p_total = polars[case]
        if m_height != p_height:
            raise RuntimeError(
                f"{case} @ {rows}: row count differs: Mojo {m_height}, "
                f"{other_name} {p_height}"
            )
        tolerance = 1e-8 * max(abs(m_total), abs(p_total), 1.0)
        if abs(m_total - p_total) > tolerance:
            raise RuntimeError(
                f"{case} @ {rows}: total differs: Mojo {m_total}, "
                f"{other_name} {p_total}"
            )


def parse_variants(text: str) -> list[str]:
    variants = text.split(",")
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown key variants {sorted(unknown)}; choose from {VARIANTS}")
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000000,10000000")
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help="comma-separated key layouts: base, shuffled, wide")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=physical_cores())
    parser.add_argument("--data-dir", type=Path, default=ROOT / "build" / "bench_polars")
    parser.add_argument("--runner", type=Path)
    args = parser.parse_args()
    variants = parse_variants(args.variants)
    os.environ["POLARS_MAX_THREADS"] = str(args.threads)
    import polars as pl

    runner = args.runner.resolve() if args.runner else build_runner()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"# polars={pl.__version__} threads={args.threads} reps={args.reps} "
        f"machine={platform.machine()} {platform.system()}",
        flush=True,
    )
    print("| join case | keys | input rows | Mojo ms | Polars ms | Mojo / Polars |", flush=True)
    print("|---|---|---:|---:|---:|---:|", flush=True)
    for rows in (int(s) for s in args.sizes.split(",")):
        if rows < 1_000_000:
            raise ValueError("join comparison requires at least 1,000,000 rows")
        generate(args.data_dir, rows)
        if variants != ["base"]:
            generate_join_variants(args.data_dir, rows)
        for variant in variants:
            mojo = run_mojo(runner, args.data_dir, rows, args.reps, args.threads, variant)
            polars = run_polars(args.data_dir, rows, args.reps, variant)
            check(rows, mojo, polars)
            for case in CASES:
                m, p = mojo[case][0], polars[case][0]
                print(
                    f"| {case} | {variant} | {rows:,} | {m:.2f} | {p:.2f} | {m/p:.2f}x |",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
