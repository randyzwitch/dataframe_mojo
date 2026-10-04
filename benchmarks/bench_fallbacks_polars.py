"""Polars counterpart of bench_fallbacks.mojo; time only the operation.

POLARS_MAX_THREADS=8 FALLBACK_CASE=sort pixi run -e oracle python ...
Input construction differs across languages; compare runtime, and compare
Mojo before/after peak RSS separately rather than treating Python RSS as
an isolated execution-memory measurement.
"""
import os
from time import perf_counter_ns

import polars as pl

mode = os.environ.get("FALLBACK_CASE", "mixed")
n = 1_000_000
groups = int(os.environ.get("FALLBACK_GROUPS", "100000"))
i = pl.Series("i", range(n), dtype=pl.Int64)
sorting = mode in ("sort", "packed", "prefix", "wide")
if sorting:
    alphabet = "abcdefghijklmnopqrstuvwxy"
    a = ["".join(alphabet[(r * 13 + j * 7) % 25] for j in range(10 + r % 31)) for r in range(775)]
    b = ["".join(alphabet[(r * 17 + j * 3) % 25] for j in range(10 + (r * 7) % 31)) for r in range(775)]
    if mode == "prefix":
        a = ["abcdefghijklmnopqrstuvwx" + str(r) for r in range(2003)]
    frame = pl.DataFrame({
        "k": i % 997,
        "a": [a[r % len(a)] for r in range(n)],
        "b": [b[r % len(b)] for r in range(n)],
        "last": (i % 37).cast(pl.Int32),
    })
    by = ["k", "last"] if mode == "packed" else ["k", "a", "b", "last"]
    descending = [True] + [False] * (len(by) - 1)
    if mode == "wide":
        frame = frame.with_columns(
            (pl.col("k") % 17).alias("extra_i"),
            pl.col("a").str.slice(3, 20).alias("extra_a"),
            pl.col("b").str.to_uppercase().alias("extra_b"),
            (pl.col("last").cast(pl.Int64) * 7).alias("extra_last"),
        )
        by += ["extra_i", "extra_a", "extra_b", "extra_last"]
        descending = [True, False, False, False, False, True, False, True]
    operation = lambda: frame.select(pl.arg_sort_by(by, descending=descending, nulls_last=True, maintain_order=True)).to_series()
else:
    frame = pl.DataFrame({"k": i * 37 % groups, "x": i % 101 - 50, "y": (i % 97).cast(pl.Float64), "z": (i % 89).cast(pl.Float64)})
    expressions = [pl.col("x").sum().alias("sx"), pl.col("y").sum().alias("sy"), pl.col("z").mean().alias("mz")]
    if mode == "mixed":
        expressions.append(pl.col("z").median().alias("median"))
    if mode == "computed":
        expressions += [(pl.col("y") + pl.col("z")).sum().alias("computed"), pl.col("x").first().alias("first")]
    operation = lambda: frame.group_by("k").agg(expressions)

times = []
for rep in range(4):
    start = perf_counter_ns()
    result = operation()
    elapsed = (perf_counter_ns() - start) / 1e6
    if rep:
        times.append(elapsed)
    if sorting:
        assert len(result) == n
        checksum = int(((i + 1) * result.cast(pl.Int64)).sum())
    else:
        assert result.height == groups
        checksum = result["sx"].sum()
print(mode, "rows", n, "groups", groups, "best_ms", min(times), "checksum", checksum)
