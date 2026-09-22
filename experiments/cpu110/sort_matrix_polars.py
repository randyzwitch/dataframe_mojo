#!/usr/bin/env python3
"""Matched Polars oracle for the isolated #108 sort/top-k matrix.

The generated arrays exactly reproduce the seeded integer arithmetic in
`radix_pipeline_108.mojo` and `topk_pipeline_108.mojo`.  Every operation uses
``maintain_order=True`` and ``nulls_last=True``. Polars' ``DataFrame.top_k``
does not expose a stable-order option in Polars 1.44. The top-k oracle adds
the original row number as an explicit ascending tie breaker, selects on
``[key, payload]``, and sorts only those 100 rows for presentation. This has
the same order as Mojo's stable key sort while retaining bounded selection.

Usage:
  POLARS_MAX_THREADS=32 pixi run -e oracle python \\
    experiments/cpu110/sort_matrix_polars.py radix 1000000 int64
  POLARS_MAX_THREADS=32 pixi run -e oracle python \\
    experiments/cpu110/sort_matrix_polars.py topk 1000000 string
"""

from __future__ import annotations

import argparse
import os
import time


MASK64 = (1 << 64) - 1
MULTIPLIER = 6364136223846793005
INCREMENT = 1442695040888963407
REPETITIONS = 4
TOPK = 100


def step(state: int) -> int:
    return (state * MULTIPLIER + INCREMENT) & MASK64


def signed64(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


def radix_values(rows: int, kind: str):
    state = 0x9E3779B97F4A7C15 if kind == "int64" else 0xD1B54A32D192ED03
    values: list[int | str] = []
    for _ in range(rows):
        state = step(state)
        if kind == "int64":
            values.append(signed64(state))
        else:
            values.append(f"key_{(state >> 23) % 250_003}")
    return values, [True] * rows


def topk_values(rows: int, kind: str):
    state = 0xA24BAED4963EE407 if kind == "int64" else 0x8CB92BA72F3D8DD7
    values: list[int | str | None] = []
    valid: list[bool] = []
    for row in range(rows):
        state = step(state)
        value = (
            int((state >> 23) % 20_003) - 10_001
            if kind == "int64"
            else f"key_{(state >> 23) % 20_003}"
        )
        present = row % 31 != 0
        values.append(value if present else None)
        valid.append(present)
    return values, valid


def stable_reference(values: list[int | str | None], valid: list[bool], descending: bool):
    """Independent stable row order: sort valid rows then append nulls."""
    present = [row for row, is_valid in enumerate(valid) if is_valid]
    present.sort(key=lambda row: values[row], reverse=descending)
    return present + [row for row, is_valid in enumerate(valid) if not is_valid]


def make_frame(values, kind: str):
    import polars as pl

    dtype = pl.Int64 if kind == "int64" else pl.String
    return pl.DataFrame(
        {
            "key": pl.Series("key", values, dtype=dtype, strict=False),
            "payload": pl.Series("payload", range(len(values)), dtype=pl.Int64),
        }
    )


def same_rows(frame, expected: list[int], limit: int | None = None) -> None:
    actual = frame["payload"].to_list()
    wanted = expected if limit is None else expected[:limit]
    if actual != wanted:
        raise AssertionError("Polars stable row order differs from independent oracle")


def best_of(operation):
    operation()  # allocator/JIT warmup; excluded from reported time.
    best = float("inf")
    result = None
    for _ in range(REPETITIONS):
        started = time.perf_counter_ns()
        result = operation()
        best = min(best, time.perf_counter_ns() - started)
    return best, result


def stable_tiebroken_topk(frame):
    """Top 100 by descending key, then ascending original input row.

    ``top_k`` always puts non-null values before nulls, which matches Mojo's
    top_k null-last contract. `reverse=True` on the unique payload chooses
    the earliest input row when keys tie. The final sort orders only 100 rows.
    """
    selected = frame.top_k(TOPK, by=["key", "payload"], reverse=[False, True])
    return selected.sort(
        ["key", "payload"],
        descending=[True, False],
        nulls_last=[True, False],
        maintain_order=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", choices=("radix", "topk"))
    parser.add_argument("rows", type=int)
    parser.add_argument("kind", choices=("int64", "string"))
    args = parser.parse_args()
    if args.rows < 0 or (args.workload == "topk" and args.rows < TOPK):
        raise ValueError("rows must be nonnegative and at least 100 for topk")

    if args.workload == "radix":
        values, valid = radix_values(args.rows, args.kind)
        descending = False
        limit = None
    else:
        values, valid = topk_values(args.rows, args.kind)
        descending = True
        limit = TOPK
    expected = stable_reference(values, valid, descending)
    frame = make_frame(values, args.kind)

    if args.workload == "topk":
        operation = lambda: stable_tiebroken_topk(frame)
        semantics = "stable_tiebroken_topk_nulls_last"
    else:
        def operation():
            return frame.sort(
                "key",
                descending=False,
                nulls_last=True,
                maintain_order=True,
            )
        semantics = "stable_sort_nulls_last"

    # Validate exact stable row order before timing. For top-k this verifies
    # the first K rows only, which is the output contract under comparison.
    same_rows(operation(), expected, limit)
    best_ns, result = best_of(operation)
    same_rows(result, expected, limit)
    print(f"workload={args.workload}")
    print(f"kind={args.kind}")
    print(f"rows={args.rows}")
    print(f"threads={os.environ.get('POLARS_MAX_THREADS', 'default')}")
    print(f"semantics={semantics}")
    print(f"best_ns={best_ns}")


if __name__ == "__main__":
    main()
