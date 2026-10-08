"""Polars differential oracle for ordered, grouped, nullable as-of joins."""
from datetime import timedelta
import itertools

import numpy as np
import polars as pl
import pyarrow as pa
from polars.testing import assert_frame_equal


def cases():
    result = []
    types = [
        pa.int8(),
        pa.uint8(),
        pa.int16(),
        pa.uint16(),
        pa.int32(),
        pa.uint32(),
        pa.int64(),
        pa.uint64(),
        pa.float32(),
        pa.float64(),
        pa.date32(),
        pa.timestamp("ms"),
        pa.timestamp("us"),
        pa.timestamp("ns"),
        pa.timestamp("us", tz="America/New_York"),
        pa.duration("ms"),
        pa.duration("us"),
        pa.duration("ns"),
    ]
    for dtype, groups, strategy, tolerance, exact in itertools.product(
        types,
        (0, 1, 2),
        ("backward", "forward", "nearest"),
        (None, 0, 2),
        (True, False),
    ):
        left_keys = [None, 0, 1, 1, 2, None, 3, 4, 5, 9]
        right_keys = [None, 1, 1, None, 3, 3, 5, None]
        # Interleave groups so global ordering is insufficient; a null group
        # tests non-matching by keys without conflating them with null on keys.
        def table(keys, left):
            values, first, second = [], [], []
            for key in keys:
                for group in range(3 if groups else 1):
                    values.append(key)
                    first.append(
                        "a" if group == 0 else "b" if group == 1 else None
                    )
                    second.append(group % 2)
            key_name = "k" if left or len(result) % 2 == 0 else "rk"
            arrays = {key_name: pa.array(values, type=dtype)}
            if groups:
                arrays["g"] = pa.array(first, type=pa.string())
            if groups == 2:
                arrays["h"] = pa.array(second, type=pa.int32())
            arrays["v"] = pa.array(range(len(values)), type=pa.int64())
            arrays["payload"] = pa.array(
                [None if i % 4 == 0 else str(i) for i in range(len(values))]
            )
            return pa.record_batch(arrays)

        lhs, rhs = table(left_keys, True), table(right_keys, False)
        # Polars numerical tolerance uses physical key units. A Duration
        # scalar is covered separately below, including unit conversion.
        result.append(
            dict(
                left=lhs,
                right=rhs,
                by=[] if not groups else ["g"] if groups == 1 else ["g", "h"],
                strategy=strategy,
                tolerance=tolerance,
                exact=exact,
                unit=None,
                different=rhs.schema.names[0] != "k",
            )
        )
    # Full-width integer keys must not pass through Float64, including
    # nearest ties and differences which overflow the signed key's width.
    for dtype, keys in [
        (pa.uint64(), [0, 2**63, 2**64 - 2, 2**64 - 1]),
        (pa.int64(), [-(2**63), -(2**63) + 1, 2**63 - 2, 2**63 - 1]),
    ]:
        for strategy, exact in itertools.product(
            ("backward", "forward", "nearest"), (True, False)
        ):
            result.append(
                dict(
                    left=pa.record_batch({"k": pa.array(keys, type=dtype)}),
                    right=pa.record_batch(
                        {"k": pa.array(keys[::2], type=dtype), "v": [1, 2]}
                    ),
                    by=[],
                    strategy=strategy,
                    tolerance=None,
                    exact=exact,
                    unit=None,
                    different=False,
                )
            )
    for dtype in (
        pa.date32(),
        pa.timestamp("ms"),
        pa.timestamp("us"),
        pa.timestamp("ns"),
        pa.duration("ms"),
        pa.duration("us"),
        pa.duration("ns"),
    ):
        for strategy in ("backward", "forward", "nearest"):
            result.append(
                dict(
                    left=pa.record_batch(
                        {"k": pa.array([0, 1, 2, 5], type=dtype)}
                    ),
                    right=pa.record_batch(
                        {"k": pa.array([0, 4], type=dtype), "v": [1, 2]}
                    ),
                    by=[],
                    strategy=strategy,
                    tolerance=1500,
                    exact=True,
                    unit="us",
                    different=False,
                )
            )
    # Exercise nonfinite floating keys and Polars' total ordering.
    for strategy, exact, tolerance in itertools.product(
        ("backward", "forward", "nearest"), (True, False), (None, 1)
    ):
        result.append(
            dict(
                left=pa.record_batch(
                    {"k": [-np.inf, 0.0, 1.0, np.inf, np.nan]}
                ),
                right=pa.record_batch(
                    {"k": [-np.inf, 1.0, np.inf, np.nan], "v": [0, 1, 2, 3]}
                ),
                by=[],
                strategy=strategy,
                tolerance=tolerance,
                exact=exact,
                unit=None,
                different=False,
            )
        )
    for case in result:
        case["has_tolerance"] = case["tolerance"] is not None
    return result


def expected(case):
    kwargs = dict(
        left_on="k",
        right_on="rk" if case["different"] else "k",
        by=case["by"] or None,
        strategy=case["strategy"],
        allow_exact_matches=case["exact"],
        suffix="_r",
        check_sortedness=False,
    )
    tolerance = case["tolerance"]
    if case["unit"]:
        tolerance = timedelta(microseconds=tolerance)
    return pl.from_arrow(case["left"]).join_asof(
        pl.from_arrow(case["right"]), tolerance=tolerance, **kwargs
    )


def check(case, actual):
    assert_frame_equal(pl.from_arrow(actual), expected(case), check_exact=True)
    return True
