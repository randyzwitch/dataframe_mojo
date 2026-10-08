"""Independent Polars oracle for irregular, duplicate and calendar windows."""
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import polars as pl
import pyarrow as pa
from polars.testing import assert_frame_equal


def cases():
    result = []
    instants = [
        datetime(2024, 1, 28),
        datetime(2024, 1, 31),
        datetime(2024, 2, 1),
        datetime(2024, 2, 1),
        datetime(2024, 2, 28),
        datetime(2024, 2, 29),
        datetime(2024, 3, 1),
        datetime(2024, 3, 31),
        datetime(2024, 5, 1),
    ]
    hourly = [
        datetime(2024, 2, 29) + timedelta(minutes=k)
        for k in [0, 1, 30, 30, 90, 120, 400, 401, 10000]
    ]
    dst = [
        datetime(2024, 3, 9, 12) + timedelta(hours=k)
        for k in [0, 1, 12, 23, 24, 25, 48, 49, 96]
    ]
    for kind, dtype, times, durations in [
        ("date", pa.date32(), instants, ["2d", "1mo", "1y"]),
        ("ms", pa.timestamp("ms"), hourly, ["1h30m", "2d"]),
        ("us", pa.timestamp("us"), instants, ["2d", "1mo", "1y"]),
        ("ns", pa.timestamp("ns"), hourly, ["1h30m", "2d"]),
        (
            "tz",
            pa.timestamp("us", tz="America/New_York"),
            dst,
            ["1d", "24h", "1mo"],
        ),
    ]:
        for grouped in [False, True]:
            rows = [
                (t, g)
                for t in times
                for g in (["a", "b"] if grouped else ["a"])
            ]
            data = pa.record_batch(
                {
                    "t": pa.array([r[0] for r in rows], type=dtype),
                    "g": pa.array([r[1] for r in rows]),
                    "v": pa.array(
                        [
                            None if i % 5 == 0 else 1e9 + ((i * 7) % 17) / 8
                            for i in range(len(rows))
                        ],
                        type=pa.float64(),
                    ),
                }
            )
            for period, closed in product(
                durations, ["left", "right", "both", "none"]
            ):
                for mode in ["by", "rolling"]:
                    result.append(
                        dict(
                            mode=mode,
                            data=data,
                            period=period,
                            every="1d",
                            offset="default",
                            closed=closed,
                            label="left",
                            grouped=grouped,
                            size=3,
                            ddof=1,
                            needed=1,
                        )
                    )
                # Daily/monthly/annual aligned bins, overlapping and gapped windows.
                for label in ["left", "right", "datapoint"]:
                    result.append(
                        dict(
                            mode="dynamic",
                            data=data,
                            period=period,
                            every=period if period != "1h30m" else "30m",
                            offset="default",
                            closed=closed,
                            label=label,
                            grouped=grouped,
                            size=3,
                            ddof=1,
                            needed=1,
                        )
                    )
    data = pa.record_batch(
        {
            "t": pa.array(hourly, type=pa.timestamp("us")),
            "g": pa.array(["a"] * 9),
            "v": pa.array([float(i) for i in range(9)]),
        }
    )
    for mode, offset, closed in product(
        ["rolling", "dynamic"],
        ["-30m", "30m", "2d"],
        ["left", "right", "both", "none"],
    ):
        result.append(
            dict(
                mode=mode,
                data=data,
                period="1h30m",
                every="30m",
                offset=offset,
                closed=closed,
                label="left",
                grouped=False,
                size=3,
                ddof=1,
                needed=1,
            )
        )
    rng = np.random.default_rng(225)
    values = 1e9 + rng.normal(0, 0.001, 1000)
    for size, ddof, needed, grouped in product(
        [1, 7, 100], [0, 1, 2], [0, 1, -1], [False, True]
    ):
        data = pa.record_batch(
            {
                "t": pa.array(range(1000), type=pa.date32()),
                "g": pa.array(["a" if i % 2 else "b" for i in range(1000)]),
                "v": pa.array(values, mask=np.arange(1000) % 13 == 0),
            }
        )
        result.append(
            dict(
                mode="row",
                data=data,
                period="2d",
                every="1d",
                offset="default",
                closed="right",
                label="left",
                grouped=grouped,
                size=size,
                ddof=ddof,
                needed=needed,
            )
        )
    template = result[0]
    for dtype in [
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
    ]:
        data = pa.record_batch(
            {
                "t": pa.array(range(9), type=pa.date32()),
                "g": pa.array(["a"] * 9),
                "v": pa.array([1, 4, None, 2, 7, 3, 4, 9, 2], type=dtype),
            }
        )
        for mode in ["row", "by", "rolling", "dynamic"]:
            for needed in [0, 1]:
                result.append(
                    dict(template, data=data, mode=mode, needed=needed)
                )
    for mode, period, every, offset in product(
        ["dynamic"],
        ["1d", "3d", "1mo", "1q"],
        ["1d", "2d", "1mo"],
        ["default", "-1d", "1d"],
    ):
        for closed in ["left", "right", "both", "none"]:
            result.append(
                dict(
                    template,
                    mode=mode,
                    data=pa.record_batch(
                        {
                            "t": pa.array(instants, type=pa.timestamp("us")),
                            "g": pa.array(["a"] * len(instants)),
                            "v": pa.array(
                                range(len(instants)), type=pa.float64()
                            ),
                        }
                    ),
                    period=period,
                    every=every,
                    offset=offset,
                    closed=closed,
                )
            )
    for values in [
        [float("nan"), 1.0, 2.0, 3.0, 4.0, 5.0],
        [1.0, float("inf"), 2.0, 3.0, 4.0, 5.0],
        [None] * 6,
    ]:
        data = pa.record_batch(
            {
                "t": pa.array(range(6), type=pa.date32()),
                "g": pa.array(["a"] * 6),
                "v": pa.array(values, type=pa.float64()),
            }
        )
        for mode, needed in product(["row", "by"], [0, 1]):
            result.append(dict(template, data=data, mode=mode, needed=needed))
    return result


def expected(case):
    frame = pl.from_arrow(case["data"])
    by = ["g"] if case["grouped"] else []
    mode = case["mode"]
    if mode == "row":
        opts = dict(
            window_size=case["size"],
            min_samples=case["size"] if case["needed"] < 0 else case["needed"],
            ddof=case["ddof"],
        )
        exprs = [
            getattr(pl.col("v"), "rolling_" + name)(**opts).alias(name)
            for name in ["std", "var"]
        ]
        if by:
            exprs = [e.over(by) for e in exprs]
        return frame.select(exprs)
    if mode == "by":
        exprs = [
            getattr(pl.col("v"), "rolling_" + name + "_by")(
                "t",
                window_size=case["period"],
                closed=case["closed"],
                min_samples=case["needed"],
            ).alias(name)
            for name in ["sum", "mean", "min", "max", "std", "var"]
        ]
        if by:
            exprs = [e.over(by) for e in exprs]
        return frame.select(exprs)
    opts = dict(
        index_column="t",
        period=case["period"],
        closed=case["closed"],
        group_by=by or None,
    )
    if case["offset"] != "default":
        opts["offset"] = case["offset"]
    request = frame.rolling(
        **opts
    ) if mode == "rolling" else frame.group_by_dynamic(
        every=case["every"], label=case["label"], **opts
    )
    return request.agg(
        [
            getattr(pl.col("v"), name)().alias(name)
            for name in ["sum", "mean", "min", "max", "std", "var"]
        ]
    )


def check(case, actual):
    actual = pl.from_arrow(actual)
    reference = expected(case)
    for name in ["mean", "std", "var"]:
        if (
            name in reference.columns
            and name in actual.columns
            and actual.schema[name] == pl.Float64
        ):
            reference = reference.with_columns(pl.col(name).cast(pl.Float64))
    if case["mode"] in ["rolling", "dynamic"] and case["grouped"]:
        actual = actual.sort(["g", "t"], maintain_order=True)
        reference = reference.sort(["g", "t"], maintain_order=True)
    # Polars' unshifted sliding updates differ from centered high-precision
    # results by up to ~1e-7 on 1e9 + 1e-3 noise. Check parity at that scale,
    # then independently require the translated result to be much tighter.
    noisy = case["mode"] == "row" and len(case["data"]) == 1000
    assert_frame_equal(
        actual,
        reference,
        rel_tol=5e-4 if noisy else 2e-5,
        abs_tol=1e-7 if noisy else 2e-8,
    )
    if noisy:
        values = case["data"]["v"].to_pylist()
        keys = case["data"]["g"].to_pylist()
        needed = case["size"] if case["needed"] < 0 else case["needed"]
        for key in dict.fromkeys(keys) if case["grouped"] else [None]:
            rows = [i for i, k in enumerate(keys) if key is None or k == key]
            for j, row in enumerate(rows):
                window = [
                    values[rows[k]]
                    for k in range(max(0, j - case["size"] + 1), j + 1)
                    if values[rows[k]] is not None
                ]
                if len(window) < needed or len(window) <= case["ddof"]:
                    continue
                centered = np.asarray(window, dtype=np.longdouble)
                centered -= centered[0]
                variance = float(np.var(centered, ddof=case["ddof"]))
                np.testing.assert_allclose(
                    actual["var"][row], variance, rtol=1e-10, atol=1e-16
                )
                np.testing.assert_allclose(
                    actual["std"][row],
                    np.sqrt(variance),
                    rtol=1e-10,
                    atol=1e-12,
                )
    return True
