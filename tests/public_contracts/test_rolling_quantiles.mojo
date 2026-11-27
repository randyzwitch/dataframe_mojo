"""Rolling medians and quantiles over fixed and duration windows match a
model that sorts each window's valid values."""
# contract-group: expr.rolling_quantiles
from std.math import abs, ceil, floor, isnan
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.coverage_generators import chunk_layout, offset_layout


def _model_quantile(
    var window: List[Float64], q: Float64, method: String
) -> Float64:
    """The quantile of `window` (sorted here, NaN above every number)."""
    var numbers = List[Float64]()
    var nans = 0
    for x in window:
        if isnan(x):
            nans += 1
        else:
            numbers.append(x)
    sort(numbers)
    for _ in range(nans):
        numbers.append(FloatLiteral.nan)
    var c = len(numbers)
    var f = Float64(c - 1) * q
    var lo = min(Int(floor(f)), c - 1)
    var hi = min(Int(ceil(f)), c - 1)
    if method == "lower":
        return numbers[lo]
    if method == "higher":
        return numbers[hi]
    if method == "nearest":
        return numbers[min(Int(floor(f + 0.5)), c - 1)]
    if method == "equiprobable":
        return numbers[min(max(Int(ceil(Float64(c) * q)) - 1, 0), c - 1)]
    if lo == hi:
        return numbers[lo]
    if method == "midpoint":
        return (numbers[lo] + numbers[hi]) / 2
    return (f - Float64(lo)) * (numbers[hi] - numbers[lo]) + numbers[lo]


def _same(actual: Float64, expected: Float64) -> Bool:
    if isnan(expected):
        return isnan(actual)
    if expected == actual:
        return True
    return abs(actual - expected) <= 1e-9 * (1 + abs(expected))


def _payload(n: Int, seed: Int) -> Tuple[List[Float64], List[Bool]]:
    """Values with repeats, nulls, NaN and infinities."""
    var values = List[Float64]()
    var valid = List[Bool]()
    var state = UInt64(seed * 2654435761 + 12345)
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var r = Int((state >> 33) % 1000)
        var x = Float64(r % 37) - 12
        if r < 20:
            x = FloatLiteral.nan
        elif r < 30:
            x = FloatLiteral.infinity
        elif r < 35:
            x = -FloatLiteral.infinity
        values.append(x)
        valid.append(r % 11 != 3)
    return (values^, valid^)


def _check_fixed(
    actual: Series,
    values: List[Float64],
    valid: List[Bool],
    groups: List[Int],
    window: Int,
    needed: Int,
    q: Float64,
    method: String,
) raises:
    """`actual` against the model: each row's window is the `window` rows
    of its group ending at it."""
    var floats = actual.cast(DataType.FLOAT64)
    for row in range(len(values)):
        var picked = List[Float64]()
        var seen = 0
        var k = row
        while k >= 0 and seen < window:
            if groups[k] == groups[row]:
                seen += 1
                if valid[k]:
                    picked.append(values[k])
            k -= 1
        var item = floats.get(row)
        if len(picked) < needed or len(picked) == 0:
            assert_true(item.is_null(), "row " + String(row) + " is null")
            continue
        assert_true(not item.is_null(), "row " + String(row) + " has a value")
        var expected = _model_quantile(picked^, q, method)
        assert_true(
            _same(item.float64(), expected),
            "row "
            + String(row)
            + " "
            + method
            + ": "
            + String(item.float64())
            + " != "
            + String(expected),
        )


def test_rolling_quantile_matches_sorted_windows_for_every_method() raises:
    var methods: List[String] = [
        "nearest",
        "lower",
        "higher",
        "midpoint",
        "linear",
        "equiprobable",
    ]
    var n = 160
    var payload = _payload(n, 7)
    var groups = List[Int64]()
    var flat = List[Int]()
    var none = List[Int]()
    for row in range(n):
        groups.append(Int64((row * 7) % 3))
        flat.append(Int((row * 7) % 3))
        none.append(0)
    var source = DataFrame(
        [
            Series("x", Column[Float64](payload[0].copy(), payload[1])),
            Series("g", Column[Int64](groups^)),
        ]
    )
    # Every method and quantile on one window; other window sizes and
    # minimums with one of them; chunked and offset layouts once.
    var configs = List[Tuple[Int, Int, Float64, String, Int]]()
    for method in methods:
        for q in [0.0, 0.3, 0.5, 1.0]:
            configs.append((5, 2, q, method, 0))
    for window in [1, 2, 64]:
        for needed in [-1, 1]:
            configs.append((window, needed, 0.3, "linear", 0))
    configs.append((7, 3, 0.75, "nearest", 1))
    configs.append((7, 3, 0.75, "midpoint", 2))
    for config in configs:
        var window = config[0]
        var minimum = window if config[1] < 0 else min(config[1], window)
        var q = config[2]
        var method = config[3]
        var frame = (
            source.copy() if config[4]
            == 0 else offset_layout(source) if config[4]
            == 1 else chunk_layout(source)
        )
        var expr = col("x").rolling_quantile(q, method, window, minimum)
        var exprs: List[Expr] = [
            expr.copy().alias("plain"),
            expr.copy().over("g").alias("grouped"),
        ]
        var eager = frame.select_exprs(exprs)
        var lazy = frame.lazy().select_exprs(exprs).collect()
        for result in [eager.copy(), lazy^]:
            assert_true(result.column("plain").dtype() == DataType.FLOAT64)
            _check_fixed(
                result.column("plain"),
                payload[0],
                payload[1],
                none,
                window,
                minimum,
                q,
                method,
            )
            _check_fixed(
                result.column("grouped"),
                payload[0],
                payload[1],
                flat,
                window,
                minimum,
                q,
                method,
            )


def test_rolling_quantile_with_dense_nulls_and_a_full_minimum() raises:
    # Windows that hold too few values for a while still move the
    # quantile's position as rows enter and leave around the nulls.
    var n = 150
    var payload = _payload(n, 13)
    var valid = List[Bool]()
    for row in range(n):
        valid.append(payload[1][row] and (row * 7) % 5 >= 2)
    var none = List[Int](length=n, fill=0)
    var source = DataFrame(
        [Series("x", Column[Float64](payload[0].copy(), valid))]
    )
    for window in [7, 9]:
        for needed in [window - 2, window]:
            for method in ["linear", "higher"]:
                var actual = source.select_exprs(
                    [col("x").rolling_quantile(0.6, method, window, needed)]
                )
                _check_fixed(
                    actual.column("x"),
                    payload[0],
                    valid,
                    none,
                    window,
                    needed,
                    0.6,
                    method,
                )


def test_rolling_median_is_the_linear_half_quantile_and_keeps_float32() raises:
    var payload = _payload(200, 11)
    var none = List[Int](length=200, fill=0)
    var source = DataFrame(
        [Series("x", Column[Float64](payload[0].copy(), payload[1]))]
    )
    for window in [1, 3, 4, 50]:
        var actual = source.select_exprs(
            [
                col("x").rolling_median(window).alias("m"),
                col("x").rolling_median(window, 1).alias("m1"),
                col("x")
                .cast(DataType.FLOAT32)
                .rolling_median(window)
                .alias("f32"),
                col("x")
                .cast(DataType.INT32, strict=False)
                .rolling_median(window, 1)
                .alias("i32"),
            ]
        )
        assert_true(actual.column("f32").dtype() == DataType.FLOAT32)
        assert_true(actual.column("i32").dtype() == DataType.FLOAT64)
        _check_fixed(
            actual.column("m"),
            payload[0],
            payload[1],
            none,
            window,
            window,
            0.5,
            "linear",
        )
        _check_fixed(
            actual.column("m1"),
            payload[0],
            payload[1],
            none,
            window,
            1,
            0.5,
            "linear",
        )
        # Float32 values are exact small integers or non-finite here, so
        # their medians are the Float64 medians, rounded once.
        var f32 = actual.column("f32").cast(DataType.FLOAT64)
        var f64 = actual.column("m")
        for row in range(200):
            assert_equal(f32.get(row).is_null(), f64.get(row).is_null())
            if not f64.get(row).is_null():
                assert_true(
                    _same(
                        f32.get(row).float64(),
                        Float64(Float32(f64.get(row).float64())),
                    )
                )


def test_rolling_quantile_of_a_decimal_keeps_its_type_and_truncates() raises:
    # Interpolated decimals keep the input's type: lo + (hi - lo) * d on
    # the unscaled integers, truncated toward zero.
    var dtype = DataType.decimal(10, 2)
    var source = DataFrame(
        [
            Series("d", Column[Int128]([101, 102, -103, 104, 0])).with_dtype(
                dtype
            )
        ]
    )
    var actual = source.select_exprs(
        [
            col("d").rolling_median(2).alias("m"),
            col("d").rolling_quantile(0.25, "linear", 3, 1).alias("q"),
            col("d").rolling_quantile(0.5, "higher", 2).alias("h"),
        ]
    )
    for name in ["m", "q", "h"]:
        assert_true(actual.column(name).dtype() == dtype)
    var m = actual.column("m").cast(DataType.FLOAT64)
    assert_true(m.get(0).is_null())
    # (1.01 + 1.02) / 2 = 1.015 -> 1.01; (1.02 - 1.03) / 2 = -0.005 -> 0.
    assert_equal(m.get(1).float64(), 1.01)
    assert_equal(m.get(2).float64(), 0.0)
    assert_equal(m.get(3).float64(), 0.0)
    assert_equal(m.get(4).float64(), 0.52)
    var h = actual.column("h").cast(DataType.FLOAT64)
    assert_equal(h.get(1).float64(), 1.02)
    assert_equal(h.get(2).float64(), 1.02)
    var q = actual.column("q").cast(DataType.FLOAT64)
    # [1.01] -> 1.01; [1.01, 1.02] -> 1.0125 -> 1.01;
    # [-1.03, 1.01, 1.02] -> -1.03 + 2.04 * 0.5 = -0.01.
    assert_equal(q.get(0).float64(), 1.01)
    assert_equal(q.get(1).float64(), 1.01)
    assert_equal(q.get(2).float64(), -0.01)


def test_rolling_quantile_by_matches_duration_windows_in_any_row_order() raises:
    var n = 120
    var payload = _payload(n, 3)
    var times = List[Int64]()
    for row in range(n):
        # Days, shuffled, with repeated days.
        times.append(Int64(((row * 37) % n) // 2))
    var source = DataFrame(
        [
            Series("x", Column[Float64](payload[0].copy(), payload[1])),
            Series("t", Column[Int64](times.copy())).with_dtype(DataType.DATE),
        ]
    )
    var methods: List[String] = ["nearest", "linear", "midpoint", "lower"]
    for closed in ["right", "left", "both", "none"]:
        for period in [1, 10]:
            var exprs = List[Expr]()
            for method in methods:
                exprs.append(
                    col("x")
                    .rolling_quantile_by(
                        "t",
                        String(period) + "d",
                        0.4,
                        method,
                        closed,
                        1,
                    )
                    .alias(method)
                )
            exprs.append(
                col("x")
                .rolling_median_by("t", String(period) + "d", closed, 2)
                .alias("median")
            )
            var eager = source.select_exprs(exprs)
            var lazy = source.lazy().select_exprs(exprs).collect()
            for result in [eager.copy(), lazy^]:
                for name in [
                    "nearest",
                    "linear",
                    "midpoint",
                    "lower",
                    "median",
                ]:
                    var q = 0.5 if name == "median" else 0.4
                    var method = String(
                        "linear"
                    ) if name == "median" else String(name)
                    var needed = 2 if name == "median" else 1
                    var column = result.column(name)
                    for row in range(n):
                        var t = times[row]
                        var picked = List[Float64]()
                        for k in range(n):
                            var start = t - Int64(period)
                            var left = closed == "left" or closed == "both"
                            var right = closed == "right" or closed == "both"
                            var lo_ok = (
                                times[k] >= start if left else times[k] > start
                            )
                            var hi_ok = times[k] <= t if right else times[k] < t
                            if lo_ok and hi_ok and payload[1][k]:
                                picked.append(payload[0][k])
                        var item = column.get(row)
                        if len(picked) < needed:
                            assert_true(item.is_null())
                            continue
                        assert_true(
                            _same(
                                item.float64(),
                                _model_quantile(picked^, q, method),
                            ),
                            name + " " + closed + " row " + String(row),
                        )


def test_rolling_quantile_rejects_bad_options() raises:
    var frame = DataFrame(
        [
            Series("x", Column[Float64]([1, 2, 3])),
            Series("s", Column[Int64]([1, 2, 3])).cast(DataType.STRING),
        ]
    )
    with assert_raises(contains="quantile must be between 0 and 1"):
        _ = frame.select_exprs([col("x").rolling_quantile(1.5, "linear", 2)])
    with assert_raises(contains="interpolation must be"):
        _ = frame.select_exprs([col("x").rolling_quantile(0.5, "cubic", 2)])
    with assert_raises(contains="min_samples cannot exceed window_size"):
        _ = frame.select_exprs([col("x").rolling_median(2, 3)])
    with assert_raises(contains="window_size must be at least 1"):
        _ = frame.select_exprs([col("x").rolling_median(0)])
    with assert_raises(contains="requires a numeric expression"):
        _ = frame.select_exprs([col("s").rolling_median(2)])
    with assert_raises(contains="quantile must be between 0 and 1"):
        _ = (
            frame.lazy()
            .select_exprs([col("x").rolling_quantile(-0.1, "linear", 2)])
            .collect()
        )


def test_time_window_medians_and_quantiles_slide_over_overlapping_windows() raises:
    # Windows of 10 days over one row a day hold every row many times, so
    # medians and quantiles are read by sliding one frame over each
    # group's windows; rows come out of time order and in two groups.
    var n = 90
    var payload = _payload(n, 5)
    var days = List[Int64]()
    var keys = List[Int64]()
    for row in range(n):
        days.append(Int64((row * 41) % n))
        keys.append(Int64(row % 2))
    var source = DataFrame(
        [
            Series("x", Column[Float64](payload[0].copy(), payload[1])),
            Series("i", Column[Float64](payload[0].copy(), payload[1])).cast(
                DataType.INT64, strict=False
            ),
            Series("t", Column[Int64](days.copy())).with_dtype(DataType.DATE),
            Series("k", Column[Int64](keys.copy())),
        ]
    )
    var aggs: List[Expr] = [
        col("x").median().alias("median"),
        col("x").quantile(0.3, "nearest").alias("nearest"),
        col("i").quantile(0.7, "linear").alias("ilinear"),
        (col("x") * 2).quantile(0.5, "midpoint").alias("midpoint"),
    ]
    var group_by: List[String] = ["k"]
    var eager = source.rolling("t", "10d", group_by=group_by).agg(aggs)
    var lazy = (
        source.lazy().rolling("t", "10d", group_by=group_by).agg(aggs).collect()
    )
    for result in [eager.copy(), lazy^]:
        assert_equal(result.height(), n)
        for name in ["median", "nearest", "ilinear", "midpoint"]:
            assert_true(result.column(name).dtype() == DataType.FLOAT64)
        for window_row in range(n):
            var k = result.item(window_row, "k").int64()
            var t = Int64(
                result.column("t").cast(DataType.INT64).get(window_row).int64()
            )
            var xs = List[Float64]()
            var ints = List[Float64]()
            for row in range(n):
                if keys[row] == k and days[row] > t - 10 and days[row] <= t:
                    if payload[1][row]:
                        xs.append(payload[0][row])
                        var as_int = source.column("i").get(row)
                        if not as_int.is_null():
                            ints.append(Float64(as_int.int64()))
            var checks: List[String] = ["median", "nearest", "midpoint"]
            for name in checks:
                var item = result.column(name).get(window_row)
                if len(xs) == 0:
                    assert_true(item.is_null())
                    continue
                var q = 0.3 if name == "nearest" else 0.5
                var method = String("midpoint")
                if name == "median":
                    method = "linear"
                elif name == "nearest":
                    method = "nearest"
                var picked = xs.copy()
                if name == "midpoint":
                    for v in range(len(picked)):
                        picked[v] = picked[v] * 2
                assert_true(
                    _same(item.float64(), _model_quantile(picked^, q, method)),
                    name + " window " + String(window_row),
                )
            var item = result.column("ilinear").get(window_row)
            if len(ints) == 0:
                assert_true(item.is_null())
            else:
                assert_true(
                    _same(item.float64(), _model_quantile(ints^, 0.7, "linear"))
                )
