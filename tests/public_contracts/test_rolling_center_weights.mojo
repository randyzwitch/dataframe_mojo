"""Centered and weighted fixed-size rolling windows."""
# contract-group: expr.rolling_quantiles
from std.math import abs, isnan, sqrt
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.coverage_generators import chunk_layout


def _same(actual: Float64, expected: Float64) -> Bool:
    if isnan(expected):
        return isnan(actual)
    if expected == actual:
        return True
    return abs(actual - expected) <= 1e-9 * (1 + abs(expected))


def _assert_values(
    column: Series, expected: List[Float64], valid: List[Bool], label: String
) raises:
    var floats = column.cast(DataType.FLOAT64)
    assert_equal(len(floats), len(expected))
    for row in range(len(expected)):
        var item = floats.get(row)
        assert_equal(
            item.is_null(), not valid[row], label + " row " + String(row)
        )
        if valid[row]:
            assert_true(
                _same(item.float64(), expected[row]),
                label
                + " row "
                + String(row)
                + ": "
                + String(item.float64())
                + " != "
                + String(expected[row]),
            )


def _model(
    values: List[Float64],
    valid: List[Bool],
    groups: List[Int],
    window: Int,
    needed: Int,
    kind: String,
    ddof: Int,
) -> Tuple[List[Float64], List[Bool]]:
    """Each row's centered window of its group (rows of a group are
    consecutive here): `window` rows around it, the extra row of an even
    window before it, cut short at the group's edges."""
    var n = len(values)
    var out = List[Float64](length=n, fill=0)
    var ok = List[Bool](length=n, fill=False)
    var after = (window + 1) // 2
    var before = window - after
    for row in range(n):
        var lo = row
        while lo > 0 and groups[lo - 1] == groups[row]:
            lo -= 1
        var hi = row
        while hi < n and groups[hi] == groups[row]:
            hi += 1
        var b = max(lo, row - before)
        var e = min(hi, row + after)
        var picked = List[Float64]()
        for k in range(b, e):
            if valid[k]:
                picked.append(values[k])
        var c = len(picked)
        if c < needed or c == 0:
            continue
        var total = Float64(0)
        var low = picked[0]
        var high = picked[0]
        for x in picked:
            total += x
            low = min(low, x)
            high = max(high, x)
        if kind == "sum":
            out[row] = total
        elif kind == "mean":
            out[row] = total / Float64(c)
        elif kind == "min":
            out[row] = low
        elif kind == "max":
            out[row] = high
        elif kind == "median":
            sort(picked)
            out[row] = (
                picked[c // 2] if c % 2
                == 1 else (picked[c // 2 - 1] + picked[c // 2]) / 2
            )
        else:
            if c <= ddof:
                continue
            var mean = total / Float64(c)
            var squares = Float64(0)
            for x in picked:
                squares += (x - mean) * (x - mean)
            var variance = squares / Float64(c - ddof)
            out[row] = sqrt(variance) if kind == "std" else variance
        ok[row] = True
    return (out^, ok^)


def test_centered_windows_match_the_model_for_every_rolling_function() raises:
    var n = 61
    var values = List[Float64]()
    var valid = List[Bool]()
    var keys = List[Int64]()
    var groups = List[Int]()
    var none = List[Int]()
    for row in range(n):
        values.append(Float64((row * 37) % 23) - 9)
        valid.append(row % 9 != 4)
        keys.append(Int64(row // 20))
        groups.append(row // 20)
        none.append(0)
    var source = DataFrame(
        [
            Series("x", Column[Float64](values.copy(), valid)),
            Series("i", Column[Float64](values.copy(), valid)).cast(
                DataType.INT64
            ),
            Series("g", Column[Int64](keys^)),
        ]
    )
    var kinds: List[String] = [
        "sum",
        "mean",
        "min",
        "max",
        "std",
        "var",
        "median",
    ]
    for window in [1, 2, 3, 4, 7, 30]:
        for wanted in [1, 2]:
            var needed = min(wanted, window)
            var exprs = List[Expr]()
            for kind in kinds:
                for name in ["x", "i"]:
                    var c = col(name)
                    var e: Expr
                    if kind == "sum":
                        e = c.rolling_sum(window, needed, center=True)
                    elif kind == "mean":
                        e = c.rolling_mean(window, needed, center=True)
                    elif kind == "min":
                        e = c.rolling_min(window, needed, center=True)
                    elif kind == "max":
                        e = c.rolling_max(window, needed, center=True)
                    elif kind == "std":
                        e = c.rolling_std(window, needed, center=True)
                    elif kind == "var":
                        e = c.rolling_var(window, needed, center=True)
                    else:
                        e = c.rolling_median(window, needed, center=True)
                    exprs.append(e.copy().alias(kind + name))
                    exprs.append(e.over("g").alias(kind + name + "g"))
            var eager = chunk_layout(source).select_exprs(exprs)
            var lazy = source.lazy().select_exprs(exprs).collect()
            for result in [eager.copy(), lazy^]:
                for kind in kinds:
                    var ddof = 1 if kind == "std" or kind == "var" else 0
                    var plain = _model(
                        values, valid, none, window, needed, kind, ddof
                    )
                    var grouped = _model(
                        values, valid, groups, window, needed, kind, ddof
                    )
                    for name in ["x", "i"]:
                        var label = kind + String(name) + " w" + String(window)
                        _assert_values(
                            result.column(kind + name),
                            plain[0],
                            plain[1],
                            label,
                        )
                        _assert_values(
                            result.column(kind + name + "g"),
                            grouped[0],
                            grouped[1],
                            label + " grouped",
                        )


def test_weighted_windows_match_polars_on_complete_input() raises:
    # Values 1..6 with weights 0.5, 1, 2: the weights line up with the
    # end of a trailing window cut short at the start, and with the
    # center of a centered one.
    var frame = DataFrame([Series("x", Column[Float64]([1, 2, 3, 4, 5, 6]))])
    var w: List[Float64] = [0.5, 1.0, 2.0]
    var all = List[Bool](length=6, fill=True)
    var exprs: List[Expr] = [
        col("x").rolling_sum(3, 1, weights=w).alias("sum"),
        col("x").rolling_sum(3, 1, center=True, weights=w).alias("csum"),
        col("x").rolling_mean(3, 1, weights=w).alias("mean"),
        col("x").rolling_min(3, 1, weights=w).alias("min"),
        col("x").rolling_min(3, 1, center=True, weights=w).alias("cmin"),
        col("x").rolling_max(3, 1, weights=w).alias("max"),
        col("x").rolling_max(3, 1, center=True, weights=w).alias("cmax"),
        col("x").rolling_var(3, 1, weights=w).alias("var"),
        col("x").rolling_std(3, 1, weights=w).alias("std"),
        col("x").rolling_median(3, 1, weights=w).alias("median"),
        col("x").rolling_median(3, 1, center=True, weights=w).alias("cmedian"),
        col("x").rolling_quantile(0.3, "linear", 3, 1, weights=w).alias("q"),
        col("x").rolling_sum(3, weights=w).alias("full"),
    ]
    var eager = frame.select_exprs(exprs)
    var lazy = frame.lazy().select_exprs(exprs).collect()
    for result in [eager.copy(), lazy^]:
        _assert_values(
            result.column("sum"), [2, 5, 8.5, 12, 15.5, 19], all, "sum"
        )
        _assert_values(
            result.column("csum"), [5, 8.5, 12, 15.5, 19, 8.5], all, "csum"
        )
        _assert_values(
            result.column("mean"),
            [
                1.0,
                1.6666666666666667,
                2.4285714285714284,
                3.4285714285714284,
                4.428571428571429,
                5.428571428571429,
            ],
            all,
            "mean",
        )
        _assert_values(result.column("min"), [2, 1, 0.5, 1, 1.5, 2], all, "min")
        _assert_values(
            result.column("cmin"), [1, 0.5, 1, 1.5, 2, 2.5], all, "cmin"
        )
        _assert_values(result.column("max"), [2, 4, 6, 8, 10, 12], all, "max")
        _assert_values(result.column("cmax"), [4, 6, 8, 10, 12, 6], all, "cmax")
        _assert_values(
            result.column("var"),
            [
                0.0,
                0.2222222222222222,
                0.5306122448979592,
                0.5306122448979592,
                0.5306122448979592,
                0.5306122448979592,
            ],
            all,
            "var",
        )
        _assert_values(
            result.column("std"),
            [
                0.0,
                0.4714045207910317,
                0.7284313590846836,
                0.7284313590846836,
                0.7284313590846836,
                0.7284313590846836,
            ],
            all,
            "std",
        )
        _assert_values(
            result.column("median"),
            [1, 1.5, 2.25, 3.25, 4.25, 5.25],
            all,
            "median",
        )
        _assert_values(
            result.column("cmedian"),
            [1.5, 2.25, 3.25, 4.25, 5.25, 5.5],
            all,
            "cmedian",
        )
        _assert_values(
            result.column("q"), [1.0, 1.3, 1.9, 2.9, 3.9, 4.9], all, "q"
        )
        var full = all.copy()
        full[0] = False
        full[1] = False
        _assert_values(
            result.column("full"), [0, 0, 8.5, 12, 15.5, 19], full, "full"
        )


def test_weighted_quantile_methods_and_centered_even_windows() raises:
    var frame = DataFrame([Series("x", Column[Float64]([1, 2, 3, 4, 5, 6]))])
    var w3: List[Float64] = [0.5, 1.0, 2.0]
    var w4: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var all = List[Bool](length=6, fill=True)
    var methods: List[String] = [
        "nearest",
        "higher",
        "linear",
        "equiprobable",
    ]
    var trailing: List[List[Float64]] = [
        [1, 1, 2, 3, 4, 5],
        [1, 2, 2, 3, 4, 5],
        [1.0, 1.3, 1.9, 2.9, 3.9, 4.9],
        [1, 2, 2, 3, 4, 5],
    ]
    var centered: List[List[Float64]] = [
        [2, 2, 3, 4, 5, 5],
        [2, 3, 4, 5, 6, 6],
        [1.6, 2.3, 3.1, 4.1, 5.1, 5.333333333333333],
        [2, 3, 4, 5, 6, 6],
    ]
    for k in range(len(methods)):
        var result = frame.select_exprs(
            [
                col("x")
                .rolling_quantile(0.3, methods[k], 3, 1, weights=w3)
                .alias("t"),
                col("x")
                .rolling_quantile(
                    0.6, methods[k], 4, 1, center=True, weights=w4
                )
                .alias("c"),
            ]
        )
        _assert_values(result.column("t"), trailing[k], all, methods[k])
        _assert_values(result.column("c"), centered[k], all, methods[k] + " c")
    # One value: lower and midpoint are that value.
    var single = frame.select_exprs(
        [
            col("x")
            .rolling_quantile(0.3, "lower", 3, 1, weights=w3)
            .alias("l"),
            col("x")
            .rolling_quantile(0.3, "midpoint", 3, 1, weights=w3)
            .alias("m"),
        ]
    )
    _assert_values(single.column("l"), [1, 1, 1, 2, 3, 4], all, "lower")
    _assert_values(single.column("m"), [1, 1.5, 1.5, 2.5, 3.5, 4.5], all, "mid")


def test_weighted_windows_skip_nulls_and_keep_types() raises:
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [1, 2, 3, 4, 5], [True, False, True, True, True]
                ),
            ),
            Series("i", Column[Int32]([1, 2, 3, 4, 5])),
        ]
    )
    var w: List[Float64] = [1.0, 2.0, 3.0]
    var result = frame.select_exprs(
        [
            col("x").rolling_sum(3, 1, weights=w).alias("sum"),
            col("x").rolling_mean(3, 2, weights=w).alias("mean"),
            col("i").rolling_sum(3, weights=w).alias("isum"),
            col("x")
            .cast(DataType.FLOAT32)
            .rolling_max(3, weights=w)
            .alias("f32"),
            col("i").rolling_mean(3, weights=w).alias("imean"),
        ]
    )
    assert_true(result.column("isum").dtype() == DataType.FLOAT64)
    assert_true(result.column("f32").dtype() == DataType.FLOAT32)
    assert_true(result.column("imean").dtype() == DataType.FLOAT64)
    # Row 2's window [1, null, 3] with weights [1, 2, 3]: 1 + 9; row 1's
    # [1, null] takes the last two weights: 2.
    _assert_values(
        result.column("sum"),
        [3, 2, 10, 18, 26],
        [True, True, True, True, True],
        "sum",
    )
    _assert_values(
        result.column("mean"),
        [0, 0, 10.0 / 4.0, 18.0 / 5.0, 26.0 / 6.0],
        [False, False, True, True, True],
        "mean",
    )
    _assert_values(
        result.column("isum"),
        [0, 0, 14, 20, 26],
        [False, False, True, True, True],
        "isum",
    )


def test_weighted_windows_reject_bad_weights() raises:
    var frame = DataFrame([Series("x", Column[Float64]([1, 2, 3]))])
    var two: List[Float64] = [1.0, 2.0]
    var zero: List[Float64] = [1.0, -1.0, 0.0]
    with assert_raises(contains="weights must hold window_size values"):
        _ = frame.select_exprs([col("x").rolling_sum(3, weights=two)])
    with assert_raises(contains="weights must not sum to zero"):
        _ = frame.select_exprs([col("x").rolling_mean(3, weights=zero)])
    with assert_raises(contains="weights must not sum to zero"):
        _ = (
            frame.lazy()
            .select_exprs([col("x").rolling_median(3, weights=zero)])
            .collect()
        )
    # A sum may have weights summing to zero.
    var diff = frame.select_exprs([col("x").rolling_sum(3, weights=zero)])
    assert_equal(diff.column("x").cast(DataType.FLOAT64).get(2).float64(), -1.0)


def test_weighted_variance_of_an_infinite_value_is_nan() raises:
    var frame = DataFrame(
        [Series("x", Column[Float64]([FloatLiteral.infinity, 1, 2]))]
    )
    var w: List[Float64] = [2.0, 1.0]
    var result = frame.select_exprs(
        [
            col("x").rolling_var(2, 1, weights=w).alias("var"),
            col("x").rolling_std(2, 1, weights=w).alias("std"),
        ]
    )
    for name in ["var", "std"]:
        var column = result.column(name)
        assert_true(isnan(column.get(0).float64()))
        assert_true(isnan(column.get(1).float64()))
        assert_true(column.get(2).float64() > 0)
