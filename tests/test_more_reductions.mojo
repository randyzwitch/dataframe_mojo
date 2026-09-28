"""The arg_min/arg_max, mode, skew, kurtosis, corr and cov reductions (#226).

Small cases pin Polars' semantics; the scripts/oracle.py `stat` operation
compares against Polars on random inputs. Large inputs here run on several
workers and through the parallel group-by paths, and are checked against
plain serial scans: arg_min/arg_max must keep the first occurrence across
worker boundaries, and the moment statistics must agree within tolerance.
"""
from std.math import isnan, sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    corr,
    cov,
    lit,
)
from dataframe.parallel import worker_count

comptime ROWS = 400_000


def nan() -> Float64:
    return Float64(0) / Float64(0)


def close(got: Float64, want: Float64, rel: Float64 = 1e-9) raises:
    if isnan(want):
        assert_true(isnan(got))
        return
    assert_true(abs(got - want) <= rel * max(1.0, abs(want)))


def big_frame() raises -> DataFrame:
    """Values with repeated extremes on both sides of worker boundaries,
    nulls, and one NaN; keys with few groups (k) and many (h)."""
    var x = List[Float64](capacity=ROWS)
    var x_valid = List[Bool](capacity=ROWS)
    var y = List[Float64](capacity=ROWS)
    var k = List[Int64](capacity=ROWS)
    var h = List[Int64](capacity=ROWS)
    for i in range(ROWS):
        var value = Float64((i * 7919) % 100_003) / 7.0
        if i == 150_000 or i == 300_000 or i == 399_999:
            value = -5.0
        if i == 250_000 or i == 350_000:
            value = 20_000.0
        if i == 5:
            value = nan()
        x.append(value)
        x_valid.append(i % 97 != 3)
        y.append(Float64((i * 104729) % 1_000_003) / 11.0 + value / 3.0)
        k.append(Int64(i % 3))
        h.append(Int64(i // 4))
    return DataFrame(
        [
            Series("x", Column[Float64](x^, x_valid^)),
            Series("y", Column[Float64](y^)),
            Series("k", Column[Int64](k^)),
            Series("h", Column[Int64](h^)),
        ]
    )


def scan_arg(
    frame: DataFrame, key: String, want_max: Bool
) raises -> Dict[Int64, Int]:
    """Reference: first best non-NaN value's index within each group."""
    ref x = frame.column("x")._data[Column[Float64]]
    ref keys = frame.column(key)._data[Column[Int64]]
    var seen = Dict[Int64, Int]()
    var best = Dict[Int64, Float64]()
    var at = Dict[Int64, Int]()
    for i in range(frame.height()):
        var g = keys._get(i)
        var index = seen.get(g, 0)
        seen[g] = index + 1
        if not x._valid(i) or isnan(x._get(i)):
            continue
        var value = x._get(i)
        if g not in best or (value > best[g] if want_max else value < best[g]):
            best[g] = value
            at[g] = index
    return at^


def test_arg_extremes_keep_first_across_workers() raises:
    var frame = big_frame()
    assert_true(worker_count(frame.height()) > 1)
    var result = frame.select_exprs(
        [col("x").arg_min().alias("lo"), col("x").arg_max().alias("hi")]
    )
    assert_equal(result.column("lo").dtype(), DataType.UINT32)
    var single = Dict[Int64, Int]()
    var ones = List[Int64](length=frame.height(), fill=0)
    var whole = frame.with_column(Series("one", Column[Int64](ones^)))
    var lo = scan_arg(whole, "one", False)
    var hi = scan_arg(whole, "one", True)
    assert_equal(Int(result.item(0, "lo").uint32()), lo[0])
    assert_equal(Int(result.item(0, "hi").uint32()), hi[0])
    assert_equal(lo[0], 150_000)
    assert_equal(hi[0], 250_000)
    _ = single^
    for key in ["k", "h"]:
        var grouped = frame.group_by(key).agg(
            [col("x").arg_min().alias("lo"), col("x").arg_max().alias("hi")]
        )
        var want_lo = scan_arg(frame, key, False)
        var want_hi = scan_arg(frame, key, True)
        for row in range(grouped.height()):
            var g = grouped.item(row, key).int64()
            assert_equal(Int(grouped.item(row, "lo").uint32()), want_lo[g])
            assert_equal(Int(grouped.item(row, "hi").uint32()), want_hi[g])
    # over() broadcasts each group's index to its rows.
    var windowed = frame.select(col("x").arg_max().over("k").alias("a"))
    var want = scan_arg(frame, "k", True)
    for row in [0, 1, 2, 399_999]:
        assert_equal(
            Int(windowed.item(row, "a").uint32()), want[Int64(row % 3)]
        )


def test_arg_extremes_semantics() raises:
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [3.0, 0.0, 1.0, nan(), 1.0], [True, False, True, True, True]
                ),
            ),
            Series("s", Column[String](["b", "a", "c", "a", "c"])),
            Series("b", Column[Bool]([False, True, True, False, True])),
            Series("u", Column[UInt64]([UInt64.MAX, 1, 0, UInt64.MAX, 7])),
        ]
    )
    var r = frame.select_exprs(
        [
            col("x").arg_min().alias("x_lo"),
            col("x").arg_max().alias("x_hi"),
            col("s").arg_min().alias("s_lo"),
            col("s").arg_max().alias("s_hi"),
            col("b").arg_max().alias("b_hi"),
            col("u").arg_max().alias("u_hi"),
            col("u").arg_min().alias("u_lo"),
        ]
    )
    assert_equal(Int(r.item(0, "x_lo").uint32()), 2)
    assert_equal(Int(r.item(0, "x_hi").uint32()), 0)
    assert_equal(Int(r.item(0, "s_lo").uint32()), 1)
    assert_equal(Int(r.item(0, "s_hi").uint32()), 2)
    assert_equal(Int(r.item(0, "b_hi").uint32()), 1)
    assert_equal(Int(r.item(0, "u_hi").uint32()), 0)
    assert_equal(Int(r.item(0, "u_lo").uint32()), 2)
    var only_nan = DataFrame(
        [Series("x", Column[Float64]([nan(), nan()]))]
    ).select(col("x").arg_min())
    assert_equal(Int(only_nan.item().uint32()), 0)
    var all_null = DataFrame(
        [Series("x", Column[Int64]([0, 0], [False, False]))]
    ).select(col("x").arg_min())
    assert_true(all_null.item().is_null())


def test_mode_rows_lists_nulls_and_types() raises:
    var frame = DataFrame(
        [
            Series("k", Column[String](["a", "a", "a", "b", "b", "c"])),
            Series(
                "n",
                Column[Int64](
                    [2, 1, 2, 0, 0, 7], [True, True, True, False, False, True]
                ),
            ),
            Series("f", Column[Float64]([-0.0, 0.0, nan(), nan(), 1.0, 2.0])),
            Series("u", Column[UInt64]([UInt64.MAX, UInt64.MAX, 1, 1, 3, 3])),
        ]
    )
    # 2 and null each occur twice.
    var modes = frame.select(col("n").mode())
    assert_equal(modes.height(), 2)
    assert_equal(modes.item(0, "n").int64(), 2)
    assert_true(modes.item(1, "n").is_null())
    # -0.0 and 0.0 are one value; NaN sorts above numbers.
    var floats = frame.select(col("f").mode())
    assert_equal(floats.height(), 2)
    assert_equal(floats.item(0, "f").float64(), 0.0)
    assert_true(isnan(floats.item(1, "f").float64()))
    var unsigned = frame.select(col("u").mode())
    assert_equal(unsigned.column("u").dtype(), DataType.UINT64)
    assert_equal(unsigned.height(), 3)
    assert_equal(unsigned.item(2, "u").uint64(), UInt64.MAX)
    var grouped = frame.group_by("k", maintain_order=True).agg(
        [col("n").mode()]
    )
    assert_equal(grouped.column("n").dtype(), DataType.list(DataType.INT64))
    var flat = grouped.explode("n")
    assert_equal(flat.height(), 3)
    assert_equal(flat.item(0, "n").int64(), 2)
    assert_true(flat.item(1, "n").is_null())
    assert_equal(flat.item(2, "n").int64(), 7)
    var empty = frame.clear().select(col("n").mode())
    assert_equal(empty.height(), 0)
    with assert_raises(contains="over()"):
        _ = frame.with_columns(col("n").mode().over("k"))
    with assert_raises(contains="row-valued"):
        _ = frame.select_exprs([col("n").mode(), col("k")])
    var dates = DataFrame(
        [Series("d", Column[Int64]([3, 3, 9])).with_dtype(DataType.DATE)]
    ).select(col("d").mode())
    assert_equal(dates.column("d").dtype(), DataType.DATE)
    assert_equal(dates.height(), 1)


def moments(values: List[Float64]) -> Tuple[Float64, Float64, Float64, Float64]:
    """Two-pass reference: count, m2, m3, m4 about the mean."""
    var n = Float64(len(values))
    var total = 0.0
    for v in values:
        total += v
    var mean = total / n
    var m2 = 0.0
    var m3 = 0.0
    var m4 = 0.0
    for v in values:
        var d = v - mean
        m2 += d * d
        m3 += d * d * d
        m4 += d * d * d * d
    return (n, m2, m3, m4)


def test_skew_kurtosis_parallel_match_two_pass() raises:
    var frame = big_frame()
    var values = List[Float64]()
    ref x = frame.column("y")._data[Column[Float64]]
    for i in range(frame.height()):
        values.append(x._get(i))
    var m = moments(values)
    var n = m[0]
    var g1 = sqrt(n) * m[2] / (m[1] * sqrt(m[1]))
    var g2 = n * m[3] / (m[1] * m[1]) - 3
    var r = frame.select_exprs(
        [
            col("y").skew().alias("s"),
            col("y").skew(bias=False).alias("su"),
            col("y").kurtosis().alias("k"),
            col("y").kurtosis(fisher=False, bias=False).alias("kp"),
        ]
    )
    close(r.item(0, "s").float64(), g1)
    close(r.item(0, "su").float64(), g1 * sqrt(n * (n - 1)) / (n - 2))
    close(r.item(0, "k").float64(), g2)
    close(
        r.item(0, "kp").float64(),
        ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3)) + 3,
    )
    # Group paths (few groups via worker ranges, many via partitions) agree
    # with the whole-frame result for a group holding every row.
    var ones = List[Int64](length=frame.height(), fill=0)
    var one = frame.with_column(Series("one", Column[Int64](ones^)))
    var grouped = one.group_by("one").agg([col("y").skew().alias("s")])
    close(grouped.item(0, "s").float64(), g1)


def test_skew_kurtosis_small_semantics() raises:
    var frame = DataFrame([Series("v", Column[Float64]([1.0, 2.0, 4.0, 9.0]))])
    var r = frame.select_exprs(
        [
            col("v").kurtosis().alias("k"),
            col("v").kurtosis(bias=False).alias("ku"),
            col("v").kurtosis(fisher=False).alias("kp"),
        ]
    )
    close(r.item(0, "k").float64(), -1.0)
    close(r.item(0, "ku").float64(), 1.5)
    close(r.item(0, "kp").float64(), 2.0)
    var three = DataFrame([Series("v", Column[Float64]([1.0, 2.0, 4.0]))])
    close(three.select(col("v").skew()).item().float64(), 0.3818017741606059)
    close(
        three.select(col("v").skew(bias=False)).item().float64(),
        0.9352195295828235,
    )
    assert_true(three.select(col("v").kurtosis(bias=False)).item().is_null())
    var constant = DataFrame([Series("v", Column[Float64]([2.0, 2.0, 2.0]))])
    assert_true(isnan(constant.select(col("v").skew()).item().float64()))
    var empty = constant.clear()
    assert_true(empty.select(col("v").skew()).item().is_null())
    var unsigned = DataFrame([Series("u", Column[UInt64]([1, 2, 4]))]).select(
        col("u").skew()
    )
    close(unsigned.item().float64(), 0.3818017741606059)
    with assert_raises(contains="numeric"):
        _ = DataFrame([Series("s", Column[String](["a"]))]).select(
            col("s").skew()
        )


def test_corr_cov_parallel_match_two_pass() raises:
    var frame = big_frame()
    ref x = frame.column("x")._data[Column[Float64]]
    ref y = frame.column("y")._data[Column[Float64]]
    var xs = List[Float64]()
    var ys = List[Float64]()
    for i in range(frame.height()):
        if x._valid(i) and not isnan(x._get(i)):
            xs.append(x._get(i))
            ys.append(y._get(i))
    var n = Float64(len(xs))
    var mx = 0.0
    var my = 0.0
    for i in range(len(xs)):
        mx += xs[i]
        my += ys[i]
    mx /= n
    my /= n
    var cxy = 0.0
    var sxx = 0.0
    var syy = 0.0
    for i in range(len(xs)):
        cxy += (xs[i] - mx) * (ys[i] - my)
        sxx += (xs[i] - mx) * (xs[i] - mx)
        syy += (ys[i] - my) * (ys[i] - my)
    var clean = frame.filter(~col("x").is_nan())
    var r = clean.select_exprs(
        [
            corr(col("x"), col("y")).alias("c"),
            cov(col("x"), col("y")).alias("v"),
            cov(col("x"), col("y"), ddof=0).alias("v0"),
        ]
    )
    close(r.item(0, "c").float64(), cxy / sqrt(sxx * syy))
    close(r.item(0, "v").float64(), cxy / (n - 1))
    close(r.item(0, "v0").float64(), cxy / n)
    # A NaN in a pair makes Pearson NaN, as in Polars.
    assert_true(isnan(frame.select(corr(col("x"), col("y"))).item().float64()))
    # Grouped paths agree with per-group two-pass values.
    var grouped = clean.group_by("k").agg([cov(col("x"), col("y")).alias("v")])
    assert_equal(grouped.height(), 3)
    ref keys = frame.column("k")._data[Column[Int64]]
    for row in range(grouped.height()):
        var g = grouped.item(row, "k").int64()
        var gx = List[Float64]()
        var gy = List[Float64]()
        for i in range(frame.height()):
            if keys._get(i) == g and x._valid(i) and not isnan(x._get(i)):
                gx.append(x._get(i))
                gy.append(y._get(i))
        var count = Float64(len(gx))
        var ax = 0.0
        var ay = 0.0
        for i in range(len(gx)):
            ax += gx[i]
            ay += gy[i]
        ax /= count
        ay /= count
        var c = 0.0
        for i in range(len(gx)):
            c += (gx[i] - ax) * (gy[i] - ay)
        close(grouped.item(row, "v").float64(), c / (count - 1))


def test_corr_cov_pairwise_spearman_and_errors() raises:
    var frame = DataFrame(
        [
            Series(
                "a",
                Column[Float64](
                    [1.0, 2.0, 0.0, 4.0], [True, True, False, True]
                ),
            ),
            Series(
                "b",
                Column[Float64](
                    [2.0, 0.0, 3.0, 8.0], [True, False, True, True]
                ),
            ),
            Series("s", Column[String](["x", "y", "z", "w"])),
        ]
    )
    var r = frame.select_exprs(
        [
            corr(col("a"), col("b")).alias("c"),
            cov(col("a"), col("b")).alias("v"),
        ]
    )
    close(r.item(0, "c").float64(), 1.0)
    close(r.item(0, "v").float64(), 9.0)
    var ties = DataFrame(
        [
            Series("a", Column[Float64]([1.0, 1.0, 2.0, 3.0])),
            Series("b", Column[Int64]([5, 6, 6, 9])),
        ]
    )
    close(
        ties.select(corr(col("a"), col("b"), method="spearman"))
        .item()
        .float64(),
        0.8333333333333334,
    )
    var one = frame.head(1)
    assert_true(isnan(one.select(corr(col("a"), col("b"))).item().float64()))
    assert_equal(one.select(cov(col("a"), col("b"))).item().float64(), 0.0)
    assert_true(frame.clear().select(cov(col("a"), col("b"))).item().is_null())
    with assert_raises(contains="numeric"):
        _ = frame.select(corr(col("a"), col("s")))
    with assert_raises(contains="pearson or spearman"):
        _ = frame.select(corr(col("a"), col("b"), method="kendall"))
    with assert_raises(contains="ddof"):
        _ = frame.select(cov(col("a"), col("b"), ddof=-1))


def test_streaming_lazy_matches_eager() raises:
    var frame = big_frame().filter(~col("x").is_nan())
    var exprs: List[Expr] = [
        col("x").arg_min().alias("lo"),
        col("x").skew().alias("s"),
        corr(col("x"), col("y")).alias("c"),
    ]
    var eager = frame.group_by("k", maintain_order=True).agg(exprs)
    var lazy = (
        frame.lazy()
        .group_by(["k"], maintain_order=True)
        .agg(exprs)
        .collect(batch_size=50_000)
    )
    assert_equal(lazy.height(), eager.height())
    for row in range(eager.height()):
        var g = eager.item(row, "k").int64()
        var other = -1
        for j in range(lazy.height()):
            if lazy.item(j, "k").int64() == g:
                other = j
        assert_equal(
            Int(lazy.item(other, "lo").uint32()),
            Int(eager.item(row, "lo").uint32()),
        )
        close(lazy.item(other, "s").float64(), eager.item(row, "s").float64())
        close(lazy.item(other, "c").float64(), eager.item(row, "c").float64())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
