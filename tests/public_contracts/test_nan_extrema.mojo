"""NaN in min, max, cum_min, cum_max, rolling_min, rolling_max and describe (#816).

The expected values are what Polars 2.0 gives for the same input (`min`, `max`,
`cum_min`, `cum_max`, `rolling_min`, `rolling_max`, `describe`), each also
taken `over` a partition and grouped. min and max skip NaN and are NaN only
when every valid value is NaN; a running min or max repeats the value so far on
a NaN row, which is NaN only before the first number; a rolling window holding
NaN is NaN. DuckDB treats NaN as the largest value instead, so its max is NaN
if any value is; this library follows Polars.
"""
# contract-group: reduce.global
# contract-group: reduce.grouped
# contract-group: expr.order_windows
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from tests.support.coverage_compare import assert_float

# Each token is one value: n is null, N is NaN, anything else a number.
# `mn`/`mx` are the global min and max, `cmin`/`cmax` the cumulative forms
# (r for reverse), `rmin`/`rmax` rolling_min/max(3, min_samples=2), and
# `dmin`/`dmax` what describe() reports.


@fieldwise_init
struct _Case(Copyable, Movable):
    var name: String
    var x: String
    var mn: String
    var mx: String
    var cmin: String
    var cmax: String
    var rcmin: String
    var rcmax: String
    var rmin: String
    var rmax: String
    var dmin: String
    var dmax: String


def _cases() -> List[_Case]:
    return [
        _Case(
            "first",
            "N,2,1,3",
            "1.0",
            "3.0",
            "N,2.0,1.0,1.0",
            "N,2.0,2.0,3.0",
            "1.0,1.0,1.0,3.0",
            "3.0,3.0,3.0,3.0",
            "n,N,N,1.0",
            "n,N,N,3.0",
            "1.0",
            "3.0",
        ),
        _Case(
            "middle",
            "2,N,1,3",
            "1.0",
            "3.0",
            "2.0,2.0,1.0,1.0",
            "2.0,2.0,2.0,3.0",
            "1.0,1.0,1.0,3.0",
            "3.0,3.0,3.0,3.0",
            "n,N,N,N",
            "n,N,N,N",
            "1.0",
            "3.0",
        ),
        _Case(
            "last",
            "2,1,3,N",
            "1.0",
            "3.0",
            "2.0,1.0,1.0,1.0",
            "2.0,2.0,3.0,3.0",
            "1.0,1.0,3.0,N",
            "3.0,3.0,3.0,N",
            "n,1.0,1.0,N",
            "n,2.0,3.0,N",
            "1.0",
            "3.0",
        ),
        _Case(
            "all_nan",
            "N,N",
            "N",
            "N",
            "N,N",
            "N,N",
            "N,N",
            "N,N",
            "n,N",
            "n,N",
            "N",
            "N",
        ),
        _Case(
            "nan_and_null",
            "n,N,2,n",
            "2.0",
            "2.0",
            "n,N,2.0,n",
            "n,N,2.0,n",
            "n,2.0,2.0,n",
            "n,2.0,2.0,n",
            "n,n,N,N",
            "n,n,N,N",
            "2.0",
            "2.0",
        ),
        _Case(
            "only_nan_and_null",
            "n,N,n",
            "N",
            "N",
            "n,N,n",
            "n,N,n",
            "n,N,n",
            "n,N,n",
            "n,n,n",
            "n,n,n",
            "N",
            "N",
        ),
        _Case(
            "many_nan",
            "N,N,5,N,4,N",
            "4.0",
            "5.0",
            "N,N,5.0,5.0,4.0,4.0",
            "N,N,5.0,5.0,5.0,5.0",
            "4.0,4.0,4.0,4.0,4.0,N",
            "5.0,5.0,5.0,4.0,4.0,N",
            "n,N,N,N,N,N",
            "n,N,N,N,N,N",
            "4.0",
            "5.0",
        ),
        _Case(
            "infinities",
            "N,-inf,inf,N",
            "-inf",
            "inf",
            "N,-inf,-inf,-inf",
            "N,-inf,inf,inf",
            "-inf,-inf,inf,N",
            "inf,inf,inf,N",
            "n,N,N,N",
            "n,N,N,N",
            "-inf",
            "inf",
        ),
        _Case(
            "no_valid",
            "n,n",
            "n",
            "n",
            "n,n",
            "n,n",
            "n,n",
            "n,n",
            "n,n",
            "n,n",
            "n",
            "n",
        ),
    ]


def _tokens(text: String) -> List[String]:
    var out = List[String]()
    for token in text.split(","):
        out.append(String(token))
    return out^


def _number(token: String) raises -> Float64:
    if token == "N":
        return Float64(0) / Float64(0)
    if token == "inf":
        return Float64(1) / Float64(0)
    if token == "-inf":
        return -Float64(1) / Float64(0)
    return Float64(token)


def _join(left: String, right: String) -> String:
    return left + "," + right if left.byte_length() else right


def _repeat(token: String, count: Int) -> String:
    var out = token
    for _ in range(count - 1):
        out += "," + token
    return out


def _frame[D: DType](x: String, groups: String) raises -> DataFrame:
    var values = List[Scalar[D]]()
    var valid = List[Bool]()
    for token in _tokens(x):
        valid.append(token != "n")
        values.append(
            _number(token).cast[D]() if token != "n" else Scalar[D](0)
        )
    var ids = List[Int64]()
    for token in _tokens(groups):
        ids.append(Int64(Int(token)))
    return DataFrame(
        [
            Series("x", Column[Scalar[D]](values^, valid)),
            Series("g", Column[Int64](ids^)),
        ]
    )


def _expect[D: DType](series: Series, expected: String, label: String) raises:
    var tokens = _tokens(expected)
    assert_true(series.dtype() == DataType.of(D), label + ": dtype")
    assert_equal(len(series), len(tokens), label + ": length")
    for i in range(len(tokens)):
        var value = series.get(i)
        if tokens[i] == "n":
            assert_true(value.is_null(), label + ": row " + String(i) + " null")
        else:
            assert_true(
                not value.is_null(), label + ": row " + String(i) + " valid"
            )
            assert_float(Float64(value.numeric[D]()), _number(tokens[i]))


def _run(
    frame: DataFrame, exprs: List[Expr], lazy: Bool, by: Bool = False
) raises -> DataFrame:
    if by:
        if lazy:
            return frame.lazy().group_by("g").agg(exprs).collect()
        return frame.group_by("g").agg(exprs)
    if lazy:
        return frame.lazy().select_exprs(exprs).collect()
    return frame.select_exprs(exprs)


def _check_extrema[D: DType](c: _Case, lazy: Bool) raises:
    var frame = _frame[D](c.x, _repeat("0", len(_tokens(c.x))))
    var n = len(_tokens(c.x))
    var label = c.name + (" lazy" if lazy else " eager")
    var reductions: List[Expr] = [
        col("x").min().alias("mn"),
        col("x").max().alias("mx"),
    ]
    var whole = _run(frame, reductions, lazy)
    _expect[D](whole.column("mn"), c.mn, label + " min")
    _expect[D](whole.column("mx"), c.mx, label + " max")
    var grouped = _run(frame, reductions, lazy, by=True)
    _expect[D](grouped.column("mn"), c.mn, label + " grouped min")
    _expect[D](grouped.column("mx"), c.mx, label + " grouped max")
    var over = _run(
        frame,
        [
            col("x").min().over("g").alias("mn"),
            col("x").max().over("g").alias("mx"),
        ],
        lazy,
    )
    _expect[D](over.column("mn"), _repeat(c.mn, n), label + " min over")
    _expect[D](over.column("mx"), _repeat(c.mx, n), label + " max over")


def _check_windows[D: DType](c: _Case, lazy: Bool, partitioned: Bool) raises:
    var frame = _frame[D](c.x, _repeat("0", len(_tokens(c.x))))
    var label = c.name + (" lazy" if lazy else " eager")
    label += " over" if partitioned else ""
    var exprs: List[Expr] = [
        col("x").cum_min(),
        col("x").cum_max(),
        col("x").cum_min(reverse=True),
        col("x").cum_max(reverse=True),
        col("x").rolling_min(3, min_samples=2),
        col("x").rolling_max(3, min_samples=2),
    ]
    var names: List[String] = [
        "cmin",
        "cmax",
        "rcmin",
        "rcmax",
        "rmin",
        "rmax",
    ]
    var expected: List[String] = [
        c.cmin,
        c.cmax,
        c.rcmin,
        c.rcmax,
        c.rmin,
        c.rmax,
    ]
    var chosen = List[Expr]()
    for i in range(len(exprs)):
        var expression = exprs[i].copy()
        if partitioned:
            expression = expression.over("g")
        chosen.append(expression.alias(names[i]))
    var result = _run(frame, chosen, lazy)
    for i in range(len(names)):
        _expect[D](result.column(names[i]), expected[i], label + " " + names[i])


def _check_describe[D: DType](c: _Case, lazy: Bool) raises:
    var frame = _frame[D](c.x, _repeat("0", len(_tokens(c.x)))).select(["x"])
    var summary = frame.lazy().describe() if lazy else frame.describe()
    var labels = summary.column("statistic")
    for row in range(len(labels)):
        var name = labels.get(row).string()
        if name != "min" and name != "max":
            continue
        var value = summary.item(row, "x")
        var token = c.dmin if name == "min" else c.dmax
        var label = (
            c.name + (" lazy" if lazy else " eager") + " describe " + name
        )
        if token == "n":
            assert_true(value.is_null(), label)
        else:
            assert_true(not value.is_null(), label)
            assert_float(value.float64(), _number(token))


def _all[D: DType]() raises:
    for c in _cases():
        for lazy in [False, True]:
            _check_extrema[D](c, lazy)
            _check_windows[D](c, lazy, False)
            _check_windows[D](c, lazy, True)
            _check_describe[D](c, lazy)


def test_float64_nan_in_extrema_and_running_windows_follows_polars() raises:
    _all[DType.float64]()


def test_float32_nan_in_extrema_and_running_windows_follows_polars() raises:
    _all[DType.float32]()


def _many[D: DType](repeat: Int) raises:
    """Every case once per repetition, each in its own group (and so its own
    partition): enough groups and rows to leave the small-input paths."""
    var x = String("")
    var g = String("")
    var count = 0
    for _ in range(repeat):
        for c in _cases():
            var n = len(_tokens(c.x))
            x = _join(x, c.x)
            g = _join(g, _repeat(String(count), n))
            count += 1
    var frame = _frame[D](x, g)
    var mn = String("")
    var mx = String("")
    var by_row = List[String]()
    for _ in range(8):
        by_row.append(String(""))
    for _ in range(repeat):
        for c in _cases():
            mn = _join(mn, c.mn)
            mx = _join(mx, c.mx)
            var parts: List[String] = [
                c.cmin,
                c.cmax,
                c.rcmin,
                c.rcmax,
                c.rmin,
                c.rmax,
                _repeat(c.mn, len(_tokens(c.x))),
                _repeat(c.mx, len(_tokens(c.x))),
            ]
            for k in range(len(parts)):
                by_row[k] = _join(by_row[k], parts[k])
    for lazy in [False, True]:
        var grouped = _run(
            frame,
            [col("x").min().alias("mn"), col("x").max().alias("mx")],
            lazy,
            by=True,
        )
        # Group order is unspecified; sort by the key to line them up.
        var ordered = grouped.sort(["g"])
        _expect[D](ordered.column("mn"), mn, "many grouped min")
        _expect[D](ordered.column("mx"), mx, "many grouped max")
        var windows = _run(
            frame,
            [
                col("x").cum_min().over("g").alias("cmin"),
                col("x").cum_max().over("g").alias("cmax"),
                col("x").cum_min(reverse=True).over("g").alias("rcmin"),
                col("x").cum_max(reverse=True).over("g").alias("rcmax"),
                col("x").rolling_min(3, min_samples=2).over("g").alias("rmin"),
                col("x").rolling_max(3, min_samples=2).over("g").alias("rmax"),
                col("x").min().over("g").alias("omn"),
                col("x").max().over("g").alias("omx"),
            ],
            lazy,
        )
        var names: List[String] = [
            "cmin",
            "cmax",
            "rcmin",
            "rcmax",
            "rmin",
            "rmax",
            "omn",
            "omx",
        ]
        for k in range(len(names)):
            _expect[D](windows.column(names[k]), by_row[k], "many " + names[k])


def test_nan_extrema_hold_across_many_groups_and_partitions() raises:
    _many[DType.float64](600)
    _many[DType.float32](600)
