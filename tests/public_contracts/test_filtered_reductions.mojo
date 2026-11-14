"""Ungrouped reductions of conditional inputs: a reduction of
`when(c).then(x)` with no otherwise sees only the rows where `c` holds
(true and not null), and `x` is never evaluated on any other row, so it
cannot raise for one. Results equal filtering first and reducing after, for
every layout, with nulls in the condition and the input, decimals, filters
that keep no row, and several reductions sharing one condition."""
# contract-group: reduce.global
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    LazyFrame,
    Series,
    col,
    lit,
    when,
)
from dataframe.lazy import WITH_COLUMNS
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import chunk_layout, offset_layout


comptime _ROWS = 5000


def _frame() raises -> DataFrame:
    """Several chunks of rows: a quantity with nulls, a float and an
    integer measure with nulls, a decimal measure, and a large integer that
    overflows when quadrupled on the rows quantity 7 marks."""
    var quantity = List[Int64]()
    var quantity_valid = List[Bool]()
    var price = List[Float64]()
    var price_valid = List[Bool]()
    var units = List[Int64]()
    var amount = List[Int64]()
    var big = List[Int64]()
    for row in range(_ROWS):
        var q = Int64((row * 7) % 40)
        quantity.append(q)
        quantity_valid.append(row % 17 != 0)
        price.append(Float64(row % 23) * 1.25)
        price_valid.append(row % 11 != 3)
        units.append(Int64((row * 13) % 97) - 40)
        amount.append(Int64(row % 1000) * 7 - 2000)
        big.append(Int64.MAX // 2 if q == 7 else Int64(row))
    var amounts = Series("amount", Column[Int64](amount^)).cast(
        DataType.decimal(12, 2, 64)
    )
    return DataFrame(
        [
            Series("quantity", Column[Int64](quantity^, quantity_valid)),
            Series("price", Column[Float64](price^, price_valid)),
            Series("units", Column[Int64](units^)),
            amounts^,
            Series("big", Column[Int64](big^)),
        ]
    )


def _layouts(frame: DataFrame) raises -> List[DataFrame]:
    return [frame.copy(), offset_layout(frame), chunk_layout(frame)]


def _band(low: Int, high: Int) raises -> Expr:
    return col("quantity").is_between(lit(Int64(low)), lit(Int64(high)))


def _conditional(low: Int, high: Int) raises -> List[Expr]:
    """Reductions of conditional inputs over one band: bare columns,
    computed ones and a decimal."""
    var band = _band(low, high)
    var tag = String(low)
    return [
        when(band).then(col("price")).end().mean().alias("mean" + tag),
        when(band).then(col("price")).end().count().alias("count" + tag),
        when(band)
        .then(col("units") * lit(Int64(3)))
        .end()
        .sum()
        .alias("sum" + tag),
        when(band).then(col("units")).end().min().alias("min" + tag),
        when(band)
        .then(col("price").cast(DataType.FLOAT64) * lit(2.0))
        .end()
        .max()
        .alias("max" + tag),
        when(band).then(col("amount")).end().sum().alias("dsum" + tag),
        when(band)
        .then(col("amount") + col("amount"))
        .end()
        .sum()
        .alias("dtwice" + tag),
    ]


def _filtered(frame: DataFrame, low: Int, high: Int) raises -> DataFrame:
    """The same reductions over the rows the band keeps, filtered first."""
    var tag = String(low)
    return (
        frame.lazy()
        .filter(_band(low, high))
        .select_exprs(
            [
                col("price").mean().alias("mean" + tag),
                col("price").count().alias("count" + tag),
                (col("units") * lit(Int64(3))).sum().alias("sum" + tag),
                col("units").min().alias("min" + tag),
                (col("price").cast(DataType.FLOAT64) * lit(2.0))
                .max()
                .alias("max" + tag),
                col("amount").sum().alias("dsum" + tag),
                (col("amount") + col("amount")).sum().alias("dtwice" + tag),
            ]
        )
        .collect(optimize=False)
    )


def test_conditional_reductions_equal_filtering_first() raises:
    for frame in _layouts(_frame()):
        for band in [(0, 9), (10, 19), (30, 39)]:
            var low = band[0]
            var high = band[1]
            var expected = _filtered(frame, low, high)
            var plan = frame.lazy().select_exprs(_conditional(low, high))
            for optimize in [False, True]:
                assert_frame(plan.collect(optimize=optimize), expected)
            assert_frame(frame.select_exprs(_conditional(low, high)), expected)


def test_reductions_sharing_a_condition_each_see_its_rows() raises:
    """Several bands at once, each condition read by several reductions."""
    for frame in _layouts(_frame()):
        var exprs = List[Expr]()
        var parts = List[Series]()
        for band in [(0, 9), (10, 19), (20, 29)]:
            for e in _conditional(band[0], band[1]):
                exprs.append(e.copy())
            var one = _filtered(frame, band[0], band[1])
            for c in one.columns():
                parts.append(one.column(c))
        var expected = DataFrame(parts^)
        var plan = frame.lazy().select_exprs(exprs)
        for optimize in [False, True]:
            assert_frame(plan.collect(optimize=optimize), expected)


def test_a_condition_that_keeps_no_row() raises:
    """No row kept: null mean, min and max, zero count and sums, as the
    reductions of no rows."""
    for frame in _layouts(_frame()):
        var plan = frame.lazy().select_exprs(_conditional(100, 200))
        var expected = _filtered(frame, 100, 200)
        assert_equal(expected.select(["count100"]).item().int64(), 0)
        assert_true(expected.select(["mean100"]).item().is_null())
        for optimize in [False, True]:
            assert_frame(plan.collect(optimize=optimize), expected)


def _raises_overflow(plan: LazyFrame, optimize: Bool) raises -> Bool:
    try:
        _ = plan.collect(optimize=optimize)
    except error:
        if String(error).find("overflow") < 0:
            raise error
        return True
    return False


def test_an_input_is_never_evaluated_on_a_row_its_condition_drops() raises:
    """Quadrupling `big` overflows only where quantity is 7: a band
    without 7 reduces it without error, also when two reductions read the
    same computed input; a band with 7 raises."""
    for frame in _layouts(_frame()):
        var outside = when(_band(10, 19)).then(col("big") * lit(Int64(4)))
        var safe = frame.lazy().select_exprs(
            [
                outside.copy().end().sum().alias("sum"),
                outside.copy().end().max().alias("max"),
                outside.copy().end().count().alias("count"),
            ]
        )
        var expected = (
            frame.lazy()
            .filter(_band(10, 19))
            .select_exprs(
                [
                    (col("big") * lit(Int64(4))).sum().alias("sum"),
                    (col("big") * lit(Int64(4))).max().alias("max"),
                    (col("big") * lit(Int64(4))).count().alias("count"),
                ]
            )
            .collect(optimize=False)
        )
        for optimize in [False, True]:
            assert_frame(safe.collect(optimize=optimize), expected)
        var inside = when(_band(0, 9)).then(col("big") * lit(Int64(4)))
        var bad = frame.lazy().select_exprs(
            [
                inside.copy().end().sum().alias("sum"),
                inside.copy().end().max().alias("max"),
            ]
        )
        for optimize in [False, True]:
            assert_true(_raises_overflow(bad, optimize))


def _shared(plan: LazyFrame) raises -> List[String]:
    """The names each with_columns of the optimized plan computes."""
    var names = List[String]()
    for node in plan._optimized()._plan._nodes:
        if node.kind == WITH_COLUMNS:
            for e in node.exprs:
                names.append(e._name)
    return names^


def test_only_a_condition_is_computed_once_for_every_row() raises:
    """A condition read by several reductions is computed once; a branch
    value repeated in several conditionals is not, since computing it for
    every row would evaluate it where its condition drops the row."""
    var frame = _frame()
    var band = _band(10, 19)
    var value = col("big") * lit(Int64(4))
    var plan = frame.lazy().select_exprs(
        [
            when(band.copy()).then(value.copy()).end().sum().alias("a"),
            when(band.copy()).then(value.copy()).end().max().alias("b"),
        ]
    )
    var shared = _shared(plan)
    assert_equal(len(shared), 1)
    assert_true(shared[0].startswith("__cse_"))
    # The condition is read on every row; the value only on band rows.
    assert_frame(plan.collect(), plan.collect(optimize=False))


def test_a_conditional_over_reductions_of_conditional_inputs() raises:
    """A count of a band's rows picks which of two filtered means each
    band reports: every reduction reads the band, the means only at its
    rows."""
    for frame in _layouts(_frame()):
        var exprs = List[Expr]()
        var parts = List[Series]()
        for band in [(0, 9), (10, 19), (20, 29), (30, 39)]:
            var low = band[0]
            var high = band[1]
            var inside = _band(low, high)
            var hits = (
                when(inside.copy())
                .then(lit(Int64(1)))
                .otherwise(lit(Int64(0)))
                .sum()
            )
            exprs.append(
                when(hits > lit(Int64(500)))
                .then(
                    when(inside.copy())
                    .then(col("price").cast(DataType.FLOAT64))
                    .end()
                    .mean()
                )
                .otherwise(
                    when(inside.copy())
                    .then(col("units").cast(DataType.FLOAT64))
                    .end()
                    .mean()
                )
                .alias("b" + String(low))
            )
            var kept = frame.lazy().filter(inside.copy())
            var reduced = kept.select_exprs(
                [
                    col("price").cast(DataType.FLOAT64).mean().alias("p"),
                    col("units").cast(DataType.FLOAT64).mean().alias("u"),
                ]
            ).collect(optimize=False)
            var pick = "p" if kept.collect().height() > 500 else "u"
            parts.append(reduced.column(pick).renamed("b" + String(low)))
        var expected = DataFrame(parts^)
        var plan = frame.lazy().select_exprs(exprs)
        for optimize in [False, True]:
            assert_frame(plan.collect(optimize=optimize), expected)
        assert_frame(frame.select_exprs(exprs), expected)
