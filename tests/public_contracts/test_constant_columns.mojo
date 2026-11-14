"""A scalar spread over every row is kept as one value (a constant column)
through the operations that only need the value, and reads exactly like the
same rows written out."""
# contract-group: expr.constant_columns
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    concat,
    lit,
    null,
    when,
)

comptime N = 5003


def _frame() raises -> DataFrame:
    var x = List[Int64](capacity=N)
    for i in range(N):
        x.append(Int64(i))
    return DataFrame([Series("x", Column[Int64](x^))])


def _values() raises -> List[Expr]:
    """One constant of every dtype family, valid and null."""
    return [
        lit(Int64(7)),
        lit(Int32(-3)),
        lit(UInt8(200)),
        lit(Float64(2.5)),
        lit(Float64(0.0)) / lit(Float64(0.0)),
        lit(Float32(-1.25)),
        lit(True),
        lit(False),
        lit("ab"),
        lit("a value longer than twelve bytes"),
        lit(""),
        lit(Int64(1234)).cast(DataType.decimal(10, 2)),
        lit(Int64(-56)).cast(DataType.decimal(9, 1, 64)),
        lit(Int64(19000)).cast(DataType.DATE),
        null(DataType.INT64),
        null(DataType.FLOAT64),
        null(DataType.BOOL),
        null(DataType.STRING),
        null(DataType.decimal(10, 2)),
    ]


def _written_out(frame: DataFrame, value: Expr) raises -> Series:
    """The same rows as a column a kernel writes row by row."""
    return frame.with_columns(
        when(col("x") >= lit(Int64(0))).then(value).otherwise(value).alias("c")
    ).column("c")


def _check(constant: Series, expected: Series) raises:
    assert_true(constant.is_constant())
    assert_equal(len(constant), len(expected))
    assert_equal(constant.dtype(), expected.dtype())
    assert_equal(constant.null_count(), expected.null_count())
    assert_true(constant.equals(expected))
    for i in [0, len(expected) // 2, len(expected) - 1]:
        if i < len(expected):
            assert_equal(String(constant.get(i)), String(expected.get(i)))
    # Reading the rows wrote them out; the column is still the constant.
    assert_true(constant.is_constant())


def test_constant_columns_every_dtype_family_eager_and_lazy() raises:
    var frame = _frame()
    for value in _values():
        var expected = _written_out(frame, value)
        assert_false(expected.is_constant())
        var eager = frame.with_columns(value.alias("c")).column("c")
        _check(eager, expected)
        var lazy = (
            frame.lazy().with_columns(value.alias("c")).collect().column("c")
        )
        _check(lazy, expected)
        var selected = frame.select_exprs([col("x"), value.alias("c")]).column(
            "c"
        )
        _check(selected, expected)


def test_constant_columns_through_slices_takes_and_filters() raises:
    var frame = _frame()
    var rows = List[Int]()
    for i in range(0, N, 7):
        rows.append(N - 1 - i)
    for value in _values():
        var with_c = frame.with_columns(value.alias("c"))
        var expected = with_c.with_columns(
            when(col("x") >= lit(Int64(0)))
            .then(value)
            .otherwise(value)
            .alias("w")
        )
        _check(
            with_c.slice(11, 2049).column("c"),
            expected.slice(11, 2049).column("w"),
        )
        _check(with_c.head(3).column("c"), expected.head(3).column("w"))
        _check(with_c.take(rows).column("c"), expected.take(rows).column("w"))
        var keep = col("x") % lit(Int64(3)) == lit(Int64(0))
        _check(
            with_c.filter(keep).column("c"),
            expected.filter(keep).column("w"),
        )
        _check(
            with_c.lazy().filter(keep).collect().column("c"),
            expected.filter(keep).column("w"),
        )
        _check(
            frame.lazy()
            .with_columns(value.alias("c"))
            .filter(keep)
            .select_exprs([col("c"), col("x")])
            .collect()
            .column("c"),
            expected.filter(keep).column("w"),
        )
        # Every row filtered out: an empty column.
        var none = with_c.filter(col("x") < lit(Int64(0))).column("c")
        assert_equal(len(none), 0)


def test_constant_columns_concatenate() raises:
    var frame = _frame()
    for value in _values():
        var a = frame.with_columns(value.alias("c"))
        var b = frame.head(100).with_columns(value.alias("c"))
        var expected = _written_out(
            concat([frame.copy(), frame.head(100)]), value
        )
        _check(concat([a.copy(), b.copy()]).column("c"), expected)
    # Different values join written out, in order.
    var a = frame.head(10).with_columns(lit(Int64(1)).alias("c"))
    var b = frame.head(5).with_columns(lit(Int64(2)).alias("c"))
    var c = frame.head(5).with_columns(null(DataType.INT64).alias("c"))
    var joined = concat([a.copy(), b.copy(), c.copy()]).column("c")
    assert_false(joined.is_constant())
    assert_equal(len(joined), 20)
    assert_equal(joined.get(9).int64(), 1)
    assert_equal(joined.get(10).int64(), 2)
    assert_true(joined.get(15).is_null())
    assert_equal(joined.null_count(), 5)


def test_constant_columns_in_arithmetic() raises:
    var frame = _frame().with_columns(
        [
            lit(Int64(7)).alias("c"),
            lit(Float64(1.5)).alias("f"),
            null(DataType.INT64).alias("n"),
        ]
    )
    # A constant with a column: as the literal would.
    var pairs: List[Tuple[Expr, Expr]] = [
        (col("c") + col("x"), lit(Int64(7)) + col("x")),
        (col("x") - col("c"), col("x") - lit(Int64(7))),
        (col("c") * col("x"), lit(Int64(7)) * col("x")),
        (col("x") > col("c"), col("x") > lit(Int64(7))),
        (
            col("f") * col("x").cast(DataType.FLOAT64),
            lit(Float64(1.5)) * col("x").cast(DataType.FLOAT64),
        ),
        (col("n") + col("x"), null(DataType.INT64) + col("x")),
        (col("x") // col("c"), col("x") // lit(Int64(7))),
    ]
    for pair in pairs:
        var got = frame.select(pair[0].alias("r")).column("r")
        var want = frame.select(pair[1].alias("r")).column("r")
        assert_false(got.is_constant())
        assert_true(got.equals(want))
        var lazy = frame.lazy().select(pair[0].alias("r")).collect()
        assert_true(lazy.column("r").equals(want))
    # Constants with constants (and literals) stay constants.
    var both = frame.select_exprs(
        [
            (col("c") * lit(Int64(2))).alias("a"),
            (col("c") + col("c")).alias("b"),
            (col("c") == lit(Int64(7))).alias("e"),
            (col("n") + col("c")).alias("m"),
            col("x"),
        ]
    )
    assert_true(both.column("a").is_constant())
    assert_equal(both.column("a").get(N - 1).int64(), 14)
    assert_true(both.column("b").is_constant())
    assert_equal(both.column("b").get(3).int64(), 14)
    assert_true(both.column("e").is_constant())
    assert_true(both.column("e").get(0).bool())
    assert_true(both.column("m").is_constant())
    assert_equal(both.column("m").null_count(), N)


def test_constant_columns_keep_overflow_errors() raises:
    var frame = _frame().with_columns(lit(Int64.MAX).alias("c"))
    assert_true(frame.column("c").is_constant())
    with assert_raises():
        _ = frame.select((col("c") + lit(Int64(1))).alias("r"))
    with assert_raises():
        _ = frame.select((col("c") + col("c")).alias("r"))
    with assert_raises():
        _ = frame.select((col("c") + col("x")).alias("r"))
    with assert_raises():
        _ = frame.lazy().select((col("c") + col("x")).alias("r")).collect()
    # Row 0 adds zero: no overflow there, and only there.
    var first = frame.head(1).select((col("c") + col("x")).alias("r"))
    assert_equal(first.column("r").get(0).int64(), Int64.MAX)


def test_scalar_aggregate_broadcast_is_constant() raises:
    var frame = _frame()
    var total = Int64(N) * Int64(N - 1) // 2
    var eager = frame.with_columns(col("x").sum().alias("s")).column("s")
    assert_true(eager.is_constant())
    assert_equal(eager.get(N - 1).int64(), total)
    var lazy = (
        frame.lazy()
        .with_columns(col("x").sum().alias("s"))
        .select_exprs([col("x"), col("s")])
        .collect()
        .column("s")
    )
    assert_true(lazy.is_constant())
    assert_equal(lazy.null_count(), 0)
    assert_equal(lazy.get(17).int64(), total)
