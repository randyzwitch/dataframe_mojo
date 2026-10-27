"""Frame null strategies preserve dtype and order; NaN filters keep nulls."""
# contract-group: expr.order_windows
# contract-group: expr.arithmetic
from std.testing import assert_equal, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    ListColumn,
    StructColumn,
    col,
    lit,
)
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


def _numeric() raises -> DataFrame:
    var valid: List[Bool] = [True, False, False, True, False]
    return DataFrame(
        [
            Series("i", Column[Int64]([1, 99, 99, 4, 99], valid)),
            Series("f", Column[Float64]([1, 99, 99, 4, 99], valid)),
            Series("n", Column[Float32]([1, 99, 99, 4, 99], valid)),
        ]
    )


def test_frame_fill_strategies_numeric_types_offsets_chunks_and_lazy() raises:
    var raw = _numeric()
    for strategy in [
        String("forward"),
        "backward",
        "min",
        "max",
        "mean",
        "zero",
        "one",
    ]:
        var values: List[Int64] = [1, 1, 1, 4, 1]
        var valid = List[Bool](length=5, fill=True)
        if strategy == "forward":
            values = [1, 1, 1, 4, 4]
        elif strategy == "backward":
            values = [1, 4, 4, 4, 0]
            valid[4] = False
        elif strategy == "max":
            values = [1, 4, 4, 4, 4]
        elif strategy == "mean":
            values = [1, 2, 2, 4, 2]
        elif strategy == "zero":
            values = [1, 0, 0, 4, 0]
        var floats = List[Float64]()
        var narrow = List[Float32]()
        for row in range(5):
            var value = Float64(values[row])
            if strategy == "mean" and row != 0 and row != 3:
                value = 2.5
            floats.append(value)
            narrow.append(Float32(value))
        var expected = DataFrame(
            [
                Series("i", Column[Int64](values^, valid)),
                Series("f", Column[Float64](floats^, valid)),
                Series("n", Column[Float32](narrow^, valid)),
            ]
        )
        for frame in [raw.copy(), offset_layout(raw), chunk_layout(raw)]:
            assert_frame(frame.fill_null(strategy=strategy), expected)
            for optimize in [False, True]:
                assert_frame(
                    frame.lazy()
                    .fill_null(strategy=strategy)
                    .collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=True,
                        batch_size=2,
                    ),
                    expected,
                )
            assert_frame(
                frame.clear().fill_null(strategy=strategy), frame.clear()
            )


def test_frame_fill_limits_nested_values_text_binary_and_boolean() raises:
    var raw = _numeric()
    var expected_forward = DataFrame(
        [
            Series(
                "i",
                Column[Int64]([1, 1, 0, 4, 4], [True, True, False, True, True]),
            ),
            raw.column("f"),
            raw.column("n"),
        ]
    )
    assert_frame(
        raw.fill_null(strategy="forward", limit=1, subset=[String("i")]),
        expected_forward,
    )
    assert_frame(
        raw.lazy()
        .fill_null(strategy="forward", limit=1, subset=[String("i")])
        .collect(engine="cpu", batch_size=1),
        expected_forward,
    )
    assert_frame(raw.fill_null(strategy="forward", limit=0), raw)
    var list_values = Series(
        "x",
        ListColumn(
            [Int64(0), 1, 1, 1, 2],
            Series("item", Column[Int64]([1, 4])),
            [UInt8(0b1001)],
        ),
    )
    var fields: List[Series] = [Series("field", Column[Int64]([1, 99, 99, 4]))]
    var struct_values = Series(
        "x", StructColumn(fields^, [UInt8(0b1001)], length=4)
    )
    for source in [list_values.copy(), struct_values.copy()]:
        var frame = DataFrame([source.copy()])
        var expected = DataFrame([source.take([Int(0), 0, 2, 3])])
        assert_frame(frame.fill_null(strategy="forward", limit=1), expected)
        assert_frame(
            frame.lazy()
            .fill_null(strategy="forward", limit=1)
            .collect(engine="cpu", batch_size=1),
            expected,
        )
        assert_frame(
            frame.fill_null(strategy="backward", limit=1),
            DataFrame([source.take([Int(0), 1, 3, 3])]),
        )
    var strings = DataFrame(
        [
            Series(
                "x", Column[String](["b", "ignored", "a"], [True, False, True])
            )
        ]
    )
    assert_frame(
        strings.fill_null(strategy="min"),
        DataFrame([Series("x", Column[String](["b", "a", "a"]))]),
    )
    assert_frame(
        strings.fill_null(strategy="zero"),
        DataFrame([Series("x", Column[String](["b", "", "a"]))]),
    )
    var binary = DataFrame(
        [Series.binary("x", [[255], [99], [0]], [True, False, True])]
    )
    assert_frame(
        binary.fill_null(strategy="zero"),
        DataFrame([Series.binary("x", [[255], [], [0]])]),
    )
    var booleans = DataFrame(
        [Series("x", Column[Bool]([True, True, False], [True, False, True]))]
    )
    assert_frame(
        booleans.fill_null(strategy="min"),
        DataFrame([Series("x", Column[Bool]([True, False, False]))]),
    )
    assert_frame(
        booleans.fill_null(strategy="one"),
        DataFrame([Series("x", Column[Bool]([True, True, False]))]),
    )
    with assert_raises(contains="not supported"):
        _ = booleans.fill_null(strategy="mean")
    with assert_raises(contains="not supported"):
        _ = strings.fill_null(strategy="one")


def test_frame_mean_fill_clamps_integer_boundaries_truncates_decimal_and_keeps_temporal_dtype() raises:
    var high = DataFrame(
        [
            Series(
                "u",
                Column[UInt64](
                    [UInt64.MAX, 0, UInt64.MAX], [True, False, True]
                ),
            ),
            Series(
                "i",
                Column[Int64]([Int64.MAX, 0, Int64.MAX], [True, False, True]),
            ),
        ]
    )
    var expected = DataFrame(
        [
            Series("u", Column[UInt64]([UInt64.MAX, UInt64.MAX, UInt64.MAX])),
            Series("i", Column[Int64]([Int64.MAX, Int64.MAX, Int64.MAX])),
        ]
    )
    assert_frame(high.fill_null(strategy="mean"), expected)
    assert_frame(
        high.lazy()
        .fill_null(strategy="mean")
        .collect(engine="cpu", batch_size=1),
        expected,
    )
    var negative = DataFrame(
        [Series("x", Column[Int64]([-1, 0, -4], [True, False, True]))]
    )
    assert_frame(
        negative.fill_null(strategy="mean"),
        DataFrame([Series("x", Column[Int64]([-1, -2, -4]))]),
    )
    for width in [32, 64, 128]:
        var dtype = DataType.decimal(8, 2, width)
        var frame = DataFrame(
            [
                Series("x", Column[Int128]([1, 0, 2], [True, False, True]))
                .with_dtype(DataType.decimal(8, 2))
                .cast(dtype)
            ]
        )
        var expected_decimal = DataFrame(
            [
                Series("x", Column[Int128]([1, 1, 2]))
                .with_dtype(DataType.decimal(8, 2))
                .cast(dtype)
            ]
        )
        assert_frame(frame.fill_null(strategy="mean"), expected_decimal)
    for dtype in [
        DataType.DATE,
        DataType.TIME,
        DataType.datetime("ns", "UTC"),
        DataType.duration("us"),
    ]:
        var frame = DataFrame(
            [
                Series(
                    "x", Column[Int64]([1, 0, 4], [True, False, True])
                ).with_dtype(dtype)
            ]
        )
        assert_frame(
            frame.fill_null(strategy="mean"),
            DataFrame(
                [Series("x", Column[Int64]([1, 2, 4])).with_dtype(dtype)]
            ),
        )
    var labels = DataFrame(
        [
            Series(
                "x", Column[String](["b", "ignored", "a"], [True, False, True])
            ).cast(DataType.CATEGORICAL)
        ]
    )
    var filled = labels.lazy().fill_null(strategy="min").collect(engine="cpu")
    assert_equal(filled.column("x").dtype().is_categorical(), True)
    assert_frame(
        DataFrame([filled.column("x").cast(DataType.STRING)]),
        DataFrame([Series("x", Column[String](["b", "a", "a"]))]),
    )


def test_frame_fill_strategy_noops_validation_and_lazy_order() raises:
    var raw = _numeric()
    var expected = DataFrame(
        [
            Series("i", Column[Int64]([1, 2])),
            Series("f", Column[Float64]([1, 2.5])),
            Series("n", Column[Float32]([1, 2.5])),
        ]
    )
    assert_frame(
        raw.lazy()
        .fill_null(strategy="mean")
        .head(2)
        .collect(engine="cpu", batch_size=1),
        expected,
    )
    var dense = DataFrame(
        [
            Series(
                "x",
                ListColumn([Int64(0), 1], Series("item", Column[Int64]([1]))),
            )
        ]
    )
    assert_frame(dense.fill_null(strategy="mean"), dense)
    var all_null = DataFrame(
        [Series.full_null("x", DataType.list(DataType.INT64), 3)]
    )
    for strategy in [String("forward"), "backward", "min", "max", "mean"]:
        assert_frame(
            all_null.lazy().fill_null(strategy=strategy).collect(engine="cpu"),
            all_null,
        )
    with assert_raises(contains="Unknown"):
        _ = raw.fill_null(strategy="bad")
    with assert_raises(contains="only valid"):
        _ = raw.lazy().fill_null(strategy="mean", limit=1)
    with assert_raises(contains="nonnegative"):
        _ = raw.fill_null(strategy="forward", limit=-1)
    with assert_raises(contains="Unknown"):
        _ = raw.fill_null(strategy="zero", subset=[String("absent")])
    with assert_raises(contains="twice"):
        _ = raw.fill_null(strategy="zero", subset=[String("i"), "i"])
    with assert_raises(contains="not supported"):
        _ = all_null.lazy().fill_null(strategy="zero").collect(engine="cpu")
    var strings = DataFrame(
        [Series("x", Column[String](["a", "bad"], [True, False]))]
    )
    assert_frame(
        strings.fill_null("min"),
        DataFrame([Series("x", Column[String](["a", "min"]))]),
    )


def test_frame_nan_helpers_preserve_nulls_dtypes_and_subset_selection() raises:
    var nan = Float64(0) / Float64(0)
    var raw = DataFrame(
        [
            Series(
                "a", Column[Float64]([1, nan, 99, 2], [True, True, False, True])
            ),
            Series(
                "b",
                Column[Float32](
                    [1, 99, Float32(nan), 2], [True, False, True, True]
                ),
            ),
            Series("i", Column[Int64]([1, 2, 3, 4])),
        ]
    )
    var filled = DataFrame(
        [
            Series(
                "a", Column[Float64]([1, 0, 99, 2], [True, True, False, True])
            ),
            Series(
                "b", Column[Float32]([1, 99, 0, 2], [True, False, True, True])
            ),
            raw.column("i"),
        ]
    )
    for frame in [raw.copy(), offset_layout(raw), chunk_layout(raw)]:
        assert_frame(frame.fill_nan(lit(Int64(0))), filled)
        assert_frame(
            frame.lazy()
            .fill_nan(lit(Int64(0)))
            .collect(engine="cpu", batch_size=1),
            filled,
        )
        assert_frame(frame.drop_nans(), raw.take([Int(0), 3]))
        assert_frame(
            frame.lazy().drop_nans().collect(engine="cpu", batch_size=1),
            raw.take([Int(0), 3]),
        )
        assert_frame(
            frame.lazy()
            .drop_nans([String("a")])
            .collect(engine="cpu", batch_size=1),
            raw.take([Int(0), 2, 3]),
        )
        assert_frame(frame.drop_nans([String("i")]), raw)
    with assert_raises(contains="Unknown"):
        _ = raw.lazy().drop_nans([String("absent")])


def test_nested_fill_expressions_validate_empty_schemas_and_parent_validity() raises:
    from tests.public_contracts.test_extended_windows import extended_frame

    for n in [0, 9]:
        for kind in [26, 27]:
            var frame = extended_frame(kind, n, 7, 1)
            var source = frame.column("x")
            for backward in [False, True]:
                var picks = List[Int](length=n, fill=-1)
                var carry = -1
                for step in range(n):
                    var row = n - step - 1 if backward else step
                    if not source.get(row).is_null():
                        carry = row
                    picks[row] = carry
                var expected = DataFrame([source.take_or_null(picks)])
                var expression = (
                    col("x")
                    .backward_fill() if backward else col("x")
                    .forward_fill()
                )
                assert_frame(frame.select(expression), expected)
                var plan = frame.lazy().select(expression)
                assert_equal(plan.collect_schema()[0].dtype, source.dtype())
                assert_frame(plan.collect(engine="cpu", batch_size=1), expected)
