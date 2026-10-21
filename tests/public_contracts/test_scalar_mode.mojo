"""Independent typed anchors for scalar mode and exact decimal histograms."""
# contract-group: public.apple_parity
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
    lit,
)
from tests.support.numeric_model import INTEGER_DTYPES, integer_bounds
from tests.support.coverage_compare import assert_frame, assert_value
from tests.support.coverage_generators import offset_layout, chunk_layout
from dataframe.aggregate import Reducer
from dataframe.expr import MODE, VALUE_COUNTS, join_names


def mode_sources() raises -> List[Series]:
    var result = List[Series]()
    var valid: List[Bool] = [True, True, True, True, False, False]
    comptime for i in range(len(INTEGER_DTYPES)):
        comptime D = INTEGER_DTYPES[i]
        var bounds = integer_bounds[D]()
        var low = bounds[0].cast[D]()
        var high = bounds[1].cast[D]()
        result.append(
            Series(
                "x",
                Column[Scalar[D]]([high, low, high, low, 0, 0], valid.copy()),
            )
        )
    result.append(
        Series(
            "x",
            Column[Bool]([True, False, True, False, True, True], valid.copy()),
        )
    )
    var text = Series(
        "x",
        StringColumn(["é🙂", "a", "é🙂", "a", "poison", "poison"], valid.copy()),
    )
    result.append(text.copy())
    result.append(text.cast(DataType.CATEGORICAL))
    result.append(
        Series("x", Column[Int64]([3, 1, 3, 1, 9, 9], valid.copy())).with_dtype(
            DataType.DATE
        )
    )
    for dtype in [
        DataType.TIME,
        DataType.datetime("ns", "UTC"),
        DataType.datetime("us"),
        DataType.duration("ns"),
        DataType.duration("us"),
        DataType.duration("ms"),
    ]:
        result.append(
            Series(
                "x", Column[Int64]([3, 1, 3, 1, 9, 9], valid.copy())
            ).with_dtype(dtype)
        )
    for width in [32, 64, 128]:
        var magnitude = Int128(1) << Int128(
            20 if width == 32 else 54 if width == 64 else 90
        )
        var source = Series(
            "x",
            Column[Int128](
                [
                    magnitude + 1,
                    -magnitude,
                    magnitude + 1,
                    -magnitude,
                    Int128.MIN,
                    0,
                ],
                valid.copy(),
            ),
        ).with_dtype(DataType.decimal(38, 2))
        if width != 128:
            source = source.cast(
                DataType.decimal(9 if width == 32 else 18, 2, width)
            )
        result.append(source^)
    return result^


def mode_expected(source: Series) raises -> DataFrame:
    var values = source.cast(
        DataType.STRING
    ) if source.dtype().is_categorical() else source.copy()
    return DataFrame([values.take([1, 0, 4])])


def float_mode_source(single: Bool) raises -> Series:
    var valid: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    if single:
        var words: List[UInt32] = [
            0x80000000,
            0,
            0x7FC00042,
            0xFFC00073,
            0xFF800000,
            0xFF800000,
            0x7F800000,
            0x7F800000,
            0x7F800001,
            0,
        ]
        var values = List[Float32]()
        for word in words:
            values.append(bitcast[DType.float32](word))
        return Series("x", Column[Float32](values^, valid))
    var words: List[UInt64] = [
        0x8000000000000000,
        0,
        0x7FF8000000000042,
        0xFFF8000000000073,
        0xFFF0000000000000,
        0xFFF0000000000000,
        0x7FF0000000000000,
        0x7FF0000000000000,
        0x7FF0000000000001,
        0,
    ]
    var values = List[Float64]()
    for word in words:
        values.append(bitcast[DType.float64](word))
    return Series("x", Column[Float64](values^, valid))


def assert_float_mode_bits(actual: DataFrame, single: Bool) raises:
    assert_equal(actual.height(), 5)
    var source = actual.column("x")
    if single:
        ref values = source._data[Column[Float32]]
        assert_equal(bitcast[DType.uint32](values._get(1)), UInt32(0x80000000))
        assert_equal(bitcast[DType.uint32](values._get(3)), UInt32(0x7FC00042))
    else:
        ref values = source._data[Column[Float64]]
        assert_equal(
            bitcast[DType.uint64](values._get(1)), UInt64(0x8000000000000000)
        )
        assert_equal(
            bitcast[DType.uint64](values._get(3)), UInt64(0x7FF8000000000042)
        )


def test_scalar_mode_typed_winners_nulls_empty_and_scalar_broadcast() raises:
    for source in mode_sources():
        var raw = DataFrame([source.copy()])
        var expected = mode_expected(source)
        for layout in range(3):
            var frame = (
                raw.copy() if layout
                == 0 else offset_layout(raw) if layout
                == 1 else chunk_layout(raw)
            )
            for streaming in [False, True]:
                assert_frame(
                    frame.lazy()
                    .select(col("x").mode())
                    .collect(engine="cpu", streaming=streaming),
                    expected,
                )
                var empty = frame.clear().select(col("x").mode())
                assert_equal(empty.height(), 0)
                assert_equal(empty.dtypes(), expected.dtypes())
                var nulls = frame.tail(2).select(col("x").mode())
                assert_equal(nulls.height(), 1)
                assert_true(nulls.item().is_null())
                var siblings = (
                    frame.lazy()
                    .select_exprs(
                        [
                            col("x").mode(),
                            lit(Int64(7)).alias("scalar"),
                            col("x").len().alias("rows"),
                        ]
                    )
                    .collect(engine="cpu", streaming=streaming)
                )
                assert_equal(siblings.height(), 3)
                for row in range(3):
                    assert_value(
                        siblings.item(row, "x"), expected.item(row, "x")
                    )
                    assert_equal(siblings.item(row, "scalar").int64(), Int64(7))
                    assert_equal(siblings.item(row, "rows").int64(), Int64(6))


def test_scalar_mode_float_equality_preserves_first_zero_and_nan_bits() raises:
    for single in [False, True]:
        var source = float_mode_source(single)
        var expected = DataFrame([source.take([4, 0, 6, 2, 8])])
        for layout in range(3):
            var raw = DataFrame([source.copy()])
            var frame = (
                raw.copy() if layout
                == 0 else offset_layout(raw) if layout
                == 1 else chunk_layout(raw)
            )
            var actual = frame.select(col("x").mode())
            assert_frame(actual, expected)
            assert_float_mode_bits(actual, single)


def test_decimal_mode_and_value_counts_preserve_exact_histograms_and_group_children() raises:
    for width in [32, 64, 128]:
        var magnitude = Int128(1) << Int128(
            20 if width == 32 else 54 if width == 64 else 90
        )
        var source = Series(
            "x",
            Column[Int128](
                [
                    magnitude + 1,
                    -magnitude,
                    magnitude + 1,
                    -magnitude,
                    magnitude + 1,
                    0,
                ],
                [True, True, True, True, True, False],
            ),
        ).with_dtype(DataType.decimal(38, 6))
        if width != 128:
            source = source.cast(
                DataType.decimal(9 if width == 32 else 18, 6, width)
            )
        var raw = DataFrame(
            [source.copy(), Series("g", Column[Int32]([0, 0, 0, 0, 0, 0]))]
        )
        for op in [MODE, VALUE_COUNTS]:
            var left = Reducer(
                op,
                source.dtype().physical(),
                1,
                0,
                0,
                text=join_names(["x", "count"]),
                logical=source.dtype(),
            )
            var right = Reducer(
                op,
                source.dtype().physical(),
                1,
                0,
                0,
                text=join_names(["x", "count"]),
                logical=source.dtype(),
            )
            left.update(source.head(3), 0, False, [])
            right.update(source.slice(3, 3), 0, False, [])
            left.merge(right)
            left.grow(2)
            var merged = left.finish()
            var first = merged.get(0).list()
            assert_equal(len(merged.get(1).list()), 0)
            assert_equal(len(first), 1 if op == MODE else 3)
            if op == MODE:
                assert_value(first.get(0), source.get(0))
            else:
                for i in range(3):
                    var item = first.get(i)
                    assert_value(
                        item.struct_field("x"),
                        source.get(1 if i == 0 else 0 if i == 1 else 5),
                    )
                    assert_equal(
                        item.struct_field("count").numeric[DType.uint32](),
                        UInt32(2 if i == 0 else 3 if i == 1 else 1),
                    )
        for grouped in [False, True]:
            var mode = raw.group_by("g").agg(
                [col("x").mode()]
            ) if grouped else raw.select(col("x").mode())
            var winners = mode.item(0, "x").list() if grouped else mode.column(
                "x"
            )
            assert_equal(winners.dtype(), source.dtype())
            assert_equal(len(winners), 1)
            assert_value(winners.get(0), source.get(0))
            for sorted in [False, True]:
                for normalize in [False, True]:
                    var expression = col("x").value_counts(
                        sorted, normalize=normalize
                    )
                    var frame = raw.group_by("g").agg(
                        [expression.copy()]
                    ) if grouped else raw.select(expression)
                    var values = frame.item(
                        0, "x"
                    ).list() if grouped else frame.column("x")
                    assert_equal(len(values), 3)
                    for i in range(3):
                        var value = values.get(i)
                        var position = (0 if i == 0 else 1) if sorted else (
                            1 if i == 0 else 0
                        )
                        var count = (3 if i == 0 else 2) if sorted else (
                            2 if i == 0 else 3
                        )
                        if i == 2:
                            position = 5
                            count = 1
                        assert_value(
                            value.struct_field("x"),
                            source.get(position),
                        )
                        if normalize:
                            assert_equal(
                                value.struct_field("proportion").float64(),
                                Float64(count) / Float64(6),
                            )
                        else:
                            assert_equal(
                                value.struct_field("count").numeric[
                                    DType.uint32
                                ](),
                                UInt32(count),
                            )
            var empty = (
                raw.clear()
                .group_by("g")
                .agg([col("x").mode()]) if grouped else raw.clear()
                .select(col("x").mode())
            )
            assert_equal(empty.height(), 0)
            assert_equal(
                empty.column("x").dtype(),
                DataType.list(source.dtype()) if grouped else source.dtype(),
            )
