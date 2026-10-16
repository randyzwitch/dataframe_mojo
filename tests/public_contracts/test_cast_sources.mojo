"""Source-side casts preserve every row, validity, and destination metadata."""
# contract-group: types.cast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from tests.support.numeric_model import INTEGER_DTYPES, FLOAT_DTYPES
from tests.support.coverage_generators import offset_layout, chunk_layout


def _layout(frame: DataFrame, layout: Int) raises -> DataFrame:
    return (
        frame.copy() if layout
        == 0 else offset_layout(frame) if layout
        == 1 else chunk_layout(frame)
    )


def test_boolean_casts_cover_every_numeric_destination_and_validity_position() raises:
    var values: List[Bool] = [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    for mode in range(3):
        var valid = List[Bool](length=9, fill=mode != 2)
        if mode == 1:
            for row in range(9):
                valid[row] = row % 3 != 1
        var raw = DataFrame(
            [Series("x", Column[Bool](values.copy(), valid.copy()))]
        )
        for layout in range(3):
            var frame = _layout(raw, layout)
            comptime numeric = (
                DType.int8,
                DType.int16,
                DType.int32,
                DType.int64,
                DType.uint8,
                DType.uint16,
                DType.uint32,
                DType.uint64,
                DType.float32,
                DType.float64,
            )
            comptime for i in range(len(numeric)):
                comptime D = numeric[i]
                for strict in [False, True]:
                    var results = [
                        frame.column("x").cast(DataType.of(D), strict=strict),
                        frame.select(
                            col("x").cast(DataType.of(D), strict=strict)
                        ).column("x"),
                    ]
                    for actual in results:
                        assert_true(actual.dtype() == DataType.of(D))
                        assert_equal(actual.name(), "x")
                        assert_equal(len(actual), 9)
                        for row in range(9):
                            assert_equal(
                                actual.get(row).is_null(), not valid[row]
                            )
                            if valid[row]:
                                assert_equal(
                                    actual.get(row).numeric[D](),
                                    Scalar[D](1 if values[row] else 0),
                                )


def test_integer_and_boolean_decimal_casts_read_every_row_at_all_storage_widths() raises:
    var valid: List[Bool] = [True, False, True, True, True, False, True]
    var columns = List[Series]()
    comptime for i in range(len(INTEGER_DTYPES)):
        comptime D = INTEGER_DTYPES[i]
        columns.append(
            Series(
                "x",
                Column[Scalar[D]](
                    [Scalar[D](0), 7, 1, 12, 3, 9, 24], valid.copy()
                ),
            )
        )
    columns.append(
        Series(
            "x",
            Column[Bool](
                [False, True, True, False, True, False, True], valid.copy()
            ),
        )
    )
    for index in range(len(columns)):
        var raw = DataFrame([columns[index].copy()])
        var numbers: List[Int128] = [0, 7, 1, 12, 3, 9, 24] if index < 8 else [
            0,
            1,
            1,
            0,
            1,
            0,
            1,
        ]
        for width in [32, 64, 128]:
            for scale in [0, 2]:
                var dtype = DataType.decimal(7, scale, width)
                var factor = Int128(1) if scale == 0 else Int128(100)
                for layout in range(3):
                    var frame = _layout(raw, layout)
                    for strict in [False, True]:
                        var results = [
                            frame.column("x").cast(dtype, strict=strict),
                            frame.select(
                                col("x").cast(dtype, strict=strict)
                            ).column("x"),
                        ]
                        for actual in results:
                            assert_true(actual.dtype() == dtype)
                            assert_equal(len(actual), 7)
                            for row in range(7):
                                assert_equal(
                                    actual.get(row).is_null(), not valid[row]
                                )
                                if valid[row]:
                                    assert_equal(
                                        actual._decimal_raw(row),
                                        numbers[row] * factor,
                                    )


def test_numeric_boolean_casts_use_nonzero_and_reject_valid_nan_only() raises:
    var columns = List[Series]()
    comptime for i in range(len(INTEGER_DTYPES)):
        comptime D = INTEGER_DTYPES[i]
        columns.append(
            Series(
                "x",
                Column[Scalar[D]](
                    [0, 1, Scalar[D].MAX, Scalar[D].MIN, 1],
                    [True, True, True, True, False],
                ),
            )
        )
    comptime for i in range(len(FLOAT_DTYPES)):
        comptime D = FLOAT_DTYPES[i]
        var nan = Scalar[D](0) / Scalar[D](0)
        var inf = Scalar[D](1) / Scalar[D](0)
        columns.append(
            Series(
                "x",
                Column[Scalar[D]](
                    [0.0, -0.0, -1, inf, nan, nan],
                    [True, True, True, True, True, False],
                ),
            )
        )
    for index in range(len(columns)):
        var raw = DataFrame([columns[index].copy()])
        for layout in range(3):
            var frame = _layout(raw, layout)
            for actual in [
                frame.column("x").cast(DataType.BOOL, strict=False),
                frame.select(col("x").cast(DataType.BOOL, strict=False)).column(
                    "x"
                ),
            ]:
                assert_true(actual.dtype() == DataType.BOOL)
                assert_equal(len(actual), 5 if index < 8 else 6)
                for row in range(len(actual)):
                    var absent = row == 4 if index < 8 else row >= 4
                    assert_equal(actual.get(row).is_null(), absent)
                    if not absent:
                        var expected = (
                            row == 1
                            or row == 2
                            or (row == 3 and index < 4) if index
                            < 8 else row
                            >= 2
                        )
                        assert_equal(actual.get(row).bool(), expected)
            if index >= 8:
                with assert_raises(contains="NaN has no Boolean value"):
                    _ = frame.select(col("x").cast(DataType.BOOL))
