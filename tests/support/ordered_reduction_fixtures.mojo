"""Typed ordered rows shared by public reduction and state-merge contracts."""
from dataframe import Column, DataType, Series
from tests.support.numeric_model import (
    INTEGER_DTYPES,
    FLOAT_DTYPES,
    integer_bounds,
)


def typed_ordered_columns(valid: List[Bool]) raises -> List[Series]:
    var columns = List[Series]()
    comptime for index in range(len(INTEGER_DTYPES)):
        comptime D = INTEGER_DTYPES[index]
        var bounds = integer_bounds[D]()
        columns.append(
            Series(
                "x",
                Column[Scalar[D]](
                    [
                        bounds[0].cast[D](),
                        bounds[1].cast[D](),
                        Scalar[D](1),
                        Scalar[D](2),
                        Scalar[D](0),
                    ],
                    valid.copy(),
                ),
            )
        )
    comptime for index in range(len(FLOAT_DTYPES)):
        comptime D = FLOAT_DTYPES[index]
        var nan = Float64(0) / Float64(0)
        columns.append(
            Series(
                "x",
                Column[Scalar[D]](
                    [
                        Scalar[D](-0.0),
                        nan.cast[D](),
                        Scalar[D](1),
                        Scalar[D](2),
                        Scalar[D](0),
                    ],
                    valid.copy(),
                ),
            )
        )
    columns.append(
        Series(
            "x", Column[Bool]([False, True, True, False, True], valid.copy())
        )
    )
    var strings = Series(
        "x",
        Column[String](
            ["", "abcdefghijklmnopé", "é", "duplicate", "poison"], valid.copy()
        ),
    )
    columns.append(strings.copy())
    columns.append(strings.cast(DataType.CATEGORICAL))
    var binary = Series.binary(
        "x", [[], [255, 0, 122], [0], [97, 98], [128]], valid
    )
    columns.append(binary.copy())
    columns.append(binary.with_dtype(DataType.geometry('{"crs":"EPSG:4326"}')))
    var ticks = Series("x", Column[Int64]([0, 123, 1, 2, 3], valid.copy()))
    for dtype in [
        DataType.DATE,
        DataType.TIME,
        DataType.datetime("ns", "UTC"),
        DataType.datetime("us"),
        DataType.datetime("ms", "America/New_York"),
        DataType.duration("ns"),
        DataType.duration("us"),
        DataType.duration("ms"),
    ]:
        columns.append(ticks.with_dtype(dtype))
    for width in [32, 64, 128]:
        columns.append(
            Series("x", Column[Int128]([125, -250, 1, 2, 3], valid.copy()))
            .with_dtype(DataType.decimal(8, 2))
            .cast(DataType.decimal(8, 2, width))
        )
    return columns^
