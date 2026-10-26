"""JSON wire forms, nested ownership and every currently supported I/O dtype."""
# contract-group: io.json
from std.os import getenv
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    ListColumn,
    StructColumn,
    Series,
    StringColumn,
)
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.io.json import read_json, read_ndjson
from dataframe.io.json_write import write_json, write_ndjson
from dataframe.io.json_wire import duration_text, parse_duration
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


def _path(name: String) -> String:
    return getenv("COVERAGE_TEST_TMP") + "/" + name


def _schema(frame: DataFrame) raises -> DataType:
    var names = List[String]()
    var types = List[DataType]()
    for column in frame._columns:
        names.append(column.name())
        types.append(column.dtype())
    return DataType.struct(names, types)


def _roundtrip(frame: DataFrame, name: String) raises:
    var schema = _schema(frame)
    for buffer_size in [1, 64, 65536]:
        var path = _path(name + ".ndjson")
        write_ndjson(frame, path, buffer_size=buffer_size)
        assert_frame(read_ndjson(path, schema, batch_size=2), frame)
        path = _path(name + ".json")
        write_json(frame, path, buffer_size=buffer_size)
        assert_frame(read_json(path, schema, batch_size=2), frame)


def test_json_writers_roundtrip_numeric_widths_and_float_bits() raises:
    comptime for D in NUMERIC_DTYPES:
        var frame = DataFrame(
            [
                Series(
                    "x",
                    Column[Scalar[D]](
                        [0, 1, 3, 4, 5], [True, True, False, True, True]
                    ),
                )
            ]
        )
        _roundtrip(frame, "json-numeric")
        _roundtrip(chunk_layout(frame), "json-numeric-chunks")
        _roundtrip(offset_layout(frame), "json-numeric-offset")
    var minimum = Int64(-9223372036854775807) - 1
    var frame = DataFrame(
        [
            Series("signed", Column[Int64]([minimum, 9223372036854775807, 0])),
            Series("unsigned", Column[UInt64]([UInt64.MAX, 0, 1])),
            Series(
                "f",
                Column[Float64](
                    [Float64("1.7976931348623157e308"), Float64("5e-324"), -0.0]
                ),
            ),
            Series(
                "f32",
                Column[Float32](
                    [
                        Float32(Float64("3.4028235e38")),
                        Float32(Float64("1e-45")),
                        -0.0,
                    ]
                ),
            ),
        ]
    )
    _roundtrip(frame, "json-boundaries")
    var path = _path("json-boundaries.ndjson")
    var actual = read_ndjson(path, _schema(frame))
    for name in [String("f"), String("f32")]:
        var cell = actual.column(name).get(2)
        var value = cell.float64() if name == "f" else Float64(cell.float32())
        assert_equal(
            bitcast[DType.uint64](value),
            bitcast[DType.uint64](Float64(-0.0)),
        )


def test_json_writers_roundtrip_temporal_decimal_categorical_and_binary() raises:
    var minimum = Int64(-9223372036854775807) - 1
    var fields = List[Series]()
    for unit in [String("ns"), String("us"), String("ms")]:
        fields.append(
            Series(
                "duration-" + unit,
                Column[Int64]([minimum, -1, 0, 1, 9223372036854775807]),
            ).with_dtype(DataType.duration(unit))
        )
        for zone in [String(""), String("UTC"), String("America/New_York")]:
            fields.append(
                Series(
                    "datetime-" + unit + zone,
                    Column[Int64](
                        [Int64(-123456789), 0, 1, 123456789, 0],
                        [True, True, True, True, False],
                    ),
                ).with_dtype(DataType.datetime(unit, zone))
            )
    fields.append(
        Series(
            "date",
            Column[Int64](
                [-1, 0, 1, 20000, 0], [True, True, True, True, False]
            ),
        ).with_dtype(DataType.DATE)
    )
    fields.append(
        Series(
            "time",
            Column[Int64](
                [0, 1, 123456789, 86399999999999, 0],
                [True, True, True, True, False],
            ),
        ).with_dtype(DataType.TIME)
    )
    fields.append(
        Series(
            "d32",
            Column[Int32](
                [-12345, 0, 1, 999999999, 0], [True, True, True, True, False]
            ),
        ).with_dtype(DataType.decimal(9, 3, 32))
    )
    fields.append(
        Series(
            "d64",
            Column[Int64](
                [-12345, 0, 1, 999999999999999999, 0],
                [True, True, True, True, False],
            ),
        ).with_dtype(DataType.decimal(18, 9, 64))
    )
    fields.append(
        Series(
            "d128",
            Column[Int128](
                [-(Int128(1) << 100), 0, 1, Int128(1) << 100, 0],
                [True, True, True, True, False],
            ),
        ).with_dtype(DataType.decimal(38, 10, 128))
    )
    fields.append(
        Series(
            "cat",
            Column[String](
                ["é", "", "a", "é", ""], [True, True, True, True, False]
            ),
        ).cast("categorical")
    )
    var bytes = StringColumn.from_bytes(
        [
            List[UInt8]([UInt8(0), 255, 128, 1]),
            List[UInt8](),
            List[UInt8]([UInt8(1)]),
            List[UInt8]([UInt8(1), 2]),
            List[UInt8](),
        ],
        [True, True, True, True, False],
    )
    fields.append(Series("binary", bytes.copy()).with_dtype(DataType.BINARY))
    fields.append(Series("geometry", bytes^).with_dtype(DataType.geometry()))
    var frame = DataFrame(fields^)
    _roundtrip(frame, "json-logical")
    _roundtrip(chunk_layout(frame), "json-logical-chunks")
    _roundtrip(offset_layout(frame), "json-logical-offset")


def test_json_writers_nested_strings_nulls_empty_tables_and_nonfinite_policy() raises:
    var frame = DataFrame(
        [
            Series(
                'escaped"\\\n',
                Column[String](
                    ['é😀\n\r\t\b\f\u0000"\\/', "", "x", "last"],
                    [True, True, False, True],
                ),
            ),
            Series(
                "b",
                Column[Bool](
                    [True, False, True, False], [True, True, False, True]
                ),
            ),
            Series.full_null("n", DataType.NULL, 4),
            Series(
                "lists",
                ListColumn(
                    [Int64(0), Int64(2), Int64(2), Int64(3), Int64(4)],
                    Series(
                        "item",
                        Column[String](
                            ["é", "", "unused", "tail"],
                            [True, False, True, True],
                        ),
                    ),
                    [UInt8(11)],
                ),
            ),
            Series(
                "sub",
                StructColumn(
                    [Series("s", Column[String](["é", "unused", "", "end"]))],
                    [UInt8(13)],
                    length=4,
                ),
            ),
        ]
    )
    _roundtrip(frame, "json-nested-write")
    _roundtrip(chunk_layout(frame), "json-nested-chunks")
    _roundtrip(offset_layout(frame), "json-nested-offset")
    _roundtrip(DataFrame([], height=2), "json-zero-fields")
    _roundtrip(frame.clear(), "json-empty-write")
    var path = _path("json-nonfinite.ndjson")
    write_ndjson(
        DataFrame(
            [
                Series(
                    "x",
                    Column[Float64](
                        [Float64("nan"), Float64("inf"), Float64("-inf"), -0.0]
                    ),
                )
            ]
        ),
        path,
    )
    var result = read_ndjson(path, DataType.struct(["x"], [DataType.FLOAT64]))
    assert_equal(result.column("x").null_count(), 3)
    assert_equal(
        bitcast[DType.uint64](result.column("x").get(3).float64()),
        bitcast[DType.uint64](Float64(-0.0)),
    )


def test_json_duration_iso_and_invalid_base64_rollback() raises:
    for unit in [String("ns"), String("us"), String("ms")]:
        var dtype = DataType.duration(unit)
        for value in [
            Int64(-9223372036854775807) - 1,
            Int64(-1),
            Int64(0),
            Int64(1),
            Int64(9223372036854775807),
        ]:
            assert_equal(
                parse_duration(duration_text(value, dtype), dtype), value
            )
        assert_equal(
            parse_duration("P1DT2H3M4S", dtype), 93784 * dtype.per_second()
        )
    for bad in [
        String("P"),
        String("PT"),
        String("P1M"),
        String("PT1H1H"),
        String("PT1.5M"),
        String("PT1Sx"),
        String("PT9223372036854775808S"),
    ]:
        with assert_raises():
            _ = parse_duration(bad, DataType.duration("ns"))
    var path = _path("json-base64-errors.ndjson")
    with open(path, "w") as file:
        file.write('{"x":"AQID!!!!"}\n{"x":"AQI="}\n{"x":null}\n{"x":""}')
    var schema = DataType.struct(["x"], [DataType.BINARY])
    with assert_raises(contains="base64"):
        _ = read_ndjson(path, schema)
    var result = read_ndjson(path, schema, ignore_errors=True)
    var expected = DataFrame(
        [
            Series(
                "x",
                StringColumn.from_bytes(
                    [
                        List[UInt8](),
                        List[UInt8]([UInt8(1), 2]),
                        List[UInt8](),
                        List[UInt8](),
                    ],
                    [False, True, False, True],
                ),
            ).with_dtype(DataType.BINARY)
        ]
    )
    assert_frame(result, expected)
