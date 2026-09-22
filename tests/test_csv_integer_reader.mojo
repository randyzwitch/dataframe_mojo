"""Public read_csv coverage for the CSV-only atoi_simd integer parser."""
from std.testing import TestSuite, assert_equal, assert_raises
from dataframe import CsvField, CsvSchema, DataType, read_csv


comptime CSV_PATH = "/tmp/dataframe_mojo_csv_integer_legacy.csv"


def _write(text: String) raises:
    with open(CSV_PATH, "w") as file:
        file.write(text)


def _schema() raises -> CsvSchema:
    return CsvSchema(
        [
            CsvField("i8", DataType.INT8, False),
            CsvField("u8", DataType.UINT8, False),
            CsvField("i16", DataType.INT16, False),
            CsvField("u16", DataType.UINT16, False),
            CsvField("i32", DataType.INT32, False),
            CsvField("u32", DataType.UINT32, False),
            CsvField.int64("i64", False),
            CsvField("u64", DataType.UINT64, False),
        ]
    )


def test_public_csv_integer_widths_and_source_boundaries() raises:
    _write(
        "i8,u8,i16,u16,i32,u32,i64,u64\n"
        "-128,255,-32768,65535,-2147483648,4294967295,"
        "-9223372036854775808,18446744073709551615\n"
        "+000000000000000000000001,0000000000000000000000002,"
        "000000000000000000000000003,0000000000000000000000000004,"
        "000000000000000000000000000005,0000000000000000000000000000006,"
        "+000000000000000000000000000000000000007,"
        "0000000000000000000000000000000000000008"
    )
    var frame = read_csv(CSV_PATH, _schema(), buffer_size=1)
    assert_equal(frame.height(), 2)
    assert_equal(frame.column("i8").int8().value(0), Int8(-128))
    assert_equal(frame.column("u8").uint8().value(0), UInt8(255))
    assert_equal(frame.column("i16").int16().value(0), Int16(-32768))
    assert_equal(frame.column("u16").uint16().value(0), UInt16(65535))
    assert_equal(frame.column("i32").int32().value(0), Int32(-2147483648))
    assert_equal(frame.column("u32").uint32().value(0), UInt32(4294967295))
    assert_equal(frame.column("i64").int64().value(0), Int64.MIN)
    assert_equal(frame.column("u64").uint64().value(0), UInt64.MAX)
    assert_equal(frame.column("i8").int8().value(1), Int8(1))
    assert_equal(frame.column("u64").uint64().value(1), UInt64(8))


def test_public_csv_integer_overflow_and_invalid_bytes() raises:
    var schema = CsvSchema([CsvField("u8", DataType.UINT8, False)])
    # atoi_simd's unsigned route rejects every negative spelling, including
    # -0. dataframe.parse keeps its historical generic-cast behavior instead.
    for text in ["u8\n256", "u8\n-1", "u8\n-0", "u8\n12x", "u8\n+", "u8\n\n"]:
        _write(text)
        with assert_raises():
            _ = read_csv(CSV_PATH, schema, buffer_size=1)


def test_public_csv_temporal_fields_keep_their_existing_parser() raises:
    var schema = CsvSchema([CsvField.date("day", nullable=False)])
    _write("day\n1970-01-02")
    var frame = read_csv(CSV_PATH, schema, buffer_size=1)
    assert_equal(frame.column("day").int64().value(0), Int64(1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
