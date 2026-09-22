"""Typed CSV builder boundaries, nulls, and ownership transfer."""
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_raises
from dataframe.csv import CsvField, CsvOptions
from dataframe.csv_buffers import CsvBuffer
from dataframe.dtype import DataType


def options() -> CsvOptions:
    return CsvOptions(",", '"', "", 0, -1, [], False, False, "utf8")


def add_text(mut buffer: CsvBuffer, text: String) raises:
    buffer.add(
        Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=text.as_bytes()
            .unsafe_ptr()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            length=text.byte_length(),
        ),
        False,
        options(),
    )


def test_integer_output_types_and_null_bitmap_boundaries() raises:
    var types: List[DataType] = [
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
        DataType.INT64,
        DataType.UINT8,
        DataType.UINT16,
        DataType.UINT32,
        DataType.UINT64,
    ]
    var bounds: List[String] = [
        "-128",
        "-32768",
        "-2147483648",
        "-9223372036854775808",
        "255",
        "65535",
        "4294967295",
        "18446744073709551615",
    ]
    for i in range(len(types)):
        var buffer = CsvBuffer(CsvField("x", types[i]), 17)
        for row in range(17):
            if row % 3 == 0:
                buffer.add_null()
            else:
                add_text(buffer, bounds[i])
        var result = buffer.finish()
        assert_equal(result.dtype(), types[i])
        assert_equal(len(result), 17)
        assert_equal(result.null_count(), 6)
        assert_equal(String(result.get(16)), bounds[i])


def test_float32_rounding_is_direct_and_float64_overflow_is_infinity() raises:
    var buffer = CsvBuffer(CsvField("x", DataType.FLOAT32), 1)
    # Just above the Float32 midpoint, but rounds to that midpoint as Float64.
    var value = String("1.000000059604644775390625000001")
    add_text(buffer, value)
    var result = buffer.finish()
    assert_equal(
        bitcast[DType.uint32](result.numeric[DType.float32]().value(0)),
        UInt32(0x3F800001),
    )
    var wide = CsvBuffer(CsvField("x", DataType.FLOAT64), 1)
    var huge = String("1e400")
    add_text(wide, huge)
    var output = wide.finish()
    assert_equal(
        bitcast[DType.uint64](output.float64().value(0)),
        UInt64(0x7FF0000000000000),
    )


def test_null_append_follows_polars_nullable_columns() raises:
    var buffer = CsvBuffer(CsvField.int64("x", False), 1)
    # Polars builders accept nulls; the clean source port does not reintroduce
    # the legacy reader's non-nullable-field enforcement.
    buffer.add_null()
    var result = buffer.finish()
    assert_equal(len(result), 1)
    assert_equal(result.null_count(), 1)


def test_string_buffer_retains_inline_long_and_escaped_values() raises:
    var buffer = CsvBuffer(CsvField.string("label"), 8)
    var expected: List[String] = [
        "short",
        "123456789012",
        "1234567890123",
        "héllo世界-long",
    ]
    for value in expected:
        add_text(buffer, value)
    var quoted = String('"long string with ""quotes"" and a newline\ninside"')
    buffer.add(
        Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=quoted.as_bytes()
            .unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            length=quoted.byte_length(),
        ),
        True,
        options(),
    )
    # Reusing the escape scratch must not overwrite previously appended bytes.
    var next_quoted = String('"next ""quoted"" value"')
    buffer.add(
        Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=next_quoted.as_bytes()
            .unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            length=next_quoted.byte_length(),
        ),
        True,
        options(),
    )
    buffer.add_null()
    var result = buffer.finish().string()
    for i in range(len(expected)):
        assert_equal(result.value(i), expected[i])
    assert_equal(
        result.value(4), 'long string with "quotes" and a newline\ninside'
    )
    assert_equal(result.value(5), 'next "quoted" value')
    assert_equal(result.null_count(), 1)
    # finish resets builder ownership; another output cannot invalidate result.
    add_text(buffer, String("another output with long bytes"))
    var other = buffer.finish().string()
    assert_equal(other.value(0), "another output with long bytes")
    assert_equal(result.value(2), "1234567890123")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
