"""write_csv/to_csv_string round-trip through read_csv, plus dialect options."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import (
    Column,
    CsvSchema,
    DataFrame,
    Series,
    read_csv,
    to_csv_string,
    write_csv,
)

comptime PATH = "/tmp/dataframe_mojo_write_test.csv"
comptime MIN = Int64(-9223372036854775807) - 1
comptime MAX = Int64(9223372036854775807)


def nan() -> Float64:
    return Float64(0) / Float64(0)


def inf() -> Float64:
    return Float64(1) / Float64(0)


def tricky() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "i",
                Column[Int64](
                    [MIN, MAX, 0, -1, 7, 2**53 + 1],
                    [True, True, True, False, True, True],
                ),
            ),
            Series(
                "f",
                Column[Float64](
                    [nan(), inf(), -inf(), -0.0, 0.1, 1e300],
                    [True, True, True, True, False, True],
                ),
            ),
            Series(
                "b",
                Column[Bool](
                    [True, False, True, True, False, False],
                    [True, True, False, True, True, True],
                ),
            ),
            Series(
                'text, with "quotes"',
                Column[String](
                    [
                        "",
                        "plain",
                        'say "hi"',
                        "comma,inside",
                        "multi\nline\r\nrecord",
                        " spaced 日本 ",
                    ],
                    [True, True, True, True, True, False],
                ),
            ),
        ]
    )


def test_round_trip_every_dtype_and_edge_value() raises:
    var frame = tricky()
    write_csv(frame, PATH)
    var back = read_csv(PATH, CsvSchema.of(frame))
    assert_true(back.equals(frame))
    # -0.0 keeps its sign and NaN stays NaN, not null.
    assert_equal(String(back.item(3, "f")), "-0.0")
    assert_false(back.item(0, "f").is_null())
    # Small buffers and CRLF endings give identical results.
    write_csv(frame, PATH, buffer_size=7, line_terminator="\r\n")
    assert_true(read_csv(PATH, CsvSchema.of(frame)).equals(frame))
    write_csv(frame, PATH, quote_style="always")
    assert_true(read_csv(PATH, CsvSchema.of(frame)).equals(frame))
    write_csv(frame, PATH, has_header=False)
    assert_true(
        read_csv(PATH, CsvSchema.of(frame), has_header=False).equals(frame)
    )


def test_exact_text_and_quote_styles() raises:
    var frame = DataFrame(
        [
            Series("n", Column[Int64]([1, 2, 3], [True, False, True])),
            Series("s", Column[String](["a", "", "x,y"], [True, True, True])),
            Series("b", Column[Bool]([True, False, True], [True, True, False])),
        ]
    )
    assert_equal(to_csv_string(frame), 'n,s,b\n1,a,true\n,"",false\n3,"x,y",\n')
    assert_equal(
        to_csv_string(frame, quote_style="always"),
        '"n","s","b"\n"1","a","true"\n,"","false"\n"3","x,y",\n',
    )
    assert_equal(
        to_csv_string(frame, quote_style="non_numeric"),
        '"n","s","b"\n1,"a","true"\n,"","false"\n3,"x,y",\n',
    )
    assert_equal(
        to_csv_string(frame, quote_style="never", has_header=False),
        "1,a,true\n,,false\n3,x,y,\n",
    )
    assert_equal(
        to_csv_string(frame, separator=";", null_value="NA"),
        'n;s;b\n1;a;true\nNA;"";false\n3;x,y;NA\n',
    )
    var looks_null = DataFrame([Series("s", Column[String](["NA", "ok"]))])
    assert_equal(to_csv_string(looks_null, null_value="NA"), 's\n"NA"\nok\n')
    assert_equal(to_csv_string(frame.clear()), "n,s,b\n")
    assert_equal(to_csv_string(DataFrame([], height=0)), "\n")


def test_option_validation() raises:
    var frame = tricky()
    with assert_raises(contains="single byte"):
        _ = to_csv_string(frame, separator="::")
    with assert_raises(contains="cannot be a quote"):
        _ = to_csv_string(frame, separator='"')
    with assert_raises(contains="quote_style must be"):
        _ = to_csv_string(frame, quote_style="sometimes")
    with assert_raises(contains="line_terminator"):
        _ = to_csv_string(frame, line_terminator="\r")
    with assert_raises(contains="null_value cannot contain"):
        _ = to_csv_string(frame, null_value="a,b")
    with assert_raises(contains="buffer_size"):
        write_csv(frame, PATH, buffer_size=0)


def test_large_frame_streams_in_chunks() raises:
    var values = List[Int64]()
    var labels = List[String]()
    for i in range(3000):
        values.append(Int64(i * 3))
        labels.append("row " + String(i))
    var frame = DataFrame(
        [
            Series("v", Column[Int64](values^)),
            Series("label", Column[String](labels^)),
        ]
    )
    write_csv(frame, PATH, buffer_size=512)
    assert_true(read_csv(PATH, CsvSchema.of(frame)).equals(frame))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
