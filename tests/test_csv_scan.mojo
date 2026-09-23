"""Scalar differential tests for the Polars-style CountLines scanner."""
from std.testing import TestSuite, assert_equal
from dataframe.csv_scan import CountLines, chunk_size


def span(text: String) -> List[UInt8]:
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    return bytes^


def scalar_count(
    bytes: Span[UInt8, ImmutAnyOrigin], quote: UInt8, quoting: Bool
) -> Tuple[Int, Int]:
    var rows = 0
    var last = 0
    var in_quotes = False
    for i in range(len(bytes)):
        if quoting and bytes[i] == quote:
            in_quotes = not in_quotes
        elif bytes[i] == 10 and not in_quotes:
            rows += 1
            last = i
    return (rows, last)


def check(text: String, quote: UInt8 = UInt8(34), quoting: Bool = True) raises:
    var owned = span(text)
    var bytes = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=owned.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(owned),
    )
    var want = scalar_count(bytes, quote, quoting)
    var got = CountLines(quote, quoting).count(bytes)
    assert_equal(got.rows, want[0])
    assert_equal(got.last_newline, want[1])
    _ = owned^


def test_every_quote_offset_and_quoted_newline() raises:
    for padding in range(0, 193):
        var text = String('"')
        for _ in range(padding):
            text += "x"
        text += '"\nnext,row\n'
        check(text)
        check('plain,"x\ny",tail\n' + text)


def test_doubled_quotes_final_eof_and_quote_disabled() raises:
    check('a,"x""\ny",z\nnext,ok\n')
    check('a,"unterminated\nlast,row')
    check('a,"x\ny",z\n', UInt8(34), False)
    check("", UInt8(34), True)


def test_find_next_doubles_only_until_a_record() raises:
    var text = String()
    for _ in range(200):
        text += "x"
    text += "\nlast"
    var owned = span(text)
    var bytes = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=owned.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(owned),
    )
    var found = CountLines().find_next(bytes, 16)
    assert_equal(found.rows, 1)
    assert_equal(found.last_newline, 200)
    assert_equal(found.chunk_size, 256)
    var eof = CountLines().find_next(bytes[201:], 16)
    assert_equal(eof.rows, 0)
    assert_equal(eof.last_newline, 0)
    _ = owned^


def test_chunk_size_scales_with_input_and_width_budget() raises:
    assert_equal(chunk_size(0, 32, 8), 4096)
    assert_equal(chunk_size(1_000_000, 1, 1), 250_000)
    # Small inputs use four ranges per worker; large inputs grow to sixteen.
    assert_equal(chunk_size(64 * 1024 * 1024, 32, 8), 512 * 1024)
    assert_equal(chunk_size(512 * 1024 * 1024, 32, 8), 1024 * 1024)
    # Width caps the number of chunks at 500k / width, but never below
    # the worker count.
    assert_equal(chunk_size(1_000_000, 32, 100_000), 31_250)
    assert_equal(chunk_size(20 * 1024 * 1024 * 1024, 1, 1), 16 * 1024 * 1024)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
