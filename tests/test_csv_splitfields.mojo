"""Polars-style borrowed CSV field splitting and quote-parity coverage."""
from std.testing import TestSuite, assert_equal, assert_true
from dataframe.csv_splitfields import CsvSplitFields


def input(text: String) -> List[UInt8]:
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    return bytes^


def split(
    text: String, quote: UInt8 = UInt8(34), quoting: Bool = True
) -> String:
    var owned = input(text)
    var bytes = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=owned.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(owned),
    )
    var fields = CsvSplitFields(UInt8(44), quote, quoting)
    var out = String()
    while True:
        var field = fields.next(bytes)
        if not field:
            break
        var value = field.value().copy()
        out += String(unsafe_from_utf8=value.bytes(bytes))
        out += "*" if value.needs_escaping else "-"
        out += "|" if value.ends_record else ","
    _ = owned^
    return out^


def test_plain_fields_and_trailing_empty() raises:
    assert_equal(split("a,b,\n"), "a-,b-,-|")
    assert_equal(split(""), "-|")


def test_quoted_delimiters_doubled_quotes_and_newlines() raises:
    assert_equal(
        split('a,"x,y","say ""hello""",z\n'),
        'a-,"x,y"*,"say ""hello"""*,z-|',
    )
    assert_equal(
        split('"line one\nline two",tail\n'), '"line one\nline two"*,tail-|'
    )


def test_structural_cache_crosses_64_byte_boundary() raises:
    var text = String('"')
    for _ in range(79):
        text += "x"
    text += '",a,b,c\n'
    var expected = String('"')
    for _ in range(79):
        expected += "x"
    expected += '"*,a-,b-,c-|'
    assert_equal(split(text), expected)


def test_quoted_close_at_every_simd_offset() raises:
    # The opening quote stays at byte zero; every closing quote position from
    # the scalar tail through three SIMD blocks must expose the same comma.
    for width in range(0, 193):
        var text = String('"')
        for _ in range(width):
            text += "x"
        text += '",tail\n'
        var expected = String('"')
        for _ in range(width):
            expected += "x"
        expected += '"*,tail-|'
        assert_equal(split(text), expected, "closing quote at " + String(width))


def test_field_bytes_borrow_the_input_and_keep_cr_for_decoder() raises:
    var owned = input("a,b\r\n")
    var bytes = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=owned.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(owned),
    )
    var fields = CsvSplitFields(UInt8(44))
    var first = fields.next(bytes).value().copy()
    var second = fields.next(bytes).value().copy()
    assert_equal(Int(first.bytes(bytes).unsafe_ptr()), Int(bytes.unsafe_ptr()))
    assert_equal(
        Int(second.bytes(bytes).unsafe_ptr()),
        Int(bytes.unsafe_ptr().unsafe_offset(2)),
    )
    assert_equal(String(unsafe_from_utf8=second.bytes(bytes)), "b\r")
    assert_true(second.ends_record)
    _ = owned^


def test_leading_quote_controls_polars_style_parity_and_consumed_offset() raises:
    var owned = input('plain"quote,tail\n')
    var bytes = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=owned.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(owned),
    )
    var fields = CsvSplitFields(UInt8(44))
    var first = fields.next(bytes).value().copy()
    # SplitFields only switches to quote parity when a field starts quoted;
    # strict CSV grammar is intentionally checked by the decoder.
    assert_equal(String(unsafe_from_utf8=first.bytes(bytes)), 'plain"quote')
    assert_true(not first.needs_escaping)
    assert_equal(first.terminator, UInt8(44))
    assert_equal(fields.consumed(), first.end + 1)
    var second = fields.next(bytes).value().copy()
    assert_equal(second.terminator, UInt8(10))
    _ = owned^


def test_custom_quote_and_quote_disabled() raises:
    assert_equal(split("'a,b',c\n", UInt8(39)), "'a,b'*,c-|")
    assert_equal(split('"a,b",c\n', UInt8(34), False), '"a-,b"-,c-|')


def test_consumed_reaches_eof_for_plain_and_quoted_fields() raises:
    var cases: List[String] = ["plain", '"quoted"', "a,"]
    for text in cases:
        var owned = input(text)
        var bytes = Span[UInt8, ImmutAnyOrigin](
            unsafe_ptr=owned.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin](),
            length=len(owned),
        )
        var fields = CsvSplitFields(UInt8(44))
        while fields.next(bytes):
            pass
        assert_equal(fields.consumed(), len(owned))
        _ = owned^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
