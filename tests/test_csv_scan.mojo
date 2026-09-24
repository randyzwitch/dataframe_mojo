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


# Tests from test_csv_bits.mojo.
# Structural bit packing and target-selected quote parity.
from std.testing import TestSuite, assert_equal
from dataframe.csv_bits import _mask64, _prefix_xor_inclusive


def test_mask_lane_order() raises:
    for bit in range(64):
        var lanes = SIMD[DType.uint8, 64](0)
        lanes[bit] = 1
        assert_equal(
            _mask64(lanes.eq(SIMD[DType.uint8, 64](1))),
            UInt64(1) << UInt64(bit),
        )


def test_prefix_parity_against_bitwise_reference() raises:
    var mask = UInt64(0)
    for _ in range(260):
        var expected = UInt64(0)
        var parity = UInt64(0)
        for bit in range(64):
            parity ^= (mask >> UInt64(bit)) & 1
            expected |= parity << UInt64(bit)
        assert_equal(_prefix_xor_inclusive(mask), expected)
        mask = mask * UInt64(6364136223846793005) + UInt64(1442695040888963407)


# Tests from test_csv_splits.mojo.
# Record-boundary coverage for the public CSV reader's CountLines scanner.
#
# The retired ``record_splits`` helper described ranges for the former reader.
# The replacement pipeline uses Polars' ``CountLines.find_next``: each returned
# range ends at the final LF outside quote parity. These tests exercise that
# observable boundary contract, including quoted CRLF records and targets at
# every alignment around a 64-byte SIMD block.
from std.testing import TestSuite, assert_equal, assert_true
from dataframe.csv_scan import CountLines


def bytes_of(text: String) -> List[UInt8]:
    var owned = List[UInt8]()
    owned.extend(text.as_bytes())
    return owned^


def check_ranges(text: String, hint: Int, quoting: Bool = True) raises:
    var owned = bytes_of(text)
    var input = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=owned.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(owned),
    )
    var scanner = CountLines(UInt8(34), quoting)
    var offset = 0
    var rows = 0
    while offset < len(input):
        var found = scanner.find_next(input[offset:], hint)
        if found.rows == 0:
            break
        var end = offset + found.last_newline + 1
        assert_true(end > offset)
        assert_equal(input[end - 1], UInt8(10))
        rows += found.rows
        offset = end
    var whole = scanner.count(input)
    assert_equal(rows, whole.rows)
    if whole.rows > 0:
        assert_equal(offset - 1, whole.last_newline)
    _ = owned^


def test_quoted_newlines_and_doubled_quotes_never_end_a_range() raises:
    var text = String('a,"x\ny"\nb,"p""\nq"\nc,3\n')
    for hint in [1, 2, 7, 16, 63, 64, 65, 128]:
        check_ranges(text, hint)


def test_quote_position_at_every_simd_alignment() raises:
    for padding in range(0, 193):
        var head = String()
        for _ in range(padding):
            head += "z"
        check_ranges(head + ',"a\nb"\ntail,1\n', 64)


def test_crlf_and_unterminated_final_record() raises:
    check_ranges('a,"x\r\ny"\r\nb,2\r\nc,3', 3)
    check_ranges('a,"unterminated\nlast,row', 8)


def test_disabled_quoting_treats_quote_bytes_as_data() raises:
    var text = String('a,"x\ny"\nb,2\n')
    var disabled = CountLines(UInt8(34), False)
    var enabled = CountLines(UInt8(34), True)
    var owned = bytes_of(text)
    var input = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=owned.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin](),
        length=len(owned),
    )
    assert_true(disabled.count(input).rows > enabled.count(input).rows)
    _ = owned^


def test_empty_and_single_record_inputs() raises:
    check_ranges("", 64)
    check_ranges("a,b\n", 64)


# Tests from test_csv_splitfields.mojo.
# Polars-style borrowed CSV field splitting and quote-parity coverage.
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


def test_relative_cached_ends_at_block_boundary_and_eof() raises:
    # A comma at byte 62 leaves structural ends at 64, 66, and 68 cached.
    # They must be relative to the position after each selected field.
    var cached = String('"')
    for _ in range(60):
        cached += "x"
    assert_equal(split(cached + '",a,b,c\n'), cached + '"*,a-,b-,c-|')
    assert_equal(split(cached + '",a,b'), cached + '"*,a-,b-|')

    # The last SIMD lane must take the no-shift-by-64 branch. A comma in the
    # first byte after the block must still be found by the scalar tail.
    var last_lane = String('"')
    for _ in range(61):
        last_lane += "x"
    assert_equal(split(last_lane + '",a,b,c\n'), last_lane + '"*,a-,b-,c-|')

    var next_block = String('"')
    for _ in range(62):
        next_block += "x"
    assert_equal(split(next_block + '",a,b,c\n'), next_block + '"*,a-,b-,c-|')


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
