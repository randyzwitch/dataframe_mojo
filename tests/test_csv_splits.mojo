"""Record-boundary coverage for the public CSV reader's CountLines scanner.

The retired ``record_splits`` helper described ranges for the former reader.
The replacement pipeline uses Polars' ``CountLines.find_next``: each returned
range ends at the final LF outside quote parity. These tests exercise that
observable boundary contract, including quoted CRLF records and targets at
every alignment around a 64-byte SIMD block.
"""
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
