"""NDJSON framing across SIMD tails, CRLF and escaped string content."""
# contract-group: kernel.json_source
from std.testing import assert_equal, assert_true
from dataframe.json_reader import json_bytes
from dataframe.io.json_source import JsonLines


def test_json_line_framing_skips_whitespace_preserves_line_numbers_and_tail() raises:
    var text = String('\n \t\r\n{"a":"escaped\\nline"}\r\n{}')
    var lines = JsonLines(json_bytes(text))
    var first = lines.next()
    assert_true(Bool(first))
    assert_equal(first.value().line, 2)
    assert_equal(first.value().row, 0)
    assert_equal(
        String(
            unsafe_from_utf8=List[UInt8](
                lines.span()[first.value().start : first.value().end]
            )
        ),
        '{"a":"escaped\\nline"}\r',
    )
    var second = lines.next()
    assert_true(Bool(second))
    assert_equal(second.value().line, 3)
    assert_equal(second.value().row, 1)
    assert_equal(second.value().end - second.value().start, 2)
    assert_true(not lines.next())
    assert_equal(lines.rows, 2)


def test_json_line_framing_all_block_boundaries_and_long_rows() raises:
    for width in [0, 1, 11, 12, 63, 64, 65, 127, 128, 129, 1024]:
        var payload = String("")
        for _ in range(width):
            payload += "x"
        var row = '{"v":"' + payload + '"}'
        var text = row + "\n" + row + "\n\n "
        var lines = JsonLines(json_bytes(text))
        for i in range(2):
            var record = lines.next()
            assert_true(Bool(record))
            assert_equal(record.value().row, i)
            assert_equal(record.value().line, i)
            assert_equal(
                record.value().end - record.value().start, row.byte_length()
            )
        assert_true(not lines.next())
