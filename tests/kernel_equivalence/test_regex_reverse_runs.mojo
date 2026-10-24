"""Reverse run dispatch preserves scalar cursors and complete-input decoding."""
# contract-group: kernel.regex
from std.testing import assert_equal
from dataframe.regex import Matcher, Regex
from dataframe.regex.pikevm import decode, previous_start
from dataframe.regex.program import Program, consumes
from dataframe.regex.reverse_run import _decode_previous, scan_reverse_run
from .test_regex_engine import _span


def _reference(
    program: Program,
    pc: Int,
    text: Span[UInt8, ImmutAnyOrigin],
    var at: Int,
    floor: Int,
) -> Int:
    while at > floor:
        var prev = previous_start(text, at)
        var cp = UInt32(decode(text, prev) >> 3)
        if prev < floor or not program.step(pc, cp):
            break
        at = prev
    return at


def _check(program: Program, text: Span[UInt8, ImmutAnyOrigin]) raises:
    for pc in range(len(program)):
        if not consumes(program.op[pc]):
            continue
        for end in range(len(text) + 1):
            for floor in [0, end // 2, max(0, end - 1), end]:
                assert_equal(
                    scan_reverse_run(program, pc, text, end, floor),
                    _reference(program, pc, text, end, floor),
                )


def test_reverse_runs_match_scalar_cursor_across_widths_and_subranges() raises:
    for pattern in [
        String("(\\w+)"),
        "([α-ω中日𐐀-𐐅]+)",
        "(α+)",
        "(?s)(.+)",
        "(.+)",
        "(\\W+)",
        "([^中]+)",
    ]:
        var regex = Regex(pattern)
        for unit in [String("abc"), "αβ", "中文", "𐐀𐐁", "中𐐀अα", "aα中𐐀"]:
            for length in [0, 1, 3, 4, 7]:
                var text = "!" + unit * length + "\n!"
                var span = _span(text)
                _check(regex._compiled[].forward, span)
                _check(regex._compiled[].reverse, span)
                _ = text


def test_reverse_runs_preserve_malformed_complete_span_decode() raises:
    var bytes = List[UInt8](length=64, fill=0)
    var ptr = (
        bytes.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    var unit: List[UInt8] = [
        0x61,
        0xCE,
        0xB1,
        0xE4,
        0xB8,
        0xAD,
        0xF0,
        0x90,
        0x90,
        0x80,
    ]
    for pattern in [String("(\\w+)"), "(α+)", "(?s)(.+)", "(.+)"]:
        var regex = Regex(pattern)
        for value in range(256):
            for position in range(8):
                for k in range(64):
                    bytes[k] = unit[k % len(unit)]
                bytes[position] = UInt8(value)
                for offset in [0, 1, 3]:
                    var span = Span[UInt8, ImmutAnyOrigin](
                        unsafe_ptr=ptr.unsafe_offset(offset), length=24
                    )
                    _check(regex._compiled[].forward, span)
    _ = bytes


def test_reverse_mixed_runs_preserve_vm_capture_priorities() raises:
    for pattern in [
        String("(\\w+) (\\w+) (\\w+)"),
        "(\\w+)(!*)",
        "((\\w)+)(!*)",
        "(?s)(.+)(!*)",
        "(.+)(!*)",
        "([α-ω中日𐐀-𐐅]+)(!*)",
        "\\b(\\w+)\\b",
        "([\\p{Han}\\p{Deseret}]+)(!*)",
        "(\\w+?)(!*)",
    ]:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref c = regex._compiled[]
        var expected = List[Int](length=c.forward.slot_count, fill=-1)
        for unit in [String("中𐐀अα"), "aα中𐐀", "αβ", "中文", "𐐀𐐁"]:
            for length in [0, 1, 3, 7, 8, 15, 16, 17, 33]:
                var text = (
                    "!"
                    + unit * length
                    + " "
                    + unit * length
                    + " "
                    + unit * length
                    + "!\n"
                )
                var span = _span(text)
                for start in [0, 1]:
                    var found = matcher.pike.search(
                        c.forward, span, start, len(span), False, expected
                    )
                    for _ in range(2):
                        assert_equal(
                            matcher.captures(span, start), found, pattern
                        )
                        if found:
                            assert_equal(matcher.slots, expected, pattern)
                matcher.forward.clear()
                matcher.reverse.clear()
                _ = text


def test_reverse_decode_covers_every_unicode_scalar() raises:
    var bytes = List[UInt8](length=5, fill=0)
    bytes[0] = 120
    var ptr = (
        bytes.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    for cp in range(0x110000):
        if cp >= 0xD800 and cp <= 0xDFFF:
            continue
        var width = 1
        if cp < 128:
            bytes[1] = UInt8(cp)
        elif cp < 0x800:
            width = 2
            bytes[1] = UInt8(0xC0 | (cp >> 6))
            bytes[2] = UInt8(0x80 | (cp & 63))
        elif cp < 0x10000:
            width = 3
            bytes[1] = UInt8(0xE0 | (cp >> 12))
            bytes[2] = UInt8(0x80 | ((cp >> 6) & 63))
            bytes[3] = UInt8(0x80 | (cp & 63))
        else:
            width = 4
            bytes[1] = UInt8(0xF0 | (cp >> 18))
            bytes[2] = UInt8(0x80 | ((cp >> 12) & 63))
            bytes[3] = UInt8(0x80 | ((cp >> 6) & 63))
            bytes[4] = UInt8(0x80 | (cp & 63))
        var span = Span[UInt8, ImmutAnyOrigin](unsafe_ptr=ptr, length=width + 1)
        var prev, actual = _decode_previous(span, len(span))
        assert_equal(prev, 1)
        assert_equal(actual, UInt32(cp))
    _ = bytes


def _check_decode(text: Span[UInt8, ImmutAnyOrigin]) raises:
    for at in range(1, len(text) + 1):
        var prev = previous_start(text, at)
        var expected = UInt32(decode(text, prev) >> 3)
        var actual_prev, actual = _decode_previous(text, at)
        assert_equal(actual_prev, prev)
        assert_equal(actual, expected)


def test_reverse_decode_matches_full_span_on_malformed_bytes() raises:
    var bytes = List[UInt8](length=10, fill=0)
    var ptr = (
        bytes.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    for a in range(256):
        for b in range(256):
            for k in range(10):
                bytes[k] = UInt8(a if k % 2 == 0 else b)
            for offset in [0, 1]:
                _check_decode(
                    Span[UInt8, ImmutAnyOrigin](
                        unsafe_ptr=ptr.unsafe_offset(offset), length=8
                    )
                )
    for a in [
        0x7F,
        0x80,
        0xBF,
        0xC0,
        0xDF,
        0xE0,
        0xED,
        0xEF,
        0xF0,
        0xF4,
        0xF7,
        0xFF,
    ]:
        for b in [0, 0x7F, 0x80, 0x9F, 0xBF, 0xC0, 0xFF]:
            for c in [0, 0x7F, 0x80, 0x9F, 0xBF, 0xC0, 0xFF]:
                for d in [0, 0x7F, 0x80, 0x9F, 0xBF, 0xC0, 0xFF]:
                    bytes[0] = UInt8(a)
                    bytes[1] = UInt8(b)
                    bytes[2] = UInt8(c)
                    bytes[3] = UInt8(d)
                    bytes[4] = 33
                    _check_decode(
                        Span[UInt8, ImmutAnyOrigin](unsafe_ptr=ptr, length=5)
                    )
    _ = bytes
