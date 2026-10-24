"""Reverse run dispatch preserves scalar cursors and complete-input decoding."""
# contract-group: kernel.regex
from std.testing import assert_equal
from dataframe.regex import Matcher, Regex
from dataframe.regex.pikevm import decode, previous_start
from dataframe.regex.program import Program, consumes
from dataframe.regex.reverse_run import scan_reverse_run
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
    for pattern in [String("(\\w+)"), "(α+)", "(?s)(.+)", "(.+)"]:
        var regex = Regex(pattern)
        for value in range(256):
            for position in range(8):
                for k in range(64):
                    bytes[k] = UInt8(
                        [
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
                        ][k % 10]
                    )
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
