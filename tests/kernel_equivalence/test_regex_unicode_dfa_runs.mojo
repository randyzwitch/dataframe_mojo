"""Unicode DFA packets preserve scalar decoding, cached edges and matches."""
# contract-group: kernel.regex
from std.testing import assert_equal
from dataframe.regex import Matcher, Regex
from dataframe.regex.pikevm import decode
from dataframe.regex.program import Program
from dataframe.regex.unicode_dfa_run import scan_unicode_self_loop
from .test_regex_engine import _span


def _reference_scan(
    program: Program,
    table: List[Int32],
    text: Span[UInt8, ImmutAnyOrigin],
    var at: Int,
    end: Int,
    expected: Int32,
) -> Int:
    if end - at < 16:
        return at
    var width = decode(text, at) & 7
    if width < 2:
        return at
    var lanes = 8 if width == 2 else 4
    while end - at >= 16:
        var pos = at
        for _ in range(lanes):
            var packed = decode(text, pos)
            if packed & 7 != width:
                return at
            var first = Int(text[pos])
            var cp = UInt32(packed >> 3)
            # The scalar engine tolerates malformed encodings. Packets
            # only prove standard UTF8; other bytes stay for that engine.
            var valid = (
                (
                    (
                        width == 2
                        and first >= 0xC2
                        and first < 0xE0
                        and cp >= 0x80
                    )
                    or (
                        width == 3
                        and first >= 0xE0
                        and first < 0xF0
                        and cp >= 0x800
                    )
                    or (
                        width == 4
                        and first >= 0xF0
                        and first <= 0xF4
                        and cp >= 0x10000
                    )
                )
                and cp <= 0x10FFFF
                and (cp < 0xD800 or cp > 0xDFFF)
            )
            for k in range(1, width):
                valid = valid and text[pos + k] >= 0x80 and text[pos + k] < 0xC0
            if not valid:
                return at
            var cls = program.class_of(cp)
            if table[7 + cls] != expected:
                return at
            pos += width
        at = pos
    return at


def test_unicode_dfa_packets_match_scalar_edges_at_bounds_and_class_changes() raises:
    var regex = Regex("([α-ω中日𐐀-𐐅]+)!\\b")
    ref program = regex._compiled[].forward
    var table = List[Int32](length=program.alphabet + 7, fill=-1)
    for expected in [Int32(7), Int32(-32)]:
        for blocked in [
            -1,
            program.class_of(0x3B1),
            program.class_of(0x4E2D),
            program.class_of(0x10400),
        ]:
            for cls in range(program.alphabet):
                table[7 + cls] = -1 if cls == blocked else expected
            for unit in [
                String("αβ"),
                "中文",
                "𐐀𐐁",
                "䶿一",
                "中𐐀",
                "α中",
                "aα中𐐀",
                "a" * 20 + "中" * 20,
            ]:
                for offset in range(8):
                    var body = unit * 24 + "!!!!!!!!!!!!!!!!"
                    var text = "x" * offset + body
                    var span = _span(text)
                    for size in [
                        0,
                        1,
                        7,
                        8,
                        9,
                        15,
                        16,
                        17,
                        23,
                        24,
                        25,
                        31,
                        32,
                        33,
                        body.byte_length() - 1,
                        body.byte_length(),
                    ]:
                        assert_equal(
                            scan_unicode_self_loop(
                                program,
                                table,
                                7,
                                span,
                                offset,
                                offset + size,
                                expected,
                            ),
                            _reference_scan(
                                program,
                                table,
                                span,
                                offset,
                                offset + size,
                                expected,
                            ),
                        )
                    _ = text


def test_unicode_dfa_packets_leave_malformed_truncated_and_mixed_input_unconsumed() raises:
    var regex = Regex("(\\w+)!")
    ref program = regex._compiled[].forward
    var table = List[Int32](length=program.alphabet + 7, fill=7)
    var bytes = List[UInt8](length=96, fill=0)
    var ptr = (
        bytes.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    var units: List[List[UInt8]] = [
        [0xCE, 0xB1],
        [0xE4, 0xB8, 0xAD],
        [0xF0, 0x90, 0x90, 0x80],
    ]
    for unit in units:
        for offset in range(4):
            for bad in range(32):
                for value in [
                    0,
                    0x80,
                    0xC0,
                    0xC1,
                    0xE0,
                    0xED,
                    0xF0,
                    0xF4,
                    0xF5,
                    0xFF,
                ]:
                    for k in range(96):
                        bytes[k] = unit[
                            (k + len(unit) - offset % len(unit)) % len(unit)
                        ]
                    bytes[offset + bad] = UInt8(value)
                    var span = Span[UInt8, ImmutAnyOrigin](
                        unsafe_ptr=ptr, length=80
                    )
                    for size in [
                        15,
                        16,
                        17,
                        23,
                        24,
                        25,
                        31,
                        32,
                        33,
                        63,
                        64,
                        65,
                    ]:
                        assert_equal(
                            scan_unicode_self_loop(
                                program,
                                table,
                                7,
                                span,
                                offset,
                                offset + size,
                                7,
                            ),
                            _reference_scan(
                                program, table, span, offset, offset + size, 7
                            ),
                        )
    _ = bytes


def test_unicode_dfa_packet_searches_preserve_vm_ends_and_capture_priority() raises:
    for pattern in [
        String("(\\w+) (\\w+) (\\w+)"),
        "(\\w+)\\b(!*)",
        "([α-ω中日𐐀-𐐅]+)(!*)",
        "([α-ω中日𐐀-𐐅]+?)(!*)",
        "(?m)^(.+)$",
        "(?s)(.+?)(!)",
    ]:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref c = regex._compiled[]
        var expected = List[Int](length=c.forward.slot_count, fill=-1)
        for unit in [
            String("αβ"),
            "中文",
            "𐐀𐐁",
            "䶿一",
            "中𐐀",
            "α中",
            "aα中𐐀",
            "a" * 20 + "中" * 20,
        ]:
            for length in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33]:
                for tail in [
                    String(""),
                    "!",
                    "!\n",
                    " beta gamma",
                    " 中文 日本",
                    " 𐐀𐐁 𐐂𐐃",
                ]:
                    var text = "!" + unit * length + tail
                    var span = _span(text)
                    for start in [0, 1]:
                        var found = matcher.pike.search(
                            c.forward, span, start, len(span), False, expected
                        )
                        for _ in range(2):
                            assert_equal(
                                matcher.forward.find_forward(
                                    c.forward,
                                    c.prefilter,
                                    span,
                                    start,
                                    False,
                                    False,
                                ),
                                expected[1] if found else -1,
                                pattern,
                            )
                            assert_equal(
                                matcher.forward.find_forward(
                                    c.forward,
                                    c.prefilter,
                                    span,
                                    start,
                                    False,
                                    True,
                                )
                                >= 0,
                                found,
                                pattern,
                            )
                        var actual = matcher.find(span, start)
                        assert_equal(
                            actual[0], expected[0] if found else -1, pattern
                        )
                        assert_equal(
                            actual[1], expected[1] if found else -1, pattern
                        )
                        assert_equal(
                            matcher.captures(span, start), found, pattern
                        )
                        if found:
                            assert_equal(matcher.slots, expected, pattern)
                    _ = text
                matcher.forward.clear()
                matcher.reverse.clear()


def test_unicode_dfa_packets_preserve_permissive_scalar_decode_results() raises:
    var units: List[List[UInt8]] = [
        [0xCE, 0xB1],
        [0xE4, 0xB8, 0xAD],
        [0xF0, 0x90, 0x90, 0x80],
    ]
    var bytes = List[UInt8](length=80, fill=0)
    var ptr = (
        bytes.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    for pattern in [String("(?s)(.+)!"), "(?s)(.+?)!", "([α-ω中日𐐀-𐐅]+)!"]:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref c = regex._compiled[]
        var expected = List[Int](length=c.forward.slot_count, fill=-1)
        for unit in units:
            for bad in range(16):
                for value in [
                    0,
                    0x21,
                    0x7F,
                    0x80,
                    0xBF,
                    0xC0,
                    0xC1,
                    0xDF,
                    0xE0,
                    0xED,
                    0xF0,
                    0xF4,
                    0xF5,
                    0xFF,
                ]:
                    for size in [16, 17, 32, 33, 64, 65]:
                        for k in range(80):
                            bytes[k] = unit[k % len(unit)]
                        bytes[bad] = UInt8(value)
                        bytes[size - 1] = 33
                        var span = Span[UInt8, ImmutAnyOrigin](
                            unsafe_ptr=ptr, length=size
                        )
                        var found = matcher.pike.search(
                            c.forward, span, 0, size, False, expected
                        )
                        for _ in range(2):
                            assert_equal(
                                matcher.forward.find_forward(
                                    c.forward,
                                    c.prefilter,
                                    span,
                                    0,
                                    False,
                                    False,
                                ),
                                expected[1] if found else -1,
                                pattern,
                            )
    _ = bytes


def test_unicode_matched_groups_preserve_last_start_across_variable_widths() raises:
    var regex = Regex("([α-ω中日𐐀-𐐅]+)!")
    var matcher = Matcher(regex)
    ref program = regex._compiled[].forward
    comptime expected = Int32(-32)
    for blocked in [
        -1,
        program.class_of(0x3B1),
        program.class_of(0x4E2D),
        program.class_of(0x10400),
    ]:
        matcher.forward.transitions = List[Int32](
            length=program.alphabet + 7, fill=expected
        )
        for cls in range(program.alphabet):
            if cls == blocked:
                matcher.forward.transitions[7 + cls] = -1
        for unit in [String("αβ"), "中文", "𐐀𐐁", "中𐐀अα", "aα中𐐀"]:
            for offset in range(8):
                var body = unit * 24
                var text = "x" * offset + body
                var span = _span(text)
                for size in [
                    0,
                    1,
                    7,
                    8,
                    15,
                    16,
                    17,
                    31,
                    32,
                    33,
                    body.byte_length(),
                ]:
                    var end = offset + size
                    # Bound the scalar decoder to the same logical slice.
                    var bounded = span[:end]
                    var stop, last = matcher.forward._advance_matched_unicode(
                        program, bounded, 7, offset, end, expected
                    )
                    var pos = offset
                    var last_start = offset
                    while pos < stop:
                        var packed = decode(bounded, pos)
                        var cls = program.class_of(UInt32(packed >> 3))
                        assert_equal(
                            matcher.forward.transitions[7 + cls], expected
                        )
                        last_start = pos
                        pos += packed & 7
                    assert_equal(pos, stop)
                    assert_equal(last, last_start)
                    if end - stop >= 16:
                        var packed = decode(bounded, stop)
                        var cls = program.class_of(UInt32(packed >> 3))
                        assert_equal(
                            matcher.forward.transitions[7 + cls] == expected,
                            False,
                        )
                _ = text
