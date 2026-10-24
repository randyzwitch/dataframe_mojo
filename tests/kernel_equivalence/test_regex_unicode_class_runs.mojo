"""Batched Unicode membership agrees with canonical ranges and scalar search."""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_true
from dataframe.regex import Matcher, Regex
from dataframe.regex.charclass import class_contains
from dataframe.regex.class_batch import class_batch
from dataframe.regex.class_lookup import build_unicode_membership
from dataframe.regex.pikevm import decode, previous_start
from dataframe.regex.program import Program, compile_program
from dataframe.regex.syntax import parse, LOOK_UNICODE_WORD
from dataframe.regex.unicode_run import _scan_class_width
from .test_regex_engine import _span


def _point(mut points: List[UInt32], cp: UInt32):
    if cp >= 128 and cp <= 0x10FFFF and not (cp >= 0xD800 and cp <= 0xDFFF):
        points.append(cp)


def _check_packets[
    width: Int, lanes: Int
](program: Program, id: Int, points: List[UInt32]) raises:
    var root = program.class_ascii[32 * len(program.class_ranges) + id]
    ref ranges = program.class_ranges[id]
    for begin in range(0, len(points), lanes):
        var cp = SIMD[DType.uint32, lanes]()
        for k in range(lanes):
            cp[k] = points[(begin + k) % len(points)]
        var accepted = class_batch[width, lanes](program, id, cp)
        for k in range(lanes):
            var expected = class_contains(ranges, cp[k])
            assert_true(not accepted[k] or expected)
            if width == 2 or root == 1 or (root >> 62) != 0:
                assert_equal(accepted[k], expected)


def test_batched_unicode_classes_match_ranges_pages_and_directory_budgets() raises:
    var patterns: List[String] = [
        "(\\w+)",
        "([α-ω]+)",
        "([一-丂𐐀-𐐂]+)",
        "([α-ω一-丂𐐀-𐐂]+)",
        "([^一]+)",
        "([a-z\\p{Cyrillic}]+)",
        "([\\p{Han}\\p{Deseret}]+)",
        "([\\p{Greek}\\p{Cyrillic}]+)",
    ]
    for pattern in patterns:
        var ast = parse(pattern)
        var program = compile_program(ast, False)
        var points = List[UInt32]()
        for page in range(0x1100):
            _point(points, UInt32(page << 8))
            _point(points, UInt32((page << 8) + 127))
            _point(points, UInt32((page << 8) + 255))
        for ranges in program.class_ranges:
            for boundary in ranges:
                _point(points, boundary)
                if boundary > 0:
                    _point(points, boundary - 1)
                _point(points, boundary + 1)
        for cp in [128, 2047, 2048, 65535, 65536, 0x10FFFF]:
            _point(points, UInt32(cp))
        var two = List[UInt32]()
        var three = List[UInt32]()
        var four = List[UInt32]()
        for cp in points:
            if cp < 0x800:
                two.append(cp)
            elif cp < 0x10000:
                three.append(cp)
            else:
                four.append(cp)
        for budget in [0, 2, 9, 137, 4096, 1 << 17]:
            build_unicode_membership(
                program.class_ascii,
                program.class_ranges,
                program.word,
                (program.looks & LOOK_UNICODE_WORD) != 0,
                budget,
            )
            for id in range(len(program.class_ranges)):
                _check_packets[2, 8](program, id, two)
                _check_packets[3, 4](program, id, three)
                _check_packets[4, 4](program, id, four)


def _check_consumed[
    width: Int
](program: Program, text: Span[UInt8, ImmutAnyOrigin]) raises:
    var stop = _scan_class_width[width, False](program, 0, text, 0, len(text))
    assert_true(stop >= 0 and stop <= len(text))
    var at = 0
    while at < stop:
        var packed = decode(text, at)
        assert_equal(packed & 7, width)
        assert_true(
            class_contains(program.class_ranges[0], UInt32(packed >> 3))
        )
        at += width
    assert_equal(at, stop)
    stop = _scan_class_width[width, True](program, 0, text, len(text), 0)
    assert_true(stop >= 0 and stop <= len(text))
    at = len(text)
    while at > stop:
        var prev = previous_start(text, at)
        var packed = decode(text, prev)
        assert_equal(at - prev, width)
        assert_equal(packed & 7, width)
        assert_true(
            class_contains(program.class_ranges[0], UInt32(packed >> 3))
        )
        at = prev
    assert_equal(at, stop)


def test_unicode_class_runs_keep_malformed_and_truncated_gathers_in_bounds() raises:
    var ast = parse("(\\w+)")
    var program = compile_program(ast, False)
    var bytes = List[UInt8](length=96, fill=0)
    var ptr = (
        bytes.unsafe_ptr()
        .unsafe_mut_cast[False]()
        .unsafe_origin_cast[ImmutAnyOrigin]()
    )
    for a in range(256):
        for b in range(256):
            for k in range(0, 64, 2):
                bytes[k] = UInt8(a)
                bytes[k + 1] = UInt8(b)
            var span = Span[UInt8, ImmutAnyOrigin](unsafe_ptr=ptr, length=64)
            _check_consumed[2](program, span)
            _check_consumed[3](program, span)
            _check_consumed[4](program, span)
    var values: List[UInt8] = [
        0x80,
        0x9F,
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
    ]
    for a in values:
        for b in values:
            for c in values:
                for d in values:
                    for k in range(0, len(bytes), 4):
                        bytes[k] = a
                        bytes[k + 1] = b
                        bytes[k + 2] = c
                        bytes[k + 3] = d
                    for offset in [0, 1, 3, 15]:
                        for length in [15, 16, 17, 31, 32, 33]:
                            var span = Span[UInt8, ImmutAnyOrigin](
                                unsafe_ptr=ptr.unsafe_offset(offset),
                                length=length,
                            )
                            _check_consumed[2](program, span)
                            _check_consumed[3](program, span)
                            _check_consumed[4](program, span)
    _ = bytes


def test_unicode_class_vector_captures_match_independent_vm() raises:
    var patterns: List[String] = [
        "^(\\w+)(!*)",
        "(\\w+)(!*)",
        "(\\w+) (\\w+) (\\w+)",
        "^([\\p{Han}\\p{Deseret}]+)(!*)",
        "([\\p{Greek}\\p{Cyrillic}]+)(!*)",
        "([^一]+)(一*)",
    ]
    var words: List[String] = [
        "один",
        "中文",
        "𐐀",
        "α",
        "123",
        "中𐐀",
        "䶿一",
        "α" * 32 + "я" * 32,
        "一" * 32 + "㐀" * 32,
    ]
    var suffixes: List[String] = ["", "!", "\n", "a", "一", "𐐀", " 123 文字"]
    for pattern in patterns:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref program = regex._compiled[].forward
        var expected = List[Int](length=program.slot_count, fill=-1)
        for word in words:
            for length in [0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65]:
                for suffix in suffixes:
                    var text = word * length + suffix
                    var span = _span(text)
                    var found = matcher.pike.search(
                        program, span, 0, len(span), False, expected
                    )
                    assert_equal(matcher.captures(span), found, pattern)
                    if found:
                        assert_equal(matcher.slots, expected, pattern)
                    _ = text
