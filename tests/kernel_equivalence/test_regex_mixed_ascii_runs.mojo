"""ASCII vector runs inside Unicode searches preserve capture boundaries."""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_true
from dataframe.regex import Matcher, Regex
from .test_regex_engine import _span


def test_mixed_ascii_runs_preserve_unicode_tail_and_capture_slots() raises:
    var patterns: List[String] = [
        "(a+)(b*)",
        "([ab]+)([!中]*)",
        "(\\w+) (\\w+) (\\w+)",
        "([\\p{Han}a-z]+)([!𐐀]*)",
        "(a+)\\b(!*)",
        "((a)+)(b*)",
        "(?s)(.+)",
        "(a+?)(b*)",
    ]
    for pattern in patterns:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref program = regex._compiled[].forward
        var expected = List[Int](length=program.slot_count, fill=-1)
        for length in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129]:
            for prefix in [String("🙂"), "中!", "𐐀!中!", "中"]:
                for suffix in [String("中"), "𐐀", "!", "b中", " 123 文字"]:
                    var text = prefix + "a" * length + suffix
                    var span = _span(text)
                    for start in [0, prefix.byte_length()]:
                        var found = matcher.pike.search(
                            program, span, start, len(span), False, expected
                        )
                        assert_equal(
                            matcher.captures(span, start), found, pattern
                        )
                        if found:
                            assert_equal(matcher.slots, expected, pattern)
                    _ = text


def test_unicode_onepass_vector_runs_respect_subranges_and_full_assertions() raises:
    var patterns: List[String] = [
        "(a+)(b*)",
        "([ab]+)([!中]*)",
        "(\\w+) (\\w+) (\\w+)",
        "(a+)\\b(!*)",
        "(?s)(.+)",
        "(a+?)(b*)",
    ]
    for pattern in patterns:
        var regex = Regex(pattern)
        assert_true(Bool(regex._compiled[].onepass), pattern)
        var matcher = Matcher(regex)
        ref program = regex._compiled[].forward
        ref onepass = regex._compiled[].onepass.value()[].forward
        var expected = List[Int](length=program.slot_count, fill=-1)
        for length in [15, 16, 17, 31, 32, 33, 64, 65, 129]:
            for tail in [String(""), "b", "中", " 123 文字", "!𐐀"]:
                var body = "a" * length + tail
                var text = "🙂" + body + "𐐀!"
                var span = _span(text)
                var begin = 4
                var end = begin + body.byte_length()
                var found = matcher.pike.search(
                    program, span, begin, end, True, expected
                )
                assert_equal(
                    onepass.search[True](
                        program,
                        span,
                        begin,
                        end,
                        matcher.pike.scratch,
                        matcher.slots,
                    ),
                    found,
                    pattern,
                )
                if found:
                    assert_equal(matcher.slots, expected, pattern)
                _ = text
