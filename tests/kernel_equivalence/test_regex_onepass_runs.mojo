"""Skipping consuming self-loops preserves captures and match priority."""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_true

from dataframe.regex import Matcher, Regex
from dataframe.regex.onepass import ascii_span
from .test_regex_engine import _span


def test_onepass_run_metadata_excludes_capture_and_assertion_changes() raises:
    var patterns: List[String] = [
        "^(a+)(b*)",
        "^(a+?)",
        "^((a)+)$",
        "^(a+)$",
        "^((?:ab)+)",
    ]
    var expected: List[Bool] = [True, False, False, False, False]
    for i in range(len(patterns)):
        var regex = Regex(patterns[i])
        assert_true(Bool(regex._compiled[].onepass), patterns[i])
        assert_equal(
            regex._compiled[].onepass.value()[].forward.has_loops,
            expected[i],
            patterns[i],
        )


def test_onepass_runs_match_vm_for_priority_captures_and_unicode() raises:
    var patterns: List[String] = [
        "^(a+)(b*)",
        "^(a+?)(b*)",
        "^((a)+)(b*)",
        "^([ab]+)([!中]*)",
        "^(\\w+)(!*)",
        "^(\\p{Han}+)(\\p{Deseret}*)",
        "^(a+)\\b(!*)",
        "^(a+)(\\b)",
        "(?s)^(.+)",
        "(?s)^(.+?)",
        "^([^\\n]+)(\\n*)",
        "^((?:ab)+)(!*)",
        "^(a+)$",
        "^((a)+)$",
    ]
    var alphabet: List[String] = ["a", "b", "中", "𐐀", "!", "\n"]
    for pattern in patterns:
        var regex = Regex(pattern)
        assert_true(Bool(regex._compiled[].onepass), pattern)
        var matcher = Matcher(regex)
        ref program = regex._compiled[].forward
        var slots = List[Int](length=program.slot_count, fill=-1)
        var combinations = 1
        for length in range(5):
            for encoded in range(combinations):
                var digits = encoded
                var text = String()
                for _ in range(length):
                    text += alphabet[digits % len(alphabet)]
                    digits //= len(alphabet)
                var span = _span(text)
                var expected = matcher.pike.search(
                    program, span, 0, len(span), True, slots
                )
                assert_equal(matcher.captures(span), expected, pattern)
                if expected:
                    for k in range(len(slots)):
                        assert_equal(matcher.slots[k], slots[k], pattern)
                _ = text
            combinations *= len(alphabet)


def test_onepass_runs_preserve_subrange_bounds_and_full_text_assertions() raises:
    var patterns: List[String] = [
        "(\\w+)(!*)",
        "(a+)(\\b)",
        "(a+)$",
        "(?s)(.+)",
        "(a+?)",
    ]
    var words: List[String] = ["", "a", "aaaa", "中𐐀中", "a" * 4096, "中" * 4096]
    var suffixes: List[String] = ["", "!", "a", "\n"]
    for pattern in patterns:
        var regex = Regex(pattern)
        assert_true(Bool(regex._compiled[].onepass), pattern)
        var matcher = Matcher(regex)
        ref program = regex._compiled[].forward
        ref onepass = regex._compiled[].onepass.value()[].forward
        var expected_slots = List[Int](length=program.slot_count, fill=-1)
        for word in words:
            for suffix in suffixes:
                var text = "!" + word + suffix
                var span = _span(text)
                var end = 1 + word.byte_length()
                var expected = matcher.pike.search(
                    program, span, 1, end, True, expected_slots
                )
                assert_equal(
                    onepass.search[True](
                        program,
                        span,
                        1,
                        end,
                        matcher.pike.scratch,
                        matcher.slots,
                    ),
                    expected,
                    pattern,
                )
                if expected:
                    for k in range(len(expected_slots)):
                        assert_equal(
                            matcher.slots[k], expected_slots[k], pattern
                        )
                if ascii_span(span, 1, end):
                    assert_equal(
                        onepass.search[False](
                            program,
                            span,
                            1,
                            end,
                            matcher.pike.scratch,
                            matcher.slots,
                        ),
                        expected,
                        pattern,
                    )
                    if expected:
                        for k in range(len(expected_slots)):
                            assert_equal(
                                matcher.slots[k], expected_slots[k], pattern
                            )
                _ = text
