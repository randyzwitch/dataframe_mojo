"""Reverse deterministic captures agree with the forward Pike VM."""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_true
from dataframe.regex import Matcher, Regex
from dataframe.regex.onepass import ascii_span, compile_onepass
from dataframe.regex.program import compile_program
from dataframe.regex.syntax import parse
from .test_regex_engine import _span


def test_reverse_onepass_capture_priorities_and_repeated_groups() raises:
    var patterns: List[String] = [
        "(a+)(b*)",
        "(a+?)(b*)",
        "((a)+)(b*)",
        "((a)?b)+",
        "((ab)?c)+",
        "()(a*)",
        "(a*)()",
        "(a?)(b*)",
        "(a*?)(b*)",
        "(a*)(b*?)",
        "(a)|(b)",
        "((a|b)+)",
        "(a+)\\b",
        "\\b(a+)",
        "(a+)$",
        "(?m)^(a+)",
        "(\\p{Han}+)(\\p{Deseret}*)",
        "([^x]+)",
        "(?s)(.*)",
        "((?:ab)+)",
        "(a+)(!)?",
        "(\\w+)(!*)",
    ]
    var alphabet: List[String] = ["a", "b", "c", "中", "𐐀", "!", "\n"]
    var checked = 0
    for pattern in patterns:
        var regex = Regex(pattern)
        if not Bool(regex._compiled[].onepass):
            continue
        var ast = parse(pattern)
        var reverse = compile_program(ast, True, True)
        var engine = compile_onepass(reverse)
        if not engine.active():
            continue
        var matcher = Matcher(regex)
        ref forward = regex._compiled[].forward
        var expected = List[Int](length=forward.slot_count, fill=-1)
        var current = List[Int](length=forward.slot_count, fill=-1)
        var actual = List[Int](length=forward.slot_count, fill=-1)
        var combinations = 1
        for length in range(5):
            for encoded in range(combinations):
                var text = String()
                var floors: List[Int] = [0]
                var digits = encoded
                for _ in range(length):
                    text += alphabet[digits % len(alphabet)]
                    digits //= len(alphabet)
                    floors.append(text.byte_length())
                var span = _span(text)
                for floor in floors:
                    if not matcher.pike.search(
                        forward, span, floor, len(span), False, expected
                    ):
                        continue
                    var message = pattern + " @" + String(floor) + " " + text
                    assert_true(
                        engine.search_reverse[True](
                            reverse, span, expected[1], floor, current, actual
                        ),
                        message,
                    )
                    assert_equal(actual, expected, message)
                    if ascii_span(span, floor, expected[1]):
                        assert_true(
                            engine.search_reverse[False](
                                reverse,
                                span,
                                expected[1],
                                floor,
                                current,
                                actual,
                            ),
                            message,
                        )
                        assert_equal(actual, expected, message)
                    checked += 1
                _ = text
            combinations *= len(alphabet)
    assert_true(checked > 10000)


def test_reverse_onepass_long_runs_and_full_text_assertions() raises:
    var patterns: List[String] = [
        "(a+)",
        "((a)+)",
        "(\\w+)(!*)",
        "(a+)\\b",
        "(?m)(a+)$",
        "(?s)(.*)",
    ]
    var words: List[String] = ["a" * 4096, "中𐐀" * 1024]
    for pattern in patterns:
        var regex = Regex(pattern)
        assert_true(Bool(regex._compiled[].onepass), pattern)
        var ast = parse(pattern)
        var reverse = compile_program(ast, True, True)
        var engine = compile_onepass(reverse)
        assert_true(engine.active(), pattern)
        var matcher = Matcher(regex)
        ref forward = regex._compiled[].forward
        var expected = List[Int](length=forward.slot_count, fill=-1)
        var current = List[Int](length=forward.slot_count, fill=-1)
        var actual = List[Int](length=forward.slot_count, fill=-1)
        for word in words:
            for suffix in [String(""), String("!"), String("\n")]:
                var text = "!" + word + suffix
                var span = _span(text)
                if matcher.pike.search(
                    forward, span, 1, len(span), False, expected
                ):
                    assert_true(
                        engine.search_reverse[True](
                            reverse, span, expected[1], 1, current, actual
                        ),
                        pattern,
                    )
                    assert_equal(actual, expected, pattern)
                _ = text
