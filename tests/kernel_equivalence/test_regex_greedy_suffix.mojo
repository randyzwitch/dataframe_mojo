"""Terminal greedy DFA shortcuts preserve priorities and capture slots."""
# contract-group: kernel.regex
from std.testing import assert_equal

from dataframe.regex import Matcher, Regex
from .test_regex_engine import _span


def test_greedy_suffix_metadata_requires_terminal_ascii_exits() raises:
    var patterns: List[String] = [
        ".*",
        "(?s).*",
        "(.+)",
        "foo(.*)",
        "[^x]*",
        "[^xy]*",
        "[^xyz]*",
        ".*?",
        ".{0,5}",
        ".*$",
        "\\w+",
        "[^wxyz]*",
        "[^中]*",
        "(?:ab)*",
    ]
    for i in range(len(patterns)):
        var regex = Regex(patterns[i])
        assert_equal(regex._compiled[].forward.has_suffix, i < 7, patterns[i])


def test_greedy_suffix_matches_vm_for_priority_and_captures() raises:
    var patterns: List[String] = [
        ".*",
        "(?s).*",
        ".+",
        "(.+)",
        "(.*)",
        "((.)*)",
        "a|.*",
        "x[^x]*|",
        "[^x]*x|",
        "\n.*|",
        "aa|.*",
        "(a)|(.*)",
        "a.*",
        ".*a",
        "a(.*)",
        "(.*)a",
        ".*?",
        "(.+?)",
        ".{0,5}",
        ".*$",
        "[^x]*",
        "([^xy]+)",
        "[^xyz]*",
        "[^wxyz]*",
        "[^中]*",
        "(?i)[^x]*",
        "[^\\x00]*",
        "(a+)(.*)",
        "(?:ab)*",
        "(?s)((.*)+)",
        "(.*)\\b",
        "\\b(.*)",
        "(?m)^(.*)",
    ]
    var alphabet: List[String] = ["a", "x", "中", "𐐀", "\n", "\x00"]
    for pattern in patterns:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref program = regex._compiled[].forward
        var slots = List[Int](length=program.slot_count, fill=-1)
        var combinations = 1
        for length in range(5):
            for encoded in range(combinations):
                var text = String()
                var digits = encoded
                for _ in range(length):
                    text += alphabet[digits % len(alphabet)]
                    digits //= len(alphabet)
                var span = _span(text)
                var expected = matcher.pike.search(
                    program, span, 0, len(span), False, slots
                )
                assert_equal(matcher.is_match(span), expected, pattern)
                assert_equal(matcher.captures(span), expected, pattern)
                if expected:
                    for k in range(len(slots)):
                        assert_equal(matcher.slots[k], slots[k], pattern)
                _ = text
            combinations *= len(alphabet)


def test_greedy_suffix_scans_vectors_and_respects_search_start() raises:
    var patterns: List[String] = [
        "(.*)",
        "(?s)(.*)",
        "(.+)",
        "([^x]+)",
        "([^xy]*)",
        "([^xyz]*)",
        "a(.*)",
        "(.*)a",
        "a|.*",
        "x[^x]*|",
        "[^x]*x|",
        "\n.*|",
        "aa|.*",
        "((.)*)",
        "([^\\x00]*)",
    ]
    var bodies: List[String] = [
        "a" * 257,
        "中𐐀" * 129,
        "a" * 31 + "x" + "中" * 31,
        "a" * 32 + "y" + "𐐀" * 32,
        "a" * 33 + "z\n\x00" + "a" * 33,
    ]
    for pattern in patterns:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref program = regex._compiled[].forward
        var slots = List[Int](length=program.slot_count, fill=-1)
        for body in bodies:
            var text = "!" + body + "\nx!"
            var span = _span(text)
            for start in range(2):
                var expected = matcher.pike.search(
                    program, span, start, len(span), False, slots
                )
                assert_equal(matcher.captures(span, start), expected, pattern)
                if expected:
                    for k in range(len(slots)):
                        assert_equal(matcher.slots[k], slots[k], pattern)
            _ = text
