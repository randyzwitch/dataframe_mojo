"""Terminal greedy DFA shortcuts preserve priorities and capture slots."""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_true

from dataframe.regex import Matcher, Regex
from dataframe.regex.program import SUFFIX_LIMIT
from dataframe.regex.suffix import suffix_end
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
    # Check every vector lane and short tail against an ordinary byte scan.
    var sample = String("a中𐐀" * 7 + "x" + "a" * 17 + "y\x00" + "a" * 33)
    var bytes = _span(sample)
    for count in range(1, 4):
        var code = Int32.MIN | Int32((count << 21) | (120 << 7) | (121 << 14))
        for start in range(len(bytes) + 1):
            for width in range(36):
                var end = min(len(bytes), start + width)
                var first = end
                var last = start
                for at in range(start, end):
                    var b = bytes[at]
                    if (
                        b == 0
                        or (count > 1 and b == 120)
                        or (count > 2 and b == 121)
                    ):
                        first = min(first, at)
                        last = at + 1
                assert_equal(suffix_end[False](bytes, start, end, code), first)
                assert_equal(suffix_end[True](bytes, end, start, code), last)
    _ = sample
    var all = String("a中𐐀" * 64)
    var all_span = _span(all)
    var all_regex = Regex("(?s)(.*)")
    var all_matcher = Matcher(all_regex)
    assert_equal(all_matcher.find(all_span), (0, len(all_span)))
    var forward_shortcut = False
    var reverse_shortcut = False
    for e in all_matcher.forward.transitions:
        forward_shortcut |= Int(e) < SUFFIX_LIMIT
    for e in all_matcher.reverse.transitions:
        reverse_shortcut |= Int(e) < SUFFIX_LIMIT
    assert_true(forward_shortcut)
    assert_true(reverse_shortcut)
    _ = all
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
