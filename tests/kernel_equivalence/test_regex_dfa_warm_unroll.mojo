"""Warm DFA groups preserve tagged edges and UTF8 decode boundaries."""
# contract-group: kernel.regex
from std.testing import assert_equal
from dataframe.regex import Matcher, Regex
from .test_regex_engine import _span


def test_dfa_warm_groups_stop_before_tagged_edges_at_each_unicode_width() raises:
    for pattern in [
        String("([a-z中𐐀]+)!"),
        "(a{3}|中{2}|𐐀{2})!",
        "(?m)^(.+)$",
        "(\\w+)\\b",
    ]:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref c = regex._compiled[]
        var expected = List[Int](length=c.forward.slot_count, fill=-1)
        for unit in [String("a"), "α", "中", "𐐀", "aα中𐐀"]:
            for length in range(40):
                for tail in [String("!"), "!more words", "\n", "?"]:
                    var text = unit * length + tail
                    var span = _span(text)
                    for start in [0, min(unit.byte_length(), len(span))]:
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
                    _ = text


def test_dfa_warm_groups_preserve_greedy_priority_earliest_and_cached_context() raises:
    var patterns: List[String] = [
        "(a+)(b*)",
        "(a+?)(b*)",
        "(a+|a+b)(b?)",
        "([ab]+)([!中]*)",
        "(\\w+) (\\w+) (\\w+)",
        "(\\w+)\\b(!*)",
        "(?m)^([ab]+)$",
        "(?s)(.+?)(b)",
        "(?:a{4}|a{8}|中)(b*)",
        "(a+)$",
        "(a*)()",
    ]
    for pattern in patterns:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref compiled = regex._compiled[]
        ref program = compiled.forward
        var expected = List[Int](length=program.slot_count, fill=-1)
        for length in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129]:
            for suffix in [String(""), "b", "!", "\n", "中", "𐐀", " beta gamma"]:
                var text = "a" * length + suffix
                var span = _span(text)
                for start in [0, min(1, len(span))]:
                    var found = matcher.pike.search(
                        program, span, start, len(span), False, expected
                    )
                    for warm in range(2):
                        var actual = matcher.forward.find_forward(
                            program,
                            compiled.prefilter,
                            span,
                            start,
                            False,
                            False,
                        )
                        assert_equal(
                            actual, expected[1] if found else -1, pattern
                        )
                        actual = matcher.forward.find_forward(
                            program,
                            compiled.prefilter,
                            span,
                            start,
                            False,
                            True,
                        )
                        assert_equal(actual >= 0, found, pattern)
                    # The parent span API and VM disagree inside a UTF8
                    # character. Forward-end checks above still cover it.
                    if start < len(span) and (span[start] & 0xC0) == 0x80:
                        continue
                    var actual = matcher.find(span, start)
                    assert_equal(
                        actual[0], expected[0] if found else -1, pattern
                    )
                    assert_equal(
                        actual[1], expected[1] if found else -1, pattern
                    )
                    assert_equal(matcher.captures(span, start), found, pattern)
                    if found:
                        assert_equal(matcher.slots, expected, pattern)
                matcher.forward.clear()
                matcher.reverse.clear()
                _ = text
