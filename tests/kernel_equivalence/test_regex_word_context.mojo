"""Word-context tables are needed only by live Unicode-word assertions."""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_true
from dataframe.regex import Matcher, Regex
from dataframe.regex.syntax import LOOK_UNICODE_WORD


def _span(text: String) -> Span[UInt8, ImmutAnyOrigin]:
    return Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=text.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
        length=text.byte_length(),
    )


def test_word_context_is_only_retained_for_live_unicode_assertions() raises:
    var patterns: List[String] = [
        "literal",
        "(\\w+)",
        "(?m)^([\\p{Han}\\p{Deseret}]+)$",
        "(?-u:\\b)(\\w+)(?-u:\\b)",
        "(?:\\b){0}a",
        "\\b(\\w+)\\b",
        "(?-u:\\b)(\\w+)\\b",
        "\\B(a+)",
        "\\<(\\w+)\\>",
    ]
    for i in range(len(patterns)):
        var regex = Regex(patterns[i])
        ref forward = regex._compiled[].forward
        ref reverse = regex._compiled[].reverse
        var required = i >= 5
        assert_equal(Bool(forward.looks & LOOK_UNICODE_WORD), required)
        assert_equal(Bool(reverse.looks & LOOK_UNICODE_WORD), required)
        assert_equal(len(forward.word) > 0, required)
        assert_equal(len(reverse.word) > 0, required)
        if required:
            assert_equal(forward.word, reverse.word)
            assert_true(
                Int(forward.word.unsafe_ptr()) != Int(reverse.word.unsafe_ptr())
            )


def test_word_context_ascii_and_unicode_assertions_match_polars() raises:
    var patterns: List[String] = [
        "(\\w+)",
        "\\b(\\w+)\\b",
        "(?-u:\\b)(\\w+)(?-u:\\b)",
        "(?-u:\\b)(\\w+)\\b",
        "\\b(\\w+)(?-u:\\b)",
    ]
    var expected: List[List[String]] = [
        ["中𐐀", "aβ", "_1"],
        ["中𐐀", "aβ", "_1"],
        ["a", "_1"],
        ["aβ", "_1"],
        ["a", "_1"],
    ]
    var text = String("中𐐀 aβ _1")
    var span = _span(text)
    for i in range(len(patterns)):
        var matcher = Matcher(Regex(patterns[i]))
        var actual = List[String]()
        while True:
            var found = matcher.next(span)
            if found[0] < 0:
                break
            actual.append(
                String(StringSlice(unsafe_from_utf8=span[found[0] : found[1]]))
            )
        assert_equal(actual, expected[i], patterns[i])
    _ = text


def test_word_context_search_offsets_keep_full_input_context() raises:
    var patterns: List[String] = ["(?-u:\\b)alpha", "\\balpha", "(alpha)"]
    var texts: List[String] = ["xalpha", "中alpha"]
    var starts: List[Int] = [1, 3]
    var expected: List[List[Tuple[Int, Int]]] = [
        [(-1, -1), (3, 8)],
        [(-1, -1), (-1, -1)],
        [(1, 6), (3, 8)],
    ]
    for i in range(len(patterns)):
        var matcher = Matcher(Regex(patterns[i]))
        for j in range(len(texts)):
            var text = texts[j]
            assert_equal(matcher.find(_span(text), starts[j]), expected[i][j])
            _ = text
