"""Regex engine properties the Polars corpus does not reach: one compiled
regex shared by several workers, the DFA cache overflowing on a pattern
with many states (and the Pike VM taking over), prefilters on long
haystacks, capture slots, replacement templates, escaping and the errors
for unsupported syntax."""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from dataframe.parallel import Job, run_jobs
from dataframe.regex import Matcher, Regex, escape


def _span(text: String) -> Span[UInt8, ImmutAnyOrigin]:
    return Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=text.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
        length=text.byte_length(),
    )


def _replace(
    pattern: String, text: String, replacement: String
) raises -> String:
    var regex = Regex(pattern)
    var matcher = Matcher(regex)
    var out = List[UInt8]()
    _ = matcher.replace(_span(text), regex.template(replacement), -1, out)
    _ = text
    return String(unsafe_from_utf8=out)


def _haystack(row: Int) -> String:
    var words: List[String] = ["alpha", "beta", "x9", "Straße", "日本", "id-42"]
    var text = String()
    for k in range(row % 7 + 1):
        text += words[(row * 3 + k) % len(words)] + " "
    return text^


struct _CountJob(Job):
    var regex: Regex
    var first: Int
    var last: Int
    var total: Int

    def __init__(out self, regex: Regex, first: Int, last: Int):
        self.regex = regex.copy()
        self.first = first
        self.last = last
        self.total = 0

    def run(mut self) raises:
        var matcher = Matcher(self.regex)
        for row in range(self.first, self.last):
            var text = _haystack(row)
            self.total += matcher.count(_span(text))
            _ = text


def test_one_regex_shared_by_workers() raises:
    var regex = Regex("\\b[a-z]+\\d*\\b|\\d+")
    var rows = 20000
    var expected = 0
    var matcher = Matcher(regex)
    for row in range(rows):
        var text = _haystack(row)
        expected += matcher.count(_span(text))
        _ = text
    var jobs = List[_CountJob]()
    for k in range(8):
        jobs.append(_CountJob(regex, k * rows // 8, (k + 1) * rows // 8))
    run_jobs(jobs)
    var total = 0
    for k in range(len(jobs)):
        total += jobs[k].total
    assert_equal(total, expected)
    assert_true(expected > rows)


def test_dfa_cache_overflow_falls_back_to_the_pike_vm() raises:
    # Unanchored, a(a|b){20}c tracks which of the last 25 code points were
    # `a`: up to 2^25 DFA states over random a/b text, past the cache.
    var regex = Regex("a(a|b){24}c")
    var matcher = Matcher(regex)
    var bytes = List[UInt8]()
    var state: UInt64 = 12345
    for _ in range(250_000):
        state = state * 6364136223846793005 + 1442695040888963407
        bytes.append(UInt8(97) if (state >> 33) & 1 else UInt8(98))
    var text = String(unsafe_from_utf8=bytes)
    var tail = text + "a" + String("b") * 24 + "c"
    assert_false(matcher.is_match(_span(text)))
    assert_true(matcher.forward.resets > 0)
    assert_true(matcher.is_match(_span(tail)))
    var m = matcher.find(_span(tail))
    assert_equal(m[0], text.byte_length())
    assert_equal(m[1], tail.byte_length())
    assert_equal(matcher.count(_span(tail + tail)), 2)
    _ = text
    _ = tail


def test_prefilters_on_long_haystacks() raises:
    var filler = String("x") * 100000
    var cases: List[String] = [
        "needle",
        "(?i)needle",
        "nee+dle",
        "[mn]eedle",
        "\\bneedle\\b",
    ]
    for pattern in cases:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        var hit = filler + " needle " + filler
        assert_true(matcher.is_match(_span(hit)), pattern)
        assert_false(matcher.is_match(_span(filler)), pattern)
        var m = matcher.find(_span(hit))
        assert_equal(m[0], 100001, pattern)
        assert_equal(m[1], 100007, pattern)
        assert_equal(matcher.count(_span(hit + hit)), 2, pattern)
        _ = hit
    _ = filler


def test_capture_slots() raises:
    var regex = Regex("(?P<year>\\d{4})-(\\d{2})(-(\\d{2}))?")
    assert_equal(regex.group_count(), 4)
    assert_equal(regex.group_index("year"), 1)
    assert_equal(regex.group_index("month"), -1)
    var matcher = Matcher(regex)
    var text = String("on 2024-01 and 1999-12-31")
    assert_true(matcher.captures(_span(text)))
    var expected: List[Int] = [3, 10, 3, 7, 8, 10, -1, -1, -1, -1]
    for k in range(len(expected)):
        assert_equal(matcher.slots[k], expected[k])
    assert_true(matcher.captures(_span(text), 10))
    expected = [15, 25, 15, 19, 20, 22, 22, 25, 23, 25]
    for k in range(len(expected)):
        assert_equal(matcher.slots[k], expected[k])
    _ = text
    # A span too long for the backtracker's visited set: the Pike VM
    # reports the same groups.
    var long = Regex("^(a+)(b*)(x|y)$")
    var both = Matcher(long)
    var short = String("aaab") + "x"
    var wide = String("a") * 200000 + String("b") * 3 + "y"
    assert_true(both.captures(_span(short)))
    expected = [0, 5, 0, 3, 3, 4, 4, 5]
    for k in range(len(expected)):
        assert_equal(both.slots[k], expected[k])
    assert_true(both.captures(_span(wide)))
    expected = [0, 200004, 0, 200000, 200000, 200003, 200003, 200004]
    for k in range(len(expected)):
        assert_equal(both.slots[k], expected[k])
    _ = short
    _ = wide


def test_replacement_templates() raises:
    var pattern = "(?P<first>\\w+) (?P<last>\\w+)"
    assert_equal(
        _replace(pattern, "ada lovelace", "$last, $first"), "lovelace, ada"
    )
    assert_equal(_replace(pattern, "ada lovelace", "${2}_$1"), "lovelace_ada")
    assert_equal(_replace(pattern, "ada lovelace", "$2x"), "")
    assert_equal(_replace(pattern, "ada lovelace", "$$1 $"), "$1 $")
    assert_equal(_replace(pattern, "ada lovelace", "${9}|${nope}|${x"), "||${x")
    assert_equal(_replace("a", "banana", "<$0>"), "b<a>n<a>n<a>")
    var regex = Regex("(a)")
    var matcher = Matcher(regex)
    var out = List[UInt8]()
    _ = matcher.replace(_span("banana"), regex.template("$1$1", False), 2, out)
    assert_equal(String(unsafe_from_utf8=out), "b$1$1n$1$1na")


def test_literal_detection_and_escape() raises:
    assert_equal(Regex("abc").literal().value(), "abc")
    assert_equal(Regex("a\\.b\\*").literal().value(), "a.b*")
    assert_false(Regex("(?i)abc").literal())
    assert_false(Regex("^abc").literal())
    assert_false(Regex("a.c").literal())
    var text = "a.b*c+(d)|[e]{f}^$#&-~\\"
    assert_equal(
        escape(text),
        "a\\.b\\*c\\+\\(d\\)\\|\\[e\\]\\{f\\}\\^\\$\\#\\&\\-\\~\\\\",
    )
    var regex = Regex(escape(text))
    assert_equal(regex.literal().value(), text)


def test_unsupported_syntax_is_rejected() raises:
    with assert_raises(contains="look-around"):
        _ = Regex("foo(?=bar)")
    with assert_raises(contains="look-around"):
        _ = Regex("(?<!x)y")
    with assert_raises(contains="backreferences are not supported"):
        _ = Regex("(a)\\1")
    with assert_raises(contains="Unknown Unicode property"):
        _ = Regex("\\p{NotAProperty}")
    with assert_raises(contains="unclosed group"):
        _ = Regex("(ab")
    with assert_raises(contains="regex is too large"):
        _ = Regex("((a{100}){100}){100}")


def test_unicode_properties_and_ascii_word_anchors_with_long_capture_spans() raises:
    var regex = Regex("(?-u:\\<)(a+)(α+)\\>")
    var matcher = Matcher(regex)
    var text = "!" + String("a") * 100000 + String("α") * 100000 + "!"
    assert_true(matcher.is_match(_span(text)))
    assert_true(matcher.captures(_span(text)))
    var expected: List[Int] = [1, 300001, 1, 100001, 100001, 300001]
    for k in range(len(expected)):
        assert_equal(matcher.slots[k], expected[k])
    assert_equal(matcher.count(_span(text)), 1)
    _ = text
