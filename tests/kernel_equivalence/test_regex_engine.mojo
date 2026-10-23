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


def test_literal_alternative_prefilter_priority_and_tails() raises:
    # Keep alternatives correlated: `fog` and `dox` pass independent byte
    # sets for fox|dog, but are not candidates for the literal-set scanner.
    var regex = Regex("(fox|dog|elephant|tiger|zebra)[0-9]")
    ref prefilter = regex._compiled[].prefilter
    assert_equal(len(prefilter.literals), 5)
    var miss = String("fog dox ") * 80
    assert_equal(prefilter._find_literals(_span(miss), 0), -1)
    # Every vector lane, the scalar tail, short spans and nonzero starts.
    for padding in range(70):
        var text = String("x") * padding + "elephant7 dog2 fox1"
        var matcher = Matcher(regex)
        var found = matcher.find(_span(text))
        assert_equal(found[0], padding)
        assert_equal(found[1], padding + 9)
        assert_equal(matcher.count(_span(text)), 3)
        assert_equal(matcher.find(_span(text), padding + 1)[0], padding + 10)
        assert_true(matcher.captures(_span(text)))
        assert_equal(matcher.slots[2], padding)
        assert_equal(matcher.slots[3], padding + 8)
        # A view must not read the backing string's excluded tail.
        for n in range(9):
            var view = _span(text)[padding : padding + n]
            assert_false(matcher.is_match(view))
            assert_equal(prefilter._find_literals(view, 0), 0 if n == 8 else -1)
        _ = text
    assert_equal(_replace("(sam|samwise)", "samwise", "<$1>"), "<sam>wise")
    assert_equal(_replace("(samwise|sam)", "samwise", "<$1>"), "<samwise>")
    assert_equal(_replace("(猫|犬犬)", "!犬犬 猫!", "<$1>"), "!<犬犬> <猫>!")
    _ = miss


def test_literal_alternative_analysis_matches_unfiltered_vm() raises:
    # The Pike VM consumes the original program without any prefilter.
    # Cover nullable branches, assertions, bounded expansion and truncation
    # rather than reproducing the prefix extraction's decisions in a test.
    var patterns: List[String] = [
        "(fox|dog) jumps",
        "(cat|elephant)[0-9]+",
        "(fox|dog|elephant|tiger|zebra)[0-9]+",
        "\\b(fox|dog|elephant|tiger|zebra)[0-9]",
        "\\b(fox|dog|elephant|tiger|zebra)\\b",
        "(?m)^(cat|dog|elephant|tiger|zebra)$",
        "(fox|dog|elephant|tiger|zebra)+",
        "(猫|犬犬|fox|dog|elephant|tiger|zebra)+",
        "(a{300}|bb|cc|dd|ee)dog",
        "(aa|bb|cc|dd|ee|)dog",
        "(sam|samwise)",
        "(samwise|sam)",
        "(猫|犬犬)+",
        "(?m)^(cat|dog)$",
        "\\b(cat|elephant)\\b",
        "(cat|elephant)?dog",
        "(cat|elephant)*?dog",
        "(cat|elephant){2,3}dog",
        "(cat|elephant){0}dog",
        "(cat|elephant|)dog",
        "(cat|elephant|[0-9])dog",
        "((ab|cd)(ef|gh)){2}x",
        "(aa|bb|cc|dd|ee|ff|gg|hh|ii)dog",
        "((aa|bb|cc)(dd|ee|ff))dog",
        "(a{300}|bb)dog",
        "(?i)(cat|elephant)dog",
    ]
    var texts: List[String] = [
        "",
        "dog",
        "fog dox",
        "fox jumps dog jumps",
        "cat7 elephant42",
        "samwise sam",
        "xfox dog tigerz",
        "!犬犬 猫!",
        "cat\ndog\n",
        "!elephant!",
        "catdog elephantdog",
        "catcatdog elephantelephantdog",
        "1dog",
        "abcdefghx cdghefghx",
        "iidog aaeedog",
        "CATDOG",
        String("a") * 300 + "dog",
    ]
    for pattern in patterns:
        var regex = Regex(pattern)
        var matcher = Matcher(regex)
        ref program = regex._compiled[].forward
        for text in texts:
            for padding in range(0, 65, 16):
                var input = String("!") * padding + text
                var span = _span(input)
                for start in range(len(span) + 1):
                    # A search start is always a UTF-8 scalar boundary.
                    if start < len(span) and (span[start] & 0xC0) == 0x80:
                        continue
                    var slots = List[Int](length=program.slot_count, fill=-1)
                    var expected = matcher.pike.search(
                        program, span, start, len(span), False, slots
                    )
                    if start == 0:
                        assert_equal(matcher.is_match(span), expected, pattern)
                    var found = matcher.find(span, start)
                    assert_equal(
                        found[0], slots[0] if expected else -1, pattern
                    )
                    assert_equal(
                        found[1], slots[1] if expected else -1, pattern
                    )
                    if expected:
                        assert_true(matcher.captures(span, start), pattern)
                        for k in range(len(slots)):
                            assert_equal(matcher.slots[k], slots[k], pattern)
                _ = input


def test_onepass_captures_preserve_priority_assertions_and_optional_groups() raises:
    var patterns: List[String] = [
        "^(a+)(b*)$",
        "(a+)(b*)",
        "(a*?)(b*)",
        "(a+)|(b+)",
        "(a(b)?)+",
        "^(\\d{4})-(\\d{2})-(\\d{2})$",
        "^ *(\\w+) +(\\w+) +(\\w+)",
        "^(\\S{8})(\\S)\\b",
        "\\b(a+)(b*)\\b",
        "\\B(a+)\\B",
        "(?m)^(a+)(b*)$",
        "()",
        "(a??)",
    ]
    var texts: List[String] = [
        "",
        "a",
        "b",
        "aaab",
        "ababaa",
        "!aabb!",
        "aab\naabb\n",
        "2024-12-31",
        "one two three",
        "  один два три",
        "abcdefgh9!",
        "abcdefghié",
        "abcdefghé!",
        "αaaβ",
    ]
    for pattern in patterns:
        var regex = Regex(pattern)
        assert_true(regex._compiled[].onepass.active(), pattern)
        var matcher = Matcher(regex)
        ref program = regex._compiled[].forward
        for text in texts:
            var span = _span(text)
            for start in range(len(span) + 1):
                if start < len(span) and span[start] & 0xC0 == 0x80:
                    continue
                var slots = List[Int](length=program.slot_count, fill=-1)
                var expected = matcher.pike.search(
                    program, span, start, len(span), False, slots
                )
                assert_equal(matcher.captures(span, start), expected, pattern)
                if expected:
                    for k in range(len(slots)):
                        assert_equal(matcher.slots[k], slots[k], pattern)
            _ = text
    # Ambiguous consuming paths retain the existing capture engines.
    var ambiguous: List[String] = ["(a*)(a*)", "(a|aa)", "(a?)*"]
    for pattern in ambiguous:
        assert_false(Regex(pattern)._compiled[].onepass.active(), pattern)
    assert_equal(_replace("(a(b)?)+", "abaaa", "<$1:$2>"), "<a:b>")
    var long = Regex("^(a+)(b*)$")
    var matcher = Matcher(long)
    var text = String("a") * 300000 + "bbb"
    assert_true(matcher.captures(_span(text)))
    assert_equal(matcher.slots[2], 0)
    assert_equal(matcher.slots[3], 300000)
    assert_equal(matcher.slots[4], 300000)
    assert_equal(matcher.slots[5], 300003)
    _ = text


struct _CaptureJob(Job):
    var regex: Regex
    var first: Int
    var last: Int
    var count: Int

    def __init__(out self, regex: Regex, first: Int, last: Int):
        self.regex = regex.copy()
        self.first = first
        self.last = last
        self.count = 0

    def run(mut self) raises:
        var matcher = Matcher(self.regex)
        for row in range(self.first, self.last):
            var text = String("key ") + String(row)
            assert_true(matcher.captures(_span(text)))
            assert_equal(matcher.slots[2], 0)
            assert_equal(matcher.slots[3], 3)
            assert_equal(matcher.slots[4], 4)
            assert_equal(matcher.slots[5], text.byte_length())
            self.count += 1
            _ = text


def test_onepass_shared_workers_and_resource_fallbacks() raises:
    var regex = Regex("^([a-z]+) ([0-9]+)$")
    assert_true(regex._compiled[].onepass.active())
    var jobs = List[_CaptureJob]()
    for k in range(8):
        jobs.append(_CaptureJob(regex, k * 500, (k + 1) * 500))
    run_jobs(jobs)
    var count = 0
    for k in range(len(jobs)):
        count += jobs[k].count
    assert_equal(count, 4000)
    # A deterministic program beyond the state budget still captures via
    # the original engines. Too many capture slots also stays supported.
    var large = Regex("(a{300})")
    assert_false(large._compiled[].onepass.active())
    var matcher = Matcher(large)
    var text = String("a") * 300
    assert_true(matcher.captures(_span(text)))
    assert_equal(matcher.slots[2], 0)
    assert_equal(matcher.slots[3], 300)
    _ = text
    var many = Regex(String("(a)") * 33)
    assert_false(many._compiled[].onepass.active())
    matcher = Matcher(many)
    text = String("a") * 33
    assert_true(matcher.captures(_span(text)))
    assert_equal(matcher.slots[66], 32)
    assert_equal(matcher.slots[67], 33)
    _ = text


def test_onepass_unicode_fallback_over_the_backtracker_budget() raises:
    var regex = Regex("^(a+)(β+)$")
    assert_true(regex._compiled[].onepass.active())
    var matcher = Matcher(regex)
    var text = String("a") * 100000 + String("β") * 100000
    assert_true(matcher.captures(_span(text)))
    assert_equal(matcher.slots[2], 0)
    assert_equal(matcher.slots[3], 100000)
    assert_equal(matcher.slots[4], 100000)
    assert_equal(matcher.slots[5], 300000)
    _ = text
