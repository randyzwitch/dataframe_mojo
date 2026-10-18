"""The regex engine agrees with Polars on a corpus of patterns and strings.

tests/fixtures/regex/polars_corpus.tsv holds, for every pattern and
haystack, what Polars 1.44.2 (Rust's `regex`) returns for `contains`,
`find`, `count_matches`, `extract_all`, `replace_all` with "<$0>" (which
shows where every match is, empty ones included), `replace` with "[${1}]",
`extract` of group 1, and `split` with `literal=False`; or ERROR for a
pattern Polars rejects. The haystacks cover Unicode, newlines, empty text
and no-match cases; the patterns cover the supported syntax, anchors,
empty matches, case folding and the errors. Regenerate it with
`pixi run -e oracle python3 scripts/regex_corpus.py`.
"""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_true

from dataframe.regex import Matcher, Regex

comptime CORPUS = "tests/fixtures/regex/polars_corpus.tsv"


def _span(text: String) -> Span[UInt8, ImmutAnyOrigin]:
    return Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=text.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
        length=text.byte_length(),
    )


def _hex(byte: UInt8) -> Int:
    return Int(byte - 48) if byte <= 57 else Int(byte - 55)


def _unescape(field: StringSlice) -> String:
    var bytes = field.as_bytes()
    var out = List[UInt8]()
    var i = 0
    while i < len(bytes):
        var b = bytes[i]
        if b == 92 and i + 1 < len(bytes):
            var c = bytes[i + 1]
            if c == 116:
                out.append(9)
            elif c == 110:
                out.append(10)
            elif c == 120:
                out.append(UInt8(_hex(bytes[i + 2]) * 16 + _hex(bytes[i + 3])))
                i += 2
            else:
                out.append(c)
            i += 2
            continue
        out.append(b)
        i += 1
    return String(unsafe_from_utf8=out)


def _escape(text: StringSlice) -> String:
    var out = String()
    for cp in text.codepoints():
        var c = Int(cp)
        if c == 92:
            out += "\\\\"
        elif c == 9:
            out += "\\t"
        elif c == 10:
            out += "\\n"
        elif c < 32 or c == 127:
            var digits = "0123456789ABCDEF"
            out += "\\x"
            out += String(digits[byte=c >> 4])
            out += String(digits[byte=c & 15])
        else:
            out += String(cp)
    return out^


def _piece(text: String, start: Int, end: Int) -> String:
    return String(text[byte=start:end])


def _listing(items: List[String]) -> String:
    var out = String(len(items))
    for item in items:
        out += "\x1f"
        out += _escape(item)
    return out^


def _expands(pattern: String) -> Bool:
    """Polars replaces a pattern with no ASCII punctuation as a literal,
    without expanding `$` references in the replacement."""
    for b in pattern.as_bytes():
        if (
            (b >= 33 and b <= 47)
            or (b >= 58 and b <= 64)
            or (b >= 91 and b <= 96)
            or (b >= 123 and b <= 126)
        ):
            return True
    return False


def _results(pattern: String, text: String) raises -> List[String]:
    """What Polars would return, computed with the engine."""
    var regex = Regex(pattern)
    var matcher = Matcher(regex)
    var t = _span(text)
    var out = List[String]()
    out.append("true" if matcher.is_match(t) else "false")
    var first = matcher.find(t)
    out.append(String(first[0]) if first[0] >= 0 else "\\N")
    out.append(String(matcher.count(t)))
    var all = List[String]()
    matcher.reset()
    while True:
        var m = matcher.next(t)
        if m[0] < 0:
            break
        all.append(_piece(text, m[0], m[1]))
    out.append(_listing(all))
    var every = List[UInt8]()
    _ = matcher.replace(t, regex.template("<$0>", _expands(pattern)), -1, every)
    out.append(_escape(String(unsafe_from_utf8=every)))
    var once = List[UInt8]()
    _ = matcher.replace(t, regex.template("[${1}]", _expands(pattern)), 1, once)
    out.append(_escape(String(unsafe_from_utf8=once)))
    if (
        matcher.captures(t)
        and regex.group_count() >= 1
        and matcher.slots[2] >= 0
    ):
        out.append(_escape(_piece(text, matcher.slots[2], matcher.slots[3])))
    else:
        out.append("\\N")
    # Polars splits on an empty pattern into code points; otherwise
    # `regex::Regex::split`: the text between matches, then the rest.
    var parts = List[String]()
    if pattern.byte_length() == 0:
        for cp in text.codepoints():
            parts.append(String(cp))
    else:
        var last = 0
        matcher.reset()
        while True:
            var m = matcher.next(t)
            if m[0] < 0:
                break
            parts.append(_piece(text, last, m[0]))
            last = m[1]
        parts.append(_piece(text, last, text.byte_length()))
    out.append(_listing(parts))
    _ = text
    return out^


def test_engine_matches_polars_corpus() raises:
    var names: List[String] = [
        "contains",
        "find",
        "count_matches",
        "extract_all",
        "replace_all",
        "replace",
        "extract",
        "split",
    ]
    var corpus = open(CORPUS, "r").read()
    var rows = 0
    var failures = List[String]()
    for line in corpus.split("\n"):
        if line.byte_length() == 0:
            continue
        var fields = line.split("\t")
        var pattern = _unescape(fields[0])
        var text = _unescape(fields[1])
        rows += 1
        if fields[2] == "ERROR":
            var raised = False
            try:
                _ = Regex(pattern)
            except:
                raised = True
            if not raised:
                failures.append("accepted invalid pattern " + String(fields[0]))
            continue
        var got: List[String]
        try:
            got = _results(pattern, text)
        except e:
            failures.append(
                "pattern " + String(fields[0]) + " raised: " + String(e)
            )
            continue
        for k in range(len(names)):
            if got[k] != String(fields[2 + k]):
                failures.append(
                    names[k]
                    + " of "
                    + String(fields[0])
                    + " on '"
                    + String(fields[1])
                    + "': expected "
                    + String(fields[2 + k])
                    + ", got "
                    + got[k]
                )
    var report = String()
    for i in range(min(len(failures), 60)):
        report += failures[i] + "\n"
    assert_true(rows > 4000)
    assert_equal(len(failures), 0, report)
