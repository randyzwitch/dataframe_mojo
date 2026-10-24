"""ASCII Perl classes retain their semantics without Unicode table parsing."""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_raises
from dataframe.regex import Matcher, Regex
from .test_regex_engine import _span


def test_ascii_perl_classes_match_explicit_ranges_and_flag_scopes() raises:
    var patterns: List[String] = [
        "(?-u:\\d)",
        "(?-u:\\w)",
        "(?-u:\\s)",
        "(?-u:[\\d])",
        "(?-u:[\\w])",
        "(?-u:[\\s])",
        "(?-u:\\w)(?u:\\w)",
        "(?-u:\\d)(?u:\\d)",
        "(?-u:\\s)(?u:\\s)",
        "(?-u:(\\w+)) (\\w+)",
        "(?-ui:(\\w+)) (\\w+)",
    ]
    var references: List[String] = [
        "[0-9]",
        "[0-9A-Z_a-z]",
        "[\\x09-\\x0d ]",
        "[0-9]",
        "[0-9A-Z_a-z]",
        "[\\x09-\\x0d ]",
        "[0-9A-Z_a-z]\\w",
        "[0-9]\\d",
        "[\\x09-\\x0d ]\\s",
        "([0-9A-Z_a-z]+) (\\w+)",
        "([0-9A-Z_a-z]+) (\\w+)",
    ]
    var texts: List[String] = [
        "",
        "a中",
        "a𐐀",
        "aé",
        "1١",
        " \u00a0",
        "_中",
        "a 中",
        "Ab_c é𐐀",
    ]
    for cp in range(128):
        texts.append(chr(cp))
    for i in range(len(patterns)):
        var regex = Regex(patterns[i])
        var reference = Regex(references[i])
        var matcher = Matcher(regex)
        var oracle = Matcher(reference)
        for text in texts:
            var span = _span(text)
            assert_equal(
                matcher.is_match(span), oracle.is_match(span), patterns[i]
            )
            var expected = oracle.captures(span)
            assert_equal(matcher.captures(span), expected, patterns[i])
            if expected:
                assert_equal(matcher.slots, oracle.slots, patterns[i])
            _ = text


def test_ascii_negated_perl_classes_still_reject_invalid_utf8_matches() raises:
    var patterns: List[String] = [
        "(?-u:\\D)",
        "(?-u:\\W)",
        "(?-u:\\S)",
        "(?-u:[\\D])",
        "(?-u:[\\W])",
        "(?-u:[\\S])",
    ]
    for pattern in patterns:
        with assert_raises():
            _ = Regex(pattern)
