"""Numeric constructors preserve the frozen Polars 1.44.2 Unicode classes."""
# contract-group: kernel.regex
from std.testing import assert_equal
from dataframe.regex.charclass import digit_class, space_class, word_class


def _hex(text: String) -> UInt32:
    var value: UInt32 = 0
    for byte in text.as_bytes():
        var digit = UInt32(byte - 48) if byte <= 57 else UInt32(byte - 55)
        value = (value << 4) | digit
    return value


def _ranges(index: Int) -> List[UInt32]:
    if index == 0:
        return word_class()
    if index == 1:
        return digit_class()
    return space_class()


def test_numeric_perl_ranges_preserve_every_frozen_endpoint() raises:
    var fixture = open(
        "tests/fixtures/regex/unicode_perl_ranges.tsv", "r"
    ).read()
    var records = fixture.split("\n")
    var names: List[String] = ["WORD", "DIGIT", "SPACE"]
    for index in range(3):
        var record = String(records[index])
        var fields = record.split("\t")
        assert_equal(String(fields[0]), names[index])
        var encoded = String(fields[1])
        var expected = List[UInt32]()
        for item in encoded.split(" "):
            var token = String(item)
            var pair = token.split("-")
            expected.append(_hex(String(pair[0])))
            expected.append(_hex(String(pair[len(pair) - 1])))
        assert_equal(_ranges(index), expected, names[index])


def test_numeric_perl_ranges_return_independent_mutable_lists() raises:
    for index in range(3):
        var original = _ranges(index)
        var changed = _ranges(index)
        changed[0] = 0
        changed.append(UInt32.MAX)
        assert_equal(_ranges(index), original)
