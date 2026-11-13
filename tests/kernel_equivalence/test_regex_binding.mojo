"""Literal and regex expression bindings share the standalone pattern cache."""
# contract-group: kernel.regex
from std.memory import Pointer
from std.testing import assert_equal, assert_true

from regex_mojo import Matcher, Regex
from dataframe.text.regex_adapter import compile_pattern


def _address(regex: Regex) -> Int:
    return Int(Pointer(to=regex._compiled[]))


def _matches(regex: Regex, text: String) -> Bool:
    var matcher = Matcher(regex)
    var span = Span[UInt8, ImmutAnyOrigin](
        unsafe_ptr=text.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](),
        length=text.byte_length(),
    )
    var matched = matcher.is_match(span)
    _ = text
    return matched


def test_cached_literal_and_regex_patterns_use_the_same_canonical_program() raises:
    var literal = compile_pattern("a.b", True)
    var escaped = compile_pattern("a\\.b", False)
    assert_equal(_address(literal), _address(escaped))
    assert_true(_matches(literal, "a.b"))
    assert_true(not _matches(literal, "axb"))
    var regex = compile_pattern("a.b", False)
    assert_true(_address(literal) != _address(regex))
    assert_true(_matches(regex, "axb"))
