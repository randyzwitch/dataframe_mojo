"""Numeric folding tables preserve the frozen Polars 1.44.2 orbits."""
# contract-group: kernel.regex
from std.testing import assert_equal, assert_true
from dataframe.regex.charclass import CaseFolds
from .test_regex_numeric_ranges import _hex


def test_numeric_folding_tables_preserve_every_pair_and_orbit() raises:
    var fixture = open("tests/fixtures/regex/unicode_folding.tsv", "r").read()
    var keys = List[UInt32]()
    var next = List[UInt32]()
    for line in fixture.split("\n"):
        if line.byte_length() == 0:
            continue
        var record = String(line)
        var pair = record.split("\t")
        keys.append(_hex(String(pair[0])))
        next.append(_hex(String(pair[1])))
    var folds = CaseFolds()
    assert_equal(folds.keys, keys)
    assert_equal(folds.next, next)
    for i in range(len(keys)):
        var cursor = folds.next[i]
        var hops = 0
        while cursor != folds.keys[i]:
            var k = folds._lower_bound(cursor)
            assert_true(k < len(keys))
            assert_equal(folds.keys[k], cursor)
            cursor = folds.next[k]
            hops += 1
            assert_true(hops < len(keys))


def test_numeric_folding_tables_return_independent_mutable_lists() raises:
    var original = CaseFolds()
    var changed = CaseFolds()
    changed.keys[0] = UInt32.MAX
    changed.next[0] = UInt32.MAX
    changed.keys.append(UInt32.MAX)
    var fresh = CaseFolds()
    assert_equal(fresh.keys, original.keys)
    assert_equal(fresh.next, original.next)
