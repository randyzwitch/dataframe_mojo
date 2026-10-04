"""Composite numbering preserves exact equality and first-occurrence ids."""
from std.testing import TestSuite, assert_equal
from dataframe import Column, Series, StringColumn
from dataframe.hashing import encode_rows, column_codes


def test_composite_nulls_and_string_storage() raises:
    var a = Series(
        "a",
        Column[Int64](
            [7, 2, 7, 2, 7, 2, 7, 2],
            [True, False, True, True, True, False, True, True],
        ),
    )
    var b = Series(
        "b",
        StringColumn(
            ["x", "y", "x", "y", "", "", "\x00", "\x00"],
            [True, True, True, True, False, False, True, True],
        ),
    )
    for reverse in [False, True]:
        var keys: List[Series] = [b.copy(), a.copy()] if reverse else [
            a.copy(),
            b.copy(),
        ]
        var included = encode_rows(keys, nulls_equal=True)
        assert_equal(included.ids, [0, 1, 0, 2, 3, 4, 5, 6])
        assert_equal(included.representatives, [0, 1, 3, 4, 5, 6, 7])
        var excluded = encode_rows(keys, nulls_equal=False)
        assert_equal(excluded.ids, [0, -1, 0, 1, -1, -1, 2, 3])
        assert_equal(excluded.representatives, [0, 3, 6, 7])


def test_integer_domain_boundaries_and_extremes() raises:
    for start in [Int64.MIN, Int64(-4000), Int64.MAX - 4096]:
        for span in [4095, 4096]:
            var a = Series(
                "a",
                Column[Int64](
                    [start + Int64(span), start, start + Int64(span), start]
                ),
            )
            var b = Series("b", Column[Int64]([5, 3, 5, 3]))
            var groups = encode_rows([a^, b^], nulls_equal=True)
            assert_equal(groups.ids, [0, 1, 0, 1])
            assert_equal(groups.representatives, [0, 1])
    var extremes = Series("a", Column[Int64]([Int64.MIN, Int64.MAX, Int64.MIN]))
    var groups = encode_rows(
        [extremes^, Series("b", Column[Int64]([1, 1, 1]))], nulls_equal=True
    )
    assert_equal(groups.ids, [0, 1, 0])


def test_codes_replace_existing_buffer_and_preserve_nulls() raises:
    var sources = [
        Series("i", Column[Int64]([8, 0, 8, 3], [True, False, True, True])),
        Series(
            "s",
            StringColumn(
                [
                    "long value outside inline storage",
                    "",
                    "long value outside inline storage",
                    "é\x00",
                ],
                [True, False, True, True],
            ),
        ),
    ]
    for source in sources:
        var codes = List[Int](length=17, fill=99)
        var nulls = List[Bool](length=4, fill=False)
        assert_equal(column_codes(source, codes, nulls), 2)
        assert_equal(codes[0], 0)
        assert_equal(codes[2], 0)
        assert_equal(codes[3], 1)
        assert_equal(len(codes), 4)
        assert_equal(nulls, [False, True, False, False])


def test_all_null_empty_and_sliced_chunked_keys() raises:
    var nulls = Series("a", Column[Int64]([0, 0, 0], [False, False, False]))
    var b = Series("b", StringColumn(["x", "x", "x"]))
    var included = encode_rows([nulls.copy(), b.copy()], nulls_equal=True)
    assert_equal(included.ids, [0, 0, 0])
    var excluded = encode_rows([nulls.copy(), b.copy()], nulls_equal=False)
    assert_equal(excluded.ids, [-1, -1, -1])
    assert_equal(excluded.count(), 0)
    var empty = encode_rows(
        [nulls.slice(0, 0), b.slice(0, 0)], nulls_equal=True
    )
    assert_equal(empty.count(), 0)
    var a = Series._from_chunks(
        [
            Series("a", Column[Int64]([99, 8, 3])),
            Series("a", Column[Int64]([8, 3, 77])),
        ]
    ).slice(1, 4)
    var s = Series._from_chunks(
        [
            Series("s", StringColumn(["ignore", "x"])),
            Series("s", StringColumn(["y", "x", "y", "ignore"])),
        ]
    ).slice(1, 4)
    var groups = encode_rows([a^, s^], nulls_equal=True)
    assert_equal(groups.ids, [0, 1, 0, 1])
    assert_equal(groups.representatives, [0, 1])


def test_boolean_first_key_requires_first_occurrence_renumbering() raises:
    for values in [
        [True, True, True],
        [False, False, False],
        [True, False, True],
    ]:
        var key = Series("b", Column[Bool](List[Bool](values)))
        var expected: List[Int] = [0, 1, 0] if values[0] != values[1] else [
            0,
            0,
            0,
        ]
        var single = encode_rows([key.copy()], nulls_equal=True)
        assert_equal(single.ids, expected)
        var composite = encode_rows(
            [key.copy(), Series("s", StringColumn(["same", "same", "same"]))],
            nulls_equal=True,
        )
        assert_equal(composite.ids, expected)


def test_unsigned_first_key_keeps_full_width_and_order() raises:
    var key = Series("u", Column[UInt64]([UInt64.MAX, 0, UInt64.MAX, 1]))
    var single = encode_rows([key.copy()], nulls_equal=True)
    assert_equal(single.ids, [0, 1, 0, 2])
    assert_equal(single.representatives, [0, 1, 3])
    var composite = encode_rows(
        [key.copy(), Series("s", StringColumn(["x", "x", "x", "x"]))],
        nulls_equal=True,
    )
    assert_equal(composite.ids, single.ids)
    assert_equal(composite.representatives, single.representatives)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
