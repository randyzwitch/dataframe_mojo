"""Composite numbering preserves exact equality and first-occurrence ids."""
from std.testing import TestSuite, assert_equal, assert_raises
from dataframe import Column, DataType, Series, StringColumn
from dataframe.dtype import CategoricalDictionary
from dataframe.series import Storage
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


def categorical_codes(
    var values: List[UInt32], var valid: List[Bool], domain: Int
) raises -> Series:
    var dictionary = CategoricalDictionary()
    for i in range(domain):
        dictionary.append(String(i))
    return Series(
        "category",
        Storage(Column[UInt32](values^, valid^)),
        DataType.categorical(dictionary^),
    )


def test_composite_categorical_keys_match_strings_and_float_equality() raises:
    var a = Series(
        "a",
        StringColumn(
            ["b", "a", "b", "\x00", "", "a"],
            [True, False, True, True, True, True],
        ),
    )
    var b = Series("b", StringColumn(["é", "x", "é", "x", "", "x"]))
    var ac = a.cast(DataType.CATEGORICAL)
    var bc = b.cast(DataType.CATEGORICAL)
    ac = Series._from_chunks([ac.slice(0, 2), ac.slice(2, 4)])
    bc = Series._from_chunks([bc.slice(0, 4), bc.slice(4, 2)])
    var f = Series(
        "f",
        Column[Float64](
            [
                Float64(0) / Float64(0),
                0.0,
                Float64(0) / Float64(0),
                -0.0,
                1.5,
                0.0,
            ]
        ),
    )
    for reverse in [False, True]:
        for nulls_equal in [False, True]:
            var raw: List[Series] = [
                b.copy(),
                a.copy(),
                f.copy(),
            ] if reverse else [a.copy(), b.copy(), f.copy()]
            var encoded: List[Series] = [
                bc.copy(),
                ac.copy(),
                f.copy(),
            ] if reverse else [ac.copy(), bc.copy(), f.copy()]
            var expected = encode_rows(raw, nulls_equal)
            var actual = encode_rows(encoded, nulls_equal)
            assert_equal(actual.ids, expected.ids)
            assert_equal(actual.representatives, expected.representatives)


def test_categorical_column_codes_cover_chunks_views_and_unused_dictionary() raises:
    var source = categorical_codes(
        [0, 2, 1, 2, 3, 1, 0], [True, True, False, True, True, True, True], 4
    ).slice(1, 5)
    var chunked = Series._from_chunks([source.slice(0, 2), source.slice(2, 3)])
    for input in [source.copy(), chunked.copy()]:
        var codes = List[Int](length=19, fill=99)
        var nulls = List[Bool](length=5, fill=False)
        assert_equal(column_codes(input, codes, nulls), 3)
        assert_equal(codes, [0, -1, 0, 1, 2])
        assert_equal(nulls, [False, True, False, False, False])
    var codes = List[Int]()
    var nulls = List[Bool]()
    assert_equal(column_codes(source.slice(0, 0), codes, nulls), 0)
    assert_equal(len(codes), 0)
    var all_null = categorical_codes(
        [UInt32.MAX, UInt32.MAX], [False, False], 0
    )
    nulls = [False, False]
    assert_equal(column_codes(all_null, codes, nulls), 0)
    assert_equal(codes, [-1, -1])
    assert_equal(nulls, [True, True])


def test_categorical_lookup_domain_limit_preserves_generic_codes() raises:
    for domain in [4096, 4097]:
        var key = categorical_codes(
            [UInt32(domain - 1), 0, UInt32(domain - 1)],
            [True, True, True],
            domain,
        )
        var codes = List[Int]()
        var nulls = List[Bool]()
        assert_equal(column_codes(key, codes, nulls), 2)
        assert_equal(codes, [0, 1, 0])


def test_composite_categorical_lookup_checks_valid_codes() raises:
    var key = categorical_codes([0, UInt32.MAX], [True, True], 2)
    var codes = List[Int]()
    var nulls = List[Bool]()
    with assert_raises():
        _ = column_codes(key, codes, nulls)
    key = categorical_codes([0, UInt32.MAX], [True, False], 2)
    nulls = [False, False]
    assert_equal(column_codes(key, codes, nulls), 1)
    assert_equal(codes, [0, -1])


def test_narrow_integer_keys_take_the_direct_table() raises:
    # Int16, UInt8 and Int32 keys number by first occurrence like Int64
    # ones, with nulls, across the whole value range of a narrow type, and
    # through the chunked route (`column_codes` on a chunked series must
    # encode every chunk, not the first alone).
    var int16 = Series(
        "k",
        Column[Int16](
            [Int16.MAX, -5, Int16.MIN, -5, 0, Int16.MAX, 0],
            [True, True, True, False, True, True, True],
        ),
    )
    var codes = List[Int]()
    var nulls = List[Bool](length=7, fill=False)
    assert_equal(column_codes(int16, codes, nulls), 4)
    assert_equal(codes, [0, 1, 2, -1, 3, 0, 3])
    assert_equal(nulls, [False, False, False, True, False, False, False])
    var keys = encode_rows([int16.copy()], nulls_equal=True)
    assert_equal(keys.ids, [0, 1, 2, 3, 4, 0, 4])
    assert_equal(keys.representatives, [0, 1, 2, 3, 4])

    var bytes = Series("k", Column[UInt8]([255, 0, 255, 7, 7, 0]))
    codes = List[Int]()
    nulls = List[Bool](length=6, fill=False)
    assert_equal(column_codes(bytes, codes, nulls), 3)
    assert_equal(codes, [0, 1, 0, 2, 2, 1])

    # A span past the direct table's limit for 32-bit keys gives the same
    # numbering through the hash table.
    var wide = Series("k", Column[Int32]([100_000, -100_000, 100_000, 3]))
    codes = List[Int]()
    nulls = List[Bool](length=4, fill=False)
    assert_equal(column_codes(wide, codes, nulls), 3)
    assert_equal(codes, [0, 1, 0, 2])

    var chunked = Series._from_chunks([int16.slice(0, 3), int16.slice(3, 4)])
    codes = List[Int]()
    nulls = List[Bool](length=7, fill=False)
    assert_equal(column_codes(chunked, codes, nulls), 4)
    assert_equal(codes, [0, 1, 2, -1, 3, 0, 3])
    var chunked_keys = encode_rows([chunked.copy()], nulls_equal=False)
    assert_equal(chunked_keys.ids, [0, 1, 2, -1, 3, 0, 3])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
