"""Order-preserving key encoding, checked against the values themselves.

A wrong encoding still produces a sorted-looking result, so these compare
the encoded order of every pair of rows against the order the values say,
rather than eyeballing an output. The cases are the ones where a plausible
encoding is silently wrong: the float sign bit, -0.0 against 0.0, NaN under
`descending`, nulls under both placements, UInt64 above 2**63, and
Int64.MIN under `descending`.
"""
from std.testing import TestSuite, assert_equal, assert_true, assert_raises

from dataframe import Column, DataFrame, DataType, Series
from dataframe.row_encode import (
    STRING_PREFIX_BYTES,
    encodable,
    encode_sort_keys,
)


def order_of(
    var column: Series, descending: Bool = False, nulls_last: Bool = True
) raises -> List[Int]:
    """Rank of each row under the encoding, by comparing encoded words."""
    var columns = List[Series]()
    columns.append(column^)
    var words = encode_sort_keys(
        columns,
        List[Bool](length=1, fill=descending),
        List[Bool](length=1, fill=nulls_last),
    )
    var rows = len(words[0])
    var rank = List[Int](length=rows, fill=0)
    for i in range(rows):
        var before = 0
        for j in range(rows):
            if i == j:
                continue
            # Lexicographic across words, row index last, as sort_indices
            # compares them.
            var decided = False
            for w in range(len(words)):
                if words[w][j] != words[w][i]:
                    if words[w][j] < words[w][i]:
                        before += 1
                    decided = True
                    break
            if not decided and j < i:
                before += 1
        rank[i] = before
    return rank^


def test_floats_order_by_sign_and_magnitude() raises:
    # Without the sign-bit toggle after the total-order map, every positive
    # sorts below every negative -- the failure mode that put 900,000 rows
    # in the wrong place when this was first prototyped.
    var values: List[Float64] = [1.0, -1.0, 0.0, -2.5, 2.5, -1e308, 1e308]
    var rank = order_of(Series("f", Column[Float64](values.copy())))
    # Expected ascending positions.
    var expected: List[Int] = [4, 2, 3, 1, 5, 0, 6]
    for i in range(len(values)):
        assert_equal(rank[i], expected[i], "value " + String(values[i]))


def test_negative_zero_equals_zero() raises:
    var values: List[Float64] = [0.0, -0.0, 0.0]
    var rank = order_of(Series("f", Column[Float64](values^)))
    # All three tie, so their order is the row order: 0, 1, 2.
    assert_equal(rank[0], 0)
    assert_equal(rank[1], 1)
    assert_equal(rank[2], 2)


def test_nan_sorts_after_numbers_in_both_directions() raises:
    var nan = Float64(0) / Float64(0)
    var values: List[Float64] = [nan, 1.0, -1.0]
    var up = order_of(Series("f", Column[Float64](values.copy())))
    assert_equal(up[0], 2, "NaN last ascending")
    var down = order_of(Series("f", Column[Float64](values^)), descending=True)
    # Descending reverses the numbers but NaN stays after them, which the
    # IEEE total order alone would not do.
    assert_equal(down[0], 2, "NaN still last descending")
    assert_equal(down[1], 0, "1.0 first descending")
    assert_equal(down[2], 1, "-1.0 second descending")


def test_nulls_take_either_end_in_either_direction() raises:
    var values: List[Int64] = [5, 0, 7]
    var valid: List[Bool] = [True, False, True]
    for descending in [False, True]:
        var last = order_of(
            Series("i", Column[Int64](values.copy(), valid.copy())),
            descending=descending,
            nulls_last=True,
        )
        assert_equal(last[1], 2, "null last")
        var first = order_of(
            Series("i", Column[Int64](values.copy(), valid.copy())),
            descending=descending,
            nulls_last=False,
        )
        assert_equal(first[1], 0, "null first")


def test_uint64_above_two_to_the_63() raises:
    # Encoded as a signed Int, these are negative; without the top-bit
    # toggle they would sort below every small value.
    var values: List[UInt64] = [1, 9223372036854775808, 0, 18446744073709551615]
    var rank = order_of(Series("u", Column[UInt64](values^)))
    var expected: List[Int] = [1, 2, 0, 3]
    for i in range(4):
        assert_equal(rank[i], expected[i], "index " + String(i))


def test_int64_extremes_under_descending() raises:
    # Descending inverts the bits rather than negating, because negating
    # Int64.MIN overflows.
    var values: List[Int64] = [
        -9223372036854775808,
        0,
        9223372036854775807,
    ]
    var up = order_of(Series("i", Column[Int64](values.copy())))
    assert_equal(up[0], 0)
    assert_equal(up[2], 2)
    var down = order_of(Series("i", Column[Int64](values^)), descending=True)
    assert_equal(down[0], 2, "MIN last descending")
    assert_equal(down[2], 0, "MAX first descending")


def test_booleans_and_narrow_integers() raises:
    var flags: List[Bool] = [True, False, True]
    var rank = order_of(Series("b", Column[Bool](flags^)))
    assert_equal(rank[1], 0, "false before true")
    var small: List[Int8] = [-128, 0, 127]
    var srank = order_of(Series("s", Column[Int8](small^)))
    assert_equal(srank[0], 0)
    assert_equal(srank[2], 2)


def test_several_keys_compare_lexicographically() raises:
    var a: List[Int64] = [1, 1, 0]
    var b: List[Int64] = [9, 2, 5]
    var columns = List[Series]()
    columns.append(Series("a", Column[Int64](a^)))
    columns.append(Series("b", Column[Int64](b^)))
    var words = encode_sort_keys(
        columns,
        List[Bool](length=2, fill=False),
        List[Bool](length=2, fill=True),
    )
    # Two keys, neither nullable nor float, so one word each.
    assert_equal(len(words), 2)
    # Row 2 has the smallest first key, so it wins regardless of the second.
    assert_true(words[0][2] < words[0][0])
    # Rows 0 and 1 tie on the first key and are split by the second.
    assert_equal(words[0][0], words[0][1])
    assert_true(words[1][1] < words[1][0])


def test_strings_order_bytewise() raises:
    var values: List[String] = ["b", "a", "c", "", "ab"]
    var rank = order_of(Series("s", Column[String](values^)))
    # "" < "a" < "ab" < "b" < "c"
    var expected: List[Int] = [3, 1, 4, 0, 2]
    for i in range(5):
        assert_equal(rank[i], expected[i], "index " + String(i))


def test_a_string_that_is_a_prefix_of_another() raises:
    # Zero padding puts the shorter first, which is the byte order; the
    # length word is what keeps that exact.
    var values: List[String] = ["abc", "ab", "abcd", "a"]
    var rank = order_of(Series("s", Column[String](values^)))
    var expected: List[Int] = [2, 1, 3, 0]
    for i in range(4):
        assert_equal(rank[i], expected[i], "index " + String(i))


def test_embedded_nul_bytes_are_not_padding() raises:
    """A NUL inside a value must not read as the end of a shorter one.

    Padding alone cannot tell "ab" from "ab\0" -- both pad to the same
    bytes -- so the length word decides, and "ab" must come first.
    """
    var with_nul = String("ab") + chr(0)
    var values: List[String] = [with_nul, "ab", "ab" + chr(0) + chr(0)]
    var rank = order_of(Series("s", Column[String](values^)))
    assert_equal(rank[1], 0, "ab first")
    assert_equal(rank[0], 1, "ab NUL second")
    assert_equal(rank[2], 2, "ab NUL NUL last")


def test_multibyte_utf8_orders_by_code_point() raises:
    # UTF-8 byte order is code point order, so no special handling.
    var values: List[String] = ["é", "a", "z", "€"]
    var rank = order_of(Series("s", Column[String](values^)))
    var expected: List[Int] = [2, 0, 1, 3]
    for i in range(4):
        assert_equal(rank[i], expected[i], "index " + String(i))


def test_strings_descending_and_null_placement() raises:
    var values: List[String] = ["b", "a", ""]
    var valid: List[Bool] = [True, False, True]
    var down = order_of(
        Series("s", Column[String](values.copy(), valid.copy())),
        descending=True,
        nulls_last=True,
    )
    assert_equal(down[0], 0, "b first descending")
    assert_equal(down[2], 1, "empty second descending")
    assert_equal(down[1], 2, "null last")
    var first = order_of(
        Series("s", Column[String](values^, valid^)),
        descending=True,
        nulls_last=False,
    )
    assert_equal(first[1], 0, "null first")


def test_long_strings_fall_back_to_ranking() raises:
    # Exactly the prefix is encodable; one byte more is not, because the
    # encoding would silently compare only the prefix.
    var fits = String()
    for _ in range(STRING_PREFIX_BYTES):
        fits += "x"
    assert_true(encodable(Series("s", Column[String]([fits.copy()]))))
    assert_true(encodable(Series("i", Column[Int64]([1]))))
    assert_true(encodable(Series("f", Column[Float64]([1.0]))))
    assert_true(encodable(Series("b", Column[Bool]([True]))))
    var over = fits + "x"
    assert_true(
        not encodable(Series("s", Column[String]([over^]))),
        "a value past the prefix must not claim to be encodable",
    )


def test_a_long_string_still_sorts_correctly() raises:
    """The fallback has to produce the same order, or the cap is a bug.

    These two differ only past the prefix, so an encoding that ignored the
    cap would call them equal and leave them in row order.
    """
    var base = String()
    for _ in range(STRING_PREFIX_BYTES):
        base += "x"
    var frame = DataFrame(
        [
            Series("s", Column[String]([base + "b", base + "a"])),
            Series("r", Column[Int64]([0, 1])),
        ]
    )
    var order = frame.arg_sort(
        ["s"],
        descending=List[Bool](length=1, fill=False),
        nulls_last=List[Bool](length=1, fill=True),
    )
    assert_equal(order[0], 1, "the smaller tail must sort first")
    assert_equal(order[1], 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
