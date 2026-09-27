"""Typed probes retain exact string bytes, nulls, slices and generic fallback."""
from std.testing import TestSuite, assert_equal
from dataframe import Column, Series, StringColumn
from dataframe.string_view import StringViewBuilder
from dataframe.join_hash import direct_hash_semi_anti_rows, count_inner_join


def strings(
    values: List[String], valid: List[Bool], view: Bool
) raises -> Series:
    if not view:
        return Series("k", Column[String](values.copy(), valid.copy()))
    var builder = StringViewBuilder(len(values))
    for i in range(len(values)):
        if valid[i]:
            builder.append(StringSlice(values[i]))
        else:
            builder.append_null()
    return Series("k", StringColumn(builder^.finish()))


def test_typed_strings_preserve_views_slices_bytes_and_nulls() raises:
    for left_view in [False, True]:
        for right_view in [False, True]:
            var left: List[Series] = [
                strings(
                    [
                        "discard",
                        "",
                        "null",
                        "same-long-prefix-A",
                        "same-long-prefix-B",
                        "é",
                        "é",
                        "a\x00b",
                        "absent",
                    ],
                    [True, True, False, True, True, True, True, True, True],
                    left_view,
                ).slice(1, 8)
            ]
            var right: List[Series] = [
                strings(
                    [
                        "discard",
                        "",
                        "same-long-prefix-A",
                        "same-long-prefix-A",
                        "é",
                        "a\x00b",
                        "null",
                    ],
                    [True, True, True, True, True, True, False],
                    right_view,
                ).slice(1, 6)
            ]
            assert_equal(
                direct_hash_semi_anti_rows(left, right, True), [0, 2, 4, 6]
            )
            assert_equal(
                direct_hash_semi_anti_rows(left, right, False), [1, 3, 5, 7]
            )
            assert_equal(count_inner_join(left, right), 5)
            assert_equal(count_inner_join(right, left), 5)


def test_typed_int64_membership_preserves_signed_bits_and_nulls() raises:
    var left: List[Series] = [
        Series(
            "k",
            Column[Int64](
                [8, Int64.MIN, Int64.MAX, 0, -1, 0],
                [True, True, True, True, True, False],
            ),
        ).slice(1, 5)
    ]
    var right: List[Series] = [
        Series(
            "k",
            Column[Int64](
                [Int64.MAX, Int64.MIN, Int64.MIN, 0], [True, True, True, False]
            ),
        )
    ]
    assert_equal(direct_hash_semi_anti_rows(left, right, True), [0, 1])
    assert_equal(direct_hash_semi_anti_rows(left, right, False), [2, 3, 4])
    assert_equal(count_inner_join(left, right), 3)


def test_compound_keys_retain_generic_equality() raises:
    var nan = Float64(0) / Float64(0)
    var left: List[Series] = [
        Series(
            "f", Column[Float64]([nan, -0.0, 5, 0], [True, True, True, False])
        ),
        Series("s", Column[String](["a", "b", "c", "a"])),
    ]
    var right: List[Series] = [
        Series("f", Column[Float64]([nan, 0.0, nan])),
        Series("s", Column[String](["a", "b", "z"])),
    ]
    assert_equal(direct_hash_semi_anti_rows(left, right, True), [0, 1])
    assert_equal(direct_hash_semi_anti_rows(left, right, False), [2, 3])
    assert_equal(count_inner_join(left, right), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
