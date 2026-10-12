"""Typed equality joins preserve complete rows, multiplicity and key identity."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, StructColumn
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.join_typed_reference import (
    typed_join_values,
    fixture_join_ranks,
    selected_ranks,
    nested_loop_pairs,
    assert_typed_join,
)


def _layout(frame: DataFrame, layout: Int) raises -> DataFrame:
    return (
        frame.copy() if layout
        == 0 else offset_layout(frame) if layout
        == 1 else chunk_layout(frame)
    )


def test_join_reference_has_literal_multiplicity_null_and_order_anchors() raises:
    assert_equal(fixture_join_ranks(DataType.UINT8), [0, 1, 2, 3, 0])
    assert_equal(fixture_join_ranks(DataType.FLOAT64), [0, 1, 2, 3, 0])
    assert_equal(fixture_join_ranks(DataType.BOOL), [0, 1, 1, 0, 1])
    var left: List[Int] = [1, -1, 1, 2]
    var right: List[Int] = [1, 3, 1, -1]
    assert_equal(
        nested_loop_pairs(left, right, "inner"),
        [(0, 0), (0, 2), (2, 0), (2, 2)],
    )
    assert_equal(
        nested_loop_pairs(left, right, "right"),
        [(0, 0), (2, 0), (-1, 1), (0, 2), (2, 2), (-1, 3)],
    )
    assert_equal(
        nested_loop_pairs(left, right, "full"),
        [(0, 0), (0, 2), (1, -1), (2, 0), (2, 2), (3, -1), (-1, 1), (-1, 3)],
    )
    assert_equal(nested_loop_pairs(left, right, "semi"), [(0, -1), (2, -1)])
    assert_equal(nested_loop_pairs(left, right, "anti"), [(1, -1), (3, -1)])


def test_extended_scalar_keys_match_independent_pairs_in_every_equality_join_kind() raises:
    var li: List[Int] = [1, 0, 1, 3, -1, 4]
    var ri: List[Int] = [3, 1, 1, 2, -1, 0]
    for base in typed_join_values():
        if base.dtype().is_nested():
            continue
        var ranks = fixture_join_ranks(base.dtype())
        var left = DataFrame(
            [
                base.take_or_null(li).renamed("k"),
                Series("p", Column[Int64]([10, 11, 12, 13, 14, 15])),
            ]
        )
        var right = DataFrame(
            [
                base.take_or_null(ri).renamed("k"),
                Series("p", Column[String](["a", "b", "c", "d", "e", "f"])),
            ]
        )
        for layout in range(3):
            var l = _layout(left, layout)
            var r = _layout(right, layout)
            for how in [
                "inner",
                "left",
                "right",
                "full",
                "semi",
                "anti",
                "cross",
            ]:
                var pairs = nested_loop_pairs(
                    selected_ranks(ranks, li), selected_ranks(ranks, ri), how
                )
                var actual = l.join(
                    r, how="cross"
                ) if how == "cross" else l.join(r, "k", how=how)
                assert_typed_join(actual, l, r, pairs, how)
                if how != "semi" and how != "anti" and how != "cross":
                    assert_typed_join(
                        l.join(r, "k", how=how, coalesce=False),
                        l,
                        r,
                        pairs,
                        how,
                        coalesce=False,
                    )


def test_join_payloads_preserve_nested_children_and_all_logical_metadata() raises:
    var li: List[Int] = [1, 0, 1, 3, -1, 4]
    var ri: List[Int] = [3, 1, 1, 2, -1, 0]
    var key = Series("k", Column[Int64]([0, 1, 2, 3, 4]))
    for payload in typed_join_values():
        var left = DataFrame(
            [key.take_or_null(li), payload.take_or_null(li).renamed("p")]
        )
        var right = DataFrame(
            [key.take_or_null(ri), payload.take_or_null(ri).renamed("p")]
        )
        for layout in range(3):
            for empty in range(4):
                var l = _layout(left, layout)
                var r = _layout(right, layout)
                var lr = li.copy()
                var rr = ri.copy()
                if empty == 1 or empty == 3:
                    l = l.clear()
                    lr = List[Int]()
                if empty == 2 or empty == 3:
                    r = r.clear()
                    rr = List[Int]()
                for how in [
                    "inner",
                    "left",
                    "right",
                    "full",
                    "semi",
                    "anti",
                    "cross",
                ]:
                    var pairs = nested_loop_pairs(lr, rr, how)
                    var actual = l.join(
                        r, how="cross"
                    ) if how == "cross" else l.join(r, "k", how=how)
                    assert_typed_join(actual, l, r, pairs, how)


def test_flat_struct_keys_obey_scalar_membership_and_reject_unsupported_join_kinds() raises:
    var li: List[Int] = [1, 0, 1, 3, -1, 4]
    var ri: List[Int] = [3, 1, 1, 2, -1, 0]
    var struct_key = Series(
        "k",
        StructColumn(
            [
                Series("n", Column[Int64]([0, 1, 2, 3, 4])),
                Series(
                    "s",
                    Column[String](["", "é", "😀abcdefghijklmnop", "z", "q"]),
                ),
            ]
        ),
    )
    var left = DataFrame(
        [
            struct_key.take_or_null(li),
            Series("p", Column[Int64]([0, 1, 2, 3, 4, 5])),
        ]
    )
    var right = DataFrame(
        [
            struct_key.take_or_null(ri),
            Series("p", Column[Int64]([10, 11, 12, 13, 14, 15])),
        ]
    )
    for layout in range(3):
        var l = _layout(left, layout)
        var r = _layout(right, layout)
        for how in ["inner", "left", "semi", "anti"]:
            assert_typed_join(
                l.join(r, "k", how=how),
                l,
                r,
                nested_loop_pairs(li, ri, how),
                how,
            )
        # docs/expressions.md explicitly excludes right/full struct-key joins.
        for how in ["right", "full"]:
            with assert_raises():
                _ = l.join(r, "k", how=how)


def test_categorical_join_keys_match_values_across_independent_dictionaries() raises:
    var left_key = Series(
        "k",
        Column[String](
            ["z", "a", "z", "left only", "hidden left"],
            [True, True, True, True, False],
        ),
    ).cast(DataType.CATEGORICAL)
    var right_key = Series(
        "k",
        Column[String](
            ["a", "z", "z", "right only", "hidden right"],
            [True, True, True, True, False],
        ),
    ).cast(DataType.CATEGORICAL)
    assert_true(left_key.dtype() != right_key.dtype())
    # The first dictionary entry/code is z on the left and a on the right;
    # equal code0 therefore cannot stand for a join match.
    assert_equal(String(left_key.dtype().dictionary()[].get(0)), "z")
    assert_equal(String(right_key.dtype().dictionary()[].get(0)), "a")
    var left = DataFrame(
        [left_key^, Series("p", Column[UInt64]([0, 1, 2, 3, UInt64.MAX]))]
    )
    var right = DataFrame(
        [right_key^, Series("p", Column[UInt64]([10, 11, 12, 13, UInt64.MAX]))]
    )
    var lr: List[Int] = [0, 1, 0, 2, -1]
    var rr: List[Int] = [1, 0, 0, 3, -1]
    for layout in range(3):
        var l = _layout(left, layout)
        var r = _layout(right, layout)
        for how in ["inner", "left", "right", "full", "semi", "anti", "cross"]:
            var pairs = nested_loop_pairs(lr, rr, how)
            var actual = l.join(r, how="cross") if how == "cross" else l.join(
                r, "k", how=how
            )
            assert_typed_join(actual, l, r, pairs, how, dictionary_union=True)
            if how != "semi" and how != "anti" and how != "cross":
                assert_typed_join(
                    l.join(r, "k", how=how, coalesce=False),
                    l,
                    r,
                    pairs,
                    how,
                    coalesce=False,
                    dictionary_union=True,
                )
