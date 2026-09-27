"""Physical build selection must preserve logical row ordering and nulls."""
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import Column, DataFrame, Series
from dataframe.frame import _smaller_build_join_rows
from dataframe.join_hash import prefer_left_build, prepare_progression_index


def test_smaller_build_restores_duplicate_order_and_unmatched() raises:
    var left = DataFrame(
        [
            Series(
                "k",
                Column[Int64]([3, 1, 3, 9, 0], [True, True, True, True, False]),
            ),
            Series("l", Column[Int64]([0, 1, 2, 3, 4])),
        ]
    )
    var keys = List[Int64]()
    var values = List[Int64]()
    var valid = List[Bool]()
    for i in range(524288):
        keys.append(Int64(100 + i))
        values.append(Int64(i))
        valid.append(i != 42)
    keys[2] = 1
    keys[7] = 3
    keys[11] = 1
    keys[40] = 3
    keys[42] = 3
    var right = DataFrame(
        [
            Series("k", Column[Int64](keys^, valid^)),
            Series("r", Column[Int64](values^)),
        ]
    )
    assert_true(prefer_left_build(left.height(), right.height()))
    assert_true(
        "[small left input; materialize unless compact progression]"
        in left.lazy().join(right.lazy(), "k").explain()
    )
    for how in ["inner", "left"]:
        var actual = left.join(right, "k", how=how)
        var expected_left: List[Int64] = [0, 0, 1, 1, 2, 2]
        var expected_right: List[Int64] = [7, 40, 2, 11, 7, 40]
        if how == "left":
            expected_left.append(3)
            expected_left.append(4)
        assert_equal(actual.height(), len(expected_left))
        for i in range(len(expected_left)):
            assert_equal(actual.item(i, "l").int64(), expected_left[i])
            if i < len(expected_right):
                assert_equal(actual.item(i, "r").int64(), expected_right[i])
            else:
                assert_true(actual.item(i, "r").is_null())
        for size in [1, 3, 65536]:
            assert_true(
                left.lazy()
                .join(right.lazy(), "k", how=how)
                .collect(batch_size=size)
                .equals(actual)
            )


def test_smaller_build_compound_float_string_equality() raises:
    var nan = Float64(0) / Float64(0)
    var left: List[Series] = [
        Series("f", Column[Float64]([nan, -0.0, 0.0, 8])),
        Series("s", Column[String](["a", "b", "a", "missing"])),
    ]
    var right: List[Series] = [
        Series(
            "f",
            Column[Float64](
                [0.0, nan, nan, -0.0, 0], [True, True, True, True, False]
            ),
        ),
        Series("s", Column[String](["a", "a", "a", "b", "missing"])),
    ]
    var pairs = _smaller_build_join_rows(left, right, True)
    assert_equal(pairs[0], [0, 0, 1, 2, 3])
    assert_equal(pairs[1], [1, 2, 3, 0, -1])


def test_build_choice_capacity_and_unknown_heights() raises:
    assert_true(not prefer_left_build(0, 100))
    assert_true(not prefer_left_build(-1, 100))
    assert_true(not prefer_left_build(1, -1))
    assert_true(not prefer_left_build(Int(Int32.MAX) + 1, Int.MAX))
    assert_true(not prefer_left_build(10, 10))
    assert_true(not prefer_left_build(4096, 262144))
    assert_true(prefer_left_build(4096, 524288))


def test_build_selection_keeps_compact_progressions() raises:
    var values = List[Int64]()
    for i in range(524288):
        values.append(Int64(i))
    var right = DataFrame([Series("k", Column[Int64](values^))])
    var left = DataFrame([Series("k", Column[Int64]([7, 2, -1]))])
    var keys: List[Series] = [right["k"].copy()]
    var prepared = prepare_progression_index(keys)
    assert_true(Bool(prepared))
    assert_true(prepared.value().progression)
    for how in ["inner", "left"]:
        assert_true(
            left.lazy()
            .join(right.lazy(), "k", how=how)
            .collect()
            .equals(left.join(right, "k", how=how))
        )
    var irregular: List[Series] = [Series("k", Column[Int64]([2, 1, 3]))]
    assert_true(not prepare_progression_index(irregular))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
