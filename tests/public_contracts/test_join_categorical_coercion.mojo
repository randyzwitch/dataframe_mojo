"""Mixed categorical/String join keys match decoded text, never integer codes."""
# contract-group: table.join
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.join_typed_reference import nested_loop_pairs
from tests.support.coverage_compare import assert_value


def test_mixed_categorical_string_keys_share_value_identity_in_both_directions() raises:
    var left_raw = Series(
        "k",
        Column[String](
            ["z", "é", "z", "left", "hidden"], [True, True, True, True, False]
        ),
    )
    var right_raw = Series(
        "k",
        Column[String](
            ["é", "z", "z", "right", "hidden"], [True, True, True, True, False]
        ),
    )
    var lr: List[Int] = [0, 1, 0, 2, -1]
    var rr: List[Int] = [1, 0, 0, 3, -1]
    for category_left in [False, True]:
        var left = DataFrame(
            [
                left_raw.cast(
                    DataType.CATEGORICAL
                ) if category_left else left_raw.copy(),
                Series("p", Column[Int64]([0, 1, 2, 3, 4])),
            ]
        )
        var right = DataFrame(
            [
                right_raw.copy() if category_left else right_raw.cast(
                    DataType.CATEGORICAL
                ),
                Series("q", Column[Int64]([10, 11, 12, 13, 14])),
            ]
        )
        for layout in range(3):
            var l = (
                left.copy() if layout
                == 0 else offset_layout(left) if layout
                == 1 else chunk_layout(left)
            )
            var r = (
                right.copy() if layout
                == 0 else offset_layout(right) if layout
                == 1 else chunk_layout(right)
            )
            for how in ["inner", "left", "right", "full", "semi", "anti"]:
                var pairs = nested_loop_pairs(lr, rr, how)
                var actual = l.join(r, on="k", how=how)
                var names: List[String] = ["k", "p"]
                if how != "semi" and how != "anti":
                    names.append("q")
                assert_equal(actual.columns(), names)
                assert_equal(actual.height(), len(pairs))
                assert_true(actual.column("k").dtype().is_categorical())
                assert_true(actual.column("p").dtype() == DataType.INT64)
                if len(names) == 3:
                    assert_true(actual.column("q").dtype() == DataType.INT64)
                for output in range(len(pairs)):
                    var i = pairs[output][0]
                    var j = pairs[output][1]
                    var use_right = how == "right" or (how == "full" and i < 0)
                    assert_value(
                        actual.item(output, "k"),
                        right_raw.get(j) if use_right else left_raw.get(i),
                    )
                    assert_equal(actual.item(output, "p").is_null(), i < 0)
                    if i >= 0:
                        assert_equal(actual.item(output, "p").int64(), Int64(i))
                    if len(names) == 3:
                        assert_equal(actual.item(output, "q").is_null(), j < 0)
                        if j >= 0:
                            assert_equal(
                                actual.item(output, "q").int64(), Int64(10 + j)
                            )
