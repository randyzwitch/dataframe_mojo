"""Full join rows and schema against a nested-loop oracle; duplicates are pairs."""
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, Series
from tests.support.reference_model import (
    generated_model,
    ReferenceRow,
    ReferenceTable,
)
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import assert_join_model


def test_all_equality_join_kinds_match_nested_loops() raises:
    var shapes: List[Tuple[Int, Int]] = [
        (0, 0),
        (0, 3),
        (3, 0),
        (1, 1),
        (9, 7),
        (35, 33),
    ]
    var key_sets: List[List[String]] = [["i"], ["s"], ["f"], ["b"], ["s", "i"]]
    for shape in shapes:
        for validity in range(3):
            var left_model = generated_model(shape[0], 3, validity)
            var right_model = generated_model(shape[1], 5, validity)
            for layout in range(3):
                var left = frame_from_model(left_model, 7 if layout == 1 else 0)
                var right = frame_from_model(
                    right_model, 1 if layout == 1 else 0
                )
                if layout == 2:
                    left = chunk_layout(left)
                    right = chunk_layout(right)
                assert_join_model(
                    left.join(right, how="cross"),
                    left_model,
                    right_model,
                    [],
                    "cross",
                )
                for keys in key_sets:
                    for how in [
                        "inner",
                        "left",
                        "right",
                        "full",
                        "semi",
                        "anti",
                    ]:
                        var result = left.join(right, keys, how=how)
                        assert_join_model(
                            result, left_model, right_model, keys, how
                        )
                        if how == "full":
                            assert_join_model(
                                left.join(right, keys, how=how, coalesce=False),
                                left_model,
                                right_model,
                                keys,
                                how,
                                coalesce=False,
                            )


def test_join_float_specials_and_validation() raises:
    var rows = List[ReferenceRow]()
    var values: List[Float64] = [
        0.0,
        -0.0,
        Float64(0) / Float64(0),
        Float64(0) / Float64(0),
        9.0,
    ]
    for index in range(len(values)):
        rows.append(
            ReferenceRow(
                Int64(index),
                values[index],
                String(index),
                False,
                True,
                index != 4,
                True,
                True,
            )
        )
    var model = ReferenceTable(["i", "f", "s", "b"], rows^)
    var frame = frame_from_model(model, 7)
    for how in ["inner", "left", "right", "full", "semi", "anti"]:
        assert_join_model(
            frame.join(frame, "f", how=how), model, model, ["f"], how
        )
    var invalid_keys: List[List[String]] = [[], ["missing"], ["i", "i"]]
    for keys in invalid_keys:
        with assert_raises():
            _ = frame.join(frame, keys)
    with assert_raises():
        _ = frame.join(frame, "i", how="unknown")
    with assert_raises():
        _ = frame.join(frame, left_on=["i"], right_on=["s"])
    with assert_raises():
        _ = frame.join(frame, left_on=["i", "s"], right_on=["i"])
    with assert_raises():
        _ = frame.join(frame, "i", suffix="")
    var right = frame.rename({"i": "key"})
    var different = frame.join(right, left_on=["i"], right_on=["key"])
    var same = frame.join(frame, "i")
    assert_equal(different.columns(), same.columns())
    # Full expected data comes from the oracle, rather than the sibling join.
    assert_join_model(different, model, model, ["i"], "inner")
