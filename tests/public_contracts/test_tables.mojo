"""Public behavior: shape, schema, order, nulls, and independent row expectations."""
# contract-group: table.schema
# contract-group: table.row_selection
# contract-group: table.filter
# contract-group: table.sort
# contract-group: harness.sensitivity
# contract-group: reference.anchors
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, Series, col
from tests.support.coverage_generators import generated_frame
from tests.support.reference_model import (
    generated_model,
    ReferenceRow,
    ReferenceTable,
)
from tests.support.coverage_compare import assert_frame, assert_model


def test_projection_and_filter_obey_row_contract() raises:
    for n in [0, 1, 7, 8, 9, 63, 64, 65]:
        for null_mode in range(3):
            var frame = generated_frame(n, 3, null_mode)
            var expected = (
                generated_model(n, 3, null_mode).filter_i_gt(0).project(["i"])
            )
            assert_model(frame.filter(col("i") > 0).select(["i"]), expected)
            assert_equal(frame.columns(), ["i", "f", "s", "b"])


def test_comparator_rejects_structural_and_cell_mutations() raises:
    var expected = DataFrame(
        [
            Series("i", Column[Int64]([1, 2])),
            Series("s", Column[String](["a", "b"])),
        ]
    )
    var swapped = DataFrame(
        [
            Series("i", Column[Int64]([2, 1])),
            Series("s", Column[String](["a", "b"])),
        ]
    )
    with assert_raises():
        assert_frame(swapped, expected)
    with assert_raises():
        assert_frame(expected.select(["s", "i"]), expected)
    with assert_raises():
        assert_frame(expected.rename({"i": "other"}), expected)
    with assert_raises():
        assert_frame(expected.slice(0, 1), expected)
    var nullable = DataFrame(
        [
            Series("i", Column[Int64]([1, 2], [True, False])),
            Series("s", Column[String](["a", "b"])),
        ]
    )
    with assert_raises():
        assert_frame(nullable, expected)
    var wrong_dtype = DataFrame(
        [
            Series("i", Column[Float64]([1, 2])),
            Series("s", Column[String](["a", "b"])),
        ]
    )
    with assert_raises():
        assert_frame(wrong_dtype, expected)
    var positive = DataFrame([Series("f", Column[Float64]([0.0]))])
    var negative = DataFrame([Series("f", Column[Float64]([-0.0]))])
    with assert_raises():
        assert_frame(positive, negative)
    var infinite = DataFrame(
        [Series("f", Column[Float64]([Float64(1) / Float64(0)]))]
    )
    with assert_raises():
        assert_frame(positive, infinite)

    var nan_frame = DataFrame(
        [Series("f", Column[Float64]([Float64(0) / Float64(0)]))]
    )
    assert_frame(nan_frame, nan_frame)
    assert_frame(infinite, infinite)
    with assert_raises():
        assert_frame(positive, nan_frame)
    var opposite = DataFrame(
        [Series("f", Column[Float64]([-Float64(1) / Float64(0)]))]
    )
    with assert_raises():
        assert_frame(infinite, opposite)
    var repeated = DataFrame(
        [
            Series("i", Column[Int64]([1, 1])),
            Series("s", Column[String](["a", "b"])),
        ]
    )
    with assert_raises():
        assert_frame(repeated, expected)
    var replaced_text = DataFrame(
        [
            Series("i", Column[Int64]([1, 2])),
            Series("s", Column[String](["x", "y"])),
        ]
    )
    with assert_raises():
        assert_frame(replaced_text, expected)


def test_reference_semantics_have_hand_computed_anchors() raises:
    var rows: List[ReferenceRow] = [
        ReferenceRow(2, 0, "first", True, True, True, True, True),
        ReferenceRow(99, 0, "null", False, False, True, True, True),
        ReferenceRow(-1, 0, "negative", False, True, True, True, True),
        ReferenceRow(2, 0, "second", True, True, True, True, True),
    ]
    var model = ReferenceTable(["i", "f", "s", "b"], rows^)
    assert_equal(model.double_i().rows[0].i, Int64(4))
    assert_equal(model.double_i().rows[2].i, Int64(-2))
    assert_equal(model.double_i().rows[1].iv, False)
    assert_equal(model.int_sum(), Int128(3))
    assert_equal(model.int_count(), Int64(3))
    assert_equal(len(model.filter_i_gt(0).rows), 2)
    assert_equal(model.filter_i_gt(0).rows[1].s, "second")
    var sorted = model.sort_i()
    assert_equal(sorted.rows[0].i, Int64(-1))
    assert_equal(sorted.rows[1].s, "first")
    assert_equal(sorted.rows[2].s, "second")
    assert_equal(sorted.rows[3].s, "null")
    assert_equal(model.slice(-2, 1).rows[0].s, "negative")
    assert_equal(len(model.project([]).rows), 4)
    with assert_raises():
        _ = model.project(["missing"])
    with assert_raises():
        _ = model.take([-1])


def test_take_slice_sort_and_empty_projection_match_reference() raises:
    for n in [0, 1, 7, 8, 9, 63, 64, 65]:
        for mode in range(3):
            var model = generated_model(n, 3, mode)
            var frame = generated_frame(n, 3, mode)
            assert_model(frame.sort("i"), model.sort_i())
            assert_model(frame.select(List[String]()), model.project([]))
            for start in [-100, -1, 0, 1, 100]:
                assert_model(frame.slice(start, 7), model.slice(start, 7))
            if n:
                var indices: List[Int] = [n - 1, 0, n - 1]
                assert_model(frame.take(indices), model.take(indices))
