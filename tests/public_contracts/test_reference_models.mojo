"""Literal anchors for scalar models; no engine participates in expected answers."""
# contract-group: reference.anchors
from std.testing import assert_equal, assert_raises
from tests.support.reference_statistics import (
    integer_quantile,
    integer_variance,
)
from tests.support.reference_model import (
    ReferenceRow,
    ReferenceTable,
    reference_groups,
    reference_join,
    sorted_model,
)
from tests.support.reference_relations import (
    distinct_indices,
    nonnull_indices,
    concatenated_model,
    pivot_model,
)
from tests.support.numeric_model import (
    integer_bounds,
    integer_fits,
    truncated_integer,
)


def test_relational_reference_has_hand_computed_anchors() raises:
    var rows: List[ReferenceRow] = [
        ReferenceRow(2, 0, "a", False, True, True, True, True),
        ReferenceRow(99, 0, "null", False, False, True, True, True),
        ReferenceRow(2, 0, "b", False, True, True, True, True),
        ReferenceRow(1, 0, "c", False, True, True, True, True),
    ]
    var model = ReferenceTable(["i", "f", "s", "b"], rows^)
    var groups = reference_groups(model, ["i"])
    assert_equal(len(groups), 3)
    assert_equal(groups[0], [0, 2])
    assert_equal(groups[1], [1])
    assert_equal(groups[2], [3])
    var right = model.take([2, 1])
    var pairs = reference_join(model, right, ["i"], "full")
    var left_indices = List[Int]()
    var right_indices = List[Int]()
    for pair in pairs:
        left_indices.append(pair.left)
        right_indices.append(pair.right)
    assert_equal(left_indices, [0, 1, 2, 3, -1])
    assert_equal(right_indices, [0, -1, 0, -1, 1])
    var modes: List[String] = [
        "inner",
        "left",
        "right",
        "semi",
        "anti",
        "cross",
    ]
    var expected_left: List[List[Int]] = [
        [0, 2],
        [0, 1, 2, 3],
        [0, 2, -1],
        [0, 2],
        [1, 3],
        [0, 0, 1, 1, 2, 2, 3, 3],
    ]
    var expected_right: List[List[Int]] = [
        [0, 0],
        [0, -1, 0, -1],
        [0, 0, 1],
        [-1, -1],
        [-1, -1],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ]
    for mode in range(len(modes)):
        var result = reference_join(model, right, ["i"], modes[mode])
        assert_equal(len(result), len(expected_left[mode]))
        for index in range(len(result)):
            assert_equal(result[index].left, expected_left[mode][index])
            assert_equal(result[index].right, expected_right[mode][index])
    var descending = sorted_model(model, ["i"], [True], [False])
    assert_equal(descending.rows[0].s, "null")
    assert_equal(descending.rows[1].s, "a")
    assert_equal(descending.rows[2].s, "b")
    assert_equal(descending.rows[3].s, "c")
    with assert_raises():
        _ = reference_join(model, right, [], "inner")
    with assert_raises():
        _ = sorted_model(model, ["i"], [], [True])


def test_statistics_reference_has_literal_anchors() raises:
    var rows: List[ReferenceRow] = [
        ReferenceRow(2, 0, "", False, True, True, True, True),
        ReferenceRow(99, 0, "", False, False, True, True, True),
        ReferenceRow(-1, 0, "", False, True, True, True, True),
        ReferenceRow(2, 0, "", False, True, True, True, True),
    ]
    var model = ReferenceTable(["i"], rows^)
    assert_equal(integer_variance(model), Float64(3))
    assert_equal(integer_quantile(model, 0.25, "linear"), Float64(0.5))
    assert_equal(integer_quantile(model, 0.25, "midpoint"), Float64(0.5))
    assert_equal(integer_quantile(model, 0.25, "nearest"), Float64(2))
    assert_equal(integer_quantile(model, 0.25, "lower"), Float64(-1))
    assert_equal(integer_quantile(model, 0.25, "higher"), Float64(2))
    assert_equal(integer_quantile(model, 0), Float64(-1))
    assert_equal(integer_quantile(model, 1), Float64(2))
    with assert_raises():
        _ = integer_variance(model, 3)
    with assert_raises():
        _ = integer_quantile(model, 2)


def test_relation_reference_has_literal_membership_schema_and_cell_anchors() raises:
    var rows: List[ReferenceRow] = [
        ReferenceRow(2, 0, "z", False, True, True, True, True),
        ReferenceRow(9, 0, "a", True, False, True, True, True),
        ReferenceRow(3, 0, "z", False, True, True, True, True),
        ReferenceRow(2, 0, "a", False, True, True, True, True),
    ]
    var model = ReferenceTable(["i", "s", "b"], rows^)
    assert_equal(distinct_indices(model, ["i"], "first"), [0, 1, 2])
    assert_equal(distinct_indices(model, ["i"], "any"), [0, 1, 2])
    assert_equal(distinct_indices(model, ["i"], "last"), [1, 2, 3])
    assert_equal(distinct_indices(model, ["i"], "none"), [1, 2])
    assert_equal(nonnull_indices(model, []), [0, 2, 3])
    var diagonal = concatenated_model(
        [
            model.take([0]).project(["s", "i"]),
            model.take([1]).project(["b", "s"]),
        ],
        "diagonal",
    )
    assert_equal(diagonal.names, ["s", "i", "b"])
    assert_equal(len(diagonal.rows), 2)
    assert_equal(diagonal.rows[0].s, "z")
    assert_equal(diagonal.rows[0].i, Int64(2))
    assert_equal(diagonal.rows[0].bv, False)
    assert_equal(diagonal.rows[1].iv, False)
    assert_equal(diagonal.rows[1].b, True)
    var horizontal = concatenated_model(
        [model.project(["b", "i"]), model.project(["s"])], "horizontal"
    )
    assert_equal(horizontal.names, ["b", "i", "s"])
    assert_equal(horizontal.rows[3].s, "a")
    assert_equal(horizontal.rows[3].i, Int64(2))
    var vertical = concatenated_model(
        [model.take([3]), model.take([1, 0])], "vertical"
    )
    assert_equal(vertical.rows[0].s, "a")
    assert_equal(vertical.rows[1].iv, False)
    assert_equal(vertical.rows[2].s, "z")
    var pivot = pivot_model(model, ["b"], False)
    assert_equal(pivot.representatives, [0, 1])
    assert_equal(pivot.labels, ["z", "a"])
    assert_equal(pivot.cells, [[0, 2], [3], [], [1]])
    var sorted = pivot_model(model, ["b"], True)
    assert_equal(sorted.labels, ["a", "z"])
    assert_equal(sorted.cells, [[3], [0, 2], [1], []])
    with assert_raises():
        _ = distinct_indices(model, ["i", "i"], "first")
    with assert_raises():
        _ = concatenated_model([model.copy(), model.project(["i"])], "vertical")


def test_numeric_reference_has_literal_width_and_truncation_anchors() raises:
    var signed = integer_bounds[DType.int8]()
    assert_equal(signed[0], Int128(-128))
    assert_equal(signed[1], Int128(127))
    var unsigned = integer_bounds[DType.uint64]()
    assert_equal(unsigned[0], Int128(0))
    assert_equal(unsigned[1], Int128(18446744073709551615))
    assert_equal(integer_fits[DType.int8](-129), False)
    assert_equal(integer_fits[DType.int8](127), True)
    assert_equal(integer_fits[DType.uint64](-1), False)
    assert_equal(truncated_integer[DType.int8](-128.75), Int128(-128))
    assert_equal(truncated_integer[DType.int8](127.75), Int128(127))
    assert_equal(truncated_integer[DType.int64](-1.75), Int128(-1))
    assert_equal(truncated_integer[DType.uint64](-0.75), Int128(0))
    with assert_raises():
        _ = truncated_integer[DType.int8](128)
    with assert_raises():
        _ = truncated_integer[DType.uint64](-1)
    with assert_raises():
        _ = truncated_integer[DType.uint64](Float64(Int128(1) << 64))
    with assert_raises():
        _ = truncated_integer[DType.int64](Float64(0) / Float64(0))
