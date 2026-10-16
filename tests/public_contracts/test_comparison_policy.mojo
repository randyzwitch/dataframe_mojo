"""The comparator must inspect extended cells and independent model mutations."""
# contract-group: harness.sensitivity
from std.testing import assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    ListColumn,
    StructColumn,
)
from tests.support.coverage_compare import (
    assert_frame,
    assert_model,
    assert_model_unordered,
)
from tests.support.coverage_generators import frame_from_model
from tests.support.reference_model import ReferenceRow, ReferenceTable


def reject_change(expected: DataFrame, changed: DataFrame) raises:
    assert_frame(expected, expected)
    with assert_raises():
        assert_frame(changed, expected)


def test_extended_comparator_reads_payloads_and_nested_children() raises:
    reject_change(
        DataFrame([Series("x", Column[Int8]([1]))]),
        DataFrame([Series("x", Column[Int8]([2]))]),
    )
    reject_change(
        DataFrame([Series("x", Column[UInt64]([UInt64.MAX]))]),
        DataFrame([Series("x", Column[UInt64]([0]))]),
    )
    reject_change(
        DataFrame([Series("x", Column[Float32]([-0.0]))]),
        DataFrame([Series("x", Column[Float32]([0.0]))]),
    )
    reject_change(
        DataFrame([Series("x", Column[Int64]([1])).with_dtype(DataType.DATE)]),
        DataFrame([Series("x", Column[Int64]([2])).with_dtype(DataType.DATE)]),
    )
    var decimal = DataType.decimal(8, 2)
    reject_change(
        DataFrame([Series("x", Column[Int128]([125])).with_dtype(decimal)]),
        DataFrame([Series("x", Column[Int128]([126])).with_dtype(decimal)]),
    )
    var bytes: List[List[UInt8]] = [[255, 0, 1]]
    var other: List[List[UInt8]] = [[255, 0, 2]]
    reject_change(
        DataFrame([Series.binary("x", bytes)]),
        DataFrame([Series.binary("x", other)]),
    )
    var list_a = Series(
        "x", ListColumn([0, 2], Series("child", Column[Int64]([1, 2])))
    )
    var list_b = Series(
        "x", ListColumn([0, 2], Series("child", Column[Int64]([2, 1])))
    )
    reject_change(DataFrame([list_a^]), DataFrame([list_b^]))
    var struct_a = Series(
        "x", StructColumn([Series("field", Column[Int64]([1]))])
    )
    var struct_b = Series(
        "x", StructColumn([Series("field", Column[Int64]([2]))])
    )
    reject_change(DataFrame([struct_a^]), DataFrame([struct_b^]))


def test_model_comparator_rejects_association_null_and_schema_mutations() raises:
    var rows: List[ReferenceRow] = [
        ReferenceRow(1, 0.0, "a", True, True, True, True, True),
        ReferenceRow(
            2, Float64(0) / Float64(0), "b", False, True, True, True, True
        ),
    ]
    var model = ReferenceTable(["i", "f", "s", "b"], rows^)
    var frame = frame_from_model(model)
    assert_model(frame, model)
    var swapped = model.copy()
    swapped.rows[0].i = 2
    swapped.rows[1].i = 1
    with assert_raises():
        assert_model(frame, swapped)
    var changed = model.copy()
    changed.rows[0].iv = False
    with assert_raises():
        assert_model(frame, changed)
    changed = model.copy()
    changed.rows[0].f = -0.0
    with assert_raises():
        assert_model(frame, changed)
    changed = model.copy()
    changed.rows[1].f = Float64(1) / Float64(0)
    with assert_raises():
        assert_model(frame, changed)
    changed = model.copy()
    changed.rows[0].s = "z"
    with assert_raises():
        assert_model(frame, changed)
    changed = model.copy()
    changed.rows[0].b = False
    with assert_raises():
        assert_model(frame, changed)
    with assert_raises():
        assert_model(frame.slice(0, 1), model)
    with assert_raises():
        assert_model(frame.rename({"i": "other"}), model)
    with assert_raises():
        assert_model(frame.select(["b", "s", "f", "i"]), model)
    with assert_raises():
        assert_model(frame.cast({"i": "float64"}), model)


def test_unordered_comparison_preserves_multiplicity_and_complete_rows() raises:
    var rows: List[ReferenceRow] = [
        ReferenceRow(1, 0.0, "a", True, True, True, True, True),
        ReferenceRow(1, 0.0, "a", True, True, True, True, True),
        ReferenceRow(2, 0.0, "b", False, True, True, True, True),
    ]
    var model = ReferenceTable(["i", "f", "s", "b"], rows^)
    var frame = frame_from_model(model)
    assert_model_unordered(frame.take([2, 0, 1]), model)
    with assert_raises():
        assert_model_unordered(frame.take([2, 2, 0]), model)
    with assert_raises():
        assert_model_unordered(frame.rename({"i": "different"}), model)
    var association = model.copy()
    association.rows[0].s = "b"
    association.rows[2].s = "a"
    with assert_raises():
        assert_model_unordered(frame, association)
