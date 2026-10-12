"""Group membership and aggregate associations use scalar key equality."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame
from tests.support.reference_model import (
    generated_model,
    reference_groups,
    ReferenceRow,
    ReferenceTable,
)
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import (
    assert_reference_cell,
    reference_dtype,
)
from tests.support.reduction_contract import (
    integer_reductions,
    assert_integer_reductions,
)


def test_grouped_integer_statistics_match_scalar_row_groups() raises:
    var key_sets: List[List[String]] = [["s"], ["b"], ["f"], ["s", "b"]]
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            var model = generated_model(n, 3, mode)
            for layout in range(3):
                var frame = frame_from_model(model, 7 if layout == 1 else 0)
                if layout == 2:
                    frame = chunk_layout(frame)
                for keys in key_sets:
                    var groups = reference_groups(model, keys)
                    var actual = frame.group_by(keys, maintain_order=True).agg(
                        integer_reductions()
                    )
                    var names = keys.copy()
                    names.extend(
                        [
                            String("sum"),
                            String("count"),
                            String("len"),
                            String("min"),
                            String("max"),
                            String("mean"),
                            String("var"),
                            String("std"),
                            String("median"),
                            String("first"),
                            String("last"),
                            String("distinct"),
                        ]
                    )
                    assert_equal(actual.columns(), names)
                    assert_equal(actual.height(), len(groups))
                    for key in keys:
                        assert_true(
                            actual.column(key).dtype() == reference_dtype(key)
                        )
                    for group in range(len(groups)):
                        for key in keys:
                            assert_reference_cell(
                                actual.item(group, key),
                                model.rows[groups[group][0]],
                                key,
                            )
                        assert_integer_reductions(
                            actual, group, model.take(groups[group])
                        )


def test_group_float_specials_and_invalid_key_sets() raises:
    var rows = List[ReferenceRow]()
    var values: List[Float64] = [
        -0.0,
        0.0,
        Float64(0) / Float64(0),
        Float64(0) / Float64(0),
        0.0,
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
    var frame = frame_from_model(model, 1)
    var groups = reference_groups(model, ["f"])
    assert_equal(len(groups), 3)
    var actual = frame.group_by("f", maintain_order=True).agg(
        integer_reductions()
    )
    assert_equal(actual.height(), len(groups))
    for group in range(len(groups)):
        assert_reference_cell(
            actual.item(group, "f"), model.rows[groups[group][0]], "f"
        )
        assert_integer_reductions(actual, group, model.take(groups[group]))
    var invalid_keys: List[List[String]] = [[], ["missing"], ["s", "s"]]
    for keys in invalid_keys:
        with assert_raises():
            _ = frame.group_by(keys).agg(integer_reductions())
