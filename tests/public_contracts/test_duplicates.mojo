"""Distinct row membership and multiplicity follow scalar key equality."""
# contract-group: table.sets_reshape
from std.testing import assert_equal, assert_raises
from dataframe import DataFrame
from tests.support.reference_model import (
    generated_model,
    reference_groups,
    ReferenceRow,
    ReferenceTable,
)
from tests.support.reference_relations import distinct_indices, nonnull_indices
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import assert_model, assert_model_unordered


def assert_distinct_contract(
    frame: DataFrame, model: ReferenceTable, subset: List[String]
) raises:
    var keys = model.names.copy() if not subset else subset.copy()
    var groups = reference_groups(model, keys)
    assert_equal(frame.n_unique(subset), len(groups))
    var duplicated = frame.is_duplicated(subset)
    var unique = frame.is_unique(subset)
    assert_equal(duplicated.name(), "is_duplicated")
    assert_equal(unique.name(), "is_unique")
    assert_equal(len(duplicated), len(model.rows))
    assert_equal(len(unique), len(model.rows))
    for group in groups:
        for row in group:
            assert_equal(duplicated.get(row).bool(), len(group) > 1)
            assert_equal(unique.get(row).bool(), len(group) == 1)
    for keep in ["any", "first", "last", "none"]:
        var indices = distinct_indices(model, subset, keep)
        assert_model(
            frame.unique(subset, keep=keep, maintain_order=True),
            model.take(indices),
        )
        assert_model_unordered(
            frame.unique(subset, keep=keep, maintain_order=False),
            model.take(indices),
        )


def test_distinct_keep_policies_and_duplicate_flags_match_scalar_groups() raises:
    var subsets: List[List[String]] = [
        [],
        ["i"],
        ["s"],
        ["b"],
        ["f"],
        ["s", "b"],
    ]
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            var model = generated_model(n, 3, mode)
            if n > 1:
                # Repeated complete rows challenge multiplicity, not just keys.
                model.rows.append(model.rows[0].copy())
                model.rows.append(model.rows[0].copy())
            for layout in range(3):
                var frame = frame_from_model(model, 7 if layout == 1 else 0)
                if layout == 2:
                    frame = chunk_layout(frame)
                for subset in subsets:
                    assert_distinct_contract(frame, model, subset)
                var null_subsets: List[List[String]] = [[], ["i"], ["s", "b"]]
                for subset in null_subsets:
                    assert_model(
                        frame.drop_nulls(subset),
                        model.take(nonnull_indices(model, subset)),
                    )


def test_distinct_float_equality_and_invalid_subsets() raises:
    var rows = List[ReferenceRow]()
    var values: List[Float64] = [
        -0.0,
        0.0,
        Float64(0) / Float64(0),
        Float64(0) / Float64(0),
        8,
        9,
    ]
    for index in range(len(values)):
        rows.append(
            ReferenceRow(
                Int64(index),
                values[index],
                String(index),
                False,
                True,
                index < 4,
                True,
                True,
            )
        )
    var model = ReferenceTable(["i", "f", "s", "b"], rows^)
    var frame = frame_from_model(model, 7)
    assert_distinct_contract(frame, model, ["f"])
    for source in [frame.copy(), frame.clear()]:
        var invalid_subsets: List[List[String]] = [["missing"], ["i", "i"]]
        for subset in invalid_subsets:
            with assert_raises():
                _ = source.unique(subset)
            with assert_raises():
                _ = source.is_duplicated(subset)
            with assert_raises():
                _ = source.is_unique(subset)
        with assert_raises():
            _ = source.unique(keep="invalid")
    with assert_raises():
        _ = frame.select(List[String]()).unique()
