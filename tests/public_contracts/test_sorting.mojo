"""Independent stable lexicographic order, including ties and floating specials."""
from std.testing import assert_equal, assert_raises
from dataframe import DataFrame
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import assert_model
from tests.support.reference_model import (
    generated_model,
    sorted_model,
    ReferenceRow,
    ReferenceTable,
)


def test_sort_directions_nulls_stability_and_top_k() raises:
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            var model = generated_model(n, 3, mode)
            for layout in range(3):
                var frame = frame_from_model(model, 1 if layout == 1 else 0)
                if layout == 2:
                    frame = chunk_layout(frame)
                for key in model.names:
                    for descending in [False, True]:
                        for nulls_last in [False, True]:
                            var expected = sorted_model(
                                model, [key], [descending], [nulls_last]
                            )
                            assert_model(
                                frame.sort(
                                    key,
                                    descending=descending,
                                    nulls_last=nulls_last,
                                ),
                                expected,
                            )
                    for count in [0, 1, 3, n, n + 1]:
                        assert_model(
                            frame.top_k(count, key),
                            sorted_model(model, [key], [True], [True]).slice(
                                0, count
                            ),
                        )
                        assert_model(
                            frame.bottom_k(count, key),
                            sorted_model(model, [key], [False], [True]).slice(
                                0, count
                            ),
                        )
                assert_model(
                    frame.sort(
                        ["b", "i"],
                        descending=[True, False],
                        nulls_last=[False, True],
                    ),
                    sorted_model(
                        model, ["b", "i"], [True, False], [False, True]
                    ),
                )
                with assert_raises():
                    _ = frame.sort(List[String]())
                with assert_raises():
                    _ = frame.sort(
                        ["i", "b"], descending=[True], nulls_last=[True, False]
                    )
                with assert_raises():
                    _ = frame.top_k(-1, "i")


def test_float_sort_specials_keep_row_associations() raises:
    var rows = List[ReferenceRow]()
    var values: List[Float64] = [
        Float64(0) / Float64(0),
        -0.0,
        0.0,
        Float64(1) / Float64(0),
        -Float64(1) / Float64(0),
        2.0,
        2.0,
        42.0,
    ]
    for index in range(len(values)):
        rows.append(
            ReferenceRow(
                Int64(index),
                values[index],
                String(index),
                False,
                True,
                index != 7,
                True,
                True,
            )
        )
    var model = ReferenceTable(["i", "f", "s", "b"], rows^)
    for descending in [False, True]:
        for nulls_last in [False, True]:
            assert_model(
                frame_from_model(model, 7).sort(
                    "f", descending=descending, nulls_last=nulls_last
                ),
                sorted_model(model, ["f"], [descending], [nulls_last]),
            )
