"""Concatenation follows independently constructed schemas and ordered rows."""
from std.testing import assert_raises
from dataframe import DataFrame, Series, concat
from tests.support.reference_model import generated_model
from tests.support.reference_relations import concatenated_model
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.coverage_compare import assert_model


def test_concat_modes_preserve_complete_cells_and_zero_column_heights() raises:
    for n in [0, 1, 7, 8, 9, 65]:
        for mode in range(3):
            var left = generated_model(n, 3, mode)
            var right = generated_model(9, 101, 1)
            for layout in range(3):
                var a = frame_from_model(left, 7 if layout == 1 else 0)
                var b = frame_from_model(right, 1 if layout == 1 else 0)
                if layout == 2:
                    a = chunk_layout(a)
                    b = chunk_layout(b)
                var expected = concatenated_model(
                    [left.copy(), right.copy()], "vertical"
                )
                assert_model(concat([a.copy(), a.clear(), b.copy()]), expected)
                assert_model(a.vstack(b), expected)
                assert_model(concat([a.copy()]), left)
                var zero_left = left.project([])
                var zero_right = right.project([])
                assert_model(
                    concat(
                        [a.select(List[String]()), b.select(List[String]())]
                    ),
                    concatenated_model([zero_left^, zero_right^], "vertical"),
                )
                var part_a = left.project(["s", "i"])
                var part_b = right.project(["f", "s", "b"])
                assert_model(
                    concat(
                        [a.select(["s", "i"]), b.select(["f", "s", "b"])],
                        "diagonal",
                    ),
                    concatenated_model([part_a^, part_b^], "diagonal"),
                )
                var aligned = generated_model(n, 101, 1)
                var horizontal = frame_from_model(aligned, 3)
                var expected_horizontal = concatenated_model(
                    [left.project(["s", "i"]), aligned.project(["b", "f"])],
                    "horizontal",
                )
                assert_model(
                    concat(
                        [a.select(["s", "i"]), horizontal.select(["b", "f"])],
                        "horizontal",
                    ),
                    expected_horizontal,
                )
                assert_model(
                    a.select(["s", "i"]).hstack(horizontal.select(["b", "f"])),
                    expected_horizontal,
                )
                assert_model(
                    a.select(["s", "i"]).hstack(
                        [horizontal.column("b"), horizontal.column("f")]
                    ),
                    expected_horizontal,
                )
                assert_model(a, left)
                assert_model(b, right)


def test_concat_rejects_incompatible_schemas_before_producing_rows() raises:
    var frame = frame_from_model(generated_model(2))
    with assert_raises():
        _ = concat(List[DataFrame]())
    with assert_raises():
        _ = concat([frame.copy()], "unknown")
    for source in [frame.copy(), frame.clear()]:
        for other in [
            source.select(["i"]),
            source.select(["s", "f", "i", "b"]),
            source.cast({"i": "float64"}),
        ]:
            with assert_raises():
                _ = concat([source.copy(), source.copy(), other.copy()])
        with assert_raises():
            _ = concat(
                [source.copy(), source.cast({"i": "float64"})], "diagonal"
            )
        with assert_raises():
            _ = source.hstack(source.select(["i"]))
        with assert_raises():
            _ = concat(
                [
                    source.select(["i"]),
                    DataFrame(List[Series](), height=source.height() + 1),
                ],
                "horizontal",
            )
