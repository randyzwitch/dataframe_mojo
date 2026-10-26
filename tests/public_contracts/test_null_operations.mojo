"""All-null JSON schemas remain usable as filter, sort and grouping keys."""
# contract-group: dtype.null
from std.testing import assert_equal
from dataframe import Column, DataFrame, DataType, Series, col
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


def test_null_logical_columns_filter_sort_group_and_keep_schema_through_layouts() raises:
    var full = DataFrame(
        [
            Series.full_null("n", DataType.NULL, 9),
            Series("x", Column[Int64]([0, 1, 2, 3, 4, 5, 6, 7, 8])),
        ]
    )
    for layout in range(3):
        var frame = full.copy() if layout == 0 else (
            offset_layout(full) if layout == 1 else chunk_layout(full)
        )
        assert_frame(frame.filter(col("n").is_null()), frame)
        assert_equal(frame.filter(col("n").is_not_null()).height(), 0)
        assert_frame(frame.sort("n"), frame)
        var expected = DataFrame(
            [
                Series.full_null("n", DataType.NULL, 1),
                Series("x", Column[Int64]([36])),
            ]
        )
        assert_frame(frame.group_by("n").agg(col("x").sum()), expected)
        for optimize in [False, True]:
            for streaming in [False, True]:
                var result = (
                    frame.lazy()
                    .group_by("n")
                    .agg(col("x").sum())
                    .collect(
                        engine="cpu",
                        optimize=optimize,
                        streaming=streaming,
                        batch_size=2,
                    )
                )
                assert_frame(result, expected)
