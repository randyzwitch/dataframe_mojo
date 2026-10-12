"""Typed CSV interchange preserves null/empty/text distinctions across buffers."""
from std.os import getenv
from dataframe import CsvSchema, read_csv, write_csv
from tests.support.coverage_generators import generated_frame, chunk_layout
from tests.support.coverage_compare import assert_model
from tests.support.reference_model import generated_model


def test_csv_preserves_typed_cells_across_buffers() raises:
    var path = getenv("COVERAGE_TEST_TMP") + "/roundtrip.csv"
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            var frame = chunk_layout(generated_frame(n, 3, mode))
            for size in [1, 7, 64]:
                write_csv(frame, path, buffer_size=size)
                assert_model(
                    read_csv(path, CsvSchema.of(frame), buffer_size=size),
                    generated_model(n, 3, mode),
                )
