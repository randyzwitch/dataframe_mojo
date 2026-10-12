"""Ownership and representation must preserve cells after the source dies."""
from dataframe import DataFrame
from tests.support.coverage_generators import generated_frame, chunk_layout
from tests.support.coverage_compare import assert_model
from tests.support.reference_model import generated_model


def escaped_window() raises -> DataFrame:
    var owner = generated_frame(70, 0, 1)
    return owner.slice(3, 65)


def test_windows_outlive_owners_and_chunks_preserve_cells() raises:
    var escaped = escaped_window()
    assert_model(escaped, generated_model(65, 3, 1))
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            var frame = generated_frame(n, 3, mode)
            var chunks = chunk_layout(frame)
            assert_model(chunks, generated_model(n, 3, mode))
            assert_model(
                chunks.slice(1, n), generated_model(n, 3, mode).slice(1, n)
            )
            for name in frame.columns():
                assert_model(
                    DataFrame([chunks.column(name).rechunk()]),
                    generated_model(n, 3, mode).project([name]),
                )
