"""Empty chunks, cross-boundary windows and cache reuse preserve typed cells."""
# contract-group: storage.chunks
from std.testing import assert_true
from dataframe import DataFrame, Series
from tests.support.reference_model import generated_model
from tests.support.coverage_generators import frame_from_model
from tests.support.coverage_compare import assert_model


def test_empty_chunks_and_repeated_rechunk_preserve_aliases() raises:
    for n in [0, 1, 9, 65]:
        for mode in range(3):
            var model = generated_model(n, 3, mode)
            var source = frame_from_model(model, 7)
            var columns = List[Series]()
            for name in model.names:
                var original = source.column(name)
                var cut = min(5, n)
                var chunked = Series._from_chunks(
                    [
                        original.slice(0, 0),
                        original.slice(0, cut),
                        original.slice(cut, 0),
                        original.slice(cut, n - cut),
                        original.slice(n, 0),
                    ]
                )
                var alias = chunked.copy()
                var first = chunked.rechunk()
                var second = alias.rechunk()
                assert_true(first._shares_buffers_with(second))
                assert_model(DataFrame([first.copy()]), model.project([name]))
                assert_model(DataFrame([alias.copy()]), model.project([name]))
                columns.append(chunked^)
            var frame = DataFrame(columns^, height=n)
            assert_model(frame, model)
            assert_model(frame.slice(3, 9), model.slice(3, 9))
            assert_model(frame.rechunk(), model)
            assert_model(source, model)
