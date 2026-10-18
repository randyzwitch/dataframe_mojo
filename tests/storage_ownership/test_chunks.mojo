"""Empty chunks, cross-boundary windows and cache reuse preserve typed cells."""
# contract-group: storage.chunks
from std.testing import assert_true
from dataframe import Column, DataFrame, DataType, Series
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


def test_numeric_chunks_merge_on_every_worker_with_their_nulls_and_dtypes() raises:
    # Chunk lengths that start rows anywhere within a validity byte, and
    # enough rows for several workers to share the merge.
    var n = 300_001
    var ints = List[Int64]()
    var small = List[Int32]()
    var floats = List[Float64]()
    var valid = List[Bool]()
    for row in range(n):
        ints.append(Int64(row * 3 - 7))
        small.append(Int32(row % 1000))
        floats.append(Float64(row) / 4.0)
        valid.append(row % 5 != 3)
    var sources: List[Series] = [
        Series("ints", Column[Int64](ints^, valid.copy())),
        Series("small", Column[Int32](small^)),
        Series("floats", Column[Float64](floats^, valid^)),
        Series("days", Column[Int64](List[Int64](length=n, fill=7))).with_dtype(
            DataType.DATE
        ),
    ]
    var cuts: List[Int] = [0, 1, 9, 70_003, 70_004, 150_000, 233_337, n]
    for source in sources:
        # A window that does not start at its buffer's first row.
        var window = source.slice(1, n - 1)
        var parts = List[Series]()
        for c in range(len(cuts) - 1):
            var lo = min(cuts[c], n - 1)
            var hi = min(cuts[c + 1], n - 1)
            parts.append(window.slice(lo, hi - lo))
        var chunked = Series._from_chunks(parts)
        assert_true(chunked.is_chunked())
        var merged = chunked.rechunk()
        assert_true(not merged.is_chunked())
        assert_true(merged.dtype() == source.dtype())
        assert_true(merged.equals(window))
        assert_true(merged.null_count() == window.null_count())
