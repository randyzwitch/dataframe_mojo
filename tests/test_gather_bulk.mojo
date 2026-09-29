"""Bulk gathers (#328): fixed-width columns and offset-storage strings copy
runs of consecutive rows with memcpy. Checked against per-row expectations
on runs, repeats, reversals, missing rows (-1), nulls and windows that do
not start at row zero, plus out-of-bounds errors."""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import Column, DataFrame, DataType, Series, col
from dataframe.column import gather_scalars
from dataframe.string_column import StringColumn


def index_sets() -> List[List[Int]]:
    return [
        List[Int](),
        [0],
        [3, 4, 5, 6, 7, 8],
        [0, 1, 2, 9, 10, 20, 21, 22, 23, 39],
        [5, 5, 5, 4, 3, 2],
        [39, 0, 38, 1, 20, 21],
        [7, -1, 8, 9, -1, -1, 30],
    ]


def test_scalar_gather_matches_rows() raises:
    var values = List[Int64]()
    var valid = List[Bool]()
    for i in range(50):
        values.append(Int64(i * 3 - 7))
        valid.append(i % 7 != 3)
    var whole = Column[Int64](values^, valid^)
    for offset in [0, 5]:
        var column = whole.slice(offset, 40)
        for indices in index_sets():
            var missing = False
            for i in indices:
                missing = missing or i == -1
            var got = gather_scalars[DType.int64](column, indices, missing)
            assert_equal(len(got), len(indices))
            for k in range(len(indices)):
                var row = indices[k]
                if row == -1:
                    assert_true(not got._valid(k))
                    continue
                assert_equal(got._valid(k), column._valid(row))
                if column._valid(row):
                    assert_equal(got._get(k), column._get(row))
    # No nulls in the source: validity stays absent.
    var dense = Column[Float64]([1.5, 2.5, 3.5, 4.5])
    var picked = gather_scalars[DType.float64](dense, [1, 2, 3, 0], False)
    assert_equal(len(picked._bits[]), 0)
    assert_equal(picked._get(3), 1.5)
    with assert_raises(contains="out of bounds"):
        _ = gather_scalars[DType.float64](dense, [2, 3, 4], False)
    with assert_raises(contains="out of bounds"):
        _ = gather_scalars[DType.float64](dense, [-1], False)


def test_string_gather_matches_rows() raises:
    var texts = List[String]()
    var valid = List[Bool]()
    for i in range(50):
        texts.append("row-" + String(i) * (i % 4))
        valid.append(i % 6 != 2)
    var whole = StringColumn(texts, valid)
    for offset in [0, 7]:
        var column = whole.slice(offset, 40)
        for indices in index_sets():
            var missing = False
            for i in indices:
                missing = missing or i == -1
            var got = column.take_or_null(
                indices, ""
            ) if missing else column.take(indices)
            assert_equal(len(got), len(indices))
            for k in range(len(indices)):
                var row = indices[k]
                if row == -1:
                    assert_true(not got._valid(k))
                    continue
                assert_equal(got._valid(k), column._valid(row))
                if column._valid(row):
                    assert_equal(String(got._get(k)), String(column._get(row)))
    with assert_raises(contains="out of bounds"):
        _ = whole.slice(0, 10).take([9, 10])


def test_series_take_keeps_dtype_and_nulls() raises:
    var dates = Series(
        "d", Column[Int64]([10, 20, 30, 40], [True, False, True, True])
    ).with_dtype(DataType.DATE)
    var picked = dates.take([3, 1, 2])
    assert_true(picked.dtype() == DataType.DATE)
    assert_true(picked.get(1).is_null())
    assert_equal(String(picked.get(0)), String(dates.get(3)))
    var joined = dates.take_or_null([0, -1, 3])
    assert_true(joined.get(1).is_null())
    assert_equal(String(joined.get(2)), String(dates.get(3)))


def test_chunked_filters_gather_in_place() raises:
    """A filter over a multi-chunk column gathers each chunk's share of the
    sorted indices in place, for numbers and strings, nulls included."""
    var numbers = List[Series]()
    var words = List[Series]()
    for c in range(3):
        var values = List[Int64]()
        var valid = List[Bool]()
        var texts = List[String]()
        # Chunks of at least 512 rows take the chunk-local filter path.
        for i in range(600):
            values.append(Int64(c * 1000 + i))
            valid.append(i % 4 != 1)
            texts.append("c" + String(c) + "-" + String(i))
        numbers.append(Series("n", Column[Int64](values^, valid.copy())))
        words.append(Series("s", StringColumn(texts, valid)))
    var n = Series._from_chunks(numbers^)
    var s = Series._from_chunks(words^)
    assert_true(n.is_chunked())
    var frame = DataFrame([n.copy(), s.copy()])
    var kept = frame.filter(col("n") % 3 != 0)
    var flat_n = n.rechunk()
    var flat_s = s.rechunk()
    var row = 0
    for i in range(len(flat_n)):
        var cell = flat_n.get(i)
        if cell.is_null() or cell.int64() % 3 == 0:
            continue
        assert_equal(kept.item(row, "n").int64(), cell.int64())
        assert_equal(kept.item(row, "s").string(), flat_s.get(i).string())
        row += 1
    assert_equal(kept.height(), row)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
