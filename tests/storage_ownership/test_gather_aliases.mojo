"""Sorted duplicates cannot impersonate an untouched full physical chunk."""
# contract-group: storage.isolation
# contract-group: storage.chunks
# contract-group: kernel.row_ops
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series
from dataframe.gather import (
    take_sorted_chunked,
    _take_sorted_chunked_partitioned,
)
from tests.support.lifecycle_jobs import WorkerSetting


def test_sorted_repeated_positions_with_full_chunk_endpoints_never_reuse_wrong_rows() raises:
    var values = List[Int64]()
    var labels = List[String]()
    var valid = List[Bool]()
    for row in range(1024):
        values.append(Int64(10000 + row))
        labels.append("independent external string payload " + String(row))
        valid.append(row % 17 != 0)
    var numbers = Series("i", Column[Int64](values^, valid.copy()))
    var text = Series("s", Column[String](labels^, valid.copy()))
    for aligned in [False, True]:
        var numeric_parts = List[Series]()
        var string_parts = List[Series]()
        for chunk in range(16):
            numeric_parts.append(numbers.slice(chunk * 64, 64))
            var first = chunk * (64 if aligned else 63)
            var length = 64 if aligned else 79 if chunk == 15 else 63
            string_parts.append(text.slice(first, length))
        var columns: List[Series] = [
            Series._from_chunks(numeric_parts),
            Series._from_chunks(string_parts),
        ]
        var rows: List[Int] = [0, 0]
        for row in range(2, 64):
            rows.append(row)
        rows.append(64)
        rows.append(64)
        rows.append(1023)
        assert_equal(rows[0], 0)
        assert_equal(rows[63], 63)
        assert_equal(len(rows), 67)
        var setting = WorkerSetting(4)
        for partitioned in [False, True]:
            var gathered = _take_sorted_chunked_partitioned(
                columns, rows.copy(), 4, allow_repeats=True
            ) if partitioned else take_sorted_chunked(
                columns, rows.copy(), workers=1, allow_repeats=True
            )
            var actual = DataFrame(gathered^)
            assert_equal(actual.height(), len(rows))
            for row in range(len(rows)):
                var original = rows[row]
                for name in ["i", "s"]:
                    assert_equal(
                        actual.item(row, name).is_null(), not valid[original]
                    )
                if valid[original]:
                    assert_equal(
                        actual.item(row, "i").int64(), Int64(10000 + original)
                    )
                    assert_equal(
                        actual.item(row, "s").string(),
                        "independent external string payload "
                        + String(original),
                    )
            # Read-only gather leaves all aliases and owners unchanged.
            assert_true(
                columns[0]
                .chunks()[0]
                .int64()
                ._shares_buffers_with(numbers.int64())
            )
            assert_equal(numbers.get(1).int64(), Int64(10001))
            assert_equal(
                text.get(1).string(), "independent external string payload 1"
            )
        _ = setting^
