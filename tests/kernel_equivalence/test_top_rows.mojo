"""Parallel per-chunk Top-N candidate rebasing and stable tie selection."""
from std.math import isnan
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series, col
from dataframe.packed_sort import chunked_top_rows, packed_top_rows
from dataframe.parallel import configured_workers
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.parallel_key_reference import (
    parallel_value,
    parallel_present,
    parallel_string,
)


def test_chunked_top_rows_rebases_four_candidates_and_preserves_null_nan_and_full_string_ties() raises:
    var setting = WorkerSetting(4)
    assert_true(configured_workers() > 1)
    var n = 196609
    assert_equal((n + 65535) // 65536, 4)
    var values = List[Float64]()
    var valid = List[Bool]()
    var text = List[String]()
    var text_valid = List[Bool]()
    var ids = List[Int64]()
    for row in range(n + 7):
        values.append(
            parallel_value(row) if parallel_present(row) else Float64.MAX
        )
        valid.append(parallel_present(row))
        text.append(parallel_string(row))
        text_valid.append(row % 17 != 3)
        ids.append(Int64(row - 7))
    var columns: List[Series] = [
        Series("value", Column[Float64](values.copy(), valid.copy())).slice(
            7, n
        ),
        Series("text", Column[String](text.copy(), text_valid.copy())).slice(
            7, n
        ),
    ]
    var frame = DataFrame(
        [
            columns[0].copy(),
            columns[1].copy(),
            Series("row", Column[Int64](ids^)).slice(7, n),
        ]
    )
    for descending in [False, True]:
        for nulls_last in [False, True]:
            var expected = List[Int]()
            for row in range(n):
                expected.append(row)

            def before(
                a: Int, b: Int
            ) {
                imm values,
                imm valid,
                imm text,
                imm text_valid,
                descending,
                nulls_last,
            } -> Bool:
                var av = valid[a + 7]
                var bv = valid[b + 7]
                if av != bv:
                    return av if nulls_last else bv
                if av:
                    var x = values[a + 7]
                    var y = values[b + 7]
                    if isnan(x) != isnan(y):
                        return not isnan(x)
                    if not isnan(x) and x != y:
                        return x > y if descending else x < y
                var ap = text_valid[a + 7]
                var bp = text_valid[b + 7]
                if ap != bp:
                    return ap if nulls_last else bp
                if ap and text[a + 7] != text[b + 7]:
                    return text[a + 7] < text[b + 7]
                return a < b

            sort(expected, before)
            for k in [0, 1, 31, 65537, n + 1]:
                var actual = chunked_top_rows(
                    columns, [descending, False], [nulls_last, nulls_last], k
                )
                assert_true(
                    Bool(actual),
                    "each of four chunk jobs must accept these packed keys",
                )
                var prefix = expected.copy()
                prefix.shrink(min(k, n))
                assert_equal(actual.take(), prefix)
                var whole = packed_top_rows(
                    columns,
                    [descending, False],
                    [nulls_last, nulls_last],
                    k,
                    threads=1,
                )
                assert_true(Bool(whole))
                assert_equal(whole.take(), prefix)
            # The public lazy TOP_K executor receives the same literal oracle
            # across ragged batches, with an offset crossing tied candidates.
            for streaming in [False, True]:
                var actual = (
                    frame.lazy()
                    .sort(
                        ["value", "text"],
                        descending=[descending, False],
                        nulls_last=[nulls_last, nulls_last],
                    )
                    .slice(3, 31)
                    .collect(engine="cpu", streaming=streaming, batch_size=1025)
                )
                assert_equal(actual.height(), 31)
                for destination in range(31):
                    var row = expected[destination + 3]
                    assert_equal(
                        actual.item(destination, "row").int64(), Int64(row)
                    )
                    var cell = actual.item(destination, "value")
                    assert_equal(cell.is_null(), not valid[row + 7])
                    if valid[row + 7]:
                        if isnan(values[row + 7]):
                            assert_true(isnan(cell.float64()))
                        else:
                            assert_equal(cell.float64(), values[row + 7])
                    var word = actual.item(destination, "text")
                    assert_equal(word.is_null(), not text_valid[row + 7])
                    if text_valid[row + 7]:
                        assert_equal(word.string(), text[row + 7])
    _ = setting^
