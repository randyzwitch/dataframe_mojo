"""Large exact serial/parallel differential for inner joins and stable CSR construction."""
from std.ffi import external_call
from std.testing import TestSuite, assert_true, assert_equal, assert_raises
from std.memory import ArcPointer
from dataframe.frame import (
    _group_index,
    _group_rows,
    _parallel_group_rows,
    _JoinCountJob,
)

from dataframe import Column, DataFrame, Series, StringColumn
from dataframe.parallel import worker_count


comptime LEFT_ROWS = 200_003
comptime RIGHT_ROWS = 160_001


def _c_string(text: String) -> List[UInt8]:
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    bytes.append(0)
    return bytes^


def _set_threads(count: Int):
    var name = _c_string("DATAFRAME_THREADS")
    var value = _c_string(String(count))
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


def _side(rows: Int, seed: UInt64, name: String) raises -> DataFrame:
    var keys = List[Int64](capacity=rows)
    var payload = List[Int64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    var texts = List[String](capacity=rows)
    var text_valid = List[Bool](capacity=rows)
    var state = seed
    for row in range(rows):
        state = state * 6364136223846793005 + 1442695040888963407
        # Repeated groups plus null keys exercise both CSR fanout and skips.
        keys.append(Int64((state >> 24) % 40_009))
        payload.append(Int64(row))
        valid.append(row % 71 != 0)
        texts.append("value_" + String(row % 101))
        text_valid.append(row % 37 != 0)
    return DataFrame(
        [
            Series("k", Column[Int64](keys^, valid^)),
            Series(name, Column[Int64](payload^)),
            Series(name + "_text", StringColumn(texts^, text_valid^)),
        ]
    )


def test_parallel_inner_exactly_matches_serial() raises:
    var left = _side(LEFT_ROWS, 7, "left_row")
    var right = _side(RIGHT_ROWS, 11, "right_row")
    _set_threads(1)
    var serial = left.join(right, "k")
    _set_threads(32)
    assert_true(worker_count(left.height()) > 1, "parallel path not reached")
    var parallel = left.join(right, "k")
    assert_true(parallel.equals(serial), "inner join row order differs")
    assert_true(parallel.column("left_row_text").n_chunks() > 1)
    assert_true(parallel.column("right_row_text").n_chunks() > 1)
    _set_threads(1)


def test_dense_right_join_matches_fallback_with_nulls_and_chunks() raises:
    var left_key = Series(
        "k",
        Column[Int64](
            [11, 9, 11, 13, 12, 10], [True, True, True, True, False, True]
        ),
    )
    var left = DataFrame(
        [
            Series._from_chunks([left_key.slice(0, 2), left_key.slice(2, 4)]),
            Series("left_row", Column[Int64]([0, 1, 2, 3, 4, 5])),
        ]
    )
    var right_key = Series("k", Column[Int64]([10, 11, 12]))
    var dense = DataFrame(
        [
            Series._from_chunks([right_key.slice(0, 1), right_key.slice(1, 2)]),
            Series("right_value", Column[Int64]([100, 110, 120])),
        ]
    )
    var shuffled = DataFrame(
        [
            Series("k", Column[Int64]([12, 10, 11])),
            Series("right_value", Column[Int64]([120, 100, 110])),
        ]
    )
    var actual = left.join(dense, "k")
    var expected = left.join(shuffled, "k")
    assert_true(actual.equals(expected))
    assert_equal(actual.height(), 3)
    assert_equal(actual.item(0, "left_row").int64(), 0)
    assert_equal(actual.item(1, "left_row").int64(), 2)
    assert_equal(actual.item(2, "left_row").int64(), 5)
    assert_equal(actual.item(0, "right_value").int64(), 110)
    assert_equal(actual.item(2, "right_value").int64(), 100)

    var high = DataFrame(
        [
            Series("k", Column[Int64]([Int64.MAX - 1, Int64.MAX])),
            Series("right_value", Column[Int64]([1, 2])),
        ]
    )
    var near_max = DataFrame(
        [
            Series("k", Column[Int64]([Int64.MAX, Int64.MAX - 1])),
            Series("left_row", Column[Int64]([0, 1])),
        ]
    )
    var edge = near_max.join(high, "k")
    assert_equal(edge.item(0, "right_value").int64(), 2)
    assert_equal(edge.item(1, "right_value").int64(), 1)

    var key_only = near_max.join(
        DataFrame([Series("k", Column[Int64]([Int64.MAX - 1, Int64.MAX]))]),
        "k",
    )
    assert_equal(key_only.height(), 2)
    assert_equal(key_only.width(), 2)


def test_parallel_unique_right_preserves_left_buffers() raises:
    var left_keys = List[Int64](capacity=LEFT_ROWS)
    var left_values = List[Int64](capacity=LEFT_ROWS)
    for i in range(LEFT_ROWS):
        left_keys.append(Int64(i % 50003))
        left_values.append(Int64(i))
    var right_keys = List[Int64](capacity=50003)
    var right_values = List[Int64](capacity=50003)
    for i in range(50003):
        right_keys.append(Int64(i))
        right_values.append(Int64(i * 2))
    var left = DataFrame(
        [
            Series("k", Column[Int64](left_keys^)),
            Series("v", Column[Int64](left_values^)),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int64](right_keys^)),
            Series("r", Column[Int64](right_values^)),
        ]
    )
    _set_threads(32)
    assert_true(worker_count(left.height()) > 1)
    var joined = left.join(right, "k")
    assert_equal(joined.height(), LEFT_ROWS)
    assert_true(
        joined.column("v")
        .int64()
        ._shares_buffers_with(left.column("v").int64())
    )
    for i in [0, 1, 50002, 50003, LEFT_ROWS - 1]:
        assert_equal(joined.item(i, "v").int64(), Int64(i))
        assert_equal(joined.item(i, "r").int64(), Int64((i % 50003) * 2))
    _set_threads(1)


def test_parallel_csr_stability_and_empty_groups() raises:
    _set_threads(32)
    for groups in [0, 1, 3, 33, 1009]:
        var ids = List[Int](capacity=4003)
        for row in range(4003):
            ids.append(
                -1 if groups == 0 or row % 7 == 0 else (row * 53) % groups
            )
        var starts = _group_index(ids, groups)
        var expected = _group_rows(ids, starts)
        for workers in [2, 8, 32]:
            var actual = _parallel_group_rows(ids, starts, workers)
            assert_equal(len(actual), len(expected))
            for i in range(len(actual)):
                assert_equal(actual[i], expected[i])
    _set_threads(1)


def test_parallel_count_overflow_precedes_output_allocation() raises:
    var job = _JoinCountJob(
        ArcPointer(List[Int]([0, 0])),
        ArcPointer(List[Int]([0, Int.MAX])),
        0,
        2,
    )
    with assert_raises(contains="Join output row count overflows"):
        job.run()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
