"""Exact raw-key partitioned join checks for the isolated #105 experiment."""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal

from dataframe import Column, DataFrame, Series
from dataframe.frame import _joint_key_ids
from dataframe.string_column import StringColumn
from raw_partitioned_join_lib import (
    raw_partitioned_int64_inner,
    raw_partitioned_string_inner,
)


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


def int_frame(
    var keys: List[Int64], var valid: List[Bool], var payload_name: String
) raises -> DataFrame:
    var payload = List[Int64](capacity=len(keys))
    for row in range(len(keys)):
        payload.append(Int64(row))
    return DataFrame(
        [
            Series("k", Column[Int64](keys^, valid^)),
            Series(payload_name, Column[Int64](payload^)),
        ]
    )


def string_frame(
    var keys: List[String], var valid: List[Bool], var payload_name: String
) raises -> DataFrame:
    var payload = List[Int64](capacity=len(keys))
    for row in range(len(keys)):
        payload.append(Int64(row))
    return DataFrame(
        [
            Series("k", Column[String](keys^, valid^)),
            Series(payload_name, Column[Int64](payload^)),
        ]
    )


def expected_int(
    left: DataFrame, right: DataFrame
) -> Tuple[List[Int], List[Int]]:
    ref left_key = left._columns[0]._data[Column[Int64]]
    ref right_key = right._columns[0]._data[Column[Int64]]
    var left_rows = List[Int]()
    var right_rows = List[Int]()
    for i in range(len(left_key)):
        if left_key._valid(i):
            for j in range(len(right_key)):
                if right_key._valid(j) and left_key._get(i) == right_key._get(
                    j
                ):
                    left_rows.append(i)
                    right_rows.append(j)
    return (left_rows^, right_rows^)


def expected_string(
    left: DataFrame, right: DataFrame
) -> Tuple[List[Int], List[Int]]:
    ref left_key = left._columns[0]._data[StringColumn]
    ref right_key = right._columns[0]._data[StringColumn]
    var left_rows = List[Int]()
    var right_rows = List[Int]()
    for i in range(len(left_key)):
        if left_key._valid(i):
            for j in range(len(right_key)):
                if right_key._valid(j) and left_key._get(i) == right_key._get(
                    j
                ):
                    left_rows.append(i)
                    right_rows.append(j)
    return (left_rows^, right_rows^)


def assert_int_pairs(
    left: DataFrame,
    right: DataFrame,
    workers: Int,
    suffix: String = "_right",
) raises:
    var expected = expected_int(left, right)
    var actual = raw_partitioned_int64_inner(
        left, right, 0, 0, workers, suffix=suffix
    )
    assert_equal(actual.left_rows, expected[0])
    assert_equal(actual.right_rows, expected[1])
    var joined = left.join(right, "k", suffix=suffix)
    assert_equal(actual.frame.equals(joined), True)


def assert_string_pairs(
    left: DataFrame,
    right: DataFrame,
    workers: Int,
    suffix: String = "_right",
) raises:
    var expected = expected_string(left, right)
    var actual = raw_partitioned_string_inner(
        left, right, 0, 0, workers, suffix=suffix
    )
    assert_equal(actual.left_rows, expected[0])
    assert_equal(actual.right_rows, expected[1])
    var joined = left.join(right, "k", suffix=suffix)
    assert_equal(actual.frame.equals(joined), True)


def test_int64_low_cardinality_nullable_duplicates_and_sparse_extremes() raises:
    _set_threads(8)
    var left = int_frame(
        [Int64.MIN, -9, 3, 3, 700_000_003, 0, Int64.MAX],
        [True, True, True, True, True, False, True],
        "payload",
    )
    var right = int_frame(
        [3, Int64.MIN, 3, -9, Int64.MAX, 12, 700_000_003],
        [True, True, True, True, True, False, True],
        "payload",
    )
    # The production dense-range path must reject this astronomical span.
    var ids = _joint_key_ids(left, right, [0], [0])
    assert_equal(ids[2], 5)
    # This also checks right payload suffixing against DataFrame.join.
    assert_int_pairs(left, right, 8, suffix="_rhs")
    _set_threads(1)


def test_int64_high_cardinality_and_skew_keep_left_major_order() raises:
    _set_threads(8)
    var left_keys = List[Int64]()
    var left_valid = List[Bool]()
    for row in range(2003):
        left_keys.append(Int64(17) if row % 5 else Int64(row % 257))
        left_valid.append(row % 71 != 0)
    var right_keys = List[Int64]()
    for key in range(257):
        right_keys.append(Int64(key))
    var left = int_frame(left_keys^, left_valid^, "left_row")
    var right = int_frame(
        right_keys^, List[Bool](length=257, fill=True), "right_row"
    )
    assert_int_pairs(left, right, 8)
    _set_threads(1)


def test_string_low_high_and_skew_nullable_duplicates() raises:
    _set_threads(8)
    var left_keys = List[String]()
    var left_valid = List[Bool]()
    var right_keys = List[String]()
    var right_valid = List[Bool]()
    for key in range(257):
        right_keys.append("key_" + String(key))
        right_valid.append(True)
    right_keys.append("hot")
    right_keys.append("hot")
    right_valid.append(True)
    right_valid.append(True)
    for row in range(2003):
        left_keys.append("hot" if row % 5 else "key_" + String(row % 257))
        left_valid.append(row % 67 != 0)
    left_keys.append("missing")
    left_keys.append("hot")
    left_valid.append(False)
    left_valid.append(True)
    var left = string_frame(left_keys^, left_valid^, "payload")
    var right = string_frame(right_keys^, right_valid^, "payload")
    assert_string_pairs(left, right, 8)
    _set_threads(1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
