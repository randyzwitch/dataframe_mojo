"""Large exact serial/parallel differential for the isolated inner path."""
from std.ffi import external_call
from std.testing import TestSuite, assert_true

from dataframe import Column, DataFrame, Series
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
    var state = seed
    for row in range(rows):
        state = state * 6364136223846793005 + 1442695040888963407
        # Repeated groups plus null keys exercise both CSR fanout and skips.
        keys.append(Int64((state >> 24) % 40_009))
        payload.append(Int64(row))
        valid.append(row % 71 != 0)
    return DataFrame(
        [
            Series("k", Column[Int64](keys^, valid^)),
            Series(name, Column[Int64](payload^)),
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
    _set_threads(1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
