"""Exact row-hash join matches across nulls, duplicates, and key dtypes."""
from std.testing import TestSuite, assert_equal
from dataframe import Column, DataFrame, Series
from dataframe.join_hash import direct_hash_join_rows
from dataframe.frame import _bounded_int64_join_rows
from dataframe.partition import Partitioner


def test_bounded_int64_rows_keep_duplicates_nulls_and_extremes() raises:
    var low = Int64.MIN
    var left = Series(
        "k",
        Column[Int64](
            [low, low + 1, low + 2, Int64.MAX, 0],
            [True, True, True, True, False],
        ),
    )
    var right = Series(
        "k",
        Column[Int64]([low, low, low + 2, 0], [True, True, True, False]),
    )
    var inner = _bounded_int64_join_rows(left, right, False)
    assert_equal(inner[0], True)
    assert_equal(inner[1], [0, 0, 2])
    assert_equal(inner[2], [0, 1, 2])
    var outer = _bounded_int64_join_rows(left, right, True)
    assert_equal(outer[1], [0, 0, 1, 2, 3, 4])
    assert_equal(outer[2], [0, 1, -1, 2, -1, -1])
    var wide = Series("k", Column[Int64]([Int64.MIN, Int64.MAX]))
    var extreme = _bounded_int64_join_rows(left, wide, False)
    assert_equal(extreme[0], True)
    assert_equal(extreme[1], [0, 3])
    assert_equal(extreme[2], [0, 1])


def test_parallel_bounded_index_keeps_duplicate_row_order() raises:
    var keys = List[Int64](capacity=2_100_000)
    var valid = List[Bool](capacity=2_100_000)
    for row in range(2_100_000):
        keys.append(Int64(row if row < 2_000_000 else row - 2_000_000))
        valid.append(row != 2_000_001)
    var build = Series("k", Column[Int64](keys^, valid^))
    var probe = Series("k", Column[Int64]([0, 1, 99_999, 1_999_999, 2_000_000]))
    var rows = _bounded_int64_join_rows(probe, build, True)
    assert_equal(rows[0], True)
    assert_equal(rows[1], [0, 0, 1, 2, 2, 3, 4])
    assert_equal(rows[2], [0, 2_000_000, 1, 99_999, 2_099_999, 1_999_999, -1])


def test_integer_duplicates_nulls_and_unmatched() raises:
    var left = Series(
        "k", Column[Int64]([5, 1, 5, 9, 0], [True, True, True, False, True])
    )
    var right = Series(
        "k", Column[Int64]([5, 5, 1, 9], [True, True, True, False])
    )
    var inner = direct_hash_join_rows([left.copy()], [right.copy()], False)
    assert_equal(inner[0], [0, 0, 1, 2, 2])
    assert_equal(inner[1], [0, 1, 2, 0, 1])
    var outer = direct_hash_join_rows([left.copy()], [right.copy()], True)
    assert_equal(outer[0], [0, 0, 1, 2, 2, 3, 4])
    assert_equal(outer[1], [0, 1, 2, 0, 1, -1, -1])


def test_packed_int64_hash_keys_keep_extremes_and_nulls() raises:
    var left = Series(
        "k",
        Column[Int64](
            [Int64.MIN, -17, Int64.MAX, 0],
            [True, True, True, False],
        ),
    )
    var right = Series(
        "k",
        Column[Int64](
            [Int64.MAX, Int64.MIN, Int64.MIN, -17, 0],
            [True, True, True, True, False],
        ),
    )
    var inner = direct_hash_join_rows([left.copy()], [right.copy()], False)
    assert_equal(inner[0], [0, 0, 1, 2])
    assert_equal(inner[1], [1, 2, 3, 0])
    var outer = direct_hash_join_rows([left.copy()], [right.copy()], True)
    assert_equal(outer[0], [0, 0, 1, 2, 3])
    assert_equal(outer[1], [1, 2, 3, 0, -1])


def test_hash_slot_collision_does_not_match_different_values() raises:
    var values = List[Int64]()
    for i in range(1_000):
        values.append(Int64(i))
    var source = Series("k", Column[Int64](values^))
    var hashes = Partitioner([source.copy()], 1)
    var collision = -1
    for i in range(1, 1_000):
        if hashes.hashes[i] >> 63 == hashes.hashes[0] >> 63 and hashes.hashes[
            i
        ] & UInt64(1) == hashes.hashes[0] & UInt64(1):
            collision = i
            break
    assert_equal(collision >= 0, True)
    var left = Series("k", Column[Int64]([Int64(collision)]))
    var right = Series("k", Column[Int64]([0]))
    var inner = direct_hash_join_rows([left.copy()], [right.copy()], False)
    assert_equal(len(inner[0]), 0)
    var outer = direct_hash_join_rows([left.copy()], [right.copy()], True)
    assert_equal(outer[0], [0])
    assert_equal(outer[1], [-1])


def test_string_keys_are_exact_and_nulls_do_not_match() raises:
    var left = Series(
        "k",
        Column[String](
            ["a", "a much longer string", "a", "", "missing"],
            [True, True, False, True, True],
        ),
    )
    var right = Series(
        "k",
        Column[String](
            ["a much longer string", "a", "", "a"],
            [True, True, True, False],
        ),
    )
    var rows = direct_hash_join_rows([left.copy()], [right.copy()], True)
    assert_equal(rows[0], [0, 1, 2, 3, 4])
    assert_equal(rows[1], [1, 0, -1, 2, -1])


def test_composite_float_nan_and_signed_zero() raises:
    var nan = Float64(0) / Float64(0)
    var left_float = Series("f", Column[Float64]([nan, -0.0, 0.0, 1.0]))
    var right_float = Series("f", Column[Float64]([nan, 0.0, -0.0, 1.0]))
    var left_text = Series("s", Column[String](["x", "x", "y", "z"]))
    var right_text = Series("s", Column[String](["x", "x", "y", "z"]))
    var rows = direct_hash_join_rows(
        [left_float.copy(), left_text.copy()],
        [right_float.copy(), right_text.copy()],
        False,
    )
    assert_equal(rows[0], [0, 1, 2, 3])
    assert_equal(rows[1], [0, 1, 2, 3])


def test_large_sparse_join_uses_exact_index_in_frame() raises:
    var left_keys = List[Int64]()
    var left_values = List[Int64]()
    for i in range(140_000):
        left_keys.append(Int64(i * 17))
        left_values.append(Int64(i))
    var right_keys = List[Int64]()
    var right_values = List[Int64]()
    for i in range(70_000):
        right_keys.append(Int64(i * 17))
        right_values.append(Int64(i * 3))
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
    var inner = left.join(right, "k")
    assert_equal(inner.height(), 70_000)
    for i in range(inner.height()):
        assert_equal(inner.item(i, "v").int64(), Int64(i))
        assert_equal(inner.item(i, "r").int64(), Int64(i * 3))
    var outer = left.join(right, "k", how="left")
    assert_equal(outer.height(), 140_000)
    for i in range(outer.height()):
        assert_equal(outer.item(i, "v").int64(), Int64(i))
        if i < 70_000:
            assert_equal(outer.item(i, "r").int64(), Int64(i * 3))
        else:
            assert_equal(outer.item(i, "r").is_null(), True)


def test_direct_join_gathers_nonidentity_rows_at_equal_output_height() raises:
    var left_keys = List[Int64]()
    var left_values = List[Int64]()
    var right_keys = List[Int64]()
    var right_values = List[Int64]()
    for i in range(140_000):
        left_keys.append(Int64(i))
        left_values.append(Int64(i))
        right_keys.append(Int64(i // 2))
        right_values.append(Int64(i))
    var key = Series("k", Column[Int64](left_keys^))
    var value = Series("v", Column[Int64](left_values^))
    var key_chunks = List[Series]()
    var value_chunks = List[Series]()
    for chunk in range(16):
        key_chunks.append(key.slice(chunk * 8750, 8750))
        value_chunks.append(value.slice(chunk * 8750, 8750))
    var left = DataFrame(
        [
            Series._from_chunks(key_chunks^),
            Series._from_chunks(value_chunks^),
        ]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int64](right_keys^)),
            Series("r", Column[Int64](right_values^)),
        ]
    )
    var inner = left.join(right, "k")
    assert_equal(inner.height(), left.height())
    for i in range(70_000):
        assert_equal(inner.item(2 * i, "v").int64(), Int64(i))
        assert_equal(inner.item(2 * i + 1, "v").int64(), Int64(i))
        assert_equal(inner.item(2 * i, "r").int64(), Int64(2 * i))
        assert_equal(inner.item(2 * i + 1, "r").int64(), Int64(2 * i + 1))


def test_high_cardinality_right_join_preserves_right_order() raises:
    var left_keys = List[Int64]()
    var left_values = List[Int64]()
    var right_keys = List[Int64]()
    var right_values = List[Int64]()
    for i in range(140_000):
        left_keys.append(Int64((i // 2) * 17))
        left_values.append(Int64(i))
        right_keys.append(Int64(i * 17))
        right_values.append(Int64(i))
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
    var result = left.join(right, "k", how="right")
    assert_equal(result.height(), 210_000)
    for i in range(70_000):
        assert_equal(result.item(2 * i, "k").int64(), Int64(i * 17))
        assert_equal(result.item(2 * i, "v").int64(), Int64(2 * i))
        assert_equal(result.item(2 * i + 1, "v").int64(), Int64(2 * i + 1))
        assert_equal(result.item(2 * i, "r").int64(), Int64(i))
        assert_equal(result.item(2 * i + 1, "r").int64(), Int64(i))
    for i in range(70_000, 140_000):
        var at = i + 70_000
        assert_equal(result.item(at, "k").int64(), Int64(i * 17))
        assert_equal(result.item(at, "r").int64(), Int64(i))
        assert_equal(result.item(at, "v").is_null(), True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
