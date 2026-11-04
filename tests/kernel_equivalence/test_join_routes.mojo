"""Progression facts and joins built on the smaller left side."""
# contract-group: table.join
# contract-group: kernel.row_ops
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, StringColumn
from dataframe.kernels.join_keys import int64_progression
from dataframe.join_type import JOIN_LEFT
from dataframe.parallel import worker_count
from tests.support.lifecycle_jobs import WorkerSetting


def test_progression_found_once_per_buffer_holds_for_its_windows() raises:
    var values = List[Int64]()
    for i in range(10):
        values.append(Int64(100 + 3 * i))
    var key = Series("k", Column[Int64](values^))
    var found = int64_progression(key)
    assert_true(found[0])
    assert_equal(found[1], Int64(100))
    assert_equal(found[2], UInt64(3))
    assert_equal(key._data[Column[Int64]]._facts[].progression_state(), 1)
    # A window onto the buffer starts at its own first value.
    var window = int64_progression(key.slice(2, 5))
    assert_true(window[0])
    assert_equal(window[1], Int64(106))
    assert_equal(window[2], UInt64(3))
    # Windows of one progression are one only where each meets the last.
    var joined = Series._from_chunks([key.slice(0, 4), key.slice(4, 6)])
    var whole = int64_progression(joined)
    assert_true(whole[0])
    assert_equal(whole[1], Int64(100))
    assert_equal(whole[2], UInt64(3))
    var gap = Series._from_chunks([key.slice(0, 4), key.slice(5, 5)])
    assert_true(not int64_progression(gap)[0])
    var swapped = Series._from_chunks([key.slice(5, 5), key.slice(0, 5)])
    assert_true(not int64_progression(swapped)[0])
    var single = Series._from_chunks([key.slice(0, 1), key.slice(1, 9)])
    assert_true(int64_progression(single)[0])
    # A buffer that is not one is known not to be; a window of it that is
    # one is still checked on its own rows.
    var irregular = Series("k", Column[Int64]([1, 2, 4, 5]))
    assert_true(not int64_progression(irregular)[0])
    assert_equal(irregular._data[Column[Int64]]._facts[].progression_state(), 2)
    assert_true(int64_progression(irregular.slice(0, 2))[0])
    assert_true(not int64_progression(irregular.slice(1, 3))[0])
    # A null anywhere in the buffer: never a progression.
    var nulls = Series(
        "k", Column[Int64]([1, 2, 3, 4], [True, True, True, False])
    )
    assert_true(not int64_progression(nulls)[0])
    assert_true(int64_progression(nulls.slice(0, 3))[0])


def test_unordered_left_build_over_many_chunks_gathers_every_left_row() raises:
    # A left side smaller than the right builds the hash table; an inner
    # join whose consumers ignore order then gets its pairs in the order the
    # right side probed, not in left row order. A left of many chunks must
    # still gather each pair's own left row.
    var right_keys = List[Int64]()
    var right_values = List[Int64]()
    for i in range(20_000):
        var key = Int64((i * 7_919) % 20_011)
        right_keys.append(key)
        right_values.append(key * 7)
    var right = DataFrame(
        [
            Series("k", Column[Int64](right_keys^)),
            Series("r", Column[Int64](right_values^)),
        ]
    )
    var keys = List[Series]()
    var values = List[Series]()
    for c in range(32):
        var chunk_keys = List[Int64]()
        var chunk_values = List[Int64]()
        for i in range(12):
            var at = ((c * 12 + i) * 104_729) % 20_000
            var key = Int64((at * 7_919) % 20_011)
            chunk_keys.append(key)
            chunk_values.append(key * 10)
        keys.append(Series("k", Column[Int64](chunk_keys^)))
        values.append(Series("l", Column[Int64](chunk_values^)))
    var left = DataFrame(
        [Series._from_chunks(keys), Series._from_chunks(values)]
    )
    var joined = left._join_impl(
        right, left_on=["k"], right_on=["k"], keep_order=False
    )
    assert_equal(joined.height(), 384)
    for row in range(joined.height()):
        var key = joined.item(row, "k").int64()
        assert_equal(joined.item(row, "l").int64(), key * 10)
        assert_equal(joined.item(row, "r").int64(), key * 7)


def test_unordered_left_join_on_a_smaller_left_emits_unmatched_left_rows_after_the_pairs() raises:
    # A left join whose left side is the smaller builds on it and, when its
    # consumers ignore order, returns the probed pairs and then the left rows
    # no right row matched (a null key and keys the right does not hold),
    # each once with a null right payload.
    var left_keys = List[Int64]()
    var left_valid = List[Bool]()
    var left_ids = List[Int64]()
    for i in range(50):
        left_keys.append(Int64(i % 13))
        left_valid.append(i != 7)
        left_ids.append(Int64(i))
    var left = DataFrame(
        [
            Series("k", Column[Int64](left_keys^, left_valid)),
            Series("l", Column[Int64](left_ids^)),
        ]
    )
    var right_keys = List[Int64]()
    var right_ids = List[Int64]()
    for j in range(6000):
        right_keys.append(Int64(j % 10))
        right_ids.append(Int64(j))
    var right = DataFrame(
        [
            Series("k", Column[Int64](right_keys^)),
            Series("r", Column[Int64](right_ids^)),
        ]
    )
    var expected_rows = 0
    var expected_unmatched = 0
    for i in range(50):
        if i != 7 and i % 13 < 10:
            expected_rows += 600
        else:
            expected_rows += 1
            expected_unmatched += 1
    for ordered in [False, True]:
        var joined = left._join_impl(
            right,
            left_on=["k"],
            right_on=["k"],
            how=JOIN_LEFT,
            keep_order=ordered,
        )
        assert_equal(joined.height(), expected_rows)
        var seen = List[Int](length=50, fill=0)
        var unmatched = 0
        for row in range(joined.height()):
            var id = Int(joined.item(row, "l").int64())
            seen[id] += 1
            if joined.item(row, "r").is_null():
                unmatched += 1
                assert_true(id == 7 or id % 13 >= 10)
            else:
                assert_equal(Int(joined.item(row, "r").int64()) % 10, id % 13)
                assert_equal(Int(joined.item(row, "k").int64()), Int(id % 13))
        assert_equal(unmatched, expected_unmatched)
        for id in range(50):
            assert_true(seen[id] >= 1)
        if ordered:
            # Left-major order: each left row's rows are together, ascending.
            var last = Int64(-1)
            for row in range(joined.height()):
                var id = joined.item(row, "l").int64()
                assert_true(id >= last)
                last = id

