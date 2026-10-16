"""Join progression, identity gathering, and stable parallel CSR contracts."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, StringColumn
from dataframe.frame import (
    _dense_right_int64_rows,
    _parallel_group_rows,
    _parallel_join_rows,
    _JoinCountJob,
)
from dataframe.join_hash import int64_progression
from dataframe.parallel import worker_count
from tests.support.lifecycle_jobs import WorkerSetting


def test_dense_join_progression_eligibility_nullable_chunk_probes_and_integer_edges_have_literal_rows() raises:
    for base in [Int64(10), Int64.MAX - 2]:
        var rk: List[Int64] = [base, base + 1, base + 2]
        var lk: List[Int64] = [
            base + 2,
            base - 1,
            base + 1,
            base + 2,
            base,
            base,
        ]
        var lv: List[Bool] = [True, True, True, True, False, True]
        var left_key = Series("k", Column[Int64](lk.copy(), lv.copy()))
        var right_key = Series("k", Column[Int64](rk.copy()))
        for chunked in [False, True]:
            var l = left_key.copy()
            var r = right_key.copy()
            if chunked:
                l = Series._from_chunks([l.slice(0, 1), l.slice(1, 5)])
                r = Series._from_chunks([r.slice(0, 2), r.slice(2, 1)])
            for unmatched in [False, True]:
                var selected = _dense_right_int64_rows(l, r, unmatched)
                assert_true(selected[0])
                var expected_left: List[Int] = [0, 2, 3, 5]
                var expected_right: List[Int] = [2, 1, 2, 0]
                if unmatched:
                    expected_left = [0, 1, 2, 3, 4, 5]
                    expected_right = [2, -1, 1, 2, -1, 0]
                assert_equal(selected[1], expected_left)
                assert_equal(selected[2], expected_right)
                var left = DataFrame(
                    [
                        l.copy(),
                        Series("left", Column[Int64]([0, 1, 2, 3, 4, 5])),
                    ]
                )
                var right = DataFrame(
                    [
                        r.copy(),
                        Series("right", Column[String](["zero", "one", "two"])),
                    ]
                )
                var actual = left.join(
                    right, "k", how="left" if unmatched else "inner"
                )
                assert_equal(
                    actual.columns(), List[String](["k", "left", "right"])
                )
                assert_equal(actual.height(), len(expected_left))
                for row in range(actual.height()):
                    var i = expected_left[row]
                    var j = expected_right[row]
                    assert_equal(actual.item(row, "left").int64(), Int64(i))
                    assert_equal(actual.item(row, "k").is_null(), not lv[i])
                    if lv[i]:
                        assert_equal(actual.item(row, "k").int64(), lk[i])
                    assert_equal(actual.item(row, "right").is_null(), j < 0)
                    if j >= 0:
                        assert_equal(
                            actual.item(row, "right").string(),
                            List[String](["zero", "one", "two"])[j],
                        )
        for bad in [
            Series("k", Column[Int64]([base, base + 2, base + 1])),
            Series("k", Column[Int64]([base, base, base + 1])),
            Series("k", Column[Int64]([base + 2, base + 1, base])),
            Series("k", Column[Int64](rk.copy(), [True, False, True])),
        ]:
            assert_true(not _dense_right_int64_rows(left_key, bad)[0])
    var probe = Series("k", Column[Int64]([0, 1, 2]))
    var dates = probe.with_dtype(DataType.DATE)
    assert_true(not _dense_right_int64_rows(probe, dates)[0])


def test_progression_found_once_per_buffer_holds_for_its_windows() raises:
    var values = List[Int64]()
    for i in range(10):
        values.append(Int64(100 + 3 * i))
    var key = Series("k", Column[Int64](values^))
    var found = int64_progression(key)
    assert_true(found[0])
    assert_equal(found[1], Int64(100))
    assert_equal(found[2], UInt64(3))
    assert_equal(key._data[Column[Int64]]._facts[].progression, 1)
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
    assert_equal(irregular._data[Column[Int64]]._facts[].progression, 2)
    assert_true(int64_progression(irregular.slice(0, 2))[0])
    assert_true(not int64_progression(irregular.slice(1, 3))[0])
    # A null anywhere in the buffer: never a progression.
    var nulls = Series(
        "k", Column[Int64]([1, 2, 3, 4], [True, True, True, False])
    )
    assert_true(not int64_progression(nulls)[0])
    assert_true(int64_progression(nulls.slice(0, 3))[0])


def test_parallel_unique_build_keys_reuse_all_left_buffers_and_preserve_every_typed_cell() raises:
    var setting = WorkerSetting(4)
    var n = 196609
    var groups = 4099
    assert_true(worker_count(n) > 1)
    var keys = List[Int64]()
    var ids = List[Int64]()
    var texts = List[String]()
    var valid = List[Bool]()
    for row in range(n + 7):
        keys.append(Int64(row % groups))
        ids.append(Int64(row))
        texts.append("external nullable payload " + String(row % 11))
        valid.append(row % 13 != 2)
    var left = DataFrame(
        [
            Series("k", Column[Int64](keys^)).slice(7, n),
            Series("id", Column[Int64](ids^)).slice(7, n),
            Series("text", Column[String](texts^, valid^)).slice(7, n),
        ]
    )
    for shuffled in [False, True]:
        var build_keys = List[Int64]()
        var payload = List[Int64]()
        for row in range(groups):
            var key = (row * 17) % groups if shuffled else row
            build_keys.append(Int64(key))
            payload.append(Int64(key * 3 - 1))
        var right = DataFrame(
            [
                Series("k", Column[Int64](build_keys^)),
                Series("r", Column[Int64](payload^)),
            ]
        )
        assert_equal(
            _dense_right_int64_rows(left.column("k"), right.column("k"))[0],
            not shuffled,
        )
        var actual = left.join(right, "k")
        assert_equal(actual.height(), n)
        assert_true(
            actual.column("id")
            .int64()
            ._shares_buffers_with(left.column("id").int64())
        )
        assert_true(
            actual.column("text")
            ._data[StringColumn]
            ._shares_buffers_with(left.column("text")._data[StringColumn])
        )
        for row in range(n):
            var source = row + 7
            assert_equal(actual.item(row, "k").int64(), Int64(source % groups))
            assert_equal(actual.item(row, "id").int64(), Int64(source))
            assert_equal(
                actual.item(row, "r").int64(), Int64((source % groups) * 3 - 1)
            )
            assert_equal(actual.item(row, "text").is_null(), source % 13 == 2)
            if source % 13 != 2:
                assert_equal(
                    actual.item(row, "text").string(),
                    "external nullable payload " + String(source % 11),
                )
    _ = setting^


def test_parallel_csr_and_join_fanout_preserve_scalar_group_row_order_and_check_count_overflow() raises:
    var setting = WorkerSetting(4)
    for groups in [0, 1, 3, 33, 1009]:
        var ids = List[Int]()
        for row in range(4003):
            ids.append(
                -1 if groups == 0 or row % 7 == 0 else (row * 53) % groups
            )
        var starts = List[Int](length=groups + 1, fill=0)
        var expected = List[Int]()
        for group in range(groups):
            for row in range(len(ids)):
                if ids[row] == group:
                    expected.append(row)
            starts[group + 1] = len(expected)
        for workers in [1, 2, 4, 8]:
            assert_equal(_parallel_group_rows(ids, starts, workers), expected)
            for unmatched in [False, True]:
                var result = _parallel_join_rows(
                    ids, starts, expected, workers, unmatched
                )
                var at = 0
                for row in range(len(ids)):
                    var matched = False
                    for other in range(len(ids)):
                        if ids[row] >= 0 and ids[row] == ids[other]:
                            assert_equal(result[0][at], row)
                            assert_equal(result[1][at], other)
                            at += 1
                            matched = True
                    if unmatched and not matched:
                        assert_equal(result[0][at], row)
                        assert_equal(result[1][at], -1)
                        at += 1
                assert_equal(len(result[0]), at)
                assert_equal(len(result[1]), at)
    for unmatched in [False, True]:
        var job = _JoinCountJob(
            ArcPointer(List[Int]([0, 0])),
            ArcPointer(List[Int]([0, Int.MAX])),
            0,
            2,
            unmatched,
        )
        with assert_raises(contains="Join output row count overflows"):
            job.run()
    _ = setting^
