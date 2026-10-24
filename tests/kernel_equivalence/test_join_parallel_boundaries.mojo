"""Large join index chains and nonidentity assembly obey exact row positions."""
# contract-group: kernel.order_keys
from std.memory import Pointer
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, Series
from dataframe.frame import (
    _RangeIndexBuildJob,
    _parallel_range_build_workers,
    _bounded_int64_join_rows,
    _dense_right_int64_rows,
    _PROGRESSION_PARALLEL_KEY_BYTES,
)
from dataframe.join_fuse import FusedJoin
from dataframe.join_hash import direct_hash_join_rows
from dataframe.parallel import Pool, worker_count
from tests.support.join_worker_setting import JoinWorkers


def test_parallel_bounded_chain_build_and_large_progression_probe_keep_every_duplicate_null_and_position() raises:
    var setting = JoinWorkers(4)
    var domain = 2000000
    var n = domain + 100003
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(n):
        values.append(Int64(row % domain))
        valid.append(row % 97 != 2)
    var build = Series("k", Column[Int64](values^, valid))
    assert_true(worker_count(n) > 1)
    var selected_workers = _parallel_range_build_workers(n, domain)
    assert_true(selected_workers >= 1 and selected_workers <= 4)
    # Explicit four-way slot ownership exercises the parallel chain builder
    # even on a CI host whose physical-core policy selects one worker.
    var heads = List[Int](length=domain, fill=-1)
    var next_rows = List[Int](length=n, fill=-1)
    var jobs = List[_RangeIndexBuildJob]()
    for worker in range(4):
        jobs.append(
            _RangeIndexBuildJob(
                build._data[Column[Int64]],
                Int(Pointer(to=heads)),
                Int(Pointer(to=next_rows)),
                0,
                domain * worker // 4,
                domain * (worker + 1) // 4,
            )
        )
    var pool = Pool(4)
    pool.run(jobs)
    pool.release()
    for key in range(domain):
        var first = (
            key if key % 97 != 2 else key + domain if key + domain < n
            and (key + domain) % 97 != 2 else -1
        )
        var second = (
            key + domain if first == key
            and key + domain < n
            and (key + domain) % 97 != 2 else -1
        )
        assert_equal(heads[key], first)
        if first >= 0:
            assert_equal(next_rows[first], second)
        if second >= 0:
            assert_equal(next_rows[second], -1)
    var probes: List[Int64] = [
        0,
        2,
        100002,
        100003,
        Int64(domain - 1),
        Int64(domain),
        Int64.MAX,
    ]
    var probe = Series("k", Column[Int64](probes.copy()))
    for outer in [False, True]:
        var expected_left = List[Int]()
        var expected_right = List[Int]()
        for i in range(len(probes)):
            var found = False
            for j in range(n):
                if j % 97 != 2 and probes[i] == Int64(j % domain):
                    expected_left.append(i)
                    expected_right.append(j)
                    found = True
            if outer and not found:
                expected_left.append(i)
                expected_right.append(-1)
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        var unfused = FusedJoin()
        assert_true(
            _bounded_int64_join_rows(
                probe, build, outer, left_rows, right_rows, unfused
            )
        )
        assert_equal(left_rows, expected_left)
        assert_equal(right_rows, expected_right)
    var rows = _PROGRESSION_PARALLEL_KEY_BYTES // 8 + 3
    var keys = List[Int64]()
    var bits = List[Bool]()
    for row in range(rows + 7):
        keys.append(Int64((row % 6 - 2) * 60))
        bits.append(row % 29 != 2)
    var flat = Series("k", Column[Int64](keys^, bits)).slice(7, rows)
    var progression = Series("k", Column[Int64]([-120, -60, 0, 60]))
    for chunked in [False, True]:
        var source = flat.copy()
        if chunked:
            source = Series._from_chunks(
                [flat.slice(0, 770001), flat.slice(770001, rows - 770001)]
            )
        var actual = _dense_right_int64_rows(source, progression, True)
        assert_true(actual[0])
        assert_equal(len(actual[1]), rows)
        assert_equal(len(actual[2]), rows)
        for row in range(rows):
            assert_equal(actual[1][row], row)
            var physical = row + 7
            assert_equal(
                actual[2][row],
                physical % 6 if physical % 29 != 2 and physical % 6 < 4 else -1,
            )
    _ = setting^


def test_equal_output_height_does_not_imply_identity_gather_and_parallel_full_join_appends_unmatched_rows() raises:
    var setting = JoinWorkers(4)
    var n = 196608
    var keys = List[Int64]()
    var ids = List[Int64]()
    var right_keys = List[Int64]()
    for row in range(n):
        keys.append(Int64(row))
        ids.append(Int64(row))
        right_keys.append(Int64(row // 2))
    var key = Series("k", Column[Int64](keys^))
    var id = Series("l", Column[Int64](ids.copy()))
    var key_chunks = List[Series]()
    var id_chunks = List[Series]()
    for chunk in range(16):
        key_chunks.append(key.slice(chunk * (n // 16), n // 16))
        id_chunks.append(id.slice(chunk * (n // 16), n // 16))
    var left = DataFrame(
        [Series._from_chunks(key_chunks), Series._from_chunks(id_chunks)]
    )
    var right = DataFrame(
        [
            Series("k", Column[Int64](right_keys^)),
            Series("r", Column[Int64](ids^)),
        ]
    )
    var direct = direct_hash_join_rows(
        [left.column("k")], [right.column("k")], False, omit_identity=True
    )
    assert_true(not direct[2])
    assert_equal(len(direct[0]), n)
    assert_equal(len(direct[1]), n)
    for row in range(n):
        assert_equal(direct[0][row], row // 2)
        assert_equal(direct[1][row], row)
    for how in ["inner", "right", "full"]:
        var actual = left.join(right, "k", how)
        assert_equal(actual.height(), n + n // 2 if how == "full" else n)
        for row in range(n):
            assert_equal(actual.item(row, "k").int64(), Int64(row // 2))
            assert_equal(actual.item(row, "l").int64(), Int64(row // 2))
            assert_equal(actual.item(row, "r").int64(), Int64(row))
        if how == "full":
            for row in range(n // 2):
                assert_equal(
                    actual.item(n + row, "k").int64(), Int64(n // 2 + row)
                )
                assert_equal(
                    actual.item(n + row, "l").int64(), Int64(n // 2 + row)
                )
                assert_true(actual.item(n + row, "r").is_null())
    _ = setting^
