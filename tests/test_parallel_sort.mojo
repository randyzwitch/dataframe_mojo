"""A parallel sort must produce the serial sort's exact order, ties included."""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Series
from dataframe.parallel import MIN_ROWS_PER_WORKER, worker_count
from dataframe.series import _merge_runs, _merge_slice, _sort_range

comptime ROWS = 200_000


def c_string(text: String) -> List[UInt8]:
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    bytes.append(0)
    return bytes^


def set_threads(n: Int):
    var name = c_string("DATAFRAME_THREADS")
    var value = c_string(String(n))
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


def frame(rows: Int, distinct: Int) raises -> DataFrame:
    """Few distinct key values, so most comparisons are ties and any
    stability failure shows up as a different row order."""
    var k = List[Int64](capacity=rows)
    var s = List[String](capacity=rows)
    var f = List[Float64](capacity=rows)
    var valid = List[Bool](capacity=rows)
    var row = List[Int64](capacity=rows)
    var state = UInt64(11)
    for i in range(rows):
        state = state * 6364136223846793005 + 1442695040888963407
        var r = Int((state >> 33) % UInt64(distinct))
        k.append(Int64(r))
        s.append("k" + String(r % 97))
        f.append(Float64(0) / Float64(0) if r % 31 == 0 else Float64(r) / 8)
        valid.append(i % 13 != 4)
        row.append(Int64(i))
    return DataFrame(
        [
            Series("k", Column[Int64](k^, valid.copy())),
            Series("s", Column[String](s^, valid.copy())),
            Series("f", Column[Float64](f^, valid^)),
            Series("row", Column[Int64](row^)),
        ]
    )


def check(df: DataFrame, by: List[String], desc: Bool, nulls_last: Bool) raises:
    set_threads(1)
    assert_equal(worker_count(df.height()), 1)
    var serial = df.arg_sort(by, descending=desc, nulls_last=nulls_last)
    set_threads(32)
    assert_true(worker_count(df.height()) > 1, "parallel sort not taken")
    var parallel = df.arg_sort(by, descending=desc, nulls_last=nulls_last)
    assert_equal(len(serial), len(parallel))
    for i in range(len(serial)):
        if serial[i] != parallel[i]:
            raise Error(
                "row order differs at "
                + String(i)
                + ": serial "
                + String(serial[i])
                + " vs parallel "
                + String(parallel[i])
            )


def test_heavy_ties_keep_input_order() raises:
    # 8 distinct values over 200,000 rows: every run is mostly ties.
    var df = frame(ROWS, 8)
    check(df, ["k"], False, True)
    check(df, ["s"], False, True)
    check(df, ["f"], False, True)


def test_directions_and_null_placement() raises:
    var df = frame(ROWS, 64)
    for desc in [False, True]:
        for nulls_last in [False, True]:
            check(df, ["k"], desc, nulls_last)


def test_composite_keys() raises:
    var df = frame(ROWS, 32)
    check(df, ["k", "s"], False, True)
    check(df, ["s", "f", "k"], True, False)


def test_mixed_directions_match_serial() raises:
    var df = frame(ROWS, 24)
    set_threads(1)
    var serial = df.sort(
        ["k", "s"], descending=[True, False], nulls_last=[False, True]
    )
    set_threads(32)
    var parallel = df.sort(
        ["k", "s"], descending=[True, False], nulls_last=[False, True]
    )
    assert_true(serial.equals(parallel))


def test_all_equal_keys_is_the_identity_permutation() raises:
    var rows = 4 * MIN_ROWS_PER_WORKER
    var same = List[Int64](length=rows, fill=1)
    var df = DataFrame([Series("k", Column[Int64](same^))])
    set_threads(32)
    var order = df.arg_sort(["k"])
    for i in range(rows):
        assert_equal(order[i], i)


def test_top_k_and_bottom_k_match_serial() raises:
    var df = frame(ROWS, 50)
    set_threads(1)
    var top_serial = df.top_k(20, "k")
    var bottom_serial = df.bottom_k(20, "k")
    set_threads(32)
    assert_true(top_serial.equals(df.top_k(20, "k")))
    assert_true(bottom_serial.equals(df.bottom_k(20, "k")))


def test_merge_slices_reproduce_the_whole_merge() raises:
    """Splitting one merge into output slices must give the same array as
    merging it in one go, for every slice count and every shape of the two
    runs -- including a slice boundary landing on the first or last output,
    and one run being empty or much longer than the other.

    This is the co-rank directly: the sort suites only reach it through a
    full sort, where a wrong split would still usually produce sorted
    output and so could pass on all but the unlucky input.
    """
    var shapes: List[List[Int]] = [
        [8, 8],
        [1, 1],
        [1, 15],
        [15, 1],
        [0, 9],
        [9, 0],
        [3, 100],
        [100, 3],
        [64, 64],
        [37, 41],
    ]
    for shape in shapes:
        var left_len = shape[0]
        var right_len = shape[1]
        var total = left_len + right_len
        if total < 1:
            continue
        # Interleave so neither run is uniformly ahead of the other, and
        # repeat values so equal ranks span the split.
        var key = List[Int](length=total, fill=0)
        for i in range(left_len):
            key[i] = (i * 2) % 7
        for i in range(right_len):
            key[left_len + i] = (i * 3) % 7
        var ranks: List[List[Int]] = [key^]

        # Each run must already be sorted for a merge to be defined.
        var source = List[Int](length=total, fill=0)
        var left_rows = _sort_range(ranks, 0, left_len)
        var right_rows = _sort_range(ranks, left_len, total)
        for i in range(len(left_rows)):
            source[i] = left_rows[i]
        for i in range(len(right_rows)):
            source[left_len + i] = right_rows[i]

        var whole = List[Int](length=total, fill=-1)
        _merge_runs(ranks, source, whole, 0, left_len, total)

        for cuts in range(1, total + 2):
            var sliced = List[Int](length=total, fill=-1)
            var s = 0
            while s < cuts:
                var first = (total * s) // cuts
                var last = (total * (s + 1)) // cuts
                if last > first:
                    _merge_slice(
                        ranks, source, sliced, 0, left_len, total, first, last
                    )
                s += 1
            for i in range(total):
                assert_equal(
                    sliced[i],
                    whole[i],
                    msg=(
                        "shape "
                        + String(left_len)
                        + "/"
                        + String(right_len)
                        + " cuts "
                        + String(cuts)
                        + " at "
                        + String(i)
                    ),
                )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
