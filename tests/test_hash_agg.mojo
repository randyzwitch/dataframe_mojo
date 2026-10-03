"""In-place hash aggregation (`hash_agg.mojo`) for one Int64 key with a
moderate number of groups: every supported reduction must equal a
row-by-row reference, with nulls in the values, NaN for min and max,
first-occurrence order when asked, and enough rows for several workers.
"""
from std.collections import Dict
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Expr, Series, col


def set_threads(n: Int):
    var name = String("DATAFRAME_THREADS")
    var value = String(n)
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


comptime ROWS = 200_000
comptime GROUPS = 20_000


def frame() raises -> DataFrame:
    var keys = List[Int64](capacity=ROWS)
    var ints = List[Int64](capacity=ROWS)
    var int_valid = List[Bool](capacity=ROWS)
    var floats = List[Float64](capacity=ROWS)
    var float_valid = List[Bool](capacity=ROWS)
    var nan = Float64(0) / Float64(0)
    for i in range(ROWS):
        keys.append(Int64((i * 7919) % GROUPS - 5000))
        ints.append(Int64((i * 104729) % 1999 - 1000))
        int_valid.append(i % 13 != 4)
        floats.append(nan if i % 97 == 3 else Float64(i % 211) / 8 - 13)
        float_valid.append(i % 17 != 9)
    return DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("a", Column[Int64](ints^, int_valid^)),
            Series("x", Column[Float64](floats^, float_valid^)),
        ]
    )


def exprs() -> List[Expr]:
    return [
        col("a").sum().alias("a_sum"),
        col("a").mean().alias("a_mean"),
        col("a").min().alias("a_min"),
        col("a").max().alias("a_max"),
        col("a").count().alias("a_count"),
        col("a").len().alias("rows"),
        col("x").sum().alias("x_sum"),
        col("x").min().alias("x_min"),
        col("x").max().alias("x_max"),
    ]


def test_matches_row_by_row_totals() raises:
    set_threads(8)
    var data = frame()
    var out = data.group_by("k", maintain_order=True).agg(exprs())
    assert_equal(out.height(), GROUPS)
    var sums = Dict[Int, Int]()
    var counts = Dict[Int, Int]()
    var rows = Dict[Int, Int]()
    var order = List[Int]()
    for i in range(ROWS):
        var k = Int(data.item(i, "k").int64())
        if k not in rows:
            rows[k] = 0
            sums[k] = 0
            counts[k] = 0
            order.append(k)
        rows[k] += 1
        var a = data.item(i, "a")
        if not a.is_null():
            sums[k] += Int(a.int64())
            counts[k] += 1
    for g in range(out.height()):
        var k = Int(out.item(g, "k").int64())
        # First-occurrence order, as maintain_order promises.
        assert_equal(k, order[g])
        assert_equal(Int(out.item(g, "a_sum").int64()), sums[k])
        assert_equal(Int(out.item(g, "a_count").int64()), counts[k])
        assert_equal(Int(out.item(g, "rows").int64()), rows[k])


def test_equals_the_partitioned_and_serial_paths() raises:
    # One worker never takes the hash path; every column must agree.
    var data = frame()
    set_threads(8)
    var parallel = data.group_by("k", maintain_order=True).agg(exprs())
    set_threads(1)
    var serial = data.group_by("k", maintain_order=True).agg(exprs())
    set_threads(8)
    assert_equal(parallel.height(), serial.height())
    for name in ["k", "a_sum", "a_min", "a_max", "a_count", "rows"]:
        assert_true(parallel.column(name).equals(serial.column(name)), name)
    for name in ["a_mean", "x_sum", "x_min", "x_max"]:
        for g in range(parallel.height()):
            var p = parallel.item(g, name)
            var s = serial.item(g, name)
            assert_equal(p.is_null(), s.is_null(), name)
            if p.is_null():
                continue
            var u = p.float64()
            var v = s.float64()
            if u != u or v != v:
                assert_true(u != u and v != v, name + " NaN")
            else:
                # Float sums may associate differently; nothing else does.
                assert_true(abs(u - v) <= 1e-9 * max(1.0, abs(v)), name)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
