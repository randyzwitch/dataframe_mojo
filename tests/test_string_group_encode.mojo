"""Grouping one string key on every worker (#381): a dominant value, nulls
and many distinct values must give first-occurrence groups with exact
counts, and lazy plans, whichever engine the streaming gate picks, must
equal the eager group-by.
"""
from std.collections import Dict
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataFrame, Series, StringColumn, col


def set_threads(n: Int):
    var name = String("DATAFRAME_THREADS")
    var value = String(n)
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


def frame(rows: Int, distinct: Int) raises -> DataFrame:
    var values = List[String](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for i in range(rows):
        valid.append(i % 17 != 3)
        if i % 3 != 0:
            values.append("")
        else:
            # Short values stay inline in a view; long ones do not.
            var v = String((i * 7919) % distinct)
            values.append(v if i % 2 == 0 else "a longer phrase number " + v)
    var ids = List[Int64](capacity=rows)
    for i in range(rows):
        ids.append(Int64(i))
    return DataFrame(
        [
            Series("s", StringColumn(values, valid)),
            Series("i", Column[Int64](ids^)),
        ]
    )


def check(data: DataFrame, grouped: DataFrame) raises:
    var want = Dict[String, Int]()
    var order = List[String]()
    var column = data.column("s")
    for i in range(data.height()):
        var cell = column.get(i)
        var key = String("<null>") if cell.is_null() else "=" + cell.string()
        if key not in want:
            want[key] = 0
            order.append(key)
        want[key] += 1
    assert_equal(grouped.height(), len(order))
    for g in range(grouped.height()):
        var cell = grouped.column("s").get(g)
        var key = String("<null>") if cell.is_null() else "=" + cell.string()
        assert_equal(key, order[g])
        assert_equal(Int(grouped.column("c").get(g).int64()), want[key])


def test_skewed_key_on_every_worker() raises:
    set_threads(8)
    var data = frame(200_000, 20_000)
    var grouped = data.group_by("s", maintain_order=True).agg(
        [col("i").len().alias("c")]
    )
    check(data, grouped)


def test_lazy_matches_eager() raises:
    set_threads(8)
    for distinct in [50, 20_000, 1_000_000]:
        var data = frame(150_000, distinct)
        var eager = data.group_by("s", maintain_order=True).agg(
            [col("i").len().alias("c")]
        )
        var lazy = (
            data.lazy()
            .group_by("s", maintain_order=True)
            .agg([col("i").len().alias("c")])
            .collect()
        )
        check(data, lazy)
        assert_true(lazy.equals(eager))
        var kept = data.filter(col("s").ne(""))
        var filtered = (
            data.lazy()
            .filter(col("s").ne(""))
            .group_by("s", maintain_order=True)
            .agg([col("i").len().alias("c")])
            .collect()
        )
        check(kept, filtered)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
