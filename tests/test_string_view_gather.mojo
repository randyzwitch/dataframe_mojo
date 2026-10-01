"""Gathered string rows are views into the source's bytes (#375).

`take`, `take_or_null` and the parallel gathers behind filters, joins and
sorts return view storage that shares the source buffer instead of copying
every byte. The values must be exactly the copied ones, and every consumer
must read them: comparisons, grouping, sorting, concat, CSV and Arrow.
"""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import (
    Column,
    DataFrame,
    Series,
    StringColumn,
    col,
    concat,
    lit,
    read_csv,
    write_csv,
)
from dataframe.arrow import ArrowArray, ArrowSchema, export_arrow, import_arrow

comptime CSV_PATH = "/tmp/dataframe_mojo_string_view_gather.csv"


def set_threads(n: Int):
    var name = String("DATAFRAME_THREADS")
    var value = String(n)
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


def words(rows: Int) -> List[String]:
    var pool: List[String] = [
        "",
        "short",
        "exactly12byt",
        "thirteen byte",
        "a much longer value that is stored outside the view",
        "ünïcode välue spanning more than twelve bytes",
    ]
    var out = List[String](capacity=rows)
    for i in range(rows):
        out.append(pool[(i * 7) % len(pool)] + String(i % 13))
    return out^


def frame(rows: Int) raises -> DataFrame:
    var valid = List[Bool](capacity=rows)
    var keys = List[Int64](capacity=rows)
    for i in range(rows):
        valid.append(i % 11 != 3)
        keys.append(Int64(i % 50))
    return DataFrame(
        [
            Series("s", StringColumn(words(rows), valid)),
            Series("k", Column[Int64](keys^)),
        ]
    )


def test_take_shares_the_source_buffer() raises:
    var data = frame(1000)
    var rows: List[Int] = [5, 999, 0, 3, 3, 500, 14]
    var taken = data.column("s").take(rows)
    ref strings = taken._data[StringColumn]
    assert_true(strings._is_view_storage())
    for k in range(len(rows)):
        var want = data.column("s").get(rows[k])
        var got = taken.get(k)
        assert_equal(got.is_null(), want.is_null())
        if not want.is_null():
            assert_equal(got.string(), want.string())
    # take_or_null: -1 is a null row.
    var with_missing: List[Int] = [-1, 4, -1, 13]
    var outer = (
        data.column("s")._data[StringColumn].take_or_null(with_missing, "")
    )
    assert_true(outer._get(1) == data.column("s").get(4).string())
    assert_true(not outer._valid(0) and not outer._valid(2))


def test_consumers_read_gathered_views() raises:
    set_threads(8)
    var data = frame(200_000)
    var picked = data.filter(col("k") < lit(Int64(20)))
    var expected = data.select_exprs([col("s"), col("k")]).filter(
        col("k") < lit(Int64(20))
    )
    assert_true(picked.equals(expected))
    # Comparison, grouping and sorting on the view column.
    assert_true(
        picked.filter(col("s") == lit("short3")).height()
        == expected.filter(col("s") == lit("short3")).height()
    )
    var grouped = picked.group_by("s", maintain_order=True).agg(
        [col("k").sum()]
    )
    assert_true(grouped.height() > 0)
    var sorted = picked.sort(["s", "k"])
    assert_true(sorted.height() == picked.height())
    # concat of a view column with an offsets column.
    var both = concat([picked.copy(), data.slice(0, 10)])
    assert_equal(both.height(), picked.height() + 10)
    # CSV and Arrow round trips.
    var small = picked.slice(0, 500)
    write_csv(small, CSV_PATH)
    var reread = read_csv(CSV_PATH)
    assert_equal(reread.height(), 500)
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow(small, array, schema)
    var back = import_arrow(array, schema)
    assert_true(back.equals(small))


def test_join_output_strings() raises:
    set_threads(8)
    var left = frame(50_000)
    var right = DataFrame(
        [
            Series("k", Column[Int64]([Int64(i) for i in range(50)])),
            Series(
                "label",
                StringColumn(["label number " + String(i) for i in range(50)]),
            ),
        ]
    )
    var joined = left.join(right, "k", "inner")
    assert_equal(joined.height(), 50_000)
    for i in range(0, 50_000, 997):
        var k = joined.column("k").get(i).int64()
        assert_equal(
            joined.column("label").get(i).string(),
            "label number " + String(k),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
