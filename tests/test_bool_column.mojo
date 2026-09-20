"""Bit-packed Boolean columns: layout, windows, and copy-on-write."""
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    Series,
    col,
    concat,
    when,
)


def sample() raises -> BoolColumn:
    var values = List[Bool]()
    var valid = List[Bool]()
    for i in range(21):
        values.append(i % 3 == 0)
        valid.append(i % 7 != 5)
    return BoolColumn(values^, valid^)


def expected(column: BoolColumn) raises -> List[String]:
    var out = List[String]()
    for i in range(len(column)):
        out.append("null" if column.is_null(i) else String(column.value(i)))
    return out^


def test_values_are_bit_packed() raises:
    var column = sample()
    assert_equal(len(column._data[]), 3)  # 21 rows -> 3 bytes, not 21
    assert_equal(len(column._bits[]), 3)
    # Rows 0, 3, 6 true -> 0b01001001 in the first byte.
    assert_equal(column._data[][0], UInt8(0b01001001))
    assert_equal(column._bits[][0], UInt8(0b11011111))
    assert_equal(column.true_count(), 7)
    assert_equal(column.null_count(), 3)


def test_windows_share_buffers_at_every_offset() raises:
    var column = sample()
    var all = expected(column)
    for start in range(len(column) + 1):
        for length in range(len(column) - start + 1):
            var window = column.slice(start, length)
            assert_true(window._shares_buffers_with(column))
            assert_equal(len(window), length)
            var rows = expected(window)
            for i in range(length):
                assert_equal(rows[i], all[start + i])
            var compact = window._compact()
            assert_false(compact._shares_buffers_with(column))
            assert_equal(expected(compact), rows)


def test_append_is_copy_on_write_and_bit_shifted() raises:
    var column = sample()
    var before = expected(column)
    for start in range(9):
        for extra in range(11):
            var left = column.slice(start, len(column) - start)
            var right = column.slice(0, extra)
            var joined = left.copy()
            joined._append_column(right)
            var want = expected(left)
            for value in expected(right):
                want.append(value)
            assert_equal(expected(joined), want)
            assert_equal(expected(column), before)


def test_take_and_broadcast() raises:
    var column = sample()
    assert_equal(
        expected(column.take([0, 5, 6, 0])), ["True", "null", "True", "True"]
    )
    assert_equal(
        expected(column.take_or_null([-1, 3, -1], True)),
        ["null", "True", "null"],
    )
    var one = BoolColumn([True])
    assert_equal(one._broadcast(9).true_count(), 9)
    assert_equal(BoolColumn._nulls(9).null_count(), 9)


def test_through_the_frame_api() raises:
    var df = DataFrame(
        [
            Series("b", sample()),
            Series("i", Column[Int64](List[Int64](length=21, fill=1))),
        ]
    )
    assert_equal(df.filter(col("b")).height(), 6)  # 7 true, one of them null
    var r = df.select_exprs(
        [
            (col("b") & True).alias("and_true"),
            (~col("b")).alias("not"),
            col("b").fill_null(False).alias("filled"),
            when(col("b")).then(1).otherwise(0).alias("pick"),
            col("b").any().alias("any"),
            col("b").all().alias("all"),
        ]
    )
    assert_true(r.item(0, "and_true").bool())
    assert_false(r.item(0, "not").bool())
    assert_false(r.item(5, "filled").bool())
    assert_equal(r.item(0, "pick").int64(), 1)
    assert_true(r.item(0, "any").bool())
    assert_false(r.item(0, "all").bool())
    # Round trips: a packed column survives concat, sort, group_by, and unique.
    var stacked = concat([df.slice(0, 5), df.slice(7, 6)])
    assert_equal(stacked.height(), 11)
    assert_equal(stacked.column("b")._data[BoolColumn].to_list()[0], True)
    assert_equal(df.sort("b").item(0, "b").bool(), False)
    assert_equal(df.group_by("b").agg(col("i").sum()).height(), 3)
    assert_equal(df.select(["b"]).unique().height(), 3)
    # A byte-per-value Column[Bool] still works and packs on construction.
    var packed = Series("b", Column[Bool]([True, False], [True, False]))
    assert_equal(len(packed._data[BoolColumn]._data[]), 1)
    assert_true(packed.get(0).bool())
    assert_true(packed.get(1).is_null())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
