"""List and struct columns: construction, access, and the type-independent
operations (take, slice, append, equals, nulls, display)."""
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, DataType, Series, StringColumn
from dataframe.nested_column import ListColumn, StructColumn


def ints(values: List[Int64]) -> Series:
    return Series("item", Column[Int64](values.copy()))


def sample_lists() raises -> Series:
    # [1, 2], [], null, [3]
    var rows = List[Series]()
    rows.append(ints([1, 2]))
    rows.append(ints(List[Int64]()))
    rows.append(ints(List[Int64]()))
    rows.append(ints([3]))
    var valid: List[Bool] = [True, True, False, True]
    return Series("xs", ListColumn.from_lists(rows, DataType.INT64, valid))


def sample_structs() raises -> Series:
    var a = Series("a", Column[Int64]([1, 2, 3]))
    var b = Series("b", StringColumn(["x", "y", "z"]))
    var valid: List[Bool] = [True, False, True]
    var bits = List[UInt8]()
    bits.append(0b101)
    return Series("s", StructColumn([a^, b^], bits^))


def test_list_dtype_and_access() raises:
    var xs = sample_lists()
    assert_equal(xs.dtype(), DataType.list(DataType.INT64))
    assert_equal(xs.dtype().name(), "list[int64]")
    assert_equal(len(xs), 4)
    assert_equal(xs.null_count(), 1)
    var first = xs.get(0).list()
    assert_equal(len(first), 2)
    assert_equal(first.get(1).int64(), 2)
    assert_equal(len(xs.get(1).list()), 0)
    assert_true(xs.get(2).is_null())
    assert_equal(String(xs.get(0)), "[1, 2]")
    assert_equal(String(xs.get(3)), "[3]")
    assert_equal(DataType.parse("list[int64]"), xs.dtype())
    assert_equal(
        DataType.list(DataType.list(DataType.STRING)).name(),
        "list[list[string]]",
    )


def test_list_take_slice_append_equals() raises:
    var xs = sample_lists()
    var taken = xs.take([3, 0, 2])
    assert_equal(len(taken), 3)
    assert_equal(String(taken.get(0)), "[3]")
    assert_equal(String(taken.get(1)), "[1, 2]")
    assert_true(taken.get(2).is_null())
    var sliced = xs.slice(1, 2)
    assert_equal(len(sliced), 2)
    assert_equal(len(sliced.get(0).list()), 0)
    assert_true(sliced.get(1).is_null())
    var joined = xs.append(taken).rechunk()
    assert_equal(len(joined), 7)
    assert_equal(String(joined.get(4)), "[3]")
    assert_equal(String(joined.get(5)), "[1, 2]")
    assert_true(xs.equals(sample_lists()))
    assert_false(xs.equals(taken))
    var nulls = Series.full_null("n", DataType.list(DataType.STRING), 3)
    assert_equal(nulls.null_count(), 3)
    assert_equal(nulls.dtype().inner(), DataType.STRING)
    var or_null = xs.take_or_null([0, -1])
    assert_true(or_null.get(1).is_null())
    assert_equal(String(or_null.get(0)), "[1, 2]")


def test_struct_dtype_and_access() raises:
    var s = sample_structs()
    assert_equal(s.dtype().name(), "struct[a: int64, b: string]")
    assert_equal(s.dtype().field_index("b"), 1)
    assert_equal(s.dtype().field_dtype(1), DataType.STRING)
    assert_equal(len(s), 3)
    assert_equal(s.null_count(), 1)
    assert_equal(s.get(0).struct_field("a").int64(), 1)
    assert_equal(s.get(2).struct_field("b").string(), "z")
    assert_true(s.get(1).is_null())
    assert_equal(String(s.get(0)), "{a: 1, b: x}")
    var column = s.struct_column()
    assert_equal(column.field("a").get(2).int64(), 3)


def test_struct_take_slice_append_equals() raises:
    var s = sample_structs()
    var taken = s.take([2, 1])
    assert_equal(taken.get(0).struct_field("b").string(), "z")
    assert_true(taken.get(1).is_null())
    var sliced = s.slice(1, 2)
    assert_equal(len(sliced), 2)
    assert_equal(sliced.get(1).struct_field("a").int64(), 3)
    var joined = s.append(taken).rechunk()
    assert_equal(len(joined), 5)
    assert_equal(joined.get(3).struct_field("a").int64(), 3)
    assert_true(s.equals(sample_structs()))
    assert_false(s.equals(taken))
    var nulls = Series.full_null("n", s.dtype(), 2)
    assert_equal(nulls.null_count(), 2)
    assert_equal(nulls.dtype(), s.dtype())


def test_frame_holds_nested_columns() raises:
    var frame = DataFrame(
        [sample_lists(), Series("k", Column[Int64]([1, 2, 3, 4]))]
    )
    assert_equal(frame.height(), 4)
    var shown = String(frame)
    assert_true("list[i64]" in shown)
    assert_true("[1, 2]" in shown)
    var head = frame.head(2)
    assert_equal(head.height(), 2)
    var raised = False
    try:
        _ = frame.sort(["xs"])
    except e:
        raised = True
        assert_true("cannot sort by a list" in String(e))
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
