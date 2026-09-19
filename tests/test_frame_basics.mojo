"""Core dataframe utilities: slicing, inspection, renaming, rows, equality."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import DataType, AnyValue, Column, DataFrame, Series


def fixture() raises -> DataFrame:
    return DataFrame(
        [
            Series("id", Column[Int64]([1, 2, 3, 4, 5])),
            Series(
                "x",
                Column[Float64](
                    [1.5, 0, 2.5, 3.5, 4.5], [True, False, True, True, True]
                ),
            ),
            Series("ok", Column[Bool]([True, False, True, False, True])),
            Series("name", Column[String](["a", "b", "", "d", "e"])),
        ]
    )


def ids(frame: DataFrame) raises -> List[Int64]:
    var column = frame.column("id").int64()
    var result = List[Int64]()
    for i in range(len(column)):
        result.append(column.value(i))
    return result^


def test_inspection() raises:
    var frame = fixture()
    assert_equal(frame.shape()[0], 5)
    assert_equal(frame.shape()[1], 4)
    assert_equal(len(frame), 5)
    assert_false(frame.is_empty())
    assert_true(frame.clear().is_empty())
    assert_equal(frame.clear().width(), 4)
    assert_equal(frame.columns()[3], "name")
    assert_equal(frame.dtypes()[1], DataType.FLOAT64)
    var nulls = frame.null_count()
    assert_equal(nulls.height(), 1)
    assert_equal(nulls.column("x").int64().value(0), Int64(1))
    assert_equal(nulls.column("id").int64().value(0), Int64(0))
    assert_equal(frame.get_column("name").string().value(3), "d")


def test_head_tail_slice_limit() raises:
    var frame = fixture()
    assert_equal(ids(frame.head(2)), [Int64(1), 2])
    assert_equal(ids(frame.head(0)), List[Int64]())
    assert_equal(ids(frame.head(99)), [Int64(1), 2, 3, 4, 5])
    assert_equal(ids(frame.head(-2)), [Int64(1), 2, 3])
    assert_equal(ids(frame.head(-99)), List[Int64]())
    assert_equal(ids(frame.tail(2)), [Int64(4), 5])
    assert_equal(ids(frame.tail(-3)), [Int64(4), 5])
    assert_equal(ids(frame.tail(99)), [Int64(1), 2, 3, 4, 5])
    assert_equal(ids(frame.limit(1)), [Int64(1)])
    assert_equal(ids(frame.slice(1, 2)), [Int64(2), 3])
    assert_equal(ids(frame.slice(3)), [Int64(4), 5])
    assert_equal(ids(frame.slice(-2, 1)), [Int64(4)])
    assert_equal(ids(frame.slice(-99, 1)), [Int64(1)])
    assert_equal(ids(frame.slice(99, 3)), List[Int64]())
    assert_true(frame.slice(1, 3).column("x").float64().is_null(0))
    with assert_raises():
        _ = frame.slice(0, -2)
    var empty = DataFrame([], height=4)
    assert_equal(empty.head(2).height(), 2)
    assert_equal(empty.tail(9).height(), 4)
    assert_equal(empty.slice(1, 2).height(), 2)


def test_reverse() raises:
    var frame = fixture()
    var reversed = frame.reverse()
    assert_equal(ids(reversed), [Int64(5), 4, 3, 2, 1])
    assert_true(reversed.column("x").float64().is_null(3))
    assert_equal(reversed.column("name").string().value(2), "")
    assert_equal(DataFrame([], height=3).reverse().height(), 3)


def test_drop_and_rename() raises:
    var frame = fixture()
    var dropped = frame.drop(["x", "ok"])
    assert_equal(dropped.columns(), [String("id"), "name"])
    assert_equal(frame.drop("id").width(), 3)
    assert_equal(frame.drop(List[String]()).width(), 4)
    assert_equal(frame.drop(["id", "x", "ok", "name"]).height(), 5)
    with assert_raises(contains="Unknown column"):
        _ = frame.drop(["x", "missing"])
    with assert_raises(contains="twice"):
        _ = frame.drop(["x", "x"])
    var renamed = frame.rename({"x": "value", "name": "label"})
    assert_equal(renamed.columns(), [String("id"), "value", "ok", "label"])
    assert_equal(renamed.column("value").float64().value(0), Float64(1.5))
    var swapped = frame.rename({"id": "x", "x": "id"})
    assert_equal(swapped.column("x").int64().value(0), Int64(1))
    with assert_raises(contains="duplicate"):
        _ = frame.rename({"x": "id"})
    with assert_raises(contains="Unknown column"):
        _ = frame.rename({"missing": "y"})


def test_with_row_index() raises:
    var frame = fixture().with_row_index()
    assert_equal(frame.columns()[0], "index")
    assert_equal(frame.column("index").int64().value(4), Int64(4))
    var offset = fixture().with_row_index("row", offset=10)
    assert_equal(offset.column("row").int64().value(0), Int64(10))
    with assert_raises(contains="collides"):
        _ = fixture().with_row_index("id")
    assert_equal(DataFrame([], height=2).with_row_index().height(), 2)


def test_rows_and_items() raises:
    var frame = fixture()
    var row = frame.row(1)
    assert_equal(len(row), 4)
    assert_equal(row[0].int64(), Int64(2))
    assert_true(row[1].is_null())
    assert_equal(row[1].dtype(), DataType.FLOAT64)
    assert_false(row[2].bool())
    assert_equal(row[3].string(), "b")
    with assert_raises(contains="null"):
        _ = row[1].float64()
    with assert_raises(contains="Expected"):
        _ = row[0].string()
    with assert_raises():
        _ = frame.row(5)
    with assert_raises():
        _ = frame.row(-1)
    assert_equal(len(frame.rows()), 5)
    assert_equal(frame.rows()[4][3].string(), "e")
    assert_equal(frame.item(2, "name").string(), "")
    assert_equal(frame.select(["id"]).head(1).item().int64(), Int64(1))
    with assert_raises(contains="exactly one cell"):
        _ = frame.item()
    assert_true(AnyValue(Int64(1)) == AnyValue(Int64(1)))
    assert_false(AnyValue(Int64(1)) == AnyValue(Float64(1)))
    assert_true(AnyValue.null("int64") == AnyValue.null("int64"))
    assert_false(AnyValue.null("int64") == AnyValue(Int64(0)))
    var nan = Float64(0) / Float64(0)
    assert_true(AnyValue(nan) == AnyValue(nan))
    assert_equal(String(AnyValue.null("string")), "null")
    assert_equal(String(AnyValue(True)), "true")


def test_equals() raises:
    var frame = fixture()
    assert_true(frame.equals(fixture()))
    assert_false(frame.equals(fixture(), null_equal=False))
    assert_false(frame.equals(frame.head(4)))
    assert_false(frame.equals(frame.rename({"x": "y"})))
    assert_false(frame.equals(frame.drop("x")))
    assert_false(frame.equals(frame.reverse()))
    assert_true(frame.drop("x").equals(frame.drop("x"), null_equal=False))
    var nan = Float64(0) / Float64(0)
    var a = DataFrame([Series("f", Column[Float64]([nan, -0.0, 1]))])
    var b = DataFrame([Series("f", Column[Float64]([nan, 0.0, 1]))])
    assert_true(a.equals(b))
    var c = DataFrame([Series("f", Column[Float64]([nan, 0.0, 2]))])
    assert_false(a.equals(c))
    # Null payloads never participate in equality.
    var d = DataFrame([Series("i", Column[Int64]([1, 7], [True, False]))])
    var e = DataFrame([Series("i", Column[Int64]([1, 9], [True, False]))])
    assert_true(d.equals(e))
    assert_true(DataFrame([], height=2).equals(DataFrame([], height=2)))
    assert_false(DataFrame([], height=2).equals(DataFrame([], height=3)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
