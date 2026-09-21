"""Vertical, diagonal, and horizontal concatenation."""
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
)
from dataframe import DataType, Column, DataFrame, Series, concat
from dataframe.column import Column as TypedColumn


def pattern(length: Int, seed: Int) -> List[Bool]:
    var valid = List[Bool](capacity=length)
    for i in range(length):
        valid.append((i * 7 + seed) % 3 != 0)
    return valid^


def make(length: Int, seed: Int) raises -> DataFrame:
    var ints = List[Int64]()
    var floats = List[Float64]()
    var bools = List[Bool]()
    var strings = List[String]()
    for i in range(length):
        ints.append(Int64(seed * 100 + i))
        floats.append(Float64(seed) + Float64(i) / 4)
        bools.append((i + seed) % 2 == 0)
        strings.append(String(seed) + ":" + String(i))
    return DataFrame(
        [
            Series("i", Column[Int64](ints^, pattern(length, seed))),
            Series("f", Column[Float64](floats^, pattern(length, seed + 1))),
            Series("b", Column[Bool](bools^, pattern(length, seed + 2))),
            Series("s", Column[String](strings^, pattern(length, seed + 3))),
        ]
    )


def test_vertical_validity_boundaries() raises:
    var lengths: List[Int] = [0, 1, 7, 8, 9, 15, 16, 17]
    for a in lengths:
        for b in lengths:
            var left = make(a, 1)
            var right = make(b, 2)
            var joined = concat([left.copy(), right.copy()])
            assert_equal(joined.height(), a + b)
            for c in range(4):
                for r in range(a + b):
                    var expected = left._columns[c].get(r) if r < a else (
                        right._columns[c].get(r - a)
                    )
                    assert_true(joined._columns[c].get(r) == expected)
                assert_equal(
                    joined._columns[c].null_count(),
                    left._columns[c].null_count()
                    + right._columns[c].null_count(),
                )
    # Three-way concatenation keeps stacking validity at odd offsets.
    var three = concat([make(3, 1), make(5, 2), make(9, 3)])
    assert_true(three.slice(8).equals(make(9, 3)))
    assert_true(three.slice(3, 5).equals(make(5, 2)))


def test_many_frames_reassemble_exactly() raises:
    """Enough frames and rows to take the parallel path, which sizes each
    output column once from the heights and text sizes of the inputs. A
    parallel CSV read reassembles exactly this way, one frame per range."""
    var frames = List[DataFrame]()
    var total = 0
    for k in range(40):
        frames.append(make(1000 + k, k))
        total += 1000 + k
    var joined = concat(frames)
    assert_equal(joined.height(), total)
    var row = 0
    for k in range(40):
        assert_true(joined.slice(row, 1000 + k).equals(make(1000 + k, k)))
        row += 1000 + k


def test_a_reservation_lands_on_the_column_that_will_be_appended_to() raises:
    """Reserving has to take ownership first.

    A column sliced out of another shares its buffers, and reserving on a
    shared buffer would size the wrong one: the append that follows copies
    before it writes, and the copy has the old capacity, so the column
    doubles its way up anyway. Nothing about the result is wrong when that
    happens, which is why this checks ownership rather than values.
    """
    var source = make(64, 5)
    var window = source.slice(0, 16)
    var column = window._columns[0].copy()
    assert_false(column._data[TypedColumn[Int64]]._owned())
    column._reserve_rows(4096, 0)
    assert_true(column._data[TypedColumn[Int64]]._owned())
    assert_true(column._data[TypedColumn[Int64]]._data[].capacity() >= 4096)
    # The window it came from still holds exactly what it did.
    assert_true(window.equals(source.slice(0, 16)))


def test_reserving_does_not_disturb_shared_buffers() raises:
    """Every input here is a window onto another frame's buffers, so sizing
    the output has to copy before it writes anything."""
    var source = make(64, 5)
    var joined = concat(
        [source.slice(0, 16), source.slice(16, 16), source.slice(32, 32)]
    )
    assert_true(joined.equals(source))
    assert_true(source.equals(make(64, 5)))
    # A second concatenation of the same windows must see them unchanged.
    var again = concat(
        [source.slice(0, 16), source.slice(16, 16), source.slice(32, 32)]
    )
    assert_true(again.equals(source))


def test_vertical_schema_errors_and_shapes() raises:
    var frame = make(3, 1)
    with assert_raises(contains="at least one"):
        _ = concat(List[DataFrame]())
    with assert_raises(contains="frame 1 column 1"):
        _ = concat([frame.copy(), frame.rename({"f": "g"})])
    with assert_raises(contains="frame 1 has 3 columns"):
        _ = concat([frame.copy(), frame.drop("s")])
    var retyped = frame.drop("i").hstack(
        [Series("i", Column[Float64]([1, 2, 3]))]
    )
    with assert_raises():
        _ = concat([frame.copy(), retyped.select(["i", "f", "b", "s"])])
    with assert_raises(contains="how must be"):
        _ = concat([frame.copy()], "sideways")
    assert_true(concat([frame.copy()]).equals(frame))
    var empty = DataFrame([], height=2)
    assert_equal(concat([empty.copy(), DataFrame([], height=3)]).height(), 5)
    assert_true(frame.vstack(frame.clear()).equals(frame))
    assert_equal(frame.vstack(frame).height(), 6)


def test_diagonal() raises:
    var a = DataFrame(
        [
            Series("x", Column[Int64]([1, 2])),
            Series("y", Column[String](["a", "b"])),
        ]
    )
    var b = DataFrame(
        [
            Series("z", Column[Bool]([True])),
            Series("x", Column[Int64]([3])),
        ]
    )
    var result = concat([a.copy(), b.copy()], "diagonal")
    assert_equal(result.columns(), [String("x"), "y", "z"])
    assert_equal(result.height(), 3)
    assert_equal(result.column("x").int64().value(2), Int64(3))
    assert_true(result.column("y").string().is_null(2))
    assert_true(result.column("z").bool().is_null(0))
    assert_true(result.column("z").bool().value(2))
    var clash = DataFrame([Series("x", Column[Float64]([1]))])
    with assert_raises(contains="column x is float64 in frame 1"):
        _ = concat([a.copy(), clash.copy()], "diagonal")


def test_horizontal() raises:
    var a = DataFrame([Series("x", Column[Int64]([1, 2]))])
    var b = DataFrame([Series("y", Column[String](["p", "q"]))])
    var result = concat([a.copy(), b.copy()], "horizontal")
    assert_equal(result.columns(), [String("x"), "y"])
    assert_equal(result.column("y").string().value(1), "q")
    with assert_raises(contains="duplicate column x"):
        _ = a.hstack(a)
    with assert_raises(contains="frame 1 has height 1"):
        _ = a.hstack(DataFrame([Series("z", Column[Int64]([1]))]))
    var zero = DataFrame([], height=2)
    assert_equal(zero.hstack(a).columns(), [String("x")])
    with assert_raises():
        _ = DataFrame([], height=3).hstack(a)


def test_series_append() raises:
    var a = Series("x", Column[Int64]([1, 2, 3], [True, False, True]))
    var b = Series("y", Column[Int64]([4], [False]))
    var result = a.append(b)
    assert_equal(result.name(), "x")
    assert_equal(len(result), 4)
    assert_equal(result.null_count(), 2)
    assert_equal(len(a), 3)
    with assert_raises(contains="Cannot append float64 to int64"):
        _ = a.append(Series("z", Column[Float64]([1])))
    var nulls = Series.full_null("n", "string", 10)
    assert_equal(nulls.null_count(), 10)
    assert_equal(nulls.dtype(), DataType.STRING)
    with assert_raises(contains="Unknown dtype"):
        _ = Series.full_null("n", "int128", 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
