"""Filtering straight from the mask's words (#376) must keep exactly the rows
a row list keeps: for masks with nulls, sliced masks whose bits start
mid-byte, every fixed-width type, Booleans, nulls in the data, strings
beside them, chunked (Parquet-like) columns, all-true and all-false masks,
and enough rows for several workers.
"""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    Series,
    StringColumn,
)


def set_threads(n: Int):
    var name = String("DATAFRAME_THREADS")
    var value = String(n)
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def frame(rows: Int, seed: UInt64) raises -> DataFrame:
    var rng = Lcg(seed)
    var ints = List[Int64](capacity=rows)
    var int_valid = List[Bool](capacity=rows)
    var floats = List[Float32](capacity=rows)
    var small = List[Int8](capacity=rows)
    var flags = List[Bool](capacity=rows)
    var flag_valid = List[Bool](capacity=rows)
    var words = List[String](capacity=rows)
    for i in range(rows):
        ints.append(Int64(i))
        int_valid.append(rng.next(9) != 0)
        floats.append(Float32(rng.next(1000)) / 7)
        small.append(Int8(rng.next(200) - 100))
        flags.append(rng.next(2) == 0)
        flag_valid.append(rng.next(7) != 0)
        words.append("w" + String(i % 97))
    return DataFrame(
        [
            Series("i", Column[Int64](ints^, int_valid^)),
            Series("f", Column[Float32](floats^)),
            Series("s", StringColumn(words)),
            Series("b", BoolColumn(flags, flag_valid)),
            Series("n", Column[Int8](small^)),
        ]
    )


def mask(rows: Int, seed: UInt64, density: Int) -> BoolColumn:
    """`density` in 0..100 percent true; about one entry in eleven null."""
    var rng = Lcg(seed)
    var values = List[Bool](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for _ in range(rows):
        values.append(rng.next(100) < density)
        valid.append(rng.next(11) != 0)
    try:
        return BoolColumn(values, valid)
    except:
        return BoolColumn(values)


def kept(m: BoolColumn) -> List[Int]:
    var rows = List[Int]()
    for i in range(len(m)):
        if m._valid(i) and m._get(i):
            rows.append(i)
    return rows^


def check(data: DataFrame, m: BoolColumn, label: String) raises:
    var got = data.filter(m)
    var want = data.take(kept(m))
    assert_equal(got.height(), want.height(), label + " height")
    assert_true(got.equals(want), label)


def chunked(data: DataFrame, cuts: List[Int]) raises -> DataFrame:
    """The same frame with every column split into arrays at `cuts`."""
    var columns = List[Series]()
    for column in data.columns():
        var parts = List[Series]()
        var at = 0
        for cut in cuts:
            parts.append(data.column(column).slice(at, cut - at))
            at = cut
        parts.append(data.column(column).slice(at, data.height() - at))
        columns.append(Series._from_chunks(parts))
    return DataFrame(columns^)


def test_masks_of_every_density() raises:
    set_threads(8)
    var data = frame(5000, 3)
    for density in [0, 1, 50, 99, 100]:
        check(
            data,
            mask(5000, UInt64(density + 1), density),
            "density " + String(density),
        )


def test_sliced_mask_and_frame() raises:
    set_threads(8)
    var data = frame(3000, 5)
    var m = mask(3003, 7, 40)
    var window = m.slice(3, 3000)
    check(data, window, "mask sliced at bit 3")
    check(data.slice(13, 2000), mask(2000, 9, 60), "frame sliced at row 13")


def test_many_rows_on_every_worker() raises:
    for threads in [8, 1]:
        set_threads(threads)
        var data = frame(300_000, 11)
        for density in [3, 50, 97]:
            check(
                data,
                mask(300_000, UInt64(density), density),
                String(threads) + " threads, density " + String(density),
            )
    set_threads(8)


def test_chunked_columns() raises:
    set_threads(8)
    var data = frame(200_000, 13)
    var parts = chunked(data, [1, 64, 65, 70_001, 131_072])
    for density in [5, 70, 100]:
        var m = mask(200_000, UInt64(density + 40), density)
        var got = parts.filter(m)
        assert_true(
            got.equals(data.take(kept(m))), "chunked " + String(density)
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
