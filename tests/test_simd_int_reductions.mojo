"""Ungrouped integer reductions and len() scan without a row loop (#333).

The SIMD kernels are compared with scalar loops over the same values for
every integer width, with nulls at every bitmap alignment (sliced windows),
lengths around each vector width, chunked columns, one worker and many,
empty and all-null input, and values at both extremes of each type.
"""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import Column, DataFrame, DataType, Series, col, concat


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

    def next(mut self) -> UInt64:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return self.state


def sample[D: DType](mut rng: Lcg) -> Scalar[D]:
    """Mostly small values, sometimes a type's extreme."""
    var r = rng.next()
    var pick = (r >> 40) % 16
    if pick == 0:
        return Scalar[D].MAX
    if pick == 1:
        return Scalar[D].MIN
    comptime if D.is_unsigned():
        return Scalar[D]((r >> 20) % 200)
    else:
        return Scalar[D](Int64((r >> 20) % 200) - 100)


def check[D: DType](column: Column[Scalar[D]], label: String) raises:
    """Every direct reduction of `column` against a scalar loop."""
    var total = Scalar[DType.int128](0)
    var count = 0
    var low = Scalar[D].MAX
    var high = Scalar[D].MIN
    for i in range(len(column)):
        if not column.is_valid(i):
            continue
        var v = column._get(i)
        total += v.cast[DType.int128]()
        count += 1
        low = min(low, v)
        high = max(high, v)
    var frame = DataFrame([Series("x", column.copy())])
    var sum_type = frame.column("x").dtype().sum_type()
    var in_range = True
    comptime for k in range(8):
        comptime T = [
            DType.int8,
            DType.int16,
            DType.int32,
            DType.int64,
            DType.uint8,
            DType.uint16,
            DType.uint32,
            DType.uint64,
        ][k]
        if sum_type == DataType.of(T):
            in_range = (
                total >= Scalar[T].MIN.cast[DType.int128]()
                and total <= Scalar[T].MAX.cast[DType.int128]()
            )
    if in_range:
        assert_equal(
            String(frame.select(col("x").sum()).item()),
            String(total),
            label + " sum",
        )
    else:
        with assert_raises(contains="sum overflow"):
            _ = frame.select(col("x").sum())
    var mean = frame.select(col("x").mean()).item()
    var min_value = frame.select(col("x").min()).item()
    var max_value = frame.select(col("x").max()).item()
    if count == 0:
        assert_true(mean.is_null(), label)
        assert_true(min_value.is_null(), label)
        assert_true(max_value.is_null(), label)
    else:
        assert_equal(
            mean.float64(),
            total.cast[DType.float64]() / Float64(count),
            label + " mean",
        )
        assert_equal(String(min_value), String(low), label + " min")
        assert_equal(String(max_value), String(high), label + " max")
    assert_equal(
        frame.select(col("x").count()).item().int64(), Int64(count), label
    )
    assert_equal(
        frame.select(col("x").len()).item().int64(),
        Int64(len(column)),
        label + " len",
    )


def column_of[
    D: DType
](rows: Int, nulls: Int, seed: UInt64) raises -> Column[Scalar[D]]:
    """`nulls`: 0 none, 1 every seventh row, 2 all."""
    var rng = Lcg(seed)
    var values = List[Scalar[D]](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for i in range(rows):
        values.append(sample[D](rng))
        valid.append(nulls == 0 or (nulls == 1 and i % 7 != 3))
    return Column[Scalar[D]](values^, valid)


def check_width[D: DType]() raises:
    # Windows at every bitmap alignment and lengths around each vector
    # width, so masks straddle bytes and loops leave tails.
    var base = column_of[D](300, 1, 7)
    for offset in range(9):
        for length in [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 129, 200]:
            if offset + length <= 300:
                check[D](
                    base.slice(offset, length),
                    String(D) + " window " + String(offset),
                )
    for nulls in range(3):
        check[D](column_of[D](5000, nulls, UInt64(nulls) + 11), String(D))


def test_every_width_one_worker() raises:
    set_threads(1)
    check_width[DType.int8]()
    check_width[DType.int16]()
    check_width[DType.int32]()
    check_width[DType.int64]()
    check_width[DType.uint8]()
    check_width[DType.uint16]()
    check_width[DType.uint32]()
    check_width[DType.uint64]()


def test_worker_tails_and_chunks() raises:
    set_threads(32)
    check[DType.int16](column_of[DType.int16](1_000_003, 0, 3), "int16")
    check[DType.int16](column_of[DType.int16](1_000_003, 1, 4), "int16")
    check[DType.int64](column_of[DType.int64](1_000_003, 1, 5), "int64")
    check[DType.uint32](column_of[DType.uint32](700_001, 0, 6), "uint32")
    # A chunked column reduces each chunk's share of every worker's range.
    var a = DataFrame([Series("x", column_of[DType.int16](400_001, 1, 8))])
    var b = DataFrame([Series("x", column_of[DType.int16](300_007, 0, 9))])
    var stacked = concat([a.copy(), b.copy()])
    assert_true(stacked.column("x").is_chunked())
    var whole = stacked.rechunk()
    for name in ["sum", "mean", "min", "max", "count", "len"]:
        var e = col("x").sum()
        if name == "mean":
            e = col("x").mean()
        elif name == "min":
            e = col("x").min()
        elif name == "max":
            e = col("x").max()
        elif name == "count":
            e = col("x").count()
        elif name == "len":
            e = col("x").len()
        assert_equal(
            String(stacked.select(e.copy()).item()),
            String(whole.select(e.copy()).item()),
            name,
        )


def test_int64_sums_stay_exact() raises:
    set_threads(1)
    # Partial sums leave Int64 but the total returns inside it: exact in
    # 128 bits whatever the lane order.
    var values = List[Int64]()
    for _ in range(1000):
        values.append(Int64.MAX)
        values.append(Int64.MIN + 1)
    values.append(42)
    var frame = DataFrame([Series("x", Column[Int64](values^))])
    assert_equal(frame.select(col("x").sum()).item().int64(), 42)
    var over = DataFrame(
        [Series("x", Column[Int64]([Int64.MAX, Int64.MAX, 1, 2, 3, 4, 5, 6]))]
    )
    with assert_raises(contains="sum overflow"):
        _ = over.select(col("x").sum())


def test_widening_cast_reads_the_column() raises:
    set_threads(4)
    var column = column_of[DType.int32](200_003, 1, 21)
    var frame = DataFrame([Series("x", column.copy())])
    var total = Scalar[DType.int128](0)
    for i in range(len(column)):
        if column.is_valid(i):
            total += column._get(i).cast[DType.int128]()
    var cast = col("x").cast(DataType.INT64)
    assert_equal(String(frame.select(cast.copy().sum()).item()), String(total))
    assert_equal(
        String(frame.select(cast.copy().min()).item()),
        String(frame.select(col("x").min()).item()),
    )
    assert_equal(frame.select(cast.copy().max()).item().dtype(), DataType.INT64)
    assert_equal(
        frame.select(cast.copy().mean()).item().float64(),
        frame.select(col("x").mean()).item().float64(),
    )


def test_empty_frame() raises:
    set_threads(1)
    var frame = DataFrame([Series("x", Column[Int16](List[Int16]()))])
    assert_equal(frame.select(col("x").len()).item().int64(), 0)
    assert_true(frame.select(col("x").max()).item().is_null())
    assert_true(frame.select(col("x").mean()).item().is_null())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
