"""Independent integer and literal references for floating reductions.

Dyadic totals use integer eighth-units. Special-value expectations are literal
IEEE contracts or exhaustive finite association sets, never CPU/GPU equality.
"""
from std.memory import bitcast
from std.math import isnan
from std.testing import assert_equal, assert_true
from dataframe import DataFrame, DataType, Column, Series, Expr, col


def check_result(
    result: DataFrame,
    row: Int,
    units: Int64,
    count: Int64,
    narrow: Bool,
) raises:
    assert_equal(result.item(row, "count").int64(), count)
    assert_true(result.item(row, "suppressed").is_null())
    assert_equal(result.item(row, "mean").is_null(), count == 0)
    var total = Float64(units) / 8.0
    if narrow:
        assert_equal(result.column("sum").dtype(), DataType.FLOAT32)
        assert_equal(result.item(row, "sum").float32(), Float32(total))
    else:
        assert_equal(result.column("sum").dtype(), DataType.FLOAT64)
        assert_equal(result.item(row, "sum").float64(), total)
    assert_equal(result.column("mean").dtype(), DataType.FLOAT64)
    if count:
        assert_equal(result.item(row, "mean").float64(), total / Float64(count))


def dyadic_fixture(
    n: Int, chunked: Bool, narrow: Bool
) raises -> Tuple[
    DataFrame, List[Int64], List[Int64], List[Bool], Int64, Int64
]:
    # Every possible partial total is a small multiple of 1/8 and therefore
    # exactly representable. The oracle adds small integers, never floats.
    var values = List[Float64]()
    var valid = List[Bool]()
    var keys = List[Int32]()
    var key_valid = List[Bool]()
    for i in range(n + 6):
        values.append(Float64(i % 31 - 15) / 8.0)
        keys.append(Int32(i % 5))
        valid.append(i % 7 != 2 and i % 5 != 4)
        key_valid.append(i % 19 != 7)
    var x = Series("x", Column[Float64](values^, valid^)).slice(3, n)
    if narrow:
        x = x.cast(DataType.FLOAT32)
    var k = Series("k", Column[Int32](keys^, key_valid^)).slice(3, n)
    if chunked:
        x = Series._from_chunks(
            [
                x.slice(0, n // 3),
                x.slice(n // 3, 0),
                x.slice(n // 3, n - n // 3),
            ]
        )
        k = Series._from_chunks(
            [k.slice(0, n // 2), k.slice(n // 2, n - n // 2)]
        )
    var frame = DataFrame([x^, k^])
    var sums = List[Int64](length=6, fill=0)
    var counts = List[Int64](length=6, fill=0)
    var seen = List[Bool](length=6, fill=False)
    var all_units = Int64(0)
    var all_count = Int64(0)
    for row in range(n):
        var i = row + 3
        var group = i % 5 if i % 19 != 7 else 5
        seen[group] = True
        if i % 7 != 2 and i % 5 != 4:
            sums[group] += Int64(i % 31 - 15)
            counts[group] += 1
            all_units += Int64(i % 31 - 15)
            all_count += 1
    return (frame^, sums^, counts^, seen^, all_units, all_count)


def exact_expressions(minimum: Int) -> List[Expr]:
    return [
        col("x").sum().alias("sum"),
        col("x").mean().alias("mean"),
        col("x").count().alias("count"),
        col("x").sum(min_count=minimum).alias("suppressed"),
    ]


def special_expressions() -> List[Expr]:
    return [col("x").sum().alias("sum"), col("x").mean().alias("mean")]


def edge_expressions() -> List[Expr]:
    return [
        col("x").sum().alias("sum"),
        col("x").mean().alias("mean"),
        col("x").count().alias("count"),
    ]


def one_group_fixture(
    values: List[Float64], narrow: Bool = False
) raises -> DataFrame:
    var source = Series("x", Column[Float64](values.copy()))
    if narrow:
        source = source.cast(DataType.FLOAT32)
    return DataFrame([source^, Series("k", Column[Int32]([1, 1, 1]))])


def special_fixture(case_index: Int) raises -> DataFrame:
    var minimum_subnormal = bitcast[DType.float64](UInt64(1))
    var cases: List[List[Float64]] = [
        [18_014_398_509_481_984.0, 1.0, -18_014_398_509_481_984.0],
        [Float64("inf"), Float64("-inf"), 1.0],
        [Float64("nan"), 1.0, 2.0],
        [Float64("inf"), 1.0, 2.0],
        [-0.0, -0.0, -0.0],
        [minimum_subnormal, minimum_subnormal, minimum_subnormal],
    ]
    return one_group_fixture(cases[case_index])


def special_kind(case_index: Int) -> Int:
    var kinds: List[Int] = [0, 1, 1, 2, 3, 4]
    return kinds[case_index]


def check_special(actual: DataFrame, kind: Int) raises:
    var total = actual.item(0, "sum").float64()
    var mean = actual.item(0, "mean").float64()
    if kind == 0:
        # Exhaustive possible associations of [2**54,1,-2**54]
        # produce 0 or 1. This is an exact finite set, not a tolerance.
        assert_true(total == 0.0 or total == 1.0)
        assert_equal(mean, total / 3.0)
    elif kind == 1:
        assert_true(isnan(total) and isnan(mean))
    elif kind == 2:
        assert_equal(total, Float64("inf"))
        assert_equal(mean, Float64("inf"))
    elif kind == 3:
        assert_equal(bitcast[DType.uint64](total), UInt64(0))
        assert_equal(bitcast[DType.uint64](mean), UInt64(0))
    else:
        assert_equal(bitcast[DType.uint64](total), UInt64(3))
        assert_equal(bitcast[DType.uint64](mean), UInt64(1))


def null_payload_fixture(narrow: Bool) raises -> DataFrame:
    var payload = Series(
        "x",
        Column[Float64](
            [Float64("nan"), Float64("inf"), Float64("-inf")],
            [False, False, False],
        ),
    )
    var x = payload.cast(DataType.FLOAT32) if narrow else payload.copy()
    return DataFrame([x^, Series("k", Column[Int32]([1, 1, 1]))])


def edge_fixture(
    case_index: Int,
) raises -> Tuple[DataFrame, Bool, Int, Float64]:
    var largest = bitcast[DType.float64](UInt64(0x7FEFFFFFFFFFFFFF))
    var tiny = bitcast[DType.float64](UInt64(1))
    var largest32 = Float64(bitcast[DType.float32](UInt32(0x7F7FFFFF)))
    var tiny32 = Float64(bitcast[DType.float32](UInt32(1)))
    var cases: List[List[Float64]] = [
        [largest, largest, -largest],
        [-tiny, 0.0, 0.0],
        [tiny, 0.0, 0.0],
        [largest32, largest32, -largest32],
        [tiny32, 0.0, 0.0],
    ]
    var narrow = case_index >= 3
    var kind = (
        0 if case_index == 0 or case_index == 3 else 1 if case_index == 1 else 2
    )
    var frame = one_group_fixture(cases[case_index], narrow)
    return (frame^, narrow, kind, cases[case_index][0])


def check_edge(
    result: DataFrame, narrow: Bool, kind: Int, anchor: Float64
) raises:
    assert_equal(result.item(0, "count").int64(), Int64(3))
    assert_equal(result.column("mean").dtype(), DataType.FLOAT64)
    var mean = result.item(0, "mean").float64()
    if narrow:
        assert_equal(result.column("sum").dtype(), DataType.FLOAT32)
        var total = result.item(0, "sum").float32()
        # These three Float32 inputs have an exact Float64 total
        # for every association. Intermediate Float32 overflow is
        # forbidden by the documented Float64 accumulation contract.
        assert_equal(total, Float32(anchor))
        assert_equal(mean, Float64(Float32(anchor)) / 3.0)
        if kind == 2:
            assert_true(mean > 0.0)
    else:
        var total = result.item(0, "sum").float64()
        if kind == 0:
            # IEEE binary64 permits either finite or +Inf here:
            # max+(max-max) versus (max+max)-max. No tolerance.
            assert_true(total == anchor or total == Float64("inf"))
            assert_equal(mean, total / 3.0)
        elif kind == 1:
            assert_equal(
                bitcast[DType.uint64](total), UInt64(0x8000000000000001)
            )
            assert_equal(
                bitcast[DType.uint64](mean), UInt64(0x8000000000000000)
            )
        else:
            assert_equal(bitcast[DType.uint64](total), UInt64(1))
            assert_equal(bitcast[DType.uint64](mean), UInt64(0))
