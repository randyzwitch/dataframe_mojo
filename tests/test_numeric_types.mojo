"""Every numeric dtype through every dispatch site, plus per-width boundaries.

`check_dtype[D]` runs one trivial operation per dispatch site (construction,
access, display, equality, take/slice/concat, sort, hashing via group_by /
join / unique, reductions, windows, casts, CSV, Arrow, comparisons,
arithmetic, when/then, fill_null) for a single dtype; the test calls it for
each entry of NUMERIC_DTYPES, so a new dtype cannot skip a site.
"""
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    CsvField,
    CsvSchema,
    Expr,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    concat,
    export_arrow,
    import_arrow,
    lit,
    read_csv,
    when,
    write_csv,
)
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.binding import bind
from dataframe.execution import evaluate

comptime PATH = "/tmp/dataframe_mojo_numeric_types.csv"


def values[D: DType]() -> List[Scalar[D]]:
    """MIN, MAX, 5, 1, 2 and 1 again (a duplicate), with row 6 null.

    Six distinct non-null values for every type (unsigned MIN is 0).
    """
    return [Scalar[D].MIN, Scalar[D].MAX, 5, 1, 2, 1, 0]


def column[D: DType]() raises -> Column[Scalar[D]]:
    return Column[Scalar[D]](
        values[D](), [True, True, True, True, True, True, False]
    )


def frame[D: DType]() raises -> DataFrame:
    return DataFrame(
        [
            Series("k", column[D]()),
            Series("g", Column[Int64]([1, 1, 2, 2, 3, 3, 3])),
        ]
    )


def assert_dtype(series: Series, expected: DataType) raises:
    assert_true(
        series.dtype() == expected,
        String("dtype ", series.dtype(), " != ", expected),
    )


def check_dtype[D: DType]() raises:
    var dtype = DataType.of(D)
    var name = dtype.name()
    var df = frame[D]()
    ref s = df._columns[0]
    # Construction, names, access, display.
    assert_dtype(s, dtype)
    assert_true(DataType.parse(name) == dtype, name)
    assert_equal(len(s), 7)
    assert_equal(s.null_count(), 1)
    assert_true(s.get(0).numeric[D]() == Scalar[D].MIN, name)
    assert_true(s.get(1).numeric[D]() == Scalar[D].MAX, name)
    assert_true(s.get(6).is_null())
    assert_true(s.numeric[D]()._get(3) == 1)
    assert_true(String(Scalar[D].MAX) in df.to_string(), name)
    assert_true(dtype.short_name() in df.to_string(), name)
    # Equality, take, slice, concat.
    assert_true(s.equals(Series("k", column[D]())))
    assert_true(s.take([1, 1, 6]).get(1).numeric[D]() == Scalar[D].MAX)
    assert_true(s.slice(2, 3).get(0).numeric[D]() == 5)
    var stacked = concat([df.slice(0, 2), df.slice(5, 2)])
    assert_dtype(stacked._columns[0], dtype)
    assert_true(stacked.item(1, "k").numeric[D]() == Scalar[D].MAX)
    # Sort (MIN first, nulls last) and hashing: unique, group_by, join.
    var sorted = df.sort("k")
    assert_true(sorted.item(0, "k").numeric[D]() == Scalar[D].MIN, name)
    assert_true(sorted.item(5, "k").numeric[D]() == Scalar[D].MAX, name)
    assert_true(sorted.item(6, "k").is_null())
    assert_equal(df.select(["k"]).unique().height(), 6)
    var grouped = df.group_by("k", maintain_order=True).agg(
        col("g").sum().alias("total")
    )
    assert_equal(grouped.height(), 6)
    assert_dtype(grouped._columns[0], dtype)
    assert_equal(grouped.item(3, "total").int64(), 5)  # key 1: rows 3 and 5
    var joined = df.join(df.select(["k"]).unique(), on="k")
    assert_equal(joined.height(), 6)
    # Reductions: min/max/first keep the dtype; sum follows sum_type().
    var r = df.select_exprs(
        [
            col("k").min().alias("min"),
            col("k").max().alias("max"),
            col("k").first().alias("first"),
            col("k").n_unique().alias("n"),
            col("k").count().alias("count"),
        ]
    )
    assert_true(r.item(0, "min").numeric[D]() == Scalar[D].MIN, name)
    assert_true(r.item(0, "max").numeric[D]() == Scalar[D].MAX, name)
    assert_true(r.item(0, "first").numeric[D]() == Scalar[D].MIN, name)
    var middle = df.slice(2, 4).select_exprs(
        [col("k").sum().alias("sum"), col("k").mean().alias("mean")]
    )
    assert_dtype(middle._columns[0], dtype.sum_type())
    var nine = "9.0" if D.is_floating_point() else "9"
    assert_equal(String(middle.item(0, "sum")), nine)  # 5 + 1 + 2 + 1
    assert_equal(middle.item(0, "mean").float64(), 2.25)
    assert_equal(r.item(0, "n").int64(), 6)
    assert_equal(r.item(0, "count").int64(), 6)
    # Windows: cum_sum type, cum_max, shift.
    var w = df.slice(2, 4).select_exprs(
        [
            col("k").cum_sum().alias("cs"),
            col("k").cum_max().alias("cm"),
            col("k").shift(1).alias("sh"),
        ]
    )
    assert_dtype(w._columns[0], dtype.sum_type())
    assert_equal(String(w.item(3, "cs")), nine)
    assert_true(w.item(3, "cm").numeric[D]() == 5)
    assert_true(w.item(1, "sh").numeric[D]() == 5)
    # Comparisons with a typed literal, arithmetic, when/then, fill_null.
    var e = df.select_exprs(
        [
            (col("k") > lit(Scalar[D](1))).alias("gt"),
            (col("k") * lit(Scalar[D](0)) + lit(Scalar[D](1))).alias("plus"),
            when(col("k").eq(lit(Scalar[D](2))))
            .then(col("k"))
            .otherwise(lit(Scalar[D](0)))
            .alias("pick"),
            col("k").fill_null(lit(Scalar[D](2))).alias("filled"),
        ]
    )
    assert_true(e.item(1, "gt").bool())
    assert_false(e.item(3, "gt").bool())
    assert_true(e.item(2, "plus").numeric[D]() == 1)
    assert_dtype(e._columns[1], dtype)
    assert_true(e.item(4, "pick").numeric[D]() == 2)
    assert_true(e.item(3, "pick").numeric[D]() == 0)
    assert_true(e.item(6, "filled").numeric[D]() == 2)
    # Casts: to string and back, and through float64 for small values.
    var text = s.cast(DataType.STRING)
    assert_equal(text.get(1).string(), String(Scalar[D].MAX))
    assert_true(text.cast(dtype).equals(s), name)
    assert_true(
        s.slice(2, 5).cast(DataType.FLOAT64).cast(dtype).equals(s.slice(2, 5))
    )
    # CSV and Arrow round trips keep dtype and every boundary value.
    write_csv(df, PATH)
    assert_true(read_csv(PATH, CsvSchema.of(df)).equals(df), name)
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_arrow(df, array, schema)
    assert_true(import_arrow(array, schema).equals(df), name)


def test_every_dtype_through_every_dispatch_site() raises:
    comptime for i in range(len(NUMERIC_DTYPES)):
        check_dtype[NUMERIC_DTYPES[i]]()


def raises_with(df: DataFrame, e: Expr, message: String) raises:
    var text = String()
    try:
        _ = df.select(e)
    except err:
        text = String(err)
    assert_true(message in text, "expected '" + message + "' in '" + text + "'")


def one[D: DType](value: Scalar[D]) raises -> DataFrame:
    return DataFrame([Series("x", Column[Scalar[D]]([value]))])


def test_integer_overflow_at_each_width() raises:
    raises_with(
        one(Int8.MAX), col("x") + lit(Int8(1)), "int8 addition overflow"
    )
    raises_with(
        one(Int8.MIN), col("x") - lit(Int8(1)), "int8 subtraction overflow"
    )
    raises_with(
        one(Int16(200)),
        col("x") * lit(Int16(200)),
        "int16 multiplication overflow",
    )
    raises_with(
        one(Int32.MIN),
        col("x") // lit(Int32(-1)),
        "int32 floor division overflow",
    )
    raises_with(
        one(UInt8(0)), col("x") - lit(UInt8(1)), "uint8 subtraction overflow"
    )
    raises_with(
        one(UInt32.MAX), col("x") + lit(UInt32(1)), "uint32 addition overflow"
    )
    raises_with(
        one(UInt64.MAX),
        col("x") * lit(UInt64(2)),
        "uint64 multiplication overflow",
    )
    raises_with(
        one(Int16(2)), col("x").pow(lit(Int16(15))), "int16 pow overflow"
    )
    raises_with(one(Int32.MIN), -col("x"), "int32 negation overflow")
    raises_with(one(UInt16(3)), -col("x"), "uint16 negation overflow")
    # In range stays exact, including 64-bit products near the limits.
    assert_equal(
        one(UInt64(4294967295))
        .select(col("x") * lit(UInt64(4294967297)))
        .item(0, "x")
        .uint64(),
        UInt64.MAX,
    )
    assert_equal(
        one(Int8(-7)).select(col("x") // lit(Int8(2))).item(0, "x").int8(), -4
    )
    assert_equal(
        one(Int8(-7)).select(col("x") % lit(Int8(2))).item(0, "x").int8(), 1
    )
    assert_equal(
        one(UInt8(7)).select(col("x") // lit(UInt8(0))).item(0, "x").is_null(),
        True,
    )
    # Division is float64 for integers and float32 for float32.
    assert_true(
        one(Int16(1)).select(col("x") / lit(Int16(4))).item(0, "x").float64()
        == 0.25
    )
    assert_true(
        one(Float32(1))
        .select(col("x") / lit(Float32(4)))
        .item(0, "x")
        .float32()
        == 0.25
    )


def test_sum_types_and_overflow() raises:
    # 8/16-bit sums widen to int64, so they cannot overflow at their width.
    var bytes = DataFrame([Series("x", Column[UInt8]([255, 255, 255]))])
    assert_equal(bytes.select(col("x").sum()).item(0, "x").int64(), 765)
    # 32- and 64-bit sums keep their type and are checked there.
    var ints = DataFrame([Series("x", Column[Int32]([Int32.MAX, 1]))])
    raises_with(ints, col("x").sum(), "int32 expression sum overflow")
    var big = DataFrame([Series("x", Column[UInt64]([UInt64.MAX - 1, 1]))])
    assert_equal(big.select(col("x").sum()).item(0, "x").uint64(), UInt64.MAX)
    var over = DataFrame([Series("x", Column[UInt64]([UInt64.MAX, 1]))])
    raises_with(over, col("x").sum(), "uint64 expression sum overflow")
    assert_equal(over.select(col("x").max()).item(0, "x").uint64(), UInt64.MAX)
    assert_equal(over.select(col("x").min()).item(0, "x").uint64(), 1)
    raises_with(ints, col("x").cum_sum(), "int32 window sum overflow")


def test_casts_from_text_check_each_width() raises:
    var text = DataFrame(
        [Series("s", StringColumn(["127", "128", "-129", "-0", "255", "256"]))]
    )
    var int8 = text.select(col("s").cast(DataType.INT8, strict=False))
    assert_equal(int8.item(0, "s").int8(), 127)
    assert_true(int8.item(1, "s").is_null())
    assert_true(int8.item(2, "s").is_null())
    var uint8 = text.select(col("s").cast(DataType.UINT8, strict=False))
    assert_equal(uint8.item(3, "s").uint8(), 0)
    assert_equal(uint8.item(4, "s").uint8(), 255)
    assert_true(uint8.item(5, "s").is_null())
    assert_true(uint8.item(2, "s").is_null())
    var big = DataFrame(
        [
            Series(
                "s",
                StringColumn(["18446744073709551615", "18446744073709551616"]),
            )
        ]
    )
    var u64 = big.select(col("s").cast(DataType.UINT64, strict=False))
    assert_equal(u64.item(0, "s").uint64(), UInt64.MAX)
    assert_true(u64.item(1, "s").is_null())
    # Float to integer truncates toward zero; out of range is an error.
    var floats = DataFrame(
        [Series("f", Column[Float64]([-1.9, 127.9, 128.0, 255.5]))]
    )
    var to_i8 = floats.select(col("f").cast(DataType.INT8, strict=False))
    assert_equal(to_i8.item(0, "f").int8(), -1)
    assert_equal(to_i8.item(1, "f").int8(), 127)
    assert_true(to_i8.item(2, "f").is_null())
    var to_u8 = floats.select(col("f").cast(DataType.UINT8, strict=False))
    assert_true(to_u8.item(0, "f").is_null())
    assert_equal(to_u8.item(3, "f").uint8(), 255)
    var message = String()
    try:
        _ = text.select(col("s").cast(DataType.INT8))
    except e:
        message = String(e)
    assert_true("cast from string to int8 failed at row 1" in message, message)


def test_csv_fields_check_each_width() raises:
    with open(PATH, "w") as f:
        f.write("a,b\n-128,4294967295\n127,0\n")
    var schema = CsvSchema(
        [CsvField("a", DataType.INT8), CsvField("b", DataType.UINT32)]
    )
    var df = read_csv(PATH, schema)
    assert_equal(df.item(0, "a").int8(), -128)
    assert_equal(df.item(0, "b").uint32(), UInt32.MAX)
    with open(PATH, "w") as f:
        f.write("a,b\n128,0\n")
    var message = String()
    try:
        _ = read_csv(PATH, schema)
    except e:
        message = String(e)
    # The Polars-derived typed builder reports numeric overflow by parser
    # category rather than the retired reader's dtype-specific text.
    assert_true("CSV signed integer overflow" in message, message)


def test_float32_nan_and_sort() raises:
    var nan = Float32(0) / Float32(0)
    var df = DataFrame(
        [
            Series(
                "f",
                Column[Float32](
                    [2.5, nan, -1.5, 0.0], [True, True, True, False]
                ),
            )
        ]
    )
    var flags = df.select(col("f").is_nan())
    assert_true(flags.item(1, "f").bool())
    var sorted = df.sort("f")
    assert_equal(sorted.item(0, "f").float32(), -1.5)
    assert_true(sorted.item(2, "f").float32() != sorted.item(2, "f").float32())
    assert_true(sorted.item(3, "f").is_null())
    var filled = df.select(col("f").fill_nan(lit(Float32(9))))
    assert_equal(filled.item(1, "f").float32(), 9)
    assert_true(
        df.select(col("f").sqrt()).get_column("f").dtype() == DataType.FLOAT32
    )


def test_mixed_widths_require_cast() raises:
    var df = DataFrame(
        [
            Series("a", Column[Int32]([1])),
            Series("b", Column[Int64]([2])),
        ]
    )
    var message = String()
    try:
        _ = df.select(col("a") + col("b"))
    except e:
        message = String(e)
    assert_true(
        "requires matching dtypes, found int32 and int64" in message, message
    )
    assert_equal(
        df.select(col("a").cast(DataType.INT64) + col("b"))
        .item(0, "a")
        .int64(),
        3,
    )


def run[
    width: Int
](frame: DataFrame, expr: Expr, batch_size: Int) raises -> Series:
    return evaluate[width](
        bind(expr, frame._columns),
        frame._columns,
        frame.height(),
        batch_size=batch_size,
    )


def test_float32_simd_widths_agree() raises:
    var values = List[Float32]()
    var valid = List[Bool]()
    for i in range(37):
        values.append(Float32(i) * 0.75 - 9)
        valid.append(i % 5 != 2)
    var frame = DataFrame([Series("x", Column[Float32](values^, valid^))])
    var e = (lit(Float32(100)) / (col("x").abs() + lit(Float32(1)))).clip(
        lit(Float32(2)), lit(Float32(50))
    ).round(3) - (col("x") // lit(Float32(4))) % lit(Float32(3))
    var reference = run[1](frame, e, 1)
    assert_true(reference.dtype() == DataType.FLOAT32)
    for size in [1, 3, 8, 64]:
        assert_true(run[4](frame, e, size).equals(reference))
        assert_true(run[8](frame, e, size).equals(reference))
        assert_true(run[16](frame, e, size).equals(reference))
    var compare = col("x") >= lit(Float32(0))
    assert_true(run[16](frame, compare, 5).equals(run[1](frame, compare, 1)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
