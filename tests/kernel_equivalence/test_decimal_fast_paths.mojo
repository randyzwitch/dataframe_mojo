"""Integer-to-decimal casts take one typed pass per source type and target
width, and a decimal product whose operands are small enough is computed
in 64 bits; both give exactly what the row-by-row arithmetic gives: the
same values and nulls, a strict cast past the precision raising with the
row and value, a lenient one giving null, and products near the 64-bit
limit taking the 128-bit path."""
from std.testing import assert_equal, assert_true, assert_raises

from dataframe import Column, DataFrame, DataType, Series, col, lit

comptime ROWS = 3_000


def integers() raises -> DataFrame:
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(ROWS):
        values.append(Int64((row * 7919) % 20_001) - 10_000)
        valid.append(row % 17 != 3)
    return DataFrame([Series("i", Column[Int64](values^, valid^))])


def test_integer_to_decimal_casts_match_the_scaled_values() raises:
    var source = integers()
    var types: List[DataType] = [
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
        DataType.INT64,
    ]
    var targets: List[DataType] = [
        DataType.decimal(9, 2, 32),
        DataType.decimal(18, 4, 64),
        DataType.decimal(38, 10, 128),
    ]
    for t in types:
        # Narrow types hold the low values only, so a lenient cast first.
        var typed = source.select_exprs(
            [col("i").cast(t, strict=False).alias("i")]
        )
        for target in targets:
            var cast = typed.select_exprs([col("i").cast(target).alias("d")])
            assert_true(cast.column("d").dtype() == target)
            var d = cast.column("d").cast(DataType.FLOAT64)
            var i = typed.column("i").cast(DataType.INT64)
            for row in range(ROWS):
                var before = i.get(row)
                var after = d.get(row)
                assert_equal(after.is_null(), before.is_null())
                if not before.is_null():
                    # Integers keep their value exactly at any scale.
                    assert_equal(after.float64(), Float64(before.int64()))


def test_integers_past_the_precision_raise_or_become_null() raises:
    var source = DataFrame(
        [
            Series(
                "i",
                Column[Int64]([Int64(5), Int64(123_456_789), Int64(-7)]),
            )
        ]
    )
    var narrow = DataType.decimal(5, 2, 32)
    with assert_raises(contains="failed at row 1 for value '123456789'"):
        _ = source.select_exprs([col("i").cast(narrow).alias("d")])
    var lenient = source.select_exprs(
        [col("i").cast(narrow, strict=False).alias("d")]
    ).column("d")
    assert_true(not lenient.get(0).is_null())
    assert_true(lenient.get(1).is_null())
    assert_equal(lenient.cast(DataType.FLOAT64).get(2).float64(), -7.0)


def cents(value: Int64) -> String:
    """A count of hundredths as decimal text, `-12.05` for -1205."""
    var negative = value < 0
    var magnitude = -value if negative else value
    var fraction = magnitude % 100
    return (
        ("-" if negative else "")
        + String(magnitude // 100)
        + "."
        + ("0" if fraction < 10 else "")
        + String(fraction)
    )


def decimals(
    name: String, values: List[Int64], dtype: DataType
) raises -> Series:
    var text = List[String]()
    for v in values:
        text.append(cents(v))
    return Series(name, Column[String](text^)).cast(dtype)


def test_decimal_products_agree_in_64_and_128_bits() raises:
    """Products of two-place decimals rounded half-even back to two places:
    from operands small enough for 64 bits, and the same rows next to one
    huge price, which sends the whole column to the 128-bit path."""
    var prices = List[Int64]()
    var rates = List[Int64]()
    for row in range(ROWS):
        prices.append(Int64((row * 104_729) % 10_000_000) - 5_000_000)
        rates.append(Int64(row % 101))
    var dtype = DataType.decimal(18, 2, 64)
    var small = DataFrame(
        [decimals("p", prices, dtype), decimals("r", rates, dtype)]
    )
    var exact = (
        small.select_exprs([(col("p") * col("r")).alias("x")])
        .column("x")
        .cast(DataType.STRING)
    )
    for row in range(ROWS):
        # Hundredths times hundredths, back to hundredths, half to even.
        var raw = Int128(prices[row]) * Int128(rates[row])
        var q = raw // 100 if raw >= 0 else -((-raw) // 100)
        var r = raw - q * 100
        var rest = abs(r)
        if rest > 50 or (rest == 50 and q % 2 != 0):
            q += Int128(1) if raw > 0 else Int128(-1)
        var text = exact.get(row).string()
        assert_equal(text, cents(Int64(q)))
    prices[0] = Int64(999_999_999_999_999)
    var wide = DataFrame(
        [decimals("p", prices, dtype), decimals("r", rates, dtype)]
    )
    var both = (
        wide.select_exprs([(col("p") * col("r")).alias("x")])
        .column("x")
        .cast(DataType.STRING)
    )
    for row in range(1, ROWS):
        assert_equal(both.get(row).string(), exact.get(row).string())
