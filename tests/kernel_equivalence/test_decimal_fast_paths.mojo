"""Integer-to-decimal casts take one typed pass per source type and target
width, and a decimal product of 64-bit operands is computed in 64 bits;
both give exactly what the row-by-row arithmetic gives: the same values
and nulls, a strict cast past the precision raising with the row and
value, a lenient one giving null, and exact products whether or not the
overflow check runs. Decimal sums and products over nulls read the
operands' combined validity 64 rows at a time: the same values and nulls
as row by row, and a null row's payload is never computed."""
# contract-group: types.cast
from std.testing import assert_equal, assert_true, assert_raises

from dataframe import Column, DataFrame, DataType, Series, col, lit

comptime ROWS = 3_000


def integers() raises -> DataFrame:
    var values = List[Int64]()
    var valid = List[Bool]()
    for row in range(ROWS):
        values.append(Int64((row * 7919) % 20_001) - 10_000)
        valid.append(row % 17 != 3)
    return DataFrame([Series("i", Column[Int64](values^, valid))])


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
        DataType.decimal(19, 0, 128),
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


def places(value: Int128, digits: Int) -> String:
    """A scaled integer as decimal text with `digits` places, `-1.0500`
    for -10500 at four."""
    var negative = value < 0
    var magnitude = -value if negative else value
    var scale = Int128(1)
    for _ in range(digits):
        scale *= 10
    var fraction = String(magnitude % scale)
    while fraction.byte_length() < digits:
        fraction = "0" + fraction
    return (
        ("-" if negative else "") + String(magnitude // scale) + "." + fraction
    )


def test_decimal_products_agree_in_64_and_128_bits() raises:
    """Products of two-place decimals are exact at four places. Test
    operands small enough for 64 bits and the same rows beside one huge
    price, which exercises the overflow check without changing products."""
    var prices = List[Int64]()
    var rates = List[Int64]()
    for row in range(ROWS):
        prices.append(Int64((row * 104_729) % 10_000_000) - 5_000_000)
        rates.append(Int64(row % 101))
    var dtype = DataType.decimal(18, 2, 64)
    var small = DataFrame(
        [decimals("p", prices, dtype), decimals("r", rates, dtype)]
    )
    var product = small.select_exprs([(col("p") * col("r")).alias("x")])
    assert_true(product.column("x").dtype() == DataType.decimal(18, 4, 64))
    var exact = product.column("x").cast(DataType.STRING)
    for row in range(ROWS):
        var raw = Int128(prices[row]) * Int128(rates[row])
        assert_equal(exact.get(row).string(), places(raw, 4))
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


def nullable_cents(
    name: String, values: List[Int64], valid: List[Bool], dtype: DataType
) raises -> Series:
    """Hundredths stored as they are under a two-place decimal type, null
    where `valid` is False, whatever the payload there."""
    return Series(name, Column[Int64](values.copy(), valid)).with_dtype(dtype)


def test_nullable_decimal_arithmetic_matches_row_by_row() raises:
    """Blocks of 64 rows all valid, all null and mixed, on either side, at
    64 and 128 bits: each row is the exact sum or product, or null where
    either side is."""
    var prices = List[Int64]()
    var rates = List[Int64]()
    var price_valid = List[Bool]()
    var rate_valid = List[Bool]()
    for row in range(ROWS):
        prices.append(Int64((row * 104_729) % 10_000_000) - 5_000_000)
        rates.append(Int64(row % 101) - 50)
        var block = row // 64
        # Whole valid blocks, whole null blocks and mixed ones.
        price_valid.append(
            block % 3 != 1 and not (block % 3 == 2 and row % 5 == 0)
        )
        rate_valid.append(row % 7 != 3)
    var narrow = DataType.decimal(9, 2, 64)
    var wide = DataType.decimal(20, 2, 128)
    for t in range(2):
        var dtype = narrow if t == 0 else wide
        var frame = DataFrame(
            [
                nullable_cents("p", prices, price_valid, narrow).cast(dtype),
                nullable_cents("r", rates, rate_valid, narrow).cast(dtype),
            ]
        )
        var out = frame.select_exprs(
            [
                (col("p") * col("r")).alias("x"),
                (col("p") + col("r")).alias("s"),
                (col("p") - col("r")).alias("d"),
            ]
        )
        var x = out.column("x").cast(DataType.STRING)
        var sums = out.column("s").cast(DataType.STRING)
        var differences = out.column("d").cast(DataType.STRING)
        for row in range(ROWS):
            var null = not price_valid[row] or not rate_valid[row]
            assert_equal(x.get(row).is_null(), null)
            assert_equal(sums.get(row).is_null(), null)
            assert_equal(differences.get(row).is_null(), null)
            if null:
                continue
            var p = Int128(prices[row])
            var r = Int128(rates[row])
            assert_equal(x.get(row).string(), places(p * r, 4))
            assert_equal(sums.get(row).string(), places(p + r, 2))
            assert_equal(differences.get(row).string(), places(p - r, 2))


def test_a_null_rows_payload_is_never_computed() raises:
    """A payload past the precision under a null raises nothing; the same
    payload on a valid row still raises."""
    var dtype = DataType.decimal(18, 2, 64)
    var values: List[Int64] = [Int64(100), Int64.MAX, Int64(300)]
    var hidden: List[Bool] = [True, False, True]
    var frame = DataFrame(
        [
            nullable_cents("p", values, hidden, dtype),
            nullable_cents(
                "r",
                [Int64(200), Int64(200), Int64(200)],
                [True, True, True],
                dtype,
            ),
        ]
    )
    var product = frame.select_exprs([(col("p") * col("r")).alias("x")]).column(
        "x"
    )
    assert_true(product.get(1).is_null())
    assert_equal(product.cast(DataType.STRING).get(2).string(), "6.0000")
    var shown = DataFrame(
        [
            nullable_cents("p", values, [True, True, True], dtype),
            nullable_cents(
                "r",
                [Int64(200), Int64(200), Int64(200)],
                [True, True, True],
                dtype,
            ),
        ]
    )
    with assert_raises():
        _ = shown.select_exprs([(col("p") * col("r")).alias("x")])
