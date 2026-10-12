"""Extreme date fields keep Gregorian meaning without signed epoch-shift wrap."""
from std.testing import assert_equal, assert_true
from dataframe import DataFrame, DataType, Series, Column, col


def test_maximum_date_weekday_does_not_wrap_the_epoch_alignment_shift() raises:
    # 2**63-1 is divisible by7; epoch Thursday plus that many days is Thursday.
    var source = DataFrame(
        [Series("d", Column[Int64]([Int64.MAX])).with_dtype(DataType.DATE)]
    )
    var actual = source.select(col("d").dt().weekday()).item(0, "d").int64()
    assert_true(
        actual == 4,
        "maximum Date weekday must be Thursday(4); got " + String(actual),
    )


def test_extreme_date_calendar_fields_do_not_wrap_the_epoch_conversion() raises:
    var source = DataFrame(
        [
            Series(
                "d",
                Column[Int64](
                    [Int64.MAX - 719468, Int64.MAX - 719467, Int64.MAX]
                ),
            ).with_dtype(DataType.DATE)
        ]
    )
    var actual = source.select_exprs(
        [
            col("d").dt().year().alias("year"),
            col("d").dt().month().alias("month"),
            col("d").dt().day().alias("day"),
            col("d").dt().ordinal_day().alias("ordinal"),
        ]
    )
    var years: List[Int64] = [
        25252734927766554,
        25252734927766554,
        25252734927768524,
    ]
    var months: List[Int64] = [9, 9, 7]
    var days: List[Int64] = [25, 26, 27]
    var ordinals: List[Int64] = [268, 269, 209]
    for row in range(3):
        var year = actual.item(row, "year").int64()
        assert_true(
            year == years[row],
            "extreme Date calendar year must remain positive; got "
            + String(year),
        )
        assert_equal(actual.item(row, "month").int64(), months[row])
        assert_equal(actual.item(row, "day").int64(), days[row])
        assert_equal(actual.item(row, "ordinal").int64(), ordinals[row])
