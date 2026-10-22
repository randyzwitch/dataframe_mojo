"""Extreme date fields keep Gregorian meaning without signed epoch-shift wrap."""
# contract-group: expr.temporal
from std.testing import assert_equal, assert_true
from dataframe import DataFrame, DataType, Series, Column, Expr, col


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


def full_date_fixture() raises -> DataFrame:
    from std.memory import bitcast

    var raw: List[Int64] = [
        0,  # Keep a nonzero bitmap offset while retaining both endpoints.
        Int64.MIN,
        Int64.MIN + 1,
        Int64.MIN + 146096,
        Int64.MIN + 719468,
        -719469,
        -719468,
        -719467,
        -146097,
        -1,
        0,
        1,
        146097,
        Int64.MAX - 719468,
        Int64.MAX - 719467,
        Int64.MAX - 3,
        Int64.MAX - 2,
        Int64.MAX - 1,
        Int64.MAX,
    ]
    var state = UInt64(0x444154454649454C)
    for _ in range(512):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        raw.append(bitcast[DType.int64](state))
    var valid = List[Bool]()
    for i in range(len(raw)):
        valid.append(i < 19 or i % 11 != 0)
    return DataFrame(
        [Series("d", Column[Int64](raw^, valid)).with_dtype(DataType.DATE)]
    ).slice(1, 529)


def full_date_fields() -> List[Expr]:
    return [
        col("d").dt().year().alias("year"),
        col("d").dt().month().alias("month"),
        col("d").dt().day().alias("day"),
        col("d").dt().weekday().alias("weekday"),
        col("d").dt().ordinal_day().alias("ordinal"),
    ]


def full_date_expected(frame: DataFrame) raises -> DataFrame:
    from tests.support.metal_domain_reference import calendar

    var source = frame.column("d").int64()
    var fields = List[Series]()
    var names = List[String](["year", "month", "day", "weekday", "ordinal"])
    for field in range(5):
        var values = List[Int64]()
        var valid = List[Bool]()
        for row in range(frame.height()):
            valid.append(source.is_valid(row))
            values.append(
                calendar(source._get(row))[field] if source.is_valid(
                    row
                ) else Int64(0)
            )
        fields.append(Series(names[field], Column[Int64](values^, valid)))
    return DataFrame(fields^)


def test_date_fields_match_independent_calendar_across_full_storage_domain() raises:
    from tests.support.coverage_compare import assert_frame

    var frame = full_date_fixture()
    var expected = full_date_expected(frame)
    assert_frame(frame.select_exprs(full_date_fields()), expected)
    assert_frame(
        frame.lazy().select_exprs(full_date_fields()).collect(engine="cpu"),
        expected,
    )
