"""Native temporal fields and fixed logical values use independent calendars."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, DataType, Series, Column, Expr, col, lit, when
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime, _capabilities
from dataframe.accelerator.constants import (
    has_literal_text_parsing,
    fold_literal_text_parsing,
)
from dataframe.accelerator.joins import has_scalar_consumers, lower_join_dag
from dataframe.accelerator.rows import lower_rows
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal import assert_native_report
from tests.support.coverage_compare import assert_frame
from tests.support.metal_domain_checks import check_domain
from tests.support.metal_domain_reference import calendar, floor128


def assert_temporal_native_plan(plan: LazyFrame) raises:
    # Validate every native shape in the portable model run, including plans
    # later in a hardware test that an earlier failure could otherwise hide.
    var bound = fold_literal_text_parsing(plan) if has_literal_text_parsing(
        plan
    ) else plan.copy()
    if has_scalar_consumers(bound):
        _ = lower_join_dag(bound, _capabilities())
    else:
        _ = lower_rows(bound, _capabilities())


def check_temporal_domain(
    plan: LazyFrame,
    expected: DataFrame,
    runtime: MetalRuntime,
    native: Bool,
    submitted: Bool = True,
    fused: Bool = False,
) raises:
    assert_temporal_native_plan(plan)
    check_domain(plan, expected, runtime, native, submitted, fused=fused)


def tagged(
    name: String, dtype: DataType, values: List[Int64], valid: List[Bool]
) raises -> Series:
    if dtype.physical() == DataType.INT32:
        var narrow = List[Int32]()
        for value in values:
            narrow.append(Int32(value))
        return Series(name, Column[Int32](narrow^, valid.copy())).with_dtype(
            dtype
        )
    return Series(name, Column[Int64](values.copy(), valid.copy())).with_dtype(
        dtype
    )


def temporal_cases(runtime: MetalRuntime, native: Bool) raises:
    var dtypes: List[DataType] = [
        DataType.DATE,
        DataType.TIME,
        DataType.datetime("ms"),
        DataType.datetime("us"),
        DataType.datetime("ns"),
    ]
    var raw: List[Int64] = [
        Int64.MIN,
        Int64.MIN + 719467,
        Int64.MIN + 719468,
        -719469,
        -719468,
        -719467,
        -25509,
        -1,
        0,
        1,
        59,
        60,
        10957,
        11016,
        11017,
        19782,
        2932896,
        86399999999999,
        86400000000000,
        Int64.MAX - 719468,
        Int64.MAX - 719467,
        Int64.MAX,
    ]
    var bits = List[Bool](length=len(raw), fill=True)
    bits[3] = False
    for dtype in dtypes:
        var safe = List[Bool]()
        var calendar_safe = List[Bool]()
        var weekday_bits = bits.copy()
        var calendar_bits = bits.copy()
        for at in range(len(raw)):
            safe.append(not dtype.is_date() or raw[at] <= Int64.MAX - 3)
            weekday_bits[at] = weekday_bits[at] and safe[at]
            calendar_safe.append(
                not dtype.is_date() or raw[at] <= Int64.MAX - 719468
            )
            calendar_bits[at] = calendar_bits[at] and calendar_safe[at]
        var source = DataFrame(
            [
                tagged("t", dtype, raw.copy(), bits.copy()),
                Series("safe_weekday", Column[Bool](safe^)),
                Series("safe_calendar", Column[Bool](calendar_safe^)),
            ]
        )
        # The public maximum-Date weekday regression is isolated separately;
        # observing its known CPU failure here would hide other native fields.
        var expressions = List[Expr]()
        if not dtype.is_time():
            expressions.extend(
                [
                    when(col("safe_weekday"))
                    .then(col("t").dt().weekday())
                    .alias("weekday"),
                    when(col("safe_calendar"))
                    .then(col("t").dt().ordinal_day())
                    .alias("ordinal"),
                ]
            )
        if not dtype.is_time():
            expressions.extend(
                [
                    when(col("safe_calendar"))
                    .then(col("t").dt().year())
                    .alias("year"),
                    when(col("safe_calendar"))
                    .then(col("t").dt().month())
                    .alias("month"),
                    when(col("safe_calendar"))
                    .then(col("t").dt().day())
                    .alias("day"),
                ]
            )
        if not dtype.is_date():
            expressions.extend(
                [
                    col("t").dt().hour().alias("hour"),
                    col("t").dt().minute().alias("minute"),
                    col("t").dt().second().alias("second"),
                    col("t").dt().nanosecond().alias("nanos"),
                ]
            )
        if dtype.is_datetime():
            expressions.extend(
                [
                    col("t").dt().date().alias("date"),
                    col("t").dt().time().alias("time"),
                ]
            )
        var weekday = List[Int64]()
        var ordinal = List[Int64]()
        var years = List[Int64]()
        var months = List[Int64]()
        var days_of_month = List[Int64]()
        var hours = List[Int64]()
        var minutes = List[Int64]()
        var seconds = List[Int64]()
        var nanos = List[Int64]()
        var dates = List[Int64]()
        var times = List[Int64]()
        for tick in raw:
            var days = Int64(tick) if dtype.is_date() else Int64(
                0
            ) if dtype.is_time() else Int64(
                floor128(Int128(tick), Int128(dtype.per_second()) * 86400)
            )
            var fields = calendar(days)
            years.append(fields[0])
            months.append(fields[1])
            days_of_month.append(fields[2])
            weekday.append(fields[3])
            ordinal.append(fields[4])
            var per = Int128(dtype.per_second())
            var within = (
                Int128(0) if dtype.is_date() else Int128(tick)
                - floor128(Int128(tick), per * 86400) * per * 86400
            )
            var whole = within // per
            hours.append(Int64(whole // 3600))
            minutes.append(Int64((whole // 60) % 60))
            seconds.append(Int64(whole % 60))
            nanos.append(Int64((within % per) * (1000000000 // per)))
            dates.append(days)
            times.append(Int64(within * (1000000000 // per)))
        var columns = List[Series]()
        if not dtype.is_time():
            columns.extend(
                [
                    tagged("weekday", DataType.INT64, weekday^, weekday_bits^),
                    tagged(
                        "ordinal",
                        DataType.INT64,
                        ordinal^,
                        calendar_bits.copy(),
                    ),
                ]
            )
        if not dtype.is_time():
            columns.extend(
                [
                    tagged(
                        "year", DataType.INT64, years^, calendar_bits.copy()
                    ),
                    tagged(
                        "month", DataType.INT64, months^, calendar_bits.copy()
                    ),
                    tagged(
                        "day",
                        DataType.INT64,
                        days_of_month^,
                        calendar_bits.copy(),
                    ),
                ]
            )
        if not dtype.is_date():
            columns.extend(
                [
                    tagged("hour", DataType.INT64, hours^, bits.copy()),
                    tagged("minute", DataType.INT64, minutes^, bits.copy()),
                    tagged("second", DataType.INT64, seconds^, bits.copy()),
                    tagged("nanos", DataType.INT64, nanos^, bits.copy()),
                ]
            )
        if dtype.is_datetime():
            columns.extend(
                [
                    tagged("date", DataType.DATE, dates^, bits.copy()),
                    tagged("time", DataType.TIME, times^, bits.copy()),
                ]
            )
        var expected = DataFrame(columns^)
        check_temporal_domain(
            source.lazy().select_exprs(expressions), expected, runtime, native
        )
        check_temporal_domain(
            source.slice(0, 0).lazy().select_exprs(expressions),
            expected.slice(0, 0),
            runtime,
            native,
            submitted=False,
        )


def test_native_metal_temporal_all_units_and_extreme_ticks_match_independent_calendar() raises:
    temporal_cases(require_metal(), True)


def extreme_date_plan() raises -> LazyFrame:
    var source = DataFrame(
        [
            tagged(
                "d",
                DataType.DATE,
                [Int64.MAX - 719467, Int64.MAX],
                [True, True],
            )
        ]
    )
    return source.lazy().select_exprs(
        [
            col("d").dt().year().alias("year"),
            col("d").dt().month().alias("month"),
            col("d").dt().day().alias("day"),
            col("d").dt().weekday().alias("weekday"),
            col("d").dt().ordinal_day().alias("ordinal"),
        ]
    )


def test_native_metal_extreme_date_epoch_shifts_keep_independent_gregorian_fields() raises:
    var runtime = require_metal()
    var plan = extreme_date_plan()
    assert_temporal_native_plan(plan)
    var expected = DataFrame(
        [
            tagged(
                "year",
                DataType.INT64,
                [25252734927766554, 25252734927768524],
                [True, True],
            ),
            tagged("month", DataType.INT64, [9, 7], [True, True]),
            tagged("day", DataType.INT64, [26, 27], [True, True]),
            tagged("weekday", DataType.INT64, [4, 4], [True, True]),
            tagged("ordinal", DataType.INT64, [269, 209], [True, True]),
        ]
    )
    # Current CPU Date epoch-shift overflow is a separate strict public
    # regression; CPU parity must not turn that implementation into an oracle.
    var measured = plan.profile(accelerator=runtime)
    var first_year = measured[0].item(0, "year").int64()
    assert_true(
        first_year == 25252734927766554,
        "native Date Gregorian year must remain positive across epoch shift; got "
        + String(first_year),
    )
    assert_frame(measured[0], expected)
    assert_native_report(measured[1])


def truncation_cases(runtime: MetalRuntime, native: Bool) raises:
    for dtype in [
        DataType.DATE,
        DataType.TIME,
        DataType.datetime("ms"),
        DataType.datetime("us"),
        DataType.datetime("ns"),
    ]:
        var periods: List[String] = ["1d", "3d", "1w", "2w"]
        var day_steps: List[Int] = [1, 3, 7, 14]
        if not dtype.is_date():
            periods.extend(["1s", "15m", "3h"])
        for at in range(len(periods)):
            var per = Int128(dtype.per_second())
            var step = Int128(day_steps[at]) * (
                1 if dtype.is_date() else per * 86400
            ) if at < 4 else per * Int128(
                1 if at == 4 else 900 if at == 5 else 10800
            )
            var origin = Int128(4) * (
                1 if dtype.is_date() else per * 86400
            ) if at == 2 or at == 3 else Int128(0)
            var raw: List[Int64] = [
                Int64(Int128(Int64.MIN) + 2 * step),
                -100,
                -1,
                0,
                1,
                100,
                Int64(Int128(Int64.MAX) - 2 * step),
                Int64.MIN,
            ]
            var bits: List[Bool] = [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
            ]
            var expected = List[Int64]()
            for value in raw:
                expected.append(
                    Int64(
                        floor128(Int128(value) - origin, step) * step + origin
                    )
                )
            var source = DataFrame([tagged("t", dtype, raw^, bits.copy())])
            check_temporal_domain(
                source.lazy().select(col("t").dt().truncate(periods[at])),
                DataFrame([tagged("t", dtype, expected^, bits.copy())]),
                runtime,
                native,
            )
    var source = DataFrame(
        [tagged("t", DataType.DATE, [0, 11016, 11017], [True, True, True])]
    )
    var plan = (
        source.lazy()
        .filter(col("t") >= lit("2000-02-29").str().strptime("date"))
        .with_columns(col("t").dt().year().alias("year"))
        .group_by("year", maintain_order=True)
        .agg(col("t").len().alias("n"))
    )
    check_temporal_domain(
        plan,
        DataFrame(
            [
                Series("year", Column[Int64]([2000])),
                Series("n", Column[Int64]([2])),
            ]
        ),
        runtime,
        native,
        # Filter, projection and aggregation form one fused native region;
        # no intermediate provider buffer needs to cross a region boundary.
        fused=True,
    )
    # A row projection makes literal broadcasting a native GPU contract;
    # scalar-only selects have a separate unsupported-plan preflight below.
    var literals: List[Expr] = [
        col("t"),
        lit("2000-02-29T12:34:56.123456")
        .str()
        .strptime("datetime[us]")
        .alias("dt"),
        lit("12:34:56.123456789").str().strptime("time").alias("clock"),
    ]
    check_temporal_domain(
        source.lazy().select_exprs(literals),
        DataFrame(
            [
                tagged(
                    "t", DataType.DATE, [0, 11016, 11017], [True, True, True]
                ),
                tagged(
                    "dt",
                    DataType.datetime("us"),
                    [951827696123456, 951827696123456, 951827696123456],
                    [True, True, True],
                ),
                tagged(
                    "clock",
                    DataType.TIME,
                    [45296123456789, 45296123456789, 45296123456789],
                    [True, True, True],
                ),
            ]
        ),
        runtime,
        native,
    )


def test_native_metal_fixed_truncation_literal_folding_and_temporal_resident_groups() raises:
    truncation_cases(require_metal(), True)


def logical_cases(runtime: MetalRuntime, native: Bool) raises:
    for dtype in [
        DataType.DATE,
        DataType.TIME,
        DataType.datetime("ns"),
        DataType.datetime("us", "America/New_York"),
        DataType.duration("ms"),
        DataType.decimal(18, 4, 64),
        DataType.decimal(9, 2, 32),
    ]:
        var x = tagged(
            "x", dtype, [-2, Int64.MIN, 0, 2], [True, False, True, True]
        )
        var y = tagged("y", dtype, [3, 4, 5, 6], [True, True, True, True])
        if dtype.physical() == DataType.INT32:
            x = Series(
                "x",
                Column[Int32]([-2, Int32.MIN, 0, 2], [True, False, True, True]),
            ).with_dtype(dtype)
            y = Series("y", Column[Int32]([3, 4, 5, 6])).with_dtype(dtype)
        var source = DataFrame([x.copy(), y^])
        var expected = tagged(
            "filled", dtype, [-2, 4, 0, 2], [True, True, True, True]
        )
        var lo = tagged("lo", dtype, [-2], [True])
        var hi = tagged("hi", dtype, [2], [True])
        if dtype.physical() == DataType.INT32:
            expected = Series(
                "filled", Column[Int32]([-2, 4, 0, 2])
            ).with_dtype(dtype)
            lo = Series("lo", Column[Int32]([-2])).with_dtype(dtype)
            hi = Series("hi", Column[Int32]([2])).with_dtype(dtype)
        check_temporal_domain(
            source.lazy().select_exprs(
                [
                    col("x"),
                    col("x").fill_null(col("y")).alias("filled"),
                    (col("x") <= col("y")).alias("cmp"),
                ]
            ),
            DataFrame(
                [
                    x^,
                    expected^,
                    Series(
                        "cmp",
                        Column[Bool](
                            [True, False, True, True], [True, False, True, True]
                        ),
                    ),
                ]
            ),
            runtime,
            native,
        )
        check_temporal_domain(
            source.lazy().select_exprs(
                [col("x").min().alias("lo"), col("x").max().alias("hi")]
            ),
            DataFrame([lo^, hi^]),
            runtime,
            native,
        )


def test_native_metal_fixed_logical_metadata_survives_nullable_rows_and_extrema() raises:
    logical_cases(require_metal(), True)


def temporal_preflight_cases() raises:
    var unavailable = MetalRuntime(device_id=999999)
    var aware = DataFrame(
        [tagged("t", DataType.datetime("us", "UTC"), [0], [True])]
    )
    with assert_raises(contains="time zone field extraction"):
        _ = (
            aware.lazy()
            .select(col("t").dt().year())
            .collect(accelerator=unavailable)
        )
    var dates = DataFrame([tagged("t", DataType.DATE, [0], [True])])
    with assert_raises(
        contains="scalar-only select is not a resident row projection"
    ):
        _ = (
            dates.lazy()
            .select_exprs(
                [
                    lit("2000-02-29T12:34:56.123456")
                    .str()
                    .strptime("datetime[us]")
                    .alias("dt"),
                    lit("12:34:56.123456789")
                    .str()
                    .strptime("time")
                    .alias("clock"),
                ]
            )
            .collect(accelerator=unavailable)
        )
    with assert_raises(contains="positive fixed interval"):
        _ = (
            dates.lazy()
            .select(col("t").dt().truncate("1mo"))
            .collect(accelerator=unavailable)
        )
    var decimals = DataFrame(
        [
            Series("d", Column[Int128]([100])).with_dtype(
                DataType.decimal(38, 2, 128)
            )
        ]
    )
    with assert_raises(contains="Decimal128 input reductions"):
        _ = (
            decimals.lazy()
            .select(col("d").arg_min())
            .collect(accelerator=unavailable)
        )


def test_native_metal_temporal_and_decimal_unsupported_requests_reject_before_device_setup() raises:
    _ = require_metal()
    temporal_preflight_cases()


def verify_metal_temporal_cpu_models() raises:
    var runtime = MetalRuntime()
    assert_temporal_native_plan(extreme_date_plan())
    temporal_cases(runtime, False)
    truncation_cases(runtime, False)
    logical_cases(runtime, False)
    temporal_preflight_cases()


def test_metal_temporal_lowering_and_cpu_models() raises:
    verify_metal_temporal_cpu_models()
