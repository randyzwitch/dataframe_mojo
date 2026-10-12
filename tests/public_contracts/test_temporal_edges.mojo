"""Parser field boundaries, zone rule horizons and unsigned time aggregates."""
from std.testing import assert_equal, assert_true, assert_raises
from std.math import sqrt
from std.sys import size_of
from tests.support.numeric_model import INTEGER_DTYPES, FLOAT_DTYPES
from tests.support.temporal_reference import members
from dataframe import Column, DataFrame, DataType, Series, col
from tests.public_contracts.test_temporal import tagged, assert_ticks
from tests.support.temporal_reference import civil_ns, civil_days
from tests.support.coverage_generators import chunk_layout, offset_layout


def test_temporal_packed_fields_and_iso_zone_designators_have_literal_epoch_meaning() raises:
    for unit in ["ms", "us", "ns"]:
        var divisor = Int64(
            1000000 if unit == "ms" else 1000 if unit == "us" else 1
        )
        var input = DataFrame(
            [
                Series(
                    "s",
                    Column[String](
                        [
                            "1970-01-01T00:00:00Z",
                            "1970-01-01T00:00:00z",
                            "1970-01-01T01:30:00+0130",
                            "1970-01-01T01:30:00+01:30",
                            "1970-01-01T01:00:00+01",
                            "1969-12-31T23:00:00-01:00",
                            "1970-01-01T00:30:00+01:00",
                        ]
                    ),
                )
            ]
        )
        assert_ticks(
            input.select(col("s").str().to_datetime(unit=unit)),
            "s",
            DataType.datetime(unit),
            [0, 0, 0, 0, 0, 0, -1800000000000 // divisor],
        )
    var packed = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    [
                        "20240229",
                        "19000301",
                        "19691231",
                        "00010101",
                    ]
                ),
            )
        ]
    )
    assert_ticks(
        packed.select(col("s").str().to_date("%Y%m%d")),
        "s",
        DataType.DATE,
        [civil_days(2024, 2, 29), civil_days(1900, 3, 1), -1, -719162],
    )
    var stamp = DataFrame(
        [Series("s", Column[String](["19700101000001", "19691231235959"]))]
    )
    assert_ticks(
        stamp.select(col("s").str().to_datetime("%Y%m%d%H%M%S")),
        "s",
        DataType.datetime("us"),
        [1000000, -1000000],
    )
    for bad in ["2024229", "202402290", "20230229"]:
        with assert_raises(contains="cannot parse"):
            _ = DataFrame([Series("s", Column[String]([bad]))]).select(
                col("s").str().to_date("%Y%m%d")
            )
    for bad in [
        "1970-01-01T00:00:00+24:00",
        "1970-01-01T00:00:00+01:60",
        "1970-01-01T00:00:00+1:00",
        "1970-01-01T00:00:00Q",
        "1970-01-01T00:00:00+01:00x",
    ]:
        with assert_raises(contains="cannot parse"):
            _ = DataFrame([Series("s", Column[String]([bad]))]).select(
                col("s").str().to_datetime()
            )
    for kind in ["date", "time"]:
        var invalid = "1970-01-01Z" if kind == "date" else "00:00:00Z"
        with assert_raises(contains="cannot parse"):
            _ = DataFrame([Series("s", Column[String]([invalid]))]).select(
                col("s").str().strptime(kind)
            )


def test_temporal_zone_footer_historical_and_half_hour_rules_apply_to_public_fields() raises:
    var zones: List[String] = [
        "America/New_York",
        "Australia/Lord_Howe",
        "+05:30",
        "-03:00",
    ]
    var ticks: List[Int64] = [
        civil_ns(2045, 7, 10) // 1000,
        civil_ns(2044, 12, 31) // 1000,
        civil_ns(1918, 6, 1) // 1000,
        civil_ns(2024, 1, 1) // 1000,
        civil_ns(2024, 6, 1) // 1000,
    ]
    var offsets: List[List[Int64]] = [
        [-14400, -18000, -14400, -18000, -14400],
        [37800, 39600, 36000, 39600, 37800],
        [19800, 19800, 19800, 19800, 19800],
        [-10800, -10800, -10800, -10800, -10800],
    ]
    for index in range(len(zones)):
        var source = DataFrame(
            [tagged("t", DataType.datetime("us", "UTC"), ticks)]
        )
        var zoned = source.select(col("t").dt().convert_time_zone(zones[index]))
        assert_ticks(zoned, "t", DataType.datetime("us", zones[index]), ticks)
        var wall = zoned.select(col("t").dt().replace_time_zone(""))
        var expected = List[Int64]()
        for row in range(len(ticks)):
            expected.append(ticks[row] + offsets[index][row] * 1000000)
        assert_ticks(wall, "t", DataType.datetime("us"), expected)
        var back = wall.select(col("t").dt().replace_time_zone(zones[index]))
        assert_ticks(back, "t", DataType.datetime("us", zones[index]), ticks)
    # A whole civil day was skipped when Apia changed UTC offset in 2011.
    var apia = DataFrame(
        [
            tagged(
                "t",
                DataType.datetime("us", "Pacific/Apia"),
                [civil_ns(2011, 12, 30, 10, 14) // 1000],
            )
        ]
    )
    with assert_raises(contains="non-existent in time zone 'Pacific/Apia'"):
        _ = apia.select(col("t").dt().offset_by("-1d"))


def test_time_windows_unsigned_extreme_totals_and_null_payloads_keep_exact_width() raises:
    var source = DataFrame(
        [
            tagged("t", DataType.DATE, [0, 1, 1, 2]),
            Series(
                "v",
                Column[UInt64](
                    [UInt64.MAX - 1, 1, UInt64.MAX, 0],
                    [True, True, False, True],
                ),
            ),
        ]
    )
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        var result = frame.select(col("v").rolling_sum_by("t", "2d"))
        assert_equal(result.column("v").dtype(), DataType.UINT64)
        var expected: List[UInt64] = [UInt64.MAX - 1, UInt64.MAX, UInt64.MAX, 1]
        for row in range(4):
            assert_equal(result.item(row, "v").uint64(), expected[row])
        var grouped = frame.rolling("t", "2d").agg(col("v").sum())
        assert_equal(grouped.column("v").dtype(), DataType.UINT64)
        for row in range(4):
            assert_equal(grouped.item(row, "v").uint64(), expected[row])


def test_time_window_moments_keep_centered_residual_precision_with_duplicates_and_evictions() raises:
    var times = List[Int64]()
    var values = List[Float64]()
    var valid = List[Bool]()
    for row in range(1001):
        times.append(Int64(row // 2))
        values.append(1e9 + Float64(row % 7) / 8)
        valid.append(row % 13 != 0)
    var source = DataFrame(
        [
            tagged("t", DataType.DATE, times),
            Series("v", Column[Float64](values.copy(), valid.copy())),
        ]
    )
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        for ddof in [0, 1, 2]:
            var result = frame.select_exprs(
                [
                    col("v")
                    .rolling_var_by("t", "30d", ddof=ddof)
                    .alias("variance"),
                    col("v")
                    .rolling_std_by("t", "30d", ddof=ddof)
                    .alias("deviation"),
                ]
            )
            for row in range(len(times)):
                var centered = List[Float64]()
                var anchor: Float64 = 0
                for member in range(len(times)):
                    if (
                        times[member] > times[row] - 30
                        and times[member] <= times[row]
                        and valid[member]
                    ):
                        if not len(centered):
                            anchor = values[member]
                        centered.append(values[member] - anchor)
                for name in ["variance", "deviation"]:
                    assert_equal(
                        result.item(row, name).is_null(), len(centered) <= ddof
                    )
                if len(centered) <= ddof:
                    continue
                var mean: Float64 = 0
                for value in centered:
                    mean += value
                mean /= Float64(len(centered))
                var squares: Float64 = 0
                for value in centered:
                    squares += (value - mean) * (value - mean)
                var expected = squares / Float64(len(centered) - ddof)
                assert_true(
                    abs(result.item(row, "variance").float64() - expected)
                    <= 1e-12 * (1 + expected),
                    "time-window variance must retain centered residual precision",
                )
                assert_true(
                    abs(
                        result.item(row, "deviation").float64() - sqrt(expected)
                    )
                    <= 1e-12 * (1 + sqrt(expected)),
                    "time-window standard deviation must retain centered residual precision",
                )


def typed_time_windows[D: DType]() raises:
    var times: List[Int64] = [0, 1, 1, 3, 10]
    var raw: List[Int] = [1, 4, 99, 2, 7]
    var valid: List[Bool] = [True, True, False, True, True]
    var payload = List[Scalar[D]]()
    for value in raw:
        payload.append(Scalar[D](value))
    var source = DataFrame(
        [
            tagged("t", DataType.DATE, times),
            Series("v", Column[Scalar[D]](payload^, valid.copy())),
        ]
    )
    comptime S = DType.int64 if D.is_integral() and size_of[
        Scalar[D]
    ]() <= 2 else D
    for layout in range(3):
        var frame = (
            source.copy() if layout
            == 0 else offset_layout(source) if layout
            == 1 else chunk_layout(source)
        )
        for closed in ["left", "right", "both", "none"]:
            for needed in [0, 1]:
                var output = frame.select_exprs(
                    [
                        col("v")
                        .rolling_sum_by(
                            "t", "2d", closed=closed, min_samples=needed
                        )
                        .alias("sum"),
                        col("v")
                        .rolling_min_by(
                            "t", "2d", closed=closed, min_samples=needed
                        )
                        .alias("min"),
                        col("v")
                        .rolling_max_by(
                            "t", "2d", closed=closed, min_samples=needed
                        )
                        .alias("max"),
                        col("v")
                        .rolling_mean_by(
                            "t", "2d", closed=closed, min_samples=needed
                        )
                        .alias("mean"),
                        col("v")
                        .rolling_var_by(
                            "t", "2d", closed=closed, min_samples=needed, ddof=1
                        )
                        .alias("var"),
                        col("v")
                        .rolling_std_by(
                            "t", "2d", closed=closed, min_samples=needed, ddof=1
                        )
                        .alias("std"),
                    ]
                )
                assert_equal(output.column("sum").dtype(), DataType.of(S))
                for name in ["min", "max"]:
                    assert_equal(output.column(name).dtype(), DataType.of(D))
                for name in ["mean", "var", "std"]:
                    assert_equal(output.column(name).dtype(), DataType.FLOAT64)
                for row in range(len(times)):
                    var selected = members(
                        times, times[row] - 2, times[row], closed
                    )
                    var count = 0
                    var total: Float64 = 0
                    var low = 999
                    var high = -999
                    for member in selected:
                        if valid[member]:
                            count += 1
                            total += Float64(raw[member])
                            low = min(low, raw[member])
                            high = max(high, raw[member])
                    assert_equal(
                        output.item(row, "sum").is_null(), count < needed
                    )
                    if count >= needed:
                        assert_equal(
                            output.item(row, "sum").numeric[S](),
                            Scalar[S](total),
                        )
                    for name in ["min", "max", "mean"]:
                        assert_equal(
                            output.item(row, name).is_null(),
                            count < needed or count == 0,
                        )
                    for name in ["var", "std"]:
                        assert_equal(
                            output.item(row, name).is_null(),
                            count < needed or count <= 1,
                        )
                    if not count:
                        continue
                    assert_equal(
                        output.item(row, "min").numeric[D](), Scalar[D](low)
                    )
                    assert_equal(
                        output.item(row, "max").numeric[D](), Scalar[D](high)
                    )
                    var mean = total / Float64(count)
                    assert_true(
                        abs(output.item(row, "mean").float64() - mean) < 1e-12
                    )
                    if count > 1:
                        var squares: Float64 = 0
                        for member in selected:
                            if valid[member]:
                                squares += (Float64(raw[member]) - mean) ** 2
                        var variance = squares / Float64(count - 1)
                        assert_true(
                            abs(output.item(row, "var").float64() - variance)
                            < 1e-12
                        )
                        assert_true(
                            abs(
                                output.item(row, "std").float64()
                                - sqrt(variance)
                            )
                            < 1e-12
                        )


def test_time_window_all_numeric_widths_follow_scalar_memberships_and_result_schema() raises:
    comptime for index in range(len(INTEGER_DTYPES)):
        typed_time_windows[INTEGER_DTYPES[index]]()
    comptime for index in range(len(FLOAT_DTYPES)):
        typed_time_windows[FLOAT_DTYPES[index]]()
