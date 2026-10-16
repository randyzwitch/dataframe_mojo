"""Civil transition contracts use an independent piecewise-offset model."""
# contract-group: expr.temporal
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, DataType, Series, Column, col
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.temporal_transitions import (
    transitions,
    fields,
    formatted_offset,
    truncate_wall,
)


def temporal_series(
    name: String, dtype: DataType, ticks: List[Int64], valid: List[Bool]
) raises -> Series:
    return Series(name, Column[Int64](ticks.copy(), valid.copy())).with_dtype(
        dtype
    )


def test_time_zone_transition_models_cover_political_half_hour_historical_footer_and_dst_boundaries() raises:
    var names: List[String] = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "weekday",
        "ordinal",
        "date",
        "time",
    ]
    for change in transitions():
        for unit in ["ms", "us", "ns"]:
            var dtype = DataType.datetime(unit, change.zone)
            var per = dtype.per_second()
            var ticks = List[Int64]()
            var wall_ticks = List[Int64]()
            var valid: List[Bool] = [True, True, True, True, True, False]
            var columns = List[List[Int64]]()
            for _ in range(10):
                columns.append(List[Int64]())
            var offset_text = List[String]()
            for delta in [-3600, -1, 0, 1, 3600, 0]:
                var utc = change.utc + Int64(delta)
                var wall = utc + change.offset(utc)
                ticks.append(utc * per)
                wall_ticks.append(wall * per)
                var expected = fields(wall)
                for at in range(10):
                    columns[at].append(expected[at])
                offset_text.append(formatted_offset(change.offset(utc)))
            var source = DataFrame(
                [
                    temporal_series(
                        "t", DataType.datetime(unit, "UTC"), ticks, valid
                    )
                ]
            )
            var expected_columns = List[Series]()
            for at in range(10):
                var field_dtype = (
                    DataType.DATE if at
                    == 8 else DataType.TIME if at
                    == 9 else DataType.INT64
                )
                expected_columns.append(
                    temporal_series(names[at], field_dtype, columns[at], valid)
                )
            expected_columns.append(
                Series("offset", Column[String](offset_text^, valid))
            )
            for layout in range(3):
                var input = (
                    source.copy() if layout
                    == 0 else offset_layout(source) if layout
                    == 1 else chunk_layout(source)
                )
                var zoned = input.select(
                    col("t").dt().convert_time_zone(change.zone)
                )
                assert_frame(
                    zoned,
                    DataFrame([temporal_series("t", dtype, ticks, valid)]),
                )
                var actual = zoned.select_exprs(
                    [
                        col("t").dt().year().alias("year"),
                        col("t").dt().month().alias("month"),
                        col("t").dt().day().alias("day"),
                        col("t").dt().hour().alias("hour"),
                        col("t").dt().minute().alias("minute"),
                        col("t").dt().second().alias("second"),
                        col("t").dt().weekday().alias("weekday"),
                        col("t").dt().ordinal_day().alias("ordinal"),
                        col("t").dt().date().alias("date"),
                        col("t").dt().time().alias("time"),
                        col("t").dt().strftime("%z %:z").alias("offset"),
                    ]
                )
                assert_frame(actual, DataFrame(expected_columns.copy()))
                assert_frame(
                    zoned.select(col("t").dt().replace_time_zone("")),
                    DataFrame(
                        [
                            temporal_series(
                                "t", DataType.datetime(unit), wall_ticks, valid
                            )
                        ]
                    ),
                )
            # Localizing the middle of the changed wall-time interval isolates
            # ambiguous and nonexistent policies independently of formatting.
            var midpoint = (
                change.utc
                + min(change.before, change.after)
                + abs(change.after - change.before) // 2
            )
            var candidates = change.candidates(midpoint)
            var naive = DataFrame(
                [
                    temporal_series(
                        "t",
                        DataType.datetime(unit),
                        [midpoint * per, Int64.MIN],
                        [True, False],
                    )
                ]
            )
            if len(candidates) == 0:
                with assert_raises(contains="non-existent"):
                    _ = naive.select(
                        col("t").dt().replace_time_zone(change.zone)
                    )
                assert_true(
                    naive.select(
                        col("t")
                        .dt()
                        .replace_time_zone(change.zone, non_existent="null")
                    )
                    .item(0, "t")
                    .is_null()
                )
            else:
                assert_equal(len(candidates), 2)
                for policy in ["earliest", "latest"]:
                    var chosen = candidates[0 if policy == "earliest" else 1]
                    assert_frame(
                        naive.select(
                            col("t")
                            .dt()
                            .replace_time_zone(change.zone, ambiguous=policy)
                        ),
                        DataFrame(
                            [
                                temporal_series(
                                    "t", dtype, [chosen * per, 0], [True, False]
                                )
                            ]
                        ),
                    )
                with assert_raises(contains="ambiguous"):
                    _ = naive.select(
                        col("t").dt().replace_time_zone(change.zone)
                    )
                assert_true(
                    naive.select(
                        col("t")
                        .dt()
                        .replace_time_zone(change.zone, ambiguous="null")
                    )
                    .item(0, "t")
                    .is_null()
                )
            var before = change.utc - 43200
            var calendar_target = before + change.offset(before) + 86400
            var shifted_candidates = change.candidates(calendar_target)
            var shifted = DataFrame(
                [temporal_series("t", dtype, [before * per], [True])]
            )
            assert_frame(
                shifted.select(col("t").dt().offset_by("24h")),
                DataFrame(
                    [
                        temporal_series(
                            "t", dtype, [(before + 86400) * per], [True]
                        )
                    ]
                ),
            )
            if not len(shifted_candidates):
                with assert_raises(contains="non-existent"):
                    _ = shifted.select(col("t").dt().offset_by("1d"))
            else:
                assert_equal(len(shifted_candidates), 1)
                assert_frame(
                    shifted.select(col("t").dt().offset_by("1d")),
                    DataFrame(
                        [
                            temporal_series(
                                "t",
                                dtype,
                                [shifted_candidates[0] * per],
                                [True],
                            )
                        ]
                    ),
                )
            for period in ["1d", "1w", "1mo"]:
                for delta in [-1, 1]:
                    var utc = change.utc + Int64(delta)
                    var wall = utc + change.offset(utc)
                    var rounded = change.candidates(truncate_wall(wall, period))
                    assert_equal(len(rounded), 1)
                    var one = DataFrame(
                        [temporal_series("t", dtype, [utc * per], [True])]
                    )
                    assert_frame(
                        one.select(col("t").dt().truncate(period)),
                        DataFrame(
                            [
                                temporal_series(
                                    "t", dtype, [rounded[0] * per], [True]
                                )
                            ]
                        ),
                    )
