"""Fixed-zone fields preserve CPU local storage wrapping and negative floors."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from dataframe.metal import MetalRuntime
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows, ROW_FIXED_ZONE_FIELDS
from dataframe.accelerator.gpu.runtime import capabilities
from dataframe.accelerator.gpu.timezone_fields import local_ticks
from tests.support.coverage_compare import assert_frame


def _fields() -> List[Expr]:
    return [
        col("t").dt().year().alias("year"),
        col("t").dt().month().alias("month"),
        col("t").dt().day().alias("day"),
        col("t").dt().hour().alias("hour"),
        col("t").dt().minute().alias("minute"),
        col("t").dt().second().alias("second"),
        col("t").dt().nanosecond().alias("nano"),
        col("t").dt().weekday().alias("weekday"),
        col("t").dt().ordinal_day().alias("ordinal"),
        col("t").dt().date().alias("date"),
        col("t").dt().time().alias("time"),
    ]


def test_mojo_gpu_fixed_zone_fields_extreme_wrapping_words_and_independent_civil_anchors() raises:
    # Generated independently with arbitrary-integer wrapping and binary
    # search of Gregorian year-start ordinals; no production calendar helper.
    # Order: ms/us/ns, -23:59/+23:59, Int64.MIN/Int64.MAX.
    var shifted: List[UInt64] = [
        UInt64(0x7FFFFFFFFADA8E60),
        UInt64(0x7FFFFFFFFADA8E5F),
        UInt64(0x80000000052571A0),
        UInt64(0x800000000525719F),
        UInt64(0x7FFFFFEBE5BC2700),
        UInt64(0x7FFFFFEBE5BC26FF),
        UInt64(0x800000141A43D900),
        UInt64(0x800000141A43D8FF),
        UInt64(0x7FFFB17966F85800),
        UInt64(0x7FFFB17966F857FF),
        UInt64(0x80004E869907A800),
        UInt64(0x80004E869907A7FF),
    ]
    var expected: List[List[Int64]] = [
        [
            292278994,
            8,
            16,
            7,
            13,
            55,
            808000000,
            6,
            228,
            106751991166,
            26035808000000,
        ],
        [
            292278994,
            8,
            16,
            7,
            13,
            55,
            807000000,
            6,
            228,
            106751991166,
            26035807000000,
        ],
        [
            -292275055,
            5,
            17,
            16,
            46,
            4,
            192000000,
            1,
            137,
            -106751991167,
            60364192000000,
        ],
        [
            -292275055,
            5,
            17,
            16,
            46,
            4,
            191000000,
            1,
            137,
            -106751991167,
            60364191000000,
        ],
        [294247, 1, 9, 4, 1, 54, 775808000, 6, 9, 106751990, 14514775808000],
        [294247, 1, 9, 4, 1, 54, 775807000, 6, 9, 106751990, 14514775807000],
        [
            -290308,
            12,
            22,
            19,
            58,
            5,
            224192000,
            1,
            357,
            -106751991,
            71885224192000,
        ],
        [
            -290308,
            12,
            22,
            19,
            58,
            5,
            224191000,
            1,
            357,
            -106751991,
            71885224191000,
        ],
        [2262, 4, 10, 23, 48, 16, 854775808, 4, 100, 106750, 85696854775808],
        [2262, 4, 10, 23, 48, 16, 854775807, 4, 100, 106750, 85696854775807],
        [1677, 9, 22, 0, 11, 43, 145224192, 3, 265, -106751, 703145224192],
        [1677, 9, 22, 0, 11, 43, 145224191, 3, 265, -106751, 703145224191],
    ]
    var runtime = MetalRuntime()
    var at = 0
    for unit in ["ms", "us", "ns"]:
        for zone in ["-23:59", "+23:59"]:
            var dtype = DataType.datetime(unit, zone)
            var offset = Int64(-86340) if zone == "-23:59" else Int64(86340)
            var frame = DataFrame(
                [
                    Series(
                        "t",
                        Column[Int64](
                            [Int64.MIN, Int64.MAX, Int64.MIN, Int64.MAX],
                            [True, True, False, False],
                        ),
                    ).with_dtype(dtype)
                ]
            )
            var query = frame.lazy().select_exprs(_fields())
            var actual = query.collect(accelerator=runtime)
            assert_frame(actual, query.collect(engine="cpu"))
            assert_equal(actual.height(), 4)
            var names = actual.columns()
            assert_equal(len(names), 11)
            for row in range(2):
                var raw = Int64.MIN if row == 0 else Int64.MAX
                assert_equal(
                    local_ticks(
                        raw.cast[DType.uint64](), offset, dtype.per_second()
                    ),
                    shifted[at + row],
                )
                for column in range(len(names)):
                    var values = actual.column(names[column]).int64()
                    assert_equal(values.value(row), expected[at + row][column])
                    assert_true(not values.is_valid(row + 2))
            assert_equal(actual.column("date").dtype(), DataType.DATE)
            assert_equal(actual.column("time").dtype(), DataType.TIME)
            at += 2
    assert_equal(at, len(shifted))


def test_mojo_gpu_fixed_zone_fields_negative_floors_nullable_chunked_empty_and_conversion() raises:
    var runtime = MetalRuntime()
    for unit in ["ms", "us", "ns"]:
        for zone in [
            "-23:59",
            "-03:30",
            "-00:01",
            "-00:00",
            "+00:00",
            "+00:01",
            "+05:45",
            "+23:59",
        ]:
            var dtype = DataType.datetime(unit, zone)
            var per = dtype.per_second()
            var day = per * 86400
            var raw: List[Int64] = [
                -day - 1,
                -per - 1,
                -1,
                0,
                1,
                per - 1,
                day - 1,
                day + 1,
            ]
            for chunked in [False, True]:
                var values = List[Int64]()
                var valid = List[Bool]()
                for row in range(263):
                    values.append(raw[row % len(raw)])
                    valid.append(row % 7 != 3)
                var source = (
                    Series("t", Column[Int64](values^, valid^))
                    .with_dtype(dtype)
                    .slice(3, 257)
                )
                if chunked:
                    source = Series._from_chunks(
                        [
                            source.head(129),
                            source.slice(129, 0),
                            source.slice(129, 128),
                        ]
                    )
                var frame = DataFrame([source^])
                var query = frame.lazy().select_exprs(_fields())
                assert_frame(
                    query.collect(accelerator=runtime),
                    query.collect(engine="cpu"),
                )
                var empty = frame.head(0).lazy().select_exprs(_fields())
                assert_frame(
                    empty.collect(accelerator=runtime),
                    empty.collect(engine="cpu"),
                )
    var source = DataFrame(
        [
            Series(
                "t", Column[Int64]([-1, 0, 1, 0], [True, True, True, False])
            ).with_dtype(DataType.datetime("ns", "UTC"))
        ]
    )
    var query = source.lazy().select_exprs(
        [
            col("t").dt().convert_time_zone("+05:45").dt().hour().alias("east"),
            col("t").dt().convert_time_zone("-03:30").dt().hour().alias("west"),
            col("t")
            .dt()
            .convert_time_zone("-03:30")
            .dt()
            .date()
            .alias("west_date"),
        ]
    )
    var expected = DataFrame(
        [
            Series(
                "east", Column[Int64]([5, 5, 5, 0], [True, True, True, False])
            ),
            Series(
                "west",
                Column[Int64]([20, 20, 20, 0], [True, True, True, False]),
            ),
            Series(
                "west_date",
                Column[Int64]([-1, -1, -1, 0], [True, True, True, False]),
            ).with_dtype(DataType.DATE),
        ]
    )
    var profiled = query.profile(accelerator=runtime)
    assert_frame(profiled[0], expected)
    assert_frame(query.collect(engine="cpu"), expected)
    assert_equal(profiled[1].item(0, "executor").string(), "metal")
    assert_true(profiled[1].item(0, "kernel_launches").int64() > 0)


def test_mojo_gpu_fixed_zone_fields_capability_descriptor_and_unsupported_local_operations() raises:
    assert_true(not RowCapabilities().fixed_zone_temporal_fields)
    for zone in ["-23:59", "+00:00", "+05:45"]:
        var frame = DataFrame(
            [
                Series("t", Column[Int64]([0])).with_dtype(
                    DataType.datetime("us", zone)
                )
            ]
        )
        var query = frame.lazy().select(col("t").dt().hour())
        var caps = capabilities()
        caps.fixed_zone_temporal_fields = False
        with assert_raises(contains="time zone field extraction"):
            _ = lower_rows(query, caps)
        caps = capabilities()
        caps.utc_temporal_conversions = False
        var plan = lower_rows(query, caps)
        assert_equal(
            plan.code[7] & ROW_FIXED_ZONE_FIELDS, ROW_FIXED_ZONE_FIELDS
        )
        assert_equal(plan.code[7] & 7, Int64(2))
        assert_equal(plan.code[7] & ((1 << 27) - 8), Int64(0))
        assert_equal(
            plan.code[6],
            Int64(-86340) if zone
            == "-23:59" else Int64(20700) if zone
            == "+05:45" else Int64(0),
        )
        var unavailable = MetalRuntime(device_id=999999)
        for expression in [
            col("t").dt().truncate("1d"),
            col("t").dt().offset_by("1d"),
        ]:
            with assert_raises(contains="time zone field extraction"):
                _ = (
                    frame.lazy()
                    .select(expression)
                    .collect(accelerator=unavailable)
                )
        with assert_raises(contains="naive or UTC"):
            _ = (
                frame.lazy()
                .select(col("t").dt().replace_time_zone("UTC"))
                .collect(accelerator=unavailable)
            )
    var iana = DataFrame(
        [
            Series("t", Column[Int64]([0])).with_dtype(
                DataType.datetime("us", "America/New_York")
            )
        ]
    )
    with assert_raises(contains="time zone field extraction"):
        _ = (
            iana.lazy()
            .select(col("t").dt().hour())
            .collect(accelerator=MetalRuntime(device_id=999999))
        )
