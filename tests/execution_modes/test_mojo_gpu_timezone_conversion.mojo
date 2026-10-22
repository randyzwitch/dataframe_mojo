"""Zone metadata conversion keeps exact ticks and uses the GPU row executor."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Expr, Series, LazyFrame, col
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.rows import lower_rows
from dataframe_metal.accelerator.gpu.runtime import capabilities
from tests.support.coverage_compare import assert_frame


def _check_local_literals(
    query: LazyFrame,
    dtype: DataType,
    ticks: List[Int64],
    valid: List[Bool] = List[Bool](),
) raises:
    var mask = valid.copy() if len(valid) else List[Bool](
        length=len(ticks), fill=True
    )
    var expected = DataFrame(
        [Series("t", Column[Int64](ticks.copy(), mask^)).with_dtype(dtype)]
    )
    assert_frame(query.collect(engine="cpu", optimize=False), expected)
    assert_frame(
        query.collect(accelerator=MetalRuntime(), optimize=False), expected
    )
    # Admission still depends on the private mutation capability and an
    # original terminal scope; disabling it must fail before device access.
    var disabled = capabilities()
    disabled.iana_zone_mutations = False
    with assert_raises(contains="time zone"):
        _ = lower_rows(query, disabled, ordered_source=True)


def _assert_ticks(actual: DataFrame, source: Series, dtype: DataType) raises:
    assert_equal(actual.columns(), ["converted"])
    assert_equal(actual.height(), len(source))
    assert_equal(actual.column("converted").dtype(), dtype)
    var values = actual.column("converted").int64()
    var expected = source.int64()
    for row in range(len(source)):
        assert_equal(values.is_valid(row), expected.is_valid(row))
        if expected.is_valid(row):
            assert_equal(values.value(row), expected.value(row))


# requires: metal
def test_mojo_gpu_timezone_conversion_extreme_ticks_nullable_layouts_and_metadata() raises:
    var raw: List[Int64] = [
        Int64.MIN,
        Int64.MIN + 1,
        -1000000001,
        -1,
        0,
        1,
        1000000001,
        Int64.MAX - 1,
        Int64.MAX,
    ]
    var sources: List[String] = ["", "UTC", "America/New_York", "+05:45"]
    var targets: List[String] = [
        "America/New_York",
        "Australia/Lord_Howe",
        "+05:45",
        "utc",
    ]
    var runtime = MetalRuntime()
    for unit in ["ms", "us", "ns"]:
        for source_zone in sources:
            for chunked in [False, True]:
                var values = List[Int64]()
                var valid = List[Bool]()
                for row in range(263):
                    values.append(raw[row % len(raw)])
                    valid.append(row % 11 != 5)
                var source = (
                    Series("t", Column[Int64](values^, valid^))
                    .with_dtype(DataType.datetime(unit, source_zone))
                    .slice(3, 257)
                )
                if chunked:
                    source = Series._from_chunks(
                        [
                            source.slice(0, 129),
                            source.slice(129, 0),
                            source.slice(129, 128),
                        ]
                    )
                var frame = DataFrame([source.copy()])
                for target in targets:
                    var query = frame.lazy().select(
                        col("t")
                        .dt()
                        .convert_time_zone(target)
                        .alias("converted")
                    )
                    var dtype = DataType.datetime(unit, target)
                    var actual = query.collect(accelerator=runtime)
                    _assert_ticks(actual, source, dtype)
                    assert_frame(actual, query.collect(engine="cpu"))
                    var empty = (
                        frame.head(0)
                        .lazy()
                        .select(
                            col("t")
                            .dt()
                            .convert_time_zone(target)
                            .alias("converted")
                        )
                    )
                    _assert_ticks(
                        empty.collect(accelerator=runtime),
                        source.head(0),
                        dtype,
                    )


# requires: metal
def test_mojo_gpu_timezone_conversion_composes_with_gpu_rows_and_utc_roundtrip() raises:
    var source = Series(
        "t",
        Column[Int64](
            [0, -1, Int64.MAX, Int64.MIN, Int64.MAX],
            [True, True, True, True, False],
        ),
    ).with_dtype(DataType.datetime("ns"))
    var frame = DataFrame(
        [source.copy(), Series("id", Column[Int64]([0, 1, 2, 3, 4]))]
    )
    var runtime = MetalRuntime()
    var query = (
        frame.lazy()
        .with_columns(
            col("t").dt().convert_time_zone("America/New_York").alias("local")
        )
        .select(["local", "id"])
        .filter(col("local").is_not_null())
        .sort(["local"])
        .slice(1, 2)
    )
    var expected = DataFrame(
        [
            Series("local", Column[Int64]([-1, 0])).with_dtype(
                DataType.datetime("ns", "America/New_York")
            ),
            Series("id", Column[Int64]([1, 0])),
        ]
    )
    var profiled = query.profile(accelerator=runtime)
    assert_frame(profiled[0], expected)
    assert_frame(query.collect(engine="cpu"), expected)
    assert_equal(profiled[1].item(0, "executor").string(), "metal")
    assert_true(profiled[1].item(0, "kernel_launches").int64() > 0)
    var roundtrip = frame.lazy().select(
        col("t")
        .dt()
        .convert_time_zone("America/New_York")
        .dt()
        .convert_time_zone("UTC")
        .dt()
        .convert_time_zone("Australia/Lord_Howe")
        .alias("converted")
    )
    var actual = roundtrip.collect(accelerator=runtime)
    _assert_ticks(
        actual, source, DataType.datetime("ns", "Australia/Lord_Howe")
    )
    assert_frame(actual, roundtrip.collect(engine="cpu"))


# requires: metal
def test_mojo_gpu_timezone_conversion_validates_zones_and_retains_local_operation_refusals() raises:
    var frame = DataFrame(
        [
            Series(
                "t",
                Column[Int64](
                    [0, 1615705200000000, 1636263000000000, Int64.MAX],
                    [True, True, True, False],
                ),
            ).with_dtype(DataType.datetime("us", "UTC"))
        ]
    )
    var unavailable = MetalRuntime(device_id=999999)
    var zoned = col("t").dt().convert_time_zone("America/New_York")
    for expression in [zoned.dt().year(), zoned.dt().hour()]:
        var query = frame.lazy().select(expression)
        _ = query.collect(engine="cpu")
        var named_caps = capabilities()
        assert_true(named_caps.iana_zone_temporal_fields)
        _ = lower_rows(query, named_caps)
        named_caps.iana_zone_temporal_fields = False
        with assert_raises(contains="time zone field extraction"):
            _ = lower_rows(query, named_caps)
    # Independent Gregorian/zoneinfo literals: NY midnight, calendar +1d
    # across DST, and a fixed +05:45 midnight. Null payloads stay unobserved.
    var fields: List[Expr] = [
        zoned.dt().truncate("1d"),
        zoned.dt().offset_by("1d"),
        col("t").dt().convert_time_zone("+05:45").dt().truncate("1d"),
    ]
    var literals: List[List[Int64]] = [
        [-68400000000, 1615698000000000, 1636257600000000, 0],
        [86400000000, 1615791600000000, 1636353000000000, 0],
        [-20700000000, 1615659300000000, 1636222500000000, 0],
    ]
    for index in range(len(fields)):
        _check_local_literals(
            frame.lazy().select(fields[index]),
            DataType.datetime(
                "us", "+05:45" if index == 2 else "America/New_York"
            ),
            literals[index],
            [True, True, True, False],
        )
    var targets: List[String] = ["", "UTC", "+05:45"]
    for index in range(len(targets)):
        var wall: List[Int64] = [
            -18000000000,
            1615690800000000,
            1636248600000000,
            0,
        ]
        if index == 2:
            wall = [-38700000000, 1615670100000000, 1636227900000000, 0]
        _check_local_literals(
            frame.lazy().select(zoned.dt().replace_time_zone(targets[index])),
            DataType.datetime("us", targets[index]),
            wall,
            [True, True, True, False],
        )
    for target in ["../invalid", "Dataframe/No_Such_Zone"]:
        var query = frame.lazy().select(col("t").dt().convert_time_zone(target))
        with assert_raises(contains="time zone"):
            _ = query.collect(engine="cpu")
        with assert_raises(contains="time zone"):
            _ = query.collect(accelerator=unavailable)
    var disabled = capabilities()
    disabled.utc_temporal_conversions = False
    var query = frame.lazy().select(zoned.alias("converted"))
    with assert_raises(contains="resident expression operation"):
        _ = lower_rows(query, disabled)
    var actual = query.collect(accelerator=MetalRuntime())
    _assert_ticks(
        actual, frame.column("t"), DataType.datetime("us", "America/New_York")
    )
