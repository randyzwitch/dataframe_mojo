"""Hash grouping at admission boundaries, with exact reducer outcomes."""
from std.testing import assert_true, assert_equal, assert_raises
from std.memory import bitcast
from dataframe import Column, DataFrame, Series, StringColumn, col
from dataframe_metal.metal import MetalRuntime
from dataframe.partition import _mix, _combine
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


# requires: mojo_gpu
def test_mojo_gpu_hash_grouping_boundaries_and_integer_states() raises:
    for n in [0, 1, 2, 127, 255, 256, 257, 511, 512, 513, 1025]:
        var keys = List[Int64]()
        var valid = List[Bool]()
        var ids = List[Int64]()
        for row in range(n):
            # Ties straddle every block; the final block is often partial.
            keys.append(Int64((n - row) % 19 - 9))
            valid.append(row % 13 != 0)
            ids.append(Int64(row))
        var frame = DataFrame(
            [
                Series("k", Column[Int64](keys^, valid^)),
                Series("id", Column[Int64](ids^)),
            ]
        )
        var grouped = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(
                [
                    col("id").first().alias("first"),
                    col("id").last().alias("last"),
                    col("id").sum().alias("sum"),
                    col("id").count().alias("count"),
                    col("id").min().alias("min"),
                    col("id").max().alias("max"),
                    col("id").mean().alias("mean"),
                    col("id").len().alias("len"),
                    col("id").null_count().alias("null_count"),
                ]
            )
        )
        assert_true(
            grouped.collect(accelerator=MetalRuntime(), engine="accel").equals(
                grouped.collect(engine="cpu")
            )
        )


# requires: mojo_gpu
def test_mojo_gpu_hash_grouping_composite_null_nan_keys_and_source_order() raises:
    var keys = List[Int64]()
    var validity = List[Bool]()
    var floating = List[Float64]()
    var values = List[Int64]()
    var flags = List[Bool]()
    var labels = List[String]()
    var bits: List[UInt64] = [
        0,
        0x8000000000000000,
        0x7FF8000000000031,
        0xFFF8000000000042,
        0x3FF0000000000000,
    ]
    for row in range(4099):
        keys.append(Int64(row % 7))
        validity.append(row % 13 != 0)
        floating.append(bitcast[DType.float64](bits[row % len(bits)]))
        values.append(Int64(row % 29 - 14))
        flags.append(row % 2 == 0)
        labels.append(String(row % 5))
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^, validity.copy())),
            Series("z", Column[Float64](floating^)),
            Series("v", Column[Int64](values^, validity.copy())),
            Series("b", Column[Bool](flags^, validity^)),
            Series("s", StringColumn(labels^)),
        ]
    )
    var query = (
        frame.lazy()
        .group_by(["k", "z"], maintain_order=True)
        .agg(
            [
                col("v").sum().alias("sum"),
                col("v").mean().alias("mean"),
                col("v").min().alias("min"),
                col("v").max().alias("max"),
                col("v").count().alias("count"),
                col("v").len().alias("len"),
                col("v").null_count().alias("nulls"),
                col("z").first().alias("first"),
                col("z").last().alias("last"),
                col("b").any(ignore_nulls=False).alias("any"),
                col("b").all(ignore_nulls=False).alias("all"),
            ]
        )
    )
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    var expected = query.collect(engine="cpu")
    assert_true(actual.equals(expected))
    for name in ["z", "first", "last"]:
        for row in range(actual.height()):
            assert_equal(
                bitcast[DType.uint64](actual.column(name).float64()._get(row)),
                bitcast[DType.uint64](
                    expected.column(name).float64()._get(row)
                ),
            )
    # Order-sensitive aggregate states retain sorted grouping.
    var ordered = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg([col("z").implode().alias("items")])
    )
    assert_true(
        ordered.collect(accelerator=MetalRuntime(), engine="accel").equals(
            ordered.collect(engine="cpu")
        )
    )
    var strings = (
        frame.lazy()
        .group_by(["s", "k"], maintain_order=True)
        .agg(
            [
                col("v").min().alias("min"),
                col("v").max().alias("max"),
                col("v").count().alias("count"),
            ]
        )
    )
    assert_true(
        strings.collect(accelerator=MetalRuntime(), engine="accel").equals(
            strings.collect(engine="cpu")
        )
    )
    with assert_raises():
        _ = strings.collect(
            accelerator=MetalRuntime(memory_limit_bytes=1), engine="accel"
        )


# requires: mojo_gpu
def test_mojo_gpu_hash_grouping_unordered_integer_reductions() raises:
    var keys = List[Int64]()
    var values = List[Int64]()
    var valid = List[Bool]()
    var ids = List[Int64]()
    for row in range(4099):
        keys.append(Int64(row % 7))
        values.append(Int64((row * 37) % 101 - 50))
        valid.append(row % 13 != 0)
        ids.append(Int64(row))
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](values^, valid^)),
            Series("id", Column[Int64](ids^)),
        ]
    )
    # Group output order is unspecified; sorting makes the CPU and Metal
    # comparison explicit while leaving the grouped step hash-eligible.
    var grouped = (
        frame.lazy()
        .group_by("k")
        .agg(
            [
                col("v").sum().alias("sum"),
                col("v").count().alias("count"),
                col("v").min().alias("min"),
                col("v").max().alias("max"),
                col("v").mean().alias("mean"),
                col("id").first().alias("first"),
                col("id").last().alias("last"),
            ]
        )
        .sort("k")
    )
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var actual = grouped.collect(accelerator=MetalRuntime(), engine="accel")
    var expected = grouped.collect(engine="cpu")
    assert_true(actual.equals(expected))
    var paths = observed.text()
    assert_true("dataframe-path: metal.hash_grouping\n" in paths)
    _ = observed^
    _ = trace^


# requires: mojo_gpu
def test_mojo_gpu_hash_grouping_parallelism_requires_a_large_hash_group() raises:
    # Several uniformly sized groups should skip integer partials even when
    # their average size would otherwise admit multiple partials per group.
    var uniform_keys = List[Int32]()
    var uniform_values = List[Int64]()
    for group in range(4):
        for row in range(1024):
            uniform_keys.append(Int32(group))
            uniform_values.append(Int64(1))
    var uniform = DataFrame(
        [
            Series("k", Column[Int32](uniform_keys^)),
            Series("v", Column[Int64](uniform_values^)),
        ]
    )
    var uniform_query = (
        uniform.lazy().group_by("k").agg(col("v").sum().alias("sum")).sort("k")
    )
    var uniform_trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var uniform_observed = StrategyTrace()
    var uniform_profile = uniform_query.profile(
        engine="accel", accelerator=MetalRuntime()
    )
    assert_true(
        uniform_profile[0].equals(
            uniform_query.collect(engine="cpu", streaming=False)
        )
    )
    assert_equal(uniform_profile[0].height(), 4)
    for group in range(4):
        assert_equal(uniform_profile[0].item(group, "k").int32(), Int32(group))
        assert_equal(uniform_profile[0].item(group, "sum").int64(), Int64(1024))
    assert_true(
        "dataframe-path: metal.hash_grouping\n" in uniform_observed.text()
    )
    assert_equal(
        uniform_profile[1]
        .item(0, "grouped_int_max_qualifying_hash_group_rows")
        .int64(),
        Int64(0),
    )
    assert_equal(
        uniform_profile[1].item(0, "grouped_int_scratch_bytes").int64(),
        Int64(0),
    )
    var uniform_peak = Int(
        uniform_profile[1].item(0, "peak_requested_bytes").int64()
    )
    with assert_raises(contains="memory budget"):
        _ = uniform_query.collect(
            engine="accel",
            accelerator=MetalRuntime(memory_limit_bytes=uniform_peak - 1),
        )
    assert_true(
        uniform_query.collect(
            engine="accel",
            accelerator=MetalRuntime(memory_limit_bytes=uniform_peak),
        ).equals(uniform_profile[0])
    )
    _ = uniform_observed^
    _ = uniform_trace^

    # The 2k/4k/8k cases validate the chosen data threshold, and 4095 checks
    # that reaching 4096 is inclusive.
    for rows in [2048, 4095, 4096, 8192]:
        var keys = List[Int32](length=rows, fill=0)
        var values = List[Int64](length=rows, fill=1)
        var frame = DataFrame(
            [
                Series("k", Column[Int32](keys^)),
                Series("v", Column[Int64](values^)),
            ]
        )
        var query = frame.lazy().group_by("k").agg(col("v").sum().alias("sum"))
        var measured = query.profile(engine="accel", accelerator=MetalRuntime())
        assert_true(
            measured[0].equals(query.collect(engine="cpu", streaming=False))
        )
        assert_equal(measured[0].height(), 1)
        assert_equal(measured[0].item(0, "sum").int64(), Int64(rows))
        var expected_qualifying = Int64(0)
        if rows >= 4096:
            expected_qualifying = Int64(rows)
        assert_equal(
            measured[1]
            .item(0, "grouped_int_max_qualifying_hash_group_rows")
            .int64(),
            expected_qualifying,
        )
        if rows < 4096:
            assert_equal(
                measured[1].item(0, "grouped_int_scratch_bytes").int64(),
                Int64(0),
            )
        else:
            assert_true(
                measured[1]
                .item(0, "grouped_int_max_partials_per_group")
                .int64()
                >= 2
            )

    # One qualifying group must retain partials even though sixteen smaller
    # groups make the average far smaller than the admission threshold.
    var skew_rows = 8192
    var skew_keys = List[Int32](length=skew_rows, fill=0)
    var skew_values = List[Int64](length=skew_rows, fill=1)
    for row in range(4096, skew_rows):
        skew_keys[row] = Int32(1 + (row - 4096) // 256)
    var skewed = DataFrame(
        [
            Series("k", Column[Int32](skew_keys^)),
            Series("v", Column[Int64](skew_values^)),
        ]
    )
    var skew_query = (
        skewed.lazy().group_by("k").agg(col("v").sum().alias("sum")).sort("k")
    )
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var skew_profile = skew_query.profile(
        engine="accel", accelerator=MetalRuntime()
    )
    assert_true(
        skew_profile[0].equals(
            skew_query.collect(engine="cpu", streaming=False)
        )
    )
    assert_equal(skew_profile[0].height(), 17)
    assert_equal(skew_profile[0].item(0, "sum").int64(), Int64(4096))
    for group in range(1, 17):
        assert_equal(skew_profile[0].item(group, "k").int32(), Int32(group))
        assert_equal(skew_profile[0].item(group, "sum").int64(), Int64(256))
    assert_true(
        skew_profile[1].item(0, "grouped_int_max_partials_per_group").int64()
        >= 2
    )
    assert_true("dataframe-path: metal.hash_grouping\n" in observed.text())
    assert_equal(
        skew_profile[1]
        .item(0, "grouped_int_max_qualifying_hash_group_rows")
        .int64(),
        Int64(4096),
    )

    # A null-key group is still a real, observed hash group for admission.
    var null_keys = List[Int32](length=4096, fill=7)
    var null_validity = List[Bool](length=4096, fill=False)
    var null_values = List[Int64](length=4096, fill=1)
    var null_group = DataFrame(
        [
            Series("k", Column[Int32](null_keys^, null_validity^)),
            Series("v", Column[Int64](null_values^)),
        ]
    )
    var null_query = (
        null_group.lazy().group_by("k").agg(col("v").sum().alias("sum"))
    )
    var null_profile = null_query.profile(
        engine="accel", accelerator=MetalRuntime()
    )
    assert_true(
        null_profile[0].equals(
            null_query.collect(engine="cpu", streaming=False)
        )
    )
    assert_equal(null_profile[0].height(), 1)
    assert_equal(null_profile[0].item(0, "sum").int64(), Int64(4096))
    assert_equal(
        null_profile[1]
        .item(0, "grouped_int_max_qualifying_hash_group_rows")
        .int64(),
        Int64(4096),
    )
    assert_true(
        null_profile[1].item(0, "grouped_int_max_partials_per_group").int64()
        >= 2
    )
    _ = observed^
    _ = trace^


# requires: mojo_gpu
def test_mojo_gpu_hash_grouping_probe_exhaustion_and_checked_sum() raises:
    # Generate a long collision chain against the complete grouping hash.
    # 256 rows select 1024 slots; 129 distinct keys exhaust 128 probes.
    var keys = List[Int64]()
    var candidate = Int64(0)
    while len(keys) < 129:
        var hashed = _combine(_combine(UInt64(0), UInt64(1)), UInt64(candidate))
        if UInt32(_mix(hashed)) & 1023 == 0:
            keys.append(candidate)
        candidate += 1
    for row in range(127):
        keys.append(keys[row])
    var frame = DataFrame(
        [
            Series("k", Column[Int64](keys^)),
            Series("v", Column[Int64](List[Int64](length=256, fill=1))),
        ]
    )
    var query = (
        frame.lazy().group_by("k").agg([col("v").sum().alias("sum")]).sort("k")
    )
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    assert_true(
        query.collect(accelerator=MetalRuntime(), engine="accel").equals(
            query.collect(engine="cpu")
        )
    )
    var paths = observed.text()
    assert_true("dataframe-path: metal.hash_grouping_probe_fallback\n" in paths)
    _ = observed^
    _ = trace^
    var overflow = (
        DataFrame(
            [
                Series("k", Column[Int64](List[Int64](length=256, fill=0))),
                Series(
                    "v", Column[Int64](List[Int64](length=256, fill=Int64.MAX))
                ),
            ]
        )
        .lazy()
        .group_by("k")
        .agg([col("v").sum()])
    )
    with assert_raises():
        _ = overflow.collect(engine="cpu")
    with assert_raises():
        _ = overflow.collect(accelerator=MetalRuntime(), engine="accel")
    # A faulted hash grouping must not poison the next execution.
    assert_true(
        query.collect(accelerator=MetalRuntime(), engine="accel").equals(
            query.collect(engine="cpu")
        )
    )
