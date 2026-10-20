"""Hash grouping at admission boundaries, with exact reducer outcomes."""
from std.testing import assert_true, assert_equal, assert_raises
from std.memory import bitcast
from dataframe import Column, DataFrame, Series, StringColumn, col
from dataframe.metal import MetalRuntime
from dataframe.partition import _mix, _combine


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
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg([col("v").sum().alias("sum")])
    )
    assert_true(
        query.collect(accelerator=MetalRuntime(), engine="accel").equals(
            query.collect(engine="cpu")
        )
    )
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
