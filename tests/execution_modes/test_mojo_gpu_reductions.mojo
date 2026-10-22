"""Direct Mojo GPU reductions with scalar anchors and CPU contracts."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe_metal.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_float64_emulation import same


def _integer_case[D: DType](n: Int, chunked: Bool) raises:
    var values = List[Scalar[D]]()
    var valid = List[Bool]()
    for row in range(n + 6):
        values.append(Scalar[D](row % 11 + 1))
        valid.append(row % 7 != 0)
    var source = Series("x", Column[Scalar[D]](values^, valid^)).slice(3, n)
    if chunked:
        source = Series._from_chunks(
            [
                source.slice(0, n // 2),
                source.slice(n // 2, 0),
                source.slice(n // 2, n - n // 2),
            ]
        )
    var frame = DataFrame([source^])
    var query = frame.lazy().select_exprs(
        [
            col("x").sum().alias("sum"),
            col("x").sum(min_count=n + 1).alias("insufficient"),
            col("x").mean().alias("mean"),
            col("x").min().alias("min"),
            col("x").max().alias("max"),
            col("x").arg_min().alias("arg_min"),
            col("x").arg_max().alias("arg_max"),
            col("x").first().alias("first"),
            col("x").last().alias("last"),
            col("x").count().alias("count"),
            col("x").null_count().alias("nulls"),
            col("x").len().alias("len"),
        ]
    )
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    assert_equal(actual.height(), 1)
    assert_equal(actual.item(0, "first").is_null(), n == 0)
    assert_equal(actual.item(0, "last").is_null(), n == 0 or (n + 2) % 7 == 0)
    if n:
        assert_equal(actual.item(0, "first").numeric[D](), Scalar[D](4))
        if (n + 2) % 7 != 0:
            assert_equal(
                actual.item(0, "last").numeric[D](), Scalar[D]((n + 2) % 11 + 1)
            )
    assert_equal(actual.column("sum").dtype(), DataType.of(D).sum_type())
    var count = 0
    var total = Int64(0)
    var minimum = -1
    var maximum = -1
    for row in range(n):
        if (row + 3) % 7 == 0:
            continue
        var value = (row + 3) % 11 + 1
        count += 1
        total += Int64(value)
        if minimum < 0 or value < (minimum + 3) % 11 + 1:
            minimum = row
        if maximum < 0 or value > (maximum + 3) % 11 + 1:
            maximum = row
    assert_equal(
        actual.column("sum").cast(DataType.INT64).int64()._get(0), total
    )
    assert_equal(actual.item(0, "count").int64(), Int64(count))
    assert_equal(actual.item(0, "nulls").int64(), Int64(n - count))
    assert_equal(actual.item(0, "len").int64(), Int64(n))
    assert_true(actual.item(0, "insufficient").is_null())
    assert_equal(actual.item(0, "mean").is_null(), count == 0)
    if count:
        assert_equal(
            actual.item(0, "mean").float64(), Float64(total) / Float64(count)
        )
        assert_equal(
            actual.item(0, "arg_min").numeric[DType.uint32](), UInt32(minimum)
        )
        assert_equal(
            actual.item(0, "arg_max").numeric[DType.uint32](), UInt32(maximum)
        )
    same(actual, query.collect(engine="cpu", streaming=False))


# requires: mojo_gpu
def test_mojo_gpu_integer_reductions_layouts_empty_nulls_and_exact_results() raises:
    comptime types = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    )
    comptime for i in range(len(types)):
        for n in [0, 1, 257, 1_027]:
            _integer_case[types[i]](n, False)
            _integer_case[types[i]](n, True)


# requires: mojo_gpu
def test_mojo_gpu_wide_reduction_cancellation_overflow_and_mean() raises:
    var runtime = MetalRuntime()
    for n in [4, 257, 65_536, 65_537]:
        var signed = List[Int64]()
        var unsigned = List[UInt64]()
        var valid = List[Bool]()
        for row in range(n):
            signed.append(Int64.MAX if row % 4 < 2 else -Int64.MAX)
            unsigned.append(UInt64.MAX - UInt64(row % 7))
            valid.append(row % 4 != 3)
        var frame = DataFrame(
            [
                Series("i", Column[Int64](signed^)),
                Series("u", Column[UInt64](unsigned^, valid^)),
            ]
        )
        var query = frame.lazy().select_exprs(
            [
                col("i").mean().alias("signed"),
                col("u").mean().alias("unsigned"),
                col("u").sum(min_count=n + 1).alias("suppressed"),
            ]
        )
        same(
            query.collect(accelerator=runtime, engine="accel"),
            query.collect(engine="cpu", streaming=False),
        )
        var cancellation = frame.lazy().head(n - n % 4).select(col("i").sum())
        var result = cancellation.collect(accelerator=runtime, engine="accel")
        assert_equal(result.item().int64(), Int64(0))
        var sliced = frame.lazy().slice(1, 257).select(col("u").mean())
        same(
            sliced.collect(accelerator=runtime, engine="accel"),
            sliced.collect(engine="cpu", streaming=False),
        )
    var limits = DataFrame(
        [
            Series(
                "i", Column[Int64]([Int64.MIN, Int64.MIN, Int64.MAX, Int64.MAX])
            ),
            Series(
                "u",
                Column[UInt64](
                    [UInt64.MAX, UInt64.MAX, UInt64.MAX, UInt64.MAX]
                ),
            ),
        ]
    )
    var query = limits.lazy().select_exprs(
        [
            col("i").sum().alias("s"),
            col("i").mean().alias("m"),
            col("u").mean().alias("u"),
        ]
    )
    var result = query.collect(accelerator=runtime, engine="accel")
    same(result, query.collect(engine="cpu", streaming=False))
    assert_equal(result.item(0, "s").int64(), Int64(-2))
    assert_equal(result.item(0, "m").float64(), Float64(-0.5))
    for name in ["i", "u"]:
        with assert_raises(contains="overflow"):
            _ = (
                limits.lazy()
                .head(2)
                .select(col(name).sum())
                .collect(accelerator=runtime, engine="accel")
            )
    var narrow = DataFrame([Series("x", Column[Int32]([Int32.MAX, 1]))])
    with assert_raises(contains="overflow"):
        _ = (
            narrow.lazy()
            .select(col("x").sum())
            .collect(accelerator=runtime, engine="accel")
        )
    # A rejected sum leaves no queued work; subsequent reductions still work.
    var profile = query.profile(accelerator=runtime, engine="accel")
    assert_equal(
        profile[1].item(0, "algorithm").string(), "mojo_gpu_reductions"
    )
    var peak = Int(profile[1].item(0, "peak_requested_bytes").int64())
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    _ = query.collect(
        accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
    )


# requires: mojo_gpu
def test_mojo_gpu_float_boolean_extrema_and_filtered_reductions() raises:
    var nan = Float64("nan")
    var inf = Float64("inf")
    var examples: List[List[Float64]] = [
        [-0.0, 0.0, nan, 3],
        [0.0, -0.0, nan, 3],
        [nan, nan, nan, nan],
        [-inf, inf, 0, nan],
        [inf, 1, inf, 2],
        [-inf, -1, -inf, -2],
    ]
    for dtype in [DataType.FLOAT32, DataType.FLOAT64]:
        for values in examples:
            var frame = DataFrame(
                [Series("x", Column[Float64](values.copy())).cast(dtype)]
            )
            var query = frame.lazy().select_exprs(
                [
                    col("x").min().alias("min"),
                    col("x").max().alias("max"),
                    col("x").arg_min().alias("arg_min"),
                    col("x").arg_max().alias("arg_max"),
                    col("x").first().alias("first"),
                    col("x").last().alias("last"),
                    col("x").sum().alias("sum"),
                    col("x").mean().alias("mean"),
                ]
            )
            same(
                query.collect(accelerator=MetalRuntime(), engine="accel"),
                query.collect(engine="cpu", streaming=False),
            )
        var tiny = Series(
            "x",
            Column[Float32](
                List[Float32](
                    length=1_027, fill=bitcast[DType.float32](UInt32(1))
                )
            ),
        ).cast(dtype)
        var small = (
            DataFrame([tiny^])
            .lazy()
            .select_exprs(
                [col("x").sum().alias("s"), col("x").mean().alias("m")]
            )
        )
        same(
            small.collect(accelerator=MetalRuntime(), engine="accel"),
            small.collect(engine="cpu", streaming=False),
        )
    for mode in range(4):
        var valid = List[Bool](length=5, fill=mode != 0)
        if mode == 1 or mode == 2:
            valid[1] = False
            valid[4] = False
        var frame = DataFrame(
            [
                Series(
                    "x",
                    Column[Bool](
                        [mode == 1, False, mode != 2, False, True], valid^
                    ),
                )
            ]
        )
        for input in [frame.copy(), frame.head(0)]:
            var query = input.lazy().select_exprs(
                [
                    col("x").any().alias("any"),
                    col("x").all().alias("all"),
                    col("x").any(ignore_nulls=False).alias("any_nulls"),
                    col("x").all(ignore_nulls=False).alias("all_nulls"),
                    col("x").count().alias("count"),
                    col("x").null_count().alias("nulls"),
                    col("x").first().alias("first"),
                    col("x").last().alias("last"),
                ]
            )
            same(
                query.collect(accelerator=MetalRuntime(), engine="accel"),
                query.collect(engine="cpu", streaming=False),
            )
    var source = DataFrame(
        [
            Series("x", Column[Int64]([Int64.MAX, 1, 2, 3])),
            Series("keep", Column[Bool]([False, True, True, True])),
        ]
    )
    var filtered = (
        source.lazy()
        .filter(col("keep"))
        .slice(-2, 2)
        .select_exprs(
            [
                col("x").sum().alias("s"),
                col("x").mean().alias("m"),
                col("x").arg_max().alias("arg"),
            ]
        )
    )
    var result = filtered.collect(accelerator=MetalRuntime(), engine="accel")
    same(result, filtered.collect(engine="cpu", streaming=False))
    assert_equal(result.item(0, "s").int64(), Int64(5))
    assert_equal(result.item(0, "m").float64(), Float64(2.5))
    assert_equal(result.item(0, "arg").numeric[DType.uint32](), UInt32(1))
    var empty = (
        source.lazy()
        .filter(~col("keep") & col("keep"))
        .select_exprs(
            [
                col("x").sum().alias("s"),
                col("x").mean().alias("m"),
                col("x").len().alias("n"),
            ]
        )
    )
    var result_empty = empty.collect(accelerator=MetalRuntime(), engine="accel")
    assert_equal(result_empty.height(), 1)
    assert_equal(result_empty.item(0, "s").int64(), Int64(0))
    assert_true(result_empty.item(0, "m").is_null())
    assert_equal(result_empty.item(0, "n").int64(), Int64(0))
