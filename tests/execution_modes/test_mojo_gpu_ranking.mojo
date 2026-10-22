"""Public rank contracts for the direct Mojo GPU backend."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col
from dataframe_metal.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.execution_modes.test_mojo_gpu_ordering import _layout


def _integer_case[D: DType]() raises:
    var values = List[Scalar[D]]()
    var validity = List[Bool]()
    var groups = List[Int32]()
    for physical in range(263):
        var value = Scalar[D](physical % 17)
        if physical % 11 == 0:
            value = Scalar[D].MAX
        elif physical % 11 == 1:
            value = Scalar[D].MIN
        values.append(value)
        validity.append(physical % 13 != 4)
        groups.append(Int32(physical % 3))
    var frame = DataFrame(
        [
            _layout(
                Series("x", Column[Scalar[D]](values^, validity^)).slice(
                    3, 257
                ),
                True,
            ),
            _layout(Series("k", Column[Int32](groups^)).slice(3, 257), True),
        ]
    )
    for method in ["average", "min", "max", "dense", "ordinal"]:
        for descending in [False, True]:
            var query = frame.lazy().select(
                col("x").rank(method, descending).over("k").alias("rank")
            )
            var actual = query.collect(
                accelerator=MetalRuntime(), engine="accel"
            )
            same(actual, query.collect(engine="cpu"))
            assert_equal(
                actual.column("rank").dtype(),
                DataType.FLOAT64 if method == "average" else DataType.INT64,
            )


# requires: mojo_gpu
def test_mojo_gpu_rank_integer_widths_chunks_methods_and_partitions() raises:
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
        _integer_case[types[i]]()
    var values = List[Int64]()
    var groups = List[Int32]()
    var valid = List[Bool]()
    for row in range(4_099):
        values.append(Int64((row * 31) % 521))
        groups.append(Int32(row % 3))
        valid.append(row % 19 != 0)
    var frame = DataFrame(
        [
            Series("x", Column[Int64](values^, valid^)),
            Series("k", Column[Int32](groups^)),
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            col("x").rank("dense").over("k").alias("dense"),
            col("x").rank("ordinal", descending=True).alias("ordinal"),
        ]
    )
    same(
        query.collect(accelerator=MetalRuntime(), engine="accel"),
        query.collect(engine="cpu"),
    )


# requires: mojo_gpu
def test_mojo_gpu_rank_floating_nan_direction_zero_ties_nulls_and_empty() raises:
    var source = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [Float64("nan"), 0, -0.0, 2, Float64("nan"), -1, 0],
                    [True, True, True, True, True, True, False],
                ),
            )
        ]
    )
    var ascending: List[List[Float64]] = [
        [5.5, 2.5, 2.5, 4, 5.5, 1],
        [5, 2, 2, 4, 5, 1],
        [6, 3, 3, 4, 6, 1],
        [4, 2, 2, 3, 4, 1],
        [5, 2, 3, 4, 6, 1],
    ]
    var descending: List[List[Float64]] = [
        [1.5, 4.5, 4.5, 3, 1.5, 6],
        [1, 4, 4, 3, 1, 6],
        [2, 5, 5, 3, 2, 6],
        [1, 3, 3, 2, 1, 4],
        [1, 4, 5, 3, 2, 6],
    ]
    var methods: List[String] = ["average", "min", "max", "dense", "ordinal"]
    for dtype in [DataType.FLOAT32, DataType.FLOAT64]:
        var frame = DataFrame([source.column("x").cast(dtype)])
        for method in range(len(methods)):
            for reverse in [False, True]:
                var query = frame.lazy().select(
                    col("x").rank(methods[method], reverse).alias("rank")
                )
                var actual = query.collect(
                    accelerator=MetalRuntime(), engine="accel"
                )
                same(actual, query.collect(engine="cpu"))
                var values = (
                    actual.column("rank").cast(DataType.FLOAT64).float64()
                )
                for row in range(6):
                    assert_equal(
                        values._get(row),
                        descending[method][row] if reverse else ascending[
                            method
                        ][row],
                    )
                assert_true(actual.item(6, "rank").is_null())
    for frame in [source.head(0), source.select_exprs([col("x") * 0]).head(0)]:
        var query = frame.lazy().select(col("x").rank("dense"))
        same(
            query.collect(accelerator=MetalRuntime(), engine="accel"),
            query.collect(engine="cpu"),
        )
    var all_null = DataFrame([Series.full_null("x", DataType.FLOAT64, 257)])
    var null_ranks = all_null.lazy().select(col("x").rank("average"))
    assert_equal(
        null_ranks.collect(accelerator=MetalRuntime(), engine="accel")
        .column("x")
        .null_count(),
        257,
    )
    var tiny = bitcast[DType.float64](UInt64(1))
    var partitions = DataFrame(
        [
            Series("x", Column[Float64]([tiny, -tiny, 0, -0.0, tiny, -tiny])),
            Series(
                "k",
                Column[Float64](
                    [0, -0.0, 0, -0.0, Float64("nan"), Float64("nan")]
                ),
            ),
        ]
    )
    for method in methods:
        var query = partitions.lazy().select(col("x").rank(method).over("k"))
        same(
            query.collect(accelerator=MetalRuntime(), engine="accel"),
            query.collect(engine="cpu"),
        )


# requires: mojo_gpu
def test_mojo_gpu_rank_compositions_key_errors_budget_and_reuse() raises:
    var frame = DataFrame(
        [
            Series("k", Column[Int32]([0, 1, 0, 1, 0, 1])),
            Series("x", Column[Float64]([3, 8, 1, 8, 5, 2])),
        ]
    )
    var rank = col("x").rank("ordinal", descending=True).over("k")
    var query = frame.lazy().filter(col("x") > 1).filter(rank <= 2).head(3)
    var runtime = MetalRuntime()
    same(
        query.collect(accelerator=runtime, engine="accel"),
        query.collect(engine="cpu"),
    )
    var nested = (
        frame.lazy()
        .with_columns(col("x").rank().rank("dense").alias("rank"))
        .slice(1, 3)
        .sort("rank")
    )
    same(
        nested.collect(accelerator=runtime, engine="accel"),
        nested.collect(engine="cpu"),
    )
    var overflow = DataFrame([Series("x", Column[Int64]([Int64.MAX, 1]))])
    var bad = (
        overflow.lazy()
        .with_columns((col("x") + 1).alias("key"))
        .select(col("key").rank("dense"))
    )
    with assert_raises():
        _ = bad.collect(accelerator=runtime, engine="accel")
    var profile = nested.profile(accelerator=runtime, engine="accel")
    var peak = Int(profile[1].item(0, "peak_requested_bytes").int64())
    with assert_raises(contains="budget"):
        _ = nested.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    same(
        nested.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ),
        profile[0],
    )
