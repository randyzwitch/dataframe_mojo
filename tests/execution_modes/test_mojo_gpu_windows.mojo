"""Cumulative, shift and fill contracts through direct Mojo GPU queries."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col, lit, when
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.execution_modes.test_mojo_gpu_ordering import _layout


def _integer_case[D: DType](n: Int, chunked: Bool) raises:
    var values = List[Scalar[D]]()
    var valid = List[Bool]()
    var groups = List[Int32]()
    var group_valid = List[Bool]()
    var flags = List[Bool]()
    for physical in range(n + 6):
        values.append(Scalar[D](physical % 5 + 1))
        valid.append(physical % 7 != 0 and physical % 7 != 1)
        groups.append(Int32(physical % 3))
        group_valid.append(physical % 11 != 4)
        flags.append(physical % 2 == 0)
    var frame = DataFrame(
        [
            _layout(
                Series("x", Column[Scalar[D]](values^, valid^)).slice(3, n),
                chunked,
            ),
            _layout(
                Series("k", Column[Int32](groups^, group_valid^)).slice(3, n),
                chunked,
            ),
            _layout(Series("b", Column[Bool](flags^)).slice(3, n), chunked),
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            col("x").cum_sum().over("k").alias("sum"),
            col("x").cum_sum(True).over("k").alias("sum_reverse"),
            col("x").cum_min().over(["k", "b"]).alias("min"),
            col("x").cum_min(True).over(["k", "b"]).alias("min_reverse"),
            col("x").cum_max().over("k").alias("max"),
            col("x").cum_max(True).over("k").alias("max_reverse"),
            col("x").cum_count().alias("count"),
            col("x").cum_count(True).alias("count_reverse"),
            col("x").shift(-3).over("k").alias("shift_negative"),
            col("x").shift(0).over("k").alias("shift_zero"),
            col("x").shift(2).over("k").alias("shift_positive"),
            col("x").shift(Int(Int64.MIN)).over("k").alias("shift_min"),
            col("x").shift(Int(Int64.MAX)).over("k").alias("shift_max"),
            col("x").forward_fill().over("k").alias("forward"),
            col("x").forward_fill(0).over("k").alias("forward_zero"),
            col("x").forward_fill(2).over("k").alias("forward_two"),
            col("x").backward_fill().over("k").alias("backward"),
            col("x").backward_fill(0).over("k").alias("backward_zero"),
            col("x").backward_fill(2).over("k").alias("backward_two"),
        ]
    )
    var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
    same(actual, query.collect(engine="cpu"))
    assert_equal(actual.column("min").dtype(), DataType.of(D))
    assert_equal(actual.column("sum").dtype(), DataType.of(D).sum_type())
    assert_equal(actual.column("shift_min").null_count(), n)
    assert_equal(actual.column("shift_max").null_count(), n)


def test_mojo_gpu_windows_integer_layouts_partitions_directions_shifts_and_fills() raises:
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
        _integer_case[types[i]](257, False)
        _integer_case[types[i]](257, True)
    _integer_case[DType.int64](0, True)
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [0, 2, 0, 0, 3], [False, True, False, False, True]
                ),
            ),
            Series("k", Column[Int32]([0, 0, 0, 0, 0])),
        ]
    )
    var result = (
        frame.lazy()
        .select_exprs(
            [
                col("x").cum_count().alias("count"),
                col("x").forward_fill(1).alias("forward"),
                col("x").backward_fill(1).alias("backward"),
            ]
        )
        .collect(accelerator=MetalRuntime(), engine="accel")
    )
    var counts: List[Int64] = [0, 1, 1, 1, 2]
    for row in range(5):
        assert_equal(result.item(row, "count").int64(), counts[row])
    assert_true(result.item(0, "forward").is_null())
    assert_equal(result.item(2, "forward").int64(), Int64(2))
    assert_true(result.item(3, "forward").is_null())
    assert_equal(result.item(0, "backward").int64(), Int64(2))
    assert_true(result.item(2, "backward").is_null())
    assert_equal(result.item(3, "backward").int64(), Int64(3))


def test_mojo_gpu_windows_float_encodings_logical_metadata_and_empty_nulls() raises:
    var tiny = bitcast[DType.float64](UInt64(1))
    var source = Series(
        "x",
        Column[Float64](
            [
                0,
                -0.0,
                tiny,
                -tiny,
                Float64("nan"),
                0,
                Float64("inf"),
                -Float64("inf"),
            ],
            [True, True, True, True, True, False, True, True],
        ),
    )
    for dtype in [DataType.FLOAT32, DataType.FLOAT64]:
        var input = source.cast(dtype)
        if dtype == DataType.FLOAT32:
            input = Series(
                "x",
                Column[Float32](
                    [
                        0,
                        -0.0,
                        bitcast[DType.float32](UInt32(1)),
                        bitcast[DType.float32](UInt32(0x80000001)),
                        bitcast[DType.float32](UInt32(0xFFC00042)),
                        0,
                        Float32(Float64("inf")),
                        -Float32(Float64("inf")),
                    ],
                    [True, True, True, True, True, False, True, True],
                ),
            )
        for chunked in [False, True]:
            var frame = DataFrame([_layout(input, chunked)])
            var query = frame.lazy().select_exprs(
                [
                    col("x").cum_sum().alias("sum"),
                    col("x").cum_sum(True).alias("reverse"),
                    col("x").cum_min().alias("min"),
                    col("x").cum_max().alias("max"),
                    col("x").shift().alias("shift"),
                    col("x").forward_fill().alias("forward"),
                    col("x").backward_fill().alias("backward"),
                ]
            )
            same(
                query.collect(accelerator=MetalRuntime(), engine="accel"),
                query.collect(engine="cpu"),
            )
    for dtype in [
        DataType.DATE,
        DataType.datetime("us", "UTC"),
        DataType.duration("ns"),
    ]:
        var frame = DataFrame(
            [
                Series(
                    "x", Column[Int64]([3, -1, 0, 2], [True, True, False, True])
                ).with_dtype(dtype)
            ]
        )
        var query = frame.lazy().select_exprs(
            [
                col("x").cum_min().alias("min"),
                col("x").cum_max(True).alias("max"),
                col("x").shift().alias("shift"),
                col("x").forward_fill().alias("forward"),
            ]
        )
        var result = query.collect(accelerator=MetalRuntime(), engine="accel")
        same(result, query.collect(engine="cpu"))
        for name in result.columns():
            assert_equal(result.column(name).dtype(), dtype)
    var all_null = DataFrame([Series.full_null("x", DataType.INT64, 257)])
    for frame in [all_null.copy(), all_null.head(0)]:
        var query = frame.lazy().select_exprs(
            [
                col("x").cum_sum().alias("sum"),
                col("x").cum_count().alias("count"),
                col("x").forward_fill().alias("forward"),
                col("x").shift(-1).alias("shift"),
            ]
        )
        same(
            query.collect(accelerator=MetalRuntime(), engine="accel"),
            query.collect(engine="cpu"),
        )


def test_mojo_gpu_windows_compositions_checked_prefix_errors_budget_and_recovery() raises:
    var runtime = MetalRuntime()
    var overflow = DataFrame([Series("x", Column[Int64]([Int64.MAX, 1, -1]))])
    var prefix = overflow.lazy().select(col("x").cum_sum().alias("sum"))
    for query in [
        prefix.copy(),
        prefix.head(1),
        prefix.filter(col("sum") < 0),
        prefix.head(0),
    ]:
        with assert_raises(contains="overflow"):
            _ = query.collect(accelerator=runtime, engine="accel")
    var hidden = overflow.lazy().select(
        when(lit(False)).then(col("x").cum_sum()).otherwise(lit(Int64(0)))
    )
    with assert_raises(contains="overflow"):
        _ = hidden.collect(accelerator=runtime, engine="accel")
    for frame in [
        DataFrame([Series("x", Column[Int32]([Int32.MAX, 1]))]),
        DataFrame([Series("x", Column[UInt32]([UInt32.MAX, 1]))]),
        DataFrame([Series("x", Column[UInt64]([UInt64(1) << 63]))]),
    ]:
        with assert_raises(contains="overflow"):
            _ = (
                frame.lazy()
                .select(col("x").cum_sum())
                .collect(accelerator=runtime, engine="accel")
            )
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [4, 1, 3, 2, 5, 0], [True, True, False, True, True, False]
                ),
            ),
            Series("k", Column[Int32]([0, 1, 0, 1, 0, 1])),
        ]
    )
    var query = (
        frame.lazy()
        .filter(col("x").is_not_null())
        .with_columns(col("x").shift().over("k").forward_fill().alias("filled"))
        .with_columns(
            col("x").rank("dense").over("k").cum_sum().alias("rank_sum")
        )
        .sort("rank_sum")
        .slice(1, 2)
        .select(["x", "filled", "rank_sum"])
    )
    var profile = query.profile(accelerator=runtime, engine="accel")
    same(profile[0], query.collect(engine="cpu"))
    var peak = Int(profile[1].item(0, "peak_requested_bytes").int64())
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
            engine="accel",
        )
    same(
        query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=peak), engine="accel"
        ),
        profile[0],
    )
