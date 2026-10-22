"""Physical chunk fragments upload with independent offsets and owned metadata."""
# contract-group: execution.backends
from dataframe_metal.accelerator.gpu.program import Program
from dataframe_metal.accelerator.gpu.runtime import (
    capabilities as _capabilities,
    estimate_resources,
)
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    DataType,
    Series,
    Column,
    StringColumn,
    col,
    lit,
)
from dataframe_metal.metal import MetalRuntime
from dataframe_metal.accelerator.capabilities import RowCapabilities
from dataframe_metal.accelerator.rows import lower_rows
from tests.execution_modes.test_backend_metal import require_metal
from dataframe.lazy import LazyFrame
from tests.execution_modes.test_backend_metal_moments import near
from tests.support.lifecycle_jobs import WorkerSetting


def check(
    query: LazyFrame,
    runtime: MetalRuntime,
    order_by: List[String] = List[String](),
) raises:
    var actual = query.collect(accelerator=runtime)
    var expected = query.collect(engine="cpu", streaming=False)
    if len(order_by):
        actual = actual.sort(order_by)
        expected = expected.sort(order_by)
    if not actual.equals(expected):
        print(query.explain(optimize=False, streaming=False))
        print("Chunked Metal actual:", actual)
        print("Chunked CPU expected:", expected)
    assert_true(actual.equals(expected))


def fragmented(column: Series, first: Int = 3) raises -> Series:
    # Each part keeps its own non-byte-aligned bitmap offset.
    return Series._from_chunks(
        [
            column.slice(1, first),
            column.slice(first + 1, 2),
            column.slice(first + 3, len(column) - first - 3),
        ]
    )


def chunk_frame() raises -> DataFrame:
    return DataFrame(
        [
            fragmented(
                Series("k", Column[Int32]([9, 2, 1, 2, 1, 3, 3, 2, 1, 2, 3]))
            ),
            fragmented(
                Series(
                    "x",
                    Column[Int64](
                        [99, 8, -3, 5, 0, 9, 4, -2, 7, 6, 1],
                        [
                            True,
                            True,
                            False,
                            True,
                            True,
                            False,
                            True,
                            True,
                            True,
                            True,
                            True,
                        ],
                    ),
                ),
                4,
            ),
        ]
    )


def test_chunked_inputs_capability_owned_descriptors_and_value_blind_estimates() raises:
    assert_true(not RowCapabilities().chunked_inputs)
    var frame = chunk_frame()
    assert_true(frame.column("x").is_chunked())
    var caps = _capabilities()
    caps.chunked_inputs = False
    with assert_raises(contains="chunked input"):
        _ = lower_rows(frame.lazy(), caps)
    var plan = lower_rows(frame.lazy(), _capabilities())
    var desc = Program(plan)
    assert_equal(len(desc.code), 0)
    assert_equal(plan.source.width(), 2)
    var keys = plan.source.column("k").chunks()
    var values = plan.source.column("x").chunks()
    assert_equal(len(keys), 3)
    assert_equal(len(values), 3)
    assert_equal(len(keys[0]), 3)
    assert_equal(len(values[0]), 4)
    assert_equal(keys[0].int32()._offset, 1)
    assert_equal(values[0].int64()._offset, 1)
    var resources = estimate_resources(plan)
    assert_equal(resources.matrix_bytes, plan.source.height() * plan.slots * 21)
    assert_true(
        "unsupported"
        not in MetalRuntime(device_id=999999).describe(frame.lazy())
    )
    assert_true(frame.column("x").is_chunked())


# requires: metal
def test_native_metal_chunked_inputs_physical_types_offsets_nulls_and_slices() raises:
    var runtime = require_metal()
    var types: List[DataType] = [
        DataType.INT8,
        DataType.UINT8,
        DataType.INT16,
        DataType.UINT16,
        DataType.INT32,
        DataType.UINT32,
        DataType.INT64,
        DataType.UINT64,
        DataType.FLOAT32,
        DataType.FLOAT64,
        DataType.BOOL,
        DataType.decimal(9, 2, 32),
        DataType.decimal(18, 2, 64),
        DataType.decimal(38, 2, 128),
    ]
    var original = Series(
        "x",
        Column[Int64](
            [99, 1, 0, 3, 4, 0, 6, 7, 8, 9, 10],
            [
                True,
                True,
                False,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
            ],
        ),
    )
    for dtype in types:
        var series = fragmented(original.cast(dtype))
        assert_true(series.is_chunked())
        var frame = DataFrame([series^])
        for offset in range(0, 9):
            check(frame.slice(offset, 5).lazy(), runtime)
        check(frame.head(0).lazy(), runtime)
        check(frame.lazy().select(col("x").is_null()), runtime)
    for dtype in List[DataType](
        [
            DataType.DATE,
            DataType.datetime("ms"),
            DataType.datetime("us"),
            DataType.datetime("ns"),
            DataType.TIME,
            DataType.duration("ns"),
        ]
    ):
        check(
            DataFrame([fragmented(original.with_dtype(dtype))]).lazy(), runtime
        )
    var strings = Series(
        "s",
        StringColumn(
            List[String](
                ["skip", "β", "", "x", "β", "🙂", "z", "x", "", "a", "b"]
            )
        ),
    )
    check(DataFrame([fragmented(strings)]).lazy(), runtime)


# requires: metal
def test_native_metal_chunked_inputs_row_consumers_groups_and_resident_joins() raises:
    var runtime = require_metal()
    var frame = chunk_frame()
    check(frame.lazy().filter(col("x") > 0).sort("x"), runtime)
    check(
        frame.lazy()
        .with_columns((col("x").cast(DataType.FLOAT64) + 2).alias("z"))
        .slice(2, 6),
        runtime,
    )
    check(frame.lazy().select(col("x").shift(2)), runtime)
    check(frame.lazy().select(col("x").median()), runtime)
    check(frame.lazy().select_exprs([col("x").sum(), col("k").mean()]), runtime)
    check(frame.lazy().group_by("k").agg(col("x").sum()), runtime, ["k"])
    check(
        frame.lazy().with_columns(col("x").sum().over("k").alias("s")), runtime
    )
    var other = DataFrame(
        [
            Series._from_chunks(
                [
                    Series("k", Column[Int32]([1, 2])),
                    Series("k", Column[Int32]([3, 4])),
                ]
            ),
            Series._from_chunks(
                [
                    Series("y", Column[Int64]([10])),
                    Series("y", Column[Int64]([20, 30, 40])),
                ]
            ),
        ]
    )
    # Align physical key dtypes before resident joins.
    other = other.with_column(other.column("k").cast(DataType.INT32))
    check(
        frame.lazy().join(other.lazy(), on="k").sort(["k", "x", "y"]), runtime
    )
    var common = frame.lazy().filter(col("x") > 0)
    check(common.join(common, on="k").sort(["k", "x", "x_right"]), runtime)


# requires: metal
def test_native_metal_chunked_inputs_profiles_exact_budget_and_float_reductions() raises:
    var runtime = require_metal()
    var query = chunk_frame().lazy().select((col("x") + 1).alias("x"))
    var estimate = runtime.estimate_auto(query)
    assert_true(";chunked-inputs-v1:3" in estimate.family)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    var exact = MetalRuntime(memory_limit_bytes=estimate.peak_bytes)
    var measured = query.profile(accelerator=exact)
    assert_equal(
        measured[1].item(0, "peak_requested_bytes").int64(),
        Int64(estimate.peak_bytes),
    )
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=estimate.peak_bytes - 1)
        )
    check(query, exact)
    for workers in List[Int]([1, 2, 4]):
        var setting = WorkerSetting(workers)
        var values = Series._from_chunks(
            [
                Series("x", Column[Float64]([1e16, 1.0])),
                Series("x", Column[Float64]([-1e16, 1.0])),
            ]
        )
        var frame = DataFrame([values^])
        # FloatSumState.merge permits reassociation: CPU eager and streaming
        # also differ on this cancellation case. Check native accuracy against
        # the exact mathematical answer, not another association's bit pattern.
        var total = (
            frame.lazy().select(col("x").sum()).collect(accelerator=runtime)
        )
        var mean = (
            frame.lazy().select(col("x").mean()).collect(accelerator=runtime)
        )
        assert_equal(total.column("x").float64()._get(0), Float64(2))
        assert_equal(mean.column("x").float64()._get(0), Float64(0.5))
        for expression in [
            col("x").var(),
            col("x").std(),
            col("x").skew(),
            col("x").median(),
        ]:
            near(
                frame.lazy().select(expression).collect(accelerator=runtime),
                frame.lazy()
                .select(expression)
                .collect(engine="cpu", streaming=False),
            )
        _ = setting^
