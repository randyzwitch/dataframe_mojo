"""Public MetalRuntime temporal arithmetic and physical-cast contracts."""
# contract-group: execution.apple_parity
# contract-group: execution.backends

from std.testing import assert_equal, assert_true, assert_raises
from std.memory import bitcast
from dataframe import (
    Column,
    BoolColumn,
    DataFrame,
    DataType,
    Series,
    col,
    lit,
    when,
)
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from dataframe.expr import ADD
from dataframe.accelerator.gpu.runtime import _preflight, capabilities
from dataframe.accelerator.gpu.dag_runtime import preflight as dag_preflight
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.temporal_descriptors import ROW_TEMPORAL_BINARY


def tagged(name: String, dtype: DataType, values: List[Int64]) raises -> Series:
    return Series(name, Column[Int64](values.copy())).with_dtype(dtype)


def layout(frame: DataFrame, index: Int) raises -> DataFrame:
    if index == 0:
        return frame.copy()
    if index == 1:
        return frame.head(1).vstack(frame).rechunk().slice(1, frame.height())
    var fields = List[Series]()
    for field in frame._columns:
        fields.append(
            Series._from_chunks(
                [field.slice(0, 1), field.slice(1, len(field) - 1)]
            )
        )
    return DataFrame(fields^, height=frame.height())


def check(
    frame: DataFrame,
    name: String,
    dtype: DataType,
    values: List[Int64],
    valid: List[Bool] = List[Bool](),
) raises:
    assert_equal(frame.height(), len(values))
    assert_true(frame.column(name).dtype() == dtype)
    for row in range(len(values)):
        var present = not len(valid) or valid[row]
        assert_equal(frame.item(row, name).is_null(), not present)
        if present:
            var value = (
                frame.item(row, name).int64() if dtype
                == DataType.INT64 else frame.item(row, name).to_physical()
            )
            assert_equal(value, values[row])


# requires: metal
def test_mojo_gpu_temporal_binary_arithmetic_extremes_layouts_descriptors_and_budget() raises:
    var d = DataType.duration("ns")
    var original = DataFrame(
        [
            tagged("t", d, [Int64.MIN, Int64.MAX, -5, 5]),
            tagged("delta", d, [0, 0, 2, -2]),
            Series("n", Column[Int64]([1, 1, 2, -2])),
        ]
    )
    for variant in range(3):
        var frame = layout(original, variant)
        var query = frame.lazy().select_exprs(
            [
                (col("t") + col("delta")).alias("plus"),
                (col("t") - col("delta")).alias("minus"),
                (col("t") * col("n")).alias("mul"),
                (col("n") * col("t")).alias("reverse"),
                (col("t") // col("n")).alias("floor"),
            ]
        )
        var actual = query.collect(accelerator=MetalRuntime(), engine="accel")
        check(actual, "plus", d, [Int64.MIN, Int64.MAX, -3, 3])
        check(actual, "minus", d, [Int64.MIN, Int64.MAX, -7, 7])
        check(actual, "mul", d, [Int64.MIN, Int64.MAX, -10, -10])
        check(actual, "reverse", d, [Int64.MIN, Int64.MAX, -10, -10])
        check(actual, "floor", d, [Int64.MIN, Int64.MAX, -3, -3])
        assert_true(actual.equals(query.collect(engine="cpu", streaming=False)))
        var plan = _preflight(query)
        var program = Program(plan)
        var found = False
        for node in range(len(program.literals)):
            if program.code[
                4 * node
            ] == ROW_TEMPORAL_BINARY and program.literals[node] >> 12 == UInt64(
                ADD
            ):
                found = True
                assert_equal(
                    program.literals[node],
                    UInt64(3 | (3 << 4) | (3 << 8)) | (UInt64(ADD) << 12),
                )
                assert_equal(
                    program.literals[node],
                    bitcast[DType.uint64](plan.literals[node]),
                )
        assert_true(found)
        if variant == 0:
            var measured = query.profile(
                accelerator=MetalRuntime(), engine="accel"
            )
            var peak = Int(measured[1].item(0, "peak_requested_bytes").int64())
            assert_true(peak > 0)
            assert_true(
                query.collect(
                    accelerator=MetalRuntime(memory_limit_bytes=peak),
                    engine="accel",
                ).equals(actual)
            )
            with assert_raises(contains="budget"):
                _ = query.collect(
                    accelerator=MetalRuntime(memory_limit_bytes=peak - 1),
                    engine="accel",
                )
    print("PASS public temporal arithmetic values/layouts/descriptors/budget")


# requires: metal
def test_mojo_gpu_temporal_physical_casts_conditional_masks_and_overflow_contract() raises:
    var target = DataType.duration("ns")
    var original = DataFrame(
        [
            tagged("t", DataType.duration("ms"), [Int64.MAX, 2, Int64.MIN, -3]),
            Series("keep", BoolColumn([False, True, False, True])),
        ]
    )
    for variant in range(3):
        var frame = layout(original, variant)
        var masked = frame.lazy().select(
            when(col("keep"))
            .then(col("t").cast(target))
            .otherwise(lit(Int64(0)).cast(target))
            .alias("result")
        )
        var actual = masked.collect(accelerator=MetalRuntime(), engine="accel")
        check(actual, "result", target, [0, 2000000, 0, -3000000])
        assert_true(
            actual.equals(masked.collect(engine="cpu", streaming=False))
        )
        var relaxed = frame.lazy().select(
            col("t").cast(target, strict=False).alias("result")
        )
        check(
            relaxed.collect(accelerator=MetalRuntime(), engine="accel"),
            "result",
            target,
            [0, 2000000, 0, -3000000],
            [False, True, False, True],
        )
        for gpu in [False, True]:
            var failing = frame.lazy().select(col("t").cast(target))
            with assert_raises(contains="temporal value overflows Int64"):
                if gpu:
                    _ = failing.collect(
                        accelerator=MetalRuntime(), engine="accel"
                    )
                else:
                    _ = failing.collect(engine="cpu", streaming=False)
    var fine = DataFrame(
        [
            tagged(
                "t",
                DataType.duration("us"),
                [-1001, 1001, Int64.MIN, Int64.MAX],
            )
        ]
    )
    check(
        fine.lazy()
        .select(col("t").cast(DataType.duration("ms")).alias("result"))
        .collect(accelerator=MetalRuntime(), engine="accel"),
        "result",
        DataType.duration("ms"),
        [-2, 1, -9223372036854776, 9223372036854775],
    )
    var instants = DataFrame(
        [
            tagged(
                "t",
                DataType.datetime("ns", "UTC"),
                [Int64.MIN, Int64.MAX, -1, 0],
            )
        ]
    )
    var zoned = DataType.datetime("ns", "America/New_York")
    check(
        instants.lazy()
        .select(col("t").cast(zoned).alias("result"))
        .collect(accelerator=MetalRuntime(), engine="accel"),
        "result",
        zoned,
        [Int64.MIN, Int64.MAX, -1, 0],
    )
    var arithmetic = DataFrame(
        [
            tagged("t", target, [Int64.MAX, 3, Int64.MIN, 7]),
            tagged("delta", target, [1, 2, -1, -2]),
            Series("keep", BoolColumn([False, True, False, True])),
        ]
    )
    check(
        arithmetic.lazy()
        .select(
            when(col("keep"))
            .then(col("t") + col("delta"))
            .otherwise(lit(Int64(0)).cast(target))
            .alias("result")
        )
        .collect(accelerator=MetalRuntime(), engine="accel"),
        "result",
        target,
        [0, 5, 0, 5],
    )
    var dates = DataFrame(
        [
            tagged("date", DataType.DATE, [-1, 1, 0, 2]),
            tagged("d", DataType.duration("ms"), [500, -500, -1, 1]),
        ]
    )
    check(
        dates.lazy()
        .select((col("date") + col("d")).alias("result"))
        .collect(accelerator=MetalRuntime(), engine="accel"),
        "result",
        DataType.datetime("ms"),
        [-86399500, 86399500, -1, 172800001],
    )
    print("PASS public temporal casts/conditional masks/error anchors")


# requires: metal
def test_mojo_gpu_temporal_arithmetic_preflight_rejects_checked_producers_and_resident_sources() raises:
    var d = DataType.duration("ns")
    var frame = DataFrame(
        [
            tagged("t", d, [1, 2]),
            tagged("delta", d, [3, 4]),
            Series("n", Column[Int64]([0, 1])),
        ]
    )
    with assert_raises(contains="binary operands must be infallible"):
        _ = _preflight(
            frame.lazy().select((col("t") + col("delta")) + col("delta"))
        )
    with assert_raises(contains="mixed checked"):
        _ = _preflight(
            frame.lazy().select(
                when((col("n") + lit(Int64(1))) > lit(Int64(0)))
                .then(col("t") + col("delta"))
                .otherwise(col("t"))
            )
        )
    with assert_raises(contains="direct scan"):
        _ = _preflight(
            frame.lazy()
            .with_columns((col("t") + col("delta")).alias("alias"))
            .select(col("alias") + col("delta"))
        )
    with assert_raises(contains="direct scan"):
        _ = _preflight(frame.lazy().select(col("t") + col("delta")).head(1))
    with assert_raises(contains="direct scan"):
        _ = _preflight(
            frame.lazy().select(col("t") + col("delta")), source_rows=2
        )
    with assert_raises(contains="before DAG submission"):
        _ = dag_preflight(
            frame.lazy()
            .join(frame.lazy(), on="n", how="inner")
            .select(col("t") + col("delta")),
            capabilities(),
        )
    var decimal = DataFrame(
        [
            Series("d", Column[Int64]([123, 456])).with_dtype(
                DataType.decimal(18, 2, 64)
            )
        ]
    )
    with assert_raises(contains="only temporal, Int64 and Bool"):
        _ = _preflight(
            decimal.lazy().select(col("d").cast(DataType.INT64).cast(d))
        )
    var dates = DataFrame([tagged("date", DataType.DATE, [0, 1])])
    with assert_raises(contains="cannot cast"):
        _ = _preflight(
            dates.lazy().select(
                when(lit(False))
                .then(col("date").cast(DataType.TIME, strict=False))
                .otherwise(lit(Int64(0)).cast(DataType.TIME))
            )
        )
    print("PASS public preflight source/provenance/fault-category restrictions")
