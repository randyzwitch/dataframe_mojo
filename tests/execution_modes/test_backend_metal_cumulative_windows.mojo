"""Ordered cumulative state and nested OVER scopes through resident Metal."""
# contract-group: execution.backends
from tests.execution_modes.test_backend_metal import require_metal
from std.testing import assert_equal, assert_true, assert_raises
from std.memory import bitcast
from dataframe import (
    DataFrame,
    Series,
    Column,
    BoolColumn,
    StringColumn,
    Expr,
    col,
    lit,
    when,
)
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.metal import MetalRuntime
from dataframe.accelerator.windows import lower_partition_aggregates
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.capabilities import RowCapabilities
from tests.execution_modes.test_backend_metal_float64_emulation import same


def check(frame: DataFrame, expression: Expr) raises:
    var query = frame.lazy().select_exprs([col("k"), expression.alias("value")])
    same(query.collect(accelerator=MetalRuntime()), query.collect(engine="cpu"))


def cumulative(expression: Expr, method: Int, reverse: Bool) -> Expr:
    if method == 0:
        return expression.cum_sum(reverse)
    if method == 1:
        return expression.cum_min(reverse)
    if method == 2:
        return expression.cum_max(reverse)
    return expression.cum_count(reverse)


def test_nested_window_lowering_preserves_cpu_scopes_without_device() raises:
    var frame = DataFrame(
        [
            Series("k", Column[Int32]([0, 1, 0, 1])),
            Series("j", Column[Int32]([0, 0, 1, 1])),
            Series("x", Column[Int64]([3, 8, 2, 5], [True, False, True, True])),
        ]
    )
    var expressions = List[Expr](
        [
            col("x").cum_sum().over("k").cum_max(True).over("j"),
            col("x").fill_null(col("x").min().over("j")).cum_max().over("k"),
            col("x").rank("dense").over("k").cum_sum().over("j"),
            col("x").rank("min").over("k").over("j"),
        ]
    )
    for expression in expressions:
        var query = frame.lazy().select(expression.alias("w"))
        same(
            lower_partition_aggregates(query).collect(engine="cpu"),
            query.collect(engine="cpu"),
        )
    with assert_raises(contains="cumulative windows require backend support"):
        _ = lower_rows(
            frame.lazy().select(col("x").cum_sum()), RowCapabilities()
        )


# requires: metal
def test_native_metal_cumulative_numeric_widths_null_partitions_and_reverse() raises:
    _ = require_metal()
    comptime for D in NUMERIC_DTYPES:
        var values = List[Scalar[D]]()
        var keys = List[Int32]()
        var valid = List[Bool]()
        var key_valid = List[Bool]()
        for i in range(519):
            values.append(Scalar[D](i % 9))
            keys.append(Int32(i % 7))
            valid.append(i % 5 != 0)
            key_valid.append(i % 11 != 0)
        var frame = DataFrame(
            [
                Series("x", Column[Scalar[D]](values^, valid^)),
                Series("k", Column[Int32](keys^, key_valid^)),
            ]
        ).slice(3, 513)
        for method in range(4):
            for reverse in List[Bool]([False, True]):
                check(frame, cumulative(col("x"), method, reverse))
                check(frame, cumulative(col("x"), method, reverse).over("k"))


# requires: metal
def test_native_metal_cumulative_floating_association_nan_zeros_and_subnormals() raises:
    _ = require_metal()
    var words: List[UInt64] = [
        0x4340000000000000,
        0x3FF0000000000000,
        0xC340000000000000,
        0x8000000000000000,
        0,
        1,
        0x8000000000000001,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000042,
    ]
    var narrow: List[UInt32] = [
        0x4B800000,
        0x3F800000,
        0xCB800000,
        0x80000000,
        0,
        1,
        0x80000001,
        0x7F800000,
        0xFF800000,
        0x7FC00042,
    ]
    var a = List[Float64]()
    var b = List[Float32]()
    var k = List[Int32]()
    var valid = List[Bool]()
    for i in range(519):
        a.append(bitcast[DType.float64](words[i % len(words)]))
        b.append(bitcast[DType.float32](narrow[i % len(narrow)]))
        k.append(Int32((i // 10) % 3))
        valid.append(i % 13 != 0)
    var frame = DataFrame(
        [
            Series("x", Column[Float64](a^, valid)),
            Series("y", Column[Float32](b^, valid^)),
            Series("k", Column[Int32](k^)),
        ]
    )
    for name in List[String](["x", "y"]):
        for method in range(4):
            for reverse in List[Bool]([False, True]):
                check(frame, cumulative(col(name), method, reverse).over("k"))


# requires: metal
def test_native_metal_cumulative_decimal_boolean_text_and_logical_storage() raises:
    _ = require_metal()
    var frame = DataFrame(
        [
            Series("k", Column[Int32]([0, 1, 0, 1, 0, 1])),
            Series(
                "p", Column[Int32]([999999999, 999999999, -999999999, 1, 1, 1])
            ).with_dtype(DataType.decimal(9, 2, 32)),
            Series(
                "d",
                Column[Int64](
                    [10, -20, 30, 40, -50, 60],
                    [True, False, True, True, True, True],
                ),
            ).with_dtype(DataType.decimal(18, 2, 64)),
            Series(
                "w",
                Column[Int128](
                    [Int128(1) << 80, 20, -(Int128(1) << 80), 40, 50, 60]
                ),
            ).with_dtype(DataType.decimal(38, 2)),
            Series("date", Column[Int64]([5, 4, 3, 2, 1, 0])).with_dtype(
                DataType.DATE
            ),
            Series(
                "b",
                BoolColumn(
                    [True, False, True, False, False, True],
                    [True, True, False, True, True, True],
                ),
            ),
            Series(
                "s",
                StringColumn(
                    List[String](["z", "a", "", "é", "a", "z"]),
                    [True, True, False, True, True, True],
                ),
            ),
        ]
    )
    for name in List[String](["p", "d", "w", "date", "b", "s"]):
        for method in range(
            0 if name == "p" or name == "d" or name == "w" else 1, 4
        ):
            for reverse in List[Bool]([False, True]):
                check(frame, cumulative(col(name), method, reverse).over("k"))


# requires: metal
def test_native_metal_cumulative_nested_scopes_resident_consumers_empty_and_budget() raises:
    _ = require_metal()
    var frame = DataFrame(
        [
            Series("k", Column[Int32]([0, 1, 0, 1, 0, 1])),
            Series("j", Column[Int32]([0, 0, 1, 1, 0, 0])),
            Series(
                "x",
                Column[Int64](
                    [8, 2, 5, 7, 1, 4], [False, True, True, False, True, True]
                ),
            ),
        ]
    )
    var value = col("x").fill_null(col("x").min().over("k"))
    check(
        frame,
        when(col("x").cum_count().over("k") > lit(Int64(0)))
        .then(value.cum_max().over("k"))
        .end(),
    )
    check(frame, col("x").cum_sum().over("k").cum_max(True).over("j"))
    check(frame, col("x").rank("dense").over("k").cum_sum().over("j"))
    var query = (
        frame.lazy()
        .filter(col("j") == lit(Int32(0)))
        .with_columns([col("x").cum_sum().over("k").alias("s")])
        .sort("s")
        .select_exprs([col("k"), col("s"), col("s").cum_count().alias("n")])
    )
    same(query.collect(accelerator=MetalRuntime()), query.collect(engine="cpu"))
    var resident = frame.lazy().select(
        col("x")
        .fill_null(col("x").min().over("k"))
        .cum_max()
        .over("j")
        .alias("s")
    )
    var measured = resident.profile(accelerator=MetalRuntime())
    same(measured[0], resident.collect(engine="cpu"))
    assert_true(measured[1].height() > 1)
    for i in range(measured[1].height() - 1):
        assert_true(
            measured[1].column("host_result_bytes").int64()._get(i) == 0
        )
        assert_true(
            measured[1].column("resident_output_bytes").int64()._get(i) > 0
        )
        assert_equal(
            measured[1].item(i, "materialization_boundary").string(),
            "Mojo GPU resident table",
        )
    var estimate = MetalRuntime().estimate_auto(
        frame.lazy().select(col("x").cum_count())
    )
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    assert_true(";cumulative-v1" in estimate.family)
    var automatic = frame.lazy().select(col("x").cum_count())
    same(
        automatic.collect(engine="auto", accelerator=MetalRuntime()),
        automatic.collect(engine="cpu"),
    )
    check(frame.head(0), col("x").cum_sum().over("k"))
    var all_null = DataFrame(
        [
            Series("k", Column[Int32]([0, 0])),
            Series("x", Column[Int64]([1, 2], [False, False])),
        ]
    )
    check(all_null, col("x").cum_count().over("k"))
    check(all_null, col("x").cum_max().over("k"))
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    same(query.collect(accelerator=MetalRuntime()), query.collect(engine="cpu"))


# requires: metal
def test_native_metal_cumulative_checked_prefixes_remain_observable_under_head_and_when() raises:
    _ = require_metal()
    var frame = DataFrame(
        [
            Series("k", Column[Int32]([0, 0, 0])),
            Series("x", Column[Int64]([0x7FFFFFFFFFFFFFFF, 1, -1])),
        ]
    )
    for reverse in List[Bool]([False, True]):
        var query = (
            frame.lazy().select(col("x").cum_sum(reverse).over("k")).head(1)
        )
        if reverse:
            same(
                query.collect(accelerator=MetalRuntime()),
                query.collect(engine="cpu"),
            )
        else:
            with assert_raises(contains="overflow"):
                _ = query.collect(engine="cpu")
            with assert_raises(contains="overflow"):
                _ = query.collect(accelerator=MetalRuntime())
    var hidden = (
        frame.lazy()
        .select(
            when(lit(False))
            .then(col("x").cum_sum().over("k"))
            .otherwise(lit(Int64(0)))
        )
        .head(0)
    )
    with assert_raises(contains="overflow"):
        _ = hidden.collect(engine="cpu")
    with assert_raises(contains="overflow"):
        _ = hidden.collect(accelerator=MetalRuntime())
    var unsigned = DataFrame([Series("x", Column[UInt64]([UInt64(1) << 63]))])
    with assert_raises(contains="overflow"):
        _ = (
            unsigned.lazy()
            .select(col("x").cum_sum())
            .collect(accelerator=MetalRuntime())
        )
    var decimal = DataFrame(
        [
            Series("x", Column[Int128]([99, 1, -1])).with_dtype(
                DataType.decimal(2, 0)
            )
        ]
    )
    # SQL SUM widens every decimal storage width to precision 38.
    # Anchor the values and type independently of CPU/GPU agreement.
    for reverse in [False, True]:
        var query = decimal.lazy().select(col("x").cum_sum(reverse))
        var actual = query.collect(accelerator=MetalRuntime())
        same(actual, query.collect(engine="cpu"))
        assert_true(actual.column("x").dtype() == DataType.decimal(38, 0))
        var expected: List[Int128] = [99, 0, -1] if reverse else [99, 100, 99]
        for row in range(3):
            assert_equal(
                actual.column("x").numeric[DType.int128]().value(row),
                expected[row],
            )

    var narrow = DataFrame([Series("x", Column[Int32]([2147483647, 1, -1]))])
    with assert_raises(contains="overflow"):
        _ = narrow.lazy().select(col("x").cum_sum()).collect(engine="cpu")
    with assert_raises(contains="overflow"):
        _ = (
            narrow.lazy()
            .select(col("x").cum_sum())
            .collect(accelerator=MetalRuntime())
        )
    var u32 = DataFrame([Series("x", Column[UInt32]([0xFFFFFFFF, 1]))])
    with assert_raises(contains="overflow"):
        _ = u32.lazy().select(col("x").cum_sum()).collect(engine="cpu")
    with assert_raises(contains="overflow"):
        _ = (
            u32.lazy()
            .select(col("x").cum_sum())
            .collect(accelerator=MetalRuntime())
        )
    var limit = Int128(1)
    for _ in range(38):
        limit *= 10
    limit -= 1
    var wide = DataFrame(
        [
            Series("x", Column[Int128]([limit, 1, -1])).with_dtype(
                DataType.decimal(38, 0)
            )
        ]
    )
    with assert_raises(contains="precision"):
        _ = wide.lazy().select(col("x").cum_sum()).collect(engine="cpu")
    with assert_raises(contains="overflow"):
        _ = (
            wide.lazy()
            .select(col("x").cum_sum())
            .collect(accelerator=MetalRuntime())
        )
    same(
        wide.lazy()
        .select(col("x").cum_sum(True))
        .collect(accelerator=MetalRuntime()),
        wide.lazy().select(col("x").cum_sum(True)).collect(engine="cpu"),
    )


# requires: metal
def test_native_metal_cumulative_large_global_partition_preserves_serial_float_order() raises:
    _ = require_metal()
    var values = List[Float64](length=1048576, fill=0)
    for i in range(len(values)):
        values[i] = Float64(9007199254740992) if i % 3 == 0 else (
            Float64(1) if i % 3 == 1 else Float64(-9007199254740992)
        )
    var frame = DataFrame([Series("x", Column[Float64](values^))])
    var query = frame.lazy().select_exprs(
        [col("x").cum_sum().alias("s"), col("x").cum_count(True).alias("n")]
    )
    same(query.collect(accelerator=MetalRuntime()), query.collect(engine="cpu"))
