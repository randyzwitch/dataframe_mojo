"""Trailing windows retain CPU recomputation order and checked sum boundaries."""
# contract-group: execution.backends
from dataframe.accelerator.gpu.program import Program
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    DataFrame,
    Series,
    Column,
    BoolColumn,
    StringColumn,
    Expr,
    LazyFrame,
    col,
    lit,
    when,
)
from dataframe.dtype import DataType, NUMERIC_DTYPES
from dataframe.categorical import encode
from dataframe.metal import MetalRuntime
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows, ROW_CUMULATIVE_KEY
from dataframe.accelerator.windows import lower_partition_aggregates
from tests.support.metal_outcomes import same_outcome
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_shift_fill import exact, check
from tests.execution_modes.test_backend_metal_float64_emulation import same
from tests.execution_modes.test_backend_metal_decimal_casts import (
    fixture as decimal_fixture,
)


def operation(
    expression: Expr, method: Int, window: Int, minimum: Int = -1
) -> Expr:
    if method == 7:
        return expression.rolling_sum(window, minimum)
    if method == 8:
        return expression.rolling_mean(window, minimum)
    if method == 9:
        return expression.rolling_min(window, minimum)
    return expression.rolling_max(window, minimum)


def test_rolling_capability_threshold_metadata_and_nested_scope_lowering() raises:
    assert_true(not RowCapabilities().rolling_windows)
    var frame = DataFrame(
        [
            Series("x", Column[Float32]([1, 2, 3, 4])),
            Series("k", Column[Int32]([0, 1, 0, 1])),
        ]
    )
    for method in range(7, 11):
        for window in List[Int]([1, 3, Int(Int64.MAX)]):
            for minimum in List[Int]([-1, 0, 2, 7]):
                var query = frame.lazy().select(
                    operation(col("x"), method, window, minimum).over("k")
                )
                var caps = _capabilities()
                caps.rolling_windows = False
                with assert_raises(
                    contains="rolling windows require backend support"
                ):
                    _ = lower_rows(query, caps)
                var plan = lower_rows(query, _capabilities())
                var found = False
                for step in plan.steps:
                    if plan.code[step.start * 4] == ROW_CUMULATIVE_KEY:
                        found = True
                        assert_equal(
                            plan.code[step.start * 4 + 3], Int64(method)
                        )
                        assert_equal(
                            plan.code[step.start * 4 + 2],
                            Int64(window if minimum < 0 else minimum),
                        )
                        assert_equal(
                            bitcast[DType.int64](plan.literals[step.start]),
                            Int64(window),
                        )
                        assert_equal(
                            plan.slot_dtypes[step.slot],
                            DataType.FLOAT64 if method
                            == 8 else DataType.FLOAT32,
                        )
                assert_true(found)
                _ = Program(plan)
                assert_true(
                    "unsupported"
                    not in MetalRuntime(device_id=999999).describe(query)
                )
    for window in List[Int]([0, -1, Int(Int64.MIN)]):
        with assert_raises(contains="window_size must be at least 1"):
            _ = lower_rows(
                frame.lazy().select(col("x").rolling_sum(window)),
                _capabilities(),
            )
    with assert_raises(contains="min_samples must be nonnegative"):
        _ = lower_rows(
            frame.lazy().select(col("x").rolling_min(3, -2)), _capabilities()
        )
    with assert_raises(contains="explicit Float64 cast"):
        _ = lower_rows(
            decimal_fixture(3, 2).lazy().select(col("wide").rolling_sum(2)),
            _capabilities(),
        )
    var nested = frame.lazy().select(
        col("x").rolling_mean(2, 1).over("k").rolling_min(3, 1)
    )
    exact(
        lower_partition_aggregates(nested).collect(engine="cpu"),
        nested.collect(engine="cpu"),
    )


# requires: metal
def test_native_metal_rolling_numeric_widths_windows_thresholds_and_literal_anchors() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    comptime for D in NUMERIC_DTYPES:
        var values = List[Scalar[D]]()
        var valid = List[Bool]()
        var keys = List[Int32]()
        for i in range(263):
            values.append(Scalar[D](i % 7))
            valid.append(i % 5 != 0)
            keys.append(Int32(i % 3))
        var frame = DataFrame(
            [
                Series("x", Column[Scalar[D]](values^, valid)),
                Series("k", Column[Int32](keys^)),
            ]
        ).slice(3, 257)
        for method in range(7, 11):
            for window in List[Int]([1, 3, 17, Int(Int64.MAX)]):
                for minimum in List[Int]([-1, 0, 1, 4, 300]):
                    check(
                        frame.lazy().select(
                            operation(col("x"), method, window, minimum)
                        ),
                        runtime,
                    )
                    check(
                        frame.lazy().select(
                            operation(col("x"), method, window, minimum).over(
                                "k"
                            )
                        ),
                        runtime,
                    )
            check(
                frame.head(0)
                .lazy()
                .select(operation(col("x"), method, 3).over("k")),
                runtime,
            )
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [1, 2, 3, 4, 5], [True, False, True, False, True]
                ),
            )
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            col("x").rolling_sum(3, 2).alias("s"),
            col("x").rolling_mean(3, 1).alias("m"),
            col("x").rolling_min(3, 0).alias("min"),
            col("x").rolling_max(3, 0).alias("max"),
        ]
    )
    var result = query.collect(accelerator=runtime)
    exact(
        result,
        DataFrame(
            [
                Series(
                    "s",
                    Column[Int64](
                        [0, 0, 4, 0, 8], [False, False, True, False, True]
                    ),
                ),
                Series("m", Column[Float64]([1, 1, 2, 3, 4])),
                Series("min", Column[Int64]([1, 1, 1, 3, 3])),
                Series("max", Column[Int64]([1, 1, 3, 3, 5])),
            ]
        ),
    )
    var nulls = DataFrame(
        [Series("x", Column[Int64]([1, 2, 3], [False, False, False]))]
    )
    for method in range(7, 11):
        check(nulls.lazy().select(operation(col("x"), method, 3, 0)), runtime)


# requires: metal
def test_native_metal_rolling_float_order_decimal_means_and_exact_logical_extrema() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
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
        0xFFF0000000000042,
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
        0xFF800042,
    ]
    var a = List[Float64]()
    var b = List[Float32]()
    var valid = List[Bool]()
    var keys = List[Int32]()
    for i in range(263):
        a.append(bitcast[DType.float64](words[i % len(words)]))
        b.append(bitcast[DType.float32](narrow[i % len(narrow)]))
        valid.append(i % 13 != 0)
        keys.append(Int32(i % 5))
    var frame = DataFrame(
        [
            Series("a", Column[Float64](a^, valid)),
            Series("b", Column[Float32](b^, valid^)),
            Series("k", Column[Int32](keys^)),
        ]
    )
    for method in range(7, 11):
        for window in List[Int]([1, 2, 3, 17, Int(Int64.MAX)]):
            var query = frame.lazy().select_exprs(
                [
                    operation(col("a"), method, window, 1).alias("a"),
                    operation(col("b"), method, window, 1).over("k").alias("b"),
                ]
            )
            if method >= 9:
                check(query, runtime)
            else:
                same(
                    query.collect(accelerator=runtime),
                    query.collect(engine="cpu"),
                )
    for scale in range(39):
        var decimals = decimal_fixture(19, scale)
        for window in List[Int]([1, 3, Int(Int64.MAX)]):
            check(
                decimals.lazy().select_exprs(
                    [
                        col("wide").rolling_mean(window, 1).over("k"),
                        col("large").rolling_mean(window, 1),
                        col("small").rolling_mean(window, 0),
                    ]
                ),
                runtime,
            )
    var mask: List[Bool] = [False, True, False, True, True, False]
    var logical = DataFrame(
        [
            Series("k", Column[Int32]([0, 1, 0, 1, 0, 1])),
            Series(
                "w",
                Column[Int128](
                    [1, Int128(1) << 100, 3, -(Int128(1) << 100), 5, 6], mask
                ),
            ).with_dtype(DataType.decimal(38, 2)),
            Series(
                "u",
                Column[UInt64](
                    [0, UInt64.MAX, 1, UInt64(1) << 63, UInt64.MAX - 1, 2], mask
                ),
            ),
            Series(
                "b", BoolColumn([False, True, True, False, False, True], mask)
            ),
            Series("date", Column[Int64]([1, 2, 3, 4, 5, 6], mask)).with_dtype(
                DataType.DATE
            ),
            Series("time", Column[Int64]([1, 2, 3, 4, 5, 6], mask)).with_dtype(
                DataType.datetime("ns")
            ),
            Series(
                "text", StringColumn(["", "β", "cat", "β", "dog", ""], mask)
            ),
            encode(
                Series(
                    "category",
                    StringColumn(["", "β", "cat", "β", "dog", ""], mask),
                )
            ),
        ]
    )
    for method in range(9, 11):
        for minimum in List[Int]([-1, 0, 1, 3]):
            var expressions = List[Expr]()
            for name in logical.columns():
                if name != "k":
                    expressions.append(
                        operation(col(name), method, 3, minimum)
                        .over("k")
                        .alias(name)
                    )
            check(logical.lazy().select_exprs(expressions), runtime)


# requires: metal
def test_native_metal_rolling_wide_sum_failures_observation_and_unsigned_inputs() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var wide = DataFrame(
        [Series("x", Column[Int64]([Int64.MAX, Int64.MAX, -Int64.MAX]))]
    )
    var cancellation = wide.lazy().select(col("x").rolling_sum(3, 3))
    check(cancellation, runtime)
    assert_equal(
        cancellation.collect(accelerator=runtime).column("x").int64()._get(2),
        Int64.MAX,
    )
    for expression in List[Expr](
        [
            col("x").rolling_sum(2, 2),
            when(lit(False))
            .then(col("x").rolling_sum(2, 2))
            .otherwise(lit(Int64(0))),
        ]
    ):
        var query = wide.lazy().select(expression).head(0)
        with assert_raises(contains="overflow"):
            _ = query.collect(engine="cpu")
        with assert_raises(contains="overflow"):
            _ = query.collect(accelerator=runtime)
    for target in List[DataType]([DataType.INT32, DataType.UINT32]):
        var frame = DataFrame(
            [
                Series(
                    "x",
                    Column[Int64](
                        [
                            Int64(2147483647) if target
                            == DataType.INT32 else Int64(4294967295),
                            1,
                        ]
                    ),
                ).cast(target)
            ]
        )
        var query = frame.lazy().select(col("x").rolling_sum(2, 2))
        with assert_raises(contains="overflow"):
            _ = query.collect(engine="cpu")
        with assert_raises(contains="overflow"):
            _ = query.collect(accelerator=runtime)
    var unsigned = DataFrame([Series("x", Column[UInt64]([1, UInt64.MAX]))])
    for expression in List[Expr](
        [col("x").rolling_sum(1, 100), col("x").rolling_mean(1, 100)]
    ):
        var query = unsigned.lazy().select(expression).head(0)
        with assert_raises(contains="overflow"):
            _ = query.collect(engine="cpu")
        with assert_raises(contains="overflow"):
            _ = query.collect(accelerator=runtime)
    var safe = DataFrame([Series("x", Column[UInt64]([UInt64(1) << 62, 1, 2]))])
    check(safe.lazy().select(col("x").rolling_mean(3, 1)), runtime)
    var unsafe = (
        DataFrame([Series("x", Column[Int64]([1, 2, 3]))])
        .lazy()
        .select((col("x") + lit(Int64(1))).rolling_min(3))
    )
    same_outcome(unsafe)
    check(wide.lazy().select(col("x").rolling_sum(2, 100)), runtime)


# requires: metal
def test_native_metal_rolling_nested_scopes_consumers_cost_budget_and_reuse() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var values = List[Int32]()
    var valid = List[Bool]()
    var keys = List[Int32]()
    var others = List[Int32]()
    for i in range(4097):
        values.append(Int32(i % 7))
        valid.append(i % 5 != 0)
        keys.append(Int32(i % 3))
        others.append(Int32(i % 2))
    var frame = DataFrame(
        [
            Series("x", Column[Int32](values^, valid^)),
            Series("k", Column[Int32](keys^)),
            Series("j", Column[Int32](others^)),
        ]
    )
    for expression in List[Expr](
        [
            col("x").rolling_mean(3, 1).over("k").rolling_min(5, 1).over("j"),
            col("x").forward_fill().over("j").rolling_max(7, 2).over("k"),
            col("x").rolling_sum(3, 1).over("k").shift(-1).over("j"),
            when(col("j") == lit(Int32(1)))
            .then(col("x").rolling_sum(3, 1).over("k"))
            .otherwise(col("x").rolling_max(3, 1)),
        ]
    ):
        check(frame.lazy().select(expression.alias("w")), runtime)
    var resident = (
        frame.lazy()
        .filter(col("j") == lit(Int32(0)))
        .with_columns(col("x").rolling_mean(3, 1).over("k").alias("w"))
        .sort("w")
        .select_exprs([col("k"), col("w").rolling_max(7, 1).alias("m")])
        .head(13)
    )
    check(resident, runtime)
    check(
        frame.lazy()
        .with_columns(col("x").rolling_sum(3, 1).alias("w"))
        .group_by("k")
        .agg(col("w").sum()),
        runtime,
    )
    var right = DataFrame([Series("k", Column[Int32]([0, 1, 2]))])
    check(
        frame.lazy()
        .with_columns(col("x").rolling_max(3, 1).alias("w"))
        .join(right.lazy(), "k")
        .select(col("w").rolling_mean(3, 1))
        .head(17),
        runtime,
    )
    check(frame.lazy().select(col("x").rolling_min(3, 1).sum()), runtime)
    for window in List[Int]([1, 3, 7, Int(Int64.MAX), 2, 3]):
        check(
            frame.head(17)
            .lazy()
            .select(col("x").rolling_mean(window, 1).over("k")),
            runtime,
        )
    var query = (
        frame.head(17).lazy().select(col("x").rolling_mean(3, 1).over("k"))
    )
    var estimate = runtime.estimate_auto(query)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    assert_true(";rolling-v1" in estimate.family)
    assert_true(";shift-fill-v1" not in estimate.family)
    exact(
        query.collect(engine="auto", accelerator=runtime),
        query.collect(engine="cpu"),
    )
    var profiled = query.profile(accelerator=runtime)
    exact(profiled[0], query.collect(engine="cpu"))
    for i in range(profiled[1].height() - 1):
        assert_equal(
            profiled[1].column("host_result_bytes").int64()._get(i), Int64(0)
        )
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    check(query, runtime)
