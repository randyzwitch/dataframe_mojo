"""Exact shift and bounded fill through resident ordered Metal windows."""
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
from tests.execution_modes.test_backend_metal_float64_emulation import same


def operation(expression: Expr, method: Int, parameter: Int) -> Expr:
    if method == 4:
        return expression.shift(parameter)
    if method == 5:
        return expression.forward_fill(parameter)
    return expression.backward_fill(parameter)


def exact(actual: DataFrame, expected: DataFrame) raises:
    same(actual, expected)
    # Copying windows preserve even NaN signs and payload bits.
    for name in actual.columns():
        if actual.column(name).dtype() == DataType.FLOAT32:
            var a = actual.column(name).float32()
            var b = expected.column(name).float32()
            for i in range(actual.height()):
                if b.is_valid(i):
                    assert_equal(
                        bitcast[DType.uint32](a._get(i)),
                        bitcast[DType.uint32](b._get(i)),
                    )
        elif actual.column(name).dtype() == DataType.FLOAT64:
            var a = actual.column(name).float64()
            var b = expected.column(name).float64()
            for i in range(actual.height()):
                if b.is_valid(i):
                    assert_equal(
                        bitcast[DType.uint64](a._get(i)),
                        bitcast[DType.uint64](b._get(i)),
                    )


def check(query: LazyFrame, runtime: MetalRuntime) raises:
    exact(query.collect(accelerator=runtime), query.collect(engine="cpu"))


def test_shift_fill_capability_raw_parameters_and_nested_scopes() raises:
    assert_true(not RowCapabilities().shift_fill_windows)
    var frame = DataFrame(
        [
            Series(
                "x", Column[Float32]([1, 2, 3, 4], [False, True, False, True])
            ),
            Series("k", Column[Int32]([0, 1, 0, 1])),
            Series("j", Column[Int32]([0, 0, 1, 1])),
        ]
    )
    for method in range(4, 7):
        for parameter in List[Int](
            [Int(Int64.MIN), -2, -1, 0, 1, 2, Int(Int64.MAX)]
        ):
            if method != 4 and parameter < -1:
                with assert_raises(contains="fill limit must be nonnegative"):
                    _ = lower_rows(
                        frame.lazy().select(
                            operation(col("x"), method, parameter)
                        ),
                        _capabilities(),
                    )
                continue
            var query = frame.lazy().select(
                operation(col("x"), method, parameter).over("k")
            )
            var caps = _capabilities()
            caps.shift_fill_windows = False
            with assert_raises(
                contains="shift and fill windows require backend support"
            ):
                _ = lower_rows(query, caps)
            var plan = lower_rows(query, _capabilities())
            var found = False
            for step in plan.steps:
                if plan.code[step.start * 4] == ROW_CUMULATIVE_KEY:
                    found = True
                    assert_equal(plan.code[step.start * 4 + 3], Int64(method))
                    assert_equal(
                        plan.code[step.start * 4 + 2], Int64(method == 6)
                    )
                    assert_equal(
                        bitcast[DType.int64](plan.literals[step.start]),
                        Int64(parameter),
                    )
                    assert_equal(plan.slot_dtypes[step.slot], DataType.FLOAT32)
            assert_true(found)
            _ = Program(plan)
            assert_true(
                "unsupported"
                not in MetalRuntime(device_id=999999).describe(query)
            )
    var nested = frame.lazy().select(
        col("x").forward_fill(1).over("k").shift(-1).backward_fill(2).over("j")
    )
    exact(
        lower_partition_aggregates(nested).collect(engine="cpu"),
        nested.collect(engine="cpu"),
    )


# requires: metal
def test_native_metal_shift_fill_numeric_widths_offsets_limits_and_slices() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    comptime for D in NUMERIC_DTYPES:
        var values = List[Scalar[D]]()
        var valid = List[Bool]()
        var keys = List[Int32]()
        for i in range(519):
            values.append(Scalar[D](i % 17))
            valid.append(i % 11 == 3 or i % 11 == 8)
            keys.append(Int32(i % 5))
        var frame = DataFrame(
            [
                Series("x", Column[Scalar[D]](values^, valid)),
                Series("k", Column[Int32](keys^)),
            ]
        ).slice(3, 513)
        for parameter in List[Int](
            [Int(Int64.MIN), -514, -3, -1, 0, 1, 3, 514, Int(Int64.MAX)]
        ):
            check(frame.lazy().select(col("x").shift(parameter)), runtime)
            check(
                frame.lazy().select(col("x").shift(parameter).over("k")),
                runtime,
            )
        for parameter in List[Int]([-1, 0, 1, 2, 7, Int(Int64.MAX)]):
            check(
                frame.lazy().select_exprs(
                    [
                        col("x").forward_fill(parameter).alias("f"),
                        col("x").backward_fill(parameter).over("k").alias("b"),
                    ]
                ),
                runtime,
            )
        for method in range(4, 7):
            check(
                frame.head(0)
                .lazy()
                .select(operation(col("x"), method, 1).over("k")),
                runtime,
            )
    var anchor = DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [9, 10, 11, 12, 13, 14, 15, 16],
                    [False, True, False, False, False, True, False, False],
                ),
            )
        ]
    )
    var result = (
        anchor.lazy()
        .select_exprs(
            [
                col("x").forward_fill(2).alias("f"),
                col("x").backward_fill(1).alias("b"),
                col("x").shift(-1).alias("s"),
            ]
        )
        .collect(accelerator=runtime)
    )
    exact(
        result,
        DataFrame(
            [
                Series(
                    "f",
                    Column[Int64](
                        [0, 10, 10, 10, 0, 14, 14, 14],
                        [False, True, True, True, False, True, True, True],
                    ),
                ),
                Series(
                    "b",
                    Column[Int64](
                        [10, 10, 0, 0, 14, 14, 0, 0],
                        [True, True, False, False, True, True, False, False],
                    ),
                ),
                Series(
                    "s",
                    Column[Int64](
                        [10, 0, 0, 0, 14, 0, 0, 0],
                        [True, False, False, False, True, False, False, False],
                    ),
                ),
            ]
        ),
    )
    var nulls = DataFrame(
        [Series("x", Column[Int64]([1, 2, 3], [False, False, False]))]
    )
    for method in range(4, 7):
        check(nulls.lazy().select(operation(col("x"), method, -1)), runtime)


# requires: metal
def test_native_metal_shift_fill_exact_words_logical_and_dictionary_payloads() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        1,
        0x8000000000000001,
        0x7FF8000000000042,
        0xFFF0000000000042,
        0x7FF0000000000000,
        0xFFF0000000000000,
    ]
    var narrow: List[UInt32] = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x7FC00042,
        0xFF800042,
        0x7F800000,
        0xFF800000,
    ]
    var a = List[Float64]()
    var b = List[Float32]()
    var valid = List[Bool]()
    var k = List[Int32]()
    for i in range(65):
        a.append(bitcast[DType.float64](words[i % len(words)]))
        b.append(bitcast[DType.float32](narrow[i % len(narrow)]))
        valid.append(i % 4 == 2)
        k.append(Int32(i % 3))
    var floats = DataFrame(
        [
            Series("a", Column[Float64](a^, valid)),
            Series("b", Column[Float32](b^, valid^)),
            Series("k", Column[Int32](k^)),
        ]
    )
    for method in range(4, 7):
        for parameter in List[Int]([-1, 0, 1, 2]):
            check(
                floats.lazy().select_exprs(
                    [
                        operation(col("a"), method, parameter)
                        .over("k")
                        .alias("a"),
                        operation(col("b"), method, parameter).alias("b"),
                    ]
                ),
                runtime,
            )
    var mask: List[Bool] = [False, True, False, False, True, False]
    var frame = DataFrame(
        [
            Series("k", Column[Int32]([0, 1, 0, 1, 0, 1])),
            Series(
                "wide",
                Column[Int128](
                    [1, Int128(1) << 100, 3, 4, -(Int128(1) << 100), 6], mask
                ),
            ).with_dtype(DataType.decimal(38, 2)),
            Series("d32", Column[Int32]([1, 2, 3, 4, 5, 6], mask)).with_dtype(
                DataType.decimal(9, 3, 32)
            ),
            Series("d64", Column[Int64]([1, 2, 3, 4, 5, 6], mask)).with_dtype(
                DataType.decimal(18, 7, 64)
            ),
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
                "duration", Column[Int64]([1, 2, 3, 4, 5, 6], mask)
            ).with_dtype(DataType.duration("us")),
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
    for method in range(4, 7):
        for parameter in List[Int](
            [Int(Int64.MIN), -2, -1, 0, 1, 2, Int(Int64.MAX)]
        ):
            if method != 4 and parameter < -1:
                continue
            var expressions = List[Expr]()
            for name in frame.columns():
                if name != "k":
                    expressions.append(
                        operation(col(name), method, parameter)
                        .over("k")
                        .alias(name)
                    )
            check(frame.lazy().select_exprs(expressions), runtime)
    var plain = DataFrame([Series("x", Column[UInt64]([UInt64.MAX, 1, 2]))])
    check(plain.lazy().select(col("x").forward_fill()), runtime)


# requires: metal
def test_native_metal_shift_fill_partition_boundaries_and_nested_resident_consumers() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    for rows in List[Int]([257, 4097, 65537]):
        var values = List[Int64]()
        var keys = List[Float64]()
        var key_valid = List[Bool]()
        var strings = List[String]()
        var valid = List[Bool]()
        for i in range(rows):
            values.append(Int64(i))
            keys.append(
                Float64(i % 3) if i
                % 3 else bitcast[DType.float64](UInt64(0x7FF8000000000042))
            )
            key_valid.append(i % 13 != 0)
            strings.append("a" if i % 2 else "β")
            valid.append(i % 11 == 3)
        var frame = DataFrame(
            [
                Series("x", Column[Int64](values^, valid)),
                Series("k", Column[Float64](keys^, key_valid^)),
                Series("j", StringColumn(strings^)),
            ]
        )
        check(
            frame.lazy().select_exprs(
                [
                    col("x").shift(3).over(["k", "j"]).alias("s"),
                    col("x").forward_fill(7).over(["k", "j"]).alias("f"),
                    col("x").backward_fill(7).over(["k", "j"]).alias("b"),
                ]
            ),
            runtime,
        )
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Int32](
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [False, True, False, True, False, False, True, False],
                ),
            ),
            Series("k", Column[Int32]([0, 1, 0, 1, 0, 1, 0, 1])),
            Series("j", Column[Int32]([0, 0, 1, 1, 0, 0, 1, 1])),
        ]
    )
    for expression in List[Expr](
        [
            col("x")
            .forward_fill(1)
            .over("k")
            .shift(-1)
            .backward_fill(2)
            .over("j"),
            col("x").shift().over("k").forward_fill().over("j").cum_max(),
            col("x")
            .fill_null(col("x").min().over("j"))
            .backward_fill(2)
            .over("k"),
            when(col("k") == lit(Int32(1)))
            .then(col("x").forward_fill().over("k"))
            .otherwise(col("x").backward_fill().over("j")),
        ]
    ):
        check(frame.lazy().select(expression.alias("w")), runtime)
    var resident = (
        frame.lazy()
        .filter(col("j") == lit(Int32(0)))
        .with_columns(col("x").forward_fill().over("k").alias("w"))
        .sort("w")
        .select_exprs([col("k"), col("w").shift(-1).alias("s")])
        .head(3)
    )
    check(resident, runtime)
    var grouped = (
        frame.lazy()
        .with_columns(col("x").backward_fill().alias("w"))
        .group_by("k")
        .agg(col("w").sum().alias("s"))
    )
    check(grouped, runtime)
    var right = DataFrame([Series("k", Column[Int32]([0, 1]))])
    check(
        frame.lazy()
        .with_columns(col("x").shift().alias("w"))
        .join(right.lazy(), "k")
        .select(col("w").forward_fill()),
        runtime,
    )
    check(frame.lazy().select(col("x").backward_fill().sum()), runtime)


# requires: metal
def test_native_metal_shift_fill_proofs_budget_cost_and_runtime_reuse() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "x", Column[Int64]([1, 2, 3, 4], [False, True, False, True])
            ),
            Series("k", Column[Int32]([0, 1, 0, 1])),
        ]
    )
    var unsafe = frame.lazy().select((col("x") + lit(Int64(1))).shift())
    same_outcome(unsafe)
    var safe = frame.lazy().select(
        (col("k").cast(DataType.INT64) + lit(Int64(1))).forward_fill().over("k")
    )
    check(safe, runtime)
    for offset in List[Int]([1, 2, -1, Int(Int64.MIN), 0, Int(Int64.MAX), 1]):
        var query = frame.lazy().select(col("x").shift(offset).over("k"))
        check(query, runtime)
    var query = frame.lazy().select(col("x").forward_fill(1).over("k"))
    var estimate = runtime.estimate_auto(query)
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    assert_true(";shift-fill-v1" in estimate.family)
    assert_true(";cumulative-v1" not in estimate.family)
    check(query, runtime)
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
