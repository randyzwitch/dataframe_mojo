"""Stable global and partitioned ranks through the resident Metal API."""
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.memory import bitcast
from tests.execution_modes.test_backend_metal import require_metal
from std.testing import assert_true, assert_raises
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
from dataframe.categorical import encode
from dataframe.metal import MetalRuntime
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.capabilities import RowCapabilities
from tests.execution_modes.test_backend_metal_float64_emulation import same


def methods() -> List[String]:
    return ["average", "min", "max", "dense", "ordinal"]


def test_native_metal_rank_float_words_unsigned_limits_and_partitions() raises:
    _ = require_metal()
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        1,
        0x8000000000000001,
        0x3FF0000000000000,
        0xBFF0000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000042,
        0xFFF8000000000043,
    ]
    var narrow: List[UInt32] = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x3F800000,
        0xBF800000,
        0x7F800000,
        0xFF800000,
        0x7FC00042,
        0xFFC00043,
    ]
    var a = List[Float64]()
    var b = List[Float32]()
    var u = List[UInt64]()
    var keys = List[Int32]()
    var floating_keys = List[Float64]()
    var valid = List[Bool]()
    var key_valid = List[Bool]()
    for i in range(513):
        a.append(bitcast[DType.float64](words[i % len(words)]))
        b.append(bitcast[DType.float32](narrow[i % len(narrow)]))
        u.append(UInt64(0xFFFFFFFFFFFFFFFF) - UInt64(i % 11))
        keys.append(Int32(i % 7))
        floating_keys.append(
            bitcast[DType.float64](words[(i // 3) % len(words)])
        )
        valid.append(i % 13 != 0)
        key_valid.append(i % 17 != 0)
    var frame = DataFrame(
        [
            Series("a", Column[Float64](a^, valid)),
            Series("b", Column[Float32](b^, valid)),
            Series("u", Column[UInt64](u^, valid)),
            Series("k", Column[Int32](keys^, key_valid)),
            Series("fk", Column[Float64](floating_keys^, key_valid^)),
        ]
    )
    for name in List[String](["a", "b", "u"]):
        for method in methods():
            for descending in List[Bool]([False, True]):
                var rank = col(name).rank(method, descending)
                for partition in List[String](["", "k", "fk"]):
                    var expression = rank.over(
                        partition
                    ) if partition else rank.copy()
                    var query = frame.lazy().select_exprs(
                        [col("k"), expression.alias("rank")]
                    )
                    same(
                        query.collect(accelerator=MetalRuntime()),
                        query.collect(engine="cpu"),
                    )


def test_native_metal_rank_all_numeric_widths_and_boolean() raises:
    _ = require_metal()
    comptime for D in NUMERIC_DTYPES:
        var values = List[Scalar[D]]()
        var valid = List[Bool]()
        for i in range(37):
            values.append(Scalar[D](i % 9))
            valid.append(i % 5 != 0)
        var frame = DataFrame([Series("x", Column[Scalar[D]](values^, valid^))])
        for method in methods():
            var query = frame.lazy().select(
                col("x").rank(method, descending=True)
            )
            same(
                query.collect(accelerator=MetalRuntime()),
                query.collect(engine="cpu"),
            )
    var boolean = DataFrame(
        [
            Series(
                "x",
                BoolColumn(
                    [True, False, True, False], [True, True, False, True]
                ),
            )
        ]
    )
    for method in methods():
        var query = boolean.lazy().select(col("x").rank(method))
        same(
            query.collect(accelerator=MetalRuntime()),
            query.collect(engine="cpu"),
        )


def test_native_metal_rank_string_and_logical_values_multiple_partition_keys() raises:
    _ = require_metal()
    var frame = DataFrame(
        [
            Series(
                "s",
                StringColumn(
                    List[String](["z", "a", "a", "", "z", "a"]),
                    [True, True, False, True, True, True],
                ),
            ),
            Series(
                "k",
                StringColumn(
                    List[String](["b", "a", "a", "b", "b", "a"]),
                    [True, True, True, False, False, True],
                ),
            ),
            Series("j", Column[Int32]([0, 0, 0, 0, 0, 1])),
            Series("d", Column[Int64]([4, -1, 4, 3, 3, 0])).with_dtype(
                DataType.DATE
            ),
            Series("p", Column[Int64]([125, 500, 125, 250, 250, 0])).with_dtype(
                DataType.decimal(18, 2, 64)
            ),
        ]
    )
    frame = frame.with_column(encode(frame.column("s").renamed("c")))
    for name in List[String](["s", "c", "d", "p"]):
        for method in methods():
            for descending in List[Bool]([False, True]):
                var rank = col(name).rank(method, descending).over(["k", "j"])
                var query = frame.lazy().select_exprs(
                    [col("j"), rank.alias("rank")]
                )
                same(
                    query.collect(accelerator=MetalRuntime()),
                    query.collect(engine="cpu"),
                )


def test_native_metal_rank_resident_filters_groups_joins_empty_and_cost() raises:
    _ = require_metal()
    var frame = DataFrame(
        [
            Series("k", Column[Int32]([0, 1, 0, 1, 0, 1])),
            Series("x", Column[Float64]([3, 8, 1, 8, 5, 2])),
        ]
    )
    var runtime = MetalRuntime()
    var expression = col("x").rank("ordinal", descending=True).over("k")
    var query = (
        frame.lazy().filter(col("x") > lit(1.0)).filter(expression <= 2).head(3)
    )
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    var conditional = frame.lazy().select(
        when(col("k") == lit(Int32(0)))
        .then(col("x").rank())
        .otherwise(lit(0.0))
    )
    same(
        conditional.collect(accelerator=runtime),
        conditional.collect(engine="cpu"),
    )
    var child = (
        frame.lazy()
        .with_columns(col("x").rank().rank("dense").alias("rank"))
        .slice(1, 3)
    )
    same(child.collect(accelerator=runtime), child.collect(engine="cpu"))
    var composed = frame.lazy().select(
        (col("x").rank("ordinal") % lit(Int64(3))).rank("min").over("k")
    )
    same(composed.collect(accelerator=runtime), composed.collect(engine="cpu"))
    var scoped_arithmetic = frame.lazy().select(
        (col("x").rank() + lit(0.5)).over("k")
    )
    same(
        scoped_arithmetic.collect(accelerator=runtime),
        scoped_arithmetic.collect(engine="cpu"),
    )
    var grouped = (
        frame.lazy()
        .group_by("k")
        .agg(col("x").mean().alias("mean"))
        .with_columns(col("mean").rank("min").alias("rank"))
    )
    same(grouped.collect(accelerator=runtime), grouped.collect(engine="cpu"))
    var joined = grouped.join(frame.lazy(), on="k").select(["k", "x", "rank"])
    same(joined.collect(accelerator=runtime), joined.collect(engine="cpu"))
    var empty = (
        frame.slice(0, 0).lazy().select(col("x").rank("dense").over("k"))
    )
    same(empty.collect(accelerator=runtime), empty.collect(engine="cpu"))
    var estimate = runtime.estimate_auto(
        frame.lazy().select(col("x").rank("dense"))
    )
    assert_true(";mojo-metal-v1;ops=" in estimate.family)
    same(
        query.collect(engine="auto", accelerator=runtime),
        query.collect(engine="cpu"),
    )
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))


def test_native_metal_rank_dense_prefix_across_blocks() raises:
    _ = require_metal()
    var values = List[Int64]()
    var keys = List[Int32]()
    var valid = List[Bool]()
    for i in range(4099):
        values.append(Int64((i * 31) % 521))
        keys.append(Int32(i % 3))
        valid.append(i % 19 != 0)
    var frame = DataFrame(
        [
            Series("x", Column[Int64](values^, valid^)),
            Series("k", Column[Int32](keys^)),
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            col("x").rank("dense").over("k").alias("dense"),
            col("x").rank("ordinal", descending=True).alias("ordinal"),
        ]
    )
    same(query.collect(accelerator=MetalRuntime()), query.collect(engine="cpu"))


def test_rank_capability_and_checked_input_preflight() raises:
    var frame = DataFrame([Series("x", Column[Int32]([1, 2, 3]))])
    var query = frame.lazy().select(col("x").rank())
    with assert_raises(contains="rank requires backend support"):
        _ = lower_rows(query, RowCapabilities())
    var plan = lower_rows(query, _capabilities())
    assert_true(len(plan.steps) == 2)
    assert_true(plan.outputs[0].slot == plan.steps[1].slot)
    var invalid = frame.lazy().select((col("x") + lit(Int32(1))).rank())
    with assert_raises(contains="checked rank inputs"):
        _ = lower_rows(invalid, _capabilities())
    var nested = frame.lazy().select(col("x").rank().over("x").over("x"))
    with assert_raises(contains="nested over scopes"):
        _ = lower_rows(nested, _capabilities())
    var unknown = frame.lazy().select(col("x").rank("invalid"))
    with assert_raises(contains="rank method must be"):
        _ = lower_rows(unknown, _capabilities())
