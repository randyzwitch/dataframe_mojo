"""GPU partition aggregates preserve null partitions, order and errors."""
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.memory import bitcast, ArcPointer
from tests.execution_modes.test_backend_metal import require_metal
from dataframe.dtype import DataType
from dataframe.categorical import encode
from std.testing import assert_true, assert_raises
from dataframe import (
    DataFrame,
    Series,
    Column,
    StringColumn,
    col,
    lit,
    when,
    Expr,
)
from dataframe.lazy import LazyFrame, SCAN_FRAME
from dataframe.metal import MetalRuntime
from dataframe.accelerator.windows import (
    has_partition_aggregates,
    lower_partition_aggregates,
)
from dataframe.accelerator.joins import lower_join_dag
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.metadata import schema_frame, prune_projections
from tests.execution_modes.test_backend_metal_float64_emulation import same


def fixture() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "g",
                Column[Int32](
                    [2, 1, 2, 9, 1, 8, 3],
                    [True, True, True, False, True, False, True],
                ),
            ),
            Series(
                "h",
                StringColumn(
                    ["b", "a", "b", "x", "b", "x", "c"],
                    [True, True, True, False, True, False, True],
                ),
            ),
            Series(
                "x",
                Column[Int16](
                    [4, -1, 8, 7, 2, 3, 0],
                    [True, True, True, True, False, True, False],
                ),
            ),
            Series(
                "__dataframe_metal_window_0",
                Column[Int32]([0, 1, 2, 3, 4, 5, 6]),
            ),
        ]
    )


def check(query: LazyFrame) raises:
    var expected = query.collect(engine="cpu")
    var actual = query.collect(accelerator=MetalRuntime())
    same(actual, expected)


def test_partition_aggregate_metadata_preflight() raises:
    var frame = fixture()
    var query = (
        frame.select(["g", "x"])
        .lazy()
        .select(col("x").sum().over("g").alias("total"))
    )
    assert_true(has_partition_aggregates(query))
    var lowered = lower_partition_aggregates(query)
    assert_true(not has_partition_aggregates(lowered))
    var dag = lower_join_dag(lowered, _capabilities())
    assert_true(dag.joins == 1)
    var null_join = False
    for task in dag.tasks:
        null_join |= task.how == 7
    assert_true(null_join)
    assert_true(
        "resident partition aggregates" in MetalRuntime().describe(query)
    )
    assert_true(
        not has_partition_aggregates(
            frame.lazy().select(col("x").rank().over("g"))
        )
    )
    var nested = frame.lazy().select(col("x").sum().over("g").sum().over("h"))
    assert_true(
        not has_partition_aggregates(lower_partition_aggregates(nested))
    )


def test_native_metal_partition_aggregates_reductions_nulls_order_and_empty() raises:
    _ = require_metal()
    var frame = fixture()
    var values: List[Expr] = [
        col("x").sum(),
        col("x").sum(min_count=1),
        col("x").sum(min_count=3),
        col("x").count(),
        col("x").len(),
        col("x").null_count(),
        col("x").min(),
        col("x").max(),
        col("x").first(),
        col("x").last(),
        col("x").mean(),
    ]
    for i in range(len(values)):
        var expression = values[i].over("g").alias("w")
        check(
            frame.lazy().select_exprs([col("g"), col("x"), expression.copy()])
        )
        check(frame.lazy().with_columns(expression))
        check(frame.slice(0, 0).lazy().select(expression))
    check(frame.lazy().select(col("x").sum().over(["g", "h"]).alias("multi")))
    var invalid = frame.lazy().select(
        col("x").sum().over(List[String]()).alias("global")
    )
    with assert_raises(contains="over requires at least one partition column"):
        _ = invalid.collect(engine="cpu")
    with assert_raises(contains="over requires at least one partition column"):
        _ = lower_partition_aggregates(invalid)


def test_native_metal_partition_aggregate_composition_filter_join_and_consumers() raises:
    _ = require_metal()
    var frame = fixture()
    check(
        frame.lazy().select_exprs(
            [
                (col("x").sum() + col("x").count()).over("g").alias("composed"),
                col("x").max().over("h").alias("other"),
            ]
        )
    )
    check(frame.lazy().filter(col("x").sum().over("g") > 5))
    check(
        frame.lazy()
        .with_columns(col("x").sum().over("g").alias("total"))
        .filter(col("total") > 5)
        .select(col("total").sum())
    )
    check(
        frame.lazy()
        .filter(col("g") < 3)
        .select(col("x").sum().over("g").alias("total"))
        .head(2)
    )
    check(
        frame.lazy().select(
            when(col("x").is_not_null())
            .then(col("x").sum().over("g"))
            .otherwise(0)
            .alias("conditional")
        )
    )
    var right = DataFrame([Series("g", Column[Int32]([1, 2, 3]))])
    check(
        frame.lazy()
        .join(right.lazy(), "g")
        .select(col("x").sum().over("g").alias("total"))
    )
    var measured = (
        frame.lazy()
        .select(col("x").sum().over("g").alias("total"))
        .profile(accelerator=MetalRuntime())
    )
    assert_true(
        measured[0].equals(
            frame.select(col("x").sum().over("g").alias("total"))
        )
    )
    assert_true(measured[1].height() >= 3)
    with assert_raises(contains="budget"):
        _ = (
            frame.lazy()
            .select(col("x").sum().over("g").alias("total"))
            .collect(accelerator=MetalRuntime(memory_limit_bytes=1))
        )


def test_native_metal_partition_checked_sum_observation() raises:
    _ = require_metal()
    var frame = DataFrame(
        [
            Series("g", Column[Int32]([1, 2, 1, 2])),
            Series("x", Column[Int32]([2147483647, 3, 1, 4])),
        ]
    )
    var expressions: List[Expr] = [
        col("x").sum().over("g").alias("w"),
        when(lit(False)).then(col("x").sum().over("g")).otherwise(0).alias("w"),
    ]
    for expression in expressions:
        var query = frame.lazy().select(expression)
        with assert_raises(contains="overflow"):
            _ = query.collect(engine="cpu")
        with assert_raises(contains="overflow"):
            _ = query.collect(accelerator=MetalRuntime())
        with assert_raises(contains="overflow"):
            _ = query.head(1).collect(engine="cpu")
        with assert_raises(contains="overflow"):
            _ = query.head(1).collect(accelerator=MetalRuntime())
        with assert_raises(contains="overflow"):
            _ = query.head(0).collect(engine="cpu")
        with assert_raises(contains="overflow"):
            _ = query.head(0).collect(accelerator=MetalRuntime())
    var valid = DataFrame(
        [
            Series("g", Column[Int32]([1, 2, 1, 2])),
            Series("x", Column[Int32]([4, 3, 1, 4])),
        ]
    )
    check(
        valid.lazy().select((col("x").sum() + 2).over("g").alias("w")).head(1)
    )
    check(valid.lazy().filter(col("x").sum().over("g") > 5))


def test_native_metal_partition_aggregate_key_types_slices_and_hash_groups() raises:
    _ = require_metal()
    var keys = List[Float64]()
    var integers = List[Int64]()
    var wide = List[Int128]()
    var values = List[Int16]()
    var valid = List[Bool]()
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        0x7FF8000000000042,
        0xFFF8000000000043,
        0x3FF0000000000000,
        0xBFF0000000000000,
    ]
    for i in range(521):
        keys.append(bitcast[DType.float64](words[i % len(words)]))
        integers.append(Int64(i % 9))
        wide.append((Int128(i % 9) << 80) | Int128(i % 3))
        values.append(Int16(i % 13 - 6))
        valid.append(i % 11 != 3)
    var frame = DataFrame(
        [
            Series("f", Column[Float64](keys^, valid)),
            Series("d", Column[Int64](integers.copy(), valid)).with_dtype(
                DataType.DATE
            ),
            Series("p", Column[Int64](integers^, valid)).with_dtype(
                DataType.decimal(18, 2, 64)
            ),
            Series("w", Column[Int128](wide^, valid^)).with_dtype(
                DataType.decimal(38, 2)
            ),
            Series("x", Column[Int16](values^)),
        ]
    ).slice(3, 513)
    for key in List[String](["f", "d", "p", "w"]):
        check(
            frame.lazy().select_exprs(
                [col(key), col("x").sum().over(key).alias("total")]
            )
        )
        check(
            frame.lazy().select(
                col("x")
                .count()
                .over(
                    List[String]([key, "p"]) if key
                    == "d" else List[String]([key, "d"])
                )
                .alias("n")
            )
        )
    var text = fixture()
    text = text.with_column(encode(text.column("h").renamed("category")))
    for key in List[String](["h", "category"]):
        check(text.lazy().select(col("x").sum().over(key).alias("total")))
    # More than 4096 build groups exercise native hash matching rather than
    # only the small sorted-key lookup. Include repeated null partitions.
    var many_keys = List[Int32]()
    var many_wide = List[Int128]()
    var many_values = List[Int16]()
    var many_valid = List[Bool]()
    for i in range(8205):
        many_keys.append(Int32(i % 4101))
        many_wide.append(Int128(i % 4101) << 80)
        many_values.append(Int16(i % 7))
        many_valid.append(i % 103 != 4)
    var many = DataFrame(
        [
            Series("g", Column[Int32](many_keys^, many_valid)),
            Series("w", Column[Int128](many_wide^, many_valid^)).with_dtype(
                DataType.decimal(38, 2)
            ),
            Series("x", Column[Int16](many_values^)),
        ]
    )
    check(many.lazy().select(col("x").sum().over("g").alias("total")))
    check(many.lazy().select(col("x").sum().over("w").alias("total")))


def test_native_metal_partition_window_checked_children_and_bool_reductions() raises:
    _ = require_metal()
    var frame = fixture()
    check(
        frame.lazy().select_exprs(
            [
                (col("x") > 1).any().over("g").alias("any"),
                (col("x") > 1).all().over("g").alias("all"),
            ]
        )
    )
    check(
        frame.lazy().select(
            (col("x") + lit(Int16(2))).sum().over("g").alias("total")
        )
    )
    check(
        frame.lazy().select(
            col("x").cast("int8").sum().over("g").alias("total")
        )
    )
    var bad = DataFrame(
        [
            Series("g", Column[Int32]([1, 1])),
            Series("x", Column[Int16]([32767, 1])),
        ]
    )
    var expression = (
        when(lit(False))
        .then((col("x") + lit(Int16(2))).sum().over("g"))
        .otherwise(0)
        .alias("w")
    )
    var query = bad.lazy().select(expression)
    with assert_raises(contains="overflow"):
        _ = query.collect(engine="cpu")
    with assert_raises(contains="overflow"):
        _ = query.collect(accelerator=MetalRuntime())
    # Ordinary checked JOIN inputs now have their own fully observed producer.
    # An overflow must be raised before matching or exporting any JOIN rows.
    var right = DataFrame([Series("g", Column[Int32]([1]))])
    var ordinary = (
        bad.lazy()
        .select_exprs([col("g"), (col("x") + lit(Int16(2))).alias("x")])
        .join(right.lazy(), "g")
    )
    with assert_raises(contains="overflow"):
        _ = ordinary.collect(engine="cpu", streaming=False, optimize=False)
    with assert_raises(contains="overflow"):
        _ = ordinary.collect(accelerator=MetalRuntime(), optimize=False)


def test_native_metal_partition_floating_reduction_bits_and_decimal_sums() raises:
    _ = require_metal()
    for n in List[Int]([513, 196609]):
        var a = List[Float64]()
        var b = List[Float32]()
        var keys = List[Int32]()
        var valid = List[Bool]()
        var pattern: List[Float64] = [1e16, 1.0, -1e16, 5.0, -2.0, 0.0]
        for i in range(n):
            a.append(pattern[i % len(pattern)])
            b.append(Float32(pattern[i % len(pattern)]))
            keys.append(Int32(i % 7))
            valid.append(i % 13 != 5)
        var frame = DataFrame(
            [
                Series("g", Column[Int32](keys^)),
                Series("a", Column[Float64](a^, valid)),
                Series("b", Column[Float32](b^, valid^)),
            ]
        )
        for name in List[String](["a", "b"]):
            check(frame.lazy().select(col(name).sum().over("g").alias("sum")))
            check(frame.lazy().select(col(name).mean().over("g").alias("mean")))
    var decimal = DataFrame(
        [
            Series("g", Column[Int32]([1, 2, 1, 2])),
            Series(
                "d",
                Column[Int64](
                    [
                        999999999999999999,
                        -999999999999999999,
                        999999999999999999,
                        -999999999999999999,
                    ]
                ),
            ).with_dtype(DataType.decimal(18, 2, 64)),
        ]
    )
    check(decimal.lazy().select(col("d").sum().over("g").alias("total")))


def test_native_metal_nested_partition_aggregates_composition_consumers_and_nulls() raises:
    _ = require_metal()
    var frame = fixture()
    var nested = col("x").sum().over("g").sum().over("h")
    check(
        frame.lazy().select_exprs(
            [
                nested.alias("nested"),
                col("x").mean().over("h").mean().over("g").alias("mean"),
                col("x").n_unique().over("g").sum().over("h").alias("distinct"),
                (col("x").sum().over("g") + col("x").count().over("h"))
                .sum()
                .over("g")
                .alias("combined"),
                col("x")
                .sum()
                .over("g")
                .mean()
                .over("h")
                .sum()
                .over("g")
                .alias("three_levels"),
            ]
        )
    )
    check(frame.lazy().filter(nested > 5))
    check(frame.lazy().with_columns(nested.alias("nested")))
    check(frame.lazy().select(nested.alias("nested")).head(0))
    check(frame.slice(0, 0).lazy().select(nested.alias("nested")))
    check(
        frame.lazy().select(
            col("x")
            .sum(min_count=3)
            .over("g")
            .sum(min_count=1)
            .over("h")
            .alias("nulls")
        )
    )
    var right = DataFrame([Series("g", Column[Int32]([1, 2, 3]))])
    check(frame.lazy().join(right.lazy(), "g").select(nested.alias("nested")))
    var measured = (
        frame.lazy()
        .select(nested.alias("nested"))
        .profile(accelerator=MetalRuntime())
    )
    same(
        measured[0],
        frame.lazy().select(nested.alias("nested")).collect(engine="cpu"),
    )
    assert_true(measured[1].height() >= 4)
    with assert_raises(contains="budget"):
        _ = (
            frame.lazy()
            .select(nested.alias("nested"))
            .collect(accelerator=MetalRuntime(memory_limit_bytes=1))
        )


def test_native_metal_nested_partition_checked_errors_are_fully_observed() raises:
    _ = require_metal()
    for outer in List[Bool]([False, True]):
        var gs: List[Int32] = [1, 1]
        var xs = List[Int32]([Int32.MAX, 1])
        if outer:
            gs = [1, 2]
            xs = [Int32.MAX // 2 + 1, Int32.MAX // 2 + 1]
        var frame = DataFrame(
            [
                Series("g", Column[Int32](gs^)),
                Series("h", Column[Int32]([1, 1])),
                Series("x", Column[Int32](xs^)),
            ]
        )
        var nested = col("x").sum().over("g").sum().over("h")
        var query = (
            frame.lazy()
            .select(when(lit(False)).then(nested).otherwise(0).alias("hidden"))
            .head(0)
        )
        with assert_raises():
            _ = query.collect(engine="cpu")
        with assert_raises():
            _ = query.collect(accelerator=MetalRuntime())
    var valid = DataFrame(
        [
            Series("g", Column[Int32]([1, 2])),
            Series("h", Column[Int32]([1, 1])),
            Series("x", Column[Int32]([1, 2])),
        ]
    )
    check(
        valid.lazy().select(
            col("x").sum().over("g").sum().over("h").alias("nested")
        )
    )


def test_native_metal_nested_partition_floating_worker_order() raises:
    _ = require_metal()
    var gs = List[Int32]()
    var hs = List[Int32]()
    var xs = List[Float64]()
    var ys = List[Float32]()
    var pattern64 = List[Float64]([1e16, 1, -1e16, 1])
    var pattern32: List[Float32] = [1e8, 1, -1e8, 1]
    for i in range(196609):
        gs.append(Int32(i % 4))
        hs.append(Int32(i % 7))
        xs.append(pattern64[i % 4])
        ys.append(pattern32[i % 4])
    var frame = DataFrame(
        [
            Series("g", Column[Int32](gs^)),
            Series("h", Column[Int32](hs^)),
            Series("x", Column[Float64](xs^)),
            Series("y", Column[Float32](ys^)),
        ]
    )
    for name in List[String](["x", "y"]):
        check(
            frame.lazy().select(
                col(name).first().over("g").sum().over("h").alias("nested")
            )
        )


def test_shared_partition_metadata_preserves_cpu_empty_execution() raises:
    var frame = fixture()
    var scalar = frame.lazy().select_exprs(
        [lit(Int32(1)).alias("g"), col("x").count().alias("n")]
    )
    var other = frame.lazy().select_exprs(
        [lit(Int32(2)).alias("g"), col("x").len().alias("n")]
    )
    var queries = List[LazyFrame]()
    queries.append(scalar.copy())
    queries.append(scalar.concat(other))
    queries.append(scalar.join(other, "g", how="full"))
    queries.append(scalar.join(other, how="cross"))
    queries.append(scalar.concat(other).sort("g", descending=True).head(1))
    queries.append(scalar.concat(other).unique(["n"]))
    queries.append(scalar.concat(other).filter(col("g") > 1))
    queries.append(scalar.concat(other).with_columns((col("n") + 1).alias("z")))
    queries.append(scalar.concat(other).group_by("n").agg(col("g").sum()))
    queries.append(scalar.concat(other).select(col("g").first().alias("first")))
    queries.append(frame.lazy().select(col("x").sum().over("g")))
    for query in queries:
        var index = len(query._nodes) - 1
        same(schema_frame(query, index), query._execute(index, True))
        var original = query.copy()
        var cached = query.copy()
        original._push_projections()
        prune_projections(cached)
        var original_text = String()
        var cached_text = String()
        original._describe(len(original._nodes) - 1, 0, original_text, False)
        cached._describe(len(cached._nodes) - 1, 0, cached_text, False)
        assert_true(original_text == cached_text)
        same(
            schema_frame(cached, len(cached._nodes) - 1),
            original._execute(len(original._nodes) - 1, True),
        )
    assert_true(schema_frame(scalar, len(scalar._nodes) - 1).height() == 1)
    var invalid = scalar.select(col("missing"))
    with assert_raises(contains="missing"):
        _ = schema_frame(invalid, len(invalid._nodes) - 1)
    # Probing a subtree must ignore invalid, unreachable later nodes.
    same(
        schema_frame(invalid, len(scalar._nodes) - 1),
        scalar._execute(len(scalar._nodes) - 1, True),
    )
    # Node indices need not follow the CPU's left-before-right traversal.
    var invalid_join = scalar.select(col("first_missing")).join(
        scalar.select(col("second_missing")), how="cross"
    )
    var root = len(invalid_join._nodes) - 1
    var earlier = invalid_join._nodes[root].left
    invalid_join._nodes[root].left = invalid_join._nodes[root].right
    invalid_join._nodes[root].right = earlier
    with assert_raises(contains="second_missing"):
        _ = invalid_join._execute(root, True)
    with assert_raises(contains="second_missing"):
        _ = schema_frame(invalid_join, root)
    var expression = col("x")
    for _ in range(24):
        expression = expression.min().over("g")
    var query = frame.select(["g", "x"]).lazy().select(expression.alias("deep"))
    var counters = List[ArcPointer[Int]]()
    for i in range(len(query._nodes)):
        if query._nodes[i].kind == SCAN_FRAME:
            var counter = ArcPointer(Int(0))
            query._nodes[i]._executions = counter.copy()
            counters.append(counter^)
    var lowered = lower_partition_aggregates(query)
    assert_true(not has_partition_aggregates(lowered))
    var metadata = schema_frame(lowered, len(lowered._nodes) - 1)
    assert_true(metadata.height() == 0)
    assert_true(metadata.columns() == List[String](["deep"]))
    assert_true(metadata._columns[0].dtype() == DataType.INT16)
    var dag = lower_join_dag(lowered, _capabilities())
    assert_true(dag.joins == 24, String(dag.joins))
    for counter in counters:
        assert_true(counter[] == 0)


def test_native_metal_deep_shared_partition_metadata_stays_resident() raises:
    _ = require_metal()
    var frame = fixture()
    var expression = col("x")
    for level in range(8):
        expression = expression.min().over("g" if level % 2 == 0 else "h")
    check(frame.lazy().select(expression.alias("deep")))
