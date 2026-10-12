"""GPU partition aggregates preserve null partitions, order and errors."""
from std.memory import bitcast
from dataframe.dtype import DataType
from dataframe.categorical import encode
from std.testing import TestSuite, assert_true, assert_raises
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
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime, metal_installed, _capabilities
from dataframe.accelerator.windows import (
    has_partition_aggregates,
    lower_partition_aggregates,
)
from dataframe.accelerator.joins import lower_join_dag
from dataframe.accelerator.rows import lower_rows
from test_metal_float64_emulation import same


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
    with assert_raises(contains="nested windows"):
        _ = lower_partition_aggregates(nested)


def test_partition_aggregates_reductions_nulls_order_and_empty() raises:
    if not metal_installed():
        return
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


def test_partition_aggregate_composition_filter_join_and_consumers() raises:
    if not metal_installed():
        return
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


def test_partition_checked_sum_observation() raises:
    if not metal_installed():
        return
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


def test_partition_aggregate_key_types_slices_and_hash_groups() raises:
    if not metal_installed():
        return
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


def test_partition_window_checked_children_and_bool_reductions() raises:
    if not metal_installed():
        return
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
    # An ordinary checked join input still has no full-window observation
    # guarantee and must keep its existing metadata rejection.
    var right = DataFrame([Series("g", Column[Int32]([1]))])
    var ordinary = (
        bad.lazy()
        .select_exprs([col("g"), (col("x") + lit(Int16(2))).alias("x")])
        .join(right.lazy(), "g")
    )
    with assert_raises(contains="terminal projections without head"):
        _ = ordinary.collect(accelerator=MetalRuntime())


def test_partition_floating_reduction_bits_and_decimal_sums() raises:
    if not metal_installed():
        return
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


def main() raises:
    var suite = TestSuite()
    suite.test[test_partition_aggregate_metadata_preflight]()
    suite.test[test_partition_aggregates_reductions_nulls_order_and_empty]()
    suite.test[test_partition_aggregate_composition_filter_join_and_consumers]()
    suite.test[test_partition_checked_sum_observation]()
    suite.test[test_partition_aggregate_key_types_slices_and_hash_groups]()
    suite.test[test_partition_window_checked_children_and_bool_reductions]()
    suite.test[test_partition_floating_reduction_bits_and_decimal_sums]()
    suite^.run()
