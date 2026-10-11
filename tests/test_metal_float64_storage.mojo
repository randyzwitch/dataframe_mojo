"""Opaque Float64 payloads retain exact IEEE bits without GPU FP64 computation."""
from std.memory import bitcast
from std.testing import TestSuite, assert_true, assert_equal, assert_raises
from dataframe import DataFrame, Series, Column, Expr, col, lit, when
from dataframe.dtype import DataType
from dataframe.metal import MetalRuntime, metal_installed


def fixture() raises -> DataFrame:
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        0x3FF0000000000000,
        0xBFF0000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000042,
        0x7FF0000000000042,
        1,
        0x8000000000000001,
        0x7FEFFFFFFFFFFFFF,
    ]
    var values = List[Float64]()
    var keys = List[Int32]()
    var valid = List[Bool]()
    var flags = List[Bool]()
    for i in range(len(words)):
        values.append(bitcast[DType.float64](words[i]))
        keys.append(Int32(i % 4))
        valid.append(i % 4 != 1)
        flags.append(i % 2 == 0)
    return DataFrame(
        [
            Series("k", Column[Int32](keys^)),
            Series("f", Column[Float64](values^, valid^)),
            Series("b", Column[Bool](flags^)),
        ]
    )


def same_bits(actual: DataFrame, expected: DataFrame) raises:
    assert_true(actual.equals(expected))
    for name in actual.columns():
        if actual.column(name).dtype() != DataType.FLOAT64:
            continue
        var left = actual.column(name).float64()
        var right = expected.column(name).float64()
        for i in range(actual.height()):
            assert_equal(left.is_valid(i), right.is_valid(i))
            if left.is_valid(i):
                assert_equal(
                    bitcast[DType.uint64](left._get(i)),
                    bitcast[DType.uint64](right._get(i)),
                )


def test_float64_storage_projection_filter_literals_and_conditionals() raises:
    if not metal_installed():
        return
    var frame = fixture()
    var runtime = MetalRuntime()
    var literal = bitcast[DType.float64](UInt64(0x8000000000000001))
    var projected = frame.lazy().select_exprs(
        [
            col("f"),
            col("f").is_null().alias("null"),
            col("f").is_not_null().alias("valid"),
            col("f").fill_null(lit(literal)).alias("filled"),
            lit(literal).alias("constant"),
            when(col("b"))
            .then(col("f"))
            .otherwise(lit(literal))
            .alias("choice"),
        ]
    )
    same_bits(
        projected.collect(accelerator=runtime), projected.collect(engine="cpu")
    )
    var filtered = (
        frame.lazy().filter(col("b")).sort("k").slice(-4, 3).select(col("f"))
    )
    same_bits(
        filtered.collect(accelerator=runtime), filtered.collect(engine="cpu")
    )
    var empty = frame.slice(0, 0).lazy().select(col("f"))
    same_bits(empty.collect(accelerator=runtime), empty.collect(engine="cpu"))


def test_float64_payload_all_joins_groups_unique_and_concat() raises:
    if not metal_installed():
        return
    var frame = fixture()
    var runtime = MetalRuntime()
    for how in List[String](
        ["inner", "left", "right", "full", "semi", "anti", "cross"]
    ):
        var query = frame.lazy().join(
            frame.slice(1, 4).lazy(), on="k", how=how
        ) if how != "cross" else frame.lazy().join(
            frame.slice(1, 4).lazy(), how="cross"
        )
        same_bits(
            query.collect(accelerator=runtime), query.collect(engine="cpu")
        )
    var grouped = (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                col("f").first().alias("first"),
                col("f").last().alias("last"),
                col("f").count().alias("n"),
            ]
        )
    )
    same_bits(
        grouped.collect(accelerator=runtime), grouped.collect(engine="cpu")
    )
    for keep in List[String](["first", "last", "none"]):
        var unique = frame.lazy().unique(["k"], keep=keep, maintain_order=True)
        same_bits(
            unique.collect(accelerator=runtime), unique.collect(engine="cpu")
        )
    var combined = frame.lazy().concat(frame.slice(-3).lazy())
    same_bits(
        combined.collect(accelerator=runtime), combined.collect(engine="cpu")
    )
    var selected = frame.lazy().select_exprs(
        [
            col("f").first().alias("first"),
            col("f").last().alias("last"),
            col("f").null_count().alias("nulls"),
            col("f").count().alias("n"),
        ]
    )
    same_bits(
        selected.collect(accelerator=runtime), selected.collect(engine="cpu")
    )


def test_remaining_float64_operators_reject_before_device_initialization() raises:
    var frame = fixture()
    var runtime = MetalRuntime(device_id=999999)
    for expression in List[Expr](
        [
            col("f").round(2),
        ]
    ):
        with assert_raises(contains="Float64 requires CPU"):
            _ = frame.lazy().select(expression).collect(accelerator=runtime)
    for expression in List[Expr]([col("f").var(), col("f").std()]):
        with assert_raises(contains="unsupported"):
            _ = frame.lazy().select(expression).collect(accelerator=runtime)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
