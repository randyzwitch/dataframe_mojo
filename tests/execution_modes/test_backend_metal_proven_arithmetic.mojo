"""Static integer error proofs preserve CPU observations across Metal stages."""
# contract-group: execution.backends
from dataframe.accelerator.gpu.runtime import capabilities as _capabilities
from std.testing import assert_equal, assert_true, assert_raises
from tests.execution_modes.test_backend_metal import require_metal
from dataframe import DataFrame, Series, Column, col, lit, Expr, when
from dataframe.binding import bind
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.integer_ranges import proven_integer_nodes
from tests.execution_modes.test_backend_metal_float64_emulation import same


def frame() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "x",
                Column[Int64](
                    [Int64.MIN, -7, -1, 0, 1, 7, Int64.MAX],
                    [True, True, False, True, True, True, True],
                ),
            ),
            Series(
                "u", Column[UInt64]([0, 1, 2, 3, 4, UInt64.MAX - 1, UInt64.MAX])
            ),
            Series("small", Column[Int8]([-128, -7, -1, 0, 1, 7, 127])),
        ]
    )


def proved(expr: Expr, source: DataFrame) raises -> Bool:
    var flags = proven_integer_nodes(bind(expr, source._columns))
    return flags[len(flags) - 1]


def check(query: LazyFrame) raises:
    same(query.collect(accelerator=MetalRuntime()), query.collect(engine="cpu"))


def test_integer_proofs_cover_unsigned_product_and_literal_boundaries() raises:
    var source = frame()
    assert_true(proved(col("x") + lit(Int64(0)), source))
    assert_true(proved(col("x") // lit(Int64(2)), source))
    assert_true(proved(col("x") // lit(Int64(-2)), source))
    assert_true(proved(col("u") // col("u"), source))
    assert_true(proved(col("x") // lit(Int64(0)), source))
    assert_true(proved(col("u") * lit(UInt64(1)), source))
    assert_true(proved(col("u") - lit(UInt64(0)), source))
    assert_true(proved(col("x").pow(lit(Int64(0))), source))
    assert_true(proved(col("x").pow(lit(Int64(1))), source))
    assert_true(proved(col("small").cast("int64") + lit(Int64(1)), source))
    assert_true(proved(col("small").cast("int64"), source))
    assert_true(not proved(col("u") * col("u"), source))
    assert_true(not proved(col("u") + lit(UInt64(1)), source))
    assert_true(not proved(col("u") - lit(UInt64(1)), source))
    assert_true(not proved(col("x") + lit(Int64(1)), source))
    assert_true(not proved(-col("x"), source))
    assert_true(not proved(col("x").abs(), source))
    assert_true(not proved(col("x") // lit(Int64(-1)), source))
    assert_true(not proved(col("x").cast("int32"), source))
    var choices = (
        when(col("x") > 0).then(lit(Int64(0))).otherwise(lit(Int64(1)))
    )
    assert_true(proved(choices * lit(Int64.MAX), source))
    assert_true(proved((col("x") > 0).cast("int64") * lit(Int64.MAX), source))
    var bounded = col("small").clip(lit(Int8(-4)), lit(Int8(4)))
    assert_true(proved(bounded * bounded, source))
    assert_true(proved(-bounded, source))
    assert_true(proved(bounded.abs(), source))
    var query = (
        source.lazy()
        .with_columns((col("x") + lit(Int64(1))).alias("bad"))
        .head(0)
    )
    var restricted = _capabilities()
    restricted.deferred_integer_errors = False
    with assert_raises(contains="terminal projections"):
        _ = lower_rows(query, restricted)
    # A proof for a parent must never hide an unproved child operation.
    with assert_raises(contains="terminal projections"):
        _ = lower_rows(
            source.lazy()
            .select(((col("x") + lit(Int64(1))) * lit(Int64(0))).alias("bad"))
            .head(0),
            restricted,
        )
    _ = lower_rows(query, _capabilities())


# requires: metal
def test_native_metal_proven_integer_gpu_stages_head_nulls_empty_and_checked_rejection() raises:
    _ = require_metal()
    var source = frame()
    var bounded = col("small").clip(lit(Int8(-4)), lit(Int8(4)))
    for count in List[Int]([0, 1, 7]):
        check(
            source.lazy()
            .select_exprs(
                [
                    (col("x") // lit(Int64(2))).alias("divided"),
                    (col("x") // lit(Int64(-2))).alias("negative_divisor"),
                    (col("u") // col("u")).alias("unsigned_division"),
                    (col("x") // lit(Int64(0))).alias("nulls"),
                    (col("u") * lit(UInt64(1))).alias("unsigned"),
                    (col("small").cast("int64") + lit(Int64(1))).alias(
                        "widened"
                    ),
                    (bounded * bounded).alias("squared"),
                    ((col("x") > 0).cast("int64") * lit(Int64.MAX)).alias(
                        "boolean"
                    ),
                    (
                        when(col("x") > 0)
                        .then(lit(Int64(0)))
                        .otherwise(lit(Int64(1)))
                        * lit(Int64.MAX)
                    ).alias("choice"),
                ]
            )
            .head(count)
        )
    check(
        source.slice(0, 0)
        .lazy()
        .with_columns((col("x") + lit(Int64(0))).alias("safe"))
        .head(0)
    )
    check(
        source.lazy()
        .with_columns((col("x") // lit(Int64(2))).alias("safe"))
        .filter(col("safe") < 0)
        .select(col("safe"))
    )
    check(source.lazy().filter((col("x") + lit(Int64(0))) < 0).head(1))
    check(source.lazy().select(col("small").cast("int64")).head(1))
    var runtime = MetalRuntime()
    var unsafe = source.lazy().with_columns(
        (col("x") + lit(Int64(1))).alias("bad")
    )
    # CPU pushes a leading slice below row-local arithmetic. The overflowing
    # last row is unobserved by these prefixes; Mojo carries its deferred fault.
    for count in [0, 1]:
        var actual = unsafe.head(count).collect(accelerator=runtime)
        same(actual, unsafe.head(count).collect(engine="cpu"))
        assert_equal(actual.columns(), ["x", "u", "small", "bad"])
        assert_equal(actual.height(), count)
        if count:
            assert_equal(actual.item(0, "bad").int64(), Int64.MIN + 1)
    with assert_raises(contains="overflow"):
        _ = unsafe.collect(engine="cpu")
    with assert_raises(contains="overflow"):
        _ = unsafe.collect(accelerator=runtime)
    var recovered = unsafe.head(1).collect(accelerator=runtime)
    assert_equal(recovered.item(0, "bad").int64(), Int64.MIN + 1)
    var restricted = _capabilities()
    restricted.deferred_integer_errors = False
    with assert_raises(contains="terminal projections"):
        _ = lower_rows(unsafe.filter(col("x") > lit(Int64(0))), restricted)
    with assert_raises(contains="strict casts"):
        _ = (
            source.lazy()
            .select(col("x").cast("int32"))
            .head(0)
            .collect(accelerator=MetalRuntime())
        )


# requires: metal
def test_native_metal_proven_integer_full_int8_domain_compositions_and_resident_joins() raises:
    _ = require_metal()
    var values = List[Int8]()
    for i in range(-128, 128):
        values.append(Int8(i))
    var source = DataFrame([Series("x", Column[Int8](values^))])
    var bounded = col("x").clip(lit(Int8(-4)), lit(Int8(4)))
    var query = source.lazy().with_columns((bounded * bounded).alias("square"))
    check(query.filter(col("square") > lit(Int8(2))).head(19))
    check(query.sort("square").head(17))
    check(
        query.join(
            source.lazy().select(col("x").alias("other")), how="cross"
        ).head(9)
    )
    check(
        source.lazy()
        .select(col("x").cast("int64").alias("wide"))
        .join(source.lazy().select(col("x").alias("other")), how="cross")
        .head(11)
    )
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
