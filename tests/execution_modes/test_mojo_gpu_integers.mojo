"""Direct query coverage for integer-only operators and membership payloads."""
from std.testing import assert_equal, assert_raises
from dataframe import Column, DataFrame, Series, Expr, col, lit, when
from dataframe.metal import MetalRuntime
from tests.support.coverage_compare import assert_frame


def test_mojo_gpu_integer_division_modulo_power_rounding_and_membership() raises:
    comptime types = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    )
    comptime for k in range(len(types)):
        comptime D = types[k]
        var left = List[Scalar[D]]()
        var right = List[Scalar[D]]()
        var valid = List[Bool]()
        for a in range(-7 if D.is_signed() else 0, 8):
            for b in range(-3 if D.is_signed() else 0, 4):
                left.append(Scalar[D](a))
                right.append(Scalar[D](b))
                valid.append(len(left) % 7 != 3)
        var x = Series("x", Column[Scalar[D]](left^, valid.copy()))
        var y = Series("y", Column[Scalar[D]](right^, valid^))
        x = Series._from_chunks(
            [x.head(9), x.slice(9, 0), x.slice(9, len(x) - 9)]
        )
        var frame = DataFrame([x^, y^])
        var expressions: List[Expr] = [
            (col("x") // col("y")).alias("quotient"),
            (col("x") % col("y")).alias("remainder"),
            col("x").pow(lit(Scalar[D](1))).alias("power"),
            col("x").floor().alias("floor"),
            col("x").ceil().alias("ceil"),
            col("x").round(2).alias("rounded"),
            col("x")
            .is_in([lit(Scalar[D](0)), lit(Scalar[D](7)), lit(Scalar[D](7))])
            .alias("member"),
        ]
        for offset in [0, 3]:
            var query = (
                frame.slice(offset, frame.height() - offset)
                .lazy()
                .select_exprs(expressions)
            )
            assert_frame(
                query.collect(accelerator=MetalRuntime()),
                query.collect(engine="cpu", streaming=False),
            )


def test_mojo_gpu_integer_operator_endpoints_conditional_errors_and_reuse() raises:
    var source = DataFrame(
        [
            Series("x", Column[Int64]([Int64.MIN, 7, -7, Int64.MAX])),
            Series("y", Column[Int64]([-1, -3, 3, 0])),
            Series("use", Column[Bool]([False, True, True, True])),
        ]
    )
    var runtime = MetalRuntime()
    var modulo = (
        source.lazy()
        .select((col("x") % col("y")).alias("v"))
        .collect(accelerator=runtime)
    )
    assert_equal(modulo.item(0, "v").int64(), Int64(0))
    assert_equal(modulo.item(1, "v").int64(), Int64(-2))
    assert_equal(modulo.item(2, "v").int64(), Int64(2))
    assert_equal(modulo.item(3, "v").is_null(), True)
    var selected = source.lazy().select(
        when(col("use"))
        .then(col("x") // col("y"))
        .otherwise(lit(Int64(0)))
        .alias("v")
    )
    var actual = selected.collect(accelerator=runtime)
    assert_equal(actual.item(0, "v").int64(), Int64(0))
    assert_equal(actual.item(1, "v").int64(), Int64(-3))
    assert_equal(actual.item(2, "v").int64(), Int64(-3))
    assert_equal(actual.item(3, "v").is_null(), True)
    with assert_raises():
        _ = (
            source.lazy()
            .select(col("x") // col("y"))
            .collect(accelerator=runtime)
        )
    var power = DataFrame(
        [
            Series("x", Column[Int8]([2, 2, -2, 0])),
            Series("y", Column[Int8]([7, 6, 7, 0])),
            Series("use", Column[Bool]([False, True, True, True])),
        ]
    )
    var chosen = (
        power.lazy()
        .select(
            when(col("use"))
            .then(col("x").pow(col("y")))
            .otherwise(lit(Int8(0)))
            .alias("v")
        )
        .collect(accelerator=runtime)
    )
    assert_equal(chosen.item(0, "v").int8(), Int8(0))
    assert_equal(chosen.item(1, "v").int8(), Int8(64))
    assert_equal(chosen.item(2, "v").int8(), Int8(-128))
    assert_equal(chosen.item(3, "v").int8(), Int8(1))
    with assert_raises():
        _ = (
            power.lazy()
            .select(col("x").pow(col("y")))
            .collect(accelerator=runtime)
        )
    var negative_power = power.with_columns(lit(Int8(-1)).alias("y")).lazy()
    with assert_raises(contains="nonnegative exponent"):
        _ = negative_power.select(col("x").pow(col("y"))).collect(
            accelerator=runtime
        )
    var masked_negative = (
        negative_power.with_columns(lit(False).alias("use"))
        .select(
            when(col("use"))
            .then(col("x").pow(col("y")))
            .otherwise(lit(Int8(0)))
            .alias("v")
        )
        .collect(accelerator=runtime)
    )
    for row in range(masked_negative.height()):
        assert_equal(masked_negative.item(row, "v").int8(), Int8(0))
    # A checked producer fault remains observable when a grouped reducer
    # independently overflows; it must not be replaced by the reducer fault.
    var grouped_fault = (
        DataFrame(
            [
                Series("g", Column[Int32]([0, 0])),
                Series("bad", Column[UInt64]([UInt64.MAX, 0])),
                Series("sum", Column[Int64]([Int64.MAX, 1])),
            ]
        )
        .lazy()
        .group_by("g")
        .agg(
            [
                col("bad").cast("int32").count().alias("count"),
                col("sum").sum().alias("sum"),
            ]
        )
    )
    with assert_raises(contains="strict cast failed"):
        _ = grouped_fault.collect(accelerator=runtime)
    assert_frame(selected.collect(accelerator=runtime), actual)
