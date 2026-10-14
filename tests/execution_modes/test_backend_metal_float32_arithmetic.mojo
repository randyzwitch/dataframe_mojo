"""Binary32 arithmetic preserves gradual underflow and each operator rounding."""
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, Series, Column, Expr, col, lit, when
from dataframe.dtype import DataType
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import require_metal
from tests.execution_modes.test_backend_metal_float64_emulation import same


def fixture() raises -> DataFrame:
    var words: List[UInt32] = [
        0,
        1,
        2,
        3,
        0x003FFFFF,
        0x007FFFFE,
        0x007FFFFF,
        0x00800000,
        0x00800001,
        0x00800002,
        0x3DCCCCCD,
        0x3F000000,
        0x3F7FFFFF,
        0x3F800000,
        0x3F800001,
        0x40000000,
        0x40400000,
        0x7F7FFFFE,
        0x7F7FFFFF,
        0x7F800000,
        0x7F800042,
        0x7FC00042,
    ]
    var n = len(words)
    for i in range(n):
        words.append(words[i] | UInt32(0x80000000))
    var left = List[Float32]()
    var right = List[Float32]()
    var valid_left = List[Bool]()
    var valid_right = List[Bool]()
    var keys = List[Int32]()
    for a in words:
        for b in words:
            left.append(bitcast[DType.float32](a))
            right.append(bitcast[DType.float32](b))
            valid_left.append(len(left) % 7 != 0)
            valid_right.append(len(left) % 11 != 0)
            keys.append(Int32(len(left) % 5))
    # Adjacent encodings at every exponent exercise cancellation, overflow,
    # tie rounding and subnormal-to-normal transitions beyond literal anchors.
    var fractions: List[UInt32] = [0, 1, 0x003FFFFF, 0x007FFFFF]
    for exponent in range(255):
        for fraction in fractions:
            var a = UInt32(exponent) << 23 | fraction
            for b in List[UInt32]([a + 1, a ^ UInt32(0x80000000)]):
                left.append(bitcast[DType.float32](a))
                right.append(bitcast[DType.float32](b))
                valid_left.append(True)
                valid_right.append(True)
                keys.append(Int32(len(left) % 5))
    return DataFrame(
        [
            Series("a", Column[Float32](left^, valid_left^)),
            Series("b", Column[Float32](right^, valid_right^)),
            Series("k", Column[Int32](keys^)),
        ]
    )


def test_native_metal_float32_arithmetic_words_nulls_consumers_and_cost() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var frame = fixture()
    var expressions: List[Expr] = [
        (col("a") + col("b")).alias("add"),
        (col("a") - col("b")).alias("subtract"),
        (col("a") * col("b")).alias("multiply"),
        (col("a") / col("b")).alias("divide"),
    ]
    for expression in expressions:
        for sample in List[DataFrame](
            [
                frame.copy(),
                frame.slice(3, frame.height() - 7),
                frame.slice(0, 0),
            ]
        ):
            var query = sample.lazy().select(expression)
            same(
                query.collect(accelerator=runtime), query.collect(engine="cpu")
            )
        var query = frame.lazy().with_columns(expression.alias("result"))
        var filtered = query.filter(col("result").is_finite()).head(71)
        same(
            filtered.collect(accelerator=runtime),
            filtered.collect(engine="cpu"),
        )
        var ordered = query.sort("result").slice(5, 31)
        same(
            ordered.collect(accelerator=runtime), ordered.collect(engine="cpu")
        )
        var joined = query.join(
            DataFrame([Series("k", Column[Int32]([0, 1, 2, 3, 4]))]).lazy(), "k"
        ).head(83)
        same(joined.collect(accelerator=runtime), joined.collect(engine="cpu"))
        var measured = runtime.estimate_auto(frame.lazy().select(expression))
        assert_true(";mojo-metal-v1;ops=" in measured.family)
    var plain = frame.select(["a", "b"])
    var homogeneous = plain.lazy().select_exprs(expressions)
    same(
        homogeneous.collect(accelerator=runtime),
        homogeneous.collect(engine="cpu"),
    )
    same(
        homogeneous.collect(engine="auto", accelerator=runtime),
        homogeneous.collect(engine="cpu"),
    )
    var tiny = bitcast[DType.float32](UInt32(1))
    var anchors = (
        DataFrame(
            [
                Series(
                    "a", Column[Float32]([tiny, tiny, -tiny, tiny, tiny, -0.0])
                ),
                Series("b", Column[Float32]([tiny, -tiny, tiny, 2, 2, 1])),
            ]
        )
        .lazy()
        .select_exprs(expressions)
        .collect(accelerator=runtime)
    )
    assert_equal(
        bitcast[DType.uint32](anchors.column("add").float32()._get(0)),
        UInt32(2),
    )
    assert_equal(
        bitcast[DType.uint32](anchors.column("add").float32()._get(1)),
        UInt32(0),
    )
    assert_equal(
        bitcast[DType.uint32](anchors.column("subtract").float32()._get(2)),
        UInt32(0x80000002),
    )
    assert_equal(
        bitcast[DType.uint32](anchors.column("multiply").float32()._get(3)),
        UInt32(2),
    )
    assert_equal(
        bitcast[DType.uint32](anchors.column("divide").float32()._get(4)),
        UInt32(0),
    )
    assert_equal(
        bitcast[DType.uint32](anchors.column("divide").float32()._get(5)),
        UInt32(0x80000000),
    )
    with assert_raises(contains="budget"):
        _ = homogeneous.collect(accelerator=MetalRuntime(memory_limit_bytes=0))
    same(
        homogeneous.collect(accelerator=runtime),
        homogeneous.collect(engine="cpu"),
    )


def test_native_metal_float32_arithmetic_conditional_scalar_and_rounding_boundaries() raises:
    _ = require_metal()
    var runtime = MetalRuntime()
    var tiny = bitcast[DType.float32](UInt32(1))
    var frame = DataFrame(
        [
            Series("x", Column[Float32]([tiny, -tiny, 1, -1, 0, -0.0])),
            Series("p", Column[Bool]([True, False, True, False, True, False])),
            Series("k", Column[Int32]([0, 0, 0, 1, 1, 1])),
        ]
    )
    for predicate in List[Expr]([lit(False), lit(True), col("p")]):
        var query = frame.lazy().select(
            when(predicate)
            .then((col("x") * col("x")).sum())
            .otherwise(col("x"))
            .alias("value")
        )
        same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
        var window = (
            frame.lazy()
            .with_columns(
                when(predicate)
                .then((col("x") + col("x")).mean().over("k"))
                .otherwise(col("x").cast(DataType.FLOAT64))
                .alias("value")
            )
            .head(3)
        )
        same(window.collect(accelerator=runtime), window.collect(engine="cpu"))
    # Separate multiply/add must not contract into FMA, even when both are
    # evaluated by the emulated underflow path.
    var query = frame.lazy().select(
        ((col("x") * lit(Float32(1.0001))) + lit(tiny)).alias("value")
    )
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
