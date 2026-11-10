"""Native numeric programs retain CPU rounding, nulls and checked work."""
# contract-group: execution.backends
from std.memory import bitcast
from max.gpu import global_idx

# Load the host kernel-launch overloads in this module's import scope.
from max.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, DataType, Expr, Series, col, when
from dataframe.expr import (
    NEG,
    ABS,
    SIGN,
    FLOOR,
    CEIL,
    TRUNCATE,
    ROUND,
    CLIP_LOW,
    CLIP_HIGH,
    DIV,
    FLOORDIV,
    MOD,
)
from dataframe_metal.accelerator.gpu.native_numeric import (
    float32_unary_bits,
    integer_unary_bits,
    float32_clip_bits,
    integer_divmod_bits,
    float32_divide_bits,
    native_numeric_binary,
)
from dataframe_metal.typed_expression_plan import compile_typed_expression
from dataframe_metal.composition_plan import plan_composition, composition_key


def test_metal_native_integer_numeric_boundaries_and_floor_remainders() raises:
    for tag in range(2):
        var smallest = Int64(Int32.MIN) if tag == 0 else Int64.MIN
        var largest = Int64(Int32.MAX) if tag == 0 else Int64.MAX
        for a in [
            smallest,
            smallest + 1,
            Int64(-9),
            Int64(-1),
            Int64(0),
            Int64(1),
            Int64(9),
            largest,
        ]:
            for b in [
                smallest,
                Int64(-7),
                Int64(-1),
                Int64(0),
                Int64(1),
                Int64(7),
                largest,
            ]:
                for op in [FLOORDIV, MOD]:
                    var result = integer_divmod_bits(a, b, UInt32(tag), op)
                    assert_equal(result[1], b != 0)
                    var overflow = op == FLOORDIV and a == smallest and b == -1
                    assert_equal(result[2], overflow)
                    if b != 0 and not overflow:
                        var expected = a // b if op == FLOORDIV else (
                            Int64(0) if b == -1 else a % b
                        )
                        assert_equal(bitcast[DType.int64](result[0]), expected)
            for op in [NEG, ABS]:
                var result = integer_unary_bits(
                    bitcast[DType.uint64](a), UInt32(tag), op
                )
                assert_equal(result[1], a == smallest)
                if a != smallest:
                    assert_equal(
                        bitcast[DType.int64](result[0]),
                        -a if op == NEG or a < 0 else a,
                    )


def test_metal_native_float_numeric_bits_keep_ieee_and_cpu_remainder() raises:
    assert_equal(float32_unary_bits(0x80000000, ROUND), UInt32(0))
    assert_equal(float32_unary_bits(0x80000001, FLOOR), UInt32(0xBF800000))
    assert_equal(float32_unary_bits(1, CEIL), UInt32(0x3F800000))
    assert_equal(float32_unary_bits(0x80000001, SIGN), UInt32(0xBF800000))
    assert_equal(float32_unary_bits(0xBFC00000, ROUND), UInt32(0xC0000000))
    assert_equal(float32_clip_bits(0x80000000, 0, CLIP_LOW), UInt32(0x80000000))
    assert_equal(
        float32_clip_bits(0x7FC12345, 0, CLIP_HIGH), UInt32(0x7FC12345)
    )
    assert_equal(float32_clip_bits(1, 0, CLIP_HIGH), UInt32(0))
    var zero_divisors: List[UInt32] = [
        1,
        0x80000001,
        0x007FFFFF,
        0x00800000,
        0x3F800000,
        0xBF800000,
        0x40000000,
        0xC0000000,
        0x7F7FFFFF,
        0xFF7FFFFF,
    ]
    for zero in [UInt32(0), UInt32(0x80000000)]:
        for divisor in zero_divisors:
            var actual = native_numeric_binary(
                UInt64(zero), UInt64(divisor), 2, MOD
            )
            var expected = bitcast[DType.uint32](
                bitcast[DType.float32](zero) % bitcast[DType.float32](divisor)
            )
            assert_equal(actual[0], UInt64(expected))
            assert_true(actual[1])
            assert_equal(actual[2], UInt32(0))
    var state = UInt64(9325421)
    for i in range(8192):
        state = state * 6364136223846793005 + 1442695040888963407
        var a = UInt32(state)
        var b = UInt32(state >> 32)
        var x = bitcast[DType.float32](a)
        var y = bitcast[DType.float32](b)
        var division = float32_divide_bits(a, b)
        if not division[1]:
            assert_equal(division[0], bitcast[DType.uint32](x / y))
        var remainder = native_numeric_binary(UInt64(a), UInt64(b), 2, MOD)
        if remainder[2] == 0:
            assert_equal(UInt32(remainder[0]), bitcast[DType.uint32](x % y))
    assert_true(float32_divide_bits(0, 0)[1])
    assert_true(float32_divide_bits(0x7F800000, 0x3F800000)[1])
    assert_equal(native_numeric_binary(1, 0x3F800000, 2, MOD)[2], UInt32(2))


def test_metal_numeric_planner_composes_native_types_and_separates_costs() raises:
    var frame = DataFrame(
        [
            Series("a", Column[Int32]([-5, 0, 7])),
            Series("b", Column[Int32]([2, 0, -3])),
            Series("f", Column[Float32]([-1.5, 0, 2.5])),
            Series("g", Column[Float32]([2, 0, -2])),
        ]
    )
    var values = List[Expr](
        [
            -col("a"),
            col("a").abs(),
            col("a").sign(),
            col("a").clip(-2, 3),
            col("a") // col("b"),
            col("a") % col("b"),
            col("f").floor(),
            col("f").ceil(),
            col("f").truncate(),
            col("f").round(),
            col("f") / col("g"),
            col("f") // col("g"),
            col("f") % col("g"),
            (-col("a").cast(DataType.FLOAT32)).clip_min(-4),
        ]
    )
    var keys = List[String]()
    for value in values:
        assert_true(Bool(compile_typed_expression(value, frame)))
        var matched = plan_composition(
            frame.lazy().select_exprs(
                [
                    value.sum().alias("s"),
                    value.count().alias("n"),
                    value.mean().alias("avg"),
                    value.min().alias("lo"),
                    value.max().alias("hi"),
                ]
            )
        )
        assert_true(Bool(matched))
        assert_equal(len(matched.value().fields), 1)
        var key = composition_key(matched.value())
        assert_false(key in keys)
        keys.append(key^)
    assert_false(Bool(compile_typed_expression(col("a") / col("b"), frame)))
    assert_false(Bool(compile_typed_expression(col("f").round(2), frame)))
    assert_false(Bool(compile_typed_expression(col("f").sqrt(), frame)))


def test_metal_group_calibration_separates_requested_output_order() raises:
    from dataframe_metal.calibration import shape_key
    from dataframe_metal.group_plan import plan_grouped, grouped_key

    var frame = DataFrame(
        [
            Series("key", Column[Int32]([2, 1, 2])),
            Series("x", Column[Int64]([-3, 4, 5])),
        ]
    )
    var unordered = frame.lazy().group_by("key").agg(col("x").abs().sum())
    var ordered = (
        frame.lazy()
        .group_by("key", maintain_order=True)
        .agg(col("x").abs().sum())
    )
    var plain = plan_grouped(unordered)
    var stable = plan_grouped(ordered)
    assert_true(Bool(plain))
    assert_true(Bool(stable))
    assert_equal(grouped_key(plain.value()), grouped_key(stable.value()))
    var plain_key = shape_key(unordered, grouped_key(plain.value()))
    var stable_key = shape_key(ordered, grouped_key(stable.value()))
    assert_true(":group_order=0" in plain_key)
    assert_true(":group_order=1" in stable_key)
    assert_false(plain_key == stable_key)


# requires: metal
def test_metal_numeric_public_nullable_chunks_groups_and_shared_states() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series(
                "a",
                Column[Int32](
                    [99, -9, -5, 0, 4, 7, 9, 99],
                    [True, True, False, True, True, True, True, True],
                ),
            ),
            Series("b", Column[Int32]([99, 2, -2, 0, -3, 2, 0, 99])),
            Series("f", Column[Float32]([99, -1.5, -0.25, 0, 0.5, 2.5, 4, 99])),
            Series("g", Column[Float32]([99, 2, -2, 1, 2, -2, 4, 99])),
            Series(
                "key",
                Column[Int64](
                    [99, 1, 1, 2, 2, 1, 2, 99],
                    [True, True, False, True, True, True, True, True],
                ),
            ),
        ]
    ).slice(1, 6)
    var chunked = List[Series]()
    for name in frame.columns():
        var source = frame.column(name)
        chunked.append(
            Series._from_chunks([source.slice(0, 2), source.slice(2, 4)])
        )
    frame = DataFrame(chunked^)
    var values = List[Expr](
        [
            col("a").abs().clip(-4, 8),
            -(col("a") // col("b")),
            col("a") % col("b"),
            col("f").round(),
            col("f").floor(),
            col("f").ceil(),
            col("f").truncate(),
            col("f").sign(),
            col("f") / col("g"),
            col("f") // col("g"),
            col("f") % col("g"),
            col("a").cast(DataType.FLOAT32).abs() / col("g"),
        ]
    )
    for value in values:
        var outputs = List[Expr](
            [
                value.sum().alias("sum"),
                value.count().alias("count"),
                value.mean().alias("mean"),
                value.min().alias("min"),
                value.max().alias("max"),
            ]
        )
        var query = frame.lazy().select_exprs(outputs)
        var expected = query.collect(engine="cpu")
        var direct = execute_timed(query)
        assert_true(direct[0].equals(expected))
        assert_true(query.collect(accelerator=runtime).equals(expected))
        var grouped = (
            frame.lazy().group_by("key", maintain_order=True).agg(outputs)
        )
        var grouped_expected = grouped.collect(engine="cpu")
        assert_true(execute_timed(grouped)[0].equals(grouped_expected))
        assert_true(
            grouped.collect(accelerator=runtime).equals(grouped_expected)
        )


# requires: metal
def test_metal_numeric_case_lazy_faults_and_cpu_overflow_errors() raises:
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series("a", Column[Int64]([Int64.MIN, -9, 5, 7])),
            Series("b", Column[Int64]([-1, 2, 0, -3])),
        ]
    )
    var selected = (
        when(col("a") > 0).then(col("a").abs() % col("b")).otherwise(0)
    )
    var query = frame.lazy().select_exprs(
        [selected.sum().alias("s"), selected.count().alias("n")]
    )
    assert_true(execute_timed(query)[0].equals(query.collect(engine="cpu")))
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )
    same_outcome(frame.lazy().select(col("a").abs().sum()))
    same_outcome(frame.lazy().select((col("a") // col("b")).count()))


def test_metal_typed_scalar_chain_keeps_nested_right_trees_on_general_path() raises:
    from dataframe import lit, null
    from dataframe_metal.expression_format import TYPED_CHAIN

    var frame = DataFrame(
        [
            Series("n", Column[Int32]([-7, 0, 4])),
            Series("w", Column[Int64]([-9, 0, 12])),
            Series("f", Column[Float32]([-1, 0, 3])),
            Series("g", Column[Float32]([2, 4, 8])),
        ]
    )
    var chains = List[Expr](
        [
            col("n").abs().clip(0, 12),
            col("w") // lit(Int64(3)),
            col("f").floor(),
            (-col("n")).cast(DataType.FLOAT32) / col("g"),
            col("f").fill_nan(null(DataType.FLOAT32)).fill_null(col("g")),
            col("w").fill_null(lit(Int64(7))),
            null(DataType.INT64).fill_null(col("w")),
        ]
    )
    var steps = [3, 1, 1, 3, 2, 1, 1]
    var unary = [1, 0, 1, 3, 0, 0, 0]
    for i in range(len(chains)):
        var value = chains[i].copy()
        var compiled = compile_typed_expression(value, frame)
        assert_true(Bool(compiled))
        assert_true(
            Bool(compiled.value().nodes[0].literal & UInt64(TYPED_CHAIN))
        )
        var schedule = compiled.value().nodes[0].literal
        assert_equal(Int((schedule >> 8) & 255), steps[i])
        assert_equal(Int(schedule >> 16), unary[i])
        var query = frame.lazy().select_exprs(
            [
                value.sum().alias("s"),
                value.count().alias("n"),
                value.mean().alias("avg"),
                value.min().alias("lo"),
                value.max().alias("hi"),
            ]
        )
        assert_true(Bool(plan_composition(query)))
    # The right expression can fault independently, even when the left is
    # null. Preserve its eager evaluation with the general interpreter.
    var trees = List[Expr](
        [
            col("n").abs() + col("n").abs(),
            col("w") // (col("w") + lit(Int64(1))),
            col("f").fill_null(col("g").abs()),
            (-col("n")).cast(DataType.FLOAT32) / col("g") * col("f").abs(),
        ]
    )
    for value in trees:
        var compiled = compile_typed_expression(value, frame)
        assert_true(Bool(compiled))
        assert_false(
            Bool(compiled.value().nodes[0].literal & UInt64(TYPED_CHAIN))
        )

    var longer = compile_typed_expression(
        col("n").abs().clip(0, 12).sign().cast(DataType.INT64), frame
    )
    assert_true(Bool(longer))
    assert_equal(Int((longer.value().nodes[0].literal >> 8) & 255), 5)
    assert_equal(Int(longer.value().nodes[0].literal >> 16), 25)
