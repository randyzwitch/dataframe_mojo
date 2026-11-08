"""Native numeric programs retain CPU rounding, nulls and checked work."""
# contract-group: execution.backends
from std.memory import Pointer, bitcast
from max.gpu import global_idx
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
from dataframe_metal.accelerator.gpu.native_numeric import numeric_unary


def _numeric_bits_probe(
    left: Pointer[UInt64, MutUntrackedOrigin],
    right: Pointer[UInt64, MutUntrackedOrigin],
    operations: Pointer[Int32, MutUntrackedOrigin],
    tags: Pointer[UInt32, MutUntrackedOrigin],
    result: Pointer[UInt64, MutUntrackedOrigin],
    present: Pointer[UInt32, MutUntrackedOrigin],
    faults: Pointer[UInt32, MutUntrackedOrigin],
    rows: UInt32,
):
    var row = global_idx.x
    if row >= Int(rows):
        return
    var op = Int(operations[unsafe_offset=row])
    var tag = tags[unsafe_offset=row]
    var a = left[unsafe_offset=row]
    var b = right[unsafe_offset=row]
    var answer = (UInt64(0), True, UInt32(0))
    if numeric_unary(op):
        if tag == 2:
            answer = (
                UInt64(float32_unary_bits(UInt32(a), op)),
                True,
                UInt32(0),
            )
        else:
            var value = integer_unary_bits(a, tag, op)
            answer = (value[0], True, UInt32(value[1]))
    else:
        answer = native_numeric_binary(a, b, tag, op)
    result[unsafe_offset=row] = answer[0]
    present[unsafe_offset=row] = UInt32(answer[1])
    faults[unsafe_offset=row] = answer[2]


# requires: metal
def test_metal_numeric_device_bits_match_independently_checked_host_primitives() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.accelerator.gpu.query_scope import Scope

    _ = require_metal()
    var left = List[UInt64]()
    var right = List[UInt64]()
    var ops = List[Int32]()
    var tags = List[UInt32]()
    var expected = List[UInt64]()
    var validity = List[UInt32]()
    var faults = List[UInt32]()
    var state = UInt64(41378391)
    for op in [
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
    ]:
        for tag in range(3):
            if op == DIV and tag != 2:
                continue
            for i in range(512):
                state = state * 6364136223846793005 + 1442695040888963407
                var a = state
                state = state * 6364136223846793005 + 1442695040888963407
                var b = state
                if tag == 0:
                    a = bitcast[DType.uint64](
                        Int64(bitcast[DType.int32](UInt32(a)))
                    )
                    b = bitcast[DType.uint64](
                        Int64(bitcast[DType.int32](UInt32(b)))
                    )
                elif tag == 2:
                    a = UInt64(UInt32(a))
                    b = UInt64(UInt32(b))
                var answer = (UInt64(0), True, UInt32(0))
                if numeric_unary(op):
                    if tag == 2:
                        answer = (
                            UInt64(float32_unary_bits(UInt32(a), op)),
                            True,
                            UInt32(0),
                        )
                    else:
                        var value = integer_unary_bits(a, UInt32(tag), op)
                        answer = (value[0], True, UInt32(value[1]))
                else:
                    answer = native_numeric_binary(a, b, UInt32(tag), op)
                left.append(a)
                right.append(b)
                ops.append(Int32(op))
                tags.append(UInt32(tag))
                expected.append(answer[0])
                validity.append(UInt32(answer[1]))
                faults.append(answer[2])
    var n = len(left)
    var scope = Scope()
    var ctx = scope.context()
    var a = ctx.enqueue_create_buffer[DType.uint64](n)
    var b = ctx.enqueue_create_buffer[DType.uint64](n)
    var operations = ctx.enqueue_create_buffer[DType.int32](n)
    var types = ctx.enqueue_create_buffer[DType.uint32](n)
    var output = ctx.enqueue_create_buffer[DType.uint64](n)
    var present = ctx.enqueue_create_buffer[DType.uint32](n)
    var flags = ctx.enqueue_create_buffer[DType.uint32](n)
    ctx.enqueue_copy(a, left.unsafe_ptr())
    ctx.enqueue_copy(b, right.unsafe_ptr())
    ctx.enqueue_copy(operations, ops.unsafe_ptr())
    ctx.enqueue_copy(types, tags.unsafe_ptr())
    var kernel = ctx.compile_function[_numeric_bits_probe]()
    ctx.enqueue_function(
        kernel,
        a,
        b,
        operations,
        types,
        output,
        present,
        flags,
        UInt32(n),
        grid_dim=(n + 255) // 256,
        block_dim=256,
    )
    var actual = List[UInt64](length=n, fill=0)
    var actual_validity = List[UInt32](length=n, fill=0)
    var actual_faults = List[UInt32](length=n, fill=0)
    ctx.enqueue_copy(actual.unsafe_ptr(), output)
    ctx.enqueue_copy(actual_validity.unsafe_ptr(), present)
    ctx.enqueue_copy(actual_faults.unsafe_ptr(), flags)
    scope.synchronize()
    assert_equal(actual, expected)
    assert_equal(actual_validity, validity)
    assert_equal(actual_faults, faults)


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
            Series._from_chunks(
                name, source.dtype(), [source.slice(0, 2), source.slice(2, 4)]
            )
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
        var grouped = frame.lazy().group_by("key").agg(outputs)
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
