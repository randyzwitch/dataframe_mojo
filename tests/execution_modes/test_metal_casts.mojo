"""Native casts preserve CPU binding, checked intermediates and null rules."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_false
from std.memory import bitcast
from dataframe import Column, DataFrame, DataType, Series, Expr, col, lit
from dataframe_metal.typed_expression_plan import compile_typed_expression
from dataframe_metal.composition_plan import plan_composition, composition_key
from dataframe_metal.accelerator.gpu.native_cast import (
    integer_float32_bits,
    float32_integer_bits,
)


def test_metal_native_cast_rounding_and_float_integer_boundaries() raises:
    var state = UInt64(914318523)
    for i in range(4096):
        state = state * 6364136223846793005 + 1442695040888963407
        var value = bitcast[DType.int64](state)
        assert_equal(
            integer_float32_bits(value), bitcast[DType.uint32](Float32(value))
        )
        var raw = UInt32(state)
        var floating = bitcast[DType.float32](raw)
        var wide = Float64(floating)
        for narrow in [False, True]:
            var valid = (
                wide >= Float64(Int32.MIN)
                and wide
                <= Float64(Int32.MAX) if narrow else wide
                >= -Float64(9223372036854775808.0)
                and wide < Float64(9223372036854775808.0)
            )
            var answer = float32_integer_bits(raw, narrow)
            assert_equal(answer[1], not valid)
            if valid:
                assert_equal(bitcast[DType.int64](answer[0]), Int64(wide))
    for value in [
        Int64.MIN,
        Int64.MAX,
        Int64(Int32.MIN),
        Int64(Int32.MAX),
        Int64(0),
        Int64(16777217),
        Int64(-16777217),
        Int64(16777219),
        Int64(-16777219),
    ]:
        assert_equal(
            integer_float32_bits(value), bitcast[DType.uint32](Float32(value))
        )
    for raw in [
        UInt32(0),
        UInt32(0x80000000),
        UInt32(1),
        UInt32(0x80000001),
        UInt32(0x7F800000),
        UInt32(0xFF800000),
        UInt32(0x7FC12345),
        UInt32(0xFFC12345),
        UInt32(0x4F000000),
        UInt32(0xCF000000),
        UInt32(0x5F000000),
        UInt32(0xDF000000),
    ]:
        var invalid = ((raw >> 23) & 255) == 255
        for narrow in [False, True]:
            var answer = float32_integer_bits(raw, narrow)
            if invalid:
                assert_true(answer[1])
            if (raw & 0x7FFFFFFF) <= 1:
                assert_false(answer[1])
                assert_equal(answer[0], UInt64(0))
    assert_true(float32_integer_bits(0x4F000000, True)[1])
    assert_equal(
        bitcast[DType.int64](float32_integer_bits(0xCF000000, True)[0]),
        Int64(Int32.MIN),
    )
    assert_true(float32_integer_bits(0x5F000000, False)[1])
    assert_equal(
        bitcast[DType.int64](float32_integer_bits(0xDF000000, False)[0]),
        Int64.MIN,
    )


def test_metal_native_cast_planner_shares_values_and_preserves_cpu_types() raises:
    var frame = DataFrame(
        [
            Series("n", Column[Int32]([1, 2])),
            Series("w", Column[Int64]([3, 4])),
            Series("f", Column[Float32]([5, 6])),
            Series("d", Column[Float64]([7, 8])),
        ]
    )
    var value = col("n").cast(DataType.INT64) * col("w")
    var plan = plan_composition(
        frame.lazy().select_exprs(
            [
                value.sum().alias("sum"),
                value.count().alias("count"),
                value.mean().alias("mean"),
            ]
        )
    )
    assert_true(Bool(plan))
    assert_equal(len(plan.value().fields), 1)
    assert_equal(plan.value().fields[0].dtype, DataType.INT64)
    assert_equal(len(plan.value().inputs), 2)
    from dataframe_metal.composition_plan import (
        has_typed_values,
        has_conditions,
        needs_general_values,
        simple_conditional_trees,
    )
    from dataframe_metal.group_general_reduce import general_case_mode
    from dataframe_metal.region_plan import normalize_region

    assert_true(has_typed_values(plan.value()))
    assert_false(has_conditions(plan.value()))
    assert_true(needs_general_values(plan.value()))
    assert_false(simple_conditional_trees(plan.value()))
    assert_equal(general_case_mode(plan.value()), 1)
    assert_true(
        composition_key(plan.value()).startswith("typed_reductions:v1:")
    )
    var stages = (
        frame.lazy()
        .with_columns([col("n").cast(DataType.INT64).alias("wide")])
        .select((col("wide") * col("w")).sum())
    )
    var normalized = normalize_region(stages)
    assert_true(Bool(normalized))
    assert_true(Bool(plan_composition(normalized.value())))
    var f32 = compile_typed_expression(
        col("w").cast(DataType.FLOAT32) * col("f"), frame
    )
    assert_true(Bool(f32))
    assert_equal(f32.value().dtype, DataType.FLOAT32)
    assert_equal(f32.value().nodes[0].literal, UInt64(0))
    assert_true(
        Bool(
            compile_typed_expression(
                col("f").cast(DataType.INT32, strict=False), frame
            )
        )
    )
    assert_equal(
        compile_typed_expression(col("f").cast(DataType.INT32), frame)
        .value()
        .nodes[0]
        .literal,
        UInt64(1),
    )
    assert_false(Bool(compile_typed_expression(col("n") + col("w"), frame)))
    assert_false(
        Bool(compile_typed_expression(col("d").cast(DataType.INT32), frame))
    )
    assert_false(
        Bool(compile_typed_expression(col("n").cast(DataType.FLOAT64), frame))
    )
    assert_false(
        Bool(
            compile_typed_expression(
                col("n").cast(DataType.INT64) / col("w"), frame
            )
        )
    )
    var strict = plan_composition(
        frame.lazy().select(col("w").cast(DataType.INT32).sum())
    )
    var relaxed = plan_composition(
        frame.lazy().select(col("w").cast(DataType.INT32, strict=False).sum())
    )
    assert_true(Bool(strict) and Bool(relaxed))
    assert_false(
        composition_key(strict.value()) == composition_key(relaxed.value())
    )
    var malformed = value.copy()
    malformed._nodes[3].left = 0
    assert_false(Bool(compile_typed_expression(malformed, frame)))


def test_metal_typed_value_program_preserves_faults_nulls_and_native_widths() raises:
    from std.memory import Pointer
    from dataframe.expr import COL, LIT_INT, CAST, ADD, MUL
    from dataframe_metal.expression_format import TYPED_HEADER
    from dataframe_metal.accelerator.gpu.expression_value import (
        expression_value,
    )
    from dataframe_metal.accelerator.gpu.multi_reduce import Bytes

    var n = List[Int32]([Int32(-2), 3, 0, Int32.MAX])
    var w = List[Int64]([Int64(5), Int64.MAX, 8, 1])
    var f = List[Float32](
        [
            Float32(2.75),
            bitcast[DType.float32](UInt32(0x7F800000)),
            bitcast[DType.float32](UInt32(0xDF000000)),
            bitcast[DType.float32](UInt32(1)),
        ]
    )
    var nb = List[UInt8]([UInt8(0b01011000)])
    var wb = List[UInt8]([UInt8(0b00111000)])
    var fb = List[UInt8]([UInt8(255)])
    var nv = Bytes(unsafe_from_address=Int(n.unsafe_ptr()))
    var wv = Bytes(unsafe_from_address=Int(w.unsafe_ptr()))
    var fv = Bytes(unsafe_from_address=Int(f.unsafe_ptr()))
    var np = Bytes(unsafe_from_address=Int(nb.unsafe_ptr()))
    var wp = Bytes(unsafe_from_address=Int(wb.unsafe_ptr()))
    var fp = Bytes(unsafe_from_address=Int(fb.unsafe_ptr()))
    for family in range(6):
        var words = List[UInt64]([UInt64(TYPED_HEADER), 0, 0, 0])
        if family == 0 or family == 1:
            words.extend(
                [UInt64(COL), 0, (UInt64(1) << 40) | (UInt64(1) << 32) | 3, 0]
            )
            if family == 1:
                words.extend([UInt64(LIT_INT), 0, UInt64(1) << 40, 1])
                words.extend([UInt64(ADD), 0, UInt64(1) << 40, 0])
            words.extend([UInt64(CAST), 0, UInt64(2) << 40, 1])
            words.extend(
                [UInt64(COL), 1, (UInt64(2) << 40) | (UInt64(1) << 32) | 3, 0]
            )
            words.extend([UInt64(MUL), 0, UInt64(2) << 40, 0])
        elif family == 2 or family == 3:
            words.extend(
                [UInt64(COL), 1, (UInt64(2) << 40) | (UInt64(1) << 32) | 3, 0]
            )
            words.extend(
                [
                    UInt64(CAST),
                    0,
                    UInt64(1) << 40,
                    UInt64(257 if family == 2 else 256),
                ]
            )
        elif family == 4:
            words[3] = 1
            words.extend([UInt64(COL), 2, UInt64(3) << 40, 0])
            words.extend([UInt64(CAST), 0, UInt64(2) << 40, 513])
        else:
            words.extend(
                [UInt64(COL), 1, (UInt64(2) << 40) | (UInt64(1) << 32) | 3, 0]
            )
            words.extend([UInt64(CAST), 0, UInt64(3) << 40, 257])
        var program = Pointer[UInt64, UntrackedOrigin[mut=False]](
            unsafe_from_address=Int(words.unsafe_ptr())
        )
        for row in range(4):
            var answer = expression_value[False, True](
                program,
                len(words) // 4,
                0,
                row,
                UInt32(1),
                False,
                nv,
                wv,
                fv,
                nv,
                nv,
                nv,
                nv,
                nv,
                np,
                wp,
                fp,
                np,
                np,
                np,
                np,
                np,
                nv,
                np,
            )
            if family == 0 or family == 1:
                if row == 0:
                    assert_true(answer[1])
                    assert_equal(
                        bitcast[DType.int64](answer[0]),
                        Int64(-10 if family == 0 else -5),
                    )
                    assert_equal(answer[2], UInt32(0))
                if row == 1:
                    assert_equal(answer[2], UInt32(1))
                if row == 2:
                    assert_false(answer[1])
                if row == 3:
                    assert_false(answer[1])
                    assert_equal(answer[2], UInt32(1 if family == 1 else 0))
            elif family == 2 or family == 3:
                if row == 1:
                    assert_false(answer[1])
                    assert_equal(answer[2], UInt32(1 if family == 2 else 0))
                elif row == 0:
                    assert_true(answer[1])
                    assert_equal(bitcast[DType.int64](answer[0]), Int64(5))
                elif row == 3:
                    assert_false(answer[1])
            elif family == 4:
                if row == 1:
                    assert_false(answer[1])
                    assert_equal(answer[2], UInt32(1))
                else:
                    assert_true(answer[1])
                    assert_equal(answer[2], UInt32(0))
                    assert_equal(
                        bitcast[DType.int64](answer[0]),
                        Int64(2) if row
                        == 0 else Int64.MIN if row
                        == 2 else Int64(0),
                    )
            elif row < 3:
                assert_true(answer[1])
                assert_equal(
                    UInt32(answer[0]), bitcast[DType.uint32](Float32(w[row]))
                )
                assert_equal(answer[2], UInt32(0))
            else:
                assert_false(answer[1])
        _ = words^
    var words = List[UInt64](
        [
            UInt64(TYPED_HEADER),
            0,
            0,
            0,
            UInt64(COL),
            2,
            UInt64(3) << 40,
            0,
            UInt64(LIT_INT),
            0,
            UInt64(3) << 40,
            UInt64(0x3F000000),
            UInt64(MUL),
            0,
            UInt64(3) << 40,
            0,
            UInt64(CAST),
            0,
            UInt64(3) << 40,
            513,
        ]
    )
    var program = Pointer[UInt64, UntrackedOrigin[mut=False]](
        unsafe_from_address=Int(words.unsafe_ptr())
    )
    for wanted in [False, True]:
        var answer = expression_value[True, True](
            program,
            5,
            0,
            3,
            UInt32(2),
            wanted,
            nv,
            wv,
            fv,
            nv,
            nv,
            nv,
            nv,
            nv,
            np,
            wp,
            fp,
            np,
            np,
            np,
            np,
            np,
            nv,
            np,
        )
        assert_true(answer[1])
        assert_equal(answer[2], UInt32(2 if wanted else 0))
    _ = words^
    _ = (n^, w^, f^, nb^, wb^, fb^)


# requires: metal
def test_metal_native_casts_all_nine_pairs_nullable_slices_and_groups() raises:
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    for source in range(3):
        var value = Series(
            "v",
            Column[Int32](
                [99, -2, 0, 3, 8, 99], [True, True, False, True, True, True]
            ),
        )
        if source == 1:
            value = Series(
                "v",
                Column[Int64](
                    [99, -2, 0, 3, 8, 99], [True, True, False, True, True, True]
                ),
            )
        elif source == 2:
            value = Series(
                "v",
                Column[Float32](
                    [99, -2.75, 0, 3.25, 8, 99],
                    [True, True, False, True, True, True],
                ),
            )
        var frame = DataFrame(
            [value^, Series("key", Column[Int32]([99, 2, 1, 2, 1, 99]))]
        ).slice(1, 4)
        for target in [DataType.INT32, DataType.INT64, DataType.FLOAT32]:
            var cast = col("v").cast(target)
            for grouped in [False, True]:
                var outputs = List[Expr](
                    [
                        cast.sum().alias("sum"),
                        cast.count().alias("count"),
                        cast.mean().alias("mean"),
                        cast.min().alias("min"),
                        cast.max().alias("max"),
                    ]
                )
                var query = frame.lazy().select_exprs(outputs)
                if grouped:
                    query = (
                        frame.lazy()
                        .group_by("key", maintain_order=True)
                        .agg(outputs)
                    )
                var cpu = query.collect(engine="cpu")
                for copied in [False, True]:
                    assert_true(
                        execute_timed(query, zero_copy=not copied)[0].equals(
                            cpu
                        )
                    )
                same_outcome(query)


# requires: metal
def test_metal_explicit_mixed_source_arithmetic_and_shared_cast_states() raises:
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series(
                "n",
                Column[Int32]([99, 2, 3, 4, 5], [True, True, True, True, True]),
            ),
            Series(
                "w",
                Column[Int64]([99, 5, 6, 7, 8], [True, True, True, True, True]),
            ),
            Series(
                "f",
                Column[Float32](
                    [99, 0.5, 1.5, 2.5, 3.5], [True, True, True, True, True]
                ),
            ),
        ]
    ).slice(1, 4)
    var columns = List[Series]()
    var names = List[String](["n", "w", "f"])
    for i in range(3):
        var split = i + 1
        columns.append(
            Series._from_chunks(
                [
                    frame.column(names[i]).slice(0, split),
                    frame.column(names[i]).slice(split, 4 - split),
                ]
            )
        )
    var chunked = DataFrame(columns^)
    for selected in [frame^, chunked^]:
        var integer = col("n").cast(DataType.INT64) * col("w")
        var floating = col("w").cast(DataType.FLOAT32) * col("f")
        var query = selected.lazy().select_exprs(
            [
                integer.sum().alias("integer"),
                integer.mean().alias("mean"),
                floating.sum().alias("floating"),
                floating.max().alias("max"),
            ]
        )
        for copied in [False, True]:
            var answer = execute_timed(query, zero_copy=not copied)[0].copy()
            assert_equal(answer.item(0, "integer").int64(), Int64(96))
            assert_equal(answer.item(0, "mean").float64(), Float64(24))
            assert_equal(answer.item(0, "floating").float32(), Float32(57))
            assert_equal(answer.item(0, "max").float32(), Float32(28))
            same_outcome(query)


# requires: metal
def test_metal_cast_faults_and_non_strict_nulls_keep_cpu_outcomes() raises:
    from std.testing import assert_raises
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series("w", Column[Int64]([Int64(Int32.MAX) + 1, 2])),
            Series("n", Column[Int32]([Int32.MAX, 1])),
            Series(
                "f",
                Column[Float32](
                    [bitcast[DType.float32](UInt32(0x7F800000)), 3.75]
                ),
            ),
            Series("nullable", Column[Int64]([0, 4], [False, True])),
            Series("p", Column[Int32]([0, 1])),
        ]
    )
    for name in ["w", "f"]:
        var strict = frame.lazy().select(col(name).cast(DataType.INT32).count())
        with assert_raises():
            _ = execute_timed(strict)
        same_outcome(strict)
        var relaxed = frame.lazy().select_exprs(
            [
                col(name).cast(DataType.INT32, strict=False).sum().alias("sum"),
                col(name)
                .cast(DataType.INT32, strict=False)
                .count()
                .alias("count"),
            ]
        )
        assert_equal(
            execute_timed(relaxed)[0].item(0, "count").int64(), Int64(1)
        )
        same_outcome(relaxed)
        var value = col(name).cast(DataType.INT32, strict=False)
        var grouped = (
            frame.lazy()
            .group_by("p", maintain_order=True)
            .agg(
                [
                    value.sum().alias("sum"),
                    value.count().alias("count"),
                    value.mean().alias("mean"),
                    value.min().alias("min"),
                    value.max().alias("max"),
                ]
            )
        )
        var groups = execute_timed(grouped)[0].copy()
        assert_equal(groups.height(), 2)
        assert_equal(groups.item(0, "count").int64(), Int64(0))
        assert_equal(groups.item(0, "sum").int32(), Int32(0))
        for output in ["mean", "min", "max"]:
            assert_true(groups.item(0, output).is_null())
        assert_equal(groups.item(1, "count").int64(), Int64(1))
        same_outcome(grouped)
        var safe = (
            frame.lazy()
            .filter(col("p") > 0)
            .select(col(name).cast(DataType.INT32).sum())
        )
        assert_equal(
            execute_timed(safe)[0].item().int32(),
            Int32(2 if name == "w" else 3),
        )
    var overflow = (
        (col("n") + 1).cast(DataType.INT64) * col("nullable")
    ).count()
    var query = frame.lazy().select(overflow)
    with assert_raises(contains="overflow"):
        _ = execute_timed(query)
    same_outcome(query)


# requires: metal
def test_metal_float_cast_count_omits_unneeded_float_work() raises:
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series(
                "f",
                Column[Float32](
                    [bitcast[DType.float32](UInt32(1)), Float32(1)]
                ),
            )
        ]
    )
    var query = frame.lazy().select(
        (col("f") * 0.5).cast(DataType.FLOAT32).count()
    )
    assert_equal(execute_timed(query)[0].item().int64(), Int64(2))
    same_outcome(query)
