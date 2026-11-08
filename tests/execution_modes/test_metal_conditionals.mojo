"""CASE programs preserve typed branches and forward-only control flow."""
# contract-group: execution.backends
from std.testing import assert_true, assert_false, assert_equal
from dataframe import Column, DataFrame, DataType, Expr, Series, col, lit, when
from dataframe_metal.conditional_plan import (
    compile_conditional,
    CASE_HEADER,
    CASE_TEST,
    CASE_JUMP,
    CASE_NULL,
    MAX_CASE_INSTRUCTIONS,
)


def test_metal_conditional_program_branches_and_mixed_predicate_types() raises:
    var frame = DataFrame(
        [
            Series("p", Column[Int32]([1, 0])),
            Series("v", Column[Int64]([3, 4])),
            Series("w", Column[Int64]([2, 1])),
        ]
    )
    var value = when(col("p") > 0).then(col("v") * col("w")).otherwise(
        col("v") - col("w")
    ) + col("v")
    var matched = compile_conditional(value, frame)
    assert_true(Bool(matched))
    ref plan = matched.value()
    assert_equal(plan.dtype, DataType.INT64)
    assert_equal(len(plan.conditions), 1)
    assert_equal(plan.nodes[0].op, CASE_HEADER)
    assert_equal(plan.nodes[1].op, CASE_TEST)
    assert_equal(plan.nodes[1].input, 0)
    assert_equal(plan.nodes[1].literal, UInt64(6))
    assert_equal(plan.nodes[5].op, CASE_JUMP)
    assert_equal(plan.nodes[5].literal, UInt64(9))
    assert_equal(len(plan.nodes), 11)
    assert_equal(
        plan.conditions[0].clauses[0].bounds.filter_dtype, DataType.INT32
    )
    assert_true(len(plan.nodes) <= MAX_CASE_INSTRUCTIONS)


def test_metal_conditional_program_chains_null_else_and_literal_values() raises:
    var frame = DataFrame(
        [
            Series("p", Column[Int32]([1, 0, -1])),
            Series("v", Column[Int32]([3, 4, 5])),
        ]
    )
    var chain = (
        when(col("p") > 0).then(col("v")).when(col("p") < 0).then(col("v") * 2)
    )
    var compiled = compile_conditional(chain.end(), frame)
    assert_true(Bool(compiled))
    ref program = compiled.value()
    assert_equal(program.dtype, DataType.INT32)
    assert_equal(len(program.conditions), 2)
    assert_equal(program.nodes[len(program.nodes) - 1].op, CASE_NULL)
    for i in range(len(program.nodes)):
        ref node = program.nodes[i]
        if node.op == CASE_TEST or node.op == CASE_JUMP:
            assert_true(node.literal > UInt64(i))
            assert_true(node.literal <= UInt64(len(program.nodes)))
    # This common conditional count has Int64 values and an Int32 predicate.
    var counts = compile_conditional(
        when(col("p") > 0).then(1).otherwise(0), frame
    )
    assert_true(Bool(counts))
    assert_equal(counts.value().dtype, DataType.INT64)


def test_metal_conditional_program_rejects_unsupported_branch_types_and_math() raises:
    var frame = DataFrame(
        [
            Series("p", Column[Int32]([1])),
            Series("v", Column[Float32]([3])),
            Series("d", Column[Float64]([4])),
        ]
    )
    assert_false(
        Bool(
            compile_conditional(
                when(col("p") > 0).then(col("v")).otherwise(col("d")), frame
            )
        )
    )
    assert_false(
        Bool(
            compile_conditional(
                when(col("p") > 0).then(col("v").sqrt()).otherwise(col("v")),
                frame,
            )
        )
    )
    assert_true(
        Bool(
            compile_conditional(
                when(col("p") > 0).then(col("v") / 2).otherwise(col("v")), frame
            )
        )
    )
    assert_false(Bool(compile_conditional(col("v") + 1, frame)))
    assert_false(
        Bool(
            compile_conditional(
                when(col("p") > 0).then(col("v")).otherwise(col("v")),
                frame,
                root=99,
            )
        )
    )


def test_metal_conditional_aggregate_states_keys_and_grouped_planning() raises:
    from dataframe_metal.composition_plan import (
        plan_composition,
        composition_key,
        only_conditional_trees,
    )
    from dataframe_metal.group_plan import plan_grouped

    var frame = DataFrame(
        [
            Series("p", Column[Int32]([1, 0])),
            Series("v", Column[Int64]([3, 4])),
            Series("w", Column[Int64]([2, 1])),
        ]
    )
    var value = (
        when(col("p") > 0)
        .then(col("v") * col("w"))
        .otherwise(col("v") - col("w"))
    )
    var outputs = List[Expr](
        [
            value.sum().alias("sum"),
            value.count().alias("count"),
            value.mean().alias("mean"),
            value.min().alias("min"),
            value.max().alias("max"),
        ]
    )
    var compiled = plan_composition(frame.lazy().select_exprs(outputs))
    assert_true(Bool(compiled))
    ref plan = compiled.value()
    assert_equal(len(plan.fields), 1)
    assert_true(only_conditional_trees(plan))
    # A plain arithmetic tree in the same projection needs the mixed evaluator.
    var mixed = plan_composition(
        frame.lazy().select_exprs(
            [
                value.sum(),
                ((col("v") * col("w")) + col("v")).sum().alias("ordinary"),
            ]
        )
    )
    assert_true(Bool(mixed))
    assert_false(only_conditional_trees(mixed.value()))
    var native = plan_composition(
        frame.lazy().select_exprs(
            [
                value.sum(),
                col("w").count().alias("native_count"),
            ]
        )
    )
    assert_true(Bool(native))
    assert_true(only_conditional_trees(native.value()))
    assert_equal(len(plan.fields[0].conditions), 1)
    assert_equal(plan.inputs, List[String](["v", "w", "p"]))
    assert_equal(plan.fields[0].conditions[0].slots, List[Int]([2]))
    for output in plan.outputs:
        assert_equal(output.field, 0)
    var changed = (
        when(col("p") > 1)
        .then(col("v") * col("w"))
        .otherwise(col("v") - col("w"))
    )
    var other = plan_composition(frame.lazy().select(changed.sum()))
    assert_true(Bool(other))
    assert_false(composition_key(plan) == composition_key(other.value()))
    var grouped = plan_grouped(frame.lazy().group_by("p").agg(outputs))
    assert_true(Bool(grouped))
    assert_equal(len(grouped.value().selected.bundle.fields[0].conditions), 1)


def test_metal_conditional_normalization_preserves_eager_checked_definitions() raises:
    from dataframe_metal.region_plan import normalize_region
    from dataframe_metal.composition_plan import plan_composition

    var frame = DataFrame(
        [
            Series("p", Column[Int32]([0, 1])),
            Series("v", Column[Int32]([Int32.MAX, 3])),
        ]
    )
    var value = when(col("p") > 0).then(col("v") + 1).otherwise(0)
    var query = (
        frame.lazy().with_columns([value.alias("x")]).select(col("x").sum())
    )
    var normalized = normalize_region(query)
    assert_true(Bool(normalized))
    assert_true(Bool(plan_composition(normalized.value())))
    # x was already evaluated for every row before CASE selected a branch.
    # Moving x inside THEN would suppress the overflow in the first row.
    var eager = (
        frame.lazy()
        .with_columns([(col("v") + 1).alias("x")])
        .select(when(col("p") > 0).then(col("x")).otherwise(0).sum())
    )
    assert_false(Bool(normalize_region(eager)))
    var dependent = (
        frame.lazy()
        .with_columns([(col("v") + 1).alias("x")])
        .with_columns(
            [when(col("p") > 0).then(col("x")).otherwise(0).alias("y")]
        )
        .select(col("y").sum())
    )
    assert_false(Bool(normalize_region(dependent)))


# requires: metal
def test_metal_conditionals_native_types_chunks_masks_and_fused_outputs() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    for kind in range(3):
        var columns = List[Series]()
        var valid = List[Bool]([True, True, False, True, True, True])
        if kind == 0:
            columns.append(
                Series("v", Column[Int32]([99, 2, 3, 4, 5, 99], valid))
            )
        elif kind == 1:
            columns.append(
                Series("v", Column[Int64]([99, 2, 3, 4, 5, 99], valid))
            )
        else:
            columns.append(
                Series("v", Column[Float32]([99, 2, 3, 4, 5, 99], valid))
            )
        columns.append(
            Series(
                "p",
                Column[Int32](
                    [99, 1, 0, 1, 1, 99], [True, True, True, False, True, True]
                ),
            )
        )
        var source = DataFrame(columns^).slice(1, 4)
        var chunked = DataFrame(
            [
                Series._from_chunks(
                    [
                        source.column("v").slice(0, 1),
                        source.column("v").slice(1, 3),
                    ]
                ),
                Series._from_chunks(
                    [
                        source.column("p").slice(0, 3),
                        source.column("p").slice(3, 1),
                    ]
                ),
            ]
        )
        for frame in [source^, chunked^]:
            var value = when(col("p") > 0).then(col("v") * 2).otherwise(1)
            # Logical values: [4,1,1,10]. A null predicate takes ELSE; the
            # null v in row1 is in the untaken branch and cannot null its 1.
            var query = frame.lazy().select_exprs(
                [
                    value.sum().alias("s"),
                    value.count().alias("n"),
                    value.mean().alias("avg"),
                    value.min().alias("lo"),
                    value.max().alias("hi"),
                    col("v").count().alias("native_n"),
                ]
            )
            for copied in [False, True]:
                var answer = execute_timed(query, zero_copy=not copied)[
                    0
                ].copy()
                assert_equal(answer.item(0, "n").int64(), Int64(4))
                assert_equal(answer.item(0, "avg").float64(), Float64(4))
                assert_equal(answer.item(0, "native_n").int64(), Int64(3))
                if kind == 0:
                    assert_equal(answer.item(0, "s").int32(), Int32(16))
                    assert_equal(answer.item(0, "lo").int32(), Int32(1))
                    assert_equal(answer.item(0, "hi").int32(), Int32(10))
                elif kind == 1:
                    assert_equal(answer.item(0, "s").int64(), Int64(16))
                    assert_equal(answer.item(0, "lo").int64(), Int64(1))
                    assert_equal(answer.item(0, "hi").int64(), Int64(10))
                else:
                    assert_equal(answer.item(0, "s").float32(), Float32(16))
                    assert_equal(answer.item(0, "lo").float32(), Float32(1))
                    assert_equal(answer.item(0, "hi").float32(), Float32(10))
                assert_true(query.collect(accelerator=runtime).equals(answer))
            var grouped = (
                frame.lazy()
                .group_by("p", maintain_order=True)
                .agg(
                    [
                        value.sum().alias("s"),
                        value.count().alias("n"),
                        value.mean().alias("avg"),
                    ]
                )
            )
            var groups = execute_timed(grouped)[0].copy()
            assert_equal(groups.height(), 3)
            assert_equal(groups.item(0, "p").int32(), Int32(1))
            assert_equal(groups.item(1, "p").int32(), Int32(0))
            assert_true(groups.item(2, "p").is_null())
            assert_equal(groups.item(0, "n").int64(), Int64(2))
            assert_equal(groups.item(0, "avg").float64(), Float64(7))
            assert_equal(groups.item(1, "avg").float64(), Float64(1))
            assert_equal(groups.item(2, "avg").float64(), Float64(1))
            assert_true(grouped.collect(accelerator=runtime).equals(groups))


# requires: metal
def test_metal_conditionals_checked_faults_only_in_selected_branches() raises:
    from std.memory import bitcast
    from std.testing import assert_raises
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    for kind in range(2):
        var v = Series(
            "v", Column[Int32]([Int32.MAX, 3])
        ) if kind == 0 else Series("v", Column[Int64]([Int64.MAX, 3]))
        var frame = DataFrame([v^, Series("p", Column[Int32]([0, 1]))])
        var safe = when(col("p") > 0).then(col("v") + 1).otherwise(0)
        var query = frame.lazy().select_exprs(
            [safe.sum().alias("s"), safe.count().alias("n")]
        )
        var answer = execute_timed(query)[0].copy()
        assert_equal(answer.item(0, "n").int64(), Int64(2))
        if kind == 0:
            assert_equal(answer.item(0, "s").int32(), Int32(4))
        else:
            assert_equal(answer.item(0, "s").int64(), Int64(4))
        same_outcome(query)
        var unsafe = when(col("p") == 0).then(col("v") + 1).otherwise(0)
        var fault = frame.lazy().select(unsafe.count())
        with assert_raises(contains="overflow"):
            _ = execute_timed(fault)
        same_outcome(fault)
    var tiny = bitcast[DType.float32](UInt32(1))
    var small = DataFrame(
        [
            Series("v", Column[Float32]([tiny, 3])),
            Series("p", Column[Int32]([0, 1])),
        ]
    )
    var safe_float = when(col("p") > 0).then(col("v") * 0.5).otherwise(0)
    assert_equal(
        execute_timed(small.lazy().select(safe_float.sum()))[0]
        .item()
        .float32(),
        Float32(1.5),
    )
    var fault_float = small.lazy().select(
        when(col("p") == 0).then(col("v") * 0.5).otherwise(0).sum()
    )
    with assert_raises(contains="gradual underflow"):
        _ = execute_timed(fault_float)
    same_outcome(fault_float)


def test_metal_conditional_constants_limits_and_ninth_predicate_binding() raises:
    from dataframe_metal.composition_plan import plan_composition
    from dataframe_metal.predicate_plan import plan_predicates

    var columns = List[Series]()
    for i in range(8):
        columns.append(
            Series("v" + String(i), Column[Int64]([Int64(i + 1), Int64(i + 2)]))
        )
    columns.append(Series("p", Column[Int32]([0, 1])))
    var frame = DataFrame(columns^)
    var constant = when(lit(True)).then(col("v0")).end()
    assert_true(Bool(plan_composition(frame.lazy().select(constant.sum()))))
    assert_true(Bool(plan_composition(frame.lazy().select(constant.count()))))
    var otherwise = when(lit(False)).then(col("v0")).otherwise(col("v1"))
    assert_true(Bool(plan_composition(frame.lazy().select(otherwise.sum()))))
    var outputs = List[Expr]()
    for i in range(4):
        var a = col("v" + String(2 * i))
        var b = col("v" + String(2 * i + 1))
        var value = when(col("p") > 0).then(a + b).otherwise(a - b)
        outputs.append(value.sum().alias("s" + String(i)))
    var compiled = plan_composition(frame.lazy().select_exprs(outputs))
    assert_true(Bool(compiled))
    assert_equal(len(compiled.value().inputs), 9)
    assert_equal(compiled.value().predicate.filter_column, "p")
    for field in compiled.value().fields:
        assert_equal(field.conditions[0].slots, List[Int]([8]))
    # A WHERE over an earlier value slot and CASE over slot9 share the
    # compound route; a range must not accidentally read the CASE predicate.
    var filtered = frame.lazy().filter(col("v0") > 0).select_exprs(outputs)
    assert_false(Bool(plan_composition(filtered)))
    var predicates = plan_predicates(filtered)
    assert_true(Bool(predicates))
    assert_equal(predicates.value().program.slots, List[Int]([0]))
    assert_equal(predicates.value().bundle.predicate.filter_column, "p")
    var chain = when(col("p") == 0).then(col("v0"))
    for i in range(1, 8):
        chain = chain.when(col("p") == i).then(col("v0"))
    assert_true(Bool(compile_conditional(chain.end(), frame)))
    assert_false(
        Bool(
            compile_conditional(
                chain.when(col("p") == 8).then(col("v0")).end(), frame
            )
        )
    )
    var deep = col("v0")
    for i in range(7):
        deep = col("v1") + deep
    assert_true(
        Bool(
            compile_conditional(
                when(col("p") > 0).then(deep).otherwise(0), frame
            )
        )
    )
    deep = col("v1") + deep
    assert_false(
        Bool(
            compile_conditional(
                when(col("p") > 0).then(deep).otherwise(0), frame
            )
        )
    )


# requires: metal
def test_metal_conditional_four_values_eight_inputs_ninth_predicate_and_group_key() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var columns = List[Series]()
    for field in range(8):
        var values = List[Int64]()
        var valid = List[Bool]()
        for row in range(6):
            values.append(Int64(2 * row + field + 1))
            valid.append(row != (field % 4 + 1))
        columns.append(
            Series("v" + String(field), Column[Int64](values^, valid^))
        )
    columns.append(
        Series(
            "p",
            Column[Int32](
                [0, 1, 0, 1, 1, 0], [True, True, True, False, True, True]
            ),
        )
    )
    columns.append(Series("k", Column[Int32]([9, 0, 1, 0, 1, 9])))
    var sliced = DataFrame(columns^).slice(1, 4)
    var chunked_columns = List[Series]()
    for field in range(10):
        var name = (
            "v" + String(field) if field < 8 else "p" if field == 8 else "k"
        )
        var split = field % 3 + 1
        chunked_columns.append(
            Series._from_chunks(
                [
                    sliced.column(name).slice(0, split),
                    sliced.column(name).slice(split, 4 - split),
                ]
            )
        )
    var chunked = DataFrame(chunked_columns^)
    var outputs = List[Expr]()
    for field in range(4):
        var a = col("v" + String(2 * field))
        var b = col("v" + String(2 * field + 1))
        var value = when(col("p") > 0).then(a + b).otherwise(a - b)
        var suffix = String(field)
        outputs.append(value.sum().alias("s" + suffix))
        outputs.append(value.count().alias("n" + suffix))
        outputs.append(value.mean().alias("avg" + suffix))
        outputs.append(value.min().alias("lo" + suffix))
        outputs.append(value.max().alias("hi" + suffix))
    for frame in [sliced^, chunked^]:
        for grouped in [False, True]:
            var query = (
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg(outputs) if grouped else frame.lazy()
                .select_exprs(outputs)
            )
            for copied in [False, True]:
                var answer = execute_timed(query, zero_copy=not copied)[
                    0
                ].copy()
                assert_equal(answer.height(), 2 if grouped else 1)
                for group in range(2 if grouped else 1):
                    for field in range(4):
                        var total = Int64(0)
                        var count = Int64(0)
                        var lo = Int64.MAX
                        var hi = Int64.MIN
                        for row in range(1, 5):
                            if grouped and (row - 1) % 2 != group:
                                continue
                            if (
                                row == (2 * field) % 4 + 1
                                or row == (2 * field + 1) % 4 + 1
                            ):
                                continue
                            var a = Int64(2 * row + 2 * field + 1)
                            var b = a + 1
                            var value = a + b if row == 1 or row == 4 else a - b
                            total += value
                            count += 1
                            lo = min(lo, value)
                            hi = max(hi, value)
                        var suffix = String(field)
                        assert_equal(
                            answer.item(group, "s" + suffix).int64(), total
                        )
                        assert_equal(
                            answer.item(group, "n" + suffix).int64(), count
                        )
                        assert_equal(
                            answer.item(group, "avg" + suffix).float64(),
                            Float64(total) / Float64(count),
                        )
                        assert_equal(
                            answer.item(group, "lo" + suffix).int64(), lo
                        )
                        assert_equal(
                            answer.item(group, "hi" + suffix).int64(), hi
                        )
                assert_true(query.collect(accelerator=runtime).equals(answer))


# requires: metal
def test_metal_conditional_rounding_ieee_and_constant_predicates() raises:
    from std.memory import bitcast
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var tiny = bitcast[DType.float32](UInt32(1))
    var nan = bitcast[DType.float32](UInt32(0x7FC00017))
    var frame = DataFrame(
        [
            Series(
                "a",
                Column[Float32](
                    [bitcast[DType.float32](UInt32(0x3F800001)), 3, 4, 5]
                ),
            ),
            Series(
                "b",
                Column[Float32](
                    [bitcast[DType.float32](UInt32(0x3F7FFFFE)), 2, 1, 0]
                ),
            ),
            Series(
                "p",
                Column[Float32](
                    [tiny, nan, -tiny, 0], [True, True, True, False]
                ),
            ),
        ]
    )
    var rounded = (
        when(col("p") > 0).then((col("a") * col("b")) - 1).otherwise(0)
    )
    assert_equal(
        execute_timed(frame.lazy().select(rounded.sum()))[0].item().float32(),
        Float32(0),
    )
    var value = (
        when(~(col("p") > 0))
        .then(col("a"))
        .when(col("p").is_null())
        .then(col("b"))
        .end()
    )
    var query = frame.lazy().select_exprs(
        [
            value.sum().alias("s"),
            value.count().alias("n"),
            value.min().alias("lo"),
            value.max().alias("hi"),
        ]
    )
    # NaN satisfies NOT(p>0), a negative subnormal does too. The missing
    # predicate is unknown in that comparison, then matches IS NULL.
    var answer = execute_timed(query)[0].copy()
    assert_equal(answer.item(0, "s").float32(), Float32(7))
    assert_equal(answer.item(0, "n").int64(), Int64(3))
    assert_equal(answer.item(0, "lo").float32(), Float32(0))
    assert_equal(answer.item(0, "hi").float32(), Float32(4))
    same_outcome(query)
    var yes = when(lit(True)).then(col("a")).end()
    var no = when(lit(False)).then(col("a")).end()
    assert_equal(
        execute_timed(frame.lazy().select(yes.count()))[0].item().int64(),
        Int64(4),
    )
    assert_equal(
        execute_timed(frame.lazy().select(no.count()))[0].item().int64(),
        Int64(0),
    )
    assert_true(
        execute_timed(frame.lazy().select(no.mean()))[0].item().is_null()
    )


def test_metal_conditional_field_loop_keeps_offsets_nulls_and_native_fields() raises:
    from std.memory import Pointer, bitcast
    from dataframe_metal.accelerator.gpu.composed_reduce import (
        conditional_tree_fields,
    )
    from dataframe_metal.accelerator.gpu.multi_reduce import Bytes
    from dataframe_metal.expression_format import TREE_OP

    for kind in range(3):
        for base in [0, 5]:
            var data = List[List[UInt8]]()
            for slot in range(8):
                var values = List[UInt8](length=72, fill=0)
                for row in range(4):
                    var number = Int64(slot * 10 + row + 1)
                    if kind == 0:
                        values.unsafe_ptr().unsafe_bitcast[Int32]()[
                            unsafe_offset=base + row
                        ] = Int32(number)
                    elif kind == 1:
                        values.unsafe_ptr().unsafe_bitcast[Int64]()[
                            unsafe_offset=base + row
                        ] = number
                    else:
                        values.unsafe_ptr().unsafe_bitcast[Float32]()[
                            unsafe_offset=base + row
                        ] = Float32(number)
                data.append(values^)
            var bitmap = List[UInt8](length=8, fill=255)
            var predicate_bitmap = List[UInt8](length=8, fill=255)
            for row in [1, 3]:
                var bit = 3 + base + row
                predicate_bitmap[bit // 8] &= ~(UInt8(1) << UInt8(bit % 8))
            var pointers = List[Bytes]()
            for slot in range(8):
                pointers.append(
                    Bytes(unsafe_from_address=Int(data[slot].unsafe_ptr()))
                )
            var valid = Bytes(unsafe_from_address=Int(bitmap.unsafe_ptr()))
            var pb = Bytes(
                unsafe_from_address=Int(predicate_bitmap.unsafe_ptr())
            )
            # Descriptors belong to a later segment, while program and clause
            # offsets remain absolute relative to the submission's root.
            var descriptor_base = 7
            var words = List[UInt64](length=descriptor_base + 48, fill=0)
            for field in range(4):
                var at = descriptor_base + 16 + field * 8
                var start = len(words)
                words[at] = UInt64(TREE_OP)
                words[at + 6] = UInt64(start)
                words[at + 7] = 5
                words.extend(
                    List[UInt64](
                        [
                            UInt64(CASE_HEADER),
                            0,
                            0,
                            0,
                            UInt64(CASE_TEST),
                            UInt64(20),
                            0,
                            4,
                            0,
                            UInt64(2 * field),
                            0,
                            0,
                            UInt64(CASE_JUMP),
                            0,
                            0,
                            5,
                            0,
                            UInt64(2 * field + 1),
                            0,
                            0,
                            1,
                            2,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            8,
                            UInt64(3) | (UInt64(1) << 32),
                            0,
                            0,
                            0,
                            4,
                        ]
                    )
                )
            var root = Pointer[UInt64, UntrackedOrigin[mut=False]](
                unsafe_from_address=Int(words.unsafe_ptr())
            )
            for native_field in [-1, 2]:
                if native_field >= 0:
                    words[descriptor_base + 16 + native_field * 8] = 0
                for first in [0, 1]:
                    for row in range(4):
                        var result: Tuple[
                            SIMD[DType.uint64, 4], SIMD[DType.uint32, 4], UInt32
                        ]
                        if kind == 2:
                            result = conditional_tree_fields[True, True](
                                root,
                                root.unsafe_offset(descriptor_base),
                                first,
                                4,
                                base,
                                row,
                                UInt32(kind),
                                15,
                                pointers[0],
                                pointers[1],
                                pointers[2],
                                pointers[3],
                                pointers[4],
                                pointers[5],
                                pointers[6],
                                pointers[7],
                                valid,
                                valid,
                                valid,
                                valid,
                                valid,
                                valid,
                                valid,
                                valid,
                                pointers[0],
                                pb,
                            )
                        else:
                            result = conditional_tree_fields[False, True](
                                root,
                                root.unsafe_offset(descriptor_base),
                                first,
                                4,
                                base,
                                row,
                                UInt32(kind),
                                15,
                                pointers[0],
                                pointers[1],
                                pointers[2],
                                pointers[3],
                                pointers[4],
                                pointers[5],
                                pointers[6],
                                pointers[7],
                                valid,
                                valid,
                                valid,
                                valid,
                                valid,
                                valid,
                                valid,
                                valid,
                                pointers[0],
                                pb,
                            )
                        assert_equal(result[2], UInt32(0))
                        for field in range(4):
                            var present = (
                                field >= first and field != native_field
                            )
                            assert_equal(result[1][field], UInt32(present))
                            if present:
                                var slot = 2 * field + Int(row % 2 == 0)
                                var value = Int64(slot * 10 + row + 1)
                                var expected = UInt64(
                                    bitcast[DType.uint32](Float32(value))
                                ) if kind == 2 else UInt64(value)
                                assert_equal(result[0][field], expected)
            _ = (data^, bitmap^, predicate_bitmap^, words^)


def test_metal_simple_case_dispatch_proves_register_layout_without_narrowing_support() raises:
    from dataframe_metal.composition_plan import (
        plan_composition,
        simple_case_layout,
        simple_conditional_trees,
    )
    from dataframe_metal.accelerator.gpu.expression_value import _case_shape
    from std.memory import Pointer

    var frame = DataFrame(
        [
            Series("p", Column[Int32]([Int32(1), 0])),
            Series("v", Column[Int64]([Int64(3), 4])),
            Series("w", Column[Int64]([Int64(2), 1])),
        ]
    )
    var values = List[Expr](
        [
            when(col("p") > 0).then(col("v")).otherwise(col("w")),
            when(col("p") > 0).then(col("v") + 1).otherwise(col("w")),
            when(col("p") > 0).then(col("v")).otherwise(col("w") * 2),
            when(col("p") > 0).then(col("v") + 1).otherwise(col("w") * 2),
            when(col("p") > 0).then(col("v") + 1).end(),
        ]
    )
    for i in range(len(values)):
        var selected = plan_composition(
            frame.lazy().select(values[i].sum()), include_native=True
        )
        assert_true(Bool(selected))
        ref plan = selected.value()
        assert_true(simple_conditional_trees(plan))
        var shape = simple_case_layout(plan.fields[0].tree)
        assert_equal(shape, i if i < 4 else 1)
        var packed = List[UInt64](length=64, fill=0)
        for j in range(len(plan.fields[0].tree)):
            ref node = plan.fields[0].tree[j]
            packed[j * 4] = UInt64(node.op)
            packed[j * 4 + 1] = UInt64(node.input)
            packed[j * 4 + 3] = node.literal
        var pointer = Pointer[UInt64, UntrackedOrigin[mut=False]](
            unsafe_from_address=Int(packed.unsafe_ptr())
        )
        assert_equal(_case_shape(pointer, len(plan.fields[0].tree)), shape)
        _ = packed^
    var complex = List[Expr](
        [
            when(col("p") > 0)
            .then((col("v") + col("w")) * col("v"))
            .otherwise(0),
            when(col("p") > 0)
            .then(col("v"))
            .when(col("p") < 0)
            .then(col("w"))
            .end(),
            values[0].copy() + col("v"),
        ]
    )
    for expression in complex:
        var selected = plan_composition(
            frame.lazy().select(expression.sum()), include_native=True
        )
        assert_true(Bool(selected))
        assert_false(simple_conditional_trees(selected.value()))
    var mixed = plan_composition(
        frame.lazy().select_exprs(
            [
                values[0].sum(),
                ((col("v") + col("w")) * col("v")).sum().alias("tree"),
            ]
        ),
        include_native=True,
    )
    assert_true(Bool(mixed))
    assert_false(simple_conditional_trees(mixed.value()))
    var native = plan_composition(
        frame.lazy().select(col("v").sum()), include_native=True
    )
    assert_true(Bool(native))
    assert_false(simple_conditional_trees(native.value()))
