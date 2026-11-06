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
