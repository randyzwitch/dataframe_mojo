"""Bounded native expression trees retain checked arithmetic identity."""
# contract-group: execution.backends
from std.testing import assert_true, assert_false, assert_equal
from dataframe import Column, DataFrame, Series, Expr, col, lit
from dataframe.expr import ADD, SUM
from dataframe_metal.expression_plan import (
    compile_expression,
    _integer_fits_float32,
)
from dataframe_metal.composition_plan import (
    plan_composition,
    has_trees,
    composition_key,
)


def test_metal_expression_tree_schema_and_shared_states() raises:
    var frame = DataFrame(
        [
            Series("a", Column[Int32]([2, 3, 4])),
            Series("b", Column[Int32]([3, 4, 5])),
            Series("c", Column[Int32]([1, 2, 3])),
            Series("f", Column[Float32]([1, 2, 3])),
        ]
    )
    var value = (col("a") * col("b")) - col("c")
    var query = frame.lazy().select_exprs(
        [
            value.sum().alias("s"),
            value.count().alias("n"),
            value.mean().alias("avg"),
            value.min().alias("lo"),
            value.max().alias("hi"),
        ]
    )
    var matched = plan_composition(query)
    assert_true(Bool(matched))
    ref plan = matched.value()
    assert_true(has_trees(plan))
    assert_equal(len(plan.fields), 1)
    assert_equal(len(plan.fields[0].tree), 5)
    assert_equal(len(plan.inputs), 3)
    for output in plan.outputs:
        assert_equal(output.field, 0)
    assert_true(composition_key(plan).startswith("expression_reductions:v1:"))
    var changed = plan_composition(
        frame.lazy().select(((col("a") * col("b")) + col("c")).sum())
    )
    assert_true(Bool(changed))
    assert_false(composition_key(plan) == composition_key(changed.value()))
    assert_false(Bool(compile_expression(col("a") + col("f"), frame)))
    assert_false(Bool(compile_expression(col("a") / col("b"), frame)))
    assert_false(Bool(compile_expression(col("missing") + col("b"), frame)))
    var bounded = col("a")
    for i in range(8):
        bounded = bounded + col("b")
    assert_false(Bool(compile_expression(bounded, frame)))
    var invalid = (col("a") + col("b")) * col("c")
    invalid._nodes[4].left = 0
    assert_false(Bool(compile_expression(invalid, frame)))


def test_metal_expression_tree_literals_are_exact() raises:
    var frame = DataFrame([Series("f", Column[Float32]([1, 2]))])
    assert_true(_integer_fits_float32(Int64.MIN))
    assert_false(_integer_fits_float32(Int64.MAX))
    assert_true(_integer_fits_float32(Int64(1) << 62))
    assert_false(_integer_fits_float32((Int64(1) << 24) + 1))
    assert_true(Bool(compile_expression((col("f") * 2) - 1, frame)))
    assert_false(Bool(compile_expression(col("f") + lit(Int64.MAX), frame)))
    assert_false(Bool(compile_expression(col("f") + 0.1, frame)))
    assert_false(Bool(compile_expression((lit(2) + lit(3)) * col("f"), frame)))


def test_metal_expression_tree_normalization_tracks_definition_versions() raises:
    from dataframe_metal.region_plan import normalize_region

    var frame = DataFrame(
        [
            Series("a", Column[Int32]([1, 2])),
            Series("b", Column[Int32]([3, 4])),
        ]
    )
    var query = (
        frame.lazy()
        .with_columns([(col("a") * col("b")).alias("x")])
        .with_columns([(col("x") - 1).alias("y")])
        .select_exprs([col("y").sum().alias("s"), col("x").max().alias("hi")])
    )
    var normalized = normalize_region(query)
    assert_true(Bool(normalized))
    var matched = plan_composition(normalized.value())
    assert_true(Bool(matched))
    assert_true(has_trees(matched.value()))
    var overwritten = (
        frame.lazy()
        .with_columns([(col("a") + col("b")).alias("a")])
        .with_columns([(col("a") * col("b")).alias("a")])
        .select(col("a").sum())
    )
    var expanded = normalize_region(overwritten)
    assert_true(Bool(expanded))
    assert_true(Bool(plan_composition(expanded.value())))
    var unused = (
        frame.lazy()
        .with_columns([(col("a") * col("b")).alias("unused")])
        .select(col("a").sum())
    )
    assert_false(Bool(normalize_region(unused)))
    var prefilter = (
        frame.lazy()
        .with_columns([(col("a") * col("b")).alias("x")])
        .filter(col("a") > 0)
        .select(col("x").sum())
    )
    assert_false(Bool(normalize_region(prefilter)))


# requires: metal
def test_metal_expression_trees_nullable_sliced_chunked_and_fused() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    for kind in range(3):
        var columns = List[Series]()
        for field in range(3):
            var values = List[Int64]([Int64(99)])
            var validity = List[Bool]([True])
            for row in range(6):
                values.append(
                    Int64(row + (2 if field == 0 else 3 if field == 1 else 1))
                )
                validity.append(
                    row != (4 if field == 0 else 1 if field == 1 else 2)
                )
            values.append(99)
            validity.append(True)
            var name = "a" if field == 0 else "b" if field == 1 else "c"
            if kind == 0:
                var narrow = List[Int32]()
                for value in values:
                    narrow.append(Int32(value))
                columns.append(Series(name, Column[Int32](narrow^, validity^)))
            elif kind == 1:
                columns.append(Series(name, Column[Int64](values^, validity^)))
            else:
                var narrow = List[Float32]()
                for value in values:
                    narrow.append(Float32(value))
                columns.append(
                    Series(name, Column[Float32](narrow^, validity^))
                )
        var sliced = DataFrame(columns^).slice(1, 6)
        var chunks = List[Series]()
        for field in range(3):
            var name = "a" if field == 0 else "b" if field == 1 else "c"
            var split = field + 1
            chunks.append(
                Series._from_chunks(
                    [
                        sliced.column(name).slice(0, split),
                        sliced.column(name).slice(split, 6 - split),
                    ]
                )
            )
        var chunked = DataFrame(chunks^)
        for frame in [sliced^, chunked^]:
            for filtered in [False, True]:
                var value = (col("a") * col("b")) - (col("c") + 1)
                var offered = frame.lazy()
                if filtered:
                    offered = offered.filter(col("a") > 2)
                var query = offered.select_exprs(
                    [
                        value.sum().alias("s"),
                        value.count().alias("n"),
                        value.mean().alias("avg"),
                        value.min().alias("lo"),
                        value.max().alias("hi"),
                    ]
                )
                for copied in [False, True]:
                    var answer = execute_timed(query, zero_copy=not copied)[
                        0
                    ].copy()
                    assert_equal(
                        answer.item(0, "n").int64(), Int64(2 if filtered else 3)
                    )
                    assert_equal(
                        answer.item(0, "avg").float64(),
                        Float64(37 if filtered else 26),
                    )
                    if kind == 0:
                        assert_equal(
                            answer.item(0, "s").int32(),
                            Int32(74 if filtered else 78),
                        )
                        assert_equal(
                            answer.item(0, "lo").int32(),
                            Int32(25 if filtered else 4),
                        )
                        assert_equal(answer.item(0, "hi").int32(), Int32(49))
                    elif kind == 1:
                        assert_equal(
                            answer.item(0, "s").int64(),
                            Int64(74 if filtered else 78),
                        )
                        assert_equal(
                            answer.item(0, "lo").int64(),
                            Int64(25 if filtered else 4),
                        )
                        assert_equal(answer.item(0, "hi").int64(), Int64(49))
                    else:
                        assert_equal(
                            answer.item(0, "s").float32(),
                            Float32(74 if filtered else 78),
                        )
                        assert_equal(
                            answer.item(0, "lo").float32(),
                            Float32(25 if filtered else 4),
                        )
                        assert_equal(
                            answer.item(0, "hi").float32(), Float32(49)
                        )
                var public = query.collect(accelerator=runtime)
                assert_equal(
                    public.item(0, "n").int64(), Int64(2 if filtered else 3)
                )
                assert_equal(
                    public.item(0, "avg").float64(),
                    Float64(37 if filtered else 26),
                )


# requires: metal
def test_metal_expression_trees_four_values_eight_inputs_and_ninth_predicate() raises:
    from tests.support.metal_outcomes import require_metal
    from tests.support.metal_matrix_reference import (
        matrix_frame,
        matrix_expressions,
        matrix_reference,
    )
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var expected = matrix_reference(False)
    for fragmented in [False, True]:
        var frame = matrix_frame(fragmented)
        var query = (
            frame.lazy()
            .filter((col("p") > 0) | col("p").is_null())
            .select_exprs(matrix_expressions())
        )
        for copied in [False, True]:
            assert_true(
                execute_timed(query, zero_copy=not copied)[0].equals(expected)
            )
        assert_true(query.collect(accelerator=runtime).equals(expected))


# requires: metal
def test_metal_expression_tree_checked_faults_masks_ieee_and_rounding() raises:
    from std.memory import bitcast
    from std.testing import assert_raises
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    for kind in range(2):
        var a = Series(
            "a", Column[Int32]([Int32.MAX, 1])
        ) if kind == 0 else Series("a", Column[Int64]([Int64.MAX, 1]))
        var b = Series(
            "b", Column[Int32]([0, 2], [False, True])
        ) if kind == 0 else Series("b", Column[Int64]([0, 2], [False, True]))
        var frame = DataFrame([a^, b^, Series("p", Column[Int32]([0, 1]))])
        var value = (col("a") + 1) * col("b")
        var fault = frame.lazy().select(value.count())
        with assert_raises(contains="overflow"):
            _ = execute_timed(fault)
        same_outcome(fault)
        var safe = frame.lazy().filter(col("p") > 0).select(value.sum())
        var answer = execute_timed(safe)[0].copy()
        if kind == 0:
            assert_equal(answer.item().int32(), Int32(4))
        else:
            assert_equal(answer.item().int64(), Int64(4))
    var tiny = bitcast[DType.float32](UInt32(1))
    var small = DataFrame([Series("f", Column[Float32]([tiny, 1]))])
    var expression = (col("f") * 0.5) + 1
    var underflow = small.lazy().select(expression.sum())
    with assert_raises(contains="gradual underflow"):
        _ = execute_timed(underflow)
    same_outcome(underflow)
    var count = small.lazy().select(expression.count())
    assert_equal(execute_timed(count)[0].item().int64(), Int64(2))
    var rounded = DataFrame(
        [
            Series(
                "a",
                Column[Float32]([bitcast[DType.float32](UInt32(0x3F800001))]),
            ),
            Series(
                "b",
                Column[Float32]([bitcast[DType.float32](UInt32(0x3F7FFFFE))]),
            ),
        ]
    )
    var rounding = rounded.lazy().select(((col("a") * col("b")) - 1).sum())
    assert_equal(execute_timed(rounding)[0].item().float32(), Float32(0))


# requires: metal
def test_metal_expression_tree_fixed_topologies_keep_native_semantics() raises:
    from tests.support.metal_outcomes import require_metal
    from tests.support.metal_matrix_reference import _typed
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var expressions = List[Expr](
        [
            col("a") * col("b") - col("c"),
            col("a") - (col("b") * col("c")),
            ((col("a") + col("b")) * col("c")) - col("d"),
            (col("a") * col("b")) - (col("c") + col("d")),
            (col("a") + (col("b") * col("c"))) - col("d"),
            col("a") - ((col("b") * col("c")) + col("d")),
            col("a") - (col("b") * (col("c") + col("d"))),
            (col("a") * col("b")) - (1 - col("c")),
        ]
    )
    var expected_values = List[Float64](
        [Float64(2), -10, 19, 1, 13, -11, -13, 9]
    )
    for kind in range(3):
        var columns = List[Series]()
        for input in range(4):
            var name = (
                "a" if input
                == 0 else "b" if input
                == 1 else "c" if input
                == 2 else "d"
            )
            var value = Float64(
                2 if input == 0 else 3 if input == 1 else 4 if input == 2 else 1
            )
            var valid = List[Bool](length=4, fill=True)
            if input == 0:
                valid[2] = False
            if input == 2:
                valid[1] = False
            columns.append(
                _typed(kind, name, List[Float64](length=4, fill=value), valid)
            )
        var frame = DataFrame(columns^)
        for index in range(len(expressions)):
            var value = expected_values[index]
            var valid = List[Bool]([True])
            var expected = DataFrame(
                [
                    _typed(kind, "s", List[Float64]([value * 2]), valid),
                    Series("n", Column[Int64]([Int64(2)])),
                    Series("avg", Column[Float64]([value])),
                    _typed(kind, "lo", List[Float64]([value]), valid),
                    _typed(kind, "hi", List[Float64]([value]), valid),
                ]
            )
            var query = frame.lazy().select_exprs(
                [
                    expressions[index].sum().alias("s"),
                    expressions[index].count().alias("n"),
                    expressions[index].mean().alias("avg"),
                    expressions[index].min().alias("lo"),
                    expressions[index].max().alias("hi"),
                ]
            )
            for copied in [False, True]:
                assert_true(
                    execute_timed(query, zero_copy=not copied)[0].equals(
                        expected
                    )
                )
            assert_true(query.collect(accelerator=runtime).equals(expected))
