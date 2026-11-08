"""Boolean and count reductions preserve empty, nullable and Kleene rules."""
# contract-group: execution.backends
from std.testing import assert_true, assert_equal
from dataframe import Column, DataFrame, DataType, Expr, Series, col
from dataframe_metal.logical_plan import (
    plan_logical,
    finish_logical,
    logical_key,
)
from dataframe_metal.predicate_plan import plan_predicates
from dataframe_metal.group_plan import plan_grouped


def _logical_outputs(value: Expr) -> List[Expr]:
    return [
        value.any().alias("any"),
        value.any(False).alias("kleene_any"),
        value.all().alias("all"),
        value.all(False).alias("kleene_all"),
        value.count().alias("valid"),
        value.null_count().alias("nulls"),
        value.len().alias("rows"),
    ]


def test_metal_logical_result_state_matches_all_small_boolean_multisets() raises:
    # Enumerate true/false/null populations independently of the GPU lowerer.
    for true_count in range(4):
        for false_count in range(4):
            for null_count in range(4):
                var bits = List[Bool]()
                var valid = List[Bool]()
                for i in range(true_count + false_count + null_count):
                    bits.append(i < true_count)
                    valid.append(i < true_count + false_count)
                var frame = DataFrame(
                    [Series("b", Column[Bool](bits^, valid^))]
                )
                var query = frame.lazy().select_exprs(
                    _logical_outputs(col("b"))
                )
                var lowered = plan_logical(query)
                assert_true(Bool(lowered))
                var actual = finish_logical(
                    lowered.value(),
                    lowered.value().region.collect(engine="cpu"),
                )
                assert_true(actual.equals(query.collect(engine="cpu")))
                assert_equal(actual.item(0, "any").bool(), true_count > 0)
                assert_equal(actual.item(0, "all").bool(), false_count == 0)
                assert_equal(
                    actual.item(0, "rows").int64(),
                    Int64(true_count + false_count + null_count),
                )
                assert_equal(actual.item(0, "nulls").int64(), Int64(null_count))
                assert_equal(
                    actual.item(0, "kleene_any").is_null(),
                    true_count == 0 and null_count > 0,
                )
                assert_equal(
                    actual.item(0, "kleene_all").is_null(),
                    false_count == 0 and null_count > 0,
                )


def test_metal_logical_plans_native_predicates_mixed_values_and_grouped_keys() raises:
    var frame = DataFrame(
        [
            Series("b", Column[Bool]([True, False, True], [True, False, True])),
            Series("n", Column[Int32]([1, 2, 3], [True, True, False])),
            Series("k", Column[Int64]([1, 2, 1])),
            Series("f", Column[Float32]([0.5, 1, 1.5], [True, False, True])),
        ]
    )
    from dataframe_metal.calibration import shape_key

    var present = DataFrame([Series("b", Column[Bool]([True, False, True]))])
    var nullable_key = shape_key(
        frame.lazy().select(col("b").any()), "bool-layout"
    )
    var present_key = shape_key(
        present.lazy().select(col("b").any()), "bool-layout"
    )
    assert_true(nullable_key != present_key)
    assert_equal(
        present_key,
        shape_key(present.lazy().select(col("b").any()), "bool-layout"),
    )
    from dataframe_metal.predicate_program import _emit, PredicateClause
    from dataframe_metal.expression_format import BOOL_COLUMN_COMPARISON

    for null_test in [False, True]:
        var expression = col("b").is_null() if null_test else col("b")
        var clauses = List[PredicateClause]()
        var code = List[Int]()
        assert_true(
            _emit(expression, len(expression._nodes) - 1, frame, clauses, code)
        )
        assert_equal(
            clauses[0].comparison, 0 if null_test else BOOL_COLUMN_COMPARISON
        )
    for predicate in [
        col("b"),
        ~col("b"),
        col("b").is_null(),
        col("b").is_not_null(),
        col("n").is_null(),
        col("b") | (col("n") > 1),
        col("n") > 1,
        col("n").eq_missing(1),
    ]:
        for grouped in [False, True]:
            var outputs = _logical_outputs(predicate)
            outputs.append(col("n").sum().alias("s"))
            outputs.append(col("f").mean().alias("avg"))
            var query = (
                frame.lazy()
                .group_by("k", maintain_order=True)
                .agg(outputs) if grouped else frame.lazy()
                .select_exprs(outputs)
            )
            var logical = plan_logical(query)
            assert_true(Bool(logical))
            if grouped:
                assert_true(Bool(plan_grouped(logical.value().region)))
            else:
                assert_true(
                    Bool(
                        plan_predicates(
                            logical.value().region, allow_simple=True
                        )
                    )
                )
            var actual = finish_logical(
                logical.value(), logical.value().region.collect(engine="cpu")
            )
            assert_true(actual.equals(query.collect(engine="cpu")))
    # Preserve the existing grouped LEN route and numeric COUNT fast path.
    assert_true(
        not Bool(plan_logical(frame.lazy().group_by("k").agg(col("n").len())))
    )
    assert_true(not Bool(plan_logical(frame.lazy().select(col("n").count()))))
    assert_true(Bool(plan_logical(frame.lazy().select(col("b").count()))))
    # A numeric COUNT of an alias must reach the existing moment matcher.
    # The logical matcher cannot bind WITH_COLUMNS outputs to the scan schema.
    from dataframe_metal.moment_plan import plan_moments

    var derived = frame.lazy().with_columns(
        [(col("n").cast(DataType.INT64) * col("k")).alias("derived")]
    )
    for grouped in [False, True]:
        var outputs: List[Expr] = [
            col("derived").var(0).alias("variance"),
            col("derived").std(0).alias("deviation"),
            col("derived").count().alias("count"),
        ]
        var query = derived.group_by("k").agg(
            outputs
        ) if grouped else derived.select_exprs(outputs)
        assert_true(not Bool(plan_logical(query)))
        assert_true(Bool(plan_moments(query)))


# requires: metal
def test_metal_logical_public_nullable_chunked_mixed_global_and_grouped() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series(
                "b",
                Column[Bool](
                    [True, False, False, False, True, False],
                    [True, True, False, True, True, False],
                ),
            ),
            Series(
                "n",
                Column[Int32](
                    [1, 2, 3, 4, 5, 6], [True, False, True, True, True, True]
                ),
            ),
            Series("k", Column[Int64]([1, 1, 2, 2, 3, 3])),
            Series(
                "f",
                Column[Float32](
                    [0.5, 1, 1.5, 2, 2.5, 3],
                    [True, True, False, True, True, False],
                ),
            ),
        ]
    )
    var chunked = List[Series]()
    for name in frame.columns():
        var source = frame.column(name)
        chunked.append(
            Series._from_chunks(
                [source.slice(0, 1), source.slice(1, 3), source.slice(4, 2)]
            )
        )
    frame = DataFrame(chunked^)
    for value in [
        col("b"),
        ~col("b"),
        col("b") & (col("n") > 2),
        col("n").eq_missing(1),
    ]:
        for grouped in [False, True]:
            for empty in [False, True]:
                var selected = frame.lazy().filter(
                    col("k") < 0
                ) if empty else frame.lazy()
                var outputs = _logical_outputs(value)
                outputs.append(col("n").sum().alias("s"))
                outputs.append(col("f").mean().alias("avg"))
                var query = selected.group_by("k", maintain_order=True).agg(
                    outputs
                ) if grouped else selected.select_exprs(outputs)
                var cpu = query.collect(engine="cpu")
                assert_true(execute_timed(query)[0].equals(cpu))
                assert_true(query.collect(accelerator=runtime).equals(cpu))


# requires: metal
def test_metal_logical_length_and_null_count_preserve_checked_value_work() raises:
    from tests.support.metal_outcomes import require_metal, same_outcome

    _ = require_metal()
    var frame = DataFrame([Series("w", Column[Int64]([Int64.MAX]))])
    var value = col("w") + 1
    same_outcome(frame.lazy().select(value.len()))
    same_outcome(frame.lazy().select(value.null_count()))
