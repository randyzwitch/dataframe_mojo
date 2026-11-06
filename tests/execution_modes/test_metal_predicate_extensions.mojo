"""Numeric predicate extensions preserve null selection independently."""
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, Expr, Series, col
from dataframe_metal.predicate_plan import (
    plan_predicates,
    predicate_key,
    NULL_CLAUSE,
)


def test_metal_null_predicate_schema_and_truth_tables() raises:
    var frame = DataFrame(
        [
            Series("i", Column[Int32]([1, 2, 3])),
            Series("p", Column[Int64]([0, 0, 0], [False, True, False])),
            Series("f", Column[Float32]([0, 1, 2])),
            Series("s", Column[String](["a", "b", "c"])),
        ]
    )
    for expression in [
        col("p").is_null(),
        col("p").is_not_null(),
        col("f").is_null(),
    ]:
        var query = frame.lazy().filter(expression).select(col("i").sum())
        var plan = plan_predicates(query)
        assert_true(Bool(plan))
        assert_equal(plan.value().program.clauses[0].kind, NULL_CLAUSE)
        assert_true(
            predicate_key(plan.value()).startswith("predicate_reductions:v2:")
        )
        assert_equal(plan.value().program.truth[0], UInt32(2))
    assert_false(
        Bool(
            plan_predicates(
                frame.lazy().filter(col("s").is_null()).select(col("i").sum())
            )
        )
    )
    var first = plan_predicates(
        frame.lazy().filter(col("p").is_null()).select(col("i").sum())
    )
    var second = plan_predicates(
        frame.lazy().filter(col("p").is_not_null()).select(col("i").sum())
    )
    assert_false(first.value().program.clauses[0].inverted)
    assert_true(second.value().program.clauses[0].inverted)
    assert_false(predicate_key(first.value()) == predicate_key(second.value()))


# requires: metal
def test_metal_null_predicates_fused_with_aggregates_and_ieee_values() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var nan = bitcast[DType.float32](UInt32(0x7FC0_0017))
    var tiny = bitcast[DType.float32](UInt32(1))
    var source = DataFrame(
        [
            Series("i", Column[Int32]([99, 1, 2, 3, 4, 99])),
            Series(
                "p",
                Column[Int64](
                    [0, 0, 0, 0, 0, 0], [True, False, True, False, True, True]
                ),
            ),
            Series(
                "f",
                Column[Float32](
                    [0, nan, 0, tiny, 1, 0],
                    [True, True, False, True, True, True],
                ),
            ),
        ]
    )
    var sliced = source.slice(1, 4)
    var chunked = DataFrame(
        [
            Series._from_chunks(
                [sliced.column("i").slice(0, 1), sliced.column("i").slice(1, 3)]
            ),
            Series._from_chunks(
                [sliced.column("p").slice(0, 3), sliced.column("p").slice(3, 1)]
            ),
            Series._from_chunks(
                [sliced.column("f").slice(0, 2), sliced.column("f").slice(2, 2)]
            ),
        ]
    )
    for frame in [sliced^, chunked^]:
        var predicates = List[Expr](
            [
                col("p").is_null(),
                col("p").is_not_null(),
                col("f").is_null(),
                col("f").is_not_null(),
                (col("p").is_null() & (col("i") > 2))
                | (col("p").is_not_null() & (col("i") < 3)),
            ]
        )
        var sums = List[Int32]([Int32(4), 6, 2, 8, 5])
        var counts = List[Int64]([Int64(2), 2, 1, 3, 2])
        for mode in range(len(predicates)):
            var query = (
                frame.lazy()
                .filter(predicates[mode])
                .select_exprs(
                    [
                        col("i").sum().alias("s"),
                        col("i").count().alias("n"),
                        col("i").min().alias("lo"),
                        col("i").max().alias("hi"),
                    ]
                )
            )
            for copied in [False, True]:
                var answer = execute_timed(query, zero_copy=not copied)[
                    0
                ].copy()
                assert_equal(answer.item(0, "s").int32(), sums[mode])
                assert_equal(answer.item(0, "n").int64(), counts[mode])
            var public = query.collect(accelerator=runtime)
            assert_equal(public.item(0, "s").int32(), sums[mode])
            assert_equal(public.item(0, "n").int64(), counts[mode])
