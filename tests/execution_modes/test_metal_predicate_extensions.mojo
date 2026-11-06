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
    for frame in [sliced, chunked]:
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


def _and_three(a: Int, b: Int) -> Int:
    return 0 if a == 0 or b == 0 else (-1 if a < 0 or b < 0 else 1)


def _or_three(a: Int, b: Int) -> Int:
    return 1 if a == 1 or b == 1 else (-1 if a < 0 or b < 0 else 0)


def test_metal_negated_predicates_preserve_three_valued_logic() raises:
    var frame = DataFrame(
        [
            Series("a", Column[Int32]([1])),
            Series("b", Column[Int64]([1])),
            Series("c", Column[Float32]([1])),
        ]
    )
    var predicates = List[Expr](
        [
            ~(col("a") > 0),
            ~((col("a") > 0) & (col("b") > 0)),
            ~((col("a") > 0) | (col("b") > 0)),
            ~((col("a") > 0) & ((col("b") > 0) | (col("c") > 0))),
        ]
    )
    for mode in range(len(predicates)):
        var matched = plan_predicates(
            frame.lazy().filter(predicates[mode]).select(col("a").sum())
        )
        assert_true(Bool(matched))
        for a in [-1, 0, 1]:
            for b in [-1, 0, 1]:
                for c in [-1, 0, 1]:
                    var value = a if mode == 0 else (
                        _and_three(a, b) if mode
                        == 1 else (
                            _or_three(a, b) if mode
                            == 2 else _and_three(a, _or_three(b, c))
                        )
                    )
                    var expected = value == 0  # NOT UNKNOWN remains UNKNOWN.
                    var mask = UInt32(0)
                    for i in range(len(matched.value().program.clauses)):
                        ref clause = matched.value().program.clauses[i]
                        var state = (
                            a if clause.bounds.filter_column
                            == "a" else (
                                b if clause.bounds.filter_column == "b" else c
                            )
                        )
                        var kept = state >= 0 and (
                            (state == 0) if clause.inverted else (state == 1)
                        )
                        mask |= UInt32(kept) << UInt32(i)
                    var word = matched.value().program.truth[Int(mask >> 5)]
                    assert_equal(
                        Bool(word & (UInt32(1) << (mask & 31))), expected
                    )
    var unequal = plan_predicates(
        frame.lazy().filter(col("a") != 0).select(col("a").sum())
    )
    assert_true(Bool(unequal))
    assert_true(unequal.value().program.clauses[0].inverted)
    assert_equal(unequal.value().program.clauses[0].bounds.low, Int64(0))
    assert_equal(unequal.value().program.clauses[0].bounds.high, Int64(0))


# requires: metal
def test_metal_negation_inequality_nan_zero_and_unknown_selection() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var nan = bitcast[DType.float32](UInt32(0x7FC0_0017))
    var tiny = bitcast[DType.float32](UInt32(1))
    var negative_zero = bitcast[DType.float32](UInt32(0x8000_0000))
    var frame = DataFrame(
        [
            Series("i", Column[Int32]([1, 2, 3, 4, 5, 6, 7, 8])),
            Series(
                "f",
                Column[Float32](
                    [nan, negative_zero, 0, tiny, -tiny, 1, -1, 0],
                    [True, True, True, True, True, True, True, False],
                ),
            ),
        ]
    )
    var predicates = List[Expr](
        [
            col("f") != 0,
            ~(col("f") > 0),
            ~(col("f").is_null()),
            ~((col("f") > 0) | (col("i") > 5)),
            ~((col("f") > 0) & (col("i") > 5)),
            ~((col("f") > 0) & (col("i") > 9)),
            ~((col("f") >= 0) & (col("f") <= 1)),
        ]
    )
    var sums = List[Int32]([Int32(23), 18, 28, 11, 22, 36, 13])
    var counts = List[Int64]([Int64(5), 5, 7, 4, 6, 8, 3])
    for mode in range(len(predicates)):
        var query = (
            frame.lazy()
            .filter(predicates[mode])
            .select_exprs(
                [col("i").sum().alias("s"), col("i").count().alias("n")]
            )
        )
        for copied in [False, True]:
            var answer = execute_timed(query, zero_copy=not copied)[0].copy()
            assert_equal(answer.item(0, "s").int32(), sums[mode])
            assert_equal(answer.item(0, "n").int64(), counts[mode])
        var public = query.collect(accelerator=runtime)
        assert_equal(public.item(0, "s").int32(), sums[mode])
        assert_equal(public.item(0, "n").int64(), counts[mode])
