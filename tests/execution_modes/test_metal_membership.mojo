"""Native numeric IN sets retain the engine's null and equality semantics."""
# contract-group: execution.backends
from std.testing import assert_true, assert_false, assert_equal
from std.memory import bitcast
from std.utils.numerics import inf
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    lit,
    null,
    when,
)
from dataframe_metal.membership_plan import (
    compile_membership,
    integer_key,
    float_key,
    MAX_MEMBERS,
)
from dataframe_metal.predicate_program import (
    compile_predicate,
    MEMBERSHIP_CLAUSE,
)
from dataframe_metal.predicate_plan import plan_predicates, predicate_key


def _native_number(
    answer: DataFrame, row: Int, name: String, kind: Int
) raises -> Float64:
    var item = answer.item(row, name)
    if kind == 0:
        return Float64(item.int32())
    if kind == 1:
        return Float64(item.int64())
    return Float64(item.float32())


def test_metal_membership_typed_integer_sets_canonicalize_order_and_duplicates() raises:
    var frame = DataFrame(
        [Series("v", Column[Int64]([Int64.MIN, -1, 0, 1, Int64.MAX]))]
    )
    var expression = col("v").is_in(
        [
            lit(Int64.MAX),
            lit(Int64(1)),
            lit(Int64.MIN),
            lit(Int64(-1)),
            lit(Int64(1)),
        ]
    )
    var compiled = compile_membership(
        expression, len(expression._nodes) - 1, frame
    )
    assert_true(Bool(compiled))
    assert_equal(
        compiled.value().keys,
        List[UInt64]([UInt64(0), integer_key(-1), integer_key(1), UInt64.MAX]),
    )
    var other = col("v").is_in(
        [lit(Int64(-1)), lit(Int64.MIN), lit(Int64.MAX), lit(Int64(1))]
    )
    var query = frame.lazy().filter(expression).select(col("v").count())
    var matched = plan_predicates(query)
    var match2 = plan_predicates(
        frame.lazy().filter(other).select(col("v").count())
    )
    assert_true(Bool(matched))
    assert_true(Bool(match2))
    assert_equal(predicate_key(matched.value()), predicate_key(match2.value()))
    assert_true(
        predicate_key(matched.value()).startswith("predicate_reductions:v3:")
    )
    var inverted = compile_predicate(~expression, frame)
    assert_true(Bool(inverted))
    assert_equal(inverted.value().clauses[0].kind, MEMBERSHIP_CLAUSE)
    assert_true(inverted.value().clauses[0].inverted)


def test_metal_membership_float_keys_preserve_ieee_and_null_candidates() raises:
    var tiny = bitcast[DType.float32](UInt32(1))
    var nan = bitcast[DType.float32](UInt32(0x7FC00017))
    var frame = DataFrame([Series("v", Column[Float32]([0, tiny, -tiny, nan]))])
    var expression = col("v").is_in(
        [
            lit(Float32(-0.0)),
            lit(Float32(0)),
            lit(tiny),
            lit(-tiny),
            lit(inf[DType.float32]()),
            lit(-inf[DType.float32]()),
            lit(nan),
            null(DataType.FLOAT32),
        ]
    )
    var compiled = compile_membership(
        expression, len(expression._nodes) - 1, frame
    )
    assert_true(Bool(compiled))
    assert_equal(
        compiled.value().keys,
        List[UInt64](
            [
                float_key(0xFF800000),
                float_key(0x80000001),
                float_key(0),
                float_key(1),
                float_key(0x7F800000),
            ]
        ),
    )
    var mismatch = col("v").is_in([lit(Int64(1))])
    assert_false(
        Bool(compile_membership(mismatch, len(mismatch._nodes) - 1, frame))
    )
    var rounded = col("v").is_in([Expr(0.1)])
    assert_false(
        Bool(compile_membership(rounded, len(rounded._nodes) - 1, frame))
    )


def test_metal_membership_empty_null_only_and_use_inside_case() raises:
    from dataframe_metal.composition_plan import (
        plan_composition,
        composition_key,
    )

    var frame = DataFrame(
        [
            Series("v", Column[Int32]([0, 1, 2])),
            Series("f", Column[Float32]([2, 3, 4])),
        ]
    )
    for expression in [
        col("v").is_in(List[Expr]()),
        col("v").is_in([null(DataType.INT32)]),
    ]:
        var compiled = compile_membership(
            expression, len(expression._nodes) - 1, frame
        )
        assert_true(Bool(compiled))
        assert_equal(len(compiled.value().keys), 0)
    var values = List[Expr]()
    for i in range(16):
        values.append(Expr(i))
    var first = when(col("v").is_in(values)).then(col("f")).otherwise(0)
    var second = (
        when(col("v").is_in([Expr(0), Expr(1)])).then(col("f")).otherwise(0)
    )
    var a = plan_composition(frame.lazy().select(first.sum()))
    var b = plan_composition(frame.lazy().select(second.sum()))
    assert_true(Bool(a))
    assert_true(Bool(b))
    assert_false(composition_key(a.value()) == composition_key(b.value()))
    var both = plan_composition(
        frame.lazy().select_exprs(
            [first.sum().alias("a"), second.sum().alias("b")]
        )
    )
    assert_true(Bool(both))
    assert_equal(len(both.value().fields), 2)


def test_metal_membership_large_sets_and_resource_limit() raises:
    var frame = DataFrame([Series("v", Column[Int64]([0, 1]))])
    var values = List[Expr]()
    for i in reversed(range(MAX_MEMBERS)):
        values.append(lit(Int64(i)))
    var expression = col("v").is_in(values)
    var compiled = compile_membership(
        expression, len(expression._nodes) - 1, frame
    )
    assert_true(Bool(compiled))
    assert_equal(len(compiled.value().keys), MAX_MEMBERS)
    assert_equal(compiled.value().keys[0], integer_key(0))
    assert_equal(
        compiled.value().keys[MAX_MEMBERS - 1], integer_key(MAX_MEMBERS - 1)
    )
    values.append(lit(Int64(MAX_MEMBERS)))
    expression = col("v").is_in(values)
    assert_false(
        Bool(compile_membership(expression, len(expression._nodes) - 1, frame))
    )


# requires: metal
def test_metal_membership_native_pairs_chunks_nulls_negation_and_case() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    for predicate_kind in range(3):
        for value_kind in range(3):
            var pvalid = List[Bool](
                [True, True, True, True, True, False, True, True, True, True]
            )
            var vvalid = List[Bool](
                [True, True, False, True, True, True, True, True, True, True]
            )
            var p = Series(
                "p", Column[Int32]([-99, -3, -2, -1, 0, 1, 2, 3, 4, 99], pvalid)
            )
            var literals = List[Expr](
                [lit(Int32(-2)), lit(Int32(1)), lit(Int32(3))]
            )
            if predicate_kind == 1:
                p = Series(
                    "p",
                    Column[Int64]([-99, -3, -2, -1, 0, 1, 2, 3, 4, 99], pvalid),
                )
                literals = List[Expr](
                    [lit(Int64(-2)), lit(Int64(1)), lit(Int64(3))]
                )
            elif predicate_kind == 2:
                p = Series(
                    "p",
                    Column[Float32](
                        [-99, -3, -2, -1, 0, 1, 2, 3, 4, 99], pvalid
                    ),
                )
                literals = List[Expr](
                    [lit(Float32(-2)), lit(Float32(1)), lit(Float32(3))]
                )
            var v = Series(
                "v", Column[Int32]([99, 1, 2, 3, 4, 5, 6, 7, 8, 99], vvalid)
            )
            if value_kind == 1:
                v = Series(
                    "v", Column[Int64]([99, 1, 2, 3, 4, 5, 6, 7, 8, 99], vvalid)
                )
            elif value_kind == 2:
                v = Series(
                    "v",
                    Column[Float32]([99, 1, 2, 3, 4, 5, 6, 7, 8, 99], vvalid),
                )
            var source = DataFrame(
                [
                    p^,
                    v^,
                    Series(
                        "k", Column[Int32]([99, 0, 1, 0, 1, 0, 1, 0, 1, 99])
                    ),
                ]
            ).slice(1, 8)
            var chunked = DataFrame(
                [
                    Series._from_chunks(
                        [
                            source.column("p").slice(0, 3),
                            source.column("p").slice(3, 5),
                        ]
                    ),
                    Series._from_chunks(
                        [
                            source.column("v").slice(0, 5),
                            source.column("v").slice(5, 3),
                        ]
                    ),
                    source.column("k"),
                ]
            )
            for frame in [source^, chunked^]:
                var member = col("p").is_in(literals)
                for inverted in [False, True]:
                    var predicate = ~member if inverted else member.copy()
                    var query = (
                        frame.lazy()
                        .filter(predicate)
                        .select_exprs(
                            [
                                col("v").sum().alias("s"),
                                col("v").count().alias("n"),
                                col("v").mean().alias("avg"),
                                col("v").min().alias("lo"),
                                col("v").max().alias("hi"),
                            ]
                        )
                    )
                    for copied in [False, True]:
                        var answer = execute_timed(query, zero_copy=not copied)[
                            0
                        ].copy()
                        var expected = Int64(22 if inverted else 7)
                        if value_kind == 0:
                            assert_equal(
                                answer.item(0, "s").int32(), Int32(expected)
                            )
                        elif value_kind == 1:
                            assert_equal(answer.item(0, "s").int64(), expected)
                        else:
                            assert_equal(
                                answer.item(0, "s").float32(), Float32(expected)
                            )
                        assert_equal(
                            answer.item(0, "n").int64(),
                            Int64(5 if inverted else 1),
                        )
                        assert_equal(
                            answer.item(0, "avg").float64(),
                            Float64(expected) / Float64(5 if inverted else 1),
                        )
                        assert_equal(
                            _native_number(answer, 0, "lo", value_kind),
                            Float64(1 if inverted else 7),
                        )
                        assert_equal(
                            _native_number(answer, 0, "hi", value_kind),
                            Float64(8 if inverted else 7),
                        )
                        assert_true(
                            query.collect(accelerator=runtime).equals(answer)
                        )
                var value = when(~member).then(col("v") * 2).otherwise(0)
                var conditional = frame.lazy().select_exprs(
                    [
                        value.sum().alias("s"),
                        value.count().alias("n"),
                        value.mean().alias("avg"),
                    ]
                )
                var answer = execute_timed(conditional)[0].copy()
                assert_equal(
                    _native_number(answer, 0, "s", value_kind), Float64(44)
                )
                assert_equal(answer.item(0, "n").int64(), Int64(8))
                assert_equal(answer.item(0, "avg").float64(), Float64(5.5))
                assert_true(
                    conditional.collect(accelerator=runtime).equals(answer)
                )
                # Both an outer WHERE set and a CASE set have variable tails.
                # The selected WHERE rows take ELSE even when v is missing.
                var both = (
                    frame.lazy()
                    .filter(member)
                    .select_exprs(
                        [
                            value.sum().alias("s"),
                            value.count().alias("n"),
                            value.mean().alias("avg"),
                        ]
                    )
                )
                var filtered = execute_timed(both)[0].copy()
                assert_equal(
                    _native_number(filtered, 0, "s", value_kind), Float64(0)
                )
                assert_equal(filtered.item(0, "n").int64(), Int64(2))
                assert_equal(filtered.item(0, "avg").float64(), Float64(0))
                assert_true(both.collect(accelerator=runtime).equals(filtered))
                var grouped = (
                    frame.lazy()
                    .group_by("k", maintain_order=True)
                    .agg(
                        [
                            value.sum().alias("s"),
                            value.count().alias("n"),
                            value.mean().alias("avg"),
                            value.min().alias("lo"),
                            value.max().alias("hi"),
                        ]
                    )
                )
                var groups = execute_timed(grouped)[0].copy()
                assert_equal(groups.height(), 2)
                for group in range(2):
                    assert_equal(
                        _native_number(groups, group, "s", value_kind),
                        Float64(8 if group == 0 else 36),
                    )
                    assert_equal(
                        _native_number(groups, group, "lo", value_kind),
                        Float64(0),
                    )
                    assert_equal(
                        _native_number(groups, group, "hi", value_kind),
                        Float64(6 if group == 0 else 16),
                    )
                assert_equal(groups.item(0, "n").int64(), Int64(4))
                assert_equal(groups.item(1, "n").int64(), Int64(4))
                assert_equal(groups.item(0, "avg").float64(), Float64(2))
                assert_equal(groups.item(1, "avg").float64(), Float64(9))
                assert_true(grouped.collect(accelerator=runtime).equals(groups))


# requires: metal
def test_metal_membership_ieee_empty_and_large_sets() raises:
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var tiny = bitcast[DType.float32](UInt32(1))
    var nan = bitcast[DType.float32](UInt32(0x7FC00017))
    var frame = DataFrame(
        [
            Series(
                "p",
                Column[Float32](
                    [
                        -inf[DType.float32](),
                        -tiny,
                        -0.0,
                        0,
                        tiny,
                        1.5,
                        inf[DType.float32](),
                        nan,
                        0,
                    ],
                    [True, True, True, True, True, True, True, True, False],
                ),
            ),
            Series("v", Column[Int64]([1, 2, 3, 4, 5, 6, 7, 8, 9])),
        ]
    )
    var member = col("p").is_in(
        [
            lit(Float32(0)),
            lit(tiny),
            lit(inf[DType.float32]()),
            lit(nan),
            null(DataType.FLOAT32),
        ]
    )
    for inverted in [False, True]:
        var query = (
            frame.lazy()
            .filter(~member if inverted else member.copy())
            .select_exprs(
                [col("v").sum().alias("s"), col("v").count().alias("n")]
            )
        )
        var answer = execute_timed(query)[0].copy()
        assert_equal(answer.item(0, "s").int64(), Int64(17 if inverted else 19))
        assert_equal(answer.item(0, "n").int64(), Int64(4))
        same_outcome(query)
    var empty = col("p").is_in(List[Expr]())
    assert_equal(
        execute_timed(frame.lazy().filter(empty).select(col("v").count()))[0]
        .item()
        .int64(),
        Int64(0),
    )
    assert_equal(
        execute_timed(frame.lazy().filter(~empty).select(col("v").count()))[0]
        .item()
        .int64(),
        Int64(8),
    )
    var ints = DataFrame(
        [
            Series(
                "p",
                Column[Int64](
                    [Int64(-1), 0, 7, 4095, 4096, Int64.MAX, 0],
                    [True, True, True, True, True, True, False],
                ),
            ),
            Series("v", Column[Int64]([1, 2, 3, 4, 5, 6, 7])),
        ]
    )
    var wanted = List[Expr]()
    for i in reversed(range(MAX_MEMBERS)):
        wanted.append(lit(Int64(i)))
    var large = col("p").is_in(wanted)
    for inverted in [False, True]:
        var query = (
            ints.lazy()
            .filter(~large if inverted else large.copy())
            .select(col("v").sum())
        )
        assert_equal(
            execute_timed(query)[0].item().int64(), Int64(12 if inverted else 9)
        )
        same_outcome(query)
    # Exercise the largest public Float32 expansion, not just raw lookup metadata.
    var float_large = DataFrame(
        [
            Series(
                "p",
                Column[Float32](
                    [-1, 0, 7, 4095, 4096, 8192, 0],
                    [True, True, True, True, True, True, False],
                ),
            ),
            Series("v", Column[Int64]([1, 2, 3, 4, 5, 6, 7])),
        ]
    )
    var float_candidates = List[Expr]()
    for i in reversed(range(MAX_MEMBERS)):
        float_candidates.append(lit(Float32(i)))
    var float_set = col("p").is_in(float_candidates)
    for inverted in [False, True]:
        var large_query = (
            float_large.lazy()
            .filter(~float_set if inverted else float_set.copy())
            .select(col("v").sum())
        )
        assert_equal(
            execute_timed(large_query)[0].item().int64(),
            Int64(12 if inverted else 9),
        )
        same_outcome(large_query)
    # A Float32 public IN list expands past the CASE instruction limit in
    # source form but lowers to one membership leaf plus seven value nodes.
    var candidates = List[Expr]()
    for i in range(16):
        candidates.append(lit(Float32(i)))
    var floats = DataFrame(
        [
            Series(
                "p",
                Column[Float32](
                    [-1, 0, 7, 15, 16, nan, 0],
                    [True, True, True, True, True, True, False],
                ),
            ),
            Series("v", Column[Int64]([1, 2, 3, 4, 5, 6, 7])),
        ]
    )
    var conditional = (
        when(col("p").is_in(candidates)).then(col("v") * 2).otherwise(1)
    )
    var query = (
        floats.lazy()
        .with_columns([conditional.alias("x")])
        .select_exprs([col("x").sum().alias("s"), col("x").count().alias("n")])
    )
    var answer = execute_timed(query)[0].copy()
    assert_equal(answer.item(0, "s").int64(), Int64(22))
    assert_equal(answer.item(0, "n").int64(), Int64(7))
    same_outcome(query)
