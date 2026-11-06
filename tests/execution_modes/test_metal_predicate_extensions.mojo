"""Numeric predicate extensions preserve null selection independently."""
# contract-group: execution.backends
from std.memory import bitcast
from std.utils.numerics import inf
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


def test_metal_column_comparison_schema_types_bindings_and_keys() raises:
    from dataframe_metal.predicate_plan import COLUMN_CLAUSE

    var frame = DataFrame(
        [
            Series("i", Column[Int32]([1])),
            Series("j", Column[Int32]([1])),
            Series("l", Column[Int64]([1])),
            Series("m", Column[Int64]([1])),
            Series("f", Column[Float32]([1])),
            Series("g", Column[Float32]([1])),
        ]
    )
    for kind in range(3):
        var left = String("i" if kind == 0 else ("l" if kind == 1 else "f"))
        var right = String("j" if kind == 0 else ("m" if kind == 1 else "g"))
        var predicates = List[Expr](
            [
                col(left) > col(right),
                col(left) >= col(right),
                col(left) < col(right),
                col(left) <= col(right),
                col(left) == col(right),
                col(left) != col(right),
            ]
        )
        var keys = List[String]()
        for expression in predicates:
            var matched = plan_predicates(
                frame.lazy().filter(expression).select(col("i").sum())
            )
            assert_true(Bool(matched))
            ref clause = matched.value().program.clauses[0]
            assert_equal(clause.kind, COLUMN_CLAUSE)
            assert_equal(clause.other, right)
            assert_true(clause.other_input >= 0)
            var key = predicate_key(matched.value())
            assert_true(key.startswith("predicate_reductions:v2:"))
            assert_false(key in keys)
            keys.append(key^)
    assert_false(
        Bool(
            plan_predicates(
                frame.lazy().filter(col("i") > col("l")).select(col("i").sum())
            )
        )
    )
    assert_false(
        Bool(
            plan_predicates(
                frame.lazy()
                .filter((col("i") + 1) > col("j"))
                .select(col("i").sum())
            )
        )
    )


def _integer_comparison(mode: Int, a: Int64, b: Int64) -> Bool:
    if mode == 0:
        return a > b
    if mode == 1:
        return a >= b
    if mode == 2:
        return a < b
    if mode == 3:
        return a <= b
    if mode == 4:
        return a == b
    return a != b


# requires: metal
def test_metal_column_comparisons_boundaries_masks_chunks_and_ieee() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    for kind in range(3):
        var low = Int64(Int32.MIN) if kind == 0 else Int64.MIN
        var high = Int64(Int32.MAX) if kind == 0 else Int64.MAX
        var av = List[Int64]([low, high, -1, 0, 1, 2, 3, 0])
        var bv = List[Int64]([low, high - 1, 0, -1, 1, 1, 4, 0])
        var valid_a = List[Bool](
            [True, True, True, True, True, True, True, True, False, True]
        )
        var valid_b = List[Bool](
            [True, True, True, True, True, True, True, False, True, True]
        )
        var columns = List[Series](
            [Series("i", Column[Int32]([0, 1, 2, 3, 4, 5, 6, 7, 8, 0]))]
        )
        if kind == 0:
            var a = List[Int32]([Int32(0)])
            var b = List[Int32]([Int32(0)])
            for row in range(8):
                a.append(Int32(av[row]))
                b.append(Int32(bv[row]))
            a.append(0)
            b.append(0)
            columns.append(Series("a", Column[Int32](a^, valid_a)))
            columns.append(Series("b", Column[Int32](b^, valid_b)))
        elif kind == 1:
            var a = List[Int64]([Int64(0)])
            var b = List[Int64]([Int64(0)])
            for row in range(8):
                a.append(av[row])
                b.append(bv[row])
            a.append(0)
            b.append(0)
            columns.append(Series("a", Column[Int64](a^, valid_a)))
            columns.append(Series("b", Column[Int64](b^, valid_b)))
        else:
            var nan = bitcast[DType.float32](UInt32(0x7FC00017))
            var tiny = bitcast[DType.float32](UInt32(1))
            var negative_zero = bitcast[DType.float32](UInt32(0x80000000))
            columns.append(
                Series(
                    "a",
                    Column[Float32](
                        [
                            Float32(0),
                            nan,
                            negative_zero,
                            0,
                            tiny,
                            -tiny,
                            inf[DType.float32](),
                            -inf[DType.float32](),
                            0,
                            0,
                        ],
                        valid_a,
                    ),
                )
            )
            columns.append(
                Series(
                    "b",
                    Column[Float32](
                        [
                            Float32(0),
                            nan,
                            0,
                            negative_zero,
                            0,
                            0,
                            inf[DType.float32](),
                            -inf[DType.float32](),
                            0,
                            0,
                        ],
                        valid_b,
                    ),
                )
            )
        var contiguous = DataFrame(columns^).slice(1, 8)
        var chunked = DataFrame(
            [
                Series._from_chunks(
                    [
                        contiguous.column("i").slice(0, 1),
                        contiguous.column("i").slice(1, 7),
                    ]
                ),
                Series._from_chunks(
                    [
                        contiguous.column("a").slice(0, 3),
                        contiguous.column("a").slice(3, 5),
                    ]
                ),
                Series._from_chunks(
                    [
                        contiguous.column("b").slice(0, 5),
                        contiguous.column("b").slice(5, 3),
                    ]
                ),
            ]
        )
        var float_sums = List[Int32]([Int32(4), 15, 5, 16, 11, 10])
        var float_counts = List[Int64]([Int64(1), 4, 1, 4, 3, 3])
        for frame in [contiguous^, chunked^]:
            var predicates = List[Expr](
                [
                    col("a") > col("b"),
                    col("a") >= col("b"),
                    col("a") < col("b"),
                    col("a") <= col("b"),
                    col("a") == col("b"),
                    col("a") != col("b"),
                ]
            )
            for mode in range(len(predicates)):
                var sum = Int32(0)
                var count = Int64(0)
                if kind == 2:
                    sum = float_sums[mode]
                    count = float_counts[mode]
                else:
                    for row in range(6):
                        if _integer_comparison(mode, av[row], bv[row]):
                            sum += Int32(row + 1)
                            count += 1
                var query = (
                    frame.lazy()
                    .filter(predicates[mode])
                    .select_exprs(
                        [col("i").sum().alias("s"), col("i").count().alias("n")]
                    )
                )
                for copied in [False, True]:
                    var answer = execute_timed(query, zero_copy=not copied)[
                        0
                    ].copy()
                    assert_equal(answer.item(0, "s").int32(), sum)
                    assert_equal(answer.item(0, "n").int64(), count)
                var public = query.collect(accelerator=runtime)
                assert_equal(public.item(0, "s").int32(), sum)
                assert_equal(public.item(0, "n").int64(), count)


def test_metal_shared_predicate_compiler_truth_and_input_binding() raises:
    from dataframe_metal.predicate_program import (
        compile_predicate,
        bind_predicate_inputs,
    )

    var frame = DataFrame(
        [
            Series("v", Column[Float32]([1, 2])),
            Series("p", Column[Int32]([0, 1])),
            Series("q", Column[Int64]([1, 2])),
            Series("r", Column[Int64]([2, 2])),
        ]
    )
    var expression = ((col("p") > 0) & ~(col("q") == col("r"))) | col(
        "v"
    ).is_null()
    var compiled = compile_predicate(expression, frame)
    assert_true(Bool(compiled))
    var program = compiled.take()
    assert_equal(len(program.clauses), 3)
    assert_equal(len(program.slots), 0)
    for mask in range(8):
        var expected = (Bool(mask & 1) and Bool(mask & 2)) or Bool(mask & 4)
        assert_equal(Bool(program.truth[0] & (UInt32(1) << UInt32(mask))), expected)
    var names = List[String](["v"])
    assert_true(bind_predicate_inputs(program, names))
    assert_equal(names, List[String](["v", "p", "q", "r"]))
    assert_equal(program.slots, List[Int]([1, 2, 0]))
    assert_equal(program.clauses[1].other_input, 3)
    # Binding can be repeated after another aggregate installs its inputs.
    names = List[String](["r", "q", "p", "v"])
    assert_true(bind_predicate_inputs(program, names))
    assert_equal(program.slots, List[Int]([2, 1, 3]))
    assert_equal(program.clauses[1].other_input, 0)


def test_metal_shared_predicate_binding_declines_without_partial_mutation() raises:
    from dataframe_metal.predicate_program import (
        compile_predicate,
        bind_predicate_inputs,
    )

    var frame = DataFrame(
        [
            Series("a", Column[Int32]([1])),
            Series("b", Column[Int32]([2])),
            Series("s", Column[String](["x"])),
        ]
    )
    var compiled = compile_predicate(col("a") > col("b"), frame)
    assert_true(Bool(compiled))
    var program = compiled.take()
    var names = List[String](["s"])
    assert_false(bind_predicate_inputs(program, names, limit=2))
    assert_equal(names, List[String](["s"]))
    assert_equal(len(program.slots), 0)
    assert_equal(program.clauses[0].other_input, -1)
    assert_false(Bool(compile_predicate(col("s") == "x", frame)))
    assert_false(Bool(compile_predicate(col("a") > 0, frame, root=99)))
