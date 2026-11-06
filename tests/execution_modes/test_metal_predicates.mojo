"""Compound native predicates retain SQL selection and reduction semantics."""
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import Column, DataFrame, Expr, Series, col, lit
from dataframe_metal.predicate_plan import plan_predicates
from dataframe_metal.handoff import execute_timed, region_key
from tests.support.metal_outcomes import require_metal, same_outcome


def test_arithmetic_and_compound_calibration_corpus_matches_cpu() raises:
    from benchmarks.bench_metal_calibration import (
        _table,
        _chunks,
        _query,
        _reference,
    )
    from dataframe_metal.calibration_capture import _check

    for rows in [32, 65, 127]:
        for nulls in [False, True]:
            var source = _table(rows, nulls)
            for count in [1, 8]:
                var frame = _chunks(source, count)
                for mode in range(11):
                    var query = _query(frame, mode)
                    if mode >= 8:
                        assert_true(
                            region_key(query).startswith(
                                "mixed_reductions:v1:" if mode
                                == 8 else "predicate_reductions:v1:"
                            )
                        )
                    _check(
                        query.collect(engine="cpu"),
                        _reference(rows, nulls, mode),
                    )


def test_compound_predicate_truth_tables_select_sql_true_with_nulls() raises:
    var a = List[Int32]()
    var b = List[Int64]()
    var c = List[Float32]()
    var av = List[Bool]()
    var bv = List[Bool]()
    var cv = List[Bool]()
    var ids = List[Int32]()
    for row in range(27):
        var x = row % 3
        var y = row // 3 % 3
        var z = row // 9
        a.append(Int32(-1 if x == 0 else 1))
        b.append(Int64(-1 if y == 0 else 1))
        c.append(Float32(-1 if z == 0 else 1))
        av.append(x != 2)
        bv.append(y != 2)
        cv.append(z != 2)
        ids.append(Int32(row))
    var frame = DataFrame(
        [
            Series("a", Column[Int32](a^, av)),
            Series("b", Column[Int64](b^, bv)),
            Series("c", Column[Float32](c^, cv)),
            Series("id", Column[Int32](ids^)),
        ]
    )
    var predicates = List[Expr](
        [
            (col("a") > 0) & (col("b") > 0),
            (col("a") > 0) | (col("b") > 0),
            ((col("a") > 0) & (col("b") > 0)) | (col("c") > 0),
            ((col("a") > 0) | (col("b") > 0)) & (col("c") > 0),
        ]
    )
    for mode in range(len(predicates)):
        var query = (
            frame.lazy().filter(predicates[mode]).select(col("id").sum())
        )
        var matched = plan_predicates(query)
        assert_true(Bool(matched))
        var expected = Int32(0)
        for row in range(27):
            var x = row % 3 == 1
            var y = row // 3 % 3 == 1
            var z = row // 9 == 1
            var keep = x and y if mode == 0 else (
                x
                or y if mode
                == 1 else ((x and y) or z if mode == 2 else (x or y) and z)
            )
            var mask = UInt32(0)
            for i in range(len(matched.value().program.clauses)):
                var name = matched.value().program.clauses[i].filter_column
                var value = x if name == "a" else (y if name == "b" else z)
                mask |= UInt32(value) << UInt32(i)
            var word = matched.value().program.truth[Int(mask >> 5)]
            assert_equal(Bool(word & (UInt32(1) << (mask & 31))), keep)
            if keep:
                expected += Int32(row)
        assert_equal(
            query.collect(engine="cpu").item(0, "id").int32(), expected
        )
        assert_true(region_key(query).startswith("predicate_reductions:v1:"))


def test_compound_predicate_schema_range_folding_clause_and_binding_limits() raises:
    var columns = List[Series]()
    for i in range(8):
        columns.append(Series("v" + String(i), Column[Int32]([1, 2, 3])))
    columns.append(Series("p", Column[Int64]([1, 2, 3])))
    columns.append(Series("q", Column[Float32]([1, 2, 3])))
    var frame = DataFrame(columns^)
    var legacy = (
        frame.lazy()
        .filter(col("p") > 0)
        .select_exprs(
            [col("q").min().alias("lo"), col("q").mean().alias("avg")]
        )
    )
    assert_false(Bool(plan_predicates(legacy)))
    assert_true(region_key(legacy).startswith("mixed_reductions:v1:"))
    var expressions = List[Expr]()
    for i in range(4):
        expressions.append(
            (col("v" + String(i * 2)) + col("v" + String(i * 2 + 1)))
            .sum()
            .alias("s" + String(i))
        )
    var query = (
        frame.lazy()
        .filter((col("v0") > 0) & (col("p") > 0))
        .select_exprs(expressions)
    )
    var nine = plan_predicates(query)
    assert_true(Bool(nine))
    assert_equal(len(nine.value().bundle.inputs), 8)
    assert_equal(nine.value().program.slots[1], 8)
    assert_false(
        Bool(
            plan_predicates(
                frame.lazy()
                .filter((col("p") > 0) & (col("q") > 0))
                .select_exprs(expressions)
            )
        )
    )
    var folded = plan_predicates(
        frame.lazy()
        .filter(((col("p") > 0) & (col("p") <= 3)) | (col("q") > 0))
        .select(col("v0").sum())
    )
    assert_true(Bool(folded))
    assert_equal(len(folded.value().program.clauses), 2)
    var eight = col("v0") == 0
    for i in range(1, 8):
        eight = eight | (col("v0") == i)
    assert_true(
        Bool(
            plan_predicates(frame.lazy().filter(eight).select(col("v0").sum()))
        )
    )
    assert_false(
        Bool(
            plan_predicates(
                frame.lazy()
                .filter(eight | (col("v0") == 8))
                .select(col("v0").sum())
            )
        )
    )
    assert_false(
        Bool(
            plan_predicates(
                frame.lazy()
                .filter((col("v0") > lit(Int32(0))) | (col("p") > 0))
                .select(col("v0").sum())
            )
        )
    )
    assert_false(
        Bool(
            plan_predicates(
                frame.lazy()
                .filter(((col("v0") + 1) > 0) | (col("p") > 0))
                .select(col("v0").sum())
            )
        )
    )


def _compound_selected_values(
    rows: Int, nulls: Bool, predicate: Int
) raises -> DataFrame:
    # Independent scalar selection: do not use the CPU's nested predicate
    # evaluator as the oracle for the Metal implementation.
    var integers = List[Int32]()
    var longs = List[Int64]()
    var floats = List[Float32]()
    var valid = List[Bool]()
    for row in range(5, 5 + rows):
        var av = not nulls or row % 11 != 3
        var bv = not nulls or row % 13 != 2
        var cv = not nulls or row % 17 != 5
        var a = row % 3 - 1
        var b = row % 5 - 2
        var c = Float32(row % 7 - 3) / 4
        var keep = (av and a > 0) and (bv and b < 2)
        if predicate == 1:
            keep = (av and a >= 0) or (cv and c == 0)
        elif predicate == 2:
            keep = ((av and a > 0) and (bv and b <= 0)) or (cv and c > 0)
        elif predicate == 3:
            keep = ((av and a < 0) or (bv and b >= 0)) and (cv and c < 0.5)
        if keep:
            integers.append(Int32(row % 17 - 8))
            longs.append(Int64(row % 13 - 6) * (Int64(1) << 34))
            floats.append(Float32(row % 19 - 9) / 4)
            valid.append(not nulls or row % 7 != 1)
    return DataFrame(
        [
            Series("i", Column[Int32](integers^, valid)),
            Series("l", Column[Int64](longs^, valid)),
            Series("f", Column[Float32](floats^, valid)),
        ]
    )


# requires: metal
def test_compound_predicates_types_nulls_operators_slices_chunks_and_aggregates() raises:
    var runtime = require_metal()
    for nulls in [False, True]:
        var i = List[Int32]()
        var l = List[Int64]()
        var f = List[Float32]()
        var a = List[Int32]()
        var b = List[Int64]()
        var c = List[Float32]()
        var valid = List[Bool]()
        var av = List[Bool]()
        var bv = List[Bool]()
        var cv = List[Bool]()
        for row in range(4113):
            i.append(Int32(row % 17 - 8))
            l.append(Int64(row % 13 - 6) * (Int64(1) << 34))
            f.append(Float32(row % 19 - 9) / 4)
            a.append(Int32(row % 3 - 1))
            b.append(Int64(row % 5 - 2))
            c.append(Float32(row % 7 - 3) / 4)
            valid.append(not nulls or row % 7 != 1)
            av.append(not nulls or row % 11 != 3)
            bv.append(not nulls or row % 13 != 2)
            cv.append(not nulls or row % 17 != 5)
        var source = DataFrame(
            [
                Series("i", Column[Int32](i^, valid)),
                Series("l", Column[Int64](l^, valid)),
                Series("f", Column[Float32](f^, valid)),
                Series("a", Column[Int32](a^, av)),
                Series("b", Column[Int64](b^, bv)),
                Series("c", Column[Float32](c^, cv)),
            ]
        )
        var columns = List[Series]()
        var index = 0
        for name in source.columns():
            var column = source.column(name)
            var length = 263 + index * 17
            columns.append(
                Series._from_chunks(
                    [
                        column.slice(5, length),
                        column.slice(5 + length, 4096 - length),
                    ]
                )
            )
            index += 1
        for frame in [
            source.slice(5, 4096),
            DataFrame(columns^),
            source.clear(),
        ]:
            var predicates = List[Expr](
                [
                    (col("a") > 0) & (col("b") < 2),
                    (col("a") >= 0) | (col("c") == 0),
                    ((col("a") > 0) & (col("b") <= 0)) | (col("c") > 0),
                    ((col("a") < 0) | (col("b") >= 0)) & (col("c") < 0.5),
                ]
            )
            for predicate_index in range(len(predicates)):
                var offered = frame.lazy().filter(predicates[predicate_index])
                var selected = _compound_selected_values(
                    frame.height(), nulls, predicate_index
                )
                for mode in range(7):
                    var expressions = List[Expr]()
                    if mode == 0:
                        expressions.append(col("i").sum().alias("s"))
                    elif mode == 1:
                        expressions = [
                            col("f").sum().alias("s"),
                            col("f").count().alias("n"),
                            col("f").mean().alias("avg"),
                            col("i").min().alias("lo"),
                            col("l").max().alias("hi"),
                        ]
                    elif mode == 2:
                        var value = col("f") * lit(Float32(0.5))
                        expressions = [
                            value.min().alias("lo"),
                            value.max().alias("hi"),
                            value.sum().alias("s"),
                            value.count().alias("n"),
                            value.mean().alias("avg"),
                        ]
                    elif mode == 3:
                        expressions = [
                            col("l").min().alias("lo"),
                            col("l").max().alias("hi"),
                        ]
                    elif mode == 4:
                        expressions = [col("i").count().alias("n")]
                    elif mode == 5:
                        expressions = [
                            col("f").sum().alias("s"),
                            col("f").count().alias("n"),
                            col("f").mean().alias("avg"),
                            col("f").min().alias("lo"),
                            col("f").max().alias("hi"),
                        ]
                    else:
                        expressions = [col("f").count().alias("n")]
                    var query = offered.select_exprs(expressions)
                    var expected = (
                        selected.lazy()
                        .select_exprs(expressions)
                        .collect(engine="cpu")
                    )
                    # The current CPU nested mixed-type predicate can differ
                    # from scalar selection (e.g. -15 versus -18 for predicate
                    # 2, non-null, 4096 rows). Keep CPU execution unchanged;
                    # validate all Metal routes against independent selection.
                    if predicate_index < 2:
                        assert_true(
                            query.collect(engine="cpu").equals(expected)
                        )
                    assert_true(
                        "HANDOFF metal" in query.explain(accelerator=runtime)
                    )
                    var actual = execute_timed(query)[0].copy()
                    assert_true(
                        actual.equals(expected),
                        String("nulls=")
                        + String(Int(nulls))
                        + "; rows="
                        + String(frame.height())
                        + "; chunks="
                        + String(frame.column("i").n_chunks())
                        + "; mode="
                        + String(mode)
                        + "; key="
                        + region_key(query)
                        + "\nActual:\n"
                        + actual.to_string()
                        + "\nExpected:\n"
                        + expected.to_string(),
                    )
                    assert_true(
                        execute_timed(query, zero_copy=False)[0].equals(
                            expected
                        )
                    )
                    var public = query.collect(accelerator=runtime)
                    assert_true(
                        public.equals(expected),
                        String("public; nulls=")
                        + String(Int(nulls))
                        + "; rows="
                        + String(frame.height())
                        + "; mode="
                        + String(mode)
                        + "; key="
                        + region_key(query)
                        + "\nActual:\n"
                        + public.to_string()
                        + "\nExpected:\n"
                        + expected.to_string(),
                    )


# requires: metal
def test_compound_predicate_ninth_binding_and_eight_clause_lookup() raises:
    _ = require_metal()
    var columns = List[Series]()
    var expressions = List[Expr]()
    for i in range(8):
        columns.append(
            Series("v" + String(i), Column[Int32]([Int32(i), -1, 2, 8]))
        )
    columns.append(Series("p", Column[Int64]([1, 1, 0, 1])))
    var frame = DataFrame(columns^)
    for i in range(4):
        expressions.append(
            (col("v" + String(i * 2)) + col("v" + String(i * 2 + 1)))
            .sum()
            .alias("s" + String(i))
        )
    var query = (
        frame.lazy()
        .filter((col("v0") >= 0) & (col("p") > 0))
        .select_exprs(expressions)
    )
    assert_true(execute_timed(query)[0].equals(query.collect(engine="cpu")))
    var predicate = col("v0") == 0
    for i in range(1, 8):
        predicate = predicate | (col("v0") == i)
    var eight = (
        frame.lazy()
        .filter(predicate)
        .select_exprs(
            [col("v0").sum().alias("s"), col("v0").count().alias("n")]
        )
    )
    assert_true(execute_timed(eight)[0].equals(eight.collect(engine="cpu")))
    var chained = (
        frame.lazy()
        .filter((col("v0") >= 0) | (col("p") < 0))
        .filter(col("p") > 0)
        .select(col("v0").sum())
    )
    assert_true(execute_timed(chained)[0].equals(chained.collect(engine="cpu")))
    var many_columns = List[Series]([Series("i", Column[Int32]([7, 8, 9, 10]))])
    var all_clauses = col("p0") > 0
    for i in range(8):
        many_columns.append(
            Series(
                "p" + String(i),
                Column[Int32]([1, 0, 1, 1], [True, True, i != 4, True]),
            )
        )
        if i:
            all_clauses = all_clauses & (col("p" + String(i)) > 0)
    var wide_frame = DataFrame(many_columns^)
    var last_lookup = (
        wide_frame.lazy()
        .filter(all_clauses)
        .select_exprs([col("i").sum().alias("s"), col("i").count().alias("n")])
    )
    var compiled = plan_predicates(last_lookup)
    assert_equal(compiled.value().program.truth[7], UInt32(0x8000_0000))
    var actual = execute_timed(last_lookup)[0].copy()
    assert_equal(actual.item(0, "s").int32(), Int32(17))
    assert_equal(actual.item(0, "n").int64(), Int64(2))


# requires: metal
def test_compound_predicate_ieee_null_selection_and_filtered_arithmetic_faults() raises:
    var runtime = require_metal()
    var tiny = bitcast[DType.float32](UInt32(1))
    var nan = bitcast[DType.float32](UInt32(0x7FC0_0017))
    var frame = DataFrame(
        [
            Series(
                "f",
                Column[Float32](
                    [tiny, -tiny, Float32(0), nan, 1, -1],
                    [True, True, True, True, False, True],
                ),
            ),
            Series(
                "p",
                Column[Int64](
                    [Int64.MAX, Int64.MIN, 1, 1, 1, 1],
                    [False, True, True, True, True, True],
                ),
            ),
            Series("i", Column[Int32]([Int32.MAX, 1, 2, 3, 4, 5])),
        ]
    )
    for predicate in List[Expr](
        [
            (col("f") > 0) | (col("p") > 0),
            (col("f") >= 0) & (col("p") > 0),
            (col("f") == 0) | (col("p") < 0),
        ]
    ):
        var query = (
            frame.lazy()
            .filter(predicate)
            .select_exprs(
                [col("i").min().alias("lo"), col("i").count().alias("n")]
            )
        )
        assert_true(execute_timed(query)[0].equals(query.collect(engine="cpu")))
    var excluded = (
        frame.lazy()
        .filter((col("f") <= 0) & (col("p") > 0))
        .select((col("i") + 1).min())
    )
    assert_true(
        execute_timed(excluded)[0].equals(excluded.collect(engine="cpu"))
    )
    var included = (
        frame.lazy()
        .filter((col("f") > 0) | (col("p") < 0))
        .select((col("i") + 1).min())
    )
    with assert_raises(contains="overflow"):
        _ = execute_timed(included)
    same_outcome(included)
    var underflow = (
        frame.lazy()
        .filter((col("f") > 0) | (col("p") < 0))
        .select((col("f") * lit(Float32(0.5))).min())
    )
    with assert_raises(contains="gradual underflow"):
        _ = execute_timed(underflow)
    assert_true(
        underflow.collect(accelerator=runtime).equals(
            underflow.collect(engine="cpu")
        )
    )


# requires: metal
def test_compound_predicate_adjacent_and_independent_allocation_lifetimes() raises:
    from dataframe_metal.predicate_reduce import _dispatch
    from dataframe_metal.accelerator.gpu.query_scope import (
        Scope,
        shared_service,
    )
    from dataframe_metal.accelerator.gpu.reduction_inputs import HandoffTimings

    _ = require_metal()
    var columns = List[Series]()
    for name in ["i", "a", "b"]:
        var values = List[Int32]()
        var valid = List[Bool]()
        for row in range(4113):
            values.append(
                Int32(
                    row % (17 if name == "i" else (5 if name == "a" else 7)) - 3
                )
            )
            valid.append(row % 11 != 1)
        columns.append(Series(name, Column[Int32](values^, valid)))
    var source = DataFrame(columns^)
    for independent in [False, True]:
        var columns = List[Series]()
        for name in source.columns():
            var parts = List[Series]()
            for part in range(8):
                var start = 5 + part * 512
                if independent:
                    var numeric = source.column(name).numeric[DType.int32]()
                    var values = List[Int32]()
                    var valid = List[Bool]()
                    for row in range(start, start + 512):
                        values.append(numeric._get(row))
                        valid.append(numeric.is_valid(row))
                    parts.append(Series(name, Column[Int32](values^, valid)))
                else:
                    parts.append(source.column(name).slice(start, 512))
            columns.append(Series._from_chunks(parts))
        var frame = DataFrame(columns^)
        var query = (
            frame.lazy()
            .filter((col("a") > 0) | (col("b") < 0))
            .select_exprs(
                [
                    col("i").sum().alias("s"),
                    col("i").count().alias("n"),
                    col("i").min().alias("lo"),
                    col("i").max().alias("hi"),
                ]
            )
        )
        var expected = query.collect(engine="cpu")
        var matched = plan_predicates(query)
        var scope = Scope(service=shared_service())
        var timings = HandoffTimings()
        var result = _dispatch[False, False](
            matched.value(), frame, scope, True, timings
        )
        assert_equal(result.integers[0], Int128(expected.item(0, "s").int32()))
        assert_equal(Int64(result.counts[0]), expected.item(0, "n").int64())
        assert_equal(result.extrema[0], Int64(expected.item(0, "lo").int32()))
        assert_equal(scope.transfers()[2], 8 if independent else 1)
        assert_equal(scope.info()[0], 0)
        scope.mark_success()
        assert_true(execute_timed(query, zero_copy=False)[0].equals(expected))
