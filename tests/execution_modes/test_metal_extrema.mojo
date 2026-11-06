"""Native extrema shape, ownership, range, and IEEE semantics contracts."""
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import Column, DataFrame, Series, LazyFrame, col
from dataframe_metal.extrema_plan import plan_extrema, extrema_key
from dataframe_metal.handoff import execute_timed, region_key
from tests.support.metal_outcomes import require_metal, same_outcome


def test_native_extrema_schema_shares_fields_and_preserves_alias_order() raises:
    var frame = DataFrame(
        [
            Series("i", Column[Int32]([1, 2, 3])),
            Series("l", Column[Int64]([1, 2, 3])),
            Series("f", Column[Float32]([1, 2, 3])),
            Series("d", Column[Float64]([1, 2, 3])),
        ]
    )
    var query = (
        frame.lazy()
        .filter(col("i") >= 2)
        .select_exprs(
            [
                col("f").max().alias("top"),
                col("i").min().alias("bottom"),
                col("f").min().alias("first"),
                col("f").max().alias("again"),
            ]
        )
    )
    var matched = plan_extrema(query)
    assert_true(Bool(matched))
    assert_equal(len(matched.value().fields), 2)
    assert_equal(matched.value().outputs[2].field, 0)
    assert_equal(matched.value().outputs[3].field, 0)
    assert_equal(matched.value().outputs[1].name, "bottom")
    assert_equal(region_key(query), extrema_key(matched.value()))
    assert_true(region_key(query).startswith("native_extrema:v1:"))
    assert_false(Bool(plan_extrema(frame.lazy().select(col("d").min()))))
    assert_false(
        Bool(
            plan_extrema(
                frame.lazy().select_exprs(
                    [
                        col("i").min().alias("x"),
                        col("l").max().alias("x"),
                    ]
                )
            )
        )
    )
    assert_false(Bool(plan_extrema(frame.lazy().select((col("i") + 1).min()))))
    assert_false(
        Bool(
            plan_extrema(
                frame.lazy().select_exprs(
                    [
                        col("i").min().alias("m"),
                        col("i").sum().alias("s"),
                    ]
                )
            )
        )
    )


# requires: metal
def test_native_extrema_all_types_filters_slices_chunks_nulls_and_aliases() raises:
    var runtime = require_metal()
    for nulls in [False, True]:
        var integers = List[Int32]()
        var longs = List[Int64]()
        var floats = List[Float32]()
        var valid = List[Bool]()
        for i in range(4113):
            integers.append(Int32(i % 31 - 15))
            longs.append(Int64(i % 29 - 14) * (Int64(1) << 40))
            floats.append(Float32(i % 23 - 11) / 8)
            valid.append(not nulls or i % 7 != 2)
        var base = DataFrame(
            [
                Series("i", Column[Int32](integers^, valid)),
                Series("l", Column[Int64](longs^, valid)),
                Series("f", Column[Float32](floats^, valid)),
            ]
        )
        var columns = List[Series]()
        for name in base.columns():
            var source = base.column(name)
            columns.append(
                Series._from_chunks(
                    [
                        source.slice(5, 257),
                        source.slice(262, 1025),
                        source.slice(1287, 2803),
                    ]
                )
            )
        for frame in [base.slice(5, 4085), DataFrame(columns^), base.clear()]:
            for predicate in ["", "i", "l", "f"]:
                var offered = frame.lazy()
                if predicate:
                    offered = offered.filter(col(predicate) > 0)
                var query = offered.select_exprs(
                    [
                        col("i").min().alias("ilo"),
                        col("i").max().alias("ihi"),
                        col("l").max().alias("lhi"),
                        col("l").min().alias("llo"),
                        col("f").min().alias("flo"),
                        col("f").max().alias("fhi"),
                        col("f").min().alias("again"),
                    ]
                )
                var expected = query.collect(engine="cpu")
                assert_true(
                    "HANDOFF metal" in query.explain(accelerator=runtime)
                )
                assert_true(execute_timed(query)[0].equals(expected))
                assert_true(
                    execute_timed(query, zero_copy=False)[0].equals(expected)
                )
                assert_true(query.collect(accelerator=runtime).equals(expected))
        var empty = (
            base.lazy()
            .filter(col("i") > 100)
            .select_exprs(
                [
                    col("i").min().alias("lo"),
                    col("f").max().alias("hi"),
                ]
            )
        )
        assert_true(execute_timed(empty)[0].equals(empty.collect(engine="cpu")))


# requires: metal
def test_native_extrema_integer_boundaries_and_four_distinct_fields() raises:
    _ = require_metal()
    var frame = DataFrame(
        [
            Series("i", Column[Int32]([Int32.MIN, Int32.MAX, 0])),
            Series("l", Column[Int64]([Int64.MIN, Int64.MAX, 0])),
            Series("f", Column[Float32]([-1, 1, 0])),
            Series("j", Column[Int64]([Int64.MAX, Int64.MIN, 0])),
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            col("i").min().alias("imin"),
            col("i").max().alias("imax"),
            col("l").min().alias("lmin"),
            col("l").max().alias("lmax"),
            col("f").min().alias("fmin"),
            col("f").max().alias("fmax"),
            col("j").min().alias("jmin"),
            col("j").max().alias("jmax"),
        ]
    )
    for zero_copy in [True, False]:
        assert_true(
            execute_timed(query, zero_copy=zero_copy)[0].equals(
                query.collect(engine="cpu")
            )
        )


# requires: metal
def test_native_extrema_float_payloads_and_adjudication() raises:
    _ = require_metal()
    var tiny = bitcast[DType.float32](UInt32(1))
    var negative_tiny = bitcast[DType.float32](UInt32(0x8000_0001))
    var nan = bitcast[DType.float32](UInt32(0x7FC0_1234))
    var negative_zero = bitcast[DType.float32](UInt32(0x8000_0000))
    var cases: List[List[Float32]] = [
        [tiny, negative_tiny, Float32(0)],
        [Float32(0), Float32(0)],
        [negative_zero, negative_zero],
        [nan, Float32(2), Float32(-3)],
        [Float32(1) / Float32(0), -Float32(1) / Float32(0)],
    ]
    for values in cases:
        var frame = DataFrame([Series("f", Column[Float32](values.copy()))])
        var query = frame.lazy().select_exprs(
            [
                col("f").min().alias("lo"),
                col("f").max().alias("hi"),
            ]
        )
        var expected = query.collect(engine="cpu")
        for zero_copy in [True, False]:
            var actual = execute_timed(query, zero_copy=zero_copy)[0].copy()
            for name in ["lo", "hi"]:
                assert_equal(
                    bitcast[DType.uint32](actual.item(0, name).float32()),
                    bitcast[DType.uint32](expected.item(0, name).float32()),
                )
    var adjudicated: List[List[Float32]] = [
        [nan, nan],
        [negative_zero, Float32(0)],
    ]
    for values in adjudicated:
        var query = (
            DataFrame([Series("f", Column[Float32](values.copy()))])
            .lazy()
            .select_exprs(
                [
                    col("f").min().alias("lo"),
                    col("f").max().alias("hi"),
                ]
            )
        )
        with assert_raises(contains="CPU"):
            _ = execute_timed(query)
        same_outcome(query)
    var all_null = (
        DataFrame([Series("f", Column[Float32]([nan, tiny], [False, False]))])
        .lazy()
        .select_exprs(
            [
                col("f").min().alias("lo"),
                col("f").max().alias("hi"),
            ]
        )
    )
    assert_true(
        execute_timed(all_null)[0].equals(all_null.collect(engine="cpu"))
    )


def _mixed_cases(nulls: Bool) raises -> List[LazyFrame]:
    from dataframe import Expr, LazyFrame

    var columns = List[Series]()
    var valid = List[Bool]([True, not nulls, True, True])
    for i in range(4):
        columns.append(
            Series("i" + String(i), Column[Int32]([-3, -1, 1, 3], valid))
        )
        columns.append(
            Series(
                "l" + String(i),
                Column[Int64](
                    [
                        -(Int64(1) << 40),
                        -(Int64(1) << 39),
                        Int64(1) << 39,
                        Int64(1) << 40,
                    ],
                    valid,
                ),
            )
        )
        columns.append(
            Series(
                "f" + String(i),
                Column[Float32]([-0.75, -0.25, 0.25, 0.75], valid),
            )
        )
    columns.append(Series("p", Column[Int32]([0, 1, 1, 1])))
    var frame = DataFrame(columns^)
    var queries = List[LazyFrame]()
    for arity in range(1, 5):
        for floats in range(arity + 1):
            for narrow in range(arity - floats + 1):
                for means in [False, True]:
                    var expressions = List[Expr]()
                    for index in range(arity - 1, -1, -1):
                        var name = (
                            "f" if index
                            < floats else (
                                "i" if index < floats + narrow else "l"
                            )
                        ) + String(index)
                        expressions.append(col(name).max().alias("hi_" + name))
                        expressions.append(col(name).sum().alias("sum_" + name))
                        expressions.append(col(name).min().alias("lo_" + name))
                        expressions.append(col(name).count().alias("n_" + name))
                        if means:
                            expressions.append(
                                col(name).mean().alias("avg_" + name)
                            )
                    queries.append(
                        frame.lazy()
                        .filter(col("p") > 0)
                        .select_exprs(expressions)
                    )
    return queries^


def test_mixed_aggregate_calibration_reference_matches_cpu() raises:
    from benchmarks.bench_metal_calibration import (
        _table,
        _chunks,
        _reference,
        _expressions,
    )
    from dataframe_metal.calibration_capture import _check

    for rows in [32, 65, 127]:
        for nulls in [False, True]:
            var base = _table(rows, nulls)
            for chunks in [1, 8]:
                var frame = _chunks(base, chunks)
                for mode in range(5, 8):
                    var query = (
                        frame.lazy()
                        .filter(col("p") > 0)
                        .select_exprs(_expressions(mode))
                    )
                    assert_true(
                        region_key(query).startswith("mixed_reductions:v1:")
                    )
                    _check(
                        query.collect(engine="cpu"),
                        _reference(rows, nulls, mode),
                    )


def test_mixed_aggregate_schema_shares_states_and_preserves_output_order() raises:
    from dataframe_metal.composition_plan import (
        plan_composition,
        composition_key,
        has_extrema,
        single_float_composition,
    )

    var queries = _mixed_cases(False)
    assert_equal(len(queries), 68)
    for query in queries:
        var matched = plan_composition(query)
        assert_true(Bool(matched))
        ref bundle = matched.value()
        assert_true(has_extrema(bundle))
        assert_false(single_float_composition(bundle))
        assert_true(composition_key(bundle).startswith("mixed_reductions:v1:"))
        ref expressions = query._plan._nodes[len(query._plan._nodes) - 1].exprs
        for i in range(len(bundle.outputs)):
            assert_equal(
                bundle.fields[bundle.outputs[i].field].left,
                expressions[i]._nodes[0].text,
            )
            assert_equal(bundle.outputs[i].name, expressions[i]._name)
    var frame = DataFrame(
        [
            Series("i", Column[Int32]([1, 2])),
            Series("d", Column[Float64]([1, 2])),
        ]
    )
    assert_true(
        Bool(
            plan_composition(
                frame.lazy().select_exprs(
                    [
                        (col("i") + 1).min().alias("lo"),
                        col("i").sum().alias("s"),
                    ]
                )
            )
        )
    )
    assert_false(
        Bool(
            plan_composition(
                frame.lazy().select_exprs(
                    [col("i").min().alias("lo"), col("d").sum().alias("s")]
                )
            )
        )
    )
    assert_false(
        Bool(
            plan_composition(
                frame.lazy().select_exprs(
                    [col("i").min().alias("x"), col("i").sum().alias("x")]
                )
            )
        )
    )
    assert_false(
        Bool(
            plan_composition(
                frame.lazy().select_exprs(
                    [col("i").min().alias("lo"), col("i").max().alias("hi")]
                )
            )
        )
    )


# requires: metal
def test_mixed_aggregate_types_arities_nulls_and_aliases_match_cpu() raises:
    var runtime = require_metal()
    for nulls in [False, True]:
        for query in _mixed_cases(nulls):
            var expected = query.collect(engine="cpu")
            assert_true(execute_timed(query)[0].equals(expected))
            assert_true(
                execute_timed(query, zero_copy=False)[0].equals(expected)
            )
            assert_true(query.collect(accelerator=runtime).equals(expected))


# requires: metal
def test_mixed_aggregate_slices_chunk_boundaries_and_arithmetic_sums() raises:
    from dataframe import Expr, lit

    var runtime = require_metal()
    var f = List[Float32]()
    var i = List[Int32]()
    var valid = List[Bool]()
    for row in range(4113):
        f.append(Float32(row % 17 - 8) / 4)
        i.append(Int32(row % 5))
        valid.append(row % 11 != 1)
    var source = DataFrame(
        [
            Series("f", Column[Float32](f^, valid)),
            Series("i", Column[Int32](i^)),
        ]
    )
    var columns = List[Series]()
    for name in source.columns():
        var parts = List[Series]()
        var cuts = 7 if name == "f" else 11
        for part in range(cuts):
            var start = 5 + part * 4096 // cuts
            parts.append(
                source.column(name).slice(
                    start, (part + 1) * 4096 // cuts - part * 4096 // cuts
                )
            )
        columns.append(Series._from_chunks(parts))
    for frame in [source.slice(5, 4096), DataFrame(columns^), source.clear()]:
        for predicate in ["", "i", "f"]:
            var offered = frame.lazy()
            if predicate:
                offered = offered.filter(col(predicate) > 0)
            for mean in [False, True]:
                var expressions: List[Expr] = [
                    col("f").min().alias("lo"),
                    col("f").max().alias("hi"),
                    (col("f") * lit(Float32(2))).sum().alias("s"),
                    col("i").count().alias("n"),
                ]
                if mean:
                    expressions.append(col("f").mean().alias("avg"))
                var query = offered.select_exprs(expressions)
                var expected = query.collect(engine="cpu")
                assert_true(execute_timed(query)[0].equals(expected))
                assert_true(
                    execute_timed(query, zero_copy=False)[0].equals(expected)
                )
                assert_true(query.collect(accelerator=runtime).equals(expected))


# requires: metal
def test_mixed_extrema_float_bits_nan_counts_and_adjudication() raises:
    var runtime = require_metal()
    var tiny = bitcast[DType.float32](UInt32(1))
    var nan = bitcast[DType.float32](UInt32(0x7FC01234))
    var negative_zero = bitcast[DType.float32](UInt32(0x80000000))
    var cases: List[List[Float32]] = [
        [tiny, -tiny, 0],
        [negative_zero, negative_zero],
        [Float32(0), Float32(0)],
        [nan, 2, -3],
        [Float32(1) / Float32(0), -Float32(1) / Float32(0)],
    ]
    for values in cases:
        var query = (
            DataFrame([Series("f", Column[Float32](values.copy()))])
            .lazy()
            .select_exprs(
                [
                    col("f").min().alias("lo"),
                    col("f").count().alias("n"),
                    col("f").max().alias("hi"),
                ]
            )
        )
        var expected = query.collect(engine="cpu")
        for copy in [False, True]:
            var actual = execute_timed(query, zero_copy=not copy)[0].copy()
            assert_equal(
                actual.item(0, "n").int64(), expected.item(0, "n").int64()
            )
            for name in ["lo", "hi"]:
                assert_equal(
                    bitcast[DType.uint32](actual.item(0, name).float32()),
                    bitcast[DType.uint32](expected.item(0, name).float32()),
                )
    var adjudicated: List[List[Float32]] = [
        [nan, nan],
        [negative_zero, Float32(0)],
    ]
    for values in adjudicated:
        var query = (
            DataFrame([Series("f", Column[Float32](values.copy()))])
            .lazy()
            .select_exprs(
                [
                    col("f").min().alias("lo"),
                    col("f").count().alias("n"),
                    col("f").max().alias("hi"),
                ]
            )
        )
        with assert_raises(contains="CPU"):
            _ = execute_timed(query)
        assert_true(
            query.collect(accelerator=runtime).equals(
                query.collect(engine="cpu")
            )
        )


# requires: metal
def test_mixed_aggregate_empty_null_min_count_and_integer_overflow() raises:
    var runtime = require_metal()
    for valid in [True, False]:
        var frame = DataFrame(
            [
                Series("i", Column[Int32]([1, 2], [valid, valid])),
                Series("f", Column[Float32]([1, 2], [valid, valid])),
            ]
        )
        for offered in [
            frame.lazy(),
            frame.lazy().filter(col("i") > 10),
            frame.clear().lazy(),
        ]:
            var query = offered.select_exprs(
                [
                    col("i").min().alias("lo"),
                    col("f").max().alias("hi"),
                    col("i").sum(min_count=1).alias("s"),
                    col("f").mean().alias("avg"),
                    col("f").count().alias("n"),
                ]
            )
            assert_true(
                execute_timed(query)[0].equals(query.collect(engine="cpu"))
            )
    var frame = DataFrame(
        [
            Series("i", Column[Int32]([Int32.MAX, Int32.MAX])),
            Series("l", Column[Int64]([Int64.MIN, Int64.MAX])),
        ]
    )
    var means = frame.lazy().select_exprs(
        [
            col("i").min().alias("lo"),
            col("i").max().alias("hi"),
            col("i").mean().alias("avg"),
            col("l").min().alias("llo"),
            col("l").max().alias("lhi"),
        ]
    )
    assert_true(execute_timed(means)[0].equals(means.collect(engine="cpu")))
    for query in [
        frame.lazy().select_exprs(
            [col("i").min().alias("lo"), col("i").sum().alias("s")]
        ),
        frame.lazy().select_exprs(
            [col("i").max().alias("hi"), (col("i") + 1).count().alias("n")]
        ),
    ]:
        with assert_raises():
            _ = execute_timed(query)
        same_outcome(query)


# requires: metal
def test_mixed_aggregate_wide_bins_and_extrema_only_subnormal_fields() raises:
    var runtime = require_metal()
    var tiny = bitcast[DType.float32](UInt32(1))
    var huge = bitcast[DType.float32](UInt32(207) << 23)
    var frame = DataFrame(
        [
            Series(
                "tiny",
                Column[Float32]([tiny, bitcast[DType.float32](UInt32(2))]),
            ),
            Series("huge", Column[Float32]([huge, huge])),
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            col("tiny").min().alias("lo"),
            col("huge").mean().alias("avg"),
            col("tiny").max().alias("hi"),
            col("tiny").count().alias("n"),
            col("huge").max().alias("top"),
        ]
    )
    var expected = query.collect(engine="cpu")
    for copy in [False, True]:
        var actual = execute_timed(query, zero_copy=not copy)[0].copy()
        assert_true(actual.equals(expected))
        assert_equal(
            bitcast[DType.uint32](actual.item(0, "lo").float32()), UInt32(1)
        )
    var cancelled = (
        DataFrame([Series("f", Column[Float32]([huge, 1, -huge]))])
        .lazy()
        .select_exprs(
            [
                col("f").min().alias("lo"),
                col("f").sum().alias("s"),
                col("f").mean().alias("avg"),
                col("f").max().alias("hi"),
            ]
        )
    )
    with assert_raises(contains="CPU"):
        _ = execute_timed(cancelled)
    assert_true(
        cancelled.collect(accelerator=runtime).equals(
            cancelled.collect(engine="cpu")
        )
    )


# requires: metal
def test_mixed_aggregate_batches_once_and_preserves_independent_allocations() raises:
    from dataframe import DataType
    from dataframe_metal.composition_plan import plan_composition
    from dataframe_metal.composition_reduce import _mixed_dispatch
    from dataframe_metal.extrema_reduce import extrema_column
    from dataframe_metal.accelerator.gpu.reduction_inputs import HandoffTimings
    from dataframe_metal.accelerator.gpu.query_scope import Scope

    _ = require_metal()
    var values = List[Float32]()
    var predicate = List[Int64]()
    var validity = List[Bool]()
    for row in range(4113):
        values.append(Float32(row % 17 - 8) / 4)
        predicate.append(Int64(row % 4))
        validity.append(row % 11 != 1)
    var source = DataFrame(
        [
            Series("f", Column[Float32](values^, validity)),
            Series("p", Column[Int64](predicate^)),
        ]
    )
    for independent in [False, True]:
        var floats = List[Series]()
        var longs = List[Series]()
        for part in range(8):
            var start = 5 + part * 4096 // 8
            var length = (part + 1) * 4096 // 8 - part * 4096 // 8
            if independent:
                var data = source.column("f").numeric[DType.float32]()
                var words = source.column("p").numeric[DType.int64]()
                var f = List[Float32]()
                var p = List[Int64]()
                var valid = List[Bool]()
                for row in range(start, start + length):
                    f.append(data._get(row))
                    p.append(words._get(row))
                    valid.append(data.is_valid(row))
                floats.append(Series("f", Column[Float32](f^, valid)))
                longs.append(Series("p", Column[Int64](p^)))
            else:
                floats.append(source.column("f").slice(start, length))
                longs.append(source.column("p").slice(start, length))
        var frame = DataFrame(
            [Series._from_chunks(floats), Series._from_chunks(longs)]
        )
        var query = (
            frame.lazy()
            .filter(col("p") > 0)
            .select_exprs(
                [
                    col("f").min().alias("lo"),
                    col("f").sum().alias("s"),
                    col("f").mean().alias("avg"),
                    col("f").max().alias("hi"),
                    col("f").count().alias("n"),
                    col("p").count().alias("pn"),
                ]
            )
        )
        var expected = query.collect(engine="cpu")
        var matched = plan_composition(query)
        var scope = Scope()
        var timings = HandoffTimings()
        var result = _mixed_dispatch[True, False](
            matched.value(), frame, scope, True, timings
        )
        var low = extrema_column(
            "lo",
            DataType.FLOAT32,
            result.extrema[0],
            result.extrema[2] != 0,
            UInt32(result.extrema[3]),
        )
        assert_equal(
            low.numeric[DType.float32]()._get(0),
            expected.item(0, "lo").float32(),
        )
        assert_equal(
            Float32(result.floating[0]), expected.item(0, "s").float32()
        )
        assert_equal(Int64(result.counts[0]), expected.item(0, "n").int64())
        assert_equal(Int64(result.counts[1]), expected.item(0, "pn").int64())
        assert_equal(scope.transfers()[2], 8 if independent else 1)
        assert_equal(scope.info()[0], 0)
        assert_true(execute_timed(query, zero_copy=False)[0].equals(expected))


# requires: metal
def test_native_float_mixed_wide_bins_and_underflow_without_mean() raises:
    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series(
                "f",
                Column[Float32](
                    [
                        bitcast[DType.float32](UInt32(1)),
                        bitcast[DType.float32](UInt32(2)),
                        bitcast[DType.float32](UInt32(4)),
                    ]
                ),
            ),
            Series("p", Column[Int64]([1, 2, 3])),
        ]
    )
    for predicate in ["", "f", "p"]:
        var offered = frame.lazy()
        if predicate:
            offered = offered.filter(col(predicate) > 0)
        var query = offered.select_exprs(
            [
                col("f").min().alias("lo"),
                col("f").max().alias("hi"),
                col("f").sum().alias("s"),
                col("f").count().alias("n"),
                col("f").mean().alias("avg"),
            ]
        )
        var expected = query.collect(engine="cpu")
        for copy in [False, True]:
            var actual = execute_timed(query, zero_copy=not copy)[0].copy()
            assert_true(actual.equals(expected))
            assert_equal(
                bitcast[DType.uint32](actual.item(0, "lo").float32()), UInt32(1)
            )
            assert_equal(
                bitcast[DType.uint32](actual.item(0, "s").float32()), UInt32(7)
            )
        var ordinary_sum = offered.select_exprs(
            [col("f").min().alias("lo"), col("f").sum().alias("s")]
        )
        with assert_raises(contains="gradual underflow"):
            _ = execute_timed(ordinary_sum)
        assert_true(
            ordinary_sum.collect(accelerator=runtime).equals(
                ordinary_sum.collect(engine="cpu")
            )
        )


def test_arithmetic_extrema_schema_shares_expression_state() raises:
    from dataframe_metal.composition_plan import plan_composition, has_extrema
    from dataframe_metal.region_plan import normalize_region
    from dataframe import lit

    var frame = DataFrame([Series("i", Column[Int32]([1, 2, 3]))])
    var value = col("i") * 2
    var query = frame.lazy().select_exprs(
        [
            value.min().alias("lo"),
            value.max().alias("hi"),
            value.sum().alias("s"),
            value.count().alias("n"),
            value.mean().alias("avg"),
            value.min().alias("again"),
        ]
    )
    var matched = plan_composition(query)
    assert_true(Bool(matched))
    assert_true(has_extrema(matched.value()))
    assert_equal(len(matched.value().fields), 1)
    assert_equal(len(matched.value().inputs), 1)
    for output in matched.value().outputs:
        assert_equal(output.field, 0)
    assert_true(region_key(query).startswith("mixed_reductions:v1:"))
    var extrema = plan_composition(frame.lazy().select(value.min()))
    assert_true(Bool(extrema))
    assert_false(extrema.value().fields[0].needs_sum)
    var shared = (
        frame.lazy()
        .with_columns([value.alias("twice")])
        .select_exprs(
            [
                col("twice").min().alias("lo"),
                col("twice").max().alias("hi"),
            ]
        )
    )
    var normalized = normalize_region(shared)
    assert_true(Bool(normalized))
    var inlined = plan_composition(normalized.value())
    assert_true(Bool(inlined))
    assert_equal(len(inlined.value().fields), 1)
    assert_false(
        Bool(
            plan_composition(
                frame.lazy().select((col("i") + lit(Int64(1))).min())
            )
        )
    )
    assert_true(
        Bool(plan_composition(frame.lazy().select((col("i") + 1 + 1).min())))
    )


# requires: metal
def test_arithmetic_extrema_types_operands_nulls_slices_chunks_and_public_fusion() raises:
    from dataframe import Expr, lit

    var runtime = require_metal()
    for nulls in [False, True]:
        var i = List[Int32]()
        var j = List[Int32]()
        var l = List[Int64]()
        var m = List[Int64]()
        var f = List[Float32]()
        var g = List[Float32]()
        var av = List[Bool]()
        var bv = List[Bool]()
        for row in range(4113):
            i.append(Int32(row % 17 - 8))
            j.append(Int32(row % 11 - 5))
            l.append(Int64(row % 13 - 6) * (Int64(1) << 34))
            m.append(Int64(row % 7 - 3))
            f.append(Float32(row % 19 - 9) / 4)
            g.append(Float32(row % 7 - 3) / 2)
            av.append(not nulls or row % 7 != 1)
            bv.append(not nulls or row % 11 != 3)
        var source = DataFrame(
            [
                Series("i", Column[Int32](i^, av)),
                Series("j", Column[Int32](j^, bv)),
                Series("l", Column[Int64](l^, av)),
                Series("m", Column[Int64](m^, bv)),
                Series("f", Column[Float32](f^, av)),
                Series("g", Column[Float32](g^, bv)),
            ]
        )
        var columns = List[Series]()
        for name in source.columns():
            var column = source.column(name)
            var first = (
                263 if name == "i" or name == "l" or name == "f" else 519
            )
            columns.append(
                Series._from_chunks(
                    [
                        column.slice(5, first),
                        column.slice(5 + first, 4096 - first),
                    ]
                )
            )
        for frame in [
            source.slice(5, 4096),
            DataFrame(columns^),
            source.clear(),
        ]:
            for kind in range(3):
                var name = String(
                    "i" if kind == 0 else ("l" if kind == 1 else "f")
                )
                var other = String(
                    "j" if kind == 0 else ("m" if kind == 1 else "g")
                )
                var two = lit(Int32(2)) if kind == 0 else (
                    lit(Int64(2)) if kind == 1 else lit(Float32(2))
                )
                var values = List[Expr](
                    [
                        col(name) + col(other),
                        col(name) - col(other),
                        col(name) * col(other),
                        col(name) + two,
                        two - col(name),
                        col(name) * two,
                    ]
                )
                for value in values:
                    for filtered in [False, True]:
                        var offered = frame.lazy()
                        if filtered:
                            offered = offered.filter(col(name) > 0)
                        var query = offered.select_exprs(
                            [
                                value.max().alias("hi"),
                                value.min().alias("lo"),
                                value.count().alias("n"),
                                value.sum().alias("s"),
                                value.mean().alias("avg"),
                                col(name).max().alias("native"),
                                value.min().alias("again"),
                            ]
                        )
                        var expected = query.collect(engine="cpu")
                        assert_true(
                            "HANDOFF metal"
                            in query.explain(accelerator=runtime)
                        )
                        assert_true(execute_timed(query)[0].equals(expected))
                        assert_true(
                            execute_timed(query, zero_copy=False)[0].equals(
                                expected
                            )
                        )
                        assert_true(
                            query.collect(accelerator=runtime).equals(expected)
                        )
                        var pure = offered.select_exprs(
                            [value.min().alias("lo"), value.max().alias("hi")]
                        )
                        assert_true(
                            execute_timed(pure)[0].equals(
                                pure.collect(engine="cpu")
                            )
                        )


# requires: metal
def test_arithmetic_extrema_four_values_eight_inputs_and_independent_predicate() raises:
    from dataframe import Expr
    from dataframe_metal.composition_plan import plan_composition

    _ = require_metal()
    var columns = List[Series]()
    for i in range(8):
        columns.append(
            Series(
                "f" + String(i),
                Column[Float32]([Float32(i + 1), -Float32(i + 1), 2]),
            )
        )
    columns.append(Series("p", Column[Int64]([1, 0, 1])))
    var frame = DataFrame(columns^)
    var expressions = List[Expr]()
    for i in range(4):
        var value = col("f" + String(i * 2)) * col("f" + String(i * 2 + 1))
        expressions.append(value.min().alias("lo" + String(i)))
        expressions.append(value.max().alias("hi" + String(i)))
    var query = frame.lazy().filter(col("p") > 0).select_exprs(expressions)
    var matched = plan_composition(query)
    assert_equal(len(matched.value().fields), 4)
    assert_equal(len(matched.value().inputs), 8)
    assert_true(execute_timed(query)[0].equals(query.collect(engine="cpu")))
    assert_true(
        execute_timed(query, zero_copy=False)[0].equals(
            query.collect(engine="cpu")
        )
    )


# requires: metal
def test_arithmetic_extrema_overflow_masks_underflow_and_ieee_adjudication() raises:
    from dataframe import lit

    var runtime = require_metal()
    var boundary = DataFrame(
        [
            Series("i", Column[Int32]([Int32.MAX, Int32.MIN])),
            Series("l", Column[Int64]([Int64.MAX, Int64.MIN])),
        ]
    )
    for name in ["i", "l"]:
        var query = boundary.lazy().select((col(name) + 1).min())
        with assert_raises(contains="overflow"):
            _ = execute_timed(query)
        same_outcome(query)
    var large = DataFrame([Series("i", Column[Int32]([Int32.MAX, Int32.MAX]))])
    var extrema = large.lazy().select_exprs(
        [(col("i") - 1).min().alias("lo"), (col("i") - 1).max().alias("hi")]
    )
    assert_true(execute_timed(extrema)[0].equals(extrema.collect(engine="cpu")))
    var masked = DataFrame(
        [
            Series("i", Column[Int32]([Int32.MAX, 1])),
            Series("p", Column[Int32]([0, 1])),
        ]
    )
    var query = masked.lazy().filter(col("p") > 0).select((col("i") + 1).min())
    assert_true(execute_timed(query)[0].equals(query.collect(engine="cpu")))
    var invalid = DataFrame(
        [Series("i", Column[Int32]([Int32.MAX, 1], [False, True]))]
    )
    var safe = invalid.lazy().select((col("i") + 1).max())
    assert_true(execute_timed(safe)[0].equals(safe.collect(engine="cpu")))
    for raw in [UInt32(1), UInt32(0x0080_0000)]:
        var tiny = bitcast[DType.float32](raw)
        var frame = DataFrame([Series("f", Column[Float32]([tiny, tiny]))])
        var value = col("f") * lit(Float32(0.5))
        var underflow = frame.lazy().select_exprs(
            [value.min().alias("lo"), value.max().alias("hi")]
        )
        with assert_raises(contains="gradual underflow"):
            _ = execute_timed(underflow)
        assert_true(
            underflow.collect(accelerator=runtime).equals(
                underflow.collect(engine="cpu")
            )
        )
    var nan = bitcast[DType.float32](UInt32(0x7FC0_0017))
    for values in [
        List[Float32]([nan, nan]),
        List[Float32](
            [bitcast[DType.float32](UInt32(0x8000_0000)), Float32(0.0)]
        ),
    ]:
        var frame = DataFrame([Series("f", Column[Float32](values.copy()))])
        var value = col("f") * lit(Float32(1))
        var ieee = frame.lazy().select_exprs(
            [value.min().alias("lo"), value.max().alias("hi")]
        )
        with assert_raises(contains="CPU"):
            _ = execute_timed(ieee)
        assert_true(
            ieee.collect(accelerator=runtime).equals(ieee.collect(engine="cpu"))
        )
