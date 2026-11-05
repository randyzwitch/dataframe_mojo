"""Native extrema shape, ownership, range, and IEEE semantics contracts."""
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import Column, DataFrame, Series, col
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
    for values in [
        [tiny, negative_tiny, Float32(0)],
        [Float32(0), Float32(0)],
        [negative_zero, negative_zero],
        [nan, Float32(2), Float32(-3)],
        [Float32(1) / Float32(0), -Float32(1) / Float32(0)],
    ]:
        var frame = DataFrame([Series("f", Column[Float32](values))])
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
    for values in [[nan, nan], [negative_zero, Float32(0)]]:
        var query = (
            DataFrame([Series("f", Column[Float32](values))])
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
