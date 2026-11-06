"""Bounded grouped reductions accept native integer keys without scanning."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, Series, Expr, col
from dataframe_metal.group_plan import plan_grouped, grouped_key


def _numbers(
    kind: Int, name: String, values: List[Int64], valid: List[Bool]
) raises -> Series:
    if kind == 0:
        var narrowed = List[Int32]()
        for value in values:
            narrowed.append(Int32(value))
        return Series(name, Column[Int32](narrowed^, valid))
    if kind == 1:
        return Series(name, Column[Int64](values.copy(), valid))
    var narrowed = List[Float32]()
    for value in values:
        narrowed.append(Float32(value))
    return Series(name, Column[Float32](narrowed^, valid))


def test_metal_grouped_reduction_schema_and_predicate_programs() raises:
    var frame = DataFrame(
        [
            Series("k", Column[Int64]([0, 1, 0])),
            Series("v", Column[Float32]([2, 3, 4])),
            Series("p", Column[Int32]([1, 2, 3])),
            Series("s", Column[String](["a", "b", "a"])),
        ]
    )
    var outputs = List[Expr](
        [
            col("v").sum().alias("s"),
            col("v").count().alias("n"),
            col("v").mean().alias("avg"),
            col("v").min().alias("lo"),
            col("v").max().alias("hi"),
        ]
    )
    var query = frame.lazy().group_by("k", maintain_order=True).agg(outputs)
    var matched = plan_grouped(query)
    assert_true(Bool(matched))
    assert_equal(matched.value().key, "k")
    assert_equal(len(matched.value().selected.bundle.fields), 1)
    assert_equal(len(matched.value().selected.program.clauses), 0)
    assert_equal(matched.value().selected.program.truth[0], UInt32(1))
    assert_true(
        grouped_key(matched.value()).startswith("grouped_reductions:v1:")
    )
    var filtered = (
        frame.lazy()
        .filter((col("p") > 1) | col("v").is_null())
        .group_by("k")
        .agg(outputs)
    )
    var selected = plan_grouped(filtered)
    assert_true(Bool(selected))
    assert_equal(len(selected.value().selected.program.clauses), 2)
    assert_false(grouped_key(matched.value()) == grouped_key(selected.value()))
    var counted = plan_grouped(frame.lazy().group_by("k").len())
    assert_true(Bool(counted))
    assert_equal(counted.value().row_counts[0], "len")
    assert_false(Bool(plan_grouped(frame.lazy().group_by("v").agg(outputs))))
    assert_false(Bool(plan_grouped(frame.lazy().group_by("s").agg(outputs))))
    assert_false(
        Bool(plan_grouped(frame.lazy().group_by(["k", "p"]).agg(outputs)))
    )
    assert_false(
        Bool(
            plan_grouped(
                frame.lazy().group_by("k").agg(col("v").sum().alias("k"))
            )
        )
    )


def test_metal_grouped_partial_finish_retains_wide_sums_and_null_group() raises:
    from std.memory import bitcast
    from dataframe_metal.group_finish import finish_grouped
    from dataframe_metal.accelerator.gpu.group_hash import GROUP_SLOTS
    from dataframe_metal.accelerator.gpu.group_state import GROUP_STATE_WORDS

    var frame = DataFrame(
        [
            Series("k", Column[Int64]([0, 9], [False, True])),
            Series("v", Column[Int64]([0, 0])),
        ]
    )
    var matched = plan_grouped(
        frame.lazy()
        .group_by("k")
        .agg(
            [
                col("v").sum().alias("s"),
                col("v").count().alias("n"),
                col("v").mean().alias("avg"),
                col("v").min().alias("lo"),
                col("v").max().alias("hi"),
            ]
        )
    )
    assert_true(Bool(matched))
    var first = List[UInt32](length=GROUP_SLOTS, fill=UInt32.MAX)
    first[0] = 0
    first[9] = 1
    var counts = List[UInt32](length=GROUP_SLOTS * 2, fill=0)
    var states = List[UInt32](
        length=GROUP_SLOTS * 2 * GROUP_STATE_WORDS, fill=0
    )
    var extrema = List[Int64](length=GROUP_SLOTS * 2 * 2, fill=0)
    for slot in [0, 9]:
        for block in range(2):
            var index = slot * 2 + block
            var at = index * GROUP_STATE_WORDS
            var sum = Int128(3)
            var count = 1
            var low = Int64(3)
            var high = Int64(3)
            if slot == 0:
                sum = Int128(Int64.MAX) + 7 if block == 0 else -Int128(
                    Int64.MAX
                )
                count = 2 if block == 0 else 1
                low = Int64(7) if block == 0 else -Int64.MAX
                high = Int64.MAX if block == 0 else -Int64.MAX
            counts[index] = UInt32(count)
            states[at] = UInt32(count)
            states[at + 1] = UInt32(sum)
            states[at + 2] = UInt32(sum >> 32)
            states[at + 3] = UInt32(sum >> 64)
            states[at + 4] = 0
            states[at + 5] = 0
            extrema[index * 2] = low
            extrema[index * 2 + 1] = high
    var answer = finish_grouped(
        matched.value(), frame, 2, first, counts, states, extrema
    )
    assert_equal(answer.height(), 2)
    assert_true(answer.item(0, "k").is_null())
    assert_equal(answer.item(0, "s").int64(), Int64(7))
    assert_equal(answer.item(0, "n").int64(), Int64(3))
    assert_equal(answer.item(0, "avg").float64(), Float64(7) / 3)
    assert_equal(answer.item(0, "lo").int64(), -Int64.MAX)
    assert_equal(answer.item(0, "hi").int64(), Int64.MAX)
    assert_equal(answer.item(1, "k").int64(), Int64(9))
    assert_equal(answer.item(1, "s").int64(), Int64(6))
    var original_keys = frame.column("k").numeric[DType.int64]()
    for chunk in answer.column("k").chunks():
        assert_false(
            chunk.numeric[DType.int64]()._shares_buffers_with(original_keys)
        )


def test_metal_grouped_cost_properties_distinguish_cardinality_and_skew() raises:
    from dataframe_metal.group_properties import group_properties

    var uniform = List[Int32]()
    var distinct = List[Int32]()
    var skewed = List[Int32]()
    for i in range(256):
        uniform.append(Int32(i % 16))
        distinct.append(Int32(i))
        skewed.append(Int32(0 if i < 240 else i - 240))
    var low = DataFrame([Series("k", Column[Int32](uniform.copy()))])
    var high = DataFrame([Series("k", Column[Int32](distinct^))])
    var hot = DataFrame([Series("k", Column[Int32](skewed^))])
    var nulls = DataFrame(
        [
            Series(
                "k", Column[Int32](uniform^, List[Bool](length=256, fill=False))
            )
        ]
    )
    assert_equal(
        group_properties(low.lazy(), "k"), ":sample:v1:groups16:hot1:null0"
    )
    assert_equal(
        group_properties(high.lazy(), "k"), ":sample:v1:groups256:hot1:null0"
    )
    assert_equal(
        group_properties(hot.lazy(), "k"), ":sample:v1:groups16:hot8:null0"
    )
    assert_equal(
        group_properties(nulls.lazy(), "k"), ":sample:v1:groups1:hot8:null8"
    )


def test_metal_grouped_final_keys_do_not_retain_source_buffers() raises:
    from dataframe_metal.group_finish import finish_grouped

    for kind in range(2):
        var frame = DataFrame(
            [
                _numbers(
                    kind,
                    "k",
                    List[Int64]([Int64(3), 4]),
                    List[Bool]([True, False]),
                ),
                Series("v", Column[Int64]([1, 2])),
            ]
        )
        var matched = plan_grouped(
            frame.lazy().group_by("k").agg(col("v").count().alias("n"))
        )
        assert_true(Bool(matched))
        var first = List[UInt32]([UInt32(0), 1])
        var counts = List[UInt32]([UInt32(1), 1])
        var states = List[UInt32](length=64, fill=0)
        states[0] = 1
        states[32] = 1
        states[4] = UInt32.MAX
        states[36] = UInt32.MAX
        var answer = finish_grouped(
            matched.value(),
            frame,
            1,
            first,
            counts,
            states,
            List[Int64](length=4, fill=0),
        )
        assert_equal(answer.height(), 2)
        assert_true(answer.item(1, "k").is_null())
        var empty = finish_grouped(
            matched.value(),
            frame,
            0,
            List[UInt32](),
            List[UInt32](),
            List[UInt32](),
            List[Int64](),
        )
        assert_equal(empty.height(), 0)
        assert_equal(empty.column("k").dtype(), frame.column("k").dtype())
        for result in [answer^, empty^]:
            for chunk in result.column("k").chunks():
                if kind == 0:
                    assert_false(
                        chunk.numeric[DType.int32]()._shares_buffers_with(
                            frame.column("k").numeric[DType.int32]()
                        )
                    )
                else:
                    assert_false(
                        chunk.numeric[DType.int64]()._shares_buffers_with(
                            frame.column("k").numeric[DType.int64]()
                        )
                    )


def test_metal_grouped_layout_key_includes_the_group_key_column() raises:
    from dataframe_metal.calibration import shape_key

    var keys = Series("k", Column[Int32]([0, 1, 2, 3]))
    var values = Series("v", Column[Float32]([2, 3, 4, 5]))
    var flat = (
        DataFrame([keys.copy(), values.copy()])
        .lazy()
        .group_by("k")
        .agg(col("v").sum())
    )
    var chunked = (
        DataFrame(
            [
                Series._from_chunks([keys.slice(0, 1), keys.slice(1, 3)]),
                values.copy(),
            ]
        )
        .lazy()
        .group_by("k")
        .agg(col("v").sum())
    )
    var plan = plan_grouped(flat)
    assert_true(Bool(plan))
    var operator = grouped_key(plan.value())
    assert_false(shape_key(flat, operator) == shape_key(chunked, operator))


def test_metal_grouped_normalization_inlines_values_without_replacing_keys() raises:
    from dataframe_metal.region_plan import normalize_region

    var frame = DataFrame(
        [
            Series("k", Column[Int32]([0, 1, 0])),
            Series("v", Column[Float32]([2, 3, 4])),
        ]
    )
    var query = (
        frame.lazy()
        .with_columns([(col("v") * 2).alias("x")])
        .with_columns([(col("x") - 1).alias("y")])
        .group_by("k", maintain_order=True)
        .agg(col("y").sum())
    )
    var normalized = normalize_region(query)
    assert_true(Bool(normalized))
    assert_true(Bool(plan_grouped(normalized.value())))
    var changed_key = (
        frame.lazy()
        .with_columns([(col("k") + 1).alias("k")])
        .group_by("k")
        .agg(col("k").sum())
    )
    assert_false(Bool(normalize_region(changed_key)))


# requires: metal
def test_metal_grouped_reductions_null_keys_values_extrema_and_order() raises:
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    for kind in range(3):
        var key_kind = 0 if kind == 0 else 1
        var low = Int64(Int32.MIN) if key_kind == 0 else Int64.MIN
        var high = Int64(Int32.MAX) if key_kind == 0 else Int64.MAX
        var source = DataFrame(
            [
                _numbers(
                    key_kind,
                    "k",
                    List[Int64]([Int64(9), low, high, low, 0, high, 0, 9]),
                    List[Bool](
                        [True, True, True, True, False, True, False, True]
                    ),
                ),
                _numbers(
                    kind,
                    "v",
                    List[Int64]([Int64(99), 2, 3, 4, 5, 6, 7, 99]),
                    List[Bool](
                        [True, True, True, False, True, True, True, True]
                    ),
                ),
                Series("p", Column[Int32]([-1, 0, 1, 2, 3, 4, 5, -1])),
            ]
        )
        var sliced = source.slice(1, 6)
        var chunks = List[Series]()
        for name in ["k", "v", "p"]:
            var split = 2 if name == "k" else 3 if name == "v" else 4
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
                var offered = frame.lazy()
                if filtered:
                    offered = offered.filter(col("p") >= 1)
                var query = offered.group_by("k", maintain_order=True).agg(
                    [
                        col("v").sum().alias("s"),
                        col("v").count().alias("n"),
                        col("v").mean().alias("avg"),
                        col("v").min().alias("lo"),
                        col("v").max().alias("hi"),
                    ]
                )
                var all = List[Bool]([True, True, True])
                var nonempty = List[Bool]([True, not filtered, True])
                var expected = DataFrame(
                    [
                        _numbers(
                            key_kind,
                            "k",
                            List[Int64](
                                [
                                    high if filtered else low,
                                    low if filtered else high,
                                    0,
                                ]
                            ),
                            List[Bool]([True, True, False]),
                        ),
                        _numbers(
                            kind,
                            "s",
                            List[Int64](
                                [
                                    Int64(9 if filtered else 2),
                                    Int64(0 if filtered else 9),
                                    12,
                                ]
                            ),
                            all,
                        ),
                        Series(
                            "n",
                            Column[Int64](
                                [
                                    Int64(2 if filtered else 1),
                                    Int64(0 if filtered else 2),
                                    2,
                                ]
                            ),
                        ),
                        Series(
                            "avg",
                            Column[Float64](
                                [
                                    Float64(4.5 if filtered else 2.0),
                                    Float64(0.0 if filtered else 4.5),
                                    6,
                                ],
                                nonempty,
                            ),
                        ),
                        _numbers(
                            kind,
                            "lo",
                            List[Int64](
                                [
                                    Int64(3 if filtered else 2),
                                    Int64(0 if filtered else 3),
                                    5,
                                ]
                            ),
                            nonempty,
                        ),
                        _numbers(
                            kind,
                            "hi",
                            List[Int64](
                                [
                                    Int64(6 if filtered else 2),
                                    Int64(0 if filtered else 6),
                                    7,
                                ]
                            ),
                            nonempty,
                        ),
                    ]
                )
                for copied in [False, True]:
                    assert_true(
                        execute_timed(query, zero_copy=not copied)[0].equals(
                            expected
                        )
                    )
                assert_true(query.collect(accelerator=runtime).equals(expected))
                var count = offered.group_by("k", maintain_order=True).len()
                var counted = execute_timed(count)[0].copy()
                assert_equal(counted.height(), 3)
                assert_equal(counted.item(0, "len").int64(), Int64(2))
                assert_equal(
                    counted.item(1, "len").int64(), Int64(1 if filtered else 2)
                )
                assert_equal(counted.item(2, "len").int64(), Int64(2))


# requires: metal
def test_metal_grouped_four_values_eight_inputs_and_independent_key() raises:
    from tests.support.metal_outcomes import require_metal
    from tests.support.metal_matrix_reference import (
        matrix_frame,
        matrix_expressions,
        matrix_reference,
    )
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    var expected = matrix_reference(True)
    for fragmented in [False, True]:
        var frame = matrix_frame(fragmented)
        var query = (
            frame.lazy()
            .filter((col("p") > 0) | col("p").is_null())
            .group_by("k", maintain_order=True)
            .agg(matrix_expressions())
        )
        for copied in [False, True]:
            assert_true(
                execute_timed(query, zero_copy=not copied)[0].equals(expected)
            )
        assert_true(query.collect(accelerator=runtime).equals(expected))


# requires: metal
def test_metal_grouped_cardinality_limit_empty_inputs_and_count_rows() raises:
    from std.testing import assert_raises
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    for distinct in [256, 257, 4096]:
        var keys = List[Int32]()
        var values = List[Int64]()
        for row in range(distinct):
            keys.append(Int32(row))
            values.append(Int64(row + 1))
        var frame = DataFrame(
            [
                Series("k", Column[Int32](keys^)),
                Series("v", Column[Int64](values^)),
            ]
        )
        var query = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(
                [
                    col("v").sum().alias("s"),
                    col("v").count().alias("n"),
                    col("v").min().alias("lo"),
                    col("v").max().alias("hi"),
                ]
            )
        )
        if distinct == 256:
            var answer = execute_timed(query)[0].copy()
            assert_equal(answer.height(), 256)
            for row in range(256):
                assert_equal(answer.item(row, "k").int32(), Int32(row))
                assert_equal(answer.item(row, "s").int64(), Int64(row + 1))
                assert_equal(answer.item(row, "n").int64(), Int64(1))
        else:
            with assert_raises(contains="cardinality"):
                _ = execute_timed(query)
            same_outcome(query)
        var empty = (
            frame.slice(0, 0)
            .lazy()
            .group_by("k")
            .agg(
                [
                    col("v").sum(min_count=1).alias("s"),
                    col("v").mean().alias("avg"),
                    col("v").min().alias("lo"),
                    col("v").count().alias("n"),
                ]
            )
        )
        assert_equal(execute_timed(empty)[0].height(), 0)
        assert_true(
            empty.collect(accelerator=runtime).equals(
                empty.collect(engine="cpu")
            )
        )
    var nulls = DataFrame(
        [
            Series("k", Column[Int32]([0, 0, 0], [False, False, False])),
            Series("v", Column[Int64]([0, 0, 0], [False, False, False])),
        ]
    )
    var count = nulls.lazy().group_by("k").len()
    assert_equal(execute_timed(count)[0].item(0, "len").int64(), Int64(3))
    assert_true(execute_timed(count)[0].item(0, "k").is_null())


# requires: metal
def test_metal_grouped_faults_predicate_masks_and_ieee_adjudication() raises:
    from std.memory import bitcast
    from std.testing import assert_raises
    from tests.support.metal_outcomes import require_metal, same_outcome
    from dataframe_metal.handoff import execute_timed

    var runtime = require_metal()
    for kind in range(2):
        var value = Series(
            "v", Column[Int32]([Int32.MAX, 1])
        ) if kind == 0 else Series("v", Column[Int64]([Int64.MAX, 1]))
        var frame = DataFrame(
            [
                Series("k", Column[Int32]([0, 1])),
                value^,
                Series("p", Column[Int32]([0, 1])),
            ]
        )
        var expression = (col("v") + 1) * 2
        var fault = frame.lazy().group_by("k").agg(expression.count())
        with assert_raises(contains="checked-value"):
            _ = execute_timed(fault)
        same_outcome(fault)
        var safe = (
            frame.lazy()
            .filter(col("p") > 0)
            .group_by("k")
            .agg(expression.sum().alias("s"))
        )
        var answer = execute_timed(safe)[0].copy()
        assert_equal(answer.height(), 1)
        if kind == 0:
            assert_equal(answer.item(0, "s").int32(), Int32(4))
        else:
            assert_equal(answer.item(0, "s").int64(), Int64(4))
    var tiny = bitcast[DType.float32](UInt32(1))
    var small = DataFrame(
        [
            Series("k", Column[Int32]([0, 0])),
            Series("f", Column[Float32]([tiny, tiny])),
        ]
    )
    var value = (col("f") * 0.5) + 1
    var underflow = small.lazy().group_by("k").agg(value.sum())
    with assert_raises(contains="checked-value"):
        _ = execute_timed(underflow)
    same_outcome(underflow)
    var counted = small.lazy().group_by("k").agg(value.count().alias("n"))
    assert_equal(execute_timed(counted)[0].item(0, "n").int64(), Int64(2))
    var nan = bitcast[DType.float32](UInt32(0x7FC00017))
    var zero = bitcast[DType.float32](UInt32(0x80000000))
    var ieee = DataFrame(
        [
            Series("k", Column[Int32]([0, 0, 1, 1])),
            Series("f", Column[Float32]([nan, nan, zero, 0])),
        ]
    )
    var extrema = (
        ieee.lazy()
        .group_by("k")
        .agg([col("f").min().alias("lo"), col("f").max().alias("hi")])
    )
    with assert_raises(contains="CPU"):
        _ = execute_timed(extrema)
    same_outcome(extrema)
    var count = (
        ieee.lazy()
        .group_by("k", maintain_order=True)
        .agg(col("f").count().alias("n"))
    )
    var answer = execute_timed(count)[0].copy()
    assert_equal(answer.item(0, "n").int64(), Int64(2))
    assert_equal(answer.item(1, "n").int64(), Int64(2))


# requires: metal
def test_metal_grouped_shared_key_mapping_and_scratch_budget_lifetime() raises:
    from std.testing import assert_raises
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.group_plan import plan_grouped
    from dataframe_metal.group_reduce import _scan_grouped
    from dataframe_metal.accelerator.gpu.query_scope import (
        Scope,
        shared_service,
    )
    from dataframe_metal.accelerator.gpu.reduction_inputs import HandoffTimings

    _ = require_metal()
    var frame = DataFrame([Series("k", Column[Int32]([0, 1, 0, 1]))])
    var matched = plan_grouped(frame.lazy().group_by("k").len())
    assert_true(Bool(matched))
    var scope = Scope(service=shared_service())
    var timings = HandoffTimings()
    var result = _scan_grouped[1, 0, 1](
        matched.value(), frame, scope, False, timings
    )
    assert_equal(scope.transfers()[0], 16)
    assert_equal(scope.info()[0], 0)
    var peak = scope.info()[1]
    assert_true(peak > 16)
    scope.mark_success()
    _ = scope^
    var limited = Scope(budget=peak - 1, service=shared_service())
    with assert_raises(contains="memory budget"):
        _ = _scan_grouped[1, 0, 1](
            matched.value(), frame, limited, False, timings
        )
    assert_equal(limited.info()[0], 0)
    _ = result^
