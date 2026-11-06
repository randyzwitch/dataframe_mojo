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
                                    Float64(4.5 if filtered else 2),
                                    Float64(0 if filtered else 4.5),
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
