"""Exact native distinct equivalence classes and bounded schema admission."""
# contract-group: execution.backends
from std.memory import bitcast
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, Expr, Series, col
from dataframe_metal.accelerator.gpu.distinct_value import (
    distinct_canonical,
    distinct_same,
    distinct_hash,
)
from dataframe_metal.distinct_plan import plan_distinct
from dataframe_metal.distinct_layout import (
    DistinctLayout,
    MAX_DISTINCT_CAPACITY,
)


def test_metal_distinct_float_classes_match_cpu_equality_with_nan_and_null() raises:
    var values: List[UInt32] = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x3F800000,
        0xBF800000,
        0x7F800000,
        0xFF800000,
        0x7F800001,
        0x7FC12345,
        0xFFC54321,
    ]
    for a in values:
        for b in values:
            var x = bitcast[DType.float32](a)
            var y = bitcast[DType.float32](b)
            for av in [False, True]:
                for bv in [False, True]:
                    var expected = (
                        av == bv if not av
                        or not bv else x == y
                        or (isnan(x) and isnan(y))
                    )
                    assert_equal(
                        distinct_same(UInt64(a), av, UInt64(b), bv, 2), expected
                    )
                    if expected:
                        assert_equal(
                            distinct_hash(UInt64(a), av, 2, 127),
                            distinct_hash(UInt64(b), bv, 2, 127),
                        )
    assert_equal(distinct_canonical(0x80000000, 2), UInt64(0))
    assert_equal(distinct_canonical(0xFFC54321, 2), UInt64(0x7FC00000))
    assert_false(distinct_same(0, False, 0, True, 2))


def test_metal_distinct_integer_classes_preserve_wide_values_and_resource_bounds() raises:
    var values: List[Int64] = [
        Int64.MIN,
        -1,
        0,
        1,
        Int64(1 << 60),
        Int64(1 << 60) + 1,
        Int64.MAX,
    ]
    for a in values:
        for b in values:
            assert_equal(
                distinct_same(
                    bitcast[DType.uint64](a),
                    True,
                    bitcast[DType.uint64](b),
                    True,
                    1,
                ),
                a == b,
            )
    assert_equal(distinct_canonical(255, 3), UInt64(1))
    var small = DistinctLayout(512, 4, 16384)
    small.check()
    var large = DistinctLayout(131072, 4, MAX_DISTINCT_CAPACITY)
    large.check()
    assert_true(large.words() * 4 < 40000000)
    assert_true(small.unique_offset() >= small.nulls_offset() + 512 * 4)


def test_metal_distinct_plans_share_null_modes_and_support_native_tuple_groups() raises:
    var frame = DataFrame(
        [
            Series("n", Column[Int32]([1, 2, 3])),
            Series("w", Column[Int64]([1, 2, 3])),
            Series("f", Column[Float32]([1, 2, 3])),
            Series("b", Column[Bool]([True, False, True])),
            Series("k", Column[Int32]([1, 2, 1])),
            Series("j", Column[Int64]([0, 1, 0])),
        ]
    )
    var outputs = List[Expr]()
    for name in ["n", "w", "f", "b"]:
        outputs.extend(
            [
                col(name).n_unique().alias(name + "all"),
                col(name).n_unique(ignore_nulls=True).alias(name + "nonnull"),
                col(name).drop_nulls().n_unique().alias(name + "dropped"),
            ]
        )
    for grouped in [False, True]:
        var selected = frame.lazy().filter(col("b") | (col("n") > 0))
        var query = selected.group_by(["k", "j"]).agg(
            outputs
        ) if grouped else selected.select_exprs(outputs)
        var plan = plan_distinct(query)
        assert_true(Bool(plan))
        assert_equal(len(plan.value().fields), 4)
        assert_equal(len(plan.value().outputs), 12)
        assert_equal(Bool(plan.value().keys), grouped)
        for field in range(4):
            assert_false(plan.value().outputs[field * 3].ignore_nulls)
            assert_true(plan.value().outputs[field * 3 + 1].ignore_nulls)
            assert_true(plan.value().outputs[field * 3 + 2].ignore_nulls)
    assert_false(
        Bool(plan_distinct(frame.lazy().select((col("n") + 1).n_unique())))
    )


def test_metal_distinct_cpu_oracle_keeps_wide_integers_nan_zero_and_null_classes() raises:
    var base = Int64(1 << 60)
    var integers = DataFrame(
        [
            Series(
                "w",
                Column[Int64](
                    [base, base + 1, base + 2, base, 0],
                    [True, True, True, True, False],
                ),
            )
        ]
    )
    var observed = (
        integers.lazy()
        .select_exprs(
            [
                col("w").n_unique().alias("all"),
                col("w").n_unique(ignore_nulls=True).alias("nonnull"),
            ]
        )
        .collect(engine="cpu")
    )
    assert_equal(observed.item(0, "all").int64(), Int64(4))
    assert_equal(observed.item(0, "nonnull").int64(), Int64(3))
    var floats = List[Float32]()
    for raw in [
        UInt32(0),
        UInt32(0x80000000),
        UInt32(0x7FC12345),
        UInt32(0xFFC54321),
        UInt32(1),
        UInt32(0x80000001),
        UInt32(0x7F800000),
        UInt32(0xFF800000),
        UInt32(0x3F800000),
        UInt32(0),
    ]:
        floats.append(bitcast[DType.float32](raw))
    var frame = DataFrame(
        [
            Series(
                "f",
                Column[Float32](
                    floats^,
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                    ],
                ),
            )
        ]
    )
    var counts = (
        frame.lazy()
        .select_exprs(
            [
                col("f").n_unique().alias("all"),
                col("f").n_unique(ignore_nulls=True).alias("nonnull"),
            ]
        )
        .collect(engine="cpu")
    )
    assert_equal(counts.item(0, "all").int64(), Int64(8))
    assert_equal(counts.item(0, "nonnull").int64(), Int64(7))
