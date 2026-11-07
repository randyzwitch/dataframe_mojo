"""Independent exact-rational fixtures for native Metal moment finishing.

Fixture sums and squared sums use radix-2**32 units of 2**-149. Expected
Float64 bits come from Python Fraction, not a floating moment recurrence.
Public GPU routing and performance are validated separately.
"""
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe_metal.moment_finish import finish_variance, MomentMagnitude


def test_metal_moment_exact_cancellation_ddof_and_full_float32_range() raises:
    # signed_zero
    var sums_signed_zero = List[Int128]([Int128(0), 0, 0, 0, 0, 0, 0, 0, 0])
    var squares_signed_zero = List[Int128](
        [Int128(0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    var answer_signed_zero_0 = finish_variance(
        sums_signed_zero, squares_signed_zero, 2, 0, -149, False
    ).value()
    var bits_signed_zero_0 = bitcast[DType.uint64](answer_signed_zero_0)
    assert_true(abs(Int128(bits_signed_zero_0) - Int128(0)) <= 4)
    var answer_signed_zero_1 = finish_variance(
        sums_signed_zero, squares_signed_zero, 2, 0, -149, True
    ).value()
    var bits_signed_zero_1 = bitcast[DType.uint64](answer_signed_zero_1)
    assert_true(abs(Int128(bits_signed_zero_1) - Int128(0)) <= 4)
    # single
    var sums_single = List[Int128]([Int128(0), 0, 0, 0, 2097152, 0, 0, 0, 0])
    var squares_single = List[Int128](
        [Int128(0), 0, 0, 0, 0, 0, 0, 0, 0, 1024, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    var answer_single_0 = finish_variance(
        sums_single, squares_single, 1, 0, -149, False
    ).value()
    var bits_single_0 = bitcast[DType.uint64](answer_single_0)
    assert_true(abs(Int128(bits_single_0) - Int128(0)) <= 4)
    var answer_single_1 = finish_variance(
        sums_single, squares_single, 1, 0, -149, True
    ).value()
    var bits_single_1 = bitcast[DType.uint64](answer_single_1)
    assert_true(abs(Int128(bits_single_1) - Int128(0)) <= 4)
    # small_subnormals
    var sums_small_subnormals = List[Int128](
        [Int128(0), 0, 0, 0, 0, 0, 0, 0, 0]
    )
    var squares_small_subnormals = List[Int128](
        [Int128(2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    var answer_small_subnormals_0 = finish_variance(
        sums_small_subnormals, squares_small_subnormals, 3, 1, -149, False
    ).value()
    var bits_small_subnormals_0 = bitcast[DType.uint64](
        answer_small_subnormals_0
    )
    assert_true(
        abs(Int128(bits_small_subnormals_0) - Int128(3265109729843609600)) <= 4
    )
    var answer_small_subnormals_1 = finish_variance(
        sums_small_subnormals, squares_small_subnormals, 3, 1, -149, True
    ).value()
    var bits_small_subnormals_1 = bitcast[DType.uint64](
        answer_small_subnormals_1
    )
    assert_true(
        abs(Int128(bits_small_subnormals_1) - Int128(3936146074321813504)) <= 4
    )
    # largest_opposite
    var sums_largest_opposite = List[Int128](
        [Int128(0), 0, 0, 0, 0, 0, 0, 0, 0]
    )
    var squares_largest_opposite = List[Int128](
        [
            Int128(0),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            134217728,
            4293918720,
            2047,
        ]
    )
    var answer_largest_opposite_0 = finish_variance(
        sums_largest_opposite, squares_largest_opposite, 2, 0, -149, False
    ).value()
    var bits_largest_opposite_0 = bitcast[DType.uint64](
        answer_largest_opposite_0
    )
    assert_true(
        abs(Int128(bits_largest_opposite_0) - Int128(5760103922333122592)) <= 4
    )
    var answer_largest_opposite_1 = finish_variance(
        sums_largest_opposite, squares_largest_opposite, 2, 0, -149, True
    ).value()
    var bits_largest_opposite_1 = bitcast[DType.uint64](
        answer_largest_opposite_1
    )
    assert_true(
        abs(Int128(bits_largest_opposite_1) - Int128(5183643170566569984)) <= 4
    )
    # largest_neighbours
    var sums_largest_neighbours = List[Int128](
        [Int128(0), 0, 0, 0, 0, 0, 0, 2684354560, 4194303]
    )
    var squares_largest_neighbours = List[Int128](
        [
            Int128(0),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            335544320,
            4293394432,
            2047,
        ]
    )
    var answer_largest_neighbours_0 = finish_variance(
        sums_largest_neighbours, squares_largest_neighbours, 2, 1, -149, False
    ).value()
    var bits_largest_neighbours_0 = bitcast[DType.uint64](
        answer_largest_neighbours_0
    )
    assert_true(
        abs(Int128(bits_largest_neighbours_0) - Int128(5539427541665710080))
        <= 4
    )
    var answer_largest_neighbours_1 = finish_variance(
        sums_largest_neighbours, squares_largest_neighbours, 2, 1, -149, True
    ).value()
    var bits_largest_neighbours_1 = bitcast[DType.uint64](
        answer_largest_neighbours_1
    )
    assert_true(
        abs(Int128(bits_largest_neighbours_1) - Int128(5072918632464333773))
        <= 4
    )
    # large_offset
    var sums_large_offset = List[Int128](
        [Int128(0), 0, 0, 0, 12582912, 24576, 0, 0, 0]
    )
    var squares_large_offset = List[Int128](
        [
            Int128(0),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            20480,
            201326640,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    var answer_large_offset_0 = finish_variance(
        sums_large_offset, squares_large_offset, 3, 1, -149, False
    ).value()
    var bits_large_offset_0 = bitcast[DType.uint64](answer_large_offset_0)
    assert_true(
        abs(Int128(bits_large_offset_0) - Int128(4616189618054758400)) <= 4
    )
    var answer_large_offset_1 = finish_variance(
        sums_large_offset, squares_large_offset, 3, 1, -149, True
    ).value()
    var bits_large_offset_1 = bitcast[DType.uint64](answer_large_offset_1)
    assert_true(
        abs(Int128(bits_large_offset_1) - Int128(4611686018427387904)) <= 4
    )
    # cancelling_scales
    var sums_cancelling_scales = List[Int128](
        [Int128(1), 0, 0, 0, 0, 0, 0, 0, 0]
    )
    var squares_cancelling_scales = List[Int128](
        [
            Int128(1),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            134217728,
            4293918720,
            2047,
        ]
    )
    var answer_cancelling_scales_0 = finish_variance(
        sums_cancelling_scales, squares_cancelling_scales, 3, 2, -149, False
    ).value()
    var bits_cancelling_scales_0 = bitcast[DType.uint64](
        answer_cancelling_scales_0
    )
    assert_true(
        abs(Int128(bits_cancelling_scales_0) - Int128(5764607521960493088)) <= 4
    )
    var answer_cancelling_scales_1 = finish_variance(
        sums_cancelling_scales, squares_cancelling_scales, 3, 2, -149, True
    ).value()
    var bits_cancelling_scales_1 = bitcast[DType.uint64](
        answer_cancelling_scales_1
    )
    assert_true(
        abs(Int128(bits_cancelling_scales_1) - Int128(5185508622768971110)) <= 4
    )
    # one_rare_value
    var sums_one_rare_value = List[Int128](
        [Int128(0), 0, 0, 0, 0, 0, 0, 0, 9007198715772928]
    )
    var squares_one_rare_value = List[Int128](
        [
            Int128(0),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            134217728,
            67108864,
            4398045985792,
        ]
    )
    var answer_one_rare_value_0 = finish_variance(
        sums_one_rare_value, squares_one_rare_value, 4294967295, 1, -149, False
    ).value()
    var bits_one_rare_value_0 = bitcast[DType.uint64](answer_one_rare_value_0)
    assert_true(
        abs(Int128(bits_one_rare_value_0) - Int128(5399815953218273280)) <= 4
    )
    var answer_one_rare_value_1 = finish_variance(
        sums_one_rare_value, squares_one_rare_value, 4294967295, 1, -149, True
    ).value()
    var bits_one_rare_value_1 = bitcast[DType.uint64](answer_one_rare_value_1)
    assert_true(
        abs(Int128(bits_one_rare_value_1) - Int128(5003499186009145344)) <= 4
    )
    # wide_population
    var sums_wide_population = List[Int128](
        [Int128(0), 0, 0, 0, -2097152, 0, 0, 0, 0]
    )
    var squares_wide_population = List[Int128](
        [
            Int128(0),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            4294966272,
            1023,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    var answer_wide_population_0 = finish_variance(
        sums_wide_population,
        squares_wide_population,
        4294967295,
        0,
        -149,
        False,
    ).value()
    var bits_wide_population_0 = bitcast[DType.uint64](answer_wide_population_0)
    assert_true(
        abs(Int128(bits_wide_population_0) - Int128(4607182418800017408)) <= 4
    )
    var answer_wide_population_1 = finish_variance(
        sums_wide_population, squares_wide_population, 4294967295, 0, -149, True
    ).value()
    var bits_wide_population_1 = bitcast[DType.uint64](answer_wide_population_1)
    assert_true(
        abs(Int128(bits_wide_population_1) - Int128(4607182418800017408)) <= 4
    )
    # negative_ddof
    var sums_negative_ddof = List[Int128](
        [Int128(0), 0, 0, 0, 14680064, 0, 0, 0, 0]
    )
    var squares_negative_ddof = List[Int128](
        [Int128(0), 0, 0, 0, 0, 0, 0, 0, 0, 11264, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    var answer_negative_ddof_0 = finish_variance(
        sums_negative_ddof, squares_negative_ddof, 5, -3, -149, False
    ).value()
    var bits_negative_ddof_0 = bitcast[DType.uint64](answer_negative_ddof_0)
    assert_true(
        abs(Int128(bits_negative_ddof_0) - Int128(4594572339843380019)) <= 4
    )
    var answer_negative_ddof_1 = finish_variance(
        sums_negative_ddof, squares_negative_ddof, 5, -3, -149, True
    ).value()
    var bits_negative_ddof_1 = bitcast[DType.uint64](answer_negative_ddof_1)
    assert_true(
        abs(Int128(bits_negative_ddof_1) - Int128(4600648566459822664)) <= 4
    )
    # final_degree
    var sums_final_degree = List[Int128](
        [Int128(0), 0, 0, 0, 14680064, 0, 0, 0, 0]
    )
    var squares_final_degree = List[Int128](
        [Int128(0), 0, 0, 0, 0, 0, 0, 0, 0, 11264, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    var answer_final_degree_0 = finish_variance(
        sums_final_degree, squares_final_degree, 5, 4, -149, False
    ).value()
    var bits_final_degree_0 = bitcast[DType.uint64](answer_final_degree_0)
    assert_true(
        abs(Int128(bits_final_degree_0) - Int128(4608083138725491507)) <= 4
    )
    var answer_final_degree_1 = finish_variance(
        sums_final_degree, squares_final_degree, 5, 4, -149, True
    ).value()
    var bits_final_degree_1 = bitcast[DType.uint64](answer_final_degree_1)
    assert_true(
        abs(Int128(bits_final_degree_1) - Int128(4607612265384412274)) <= 4
    )
    # all_equal_max_count
    var sums_all_equal_max_count = List[Int128](
        [Int128(0), 0, 0, 0, 0, 0, 0, 536870912, 9007198715772928]
    )
    var squares_all_equal_max_count = List[Int128](
        [
            Int128(0),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            4227858432,
            67633151,
            4398045985792,
        ]
    )
    var answer_all_equal_max_count_0 = finish_variance(
        sums_all_equal_max_count,
        squares_all_equal_max_count,
        4294967295,
        1,
        -149,
        False,
    ).value()
    var bits_all_equal_max_count_0 = bitcast[DType.uint64](
        answer_all_equal_max_count_0
    )
    assert_true(abs(Int128(bits_all_equal_max_count_0) - Int128(0)) <= 4)
    var answer_all_equal_max_count_1 = finish_variance(
        sums_all_equal_max_count,
        squares_all_equal_max_count,
        4294967295,
        1,
        -149,
        True,
    ).value()
    var bits_all_equal_max_count_1 = bitcast[DType.uint64](
        answer_all_equal_max_count_1
    )
    assert_true(abs(Int128(bits_all_equal_max_count_1) - Int128(0)) <= 4)


def test_metal_moment_empty_invalid_and_non_finite_adjudication() raises:
    var empty = List[Int128]()
    assert_true(not finish_variance(empty, empty, 0, 0, -149))
    assert_true(not finish_variance(empty, empty, 1, 1, -149))
    assert_equal(finish_variance(empty, empty, 0, -1, -149).value(), Float64(0))
    # Insufficient values stay null even when a selected value was non-finite.
    assert_true(not finish_variance(empty, empty, 1, 1, -149, non_finite=True))
    with assert_raises(contains="non-finite"):
        _ = finish_variance(empty, empty, 2, 1, -149, non_finite=True)
    with assert_raises(contains="counter range"):
        _ = finish_variance(empty, empty, 4294967296, 1, -149)
    with assert_raises(contains="negative variance"):
        _ = finish_variance(List[Int128]([Int128(2)]), empty, 1, 0, 0)
    with assert_raises(contains="nonnegative"):
        _ = finish_variance(empty, List[Int128]([Int128(-1)]), 1, 0, 0)
    with assert_raises(contains="scale"):
        _ = finish_variance(empty, empty, 2, 0, -150)
    with assert_raises(contains="ddof"):
        _ = finish_variance(empty, empty, 2, Int(Int64.MIN), 0)


def test_metal_moment_signed_carries_and_integer_units() raises:
    var a = MomentMagnitude.from_bins(List[Int128]([Int128(-1), 1]))
    var b = MomentMagnitude.from_bins(List[Int128]([Int128(4294967295)]))
    assert_equal(a.compare(b), 0)
    var product = a.multiply(b)
    assert_equal(product.scaled(0), Float64(4294967295) * Float64(4294967295))
    var difference = product.subtract(product)
    assert_equal(difference.scaled(-298), Float64(0))
    # 1,2,3 have exact sample variance 1, population variance 2/3.
    var sums = List[Int128]([Int128(6)])
    var squares = List[Int128]([Int128(14)])
    assert_equal(finish_variance(sums, squares, 3, 1, 0).value(), Float64(1))
    assert_equal(
        finish_variance(sums, squares, 3, 0, 0).value(), Float64(2) / Float64(3)
    )
    assert_equal(
        finish_variance(sums, squares, 3, 1, 0, True).value(), Float64(1)
    )


def test_metal_moment_square_words_cover_every_float32_exponent() raises:
    from dataframe_metal.accelerator.gpu.moment_squares import (
        float32_square_words,
    )

    for exponent in range(255):
        for fraction in [
            UInt32(0),
            UInt32(1),
            UInt32(0x3FFFFF),
            UInt32(0x7FFFFF),
        ]:
            for sign in [UInt32(0), UInt32(0x80000000)]:
                var raw = (UInt32(exponent) << 23) | fraction | sign
                var actual = float32_square_words(raw)
                var significand = Int128(fraction)
                if exponent:
                    significand += Int128(1) << 23
                var shift = 2 * max(0, exponent - 1)
                var expected = (significand * significand) << Int128(shift % 32)
                var reconstructed = (
                    Int128(actual[1])
                    + (Int128(actual[2]) << 32)
                    + (Int128(actual[3]) << 64)
                )
                assert_equal(actual[0], UInt32(shift // 32))
                assert_equal(reconstructed, expected)
                assert_true(not actual[4])
    for raw in [
        UInt32(0x7F800000),
        UInt32(0xFF800000),
        UInt32(0x7FC00017),
        UInt32(0xFF800001),
    ]:
        assert_true(float32_square_words(raw)[4])


def test_metal_moment_integer_rounding_and_square_words() raises:
    from dataframe_metal.accelerator.gpu.moment_squares import (
        integer_moment_magnitude,
        integer_square_words,
    )

    var cases = List[Int64](
        [
            Int64.MIN,
            Int64.MIN + 1,
            Int64.MAX,
            Int64.MAX - 1,
            -9007199254740995,
            -9007199254740993,
            -9007199254740991,
            -2147483648,
            -1,
            0,
            1,
            2147483647,
            9007199254740991,
            9007199254740993,
            9007199254740995,
        ]
    )
    for value in cases:
        var raw = bitcast[DType.uint64](value)
        var magnitude = UInt64(0) - raw if value < 0 else raw
        var actual = integer_moment_magnitude(value)
        var expected = Float64(magnitude).cast[DType.uint64]()
        assert_equal(actual[0], expected)
        assert_equal(actual[1], value < 0)
    var state = UInt64(0x9E3779B97F4A7C15)
    for i in range(2048):
        state = state * UInt64(6364136223846793005) + 1
        var value = UInt64.MAX if i == 0 else (UInt64(0) if i == 1 else state)
        var actual = integer_square_words(value)
        var reconstructed = (
            UInt128(actual[0])
            + (UInt128(actual[1]) << 32)
            + (UInt128(actual[2]) << 64)
            + (UInt128(actual[3]) << 96)
        )
        assert_equal(reconstructed, UInt128(value) * UInt128(value))
        # Also compare randomized signed casts with the CPU's conversion.
        var signed = bitcast[DType.int64](state)
        var magnitude = UInt64(0) - state if signed < 0 else state
        assert_equal(
            integer_moment_magnitude(signed)[0],
            Float64(magnitude).cast[DType.uint64](),
        )


def test_metal_moment_planner_shares_values_but_preserves_each_ddof() raises:
    from dataframe import Column, DataFrame, Series, col
    from dataframe.expr import VAR, STD, SUM
    from dataframe_metal.moment_plan import plan_moments, moment_key

    var frame = DataFrame(
        [
            Series("i", Column[Int32]([1, 2, 3])),
            Series("f", Column[Float32]([2, 3, 4])),
        ]
    )
    var value = (col("i") * 2) - 1
    var query = (
        frame.lazy()
        .filter((col("i") > 0) & col("f").is_not_null())
        .select_exprs(
            [
                value.var(0).alias("population"),
                value.std(1).alias("sample_std"),
                value.var(2).alias("second_ddof"),
                value.sum().alias("sum"),
                col("f").mean().alias("float_mean"),
            ]
        )
    )
    var matched = plan_moments(query)
    assert_true(Bool(matched))
    ref plan = matched.value()
    assert_true(not Bool(plan.grouped))
    assert_equal(len(plan.selected.bundle.fields), 2)
    assert_equal(plan.selected.bundle.fields[0].dtype.name(), "float32")
    assert_equal(plan.outputs[0].field, 1)
    assert_equal(plan.outputs[0].op, VAR)
    assert_equal(plan.outputs[1].op, STD)
    assert_equal(plan.outputs[2].parameter, 2)
    assert_equal(plan.outputs[3].op, SUM)
    assert_true(not plan.needs_moments[0])
    assert_true(plan.needs_moments[1])
    for i in range(4):
        assert_equal(plan.outputs[i].field, 1)
    assert_equal(plan.outputs[0].parameter, 0)
    assert_equal(plan.outputs[1].parameter, 1)
    # Matching must not rewrite the caller's variance to its internal mean.
    assert_equal(
        query._plan._nodes[len(query._plan._nodes) - 1]
        .exprs[0]
        ._nodes[len(value._nodes)]
        .op,
        VAR,
    )
    assert_true(moment_key(plan).startswith("moments:v1:"))
    assert_true(not Bool(plan_moments(frame.lazy().select(value.var(-1)))))
    var changed = plan_moments(frame.lazy().select(value.var(1)))
    var original = plan_moments(frame.lazy().select(value.var(0)))
    assert_true(moment_key(changed.value()) != moment_key(original.value()))
    assert_true(not Bool(plan_moments(frame.lazy().select(col("i").sum()))))


def test_metal_moment_planner_grouped_and_row_local_values() raises:
    from dataframe import Column, DataFrame, Series, col
    from dataframe_metal.moment_plan import plan_moments

    var frame = DataFrame(
        [
            Series("k", Column[Int32]([1, 1, 2])),
            Series("j", Column[Int64]([2, 3, 4])),
            Series("f", Column[Float32]([2, 3, 4])),
            Series("d", Column[Float64]([1, 2, 3])),
        ]
    )
    var grouped = plan_moments(
        frame.lazy()
        .group_by(["k", "j"])
        .agg(
            [
                col("f").var(1).alias("v"),
                col("f").std(0).alias("s"),
                col("f").len().alias("rows"),
            ]
        )
    )
    assert_true(Bool(grouped))
    assert_true(Bool(grouped.value().grouped))
    assert_equal(len(grouped.value().grouped.value().keys.names), 2)
    assert_equal(len(grouped.value().selected.bundle.fields), 1)
    assert_equal(grouped.value().outputs[2].field, -1)
    assert_equal(grouped.value().outputs[0].parameter, 1)
    assert_equal(grouped.value().outputs[1].parameter, 0)
    var row_local = plan_moments(
        frame.lazy()
        .with_columns([(col("f") * 2).alias("x")])
        .select_exprs([col("x").var().alias("v"), col("x").sum().alias("sum")])
    )
    assert_true(Bool(row_local))
    assert_equal(len(row_local.value().selected.bundle.fields), 1)
    assert_true(not Bool(plan_moments(frame.lazy().select(col("d").var()))))
    assert_true(
        not Bool(plan_moments(frame.lazy().select((col("k") / 2).var())))
    )
