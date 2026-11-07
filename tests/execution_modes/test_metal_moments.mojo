"""Independent exact-rational fixtures for native Metal moment finishing.

Fixture sums and squared sums use radix-2**32 units of 2**-149. Expected
Float64 bits come from Python Fraction, not a floating moment recurrence.
Public GPU routing and performance are validated separately.
"""
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe_metal.moment_finish import finish_variance, MomentMagnitude
from dataframe import DataFrame, LazyFrame


def test_metal_moments_cpu_compatibility_adjudicates_sensitive_centering() raises:
    from dataframe_metal.moment_compatibility import (
        require_cpu_moment_compatibility,
    )

    # Exact population variance of [2**60, 2**60+256] is 16384, while
    # Mojo CPU, DuckDB and Polars return 32768. Preserve CPU completion.
    with assert_raises(contains="variance centering needs CPU adjudication"):
        require_cpu_moment_compatibility(
            List[Int128]([Int128(256), Int128(1) << 29]), 2, 0, 0, 16384
        )
    # A constant wide integer can also expose CPU block-mean rounding.
    with assert_raises(contains="variance centering needs CPU adjudication"):
        require_cpu_moment_compatibility(
            List[Int128]([Int128(0), Int128(1) << 35]), 128, 0, 0, 0
        )
    # Ordinary spread, exact narrow constants and singleton/null cases
    # retain GPU completion, including nonzero mean and ddof conversion.
    require_cpu_moment_compatibility(List[Int128]([Int128(4)]), 2, 0, 0, 1)
    require_cpu_moment_compatibility(List[Int128]([Int128(4)]), 2, 1, 0, 2)
    require_cpu_moment_compatibility(List[Int128]([Int128(128)]), 128, 0, 0, 0)
    require_cpu_moment_compatibility(List[Int128]([Int128(0)]), 1, 0, 0, 0)
    require_cpu_moment_compatibility(List[Int128]([Int128(0)]), 0, 0, 0, 0)
    # Float32 subnormal values [2**-149, 3*2**-149] are well conditioned.
    require_cpu_moment_compatibility(
        List[Int128]([Int128(4)]),
        2,
        0,
        -149,
        bitcast[DType.float64](UInt64(725) << 52),
    )


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
    assert_true(moment_key(plan).startswith("moments:v2:cpu:"))
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


def test_metal_moment_narrow_square_window_and_retry_flags() raises:
    from dataframe_metal.accelerator.gpu.moment_squares import SquareBins

    assert_equal(SquareBins[4, False].WIDTH, 32)
    assert_equal(SquareBins[4].WIDTH, 128)
    var raw = List[UInt32](
        [
            UInt32(0),
            0x80000000,
            1,
            0x25000000,
            0x24800000,
            0x3F800000,
            0xBF800000,
            0x40600000,
            0x55000000,
            0x7F7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0x25000001,
        ]
    )
    for i in range(len(raw)):
        var state = SquareBins[1, False]()
        state.add[0](raw[i])
        assert_equal(state.non_finite[0], UInt32(i >= 10 and i <= 12))
        assert_equal(
            state.outside[0], UInt32(i == 2 or i == 4 or i == 9 or i == 13)
        )
        for bin in range(8):
            var expected = Int64(0)
            if bin == 0 and (i == 3 or i == 13):
                expected = 1
            if bin == 3 and (i == 5 or i == 6):
                expected = 1024
            if bin == 3 and i == 7:
                expected = 12544
            if bin == 6 and i == 8:
                expected = 1
            assert_equal(state.words[bin], expected)
    var fields = SquareBins[4, False]()
    fields.add[0](UInt32(0x3F800000))
    fields.add[1](UInt32(0x40600000))
    fields.add[2](UInt32(0x24800000))
    fields.add[3](UInt32(0x7F800000))
    assert_equal(fields.words[3], Int64(1024))
    assert_equal(fields.words[11], Int64(12544))
    assert_equal(fields.outside[2], UInt32(1))
    assert_equal(fields.non_finite[3], UInt32(1))
    assert_equal(fields.largest[0], UInt32(0x3F800000))
    assert_equal(fields.largest[1], UInt32(0x40600000))
    assert_equal(fields.largest[3], UInt32(0x7F800000))


def test_metal_moment_math_finish_preserves_shared_outputs_and_typed_nulls() raises:
    from std.math import sqrt
    from dataframe import Column, DataFrame, Series, col
    from dataframe_metal.moment_plan import plan_moments, moment_output_sources
    from dataframe_metal.moment_reduce import (
        finish_global_moments_math,
        moment_masks,
    )
    from dataframe_metal.composition_reduce import ComposedScan

    for floating in [False, True]:
        var series = Series("v", Column[Int64]([1, 3]))
        if floating:
            series = Series("v", Column[Float32]([1, 3]))
        var frame = DataFrame([series^])
        var matched = plan_moments(
            frame.lazy().select_exprs(
                [
                    col("v").std(0).alias("population_std"),
                    col("v").var(0).alias("population"),
                    col("v").var(1).alias("sample"),
                    col("v").std(1).alias("deviation"),
                    col("v").sum().alias("total"),
                    col("v").mean().alias("average"),
                    col("v").count().alias("count"),
                    col("v").var(0).alias("population_alias"),
                ]
            )
        )
        assert_true(Bool(matched))
        ref plan = matched.value()
        assert_equal(
            moment_output_sources(plan.outputs),
            List[Int]([0, 0, 2, 2, 4, 5, 6, 0]),
        )
        assert_equal(moment_masks(plan), (UInt32(1), UInt32(1)))
        var data = List[Int128](length=28, fill=0)
        if floating:
            data[4] = 8388608
            data[18] = 10240
        else:
            data[0] = 4
            data[9] = 10
        var scan = ComposedScan(
            List[Int128]([Int128(4)]),
            List[Float64]([Float64(4)]),
            List[Int]([2]),
            List[Int64](),
            List[List[Int128]]([data^]),
            List[UInt64]([UInt64(3)]),
        )
        var answer = finish_global_moments_math(plan, scan)
        assert_equal(answer.item(0, "population").float64(), Float64(1))
        assert_equal(answer.item(0, "population_std").float64(), Float64(1))
        assert_equal(answer.item(0, "population_alias").float64(), Float64(1))
        assert_equal(answer.item(0, "sample").float64(), Float64(2))
        assert_equal(answer.item(0, "deviation").float64(), sqrt(Float64(2)))
        assert_equal(answer.item(0, "average").float64(), Float64(2))
        assert_equal(answer.item(0, "count").int64(), Int64(2))
        if floating:
            assert_equal(answer.item(0, "total").float32(), Float32(4))
        else:
            assert_equal(answer.item(0, "total").int64(), Int64(4))
        var empty = ComposedScan(
            List[Int128]([Int128(0)]),
            List[Float64]([Float64(0)]),
            List[Int]([0]),
            List[Int64](),
            List[List[Int128]]([List[Int128](length=28, fill=0)]),
            List[UInt64]([UInt64(0)]),
        )
        var nulls = finish_global_moments_math(plan, empty)
        assert_true(nulls.item(0, "population").is_null())
        assert_true(nulls.item(0, "population_std").is_null())
        assert_true(nulls.item(0, "population_alias").is_null())
        assert_true(nulls.item(0, "sample").is_null())
        assert_true(nulls.item(0, "deviation").is_null())
        assert_true(nulls.item(0, "average").is_null())
        assert_equal(nulls.item(0, "count").int64(), Int64(0))
    var legacy = ComposedScan(
        List[Int128](), List[Float64](), List[Int](), List[Int64]()
    )
    assert_equal(len(legacy.moment_data), 0)
    assert_equal(len(legacy.moment_largest), 0)


def test_metal_moment_math_keeps_cast_rounded_sums_separate_from_integer_sum() raises:
    from dataframe import Column, DataFrame, Series, col
    from dataframe_metal.moment_plan import plan_moments
    from dataframe_metal.moment_reduce import finish_global_moments_math
    from dataframe_metal.composition_reduce import ComposedScan

    var frame = DataFrame(
        [
            Series(
                "v",
                Column[Int64](
                    [Int64(1152921504606846977), 1152921504606847233]
                ),
            )
        ]
    )
    var plan = (
        plan_moments(
            frame.lazy().select_exprs(
                [
                    col("v").var(0).alias("variance"),
                    col("v").sum().alias("native_sum"),
                ]
            )
        )
        .value()
        .copy()
    )
    # Independent radix moments of the CPU's binary64-cast input values:
    # [2**60,2**60+256]. The native integer sum also retains both low bits.
    var data = List[Int128](length=28, fill=0)
    data[0] = 256
    data[1] = 536870912
    data[9] = 65536
    data[11] = 32
    data[12] = 33554432
    var native_sum = (Int128(1) << 61) + 258
    var scan = ComposedScan(
        List[Int128]([native_sum]),
        List[Float64]([Float64(0)]),
        List[Int]([2]),
        List[Int64](),
        List[List[Int128]]([data^]),
        List[UInt64]([UInt64(1152921504606847232)]),
    )
    var answer = finish_global_moments_math(plan, scan)
    assert_equal(answer.item(0, "variance").float64(), Float64(16384))
    assert_equal(
        answer.item(0, "native_sum").int64(), native_sum.cast[DType.int64]()
    )


def test_metal_grouped_moments_share_states_and_finish_all_native_types() raises:
    from std.memory import bitcast, Pointer
    from std.math import sqrt
    from dataframe import Column, DataFrame, DataType, Series, col
    from dataframe_metal.moment_plan import plan_moments
    from dataframe_metal.moment_group_reduce import finish_grouped_moments_math
    from dataframe_metal.group_partials import GroupedPartials
    from dataframe_metal.group_layout import GroupLayout, GROUP_MOMENT_WORDS
    from dataframe_metal.accelerator.gpu.group_state import (
        initial_group_word,
        add_float_sum,
        add_integer_sum,
    )
    from dataframe_metal.accelerator.gpu.group_moments import (
        add_group_float_moment,
        add_group_integer_moment,
    )

    var frame = DataFrame(
        [
            Series(
                "key",
                Column[Int32](
                    [9, 4, 9, 4, 0, 0], [True, True, True, True, False, False]
                ),
            ),
            Series(
                "f",
                Column[Float32](
                    [1, 2, 5, 6, 0, 0], [True, True, True, True, False, False]
                ),
            ),
            Series(
                "i",
                Column[Int32](
                    [1, 2, 5, 6, 0, 0], [True, True, True, True, False, False]
                ),
            ),
            Series(
                "w",
                Column[Int64](
                    [Int64(1) << 60, 3, (Int64(1) << 60) + 256, 7, 0, 0],
                    [True, True, True, True, False, False],
                ),
            ),
        ]
    )
    var plan = (
        plan_moments(
            frame.lazy()
            .group_by("key")
            .agg(
                [
                    col("f").var(0).alias("f_population"),
                    col("f").var(1).alias("f_sample"),
                    col("f").std(1).alias("f_sample_std"),
                    col("i").std(0).alias("i_std"),
                    col("w").std(0).alias("w_std"),
                    col("w").var(0).alias("w_population"),
                    col("f").sum().alias("f_sum"),
                    col("i").count().alias("i_count"),
                    col("f").len().alias("rows"),
                ]
            )
        )
        .value()
        .copy()
    )
    for blocks in [1, 2]:
        var layout = GroupLayout(
            512, 3, blocks, blocks == 1, GROUP_MOMENT_WORDS
        )
        layout.check()
        assert_equal(
            layout.words(), layout.states_offset() + layout.entries() * 80
        )
        var first = List[UInt32](length=512, fill=UInt32.MAX)
        # Deliberately put the first input group after the next one in hash order.
        first[7] = 0
        first[2] = 1
        first[11] = 4
        var rows = List[UInt32](length=512 * blocks, fill=0)
        var states = List[UInt32](length=layout.entries() * 80, fill=0)
        for entry in range(layout.entries()):
            for word in range(80):
                states[entry * 80 + word] = initial_group_word(word)
        for row in range(6):
            var slot = 7 if row % 2 == 0 else 2
            if row >= 4:
                slot = 11
            var block = row % blocks
            rows[slot * blocks + block] += 1
            if row >= 4:
                continue
            for field in range(3):
                var at = ((slot * blocks + block) * 3 + field) * 80
                states[at] += 1
                var pointer = Pointer[UInt32, MutUntrackedOrigin](
                    unsafe_from_address=Int(
                        states.unsafe_ptr().unsafe_offset(at)
                    )
                )
                if field == 0:
                    var raw = bitcast[DType.uint32](
                        frame.item(row, "f").float32()
                    )
                    add_float_sum(pointer, raw)
                    add_group_float_moment(pointer, raw)
                else:
                    var value = (
                        Int64(frame.item(row, "i").int32()) if field
                        == 1 else frame.item(row, "w").int64()
                    )
                    add_integer_sum(pointer, value)
                    add_group_integer_moment(pointer, value)
        var partials = GroupedPartials(
            first^, rows^, states^, List[Int64](), blocks
        )
        var result = finish_grouped_moments_math(plan, frame, partials)
        assert_equal(result.height(), 3)
        assert_equal(result.item(0, "key").int32(), Int32(9))
        assert_equal(result.item(1, "key").int32(), Int32(4))
        assert_true(result.item(2, "key").is_null())
        for row in range(2):
            assert_equal(result.item(row, "f_population").float64(), Float64(4))
            assert_equal(result.item(row, "f_sample").float64(), Float64(8))
            assert_equal(result.item(row, "i_std").float64(), Float64(2))
            assert_equal(result.item(row, "i_count").int64(), Int64(2))
        assert_equal(result.item(0, "w_population").float64(), Float64(16384))
        assert_equal(result.item(1, "w_population").float64(), Float64(4))
        assert_equal(result.item(0, "f_sum").float32(), Float32(6))
        assert_equal(result.item(1, "f_sum").float32(), Float32(8))
        assert_equal(result.item(2, "i_count").int64(), Int64(0))
        assert_true(result.item(2, "f_population").is_null())
        assert_true(result.item(2, "i_std").is_null())
        assert_true(result.item(2, "w_population").is_null())
        var moments_only = plan.copy()
        _ = moments_only.outputs.pop()
        _ = moments_only.outputs.pop()
        _ = moments_only.outputs.pop()
        for with_rows in [False, True]:
            if with_rows:
                moments_only.outputs.append(
                    plan.outputs[len(plan.outputs) - 1].copy()
                )
            var reduced = finish_grouped_moments_math(
                moments_only, frame, partials
            )
            assert_equal(reduced.height(), 3)
            assert_equal(reduced.width(), 8 if with_rows else 7)
            assert_equal(reduced.item(0, "key").int32(), Int32(9))
            assert_equal(reduced.item(1, "key").int32(), Int32(4))
            assert_true(reduced.item(2, "key").is_null())
            for row in range(2):
                assert_equal(
                    reduced.item(row, "f_population").float64(), Float64(4)
                )
                assert_equal(
                    reduced.item(row, "f_sample").float64(), Float64(8)
                )
                assert_equal(reduced.item(row, "i_std").float64(), Float64(2))
            assert_equal(
                reduced.item(0, "w_population").float64(), Float64(16384)
            )
            assert_true(reduced.item(2, "f_population").is_null())
            assert_true(reduced.item(2, "i_std").is_null())
            assert_true(reduced.item(2, "w_population").is_null())
            for row in range(2):
                assert_equal(
                    reduced.item(row, "f_sample_std").float64(),
                    sqrt(Float64(8)),
                )
            assert_equal(reduced.item(0, "w_std").float64(), Float64(128))
            assert_equal(reduced.item(1, "w_std").float64(), Float64(2))
            assert_true(reduced.item(2, "f_sample_std").is_null())
            assert_true(reduced.item(2, "w_std").is_null())
            if with_rows:
                for row in range(3):
                    assert_equal(reduced.item(row, "rows").int64(), Int64(2))


def test_metal_grouped_moment_atomic_carries_match_exact_variance() raises:
    from std.memory import Pointer, bitcast
    from std.testing import assert_equal
    from dataframe_metal.accelerator.gpu.group_state import add_float_sum
    from dataframe_metal.accelerator.gpu.group_moments import (
        add_group_float_moment,
        add_group_integer_moment,
    )
    from dataframe_metal.moment_finish import finish_variance

    var words = List[UInt32](length=80, fill=0)
    var p = Pointer[UInt32, MutUntrackedOrigin](
        unsafe_from_address=Int(words.unsafe_ptr())
    )
    for n in range(2056):
        var value = Int64(n % 257 - 128) * 1000003
        add_group_integer_moment(p, value)
    var sums = List[Int128](length=2, fill=0)
    var squares = List[Int128](length=4, fill=0)
    for bin in range(2):
        sums[bin] = Int128(
            bitcast[DType.int64](
                UInt64(words[68 + bin * 2])
                | (UInt64(words[69 + bin * 2]) << 32)
            )
        )
    for bin in range(4):
        squares[bin] = Int128(
            bitcast[DType.int64](
                UInt64(words[32 + bin * 2])
                | (UInt64(words[33 + bin * 2]) << 32)
            )
        )
    assert_equal(
        finish_variance(sums, squares, 2056, 0, 0).value(),
        Float64(5504033024049536),
    )
    for i in range(80):
        words[i] = 0
    for n in range(2056):
        var value = Float32((n % 257 - 128) * 256)
        var raw = bitcast[DType.uint32](value)
        add_float_sum(p, raw)
        add_group_float_moment(p, raw)
    sums = List[Int128](length=9, fill=0)
    squares = List[Int128](length=18, fill=0)
    for bin in range(9):
        sums[bin] = Int128(
            bitcast[DType.int64](
                UInt64(words[10 + bin * 2])
                | (UInt64(words[11 + bin * 2]) << 32)
            )
        )
    for bin in range(18):
        squares[bin] = Int128(
            bitcast[DType.int64](
                UInt64(words[32 + bin * 2])
                | (UInt64(words[33 + bin * 2]) << 32)
            )
        )
    assert_equal(
        finish_variance(sums, squares, 2056, 0, -149).value(),
        Float64(360710144),
    )
    _ = words^


def test_metal_grouped_moments_empty_input_retains_schema_without_device() raises:
    from dataframe import Column, DataFrame, DataType, Series, col
    from dataframe_metal.moment_group_reduce import execute_grouped_moments_math

    var frame = DataFrame(
        [
            Series("key", Column[Int32](List[Int32]())),
            Series("f", Column[Float32](List[Float32]())),
            Series("i", Column[Int64](List[Int64]())),
        ]
    )
    var answer = execute_grouped_moments_math(
        frame.lazy()
        .group_by("key")
        .agg(
            [
                col("f").var(0).alias("variance"),
                col("i").std(1).alias("deviation"),
                col("i").count().alias("count"),
            ]
        ),
        True,
    )
    assert_equal(answer[0].height(), 0)
    assert_equal(answer[0].column("key").dtype(), DataType.INT32)
    assert_equal(answer[0].column("variance").dtype(), DataType.FLOAT64)
    assert_equal(answer[0].column("deviation").dtype(), DataType.FLOAT64)
    assert_equal(answer[0].column("count").dtype(), DataType.INT64)
    assert_equal(answer[1].context_ns, 0)
    assert_equal(answer[1].kernel_ns, 0)


def test_metal_moment_word_rounding_covers_limb_boundaries_and_ties() raises:
    from dataframe_metal.moment_finish import MomentMagnitude

    var one = MomentMagnitude.from_bins(List[Int128]([Int128(1)]))
    for shift in range(1, 588):
        var power = MomentMagnitude()
        power.digits[shift // 32] = UInt32(1) << UInt32(shift % 32)
        var half = MomentMagnitude()
        half.digits[(shift - 1) // 32] = UInt32(1) << UInt32((shift - 1) % 32)
        for parity in range(3):
            # Two adjacent integer significands test ties in both directions;
            # the last also rounds through a change in the Float64 exponent.
            var significand = (Int64(1) << 52) + Int64(parity)
            if parity == 2:
                if shift == 587:
                    continue
                significand = (Int64(1) << 53) - 1
            var next = MomentMagnitude.from_bins(
                List[Int128]([Int128(significand) + 1])
            )
            var upper = next.multiply(power)
            var midpoint = upper.subtract(half)
            for side in range(3):
                var value = midpoint.copy()
                if side == 0:
                    value = midpoint.subtract(one)
                elif side == 2:
                    value = upper.subtract(half.subtract(one))
                var rounded = significand + Int64(
                    side == 2 or (side == 1 and Bool(significand & 1))
                )
                assert_equal(value.scaled(-shift), Float64(rounded))


def _cast_moment_frame() raises -> DataFrame:
    from dataframe import Column, DataFrame, Series

    return DataFrame(
        [
            Series("n", Column[Int32]([1, 3, 5, 7])),
            Series("w", Column[Int64]([2, 4, 6, 8])),
            Series("f", Column[Float32]([0.5, 1.5, 2.5, 3.5])),
            Series("p", Column[Int32]([-1, 1, -1, 1])),
            Series("k", Column[Int32]([9, 4, 9, 4])),
            Series("j", Column[Int64]([20, 30, 20, 30])),
        ]
    )


def _cast_moment_query(frame: DataFrame, grouped: Bool) raises -> LazyFrame:
    from dataframe import DataFrame, DataType, Expr, LazyFrame, col, when

    var floating = col("n").cast(DataType.FLOAT32) * col("f")
    var narrow = col("f").cast(DataType.INT32, strict=False)
    var wide = col("n").cast(DataType.INT64) * col("w")
    var conditional = when(col("p") > 0).then(col("w") + 1).otherwise(0)
    var outputs = List[Expr](
        [
            floating.var(0).alias("floating_var"),
            floating.std(1).alias("floating_std"),
            narrow.var(0).alias("narrow_var"),
            wide.var(0).alias("wide_var"),
            conditional.var(0).alias("case_var"),
            wide.sum().alias("wide_sum"),
            narrow.count().alias("narrow_count"),
        ]
    )
    if grouped:
        return (
            frame.lazy().group_by(["k", "j"], maintain_order=True).agg(outputs)
        )
    return frame.lazy().select_exprs(outputs)


def test_metal_moments_combine_native_casts_case_and_tuple_grouping() raises:
    from dataframe import DataType
    from dataframe_metal.moment_plan import plan_moments, moment_output_sources
    from dataframe_metal.moment_reduce import moment_masks
    from dataframe_metal.composition_plan import (
        needs_general_values,
        has_typed_values,
    )
    from dataframe_metal.group_general_reduce import general_case_mode

    var frame = _cast_moment_frame()
    for grouped in [False, True]:
        var matched = plan_moments(_cast_moment_query(frame, grouped))
        assert_true(Bool(matched))
        ref plan = matched.value()
        assert_equal(Bool(plan.grouped), grouped)
        assert_equal(len(plan.selected.bundle.fields), 4)
        assert_equal(plan.selected.bundle.fields[0].dtype, DataType.FLOAT32)
        assert_equal(plan.selected.bundle.fields[1].dtype, DataType.INT32)
        assert_equal(plan.selected.bundle.fields[2].dtype, DataType.INT64)
        assert_equal(plan.selected.bundle.fields[3].dtype, DataType.INT64)
        assert_true(has_typed_values(plan.selected.bundle))
        assert_true(needs_general_values(plan.selected.bundle))
        assert_equal(general_case_mode(plan.selected.bundle), 1)
        assert_equal(moment_masks(plan)[0], UInt32(15))
        assert_equal(
            moment_output_sources(plan.outputs),
            List[Int]([0, 1, 2, 3, 4, 5, 6]),
        )
        if grouped:
            assert_equal(len(plan.grouped.value().keys.names), 2)


def test_metal_moment_layout_keys_use_inlined_native_inputs() raises:
    from dataframe import DataType, col
    from dataframe_metal.moment_plan import (
        plan_moments,
        moment_key,
        moment_execution_region,
    )
    from dataframe_metal.calibration import shape_key
    from dataframe_metal.region_plan import normalize_region
    from dataframe.expr import COL, VAR, STD, MEAN
    from dataframe.lazy import WITH_COLUMNS

    var frame = _cast_moment_frame()
    var value = col("n").cast(DataType.INT64) * col("w")
    for grouped in [False, True]:
        var derived = frame.lazy().with_columns([value.alias("derived")])
        var direct = frame.lazy()
        if grouped:
            derived = derived.group_by(["k", "j"], maintain_order=True).agg(
                [
                    col("derived").var(0).alias("variance"),
                    col("derived").std(1).alias("deviation"),
                ]
            )
            direct = direct.group_by(["k", "j"], maintain_order=True).agg(
                [
                    value.var(0).alias("renamed_variance"),
                    value.std(1).alias("renamed_deviation"),
                ]
            )
        else:
            derived = derived.select_exprs(
                [
                    col("derived").var(0).alias("variance"),
                    col("derived").std(1).alias("deviation"),
                ]
            )
            direct = direct.select_exprs(
                [
                    value.var(0).alias("renamed_variance"),
                    value.std(1).alias("renamed_deviation"),
                ]
            )
        # Generic reduction normalization does not accept VAR/STD bindings.
        # Moment matching rewrites their matching surrogates before inlining.
        assert_true(not Bool(normalize_region(derived)))
        var matched = plan_moments(derived).value().copy()
        var equivalent = plan_moments(direct).value().copy()
        var executable = moment_execution_region(matched)
        ref executable_root = executable._plan._nodes[
            len(executable._plan._nodes) - 1
        ]
        ref population = executable_root.exprs[0]._nodes[
            len(executable_root.exprs[0]._nodes) - 1
        ]
        ref sample_std = executable_root.exprs[1]._nodes[
            len(executable_root.exprs[1]._nodes) - 1
        ]
        assert_equal(population.op, VAR)
        assert_equal(population.integer, Int64(0))
        assert_equal(sample_std.op, STD)
        assert_equal(sample_std.integer, Int64(1))
        for node in executable._plan._nodes:
            assert_true(node.kind != WITH_COLUMNS)
            for expression in node.exprs:
                for item in expression._nodes:
                    assert_true(item.op != COL or item.text != "derived")
        assert_equal(
            moment_key(plan_moments(executable).value()), moment_key(matched)
        )
        # The executable copy must not replace the calibration surrogate.
        ref matching_root = matched.matching_region._plan._nodes[
            len(matched.matching_region._plan._nodes) - 1
        ]
        assert_equal(
            matching_root.exprs[0]
            ._nodes[len(matching_root.exprs[0]._nodes) - 1]
            .op,
            MEAN,
        )
        for node in matched.matching_region._plan._nodes:
            assert_true(node.kind != WITH_COLUMNS)
            for expression in node.exprs:
                for item in expression._nodes:
                    assert_true(item.op != COL or item.text != "derived")
        assert_equal(moment_key(matched), moment_key(equivalent))
        assert_equal(
            shape_key(matched.matching_region, moment_key(matched)),
            shape_key(equivalent.matching_region, moment_key(equivalent)),
        )


def _assert_moment_answer(actual: DataFrame, expected: DataFrame) raises:
    from dataframe import DataType

    assert_equal(actual.columns(), expected.columns())
    assert_equal(actual.height(), expected.height())
    for name in expected.columns():
        assert_equal(actual.column(name).dtype(), expected.column(name).dtype())
        if expected.column(name).dtype() != DataType.FLOAT64:
            assert_true(actual.column(name).equals(expected.column(name)), name)
            continue
        for row in range(expected.height()):
            var value = actual.item(row, name)
            var reference = expected.item(row, name)
            assert_equal(value.is_null(), reference.is_null())
            if not reference.is_null():
                var a = Int128(bitcast[DType.uint64](value.float64()))
                var b = Int128(bitcast[DType.uint64](reference.float64()))
                assert_true(abs(a - b) <= 4, name)


def test_metal_moment_provider_layout_key_keeps_derived_ddof() raises:
    from dataframe import DataType, col
    from dataframe_metal.handoff import layout_key, region_key

    var frame = _cast_moment_frame()
    var value = col("n").cast(DataType.INT64) * col("w")
    for grouped in [False, True]:
        var derived = frame.lazy().with_columns([value.alias("derived")])
        var direct = frame.lazy()
        if grouped:
            derived = derived.group_by(["k", "j"], maintain_order=True).agg(
                [col("derived").var(1).alias("variance")]
            )
            direct = direct.group_by(["k", "j"], maintain_order=True).agg(
                [value.var(1).alias("renamed")]
            )
        else:
            derived = derived.select(col("derived").var(1).alias("variance"))
            direct = direct.select(value.var(1).alias("renamed"))
        assert_true(region_key(derived).startswith("moments:v2:cpu:"))
        assert_equal(layout_key(derived), layout_key(direct))
    assert_true(
        layout_key(frame.lazy().select(value.var(0)))
        != layout_key(frame.lazy().select(value.var(1)))
    )
    assert_true(
        layout_key(frame.lazy().select(value.var(0)))
        != layout_key(frame.lazy().select(value.mean()))
    )


def test_metal_global_moments_empty_input_keeps_schema_without_device() raises:
    from dataframe import Column, DataFrame, Series, col
    from dataframe_metal.moment_reduce import execute_global_moments_math

    var frame = DataFrame([Series("v", Column[Float32](List[Float32]()))])
    var result = execute_global_moments_math(
        frame.lazy().select_exprs(
            [col("v").var(0).alias("v"), col("v").std(1).alias("s")]
        ),
        True,
    )[0].copy()
    assert_equal(result.height(), 1)
    assert_true(result.item(0, "v").is_null())
    assert_true(result.item(0, "s").is_null())
    from dataframe import DataType

    assert_equal(result.column("v").dtype(), DataType.FLOAT64)
    assert_equal(result.column("s").dtype(), DataType.FLOAT64)


# requires: metal
def test_metal_moments_public_native_nullable_slices_chunks_and_groups() raises:
    from std.math import sqrt
    from dataframe import Column, DataFrame, Series, Expr, col
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed, supports

    _ = require_metal()
    for dtype in range(3):
        var value = Series(
            "v",
            Column[Int32](
                [99, -2, 0, 3, 8, 99], [True, True, False, True, True, True]
            ),
        )
        if dtype == 1:
            value = Series(
                "v",
                Column[Int64](
                    [99, -2, 0, 3, 8, 99], [True, True, False, True, True, True]
                ),
            )
        elif dtype == 2:
            value = Series(
                "v",
                Column[Float32](
                    [99, -2, 0, 3, 8, 99], [True, True, False, True, True, True]
                ),
            )
        var original = DataFrame(
            [value^, Series("key", Column[Int32]([99, 1, 1, 2, 2, 99]))]
        ).slice(1, 4)
        for chunked in [False, True]:
            var frame = original.copy()
            if chunked:
                var columns = List[Series]()
                for name in original.columns():
                    var column = original.column(name)
                    columns.append(
                        Series._from_chunks(
                            [column.slice(0, 1), column.slice(1, 3)]
                        )
                    )
                frame = DataFrame(columns^)
            for grouped in [False, True]:
                var outputs = List[Expr](
                    [
                        col("v").var(0).alias("population"),
                        col("v").var(1).alias("sample"),
                        col("v").std(1).alias("deviation"),
                        col("v").count().alias("count"),
                        col("v").sum().alias("sum"),
                        col("v").mean().alias("mean"),
                        col("v").min().alias("minimum"),
                        col("v").max().alias("maximum"),
                    ]
                )
                var query = frame.lazy().select_exprs(outputs)
                var expected = DataFrame(
                    [
                        Series(
                            "population", Column[Float64]([Float64(50) / 3])
                        ),
                        Series("sample", Column[Float64]([25])),
                        Series("deviation", Column[Float64]([5])),
                        Series("count", Column[Int64]([3])),
                    ]
                )
                if grouped:
                    query = (
                        frame.lazy()
                        .group_by("key", maintain_order=True)
                        .agg(outputs)
                    )
                    expected = DataFrame(
                        [
                            Series("key", Column[Int32]([1, 2])),
                            Series("population", Column[Float64]([0, 6.25])),
                            Series(
                                "sample",
                                Column[Float64]([0, 12.5], [False, True]),
                            ),
                            Series(
                                "deviation",
                                Column[Float64](
                                    [0, sqrt(Float64(12.5))], [False, True]
                                ),
                            ),
                            Series("count", Column[Int64]([1, 2])),
                        ]
                    )
                var complete = List[Series]()
                for name in expected.columns():
                    complete.append(expected.column(name))
                for statistic in range(3):
                    var name = "sum" if statistic == 0 else (
                        "minimum" if statistic == 1 else "maximum"
                    )
                    var values = List[Int64](
                        [
                            Int64(
                                9 if statistic
                                == 0 else (-2 if statistic == 1 else 8)
                            )
                        ]
                    )
                    if grouped:
                        values = List[Int64](
                            [
                                Int64(-2),
                                Int64(
                                    11 if statistic
                                    == 0 else (3 if statistic == 1 else 8)
                                ),
                            ]
                        )
                    var column = Series(name, Column[Int64](values.copy()))
                    if dtype == 0:
                        var narrow = List[Int32]()
                        for value in values:
                            narrow.append(Int32(value))
                        column = Series(name, Column[Int32](narrow^))
                    elif dtype == 2:
                        var floating = List[Float32]()
                        for value in values:
                            floating.append(Float32(value))
                        column = Series(name, Column[Float32](floating^))
                    complete.append(column^)
                    if statistic == 0:
                        complete.append(
                            Series(
                                "mean",
                                Column[Float64](
                                    List[Float64](
                                        [Float64(-2), 5.5]
                                    ) if grouped else List[Float64](
                                        [Float64(3)]
                                    )
                                ),
                            )
                        )
                expected = DataFrame(complete^)
                assert_true(supports(query))
                for copied in [False, True]:
                    _assert_moment_answer(
                        execute_timed(query, zero_copy=not copied)[0], expected
                    )
                _assert_public_moment(query, expected)


# requires: metal
def test_metal_moments_public_cancellation_subnormals_and_wide_retry() raises:
    from std.math import sqrt
    from dataframe import Column, DataFrame, Series, Expr, col
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed

    _ = require_metal()
    for family in range(3):
        var value = Series(
            "v", Column[Int64]([Int64(1) << 60, (Int64(1) << 60) + 256])
        )
        var population = Float64(16384)
        if family == 1:
            value = Series(
                "v",
                Column[Float32](
                    [
                        bitcast[DType.float32](UInt32(1)),
                        bitcast[DType.float32](UInt32(3)),
                    ]
                ),
            )
            population = bitcast[DType.float64](UInt64(725) << 52)
        elif family == 2:
            value = Series(
                "v", Column[Float32]([Float32.MAX_FINITE, -Float32.MAX_FINITE])
            )
            population = Float64(Float32.MAX_FINITE) * Float64(
                Float32.MAX_FINITE
            )
        var frame = DataFrame([value^, Series("key", Column[Int32]([7, 7]))])
        for grouped in [False, True]:
            var outputs = List[Expr](
                [
                    col("v").var(0).alias("population"),
                    col("v").var(1).alias("sample"),
                    col("v").std(1).alias("deviation"),
                ]
            )
            var query = frame.lazy().select_exprs(outputs)
            var columns = List[Series]()
            if grouped:
                query = (
                    frame.lazy()
                    .group_by("key", maintain_order=True)
                    .agg(outputs)
                )
                columns.append(Series("key", Column[Int32]([7])))
            columns.extend(
                [
                    Series("population", Column[Float64]([population])),
                    Series("sample", Column[Float64]([population * 2])),
                    Series(
                        "deviation", Column[Float64]([sqrt(population * 2)])
                    ),
                ]
            )
            var expected = DataFrame(columns^)
            if family == 0:
                # Preserve the statistical behavior users obtain from CPU,
                # DuckDB and Polars, rather than returning an exact answer
                # that materially differs from the library's CPU result.
                with assert_raises(
                    contains="variance centering needs CPU adjudication"
                ):
                    _ = execute_timed(query)
                var cpu = query.collect(engine="cpu")
                var profiled = query.profile(accelerator=require_metal())
                _assert_moment_answer(profiled[0], cpu)
                assert_true("fault_adjudication" in profiled[1].columns())
                assert_equal(
                    cpu.item(0, "population").float64(), Float64(32768)
                )
                continue
            _assert_moment_answer(execute_timed(query)[0], expected)
            _assert_public_moment(query, expected)


def _assert_public_moment(query: LazyFrame, expected: DataFrame) raises:
    from tests.support.metal_outcomes import require_metal

    var observed = query.profile(accelerator=require_metal())
    _assert_moment_answer(observed[0], expected)
    ref report = observed[1]
    assert_true("fault_adjudication" not in report.columns())
    var handed = False
    for row in range(report.height()):
        if report.item(row, "executor").string() == "accel":
            assert_equal(report.item(row, "algorithm").string(), "handoff")
            handed = True
    assert_true(handed, "Moment query must complete through the GPU handoff")


# requires: metal
def test_metal_moments_public_derived_casts_membership_and_tuple_groups() raises:
    from std.math import sqrt
    from dataframe import Column, DataFrame, Series, DataType, col, lit
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed, supports, layout_key

    _ = require_metal()
    var frame = _cast_moment_frame()
    for filtered in [False, True]:
        for grouped in [False, True]:
            var source = frame.lazy()
            if filtered:
                source = source.filter(col("p").is_in([lit(Int32(1))]))
            var value = col("n").cast(DataType.INT64) * col("w")
            source = source.with_columns([value.alias("derived")])
            var outputs = List(
                [
                    col("derived").var(0).alias("variance"),
                    col("derived").std(0).alias("deviation"),
                    col("derived").count().alias("count"),
                ]
            )
            var query = source.select_exprs(outputs)
            var columns = List[Series]()
            if grouped:
                query = source.group_by(["k", "j"], maintain_order=True).agg(
                    outputs
                )
                columns.extend(
                    [
                        Series(
                            "k",
                            Column[Int32](
                                List[Int32]([Int32(4)]) if filtered else List[
                                    Int32
                                ]([Int32(9), 4])
                            ),
                        ),
                        Series(
                            "j",
                            Column[Int64](
                                List[Int64]([Int64(30)]) if filtered else List[
                                    Int64
                                ]([Int64(20), 30])
                            ),
                        ),
                    ]
                )
            var variances = List[Float64]([Float64(484)]) if filtered else List[
                Float64
            ]([Float64(196), 484]) if grouped else List[Float64]([Float64(421)])
            var deviations = List[Float64]()
            for v in variances:
                deviations.append(sqrt(v))
            columns.extend(
                [
                    Series("variance", Column[Float64](variances^)),
                    Series("deviation", Column[Float64](deviations^)),
                    Series(
                        "count",
                        Column[Int64](
                            List[Int64]([Int64(2)]) if filtered else List[
                                Int64
                            ]([Int64(2), 2]) if grouped else List[Int64](
                                [Int64(4)]
                            )
                        ),
                    ),
                ]
            )
            var expected = DataFrame(columns^)
            assert_true(supports(query))
            assert_true(layout_key(query).startswith("moments:v2:cpu:"))
            _assert_moment_answer(execute_timed(query)[0], expected)
            _assert_public_moment(query, expected)


# requires: metal
def test_metal_moments_public_four_cast_case_values_share_one_scan() raises:
    from std.math import sqrt
    from dataframe import Column, DataFrame, Series, col, lit
    from tests.support.metal_outcomes import require_metal
    from dataframe_metal.handoff import execute_timed, supports

    _ = require_metal()
    var frame = _cast_moment_frame()
    for filtered in [False, True]:
        for grouped in [False, True]:
            var original = _cast_moment_query(frame, grouped)
            var outputs = original._plan._nodes[
                len(original._plan._nodes) - 1
            ].exprs.copy()
            var source = frame.lazy()
            if filtered:
                source = source.filter(col("p").is_in([lit(Int32(1))]))
            var query = source.select_exprs(outputs)
            var columns = List[Series]()
            if grouped:
                query = source.group_by(["k", "j"], maintain_order=True).agg(
                    outputs
                )
                columns.extend(
                    [
                        Series(
                            "k",
                            Column[Int32](
                                List[Int32]([Int32(4)]) if filtered else List[
                                    Int32
                                ]([Int32(9), 4])
                            ),
                        ),
                        Series(
                            "j",
                            Column[Int64](
                                List[Int64]([Int64(30)]) if filtered else List[
                                    Int64
                                ]([Int64(20), 30])
                            ),
                        ),
                    ]
                )
            var floating = List[Float64]([Float64(100)]) if filtered else (
                List[Float64]([Float64(36), 100]) if grouped else List[Float64](
                    [Float64(84)]
                )
            )
            var deviation = List[Float64](
                [sqrt(Float64(200))]
            ) if filtered else (
                List[Float64](
                    [sqrt(Float64(72)), sqrt(Float64(200))]
                ) if grouped else List[Float64]([sqrt(Float64(112))])
            )
            var narrow = List[Float64]([Float64(1)]) if filtered else (
                List[Float64]([Float64(1), 1]) if grouped else List[Float64](
                    [Float64(1.25)]
                )
            )
            var wide = List[Float64]([Float64(484)]) if filtered else (
                List[Float64]([Float64(196), 484]) if grouped else List[
                    Float64
                ]([Float64(421)])
            )
            var conditional = List[Float64]([Float64(4)]) if filtered else (
                List[Float64]([Float64(0), 4]) if grouped else List[Float64](
                    [Float64(14.25)]
                )
            )
            var total = List[Int64]([Int64(68)]) if filtered else (
                List[Int64]([Int64(32), 68]) if grouped else List[Int64](
                    [Int64(100)]
                )
            )
            var count = List[Int64]([Int64(2)]) if filtered else (
                List[Int64]([Int64(2), 2]) if grouped else List[Int64](
                    [Int64(4)]
                )
            )
            columns.extend(
                [
                    Series("floating_var", Column[Float64](floating^)),
                    Series("floating_std", Column[Float64](deviation^)),
                    Series("narrow_var", Column[Float64](narrow^)),
                    Series("wide_var", Column[Float64](wide^)),
                    Series("case_var", Column[Float64](conditional^)),
                    Series("wide_sum", Column[Int64](total^)),
                    Series("narrow_count", Column[Int64](count^)),
                ]
            )
            var expected = DataFrame(columns^)
            assert_true(supports(query))
            for copied in [False, True]:
                _assert_moment_answer(
                    execute_timed(query, zero_copy=not copied)[0], expected
                )
            _assert_public_moment(query, expected)


# requires: metal
def test_metal_moments_public_faults_and_invalid_ddof_keep_cpu_rules() raises:
    from dataframe import Column, DataFrame, Series, DataType, col, lit
    from tests.support.metal_outcomes import require_metal, same_outcome

    var runtime = require_metal()
    var frame = DataFrame(
        [
            Series(
                "v",
                Column[Float32](
                    [bitcast[DType.float32](UInt32(0x7F800000)), 1]
                ),
            )
        ]
    )
    var query = frame.lazy().select_exprs(
        [col("v").var(0).alias("v"), col("v").std(1).alias("s")]
    )
    same_outcome(query)
    var profiled = query.profile(accelerator=runtime)
    assert_true("fault_adjudication" in profiled[1].columns())
    same_outcome(frame.lazy().select(col("v").cast(DataType.INT32).var(0)))
    same_outcome(frame.lazy().select(col("v").var(-1)))
    var tiny = bitcast[DType.float32](UInt32(1))
    var finite = DataFrame([Series("v", Column[Float32]([tiny, tiny * 2]))])
    same_outcome(finite.lazy().select((col("v") * lit(Float32(1))).var(0)))
