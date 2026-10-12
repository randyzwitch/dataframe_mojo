"""Native reductions: scalar memberships, exact widths and checked boundaries.

Public tests require the real Metal capability. CPU model verification is a
separate callable fixture audit; it is never reported as native execution.
"""
from std.testing import assert_equal, assert_true, assert_raises
from std.memory import bitcast
from dataframe import DataFrame, DataType, Series, Column, Expr, col, lit, when
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from tests.execution_modes.test_backend_metal import (
    require_metal,
    NATIVE_NUMERIC,
)
from tests.support.metal_domain_checks import check_domain
from tests.support.metal_domain_reference import (
    value,
    valid,
    group_key,
    groups,
    selected,
    floor128,
)


def reduction_source[D: DType](n: Int) raises -> DataFrame:
    var payload = List[Scalar[D]]()
    var bits = List[Bool]()
    var keys = List[Int32]()
    var key_bits = List[Bool]()
    for row in range(n + 5):
        payload.append(Scalar[D](value(row) if valid(row) else 99))
        bits.append(valid(row))
        keys.append(Int32(max(0, group_key(row))))
        key_bits.append(group_key(row) >= 0)
    return DataFrame(
        [
            Series("key", Column[Int32](keys^, key_bits^)),
            Series("x", Column[Scalar[D]](payload^, bits^)),
        ]
    ).slice(5, n)


def reduction_expressions[D: DType]() -> List[Expr]:
    var expressions: List[Expr] = [
        col("x").min().alias("lo"),
        col("x").max().alias("hi"),
        col("x").first().alias("first"),
        col("x").last().alias("last"),
        col("x").count().alias("count"),
        col("x").len().alias("length"),
        col("x").null_count().alias("nulls"),
        col("x").arg_min().alias("argmin"),
        col("x").arg_max().alias("argmax"),
    ]
    comptime if not D.is_floating_point():
        expressions.append(col("x").sum().alias("sum"))
        expressions.append(col("x").sum(min_count=2).alias("sum2"))
    return expressions^


def reduction_expected[
    D: DType
](memberships: List[List[Int]], grouped: Bool) raises -> DataFrame:
    var key = List[Int32]()
    var key_bits = List[Bool]()
    var lows = List[Scalar[D]]()
    var highs = List[Scalar[D]]()
    var firsts = List[Scalar[D]]()
    var lasts = List[Scalar[D]]()
    var nonempty = List[Bool]()
    var first_valid = List[Bool]()
    var last_valid = List[Bool]()
    var counts = List[Int64]()
    var lengths = List[Int64]()
    var nulls = List[Int64]()
    var argmin = List[UInt32]()
    var argmax = List[UInt32]()
    var sums = List[Int64]()
    var sums2_valid = List[Bool]()
    for rows in memberships:
        if grouped:
            var k = group_key(rows[0])
            key.append(Int32(max(0, k)))
            key_bits.append(k >= 0)
        var count = Int64(0)
        var total = Int64(0)
        var low = 0
        var high = 0
        var low_at = UInt32(0)
        var high_at = UInt32(0)
        for at in range(len(rows)):
            var row = rows[at]
            if not valid(row):
                continue
            var v = value(row)
            if count == 0 or v < low:
                low = v
                low_at = UInt32(at)
            if count == 0 or v > high:
                high = v
                high_at = UInt32(at)
            count += 1
            total += Int64(v)
        lows.append(Scalar[D](low))
        highs.append(Scalar[D](high))
        nonempty.append(count > 0)
        firsts.append(Scalar[D](value(rows[0]) if len(rows) else 0))
        lasts.append(Scalar[D](value(rows[len(rows) - 1]) if len(rows) else 0))
        first_valid.append(len(rows) > 0 and valid(rows[0]))
        last_valid.append(len(rows) > 0 and valid(rows[len(rows) - 1]))
        counts.append(count)
        lengths.append(Int64(len(rows)))
        nulls.append(Int64(len(rows)) - count)
        argmin.append(low_at)
        argmax.append(high_at)
        sums.append(total)
        sums2_valid.append(count >= 2)
    var columns = List[Series]()
    if grouped:
        columns.append(Series("key", Column[Int32](key^, key_bits^)))
    columns.append(Series("lo", Column[Scalar[D]](lows^, nonempty)))
    columns.append(Series("hi", Column[Scalar[D]](highs^, nonempty)))
    columns.append(Series("first", Column[Scalar[D]](firsts^, first_valid^)))
    columns.append(Series("last", Column[Scalar[D]](lasts^, last_valid^)))
    columns.append(Series("count", Column[Int64](counts^)))
    columns.append(Series("length", Column[Int64](lengths^)))
    columns.append(Series("nulls", Column[Int64](nulls^)))
    columns.append(Series("argmin", Column[UInt32](argmin^, nonempty)))
    columns.append(Series("argmax", Column[UInt32](argmax^, nonempty)))
    comptime if not D.is_floating_point():
        comptime S = DType.int64 if D == DType.int8 or D == DType.int16 or D == DType.uint8 or D == DType.uint16 else D
        var typed_sums = List[Scalar[S]]()
        for total in sums:
            typed_sums.append(Scalar[S](total))
        columns.append(Series("sum", Column[Scalar[S]](typed_sums.copy())))
        columns.append(
            Series("sum2", Column[Scalar[S]](typed_sums^, sums2_valid^))
        )
    return DataFrame(columns^)


def numeric_reductions(runtime: MetalRuntime, native: Bool) raises:
    comptime for at in range(len(NATIVE_NUMERIC)):
        comptime D = NATIVE_NUMERIC[at]
        for n in [0, 1, 65, 513]:
            var source = reduction_source[D](n)
            var expressions = reduction_expressions[D]()
            for filtered in [False, True]:
                var plan = source.lazy()
                if filtered:
                    plan = plan.filter(col("key") >= lit(Int32(2)))
                var rows = selected(n, filtered)
                check_domain(
                    plan.select_exprs(expressions),
                    reduction_expected[D]([rows.copy()], False),
                    runtime,
                    native,
                    n > 0,
                )
                check_domain(
                    plan.group_by("key", maintain_order=True).agg(expressions),
                    reduction_expected[D](groups(rows), True),
                    runtime,
                    native,
                    n > 0,
                )


def test_native_metal_grouped_and_global_reductions_match_scalar_memberships_all_widths() raises:
    numeric_reductions(require_metal(), True)


def composition_cases(runtime: MetalRuntime, native: Bool) raises:
    var source = DataFrame(
        [
            Series("key", Column[Int32]([2, 1, 2, 3, 1, 3])),
            Series(
                "x",
                Column[Int16](
                    [4, 8, 9, 99, 2, 99], [True, True, True, False, True, False]
                ),
            ),
            Series(
                "__dataframe_metal_aggregate_0",
                Column[Int32]([1, 2, 3, 4, 5, 6]),
            ),
        ]
    )
    var expressions: List[Expr] = [
        (col("x").len() + 1).alias("rows"),
        (col("x").max().cast("int64") - col("x").min().cast("int64")).alias(
            "range"
        ),
        (col("x").sum() + col("x").count()).alias("total"),
        when(col("x").count() > 0)
        .then(col("x").min())
        .otherwise(lit(Int16(-7)))
        .alias("chosen"),
    ]
    var grouped = (
        source.lazy().group_by("key", maintain_order=True).agg(expressions)
    )
    var expected = DataFrame(
        [
            Series("key", Column[Int32]([2, 1, 3])),
            Series("rows", Column[Int64]([3, 3, 3])),
            Series("range", Column[Int64]([5, 6, 0], [True, True, False])),
            Series("total", Column[Int64]([15, 12, 0])),
            Series("chosen", Column[Int16]([4, 2, -7])),
        ]
    )
    # Grouped aggregation and its projection share one native region. Global
    # scalar aggregation below deliberately exercises a separate resident edge.
    check_domain(grouped, expected, runtime, native, fused=True)
    check_domain(
        source.lazy().select_exprs(expressions),
        DataFrame(
            [
                Series("rows", Column[Int64]([7])),
                Series("range", Column[Int64]([7])),
                Series("total", Column[Int64]([27])),
                Series("chosen", Column[Int16]([2])),
            ]
        ),
        runtime,
        native,
        resident=True,
    )
    check_domain(
        source.slice(0, 0).lazy().select_exprs(expressions),
        DataFrame(
            [
                Series("rows", Column[Int64]([1])),
                Series("range", Column[Int64]([0], [False])),
                Series("total", Column[Int64]([0])),
                Series("chosen", Column[Int16]([-7])),
            ]
        ),
        runtime,
        native,
        submitted=False,
    )
    var chain = (
        source.lazy()
        .group_by("key", maintain_order=True)
        .agg(col("x").count().alias("n"))
        .filter(col("n") > 0)
        .sort("key")
        .select((col("n") + 2).alias("answer"))
    )
    check_domain(
        chain,
        DataFrame([Series("answer", Column[Int64]([4, 4]))]),
        runtime,
        native,
        fused=True,
    )


def test_native_metal_aggregate_compositions_and_consumers_stay_resident() raises:
    composition_cases(require_metal(), True)


def bool_and_float_cases(runtime: MetalRuntime, native: Bool) raises:
    for size in [0, 1, 3, 4]:
        var values: List[Bool] = [False, False, True, True]
        var bits: List[Bool] = [False, True, True, False]
        values.resize(size, False)
        bits.resize(size, False)
        var source = DataFrame([Series("b", Column[Bool](values^, bits^))])
        var expressions: List[Expr] = [
            col("b").any().alias("any"),
            col("b").all().alias("all"),
            col("b").any(ignore_nulls=False).alias("kleene_any"),
            col("b").all(ignore_nulls=False).alias("kleene_all"),
        ]
        var any_value = size >= 3
        var all_value = size <= 1
        var expected = DataFrame(
            [
                Series("any", Column[Bool]([any_value])),
                Series("all", Column[Bool]([all_value])),
                Series(
                    "kleene_any",
                    Column[Bool]([any_value], [size == 0 or size >= 3]),
                ),
                Series(
                    "kleene_all",
                    Column[Bool]([all_value], [size == 0 or size >= 2]),
                ),
            ]
        )
        check_domain(
            source.lazy().select_exprs(expressions),
            expected,
            runtime,
            native,
            size > 0,
        )
    var nan = bitcast[DType.float32](UInt32(0x7FC00035))
    var tiny = bitcast[DType.float32](UInt32(1))
    var source = DataFrame(
        [
            Series("g", Column[Int32]([0, 0, 1, 1, 2, 2, 3, 3])),
            Series(
                "f", Column[Float32]([nan, 2, nan, nan, -0.0, 0.0, tiny, -tiny])
            ),
        ]
    )
    var expressions: List[Expr] = [
        col("f").min().alias("lo"),
        col("f").max().alias("hi"),
        col("f").arg_min().alias("argmin"),
        col("f").arg_max().alias("argmax"),
    ]
    var expected = DataFrame(
        [
            Series("g", Column[Int32]([0, 1, 2, 3])),
            Series("lo", Column[Float32]([2, nan, -0.0, -tiny])),
            Series("hi", Column[Float32]([nan, nan, -0.0, tiny])),
            Series("argmin", Column[UInt32]([1, 0, 0, 1])),
            Series("argmax", Column[UInt32]([1, 0, 0, 0])),
        ]
    )
    check_domain(
        source.lazy().group_by("g", maintain_order=True).agg(expressions),
        expected,
        runtime,
        native,
    )


def test_native_metal_boolean_kleene_and_float_extrema_keep_null_nan_zero_and_subnormal_rules() raises:
    bool_and_float_cases(require_metal(), True)


def wide_sum_cases(runtime: MetalRuntime, native: Bool) raises:
    var source = DataFrame(
        [
            Series("g", Column[Int32]([1, 1, 2, 2])),
            Series(
                "i", Column[Int64]([7, -3, 99, 4], [True, True, False, True])
            ),
            Series(
                "u",
                Column[UInt64](
                    [7, 3, UInt64.MAX, 4], [True, True, False, True]
                ),
            ),
            Series(
                "d",
                Column[Int64]([7, -3, Int64.MIN, 4], [True, True, False, True]),
            ).with_dtype(DataType.duration("us")),
        ]
    )
    var expressions: List[Expr] = [
        col("i").sum(),
        col("u").sum(),
        col("d").sum(),
    ]
    check_domain(
        source.lazy().select_exprs(expressions),
        DataFrame(
            [
                Series("i", Column[Int64]([8])),
                Series("u", Column[UInt64]([14])),
                Series("d", Column[Int64]([8])).with_dtype(
                    DataType.duration("us")
                ),
            ]
        ),
        runtime,
        native,
    )
    check_domain(
        source.lazy().group_by("g", maintain_order=True).agg(expressions),
        DataFrame(
            [
                Series("g", Column[Int32]([1, 2])),
                Series("i", Column[Int64]([4, 4])),
                Series("u", Column[UInt64]([10, 4])),
                Series("d", Column[Int64]([4, 4])).with_dtype(
                    DataType.duration("us")
                ),
            ]
        ),
        runtime,
        native,
    )
    var singleton = DataFrame(
        [
            Series("g", Column[Int32]([1, 2])),
            Series("u", Column[UInt64]([UInt64.MAX, UInt64.MAX - 1])),
        ]
    )
    check_domain(
        singleton.lazy().group_by("g", maintain_order=True).agg(col("u").sum()),
        singleton,
        runtime,
        native,
    )
    var cancel32 = DataFrame([Series("x", Column[Int32]([Int32.MAX, 1, -1]))])
    check_domain(
        cancel32.lazy().select(col("x").sum()),
        DataFrame([Series("x", Column[Int32]([Int32.MAX]))]),
        runtime,
        native,
    )
    if native:
        var cancel = DataFrame(
            [Series("x", Column[Int64]([Int64.MAX, -Int64.MAX]))]
        )
        check_domain(
            cancel.lazy().select(col("x").sum()),
            DataFrame([Series("x", Column[Int64]([0]))]),
            runtime,
            native,
        )
        var bad32 = DataFrame([Series("x", Column[Int32]([Int32.MAX, 1]))])
        with assert_raises(contains="overflow"):
            _ = bad32.lazy().select(col("x").sum()).collect(accelerator=runtime)
        var unsafe = DataFrame([Series("x", Column[UInt64]([UInt64.MAX, 1]))])
        with assert_raises(contains="Integer overflow"):
            _ = (
                unsafe.lazy()
                .select(col("x").sum())
                .collect(accelerator=runtime)
            )
        var safe_filtered = (
            unsafe.lazy()
            .filter(col("x") == lit(UInt64(1)))
            .select(col("x").sum())
        )
        check_domain(
            safe_filtered,
            DataFrame([Series("x", Column[UInt64]([1]))]),
            runtime,
            native,
        )
        # Recovery uses the same runtime after every rejected native submission.
        check_domain(
            cancel32.lazy().select(col("x").sum()),
            DataFrame([Series("x", Column[Int32]([Int32.MAX]))]),
            runtime,
            native,
        )


def test_native_metal_wide_sum_proofs_exact_cancellation_and_error_recovery() raises:
    wide_sum_cases(require_metal(), True)


def verify_metal_reduction_cpu_models() raises:
    var runtime = MetalRuntime()
    numeric_reductions(runtime, False)
    composition_cases(runtime, False)
    bool_and_float_cases(runtime, False)
    wide_sum_cases(runtime, False)


def integer_expression_cases[
    D: DType
](runtime: MetalRuntime, native: Bool) raises:
    var raw: List[Scalar[D]]
    var divisors: List[Scalar[D]]
    comptime if D.is_signed():
        raw = [-7, -1, 0, 1, 7, Scalar[D].MIN]
        divisors = [3, -3, 0, -1, 2, 1]
    else:
        raw = [0, 1, 2, 7, 15, Scalar[D].MAX]
        divisors = [3, 3, 0, 1, 2, 1]
    var bits: List[Bool] = [True, True, True, True, True, False]
    var source = DataFrame(
        [
            Series("x", Column[Scalar[D]](raw.copy(), bits)),
            Series("y", Column[Scalar[D]](divisors.copy())),
        ]
    )
    var quotient = List[Scalar[D]]()
    var remainder = List[Scalar[D]]()
    var squares = List[Scalar[D]]()
    var absolute = List[Scalar[D]]()
    var clipped = List[Scalar[D]]()
    var division_valid = List[Bool]()
    for at in range(len(raw)):
        var x = Int128(raw[at]) if bits[at] else Int128(0)
        var y = Int128(divisors[at])
        var q = floor128(x, y) if y != 0 else Int128(0)
        quotient.append(Scalar[D](q))
        remainder.append(Scalar[D](x - q * y))
        squares.append(Scalar[D](x * x))
        absolute.append(Scalar[D](-x if x < 0 else x))
        clipped.append(Scalar[D](min(max(x, Int128(0)), Int128(2))))
        division_valid.append(bits[at] and y != 0)
    check_domain(
        source.lazy().select_exprs(
            [
                (col("x") // col("y")).alias("div"),
                (col("x") % col("y")).alias("mod"),
                col("x").pow(lit(Scalar[D](2))).alias("pow"),
                col("x").abs().alias("abs"),
                col("x").floor().alias("floor"),
                col("x").ceil().alias("ceil"),
                col("x").round(3).alias("round"),
                col("x")
                .clip(lit(Scalar[D](0)), lit(Scalar[D](2)))
                .alias("clip"),
                col("x")
                .is_in(
                    [lit(Scalar[D].MAX), lit(Scalar[D](0)), lit(Scalar[D].MAX)]
                )
                .alias("hit"),
            ]
        ),
        DataFrame(
            [
                Series("div", Column[Scalar[D]](quotient^, division_valid)),
                Series("mod", Column[Scalar[D]](remainder^, division_valid)),
                Series("pow", Column[Scalar[D]](squares^, bits)),
                Series("abs", Column[Scalar[D]](absolute^, bits)),
                Series("floor", Column[Scalar[D]](raw.copy(), bits)),
                Series("ceil", Column[Scalar[D]](raw.copy(), bits)),
                Series("round", Column[Scalar[D]](raw.copy(), bits)),
                Series("clip", Column[Scalar[D]](clipped^, bits)),
                Series(
                    "hit",
                    Column[Bool](
                        [
                            raw[0] == 0,
                            raw[1] == 0,
                            raw[2] == 0,
                            raw[3] == 0,
                            raw[4] == 0,
                            False,
                        ],
                        bits,
                    ),
                ),
            ]
        ),
        runtime,
        native,
    )


def integer_cast_case[
    S: DType, T: DType
](runtime: MetalRuntime, native: Bool) raises:
    var raw: List[Scalar[S]] = [0, 1, 2, 127, Scalar[S].MAX, Scalar[S].MIN, 0]
    var bits: List[Bool] = [True, True, True, True, True, True, False]
    var expected = List[Scalar[T]]()
    var expected_bits = List[Bool]()
    for row in range(len(raw)):
        var value = Int128(raw[row])
        comptime if T == DType.float32:
            expected.append(Scalar[T](Float32(Float64(value))))
            expected_bits.append(bits[row])
        else:
            var representable = value >= Int128(
                Scalar[T].MIN
            ) and value <= Int128(Scalar[T].MAX)
            expected.append(Scalar[T](value) if representable else Scalar[T](0))
            expected_bits.append(bits[row] and representable)
    var source = DataFrame([Series("x", Column[Scalar[S]](raw^, bits^))])
    check_domain(
        source.lazy().select(col("x").cast(DataType.of(T), strict=False)),
        DataFrame([Series("x", Column[Scalar[T]](expected^, expected_bits^))]),
        runtime,
        native,
    )


def integer_numeric_cases(runtime: MetalRuntime, native: Bool) raises:
    comptime for at in range(8):
        comptime S = NATIVE_NUMERIC[at]
        integer_expression_cases[S](runtime, native)
        comptime for target in range(len(NATIVE_NUMERIC)):
            comptime T = NATIVE_NUMERIC[target]
            integer_cast_case[S, T](runtime, native)
        var values: List[Scalar[S]] = [0, 1, Scalar[S].MAX, Scalar[S].MIN, 99]
        var bits: List[Bool] = [True, True, True, True, False]
        check_domain(
            DataFrame([Series("x", Column[Scalar[S]](values^, bits))])
            .lazy()
            .select(col("x").cast("bool")),
            DataFrame(
                [
                    Series(
                        "x",
                        Column[Bool](
                            [
                                False,
                                True,
                                Scalar[S].MAX != 0,
                                Scalar[S].MIN != 0,
                                False,
                            ],
                            bits,
                        ),
                    )
                ]
            ),
            runtime,
            native,
        )


def test_native_metal_integer_operators_membership_and_cast_matrix_keep_exact_native_words() raises:
    integer_numeric_cases(require_metal(), True)


def masked_cases(runtime: MetalRuntime, native: Bool) raises:
    var source = DataFrame(
        [
            Series(
                "p",
                Column[Bool](
                    [False, False, True, True], [True, False, True, True]
                ),
            ),
            Series("i", Column[Int8]([Int8.MIN, Int8.MIN, 2, 3])),
            Series("u", Column[UInt64]([UInt64.MAX, UInt64.MAX, 2, 3])),
            Series("f", Column[Float32]([1e-20, 1e-20, 1e20, 1e20])),
        ]
    )
    var plan = source.lazy().select_exprs(
        [
            when(col("p"))
            .then(col("i").abs())
            .otherwise(col("i"))
            .alias("abs"),
            when(col("p"))
            .then(col("u").cast("int8"))
            .otherwise(lit(Int8(0)))
            .alias("cast"),
            when(col("p"))
            .then(col("f") * lit(Float32(1e-20)))
            .otherwise(col("f"))
            .alias("product"),
            when(col("p")).then(col("i")).alias("nullable"),
        ]
    )
    var expected_product = Float32(Float32(1e20) * Float32(1e-20))
    check_domain(
        plan,
        DataFrame(
            [
                Series("abs", Column[Int8]([Int8.MIN, Int8.MIN, 2, 3])),
                Series("cast", Column[Int8]([0, 0, 2, 3])),
                Series(
                    "product",
                    Column[Float32](
                        [1e-20, 1e-20, expected_product, expected_product]
                    ),
                ),
                Series(
                    "nullable",
                    Column[Int8]([0, 0, 2, 3], [False, False, True, True]),
                ),
            ]
        ),
        runtime,
        native,
    )
    var casts = DataFrame(
        [
            Series(
                "f",
                Column[Float32](
                    [
                        -128.5,
                        -0.5,
                        1.5,
                        255,
                        bitcast[DType.float32](UInt32(0x7FC00001)),
                        bitcast[DType.float32](UInt32(0x7F800000)),
                        99,
                    ],
                    [True, True, True, True, True, True, False],
                ),
            )
        ]
    )
    check_domain(
        casts.lazy().select_exprs(
            [
                col("f").cast("int8", strict=False).alias("i"),
                col("f").cast("uint8", strict=False).alias("u"),
                col("f").cast("bool", strict=False).alias("b"),
            ]
        ),
        DataFrame(
            [
                Series(
                    "i",
                    Column[Int8](
                        [-128, 0, 1, 0, 0, 0, 0],
                        [True, True, True, False, False, False, False],
                    ),
                ),
                Series(
                    "u",
                    Column[UInt8](
                        [0, 0, 1, 255, 0, 0, 0],
                        [False, True, True, True, False, False, False],
                    ),
                ),
                Series(
                    "b",
                    Column[Bool](
                        [True, True, True, True, False, True, False],
                        [True, True, True, True, False, True, False],
                    ),
                ),
            ]
        ),
        runtime,
        native,
    )
    if native:
        with assert_raises(contains="nonnegative exponent"):
            _ = (
                source.lazy()
                .select(col("i").pow(lit(Int8(-1))))
                .collect(accelerator=runtime)
            )
        with assert_raises(contains="overflow"):
            _ = (
                source.lazy()
                .select(
                    when(~col("p")).then(col("i").abs()).otherwise(col("i"))
                )
                .collect(accelerator=runtime)
            )
        with assert_raises(contains="strict cast failed"):
            _ = (
                source.lazy()
                .select(col("u").cast("int8"))
                .collect(accelerator=runtime)
            )
        var underflow = source.lazy().select(col("f") * lit(Float32(1e-20)))
        assert_true(
            underflow.collect(accelerator=runtime).equals(
                underflow.collect(engine="cpu")
            )
        )
        var boundary = DataFrame(
            [
                Series(
                    "u",
                    Column[UInt64]([(UInt64(1) << 63) + (UInt64(1) << 39) - 1]),
                )
            ]
        )
        var rounded_cast = boundary.lazy().select(col("u").cast("float32"))
        assert_true(
            rounded_cast.collect(accelerator=runtime).equals(
                rounded_cast.collect(engine="cpu")
            )
        )
        check_domain(
            plan,
            DataFrame(
                [
                    Series("abs", Column[Int8]([Int8.MIN, Int8.MIN, 2, 3])),
                    Series("cast", Column[Int8]([0, 0, 2, 3])),
                    Series(
                        "product",
                        Column[Float32](
                            [1e-20, 1e-20, expected_product, expected_product]
                        ),
                    ),
                    Series(
                        "nullable",
                        Column[Int8]([0, 0, 2, 3], [False, False, True, True]),
                    ),
                ]
            ),
            runtime,
            native,
        )


def test_native_metal_branch_observation_masks_and_precision_errors_leave_runtime_reusable() raises:
    masked_cases(require_metal(), True)


def verify_metal_numeric_cpu_models() raises:
    var runtime = MetalRuntime()
    integer_numeric_cases(runtime, False)
    masked_cases(runtime, False)
    float_value_cases(runtime, False)
    checked_boundary_preflight()


def float_value_cases(runtime: MetalRuntime, native: Bool) raises:
    var tiny = bitcast[DType.float32](UInt32(1))
    var nan = bitcast[DType.float32](UInt32(0x7FC00031))
    var positive_inf = bitcast[DType.float32](UInt32(0x7F800000))
    var negative_inf = bitcast[DType.float32](UInt32(0xFF800000))
    var raw: List[Float32] = [
        tiny,
        -tiny,
        -0.0,
        0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        nan,
        positive_inf,
        negative_inf,
        99,
    ]
    var bits: List[Bool] = [
        True,
        True,
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
    ]
    var source = DataFrame([Series("x", Column[Float32](raw^, bits))])
    var expected = DataFrame(
        [
            Series(
                "abs",
                Column[Float32](
                    [
                        tiny,
                        tiny,
                        0.0,
                        0.0,
                        0.5,
                        0.5,
                        1.5,
                        1.5,
                        nan,
                        positive_inf,
                        positive_inf,
                        0,
                    ],
                    bits,
                ),
            ),
            Series(
                "floor",
                Column[Float32](
                    [
                        0.0,
                        -1,
                        -0.0,
                        0.0,
                        0,
                        -1,
                        1,
                        -2,
                        nan,
                        positive_inf,
                        negative_inf,
                        0,
                    ],
                    bits,
                ),
            ),
            Series(
                "ceil",
                Column[Float32](
                    [
                        1,
                        -0.0,
                        -0.0,
                        0.0,
                        1,
                        -0.0,
                        2,
                        -1,
                        nan,
                        positive_inf,
                        negative_inf,
                        0,
                    ],
                    bits,
                ),
            ),
            Series(
                "round",
                Column[Float32](
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1,
                        -1,
                        2,
                        -2,
                        nan,
                        positive_inf,
                        negative_inf,
                        0,
                    ],
                    bits,
                ),
            ),
            Series(
                "nan",
                Column[Bool](
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                    ],
                    bits,
                ),
            ),
            Series(
                "finite",
                Column[Bool](
                    [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        False,
                        False,
                        False,
                        False,
                    ],
                    bits,
                ),
            ),
            Series(
                "infinite",
                Column[Bool](
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True,
                        False,
                    ],
                    bits,
                ),
            ),
            Series(
                "positive",
                Column[Bool](
                    [
                        True,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                        False,
                        True,
                        False,
                        False,
                    ],
                    bits,
                ),
            ),
            Series(
                "clip",
                Column[Float32](
                    [tiny, -tiny, -0.0, 0.0, 0.5, -0.5, 1, -1, nan, 1, -1, 0],
                    bits,
                ),
            ),
            Series(
                "fill",
                Column[Float32](
                    [
                        tiny,
                        -tiny,
                        -0.0,
                        0.0,
                        0.5,
                        -0.5,
                        1.5,
                        -1.5,
                        2,
                        positive_inf,
                        negative_inf,
                        0,
                    ],
                    bits,
                ),
            ),
        ]
    )
    check_domain(
        source.lazy().select_exprs(
            [
                col("x").abs().alias("abs"),
                col("x").floor().alias("floor"),
                col("x").ceil().alias("ceil"),
                # Whole-number rounding promises half-away values, while the
                # sign of an exact zero is unspecified. Normalize only this
                # result; other columns retain their exact signed-zero policy.
                when(col("x").round() == lit(Float32(0)))
                .then(lit(Float32(0)))
                .otherwise(col("x").round())
                .alias("round"),
                col("x").is_nan().alias("nan"),
                col("x").is_finite().alias("finite"),
                col("x").is_infinite().alias("infinite"),
                (col("x") > lit(Float32(0))).alias("positive"),
                col("x").clip(lit(Float32(-1)), lit(Float32(1))).alias("clip"),
                col("x").fill_nan(lit(Float32(2))).alias("fill"),
            ]
        ),
        expected,
        runtime,
        native,
    )
    var separate = DataFrame(
        [
            Series(
                "x",
                Column[Float32](
                    [Float32(1) + Float32(0.00000011920928955078125)]
                ),
            )
        ]
    )
    check_domain(
        separate.lazy().select(
            (
                (
                    col("x")
                    * lit(Float32(1) - Float32(0.00000011920928955078125))
                )
                - lit(Float32(1))
            ).alias("result")
        ),
        DataFrame([Series("result", Column[Float32]([0.0]))]),
        runtime,
        native,
    )


def test_native_metal_float32_encodings_and_separate_operator_rounding_match_literal_results() raises:
    float_value_cases(require_metal(), True)


def checked_boundary_preflight() raises:
    var unavailable = MetalRuntime(device_id=999999)
    var source = DataFrame([Series("x", Column[Int64]([1, 2]))])
    with assert_raises(contains="terminal projections without head"):
        _ = (
            source.lazy()
            .select(col("x") + 1)
            .head(0)
            .collect(accelerator=unavailable)
        )
    with assert_raises(contains="strict casts require terminal"):
        _ = (
            source.lazy()
            .with_columns(col("x").cast("int8").alias("n"))
            .filter(col("n") > lit(Int8(0)))
            .collect(accelerator=unavailable)
        )
    with assert_raises(contains="terminal boundary"):
        _ = (
            source.lazy()
            .select(col("x").sum())
            .join(source.lazy(), "x")
            .collect(accelerator=unavailable)
        )
    with assert_raises(contains="terminal boundary"):
        _ = (
            source.lazy()
            .select(col("x").sum())
            .head(0)
            .collect(accelerator=unavailable)
        )
    var float_source = DataFrame([Series("f", Column[Float32]([1.5]))])
    with assert_raises(contains="Float64 intermediate"):
        _ = (
            float_source.lazy()
            .select(col("f").round(2))
            .collect(accelerator=unavailable)
        )


def test_native_metal_checked_operation_boundaries_reject_before_device_initialization() raises:
    _ = require_metal()
    checked_boundary_preflight()
