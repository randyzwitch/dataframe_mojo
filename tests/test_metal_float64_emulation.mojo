"""Integer-only binary64 arithmetic through the public Metal query API."""
from std.memory import bitcast
from std.testing import TestSuite, assert_true, assert_equal, assert_raises
from dataframe import DataFrame, Series, Column, Expr, col, lit, when
from dataframe.dtype import DataType
from dataframe.metal import MetalRuntime, metal_installed


def fixture() raises -> DataFrame:
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        1,
        0x8000000000000001,
        0x000FFFFFFFFFFFFF,
        0x0010000000000000,
        0x3690000000000000,
        0x36A0000000000000,
        0x380FFFFFC0000000,
        0x3810000000000000,
        0x3CA0000000000000,
        0x3FEFFFFFFFFFFFFF,
        0x3FF0000000000000,
        0x3FF0000000000001,
        0x4000000000000000,
        0xBFF0000000000000,
        0x4340000000000000,
        0x7FEFFFFFFFFFFFFF,
        0xFFEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000042,
        0x7FF0000000000042,
    ]
    var left = List[Float64]()
    var right = List[Float64]()
    var valid = List[Bool]()
    var keys = List[Int32]()
    for a in words:
        for b in words:
            left.append(bitcast[DType.float64](a))
            right.append(bitcast[DType.float64](b))
            valid.append(len(left) % 7 != 0)
            keys.append(Int32(len(left) % 5))
    return DataFrame(
        [
            Series("a", Column[Float64](left^, valid^)),
            Series("b", Column[Float64](right^)),
            Series("k", Column[Int32](keys^)),
        ]
    )


def same(actual: DataFrame, expected: DataFrame) raises:
    if not actual.equals(expected):
        print("Float64 actual:", actual)
        print("Float64 expected:", expected)
    assert_true(actual.equals(expected))
    for name in actual.columns():
        if actual.column(name).dtype() == DataType.FLOAT32:
            var a = actual.column(name).float32()
            var b = expected.column(name).float32()
            for i in range(actual.height()):
                if not b.is_valid(i):
                    continue
                var bits = bitcast[DType.uint32](b._get(i))
                if (bits & 0x7FFFFFFF) <= 0x7F800000:
                    assert_equal(bitcast[DType.uint32](a._get(i)), bits)
            continue
        if actual.column(name).dtype() != DataType.FLOAT64:
            continue
        var a = actual.column(name).float64()
        var b = expected.column(name).float64()
        for i in range(actual.height()):
            assert_equal(a.is_valid(i), b.is_valid(i))
            if not a.is_valid(i):
                continue
            var bits = bitcast[DType.uint64](b._get(i))
            # Native hosts differ in NaN sign/payload after arithmetic. All
            # other encodings, including signed zeros, must be identical.
            if (bits & 0x7FFFFFFFFFFFFFFF) <= 0x7FF0000000000000:
                assert_equal(bitcast[DType.uint64](a._get(i)), bits)


def test_float64_arithmetic_rounding_subnormals_and_nulls() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = fixture()
    var query = frame.lazy().select_exprs(
        [
            (col("a") + col("b")).alias("add"),
            (col("a") - col("b")).alias("subtract"),
            (col("a") * col("b")).alias("multiply"),
            (col("a") / col("b")).alias("divide"),
            (col("a") // col("b")).alias("floor_divide"),
            (-col("a")).alias("negative"),
            col("a").abs().alias("absolute"),
            col("a").floor().alias("floor"),
            col("a").ceil().alias("ceil"),
            col("a").round(0).alias("round"),
            ((col("a") + col("b")) * lit(Float64(0.5))).alias("composed"),
        ]
    )
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    var empty = frame.slice(0, 0).lazy().select(col("a") + col("b"))
    same(empty.collect(accelerator=runtime), empty.collect(engine="cpu"))


def test_float64_predicates_nan_filling_and_clipping() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = fixture()
    var query = frame.lazy().select_exprs(
        [
            (col("a") == col("b")).alias("equal"),
            (col("a") != col("b")).alias("different"),
            (col("a") < col("b")).alias("less"),
            (col("a") <= col("b")).alias("less_equal"),
            (col("a") > col("b")).alias("greater"),
            (col("a") >= col("b")).alias("greater_equal"),
            col("a").is_nan().alias("nan"),
            col("a").is_not_nan().alias("not_nan"),
            col("a").is_finite().alias("finite"),
            col("a").is_infinite().alias("infinite"),
            col("a").fill_nan(col("b")).alias("filled"),
            col("a").clip(lit(Float64(-1)), lit(Float64(1))).alias("clipped"),
        ]
    )
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))


def test_float64_emulation_filters_conditionals_and_join_regions() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = fixture()
    var query = (
        frame.lazy()
        .with_columns((col("a") + col("b")).alias("sum"))
        .filter(col("sum") > lit(Float64(0)))
        .sort("k")
        .select_exprs(
            [
                when(col("sum").is_finite())
                .then(col("sum") * col("b"))
                .otherwise(col("a"))
                .alias("value"),
            ]
        )
    )
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    var child = (
        frame.slice(0, 8)
        .lazy()
        .with_columns((col("a") * lit(Float64(2))).alias("a"))
    )
    var joined = child.join(frame.slice(2, 5).lazy(), "k")
    same(joined.collect(accelerator=runtime), joined.collect(engine="cpu"))


def test_float64_emulation_has_separate_automatic_cost_family() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = DataFrame([Series("x", Column[Float64]([1, 2, 3]))])
    for expression in List[Expr](
        [col("x") + lit(Float64(0.5)), col("x") // lit(Float64(0.5))]
    ):
        var query = frame.lazy().select(expression)
        var facts = runtime.estimate_auto(query)
        assert_true("metal-emulated-float64-v1" in facts.family)
        same(
            query.collect(engine="auto", accelerator=runtime),
            query.collect(engine="cpu"),
        )
    var narrow = DataFrame([Series("x", Column[Float32]([1, 2, 3]))])
    var reduction = narrow.lazy().select(col("x").sum())
    assert_true(
        "metal-emulated-float64-v1" in runtime.estimate_auto(reduction).family
    )
    assert_equal(
        runtime.estimate_auto(
            narrow.lazy().select(col("x") + lit(Float32(1)))
        ).family,
        "float32",
    )


def test_float64_numeric_casts_and_integer_division() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = fixture()
    var converted = List[Expr]()
    for dtype in List[DataType](
        [
            DataType.FLOAT32,
            DataType.INT8,
            DataType.INT16,
            DataType.INT32,
            DataType.INT64,
            DataType.UINT8,
            DataType.UINT16,
            DataType.UINT32,
            DataType.UINT64,
            DataType.BOOL,
        ]
    ):
        converted.append(col("a").cast(dtype, strict=False).alias(dtype.name()))
    var query = frame.lazy().select_exprs(converted)
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    with assert_raises(contains="strict cast failed"):
        _ = (
            frame.lazy()
            .select(col("a").cast(DataType.INT64))
            .collect(accelerator=runtime)
        )
    with assert_raises(contains="strict cast failed"):
        _ = (
            frame.lazy()
            .select(col("a").cast(DataType.BOOL))
            .collect(accelerator=runtime)
        )
    var integers = DataFrame(
        [
            Series(
                "i",
                Column[Int64](
                    [
                        Int64.MIN,
                        Int64.MAX,
                        9007199254740993,
                        -9007199254740993,
                        0,
                        1,
                        -1,
                    ]
                ),
            ),
            Series(
                "u",
                Column[UInt64](
                    [
                        0,
                        UInt64.MAX,
                        9007199254740993,
                        9223372036854775808,
                        0,
                        1,
                        2,
                    ]
                ),
            ),
        ]
    )
    var casted = integers.lazy().select_exprs(
        [
            col("i").cast(DataType.FLOAT64).alias("signed"),
            col("u").cast(DataType.FLOAT64).alias("unsigned"),
            (col("i") / lit(Int64(3))).alias("signed_divide"),
            (col("u") / lit(UInt64(3))).alias("unsigned_divide"),
            (col("i") / lit(Int64(0))).alias("zero_divide"),
        ]
    )
    same(casted.collect(accelerator=runtime), casted.collect(engine="cpu"))
    var words: List[UInt32] = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x00800000,
        0x7F7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00042,
    ]
    var values = List[Float32]()
    for word in words:
        values.append(bitcast[DType.float32](word))
    var floats = DataFrame([Series("x", Column[Float32](values^))])
    var widened = floats.lazy().select(col("x").cast(DataType.FLOAT64))
    same(widened.collect(accelerator=runtime), widened.collect(engine="cpu"))
    var resident = (
        integers.lazy()
        .with_columns(col("i").cast(DataType.FLOAT64).alias("wide"))
        .filter(col("wide") >= lit(Float64(0)))
        .head(3)
    )
    same(resident.collect(accelerator=runtime), resident.collect(engine="cpu"))
    var boundary = DataFrame(
        [
            Series(
                "u",
                Column[UInt64](
                    [
                        (UInt64(1) << 63) + (UInt64(1) << 39) - 1,
                        (UInt64(1) << 63) + (UInt64(1) << 39),
                        (UInt64(1) << 63) + (UInt64(1) << 39) + 1,
                    ]
                ),
            )
        ]
    )
    var narrowed = (
        boundary.lazy()
        .with_columns(col("u").cast(DataType.FLOAT32).alias("narrow"))
        .head(2)
    )
    same(narrowed.collect(accelerator=runtime), narrowed.collect(engine="cpu"))


def test_float64_keys_extrema_and_joins() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var words: List[UInt64] = [
        0,
        0x8000000000000000,
        1,
        0x8000000000000001,
        0x3FF0000000000000,
        0xBFF0000000000000,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x7FF8000000000042,
        0x7FF8000000000043,
        0,
        0x3FF0000000000000,
    ]
    var keys = List[Float64]()
    var ids = List[Int32]()
    var valid = List[Bool]()
    for i in range(len(words)):
        keys.append(bitcast[DType.float64](words[i]))
        ids.append(Int32(i))
        valid.append(i != 11)
    var frame = DataFrame(
        [
            Series("f", Column[Float64](keys^, valid^)),
            Series("id", Column[Int32](ids^)),
        ]
    )
    for descending in List[Bool]([False, True]):
        for nulls_last in List[Bool]([False, True]):
            var ordered = frame.lazy().sort(
                ["f"], descending=descending, nulls_last=nulls_last
            )
            same(
                ordered.collect(accelerator=runtime),
                ordered.collect(engine="cpu"),
            )
    var extrema = frame.lazy().select_exprs(
        [
            col("f").min().alias("minimum"),
            col("f").max().alias("maximum"),
            col("f").arg_min().alias("arg_min"),
            col("f").arg_max().alias("arg_max"),
        ]
    )
    same(extrema.collect(accelerator=runtime), extrema.collect(engine="cpu"))
    for ordered in List[Bool]([False, True]):
        var grouped = (
            frame.lazy()
            .group_by("f", maintain_order=ordered)
            .agg([col("id").count()])
        )
        if not ordered:
            # Default group order is unspecified. Normalize the complete,
            # unlimited result; maintain_order=True is checked as returned.
            grouped = grouped.sort("f")
        same(
            grouped.collect(accelerator=runtime), grouped.collect(engine="cpu")
        )
    for how in List[String](["inner", "left", "right", "full", "semi", "anti"]):
        var joined = frame.lazy().join(
            frame.slice(1, 9).lazy(), on="f", how=how
        )
        same(joined.collect(accelerator=runtime), joined.collect(engine="cpu"))
    for keep in List[String](["first", "last", "none"]):
        var unique = frame.lazy().unique(["f"], keep=keep, maintain_order=True)
        same(unique.collect(accelerator=runtime), unique.collect(engine="cpu"))
    var large_keys = List[Float64]()
    var large_ids = List[Int32]()
    var large_valid = List[Bool]()
    for i in range(4097):
        large_keys.append(
            bitcast[DType.float64](words[(i // 128) % len(words)]) if i % 128
            == 0 else Float64(i - 2048)
        )
        large_ids.append(Int32(i))
        large_valid.append(i % 257 != 0)
    var large = DataFrame(
        [
            Series("f", Column[Float64](large_keys^, large_valid^)),
            Series("id", Column[Int32](large_ids^)),
        ]
    )
    var hashed = (
        large.lazy().group_by("f", maintain_order=True).agg([col("id").count()])
    )
    same(hashed.collect(accelerator=runtime), hashed.collect(engine="cpu"))
    var hashed_join = large.lazy().join(
        large.slice(17).lazy(), on="f", how="full"
    )
    same(
        hashed_join.collect(accelerator=runtime),
        hashed_join.collect(engine="cpu"),
    )


def test_float64_accumulation_scalar_grouped_and_resident() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    for n in List[Int]([0, 1, 257, 4097]):
        var wide = List[Float64]()
        var narrow = List[Float32]()
        var keys = List[Int32]()
        var valid = List[Bool]()
        for i in range(n):
            wide.append(Float64(i % 31 - 15) * 0.25)
            narrow.append(Float32(i % 31 - 15) * Float32(0.25))
            keys.append(Int32(i % 11))
            valid.append(i % 7 != 2)
        var frame = DataFrame(
            [
                Series("f", Column[Float64](wide^, valid)),
                Series("s", Column[Float32](narrow^, valid)),
                Series("k", Column[Int32](keys^)),
            ]
        )
        var selected = frame.lazy().select_exprs(
            [
                col("f").sum().alias("wide_sum"),
                col("s").sum().alias("narrow_sum"),
                col("f").mean().alias("wide_mean"),
                col("s").mean().alias("narrow_mean"),
                col("f").sum(min_count=n + 1).alias("not_enough"),
            ]
        )
        same(
            selected.collect(accelerator=runtime),
            selected.collect(engine="cpu"),
        )
        var filtered = (
            frame.lazy().filter(col("k") < lit(Int32(3))).select(col("f").sum())
        )
        same(
            filtered.collect(accelerator=runtime),
            filtered.collect(engine="cpu"),
        )
        var grouped = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(
                [
                    col("f").sum().alias("wide_sum"),
                    col("s").sum().alias("narrow_sum"),
                    col("f").mean().alias("wide_mean"),
                    col("s").mean().alias("narrow_mean"),
                ]
            )
        )
        same(
            grouped.collect(accelerator=runtime), grouped.collect(engine="cpu")
        )
        var consumed = grouped.with_columns(
            (col("wide_sum") + col("narrow_mean")).alias("value")
        )
        same(
            consumed.collect(accelerator=runtime),
            consumed.collect(engine="cpu"),
        )
    var specials = (
        fixture().lazy().select_exprs([col("a").sum(), col("b").mean()])
    )
    same(specials.collect(accelerator=runtime), specials.collect(engine="cpu"))


def test_integer_mean_preserves_exact_accumulation_before_float64_division() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "i",
                Column[Int64]([-9007199254740993, 9007199254740993, 1, 2, 3]),
            ),
            Series("u", Column[UInt64]([9007199254740993, 1, 2, 3, 4])),
            Series("k", Column[Int32]([0, 0, 1, 1, 1])),
        ]
    )
    var query = frame.lazy().select_exprs([col("i").mean(), col("u").mean()])
    same(query.collect(accelerator=runtime), query.collect(engine="cpu"))
    var grouped = (
        frame.lazy().group_by("k").agg([col("i").mean(), col("u").mean()])
    )
    same(grouped.collect(accelerator=runtime), grouped.collect(engine="cpu"))
    var unproved = DataFrame(
        [Series("i", Column[Int64]([Int64.MIN, Int64.MAX]))]
    )
    with assert_raises(contains="accumulator"):
        _ = unproved.lazy().select(col("i").mean()).collect(accelerator=runtime)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
