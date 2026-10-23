"""Default QCUT labels remain String across immutable resident text domains."""
# contract-group: execution.apple_parity
# contract-group: execution.backends
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, col, lit, when
from dataframe_metal.metal import MetalRuntime
from dataframe.dtype import CategoricalDictionary
from tests.support.coverage_compare import assert_frame
from tests.support.coverage_generators import offset_layout, chunk_layout


def _source() raises -> DataFrame:
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000042))
    var dictionary = CategoricalDictionary()
    for value in ["z", "a", "é🙂", "unused"]:
        dictionary.append(value)
    var category = Series("cat", Column[UInt32]([0, 1, 2, 0, 0, 1])).with_dtype(
        DataType.categorical(dictionary^)
    )
    return DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [0, 1, 2, 3, nan, 99], [True, True, True, True, True, False]
                ),
            ),
            Series("id", Column[Int64]([10, 11, 12, 13, 14, 15])),
            Series("text", Column[String](["z", "a", "é🙂", "m", "z", "a"])),
            Series("numeric", Column[UInt32]([0, 1, 2, 3, 4, 5])),
            category^,
        ]
    )


# requires: metal
def test_mojo_gpu_qcut_default_literal_labels_nulls_closure_and_layouts() raises:
    var runtime = MetalRuntime()
    var raw = _source()
    for layout in range(3):
        var source = (
            raw.copy() if layout
            == 0 else offset_layout(raw) if layout
            == 1 else chunk_layout(raw)
        )
        for closed in [False, True]:
            var query = source.lazy().select(
                col("x").qcut([0.5], left_closed=closed).alias("bin")
            )
            var actual = query.collect(accelerator=runtime)
            assert_frame(
                actual,
                query.collect(engine="cpu", optimize=False, streaming=False),
            )
            assert_equal(actual.column("bin").dtype(), DataType.STRING)
            for row in range(4):
                assert_equal(
                    actual.item(row, "bin").string(),
                    "[-inf, 1.5)" if closed
                    and row
                    < 2 else "[1.5, inf)" if closed else "(-inf, 1.5]" if row
                    < 2 else "(1.5, inf]",
                )
            assert_true(actual.item(4, "bin").is_null())
            assert_true(actual.item(5, "bin").is_null())
    for source in [raw.clear(), raw.head(1)]:
        var query = source.lazy().select(
            col("x").qcut(List[Float64]()).alias("bin")
        )
        var actual = query.collect(accelerator=runtime)
        assert_frame(
            actual, query.collect(engine="cpu", optimize=False, streaming=False)
        )
        assert_equal(actual.column("bin").dtype(), DataType.STRING)
        if source.height():
            assert_equal(actual.item().string(), "(-inf, inf]")


# requires: metal
def test_mojo_gpu_qcut_default_downstream_filter_join_and_domain_union() raises:
    var source = _source()
    var runtime = MetalRuntime()
    var q = source.lazy().with_columns(col("x").qcut([0.5]).alias("bin"))
    var filtered = q.filter(col("bin") == "(1.5, inf]").select(
        ["id", "bin", "text", "numeric"]
    )
    var actual = filtered.collect(accelerator=runtime)
    assert_frame(
        actual, filtered.collect(engine="cpu", optimize=False, streaming=False)
    )
    assert_equal(actual.item(0, "id").int64(), Int64(12))
    assert_equal(actual.item(1, "id").int64(), Int64(13))
    assert_equal(actual.item(0, "numeric").uint32(), UInt32(2))
    var mixed = (
        source.lazy()
        .with_columns(
            [
                col("x").qcut([0.5]).alias("bin"),
                col("x").qcut([0.5], ["low", "high"]).alias("custom"),
            ]
        )
        .select(["id", "bin", "custom", "text", "cat"])
    )
    actual = mixed.collect(accelerator=runtime)
    assert_frame(
        actual, mixed.collect(engine="cpu", optimize=False, streaming=False)
    )
    assert_equal(actual.item(0, "custom").string(), "low")
    assert_equal(actual.item(2, "custom").string(), "high")
    var right = DataFrame(
        [
            Series(
                "bin", Column[String](["(1.5, inf]", "absent", "(-inf, 1.5]"])
            ),
            Series("label", Column[String](["upper", "none", "lower"])),
        ]
    )
    var joined = q.join(right.lazy(), on=["bin"], how="left").select(
        ["id", "bin", "label", "text"]
    )
    actual = joined.collect(accelerator=runtime)
    assert_frame(
        actual, joined.collect(engine="cpu", optimize=False, streaming=False)
    )
    assert_equal(actual.item(0, "label").string(), "lower")
    assert_equal(actual.item(2, "label").string(), "upper")
    # Two different dynamic dictionaries coexist until their ordered union.
    var second = (
        source.lazy()
        .with_columns(col("x").qcut([0.25, 0.75]).alias("other"))
        .select(["id", "other"])
    )
    var both = q.join(second, on=["id"]).select(["id", "bin", "other", "text"])
    actual = both.collect(accelerator=runtime)
    assert_frame(
        actual, both.collect(engine="cpu", optimize=False, streaming=False)
    )
    assert_equal(actual.item(0, "other").string(), "(-inf, 0.75]")
    assert_equal(actual.item(1, "other").string(), "(0.75, 2.25]")
    assert_equal(actual.item(3, "other").string(), "(2.25, inf]")


# requires: metal
def test_mojo_gpu_qcut_default_breakpoints_over_keys_and_exact_controls() raises:
    var source = _source()
    var runtime = MetalRuntime()
    var packed = source.lazy().select(
        col("x").qcut([0.5], include_breaks=True).alias("cut")
    )
    var actual = packed.collect(accelerator=runtime)
    assert_frame(
        actual, packed.collect(engine="cpu", optimize=False, streaming=False)
    )
    assert_equal(
        actual.item(0, "cut").struct_field("category").string(), "(-inf, 1.5]"
    )
    assert_equal(
        bitcast[DType.uint64](
            actual.item(0, "cut").struct_field("breakpoint").float64()
        ),
        UInt64(0x3FF8000000000000),
    )
    var q = (
        source.lazy()
        .with_columns(col("x").qcut([0.5]).alias("bin"))
        .with_columns(col("id").sum().over("bin").alias("total"))
        .select(["id", "bin", "total"])
    )
    actual = q.collect(accelerator=runtime)
    assert_frame(
        actual, q.collect(engine="cpu", optimize=False, streaming=False)
    )
    assert_equal(actual.item(0, "total").int64(), Int64(21))
    assert_equal(actual.item(2, "total").int64(), Int64(25))
    # Same query graph/width/cardinality: only default labels download K edges.
    # The custom labels' last region remaps no codes, so it is a view of the
    # resident table (#857). The default labels' dictionary is built from the
    # edges, so that region remaps its text codes into it: two more
    # synchronizations, no more.
    var defaults = source.lazy().select(col("x").qcut([0.5]).alias("bin"))
    var custom = source.lazy().select(
        col("x").qcut([0.5], ["a", "b"]).alias("bin")
    )
    var d = defaults.profile(accelerator=runtime)
    var c = custom.profile(accelerator=runtime)
    var dd = d[1].item(0, "query_download_bytes").int64()
    var cd = c[1].item(0, "query_download_bytes").int64()
    assert_equal(dd - cd, Int64(8))
    assert_equal(
        d[1].item(0, "query_synchronizations").int64()
        - c[1].item(0, "query_synchronizations").int64(),
        Int64(2),
    )
    var peak = Int(d[1].item(0, "query_peak_requested_bytes").int64())
    var exact = MetalRuntime(memory_limit_bytes=peak)
    assert_frame(defaults.collect(accelerator=exact), d[0])
    var short = MetalRuntime(memory_limit_bytes=peak - 1)
    with assert_raises(contains="budget"):
        _ = defaults.collect(accelerator=short)
    assert_frame(defaults.collect(accelerator=runtime), d[0])


# requires: metal
def test_mojo_gpu_qcut_default_duplicates_signed_zero_and_ordered_failures() raises:
    var runtime = MetalRuntime()
    var source = DataFrame([Series("x", Column[Float64]([0, 0, 0, 1]))])
    var q = source.lazy().select(
        col("x").qcut([0, 0.25, 0.5], allow_duplicates=True).alias("bin")
    )
    var actual = q.collect(accelerator=runtime)
    assert_frame(
        actual, q.collect(engine="cpu", optimize=False, streaming=False)
    )
    assert_equal(actual.item(0, "bin").string(), "(-inf, 0]")
    assert_equal(actual.item(3, "bin").string(), "(0, inf]")
    var zeros = DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [
                        bitcast[DType.float64](UInt64(0x8000000000000000)),
                        bitcast[DType.float64](UInt64(0x8000000000000000)),
                        bitcast[DType.float64](UInt64(0x8000000000000000)),
                    ]
                ),
            )
        ]
    )
    q = zeros.lazy().select(
        col("x").qcut([0.5], include_breaks=True).alias("cut")
    )
    actual = q.collect(accelerator=runtime)
    assert_frame(
        actual, q.collect(engine="cpu", optimize=False, streaming=False)
    )
    assert_equal(
        bitcast[DType.uint64](
            actual.item(0, "cut").struct_field("breakpoint").float64()
        ),
        UInt64(0x8000000000000000),
    )
    var pair_query = (
        zeros.head(2)
        .lazy()
        .select(col("x").qcut([0.5], include_breaks=True).alias("cut"))
    )
    var pair = pair_query.collect(accelerator=runtime)
    assert_frame(
        pair, pair_query.collect(engine="cpu", optimize=False, streaming=False)
    )
    assert_equal(
        bitcast[DType.uint64](
            pair.item(0, "cut").struct_field("breakpoint").float64()
        ),
        UInt64(0),
    )
    var infinities = DataFrame(
        [Series("x", Column[Float64]([Float64("-inf"), Float64("inf")]))]
    )
    q = infinities.lazy().select(
        col("x").qcut([0.5], include_breaks=True).alias("cut")
    )
    actual = q.collect(accelerator=runtime)
    assert_frame(
        actual, q.collect(engine="cpu", optimize=False, streaming=False)
    )
    for row in range(2):
        assert_equal(
            bitcast[DType.uint64](
                actual.item(row, "cut").struct_field("breakpoint").float64()
            ),
            UInt64(0x7FF8000000000000),
        )
        assert_equal(
            actual.item(row, "cut").struct_field("category").string(),
            "(-inf, nan]",
        )
    var overflow = DataFrame([Series("x", Column[Int64]([Int64.MAX]))])
    q = overflow.lazy().select(
        (col("x") + lit(Int64(1))).qcut([-1]).alias("bin")
    )
    with assert_raises(contains="overflow"):
        _ = q.collect(engine="cpu", optimize=False, streaming=False)
    with assert_raises(contains="overflow"):
        _ = q.collect(accelerator=runtime)
    q = source.lazy().select(col("x").qcut([-1]).alias("bin"))
    with assert_raises(contains="between 0 and 1"):
        _ = q.collect(accelerator=runtime)
    assert_frame(
        source.lazy().select(col("x").qcut([0.5])).collect(accelerator=runtime),
        source.lazy()
        .select(col("x").qcut([0.5]))
        .collect(engine="cpu", optimize=False, streaming=False),
    )
