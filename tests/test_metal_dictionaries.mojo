"""String and categorical operations share ordered codes on Metal."""
from std.testing import TestSuite, assert_true, assert_equal, assert_raises
from dataframe import DataFrame, Series, StringColumn, Column, col, lit
from dataframe.categorical import encode
from dataframe.metal import MetalRuntime, metal_installed


def test_string_filter_sort_group_and_join() raises:
    assert_true(metal_installed())
    var runtime = MetalRuntime()
    var frame = DataFrame([
        Series("k", StringColumn(List[String](["z", "a", "é", "", "a", "null"]), List[Bool]([True, True, True, True, True, False]))),
        Series("v", Column[Int32]([1, 2, 3, 4, 5, 6])),
    ])
    var filtered = frame.lazy().filter(col("k") < lit("z")).sort("k")
    assert_true(filtered.collect(accelerator=runtime).equals(filtered.collect(engine="cpu")))
    var grouped = frame.lazy().group_by("k", maintain_order=True).agg(col("v").sum())
    assert_true(grouped.collect(accelerator=runtime).equals(grouped.collect(engine="cpu")))
    var right = DataFrame([Series("k", StringColumn(List[String](["a", "z", "missing"]))), Series("w", Column[Int32]([10, 20, 30]))])
    var joined = frame.lazy().join(right.lazy(), "k", how="left")
    assert_true(joined.collect(accelerator=runtime).equals(joined.collect(engine="cpu")))
    with assert_raises(contains="dictionary staging"):
        _ = filtered.collect(accelerator=MetalRuntime(memory_limit_bytes=0))


def test_categorical_projection_preserves_original_dictionary() raises:
    assert_true(metal_installed())
    var runtime = MetalRuntime()
    var source = encode(Series("k", StringColumn(List[String](["z", "a", "z", "é"]))))
    var frame = DataFrame([source.copy()])
    var query = frame.lazy().sort("k").head(3)
    assert_true(query.collect(accelerator=runtime).equals(query.collect(engine="cpu")))


def test_different_categorical_dictionaries_join_all_styles() raises:
    assert_true(metal_installed())
    var runtime = MetalRuntime(memory_limit_bytes=64 * 1024 * 1024)
    var a = encode(Series("k", StringColumn(List[String](["z", "a", "z", "left", "null"]), List[Bool]([True, True, True, True, False]))))
    var b = encode(Series("k", StringColumn(List[String](["a", "z", "right", "null"]), List[Bool]([True, True, True, False]))))
    assert_true(a.dtype() != b.dtype())
    var left = DataFrame([a.copy(), Series("v", Column[Int32]([1, 2, 3, 4, 5]))])
    var right = DataFrame([b.copy(), Series("w", Column[Int32]([10, 20, 30, 40]))])
    for how in List[String](["inner", "left", "right", "full", "semi", "anti", "cross"]):
        var query = left.lazy().join(right.lazy(), "k", how=how) if how != "cross" else left.lazy().join(right.lazy(), how="cross")
        var actual = query.collect(accelerator=runtime)
        var expected = query.collect(engine="cpu")
        assert_true(actual.equals(expected))
        assert_equal(actual.column("k").dtype(), expected.column("k").dtype())


def test_dictionary_profile_includes_host_staging_and_restoration() raises:
    assert_true(metal_installed())
    var frame = DataFrame([Series("k", StringColumn(List[String](["long original value", "a", "b"])))])
    var measured = frame.lazy().sort("k").profile(accelerator=MetalRuntime())
    assert_true(measured[0].equals(frame.lazy().sort("k").collect(engine="cpu")))
    var report = measured[1].copy()
    assert_true(report.item(0, "dictionary_staging_bytes").int64() > 0)
    var encode_ms = report.item(0, "dictionary_encode_ms").float64()
    var decode_ms = report.item(0, "dictionary_decode_ms").float64()
    assert_true(report.item(0, "query_wall_ms").float64() >= encode_ms + decode_ms)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
