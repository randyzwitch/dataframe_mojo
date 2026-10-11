"""Distinct row selection executes through resident Metal groups."""
from std.testing import TestSuite, assert_true, assert_raises
from std.memory import bitcast
from dataframe import DataFrame, Series, Column, StringColumn, col
from dataframe.metal import MetalRuntime, metal_installed


def fixture() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "key",
                Column[Int16](
                    [3, 1, 3, 2, 4, 4, 1, 9],
                    [True, True, True, True, False, False, True, True],
                ),
            ),
            Series(
                "value",
                Column[Int32](
                    [10, 20, 30, 40, 50, 60, 70, 80],
                    [False, True, True, True, True, True, True, True],
                ),
            ),
            Series(
                "text", StringColumn(["é", "b", "c", "d", "e", "f", "g", "h"])
            ),
            Series(
                "__dataframe_unique_count",
                Column[Bool](
                    [True, False, True, False, True, False, True, False]
                ),
            ),
        ]
    )


def test_unique_subset_keeps_first_and_singletons() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = fixture()
    for keep in List[String](["first", "last", "any", "none"]):
        for ordered in List[Bool]([False, True]):
            var query = frame.lazy().unique(
                ["key"], keep=keep, maintain_order=ordered
            )
            assert_true(
                query.collect(accelerator=runtime).equals(
                    query.collect(engine="cpu")
                )
            )
            var empty = (
                frame.slice(0, 0)
                .lazy()
                .unique(["key"], keep=keep, maintain_order=ordered)
            )
            assert_true(
                empty.collect(accelerator=runtime).equals(
                    empty.collect(engine="cpu")
                )
            )
            var composed = (
                query.filter(col("value").is_not_null())
                .select_exprs([col("text"), col("key")])
                .head(2)
            )
            assert_true(
                composed.collect(accelerator=runtime).equals(
                    composed.collect(engine="cpu")
                )
            )


def test_unique_whole_rows_and_string_keys() raises:
    if not metal_installed():
        return
    var runtime = MetalRuntime()
    var frame = DataFrame(
        [
            Series(
                "key",
                StringColumn(
                    ["é", "a", "é", "a", "x", "x"],
                    [True, True, True, True, False, False],
                ),
            ),
            Series("v", Column[UInt64]([UInt64.MAX, 1, UInt64.MAX, 1, 4, 4])),
        ]
    )
    for keep in List[String](["first", "last", "any", "none"]):
        var query = frame.lazy().unique(keep=keep, maintain_order=True)
        assert_true(
            query.collect(accelerator=runtime).equals(
                query.collect(engine="cpu")
            )
        )
        var grouped = query.group_by("key", maintain_order=True).agg(
            [col("v").count().alias("n")]
        )
        assert_true(
            grouped.collect(accelerator=runtime).equals(
                grouped.collect(engine="cpu")
            )
        )
        var joined = query.join(frame.lazy(), on="key", how="full")
        assert_true(
            joined.collect(accelerator=runtime).equals(
                joined.collect(engine="cpu")
            )
        )


def test_unique_float_nan_zero_and_recovery() raises:
    if not metal_installed():
        return
    var nan = bitcast[DType.float32](UInt32(0x7FC00000))
    var frame = DataFrame(
        [Series("x", Column[Float32]([nan, -0.0, nan, 0.0, 5.0]))]
    )
    var runtime = MetalRuntime()
    for keep in List[String](["first", "last", "any", "none"]):
        var query = frame.lazy().unique(keep=keep, maintain_order=True)
        assert_true(
            query.collect(accelerator=runtime).equals(
                query.collect(engine="cpu")
            )
        )
    with assert_raises(contains="keep must be"):
        _ = (
            frame.lazy()
            .unique(keep="invalid", maintain_order=True)
            .collect(accelerator=runtime)
        )
    var query = frame.lazy().unique(maintain_order=True)
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
