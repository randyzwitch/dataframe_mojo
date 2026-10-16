"""Shared Mojo normalization must precede direct GPU dispatch."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    Series,
    StringColumn,
    Expr,
    col,
    lit,
    when,
    concat_str,
    all,
    by_dtype,
    nth,
)
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from dataframe.accelerator.dictionaries import DictionaryCodes, encode_query
from dataframe.accelerator.metadata import schema_frame
from tests.support.coverage_compare import assert_frame


def fixture() raises -> DataFrame:
    return DataFrame(
        [
            Series("id", Column[Int32]([0, 1, 2, 3, 4, 5])),
            Series("g", Column[Int32]([0, 1, 0, 1, 0, 1])),
            Series(
                "x",
                Column[Float64](
                    [1, 2, 3, 4, 5, 6], [True, False, True, True, True, True]
                ),
            ),
            Series(
                "s",
                StringColumn(
                    ["a", "é🙂", "a", "", "poison", "é🙂"],
                    [True, True, True, True, False, True],
                ),
            ),
        ]
    )


def check(query: LazyFrame) raises:
    assert_frame(
        query.collect(accelerator=MetalRuntime(), engine="accel"),
        query.collect(engine="cpu", streaming=False),
    )


# requires: mojo_gpu
def test_mojo_gpu_normalization_dictionary_unicode_operations_and_selectors() raises:
    var frame = fixture()
    var expressions: List[Expr] = [
        col("s").str().to_uppercase().alias("upper"),
        col("s").str().reverse().alias("reverse"),
        col("s").str().len_chars().alias("chars"),
        col("s").str().len_bytes().alias("bytes"),
        col("s").str().contains("🙂").alias("contains"),
        concat_str([col("s"), lit("!")]).alias("concat"),
        col("s").fill_null(lit("missing")).alias("filled"),
    ]
    for rows in [0, 1, frame.height()]:
        check(frame.head(rows).lazy().select_exprs(expressions))
        check(frame.head(rows).lazy().select(all().name_prefix("copy_")))
    check(frame.lazy().select(nth(-1).str().to_uppercase().alias("upper")))
    check(frame.lazy().with_columns(by_dtype(["float64"]) + lit(Float64(2))))
    var projected = frame.lazy().with_columns(
        concat_str([col("s"), lit("!")]).alias("name")
    )
    check(projected.filter(col("name").str().contains("é")).sort("id"))
    check(projected.group_by("name", maintain_order=True).agg(col("x").sum()))
    check(projected.join(projected, ["name"], how="full"))
    var measured = projected.profile(accelerator=MetalRuntime())
    assert_true(measured[1].item(0, "dictionary_staging_bytes").int64() > 0)
    assert_equal(measured[0].item(1, "name").string(), "é🙂!")
    with assert_raises(contains="budget"):
        _ = projected.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    check(projected)

    # Independently account simultaneous dictionary owners across execution
    # and restoration, keeping the inner region's requested peak unchanged.
    var query = frame.lazy().select(col("s"))
    var dictionary = DictionaryCodes(query)
    var encoded = MetalRuntime().execute_profiled(
        encode_query(query, dictionary)
    )
    var region_peak = Int(encoded[1].item(0, "peak_requested_bytes").int64())
    var staging = dictionary.staging_bytes(query)
    var restoration = dictionary.restoration_bytes(
        encoded[0], schema_frame(query, len(query._nodes) - 1)
    )
    var expected_peak = staging + max(region_peak, restoration)
    var normalized = query.profile(accelerator=MetalRuntime())
    assert_equal(
        normalized[1].item(0, "query_peak_requested_bytes").int64(),
        Int64(expected_peak),
    )
    assert_equal(
        normalized[1].item(0, "peak_requested_bytes").int64(),
        Int64(region_peak),
    )
    var exact = query.profile(
        accelerator=MetalRuntime(memory_limit_bytes=expected_peak)
    )
    assert_frame(exact[0], normalized[0])
    assert_equal(
        exact[1].item(0, "query_peak_requested_bytes").int64(),
        Int64(expected_peak),
    )
    with assert_raises(contains="budget"):
        _ = query.collect(
            accelerator=MetalRuntime(memory_limit_bytes=expected_peak - 1)
        )


# requires: mojo_gpu
def test_mojo_gpu_normalization_union_distinct_nulls_and_row_order() raises:
    var frame = fixture()
    var query = (
        frame.lazy()
        .concat(frame.slice(1, 3).lazy())
        .concat(frame.head(2).lazy())
    )
    check(query)
    check(query.slice(-5, 3))
    for keep in [String("first"), String("last"), String("none")]:
        var unique = query.unique(["s"], keep=keep, maintain_order=True)
        check(unique)
        check(unique.slice(-2, 1))
    var first = query.unique(["s"], keep="first", maintain_order=True).collect(
        accelerator=MetalRuntime()
    )
    assert_equal(first.height(), 4)
    for row in range(4):
        assert_equal(first.item(row, "id").int32(), Int32([0, 1, 3, 4][row]))
    var singleton = query.unique(
        ["s"], keep="none", maintain_order=True
    ).collect(accelerator=MetalRuntime())
    assert_equal(singleton.height(), 1)
    assert_equal(singleton.item(0, "id").int32(), Int32(4))


# requires: mojo_gpu
def test_mojo_gpu_normalization_distinct_counts_partition_and_aggregate_compositions() raises:
    var frame = fixture()
    check(
        frame.lazy().select_exprs(
            [
                col("s").n_unique().alias("distinct"),
                col("x").count().alias("count"),
                col("x").sum().alias("sum"),
            ]
        )
    )
    check(
        frame.lazy()
        .group_by("g", maintain_order=True)
        .agg(
            [
                (col("s").n_unique() + col("x").count()).alias("combined"),
                col("x").mean().alias("mean"),
            ]
        )
    )
    var partitioned = frame.lazy().with_columns(
        [
            col("x").sum().over("g").alias("total"),
            col("x").max().over("s").alias("maximum"),
        ]
    )
    check(partitioned)
    check(
        partitioned.filter(col("total") > lit(Float64(5))).select(
            col("maximum").sum()
        )
    )
    check(
        frame.lazy().select(
            (col("x").sum() / col("x").count().cast("float64")).alias("average")
        )
    )
    check(frame.head(0).lazy().select(col("s").n_unique()))


# requires: mojo_gpu
def test_mojo_gpu_normalization_scalar_literals_text_parsing_and_empty_shapes() raises:
    var frame = fixture()
    for rows in [0, 1, frame.height()]:
        var query = (
            frame.head(rows)
            .lazy()
            .select_exprs(
                [
                    lit(Int32(7)).alias("value"),
                    lit("β🙂").alias("text"),
                    lit("2000-02-29T12:34:56.123456")
                    .str()
                    .strptime("datetime[us]")
                    .alias("dt"),
                ]
            )
        )
        check(query)
        check(query.head(0))
        check(query.filter(col("value") == lit(Int32(7))))
        check(query.select(col("value").sum()))
        assert_equal(query.collect(accelerator=MetalRuntime()).height(), 1)
    var invalid = frame.lazy().select(col(["missing"]))
    with assert_raises(contains="missing"):
        _ = invalid.collect(accelerator=MetalRuntime(device_id=999999))
    assert_true(
        "unsupported"
        not in MetalRuntime(device_id=999999).describe(
            frame.lazy().select(all())
        )
    )
