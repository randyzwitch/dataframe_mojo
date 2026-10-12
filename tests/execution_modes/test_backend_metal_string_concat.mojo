"""Bounded dictionary pairs for resident row-wise String concatenation."""
from tests.execution_modes.test_backend_metal import require_metal
from std.testing import assert_true, assert_equal, assert_raises
from dataframe import (
    DataFrame,
    Series,
    Column,
    StringColumn,
    col,
    lit,
    when,
    concat_str,
)
from dataframe.accelerator.dictionaries import lower_dictionary_operations
from dataframe.metal import MetalRuntime
from dataframe.lazy import LazyFrame


def concat_frame() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "a",
                StringColumn(
                    List[String](["a", "ab", "é", "", "null", "a", "🙂", "z"]),
                    List[Bool](
                        [True, True, True, True, False, True, True, True]
                    ),
                ),
            ),
            Series(
                "b",
                StringColumn(
                    List[String](["bc", "c", "🙂", "", "x", "bc", "null", "q"]),
                    List[Bool](
                        [True, True, True, True, True, True, False, True]
                    ),
                ),
            ),
            Series("v", Column[Int16]([1, 2, 3, 4, 5, 6, 7, 8])),
        ]
    )


def compare_concat(query: LazyFrame) raises:
    var expected = query.collect(engine="cpu")
    var lowered = lower_dictionary_operations(query)
    assert_true(lowered[0].collect(engine="cpu").equals(expected))
    assert_true(lowered[1] >= 0)
    assert_true(lowered[2] > 0)
    assert_true(query.collect(accelerator=MetalRuntime()).equals(expected))


def test_native_metal_concat_dictionary_pairs_nulls_unicode_literals_and_chains() raises:
    _ = require_metal()
    var frame = concat_frame()
    var expressions = [
        concat_str([col("a"), col("b")]),
        concat_str([col("a"), col("b")], separator="🙂é"),
        concat_str(
            [lit("prefix"), col("a"), lit(", "), col("b"), lit("suffix")]
        ),
        concat_str(
            [col("a").fill_null(lit("")), col("b").fill_null(lit(""))],
            separator=", ",
        ),
        concat_str([col("a").str().to_uppercase(), col("b").str().reverse()])
        .str()
        .len_chars(),
        concat_str(
            [
                when(col("v") > lit(Int16(3)))
                .then(col("a"))
                .otherwise(col("b")),
                col("a"),
            ]
        )
        .str()
        .to_uppercase(),
        concat_str([col("a"), lit(""), col("b")]).str().contains("abc"),
    ]
    for expression in expressions:
        for offset in range(8):
            var sliced = frame.slice(offset, 8 - offset)
            compare_concat(sliced.lazy().select(expression.alias("value")))
        compare_concat(
            frame.slice(0, 0).lazy().select(expression.alias("value"))
        )
    var nulls = frame.with_column(
        Series(
            "a",
            StringColumn(
                List[String](["", "", "", "", "", "", "", ""]),
                List[Bool](
                    [False, False, False, False, False, False, False, False]
                ),
            ),
        )
    )
    compare_concat(
        nulls.lazy().select(concat_str([col("a"), col("b")]).alias("value"))
    )


def test_native_metal_concat_dictionary_domains_replacement_aliases_and_internal_names() raises:
    _ = require_metal()
    var frame = concat_frame().with_column(
        Series(
            "__dataframe_metal_dictionary_key_0_rhs",
            Column[Int16]([1, 2, 3, 4, 5, 6, 7, 8]),
        )
    )
    var replaced = frame.lazy().with_columns(
        [col("b").str().to_uppercase().alias("a")]
    )
    compare_concat(
        replaced.with_columns(
            [concat_str([col("a"), col("b")], separator="!").alias("a")]
        )
    )
    var renamed = frame.lazy().select_exprs(
        [col("a").alias("x"), col("b").alias("y")]
    )
    compare_concat(
        renamed.select(concat_str([col("x"), col("y")]).alias("name"))
    )
    var joined = frame.lazy().join(
        frame.lazy(), left_on=["v"], right_on=["v"], how="left"
    )
    compare_concat(
        joined.select(concat_str([col("a_right"), col("b")]).alias("name"))
    )


def test_native_metal_concat_dictionary_group_filter_join_order_and_duplicate_outputs() raises:
    _ = require_metal()
    var frame = concat_frame()
    var query = frame.lazy().with_columns(
        [concat_str([col("a"), col("b")]).alias("name")]
    )
    compare_concat(query)
    compare_concat(query.filter(col("name") == lit("abc")).sort("v"))
    compare_concat(
        query.group_by("name", maintain_order=True).agg(
            [col("v").sum().alias("s")]
        )
    )
    var other = DataFrame(
        [
            Series(
                "name", StringColumn(List[String](["abc", "é🙂", "", "missing"]))
            ),
            Series("w", Column[Int16]([10, 20, 30, 40])),
        ]
    )
    compare_concat(
        query.join(
            other.lazy(), left_on=["name"], right_on=["name"], how="full"
        )
    )
    compare_concat(query.sort("name").head(5))


def test_native_metal_concat_dictionary_pair_capacity_byte_budget_and_recovery() raises:
    _ = require_metal()
    var query = (
        concat_frame()
        .lazy()
        .select(concat_str([col("a"), col("b")]).alias("value"))
    )
    with assert_raises(contains="staging exceeds payload budget"):
        _ = lower_dictionary_operations(query, 1)
    for count in [800, 1100]:
        var values = List[String]()
        for i in range(count):
            values.append(String(i))
        var frame = DataFrame(
            [
                Series("a", StringColumn(values.copy())),
                Series("b", StringColumn(values^)),
            ]
        )
        var large = frame.lazy().select(
            concat_str([col("a"), col("b")]).alias("value")
        )
        with assert_raises(contains="capacity"):
            _ = lower_dictionary_operations(large)
    compare_concat(query)
    var runtime = MetalRuntime()
    var measured = query.profile(accelerator=runtime)
    assert_true(measured[0].equals(query.collect(engine="cpu")))
    assert_equal(
        measured[1].item(0, "dictionary_operation_tables").int64(), Int64(1)
    )
    assert_true(
        measured[1].item(0, "dictionary_operation_staging_bytes").int64() > 0
    )
    with assert_raises(contains="budget"):
        _ = query.collect(accelerator=MetalRuntime(memory_limit_bytes=1))
    assert_true(
        query.collect(accelerator=runtime).equals(query.collect(engine="cpu"))
    )


