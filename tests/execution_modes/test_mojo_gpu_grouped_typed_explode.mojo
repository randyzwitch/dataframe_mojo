"""Literal typed grouped→EXPLODE oracles, including downstream consumers."""
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
)
from dataframe.dtype import CategoricalDictionary
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime
from tests.public_contracts.test_grouped_key_domains import _key, _coded
from dataframe.accelerator.grouped_key_domains import STRING, BINARY, GEOMETRY


def _collect(query: LazyFrame) raises -> DataFrame:
    var expected = query.collect(engine="cpu", optimize=False, streaming=False)
    var actual = query.collect(
        accelerator=MetalRuntime(),
        engine="accel",
        optimize=False,
        streaming=False,
    )
    assert_true(actual.equals(expected))
    return actual^


def test_grouped_typed_expand_literal_raw_domains_and_consumers() raises:
    for domain in [STRING, BINARY, GEOMETRY]:
        var keys = _key(
            domain, [255, 1, 255, 1, 0], [True, True, True, True, False]
        )
        if domain == STRING:
            keys = Series(
                "k",
                StringColumn(
                    ["é", "a\x00b", "é", "a\x00b", ""],
                    [True, True, True, True, False],
                ),
            )
        var source = DataFrame(
            [keys.copy(), Series("v", Column[Int64]([10, 20, 30, 40, 50]))]
        )
        var expanded = (
            source.lazy()
            .group_by("k", maintain_order=True)
            .agg(col("v").implode().alias("items"))
            .explode(["items"])
        )
        var result = _collect(expanded)
        assert_equal(result.height(), 5)
        assert_equal(
            result.column("items").int64().to_list(), [10, 30, 20, 40, 50]
        )
        assert_equal(result.column("k").dtype(), source.column("k").dtype())
        assert_true(
            result.column("k").equals(source.column("k").take([0, 2, 1, 3, 4]))
        )
        if domain == STRING:
            var matched = _collect(expanded.filter(col("k") == lit("a\x00b")))
            assert_equal(matched.column("items").int64().to_list(), [20, 40])
        var grouped = _collect(
            expanded.group_by("k", maintain_order=True).agg(
                col("items").sum().alias("answer")
            )
        )
        assert_equal(grouped.column("answer").int64().to_list(), [40, 60, 50])
        var over = _collect(
            expanded.with_columns(col("items").sum().over("k").alias("total"))
        )
        assert_equal(
            over.column("total").int64().to_list(), [40, 40, 60, 60, 50]
        )
        var projected = _collect(
            expanded.select_exprs(
                [
                    col("k").alias("key"),
                    (col("items") + lit(Int64(1))).alias("next"),
                ]
            ).head(3)
        )
        assert_equal(projected.column("next").int64().to_list(), [11, 31, 21])
        var right = (
            source.lazy()
            .group_by("k", maintain_order=True)
            .agg(col("v").sum().alias("right"))
        )
        var joined = _collect(expanded.join(right, on="k", how="left"))
        assert_equal(joined.height(), 5)


def test_grouped_typed_expand_duplicate_categorical_and_coded_string_identity() raises:
    var dictionary = CategoricalDictionary()
    dictionary.append("same")
    dictionary.append("same")
    dictionary.append("other")
    var category = Series("k", Column[UInt32]([0, 1, 0, 1, 2])).with_dtype(
        DataType.categorical(dictionary^)
    )
    for coded in [False, True]:
        var keys = _coded(category) if coded else category.copy()
        var frame = DataFrame(
            [keys.copy(), Series("v", Column[Int64]([10, 20, 30, 40, 50]))]
        )
        var expanded = (
            frame.lazy()
            .group_by("k", maintain_order=True)
            .agg(col("v").implode().alias("items"))
            .explode(["items"])
        )
        var output = _collect(expanded)
        assert_equal(
            output.column("items").int64().to_list(), [10, 30, 20, 40, 50]
        )
        assert_equal(output.column("k").dtype(), frame.column("k").dtype())
        if coded:
            assert_true(not Bool(output.column("k")._dictionary_codes()))
            assert_true(not Bool(output.column("k")._field_metadata))
        else:
            assert_true(
                output.column("k").dtype().dictionary().ptr()
                == category.dtype().dictionary().ptr()
            )
        var grouped = _collect(
            expanded.group_by("k", maintain_order=True).agg(
                col("items").sum().alias("answer")
            )
        )
        assert_equal(grouped.height(), 2 if coded else 3)
        var expected_grouped: List[Int64] = [100, 50] if coded else [40, 60, 50]
        assert_equal(
            grouped.column("answer").int64().to_list(), expected_grouped
        )
        var over = _collect(
            expanded.with_columns(col("items").sum().over("k").alias("total"))
        )
        var expected_over: List[Int64] = [
            100,
            100,
            100,
            100,
            50,
        ] if coded else [40, 40, 60, 60, 50]
        assert_equal(over.column("total").int64().to_list(), expected_over)


def test_grouped_typed_owner_contract_metadata_without_device_context() raises:
    from tests.support.grouped_typed_owner_reference import (
        scoped_grouped_typed_owner_contract,
    )

    scoped_grouped_typed_owner_contract()
