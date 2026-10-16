"""Arrow field metadata follows source fields, not computed result values."""
# contract-group: public.accelerator_staging
# contract-group: public.apple_parity
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, DataType, Series, StringColumn, col
from dataframe.lazy import LazyFrame
from dataframe.accelerator.capabilities import RowCapabilities
from dataframe.accelerator.rows import lower_rows
from dataframe.accelerator.metadata import schema_frame


def field_source() raises -> DataFrame:
    var key = Series("k", Column[Int64]([0, 1, 0, 1]))
    var value = Series(
        "x", Column[Int64]([30, 90, 10, 40], [True, False, True, True])
    )
    # One Arrow metadata pair k=<g|v>. These are distinct immutable owners.
    var key_metadata: List[UInt8] = [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        107,
        1,
        0,
        0,
        0,
        103,
    ]
    var value_metadata: List[UInt8] = [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        107,
        1,
        0,
        0,
        0,
        118,
    ]
    key._field_metadata = ArcPointer(key_metadata^)
    value._field_metadata = ArcPointer(value_metadata^)
    return DataFrame([key^, value^])


def assert_field_owner(actual: Series, original: Series) raises:
    assert_true(Bool(actual._field_metadata))
    assert_true(Bool(original._field_metadata))
    assert_equal(
        actual._field_metadata.value()[], original._field_metadata.value()[]
    )
    assert_true(
        actual._field_metadata.value().ptr()
        == original._field_metadata.value().ptr()
    )


def field_row_query(frame: DataFrame) raises -> LazyFrame:
    return (
        frame.lazy()
        .with_columns(col("x").alias("copy"))
        .with_columns((col("x") + 1).alias("computed"))
        .filter(col("k") >= 0)
        .select_exprs([col("copy").alias("alias"), col("computed"), col("k")])
        .sort("alias")
        .slice(1, 2)
    )


def field_group_query(frame: DataFrame) raises -> LazyFrame:
    return (
        frame.lazy()
        .group_by("k", maintain_order=True)
        .agg(
            [
                col("x").sum().alias("total"),
                col("x").first().alias("first"),
                col("x").last().alias("last"),
                col("x").min().alias("low"),
            ]
        )
    )


def field_text_source(categorical: Bool) raises -> DataFrame:
    var text = Series(
        "label",
        StringColumn(["z", "poison", "a", "z"], [True, False, True, True]),
    )
    if categorical:
        text = text.cast(DataType.CATEGORICAL)
    var text_metadata: List[UInt8] = [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        107,
        1,
        0,
        0,
        0,
        115,
    ]
    text._field_metadata = ArcPointer(text_metadata^)
    return field_source().with_column(text^)


def _metadata_capabilities(reuse: Bool) -> RowCapabilities:
    return RowCapabilities(
        "metadata",
        mixed_types=True,
        grouped=True,
        order=True,
        slices=True,
        extrema=True,
        extra_reductions=True,
        reuse_column_slots=reuse,
        deferred_integer_errors=True,
    )


def test_accelerator_field_metadata_bare_alias_movement_and_computed_contract() raises:
    var original = field_source()
    for empty in [False, True]:
        var frame = original.clear() if empty else original.copy()
        var query = field_row_query(frame)
        var expected = query.collect(engine="cpu", streaming=False)
        assert_equal(expected.height(), 0 if empty else 2)
        assert_field_owner(expected.column("alias"), original.column("x"))
        assert_field_owner(expected.column("k"), original.column("k"))
        assert_false(Bool(expected.column("computed")._field_metadata))
        if not empty:
            assert_equal(
                expected.column("alias").int64().to_list(),
                [Int64(30), Int64(40)],
            )
        for reuse in [False, True]:
            var plan = lower_rows(query, _metadata_capabilities(reuse))
            assert_equal(len(plan.outputs), 3)
            assert_equal(
                plan.outputs[0].field_metadata.value()[],
                original.column("x")._field_metadata.value()[],
            )
            assert_true(
                plan.outputs[0].field_metadata.value().ptr()
                == original.column("x")._field_metadata.value().ptr()
            )
            assert_false(Bool(plan.outputs[1].field_metadata))
            assert_true(
                plan.outputs[2].field_metadata.value().ptr()
                == original.column("k")._field_metadata.value().ptr()
            )


def test_accelerator_field_metadata_group_keys_and_reductions_follow_cpu() raises:
    var original = field_source()
    for empty in [False, True]:
        var frame = original.clear() if empty else original.copy()
        var query = field_group_query(frame)
        var expected = query.collect(engine="cpu", streaming=False)
        assert_field_owner(expected.column("k"), original.column("k"))
        for name in ["total", "first", "last", "low"]:
            assert_false(Bool(expected.column(name)._field_metadata))
        var plan = lower_rows(query, _metadata_capabilities(True))
        assert_true(
            plan.outputs[0].field_metadata.value().ptr()
            == original.column("k")._field_metadata.value().ptr()
        )
        for output in range(1, len(plan.outputs)):
            assert_false(Bool(plan.outputs[output].field_metadata))
        var scalar = frame.lazy().select_exprs(
            [col("x").sum().alias("total"), col("x").first().alias("first")]
        )
        expected = scalar.collect(engine="cpu", streaming=False)
        plan = lower_rows(scalar, _metadata_capabilities(True))
        for output in range(len(plan.outputs)):
            assert_false(Bool(expected._columns[output]._field_metadata))
            assert_false(Bool(plan.outputs[output].field_metadata))


def test_accelerator_field_metadata_dictionary_transport_and_computed_schema() raises:
    from dataframe.accelerator.dictionaries import (
        DictionaryCodes,
        encode_query,
        restore_frame,
    )

    for categorical in [False, True]:
        var original = field_text_source(categorical)
        for empty in [False, True]:
            var source = original.clear() if empty else original.copy()
            var query = source.lazy().select_exprs(
                [
                    col("label").alias("alias"),
                    col("label").str().to_uppercase().alias("computed"),
                ]
            )
            var schema = schema_frame(query, len(query._nodes) - 1)
            assert_field_owner(schema.column("alias"), original.column("label"))
            assert_false(Bool(schema.column("computed")._field_metadata))
            var dictionary = DictionaryCodes(query)
            # Test the bare field transport independently of lookup lowering.
            var bare = source.lazy().select(col("label").alias("alias"))
            var encoded = encode_query(bare, dictionary)
            assert_field_owner(
                encoded._frames[0].column("label"), original.column("label")
            )
            var codes = encoded.collect(engine="cpu", streaming=False)
            assert_field_owner(codes.column("alias"), original.column("label"))
            var restored = restore_frame(
                codes, schema_frame(bare, len(bare._nodes) - 1), dictionary
            )
            assert_field_owner(
                restored.column("alias"), original.column("label")
            )
