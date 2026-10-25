"""Arrow field metadata follows source fields, not computed result values."""
# contract-group: public.accelerator_staging
# contract-group: public.apple_parity
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false
from dataframe import Column, DataFrame, Series, col
from dataframe.lazy import LazyFrame


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


def test_accelerator_field_metadata_group_keys_and_reductions_follow_cpu() raises:
    var original = field_source()
    for empty in [False, True]:
        var frame = original.clear() if empty else original.copy()
        var query = field_group_query(frame)
        var expected = query.collect(engine="cpu", streaming=False)
        assert_field_owner(expected.column("k"), original.column("k"))
        for name in ["total", "first", "last", "low"]:
            assert_false(Bool(expected.column(name)._field_metadata))
        var scalar = frame.lazy().select_exprs(
            [col("x").sum().alias("total"), col("x").first().alias("first")]
        )
        expected = scalar.collect(engine="cpu", streaming=False)
        for output in range(expected.width()):
            assert_false(Bool(expected._columns[output]._field_metadata))
