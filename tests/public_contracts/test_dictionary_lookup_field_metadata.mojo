"""Internal dictionary joins retain matching private-key Arrow owners."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import DataFrame, Series, StringColumn, col, concat_str
from dataframe.lazy import JOIN
from dataframe.accelerator.dictionaries import (
    lower_dictionary_operations,
    DictionaryCodes,
    encode_query,
)
from dataframe.accelerator.metadata import schema_frame
from dataframe.accelerator.joins import has_decimal_consumers
from tests.public_contracts.test_accelerator_field_metadata import (
    field_text_source,
    assert_field_owner,
)


def test_dictionary_lookup_metadata_schema_cpu_literals_and_decimal_admission() raises:
    for categorical in [False, True]:
        var original = field_text_source(categorical)
        for empty in [False, True]:
            var source = original.clear() if empty else original.copy()
            var query = source.lazy().select_exprs(
                [
                    col("label").alias("alias"),
                    col("label").str().to_uppercase().alias("upper"),
                    concat_str([col("label"), col("label")], ":").alias("pair"),
                    concat_str(
                        [col("label").str().to_uppercase(), col("label")], ":"
                    ).alias("nested"),
                ]
            )
            var lowered = lower_dictionary_operations(query)
            assert_true(lowered[2] >= 4)
            # Before the fix this metadata-only traversal fails on the first
            # private key: its source alias has metadata but its lookup does not.
            var schema = schema_frame(lowered[0], len(lowered[0]._nodes) - 1)
            assert_field_owner(schema.column("alias"), original.column("label"))
            for name in ["upper", "pair", "nested"]:
                assert_false(Bool(schema.column(name)._field_metadata))
            for operation in lowered[0]._nodes:
                if operation.kind != JOIN:
                    continue
                var left = schema_frame(lowered[0], operation.left)
                var right = schema_frame(lowered[0], operation.right)
                for name in operation.names:
                    var a = left.column(name)
                    var b = right.column(name)
                    assert_equal(
                        Bool(a._field_metadata), Bool(b._field_metadata)
                    )
                    if a._field_metadata:
                        assert_field_owner(b, a)
            assert_false(has_decimal_consumers(lowered[0]))
            var dictionary = DictionaryCodes(lowered[0])
            var encoded = encode_query(lowered[0], dictionary)
            assert_false(has_decimal_consumers(encoded))
            var actual = lowered[0].collect(
                engine="cpu", streaming=False, optimize=False
            )
            var expected = query.collect(
                engine="cpu", streaming=False, optimize=False
            )
            assert_true(actual.equals(expected))
            assert_field_owner(actual.column("alias"), original.column("label"))
            for name in ["upper", "pair", "nested"]:
                assert_false(Bool(actual.column(name)._field_metadata))
            assert_equal(actual.height(), 0 if empty else 4)
            if not empty:
                var upper: List[String] = ["Z", "", "A", "Z"]
                var pair: List[String] = ["z:z", "", "a:a", "z:z"]
                var nested: List[String] = ["Z:z", "", "A:a", "Z:z"]
                for row in range(4):
                    for name in ["upper", "pair", "nested"]:
                        assert_equal(actual.item(row, name).is_null(), row == 1)
                    if row != 1:
                        assert_equal(
                            actual.item(row, "upper").string(), upper[row]
                        )
                        assert_equal(
                            actual.item(row, "pair").string(), pair[row]
                        )
                        assert_equal(
                            actual.item(row, "nested").string(), nested[row]
                        )


def test_dictionary_lookup_does_not_change_public_metadata_conflicts() raises:
    var original = field_text_source(False)
    var fresh = DataFrame([Series("label", StringColumn(["z"]))])
    var invalid = original.lazy().join(fresh.lazy(), on="label", how="left")
    with assert_raises(contains="Conflicting Arrow field metadata"):
        _ = schema_frame(invalid, len(invalid._nodes) - 1)
    with assert_raises(contains="Conflicting Arrow field metadata"):
        _ = invalid.collect(engine="cpu", streaming=False, optimize=False)
