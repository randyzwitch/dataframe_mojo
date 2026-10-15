"""Internal dictionary joins retain matching private-key Arrow owners."""
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
    concat_str,
    null,
)
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


def _lookup_case(categorical: Bool, empty: Bool, shape: Int, utf8: Bool) raises:
    var label = (
        ("CAT" if categorical else "String")
        + ("/empty/" if empty else "/rows/")
        + List[String](["UPPER", "CONCAT", "nested", "combined"])[shape]
    )
    label += "/UTF8+NUL" if utf8 else "/ASCII"
    var stage = String("source")
    try:
        var original = field_text_source(categorical)
        if utf8:
            var text = Series(
                "label",
                StringColumn(
                    ["☃z", "poison", "☃a", "a\x00z"], [True, False, True, True]
                ),
            )
            if categorical:
                text = text.cast(DataType.CATEGORICAL)
            text._field_metadata = original.column("label")._field_metadata
            original = original.with_column(text^)
        var source = original.clear() if empty else original.copy()
        var expressions = List[Expr]([col("label").alias("alias")])
        if shape == 0 or shape == 3:
            expressions.append(col("label").str().to_uppercase().alias("upper"))
        if shape == 1 or shape == 3:
            expressions.append(
                concat_str([col("label"), col("label")], ":").alias("pair")
            )
        if shape == 2 or shape == 3:
            expressions.append(
                concat_str(
                    [col("label").str().to_uppercase(), col("label")], ":"
                ).alias("nested")
            )
        var query = source.lazy().select_exprs(expressions)
        stage = "lower"
        print("CHECK dictionary", label, stage, flush=True)
        var lowered = lower_dictionary_operations(query)
        assert_true(lowered[2] >= (4 if shape == 3 else 2 if shape == 2 else 1))
        stage = "cached metadata schema"
        print("CHECK dictionary", label, stage, flush=True)
        var schema = schema_frame(lowered[0], len(lowered[0]._nodes) - 1)
        assert_field_owner(schema.column("alias"), original.column("label"))
        for field in schema._columns:
            if field.name() != "alias":
                assert_false(Bool(field._field_metadata))
        stage = "each private JOIN cached and lazy-empty inputs"
        print("CHECK dictionary", label, stage, flush=True)
        for operation in lowered[0]._nodes:
            if operation.kind != JOIN:
                continue
            var left = schema_frame(lowered[0], operation.left)
            var right = schema_frame(lowered[0], operation.right)
            var lazy_left = lowered[0]._execute(operation.left, True)
            var lazy_right = lowered[0]._execute(operation.right, True)
            for name in operation.names:
                print(
                    "CHECK dictionary", label, "private key", name, flush=True
                )
                for view in [
                    left.copy(),
                    right.copy(),
                    lazy_left.copy(),
                    lazy_right.copy(),
                ]:
                    var key = view.column(name)
                    assert_equal(key.dtype(), DataType.STRING)
                    assert_false(Bool(key._field_metadata))
        stage = "decimal admission"
        print("CHECK dictionary", label, stage, flush=True)
        assert_false(has_decimal_consumers(lowered[0]))
        var dictionary = DictionaryCodes(lowered[0])
        var encoded = encode_query(lowered[0], dictionary)
        assert_false(has_decimal_consumers(encoded))
        stage = "encoded metadata schema"
        print("CHECK dictionary", label, stage, flush=True)
        _ = schema_frame(encoded, len(encoded._nodes) - 1)
        stage = "normalized eager CPU"
        print("CHECK dictionary", label, stage, flush=True)
        var actual = lowered[0].collect(
            engine="cpu", streaming=False, optimize=False
        )
        stage = "original eager CPU"
        print("CHECK dictionary", label, stage, flush=True)
        var expected = query.collect(
            engine="cpu", streaming=False, optimize=False
        )
        stage = "independent literals and owners"
        assert_true(actual.equals(expected))
        assert_field_owner(actual.column("alias"), original.column("label"))
        for field in actual._columns:
            if field.name() != "alias":
                assert_false(Bool(field._field_metadata))
        assert_equal(actual.height(), 0 if empty else 4)
        if not empty:
            var upper: List[String] = ["Z", "", "A", "Z"]
            var pair: List[String] = ["z:z", "", "a:a", "z:z"]
            var nested: List[String] = ["Z:z", "", "A:a", "Z:z"]
            if utf8:
                upper = ["☃Z", "", "☃A", "A\x00Z"]
                pair = ["☃z:☃z", "", "☃a:☃a", "a\x00z:a\x00z"]
                nested = ["☃Z:☃z", "", "☃A:☃a", "A\x00Z:a\x00z"]
            for row in range(4):
                for field in actual._columns:
                    if field.name() == "alias":
                        continue
                    assert_equal(
                        actual.item(row, field.name()).is_null(), row == 1
                    )
                    if row != 1:
                        var literal = (
                            upper[row] if field.name()
                            == "upper" else pair[row] if field.name()
                            == "pair" else nested[row]
                        )
                        assert_equal(
                            actual.item(row, field.name()).string(), literal
                        )
        print("PASS dictionary", label, flush=True)
    except error:
        raise Error(
            "Dictionary lookup " + label + " at " + stage + ": " + String(error)
        )


def _private_key_phase_witness() raises:
    var source = field_text_source(False)
    # Pin the two existing CPU metadata paths independently. A bare internal
    # alias is not a stable private key owner certificate across these paths.
    var alias = source.lazy().with_columns(col("label").alias("private"))
    var cached = schema_frame(alias, len(alias._nodes) - 1)
    var lazy = alias._execute(len(alias._nodes) - 1, True)
    assert_field_owner(cached.column("private"), source.column("label"))
    assert_false(Bool(lazy.column("private")._field_metadata))
    var identity = source.lazy().with_columns(
        col("label").fill_null(null(DataType.STRING)).alias("private")
    )
    cached = schema_frame(identity, len(identity._nodes) - 1)
    lazy = identity._execute(len(identity._nodes) - 1, True)
    assert_false(Bool(cached.column("private")._field_metadata))
    assert_false(Bool(lazy.column("private")._field_metadata))
    print("PASS dictionary private key CPU metadata phase witness", flush=True)


def test_dictionary_lookup_metadata_schema_cpu_literals_and_decimal_admission() raises:
    _private_key_phase_witness()
    var failures = 0
    for utf8 in [False, True]:
        for categorical in [False, True]:
            for empty in [False, True]:
                for shape in range(4):
                    try:
                        _lookup_case(categorical, empty, shape, utf8)
                    except error:
                        failures += 1
                        print("FAIL", String(error), flush=True)
    assert_equal(failures, 0)


def test_dictionary_lookup_does_not_change_public_metadata_conflicts() raises:
    # Both CAT sides preserve their field owners during dictionary unification.
    # Four equal right keys force low_cardinality's dictionary path for FULL;
    # tiny LEFT/FULL direct hash shortcuts do not append/check key metadata.
    var original = field_text_source(True)
    var fresh = DataFrame(
        [
            Series("label", StringColumn(["z", "z", "z", "z"])).cast(
                DataType.CATEGORICAL
            )
        ]
    )
    var invalid = original.lazy().join(fresh.lazy(), on="label", how="full")
    with assert_raises(contains="Conflicting Arrow field metadata"):
        _ = schema_frame(invalid, len(invalid._nodes) - 1)
    with assert_raises(contains="Conflicting Arrow field metadata"):
        _ = invalid.collect(engine="cpu", streaming=False, optimize=False)
