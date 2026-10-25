"""Internal dictionary joins retain matching private-key Arrow owners."""
from std.memory import ArcPointer
from std.testing import assert_raises
from dataframe import Column, DataFrame, DataType, Series, StringColumn


def _labelled_source() raises -> DataFrame:
    var key = Series("k", Column[Int64]([0, 1, 0, 1]))
    var text = Series(
        "label",
        StringColumn(["z", "poison", "a", "z"], [True, False, True, True]),
    ).cast(DataType.CATEGORICAL)
    # One Arrow metadata pair k=s on the label field.
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 115]
    text._field_metadata = ArcPointer(metadata^)
    return DataFrame([key^, text^])


def test_dictionary_lookup_does_not_change_public_metadata_conflicts() raises:
    # Both CAT sides preserve their field owners during dictionary unification.
    # Four equal right keys force low_cardinality's dictionary path for FULL;
    # tiny LEFT/FULL direct hash shortcuts do not append/check key metadata.
    var original = _labelled_source()
    var fresh = DataFrame(
        [
            Series("label", StringColumn(["z", "z", "z", "z"])).cast(
                DataType.CATEGORICAL
            )
        ]
    )
    var invalid = original.lazy().join(fresh.lazy(), on="label", how="full")
    with assert_raises(contains="Conflicting Arrow field metadata"):
        _ = invalid.collect(engine="cpu", streaming=False, optimize=False)
