"""Internal dictionary joins retain the left key's Arrow field owner."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true
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


def test_dictionary_lookup_keeps_the_left_key_field_metadata() raises:
    # Both CAT sides preserve their field owners during dictionary
    # unification. A full join's coalesced key takes rows of both keys and
    # is the left key's field: it keeps the left key's metadata whatever
    # the right key carries (#1478).
    var original = _labelled_source()
    var fresh = DataFrame(
        [
            Series("label", StringColumn(["z", "z", "z", "z"])).cast(
                DataType.CATEGORICAL
            )
        ]
    )
    var joined = (
        original.lazy()
        .join(fresh.lazy(), on="label", how="full")
        .collect(engine="cpu", streaming=False, optimize=False)
    )
    var left = original.column("label")._field_metadata.value()
    var kept = joined.column("label")._field_metadata
    assert_true(Bool(kept))
    assert_equal(len(kept.value()[]), len(left[]))
    for i in range(len(left[])):
        assert_equal(kept.value()[][i], left[][i])
    # 2 "z" rows on the left times 4 on the right, plus the left's "a" and
    # null rows.
    assert_equal(joined.height(), 10)
