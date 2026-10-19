"""Retained code transport preserves values without hashing String rows."""
# contract-group: execution.backends
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, StringColumn, col
from dataframe.dtype import CategoricalDictionary
from dataframe.string_column import StringCodes
from dataframe.accelerator.dictionaries import DictionaryCodes, _staging_values
from tests.support.coverage_compare import assert_frame
from tests.execution_modes.test_backend_metal import require_metal


def source() raises -> Series:
    var dictionary = CategoricalDictionary()
    for label in ["b", "a", "é🙂", "unused", "a"]:
        dictionary.append(label)
    var strings = StringColumn(
        ["é🙂", "b", "a", "", "é🙂", "a", "b"],
        [True, True, True, False, True, True, True],
    )
    var codes: List[UInt32] = [2, 0, 1, UInt32.MAX, 2, 4, 0]
    strings._codes = ArcPointer(
        StringCodes(
            ArcPointer(codes^),
            DataType.categorical(dictionary^),
        )
    )
    return Series("key", strings^)


def verify(column: Series) raises:
    var frame = DataFrame([column.copy()])
    var dictionary = DictionaryCodes(frame.lazy())
    assert_true("unused" not in dictionary.lookup)
    var encoded = dictionary.encode(column)
    var restored = dictionary.restore(encoded, DataType.STRING)
    assert_frame(DataFrame([restored^]), frame)


def test_metal_retained_dictionary_staging_slices_gathers_and_chunks() raises:
    var key = source()
    var values = _staging_values(key)
    assert_equal(len(values), 3)
    assert_equal(String(values.get(0)), "é🙂")
    assert_equal(String(values.get(1)), "b")
    assert_equal(String(values.get(2)), "a")
    verify(key)
    verify(key.slice(1, 5))
    verify(key.take([6, 3, 0, 2]))
    verify(Series._from_chunks([key.slice(1, 2), key.slice(3, 4)]))
    verify(key.slice(0, 0))


def test_metal_retained_dictionary_staging_preserves_unused_categories() raises:
    var retained = source()._dictionary_codes()
    var key = retained.take()
    var frame = DataFrame([key.copy()])
    var dictionary = DictionaryCodes(frame.lazy())
    assert_equal(len(dictionary.dtype.dictionary()[]), 4)
    assert_true("unused" in dictionary.lookup)
    var encoded = dictionary.encode(key)
    assert_frame(
        DataFrame([dictionary.restore(encoded, DataType.STRING)]),
        DataFrame([source()]),
    )


# requires: metal
def test_metal_retained_dictionary_staging_public_aggregate() raises:
    var frame = DataFrame(
        [
            source().slice(0, 5),
            Series("value", Column[Int64]([1, 2, 3, 4, 5])),
        ]
    )
    var query = (
        frame.lazy()
        .group_by("key")
        .agg([col("value").sum().alias("sum"), col("value").max().alias("max")])
    )
    assert_frame(
        query.collect(accelerator=require_metal()).sort("key"),
        query.collect(engine="cpu").sort("key"),
    )
