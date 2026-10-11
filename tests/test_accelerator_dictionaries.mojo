"""Shared string code staging must work without an accelerator or driver."""
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
)
from dataframe.accelerator.dictionaries import DictionaryCodes
from dataframe.categorical import encode


def test_ordered_codes_include_literals_and_preserve_nulls() raises:
    var source = Series(
        "k",
        StringColumn(
            List[String](["z", "", "é", "a", "ignored", "a"]),
            List[Bool]([True, True, True, True, False, True]),
        ),
    )
    var frame = DataFrame([source.copy()])
    var dictionary = DictionaryCodes(
        frame.lazy().filter(col("k") != lit("missing"))
    )
    assert_true(dictionary.code("") < dictionary.code("a"))
    assert_true(dictionary.code("a") < dictionary.code("missing"))
    assert_true(dictionary.code("missing") < dictionary.code("z"))
    assert_true(dictionary.code("z") < dictionary.code("é"))
    var codes = dictionary.encode(source)
    assert_equal(codes.numeric[DType.uint32]()._get(0), dictionary.code("z"))
    assert_equal(
        codes.numeric[DType.uint32]()._get(3),
        codes.numeric[DType.uint32]()._get(5),
    )
    assert_true(codes.get(4).is_null())
    var restored = dictionary.restore(codes, DataType.STRING)
    assert_true(DataFrame([restored^]).equals(frame))


def test_categorical_inputs_reconcile_different_code_orders() raises:
    var first = encode(Series("k", StringColumn(List[String](["z", "a", "z"]))))
    var second = encode(
        Series("k", StringColumn(List[String](["a", "z", "other"])))
    )
    assert_true(first.dtype() != second.dtype())
    var left = DataFrame([first.copy()])
    var right = DataFrame([second.copy()])
    var dictionary = DictionaryCodes(left.lazy().join(right.lazy(), "k"))
    var a = dictionary.encode(first)
    var b = dictionary.encode(second)
    assert_equal(a.dtype(), b.dtype())
    assert_equal(
        a.numeric[DType.uint32]()._get(0), b.numeric[DType.uint32]()._get(1)
    )
    assert_equal(
        a.numeric[DType.uint32]()._get(1), b.numeric[DType.uint32]()._get(0)
    )
    var restored = dictionary.restore(a, first.dtype())
    assert_equal(restored.dtype(), first.dtype())
    assert_true(DataFrame([restored^]).equals(left))


def test_empty_categorical_retains_unused_members() raises:
    var categorical = encode(
        Series("k", StringColumn(List[String](["unused", "a"])))
    )
    var empty = categorical.slice(0, 0)
    var dictionary = DictionaryCodes(DataFrame([empty.copy()]).lazy())
    assert_true(dictionary.code("a") < dictionary.code("unused"))
    var coded = dictionary.encode(empty)
    var restored = dictionary.restore(coded, categorical.dtype())
    assert_equal(restored.dtype(), categorical.dtype())
    assert_equal(len(restored), 0)
    assert_equal(len(restored.dtype().dictionary()[]), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
