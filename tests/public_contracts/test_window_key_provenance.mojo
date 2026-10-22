"""CPU partition provenance survives OVER splitting and dictionary recoding."""
# contract-group: public.apple_parity
# contract-group: expr.decimal_extrema
from std.testing import assert_equal, assert_true, assert_false
from std.memory import Pointer
from std.sys import size_of
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.dtype import CategoricalDictionary
from dataframe.lazy import AGG, LazyFrame
from dataframe.plan_types import WindowKey
from dataframe_metal.accelerator.code_domains import (
    CategoryOwner,
    CategoryField,
)
from dataframe_metal.accelerator.windows import lower_partition_aggregates
from dataframe_metal.accelerator.dictionaries import (
    DictionaryCodes,
    encode_query,
)


def _keys(query: LazyFrame) -> List[WindowKey]:
    var keys = List[WindowKey]()
    for operation in query._nodes:
        if operation.kind == AGG:
            keys.extend(operation.window_keys.copy())
    return keys^


def test_window_key_provenance_original_codes_and_explicit_column_reads() raises:
    var dictionary = CategoricalDictionary()
    for text in ["z", "a", "é🙂", ""]:
        dictionary.append(text)
    var dtype = DataType.categorical(dictionary^)
    var frame = DataFrame(
        [
            Series("k", Column[UInt32]([0, 1, 2, 3])).with_dtype(dtype),
            Series("x", Column[Int64]([1, 2, 3, 4])),
            Series(
                "other",
                StringColumn(["outside", "outside", "outside", "outside"]),
            ),
        ]
    )
    var query = frame.lazy().select_exprs(
        [
            col("x").sum().over("k").alias("raw"),
            when(col("k") == lit("z"))
            .then(col("x").sum().over("k"))
            .otherwise(lit(Int64(0)))
            .alias("decoded"),
        ]
    )
    var staged = lower_partition_aggregates(query)
    var original = _keys(staged)
    assert_equal(len(original), 2)
    assert_equal(original[0].name, "k")
    assert_equal(original[0].dtype, dtype)
    assert_false(original[0].string_hash)
    assert_true(original[1].string_hash)
    assert_equal(len(original[0].values[]), 0)
    var codes = DictionaryCodes(staged)
    # Four UInt32 codes plus one validity byte for each dictionary column,
    # then two retained five-entry UInt64 provenance tables. Only the one
    # categorical scan column retains an original-code category owner and
    # field; its unchunked input contributes one Column[UInt32] chunk header.
    # Its four source-dictionary entries also reserve eight bytes apiece for
    # the code map, presence flags and discovery scratch.
    assert_equal(
        codes.staging_bytes(staged),
        codes.bytes
        + 34
        + 80
        + 4 * 8
        + size_of[CategoryOwner]()
        + size_of[CategoryField]()
        + size_of[Column[UInt32]](),
    )
    var encoded = encode_query(staged, codes)
    var keys = _keys(encoded)
    var copied = encoded.copy()
    var copied_keys = _keys(copied)
    assert_equal(
        Int(Pointer(to=keys[0].values[])),
        Int(Pointer(to=copied_keys[0].values[])),
    )
    assert_equal(len(keys[0].values[]), 5)
    assert_equal(keys[0].values[][Int(codes.code("z"))], UInt64(0))
    assert_equal(keys[0].values[][Int(codes.code("a"))], UInt64(1))
    assert_equal(keys[0].values[][Int(codes.code("é🙂"))], UInt64(2))
    assert_equal(keys[0].values[][Int(codes.code(""))], UInt64(3))
    assert_equal(keys[0].values[][Int(codes.code("outside"))], UInt64(0))
    # Independently packed short UTF-8 keys, including their byte lengths.
    assert_equal(
        keys[1].values[][Int(codes.code("z"))], UInt64(0xCAF29CE48422235F)
    )
    assert_equal(
        keys[1].values[][Int(codes.code("a"))], UInt64(0xCAF29CE484222344)
    )
    assert_equal(
        keys[1].values[][Int(codes.code("é🙂"))], UInt64(0xCDF21E7D1BD28AE6)
    )
    assert_equal(
        keys[1].values[][Int(codes.code(""))], UInt64(0xCBF29CE484222325)
    )
    assert_equal(
        keys[1].values[][Int(codes.code("outside"))], UInt64(0xCC97F88DF756564A)
    )
    assert_equal(keys[0].dtype, dtype)
    assert_equal(encoded._frames[0]._columns[0].dtype(), DataType.UINT32)


def test_window_key_provenance_strings_and_empty_categorical_dictionaries() raises:
    var strings = DataFrame(
        [
            Series("k", StringColumn(["z", "a", ""])),
            Series("x", Column[Int64]([1, 2, 3])),
        ]
    )
    var staged = lower_partition_aggregates(
        strings.lazy().select(col("x").sum().over("k"))
    )
    var codes = DictionaryCodes(staged)
    var keys = _keys(encode_query(staged, codes))
    assert_equal(keys[0].dtype, DataType.STRING)
    assert_true(keys[0].string_hash)
    assert_equal(
        keys[0].values[][Int(codes.code("z"))], UInt64(0xCAF29CE48422235F)
    )

    var dictionary = CategoricalDictionary()
    var dtype = DataType.categorical(dictionary^)
    for rows in [0, 1]:
        var frame = DataFrame(
            [
                Series(
                    "k",
                    Column[UInt32](
                        List[UInt32](length=rows, fill=UInt32.MAX),
                        List[Bool](length=rows, fill=False),
                    ),
                ).with_dtype(dtype),
                Series("x", Column[Int64](List[Int64](length=rows, fill=1))),
            ]
        )
        var empty = lower_partition_aggregates(
            frame.lazy().select(col("x").sum().over("k"))
        )
        var empty_codes = DictionaryCodes(empty)
        var empty_keys = _keys(encode_query(empty, empty_codes))
        assert_equal(len(empty_keys), 1)
        assert_equal(empty_keys[0].dtype, dtype)
        assert_false(empty_keys[0].string_hash)
        assert_equal(len(empty_keys[0].values[]), 0)
