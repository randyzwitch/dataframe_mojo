"""Context-independent logical text owners, literal tags and partition hashes."""
# contract-group: public.apple_parity
from std.testing import assert_equal, assert_true, assert_false, assert_raises
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
from dataframe_metal.accelerator.code_domains import (
    TextQuery,
    TEXT_LITERAL,
    code,
    dictionary_bytes,
    same_dictionary,
    text_less,
)
from dataframe.plan_types import WindowKey
from dataframe_metal.accelerator.text_key_domains import append_partition_values
from dataframe_metal.accelerator.dictionaries import (
    DictionaryCodes,
    encode_query,
)
from dataframe.expr import LIT_INT
from dataframe.lazy import SELECT
from dataframe_metal.accelerator.metadata import prune_projections


def _domain(values: List[String]) -> DataType:
    var count = 0
    for value in values:
        count += value.byte_length()
    var bytes = List[UInt8](capacity=count)
    var offsets = List[Int64](capacity=len(values) + 1)
    offsets.append(0)
    var dictionary = CategoricalDictionary(bytes^, offsets^)
    for value in values:
        dictionary.append(value)
    return DataType.categorical(dictionary^)


def test_text_domain_metadata_owners_and_logical_literal_tags() raises:
    var base = _domain(["", "a", "z"])
    var dynamic = _domain(["é", "🙂"])
    var separate = _domain(["é", "🙂"])
    assert_true(same_dictionary(dynamic, dynamic))
    assert_false(same_dictionary(dynamic, separate))
    assert_true(dynamic.dictionary()[].same_values(separate.dictionary()[]))
    # Exact requested UTF8 bytes6 plus three Int64 offsets. Identical owners
    # are charged once; equal values in a different retained owner are distinct.
    assert_equal(dictionary_bytes(dynamic), 30)
    var query = TextQuery(base)
    var owners: List[Optional[DataType]] = [
        base,
        None,
        dynamic,
        dynamic,
        separate,
    ]
    query.record_final(owners)
    assert_equal(query.final_owned_bytes, 60)
    owners[2] = None
    assert_true(Bool(query.final_domains[2]))
    assert_true(same_dictionary(query.final_domains[2].value(), dynamic))
    query.record_final([base, None])
    assert_equal(query.final_owned_bytes, 0)
    assert_equal(code(base, ""), UInt32(0))
    assert_equal(code(base, "z"), UInt32(2))
    assert_true(text_less("z", "é"))
    assert_true(text_less("é", "🙂"))
    with assert_raises(contains="missing a required dictionary value"):
        _ = code(base, "absent")
    with assert_raises(contains="dictionary metadata"):
        _ = dictionary_bytes(DataType.UINT32)
    var source = DataFrame(
        [
            Series("s", StringColumn(["a", "z"])),
            Series("u", Column[UInt32]([0, 2])),
        ]
    )
    var original = source.lazy().select_exprs(
        [
            col("s"),
            col("u"),
            lit("").alias("empty"),
            lit(UInt32(0)).alias("numeric"),
        ]
    )
    var codes = DictionaryCodes(original)
    var encoded = encode_query(original, codes, dynamic_text=True)
    ref root = encoded._nodes[len(encoded._nodes) - 1]
    assert_equal(root.exprs[2]._nodes[0].op, LIT_INT)
    assert_equal(root.exprs[2]._nodes[0].min_count, TEXT_LITERAL)
    assert_equal(root.exprs[2]._nodes[0].text2, "")
    assert_equal(root.exprs[3]._nodes[0].op, LIT_INT)
    assert_true(root.exprs[3]._nodes[0].min_count != TEXT_LITERAL)
    assert_equal(root.exprs[3]._nodes[0].integer, Int64(0))
    assert_equal(len(root.text_fields), 2)
    assert_equal(root.text_fields[0].name, "s")
    assert_equal(root.text_fields[0].dtype, DataType.STRING)
    assert_equal(root.text_fields[1].name, "empty")
    assert_equal(root.text_fields[1].dtype, DataType.STRING)
    assert_equal(source.column("s").dtype(), DataType.STRING)
    assert_equal(source.column("u").dtype(), DataType.UINT32)
    assert_equal(encoded._frames[0].column("u").uint32().value(1), UInt32(2))
    # Join input pruning adds fresh SELECT nodes after recoding. Their output
    # metadata must preserve text domains while keeping numeric UInt32 aliases
    # separate; otherwise independent dynamic dictionaries cannot be remapped.
    var right = DataFrame(
        [
            Series("u", Column[UInt32]([0, 2])),
            Series("t", StringColumn(["m", "é"])),
            Series("right_unused", Column[UInt32]([3, 4])),
        ]
    )
    var paired = (
        source.lazy()
        .with_columns(
            [
                col("u").alias("numeric"),
                lit(UInt32(9)).alias("left_unused"),
            ]
        )
        .join(
            right.lazy().with_columns(lit(UInt32(7)).alias("extra")), on=["u"]
        )
        .select(["u", "s", "numeric", "t"])
    )
    var paired_codes = DictionaryCodes(paired)
    var narrowed = encode_query(paired, paired_codes, dynamic_text=True)
    prune_projections(narrowed)
    ref joined = narrowed._nodes[narrowed._nodes[len(narrowed._nodes) - 1].left]
    ref left_projection = narrowed._nodes[joined.left]
    ref right_projection = narrowed._nodes[joined.right]
    assert_equal(left_projection.kind, SELECT)
    assert_equal(right_projection.kind, SELECT)
    assert_equal(len(left_projection.text_fields), 1)
    assert_equal(len(right_projection.text_fields), 1)
    assert_equal(left_projection.text_fields[0].name, "s")
    assert_equal(right_projection.text_fields[0].name, "t")
    assert_equal(left_projection.text_fields[0].dtype, DataType.STRING)
    assert_equal(right_projection.text_fields[0].dtype, DataType.STRING)
    assert_true(
        same_dictionary(left_projection.code_domain.value(), paired_codes.dtype)
    )
    assert_true(
        same_dictionary(
            right_projection.code_domain.value(), paired_codes.dtype
        )
    )


def test_text_domain_metadata_keeps_raw_hash_and_original_code_provenance() raises:
    var staged = _domain(["a", "m", "z"])
    var original = _domain(["z", "a"])
    var string_key = WindowKey("s", DataType.STRING, True)
    var categorical_key = WindowKey("c", original, False)
    var hashes: List[UInt64] = [0x123456789ABCDEF0]
    append_partition_values(hashes, staged, string_key)
    assert_equal(hashes[0], UInt64(0x123456789ABCDEF0))
    # Independently packed short UTF8 byte hashes, before CPU's per-key mix.
    assert_equal(hashes[1], UInt64(0xCAF29CE484222344))
    assert_equal(hashes[2], UInt64(0xCAF29CE484222348))
    assert_equal(hashes[3], UInt64(0xCAF29CE48422235F))
    assert_equal(string_key.dtype, DataType.STRING)
    assert_true(string_key.string_hash)
    var codes = List[UInt64]()
    append_partition_values(codes, staged, categorical_key)
    assert_equal(codes[0], UInt64(1))
    assert_equal(codes[2], UInt64(0))
    assert_equal(categorical_key.dtype, original)
    assert_false(categorical_key.string_hash)
    assert_true(same_dictionary(categorical_key.dtype, original))
    # Physical UInt32 and opaque bytes never acquire text/hash provenance.
    var numeric = WindowKey("u", DataType.UINT32, False)
    var opaque = WindowKey("b", DataType.BINARY, True)
    with assert_raises(contains="opaque/numeric"):
        append_partition_values(codes, staged, numeric)
    with assert_raises(contains="opaque/numeric"):
        append_partition_values(codes, staged, opaque)
    assert_equal(numeric.dtype, DataType.UINT32)
    assert_equal(opaque.dtype, DataType.BINARY)
