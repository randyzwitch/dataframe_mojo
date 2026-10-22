"""Logical text provenance and immutable union/map metadata, no GPU needed."""
# contract-group: public.apple_parity
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import DataFrame, DataType, Series
from dataframe.dtype import CategoricalDictionary
from dataframe_metal.accelerator.code_domains import (
    TextField,
    TextQuery,
    same_dictionary,
    code,
)
from dataframe.plan_types import WindowKey
from dataframe_metal.accelerator.text_key_domains import append_partition_values
from dataframe_metal.accelerator.gpu.residency import (
    Scope,
    TextDomain,
    TextColumn,
    Table,
)
from dataframe_metal.accelerator.gpu.text_domains import (
    union_domain,
    ColumnMap,
    MapPlan,
)


def _dtype(values: List[String]) -> DataType:
    var bytes_count = 0
    for value in values:
        bytes_count += value.byte_length()
    var bytes = List[UInt8](capacity=bytes_count)
    var offsets = List[Int64](capacity=len(values) + 1)
    offsets.append(0)
    var dictionary = CategoricalDictionary(bytes^, offsets^)
    for value in values:
        dictionary.append(value)
    return DataType.categorical(dictionary^)


# requires: mojo_gpu
def test_dynamic_text_domain_union_maps_are_metadata_and_private() raises:
    var scope = Scope(99999)
    var original = _dtype(["a", "z"])
    var additional = _dtype(["m"])
    var base_domain = ArcPointer(TextDomain(scope, original, False))
    var dynamic = ArcPointer(TextDomain(scope, additional, True))
    var fields: List[Optional[TextColumn]] = [
        TextColumn(DataType.STRING, base_domain.copy()),
        None,
        TextColumn(DataType.STRING, dynamic.copy()),
        None,
    ]
    var merged = union_domain(scope, base_domain, fields)
    assert_false(scope.initialized())
    assert_equal(code(merged[].dtype, "a"), UInt32(0))
    assert_equal(code(merged[].dtype, "m"), UInt32(1))
    assert_equal(code(merged[].dtype, "z"), UInt32(2))
    var again = union_domain(
        scope, base_domain, [TextColumn(DataType.STRING, merged.copy())]
    )
    assert_true(same_dictionary(again[].dtype, merged[].dtype))
    _ = again^
    var maps: List[ColumnMap] = [
        ColumnMap(0, 3, base_domain.copy()),
        ColumnMap(2, 2, dynamic.copy()),
    ]
    var plan = MapPlan(merged, maps)
    assert_equal(len(plan.words), 3)
    assert_equal(plan.words[0], UInt32(0))
    assert_equal(plan.words[1], UInt32(2))
    assert_equal(plan.words[2], UInt32(1))
    assert_equal(plan.descriptors[0], Int64(0))
    assert_equal(plan.descriptors[4], Int64(2))
    # 3 UInt32 host/device map entries +2 four-Int64 descriptors on host/device.
    assert_equal(plan.bytes(), 3 * 8 + 2 * 64)
    var table = Table.empty(
        DataFrame(
            [
                Series.full_null("text", DataType.UINT32, 0),
                Series.full_null("number", DataType.UINT32, 0),
                Series.full_null("opaque", DataType.UINT32, 0),
            ]
        )
    )
    # Original pre-pruning metadata may mention fields removed by a scan or
    # consumer projection. Only actual retained columns acquire owners.
    table.attach_text(
        [
            TextField("text", DataType.STRING),
            TextField("dropped_text", DataType.STRING),
            TextField("dropped_cat", additional),
        ],
        merged,
    )
    assert_true(Bool(table.text[0]))
    assert_false(Bool(table.text[1]))
    assert_false(Bool(table.text[2]))
    assert_equal(table.schema._columns[0].dtype(), DataType.UINT32)
    assert_equal(table.text[0].value().logical, DataType.STRING)
    assert_equal(len(table.text), 3)
    var invalid = Table.empty(
        DataFrame([Series.full_null("text", DataType.FLOAT64, 0)])
    )
    with assert_raises(contains="UInt32 transport storage"):
        invalid.attach_text([TextField("text", DataType.STRING)], merged)
    _ = invalid^
    _ = plan^
    _ = table^
    _ = fields^
    _ = maps^
    _ = merged^
    _ = dynamic^
    _ = base_domain^
    assert_equal(scope.info()[0], 0)


# requires: mojo_gpu
def test_dynamic_text_domains_keep_original_categorical_and_hash_provenance() raises:
    var scope = Scope(99999)
    var ordered = _dtype(["a", "m", "z"])
    var categorical = _dtype(["z", "a"])
    var domain = ArcPointer(TextDomain(scope, ordered, False))
    var table = Table.empty(
        DataFrame(
            [
                Series.full_null("s", DataType.UINT32, 0),
                Series.full_null("c", DataType.UINT32, 0),
            ]
        )
    )
    table.attach_text(
        [TextField("s", DataType.STRING), TextField("c", categorical)], domain
    )
    assert_equal(table.text[0].value().logical, DataType.STRING)
    assert_true(
        table.text[1]
        .value()
        .logical.dictionary()[]
        .same_values(categorical.dictionary()[])
    )
    var string_key = WindowKey("s", DataType.STRING, True)
    var category_key = WindowKey("c", categorical, False)
    var hashes = List[UInt64]()
    append_partition_values(hashes, ordered, string_key)
    assert_true(string_key.string_hash)
    assert_equal(string_key.dtype, DataType.STRING)
    # Independently packed one-byte UTF8 keys, not hashes of carrier codes.
    assert_equal(hashes[0], UInt64(0xCAF29CE484222344))
    assert_equal(hashes[1], UInt64(0xCAF29CE484222348))
    assert_equal(hashes[2], UInt64(0xCAF29CE48422235F))
    var codes = List[UInt64]()
    append_partition_values(codes, ordered, category_key)
    # Original categorical z->0,a->1 is independent of current ordered codes.
    assert_equal(codes[0], UInt64(1))
    assert_equal(codes[2], UInt64(0))
    assert_false(category_key.string_hash)
    assert_equal(category_key.dtype, categorical)
    var opaque = WindowKey("b", DataType.BINARY, True)
    with assert_raises(contains="opaque/numeric"):
        append_partition_values(hashes, ordered, opaque)
    var query = TextQuery(ordered)
    var owners: List[Optional[DataType]] = [ordered, None, ordered]
    query.record_final(owners)
    assert_equal(query.final_owned_bytes, 0)
    assert_false(scope.initialized())


# requires: mojo_gpu
def test_dynamic_text_domain_guards_precede_device_creation() raises:
    var scope = Scope(99999)
    var denied = Scope(99999, 0)
    with assert_raises(contains="budget"):
        _ = TextDomain(denied, _dtype(["a"]), True)
    assert_equal(denied.info()[0], 0)
    assert_false(denied.initialized())
    var foreign = Scope(99999)
    var base = ArcPointer(TextDomain(scope, _dtype(["a"]), False))
    var other = ArcPointer(TextDomain(foreign, _dtype(["b"]), False))
    with assert_raises(contains="different query scope"):
        _ = union_domain(scope, other, List[Optional[TextColumn]]())
    var table = Table.rows_only(scope, 0)
    with assert_raises(contains="different query scope"):
        table.attach_text(List[TextField](), other)
    var columns: List[ColumnMap] = [ColumnMap(0, 0, base.copy())]
    var target = ArcPointer(TextDomain(scope, _dtype(["a", "z"]), False))
    var plan = MapPlan(target, columns)
    with assert_raises(contains="different query scope"):
        plan.validate_shape(foreign, 0, 0, 0, 0)
    # Empty shape is valid and enqueue returns before any context operation.
    plan.validate_shape(scope, 0, 0, 0, 0)
    with assert_raises(contains="matrix shape"):
        plan.validate_shape(scope, Int(UInt32.MAX) + 1, 0, 0, 0)
    with assert_raises(contains="matrix shape"):
        plan.validate_shape(scope, 1, 1, 1, 1)
    plan.descriptors[3] = 2
    with assert_raises(contains="outside its matrix"):
        plan.validate_shape(scope, 1, 2, 1, 1)
    plan.descriptors[3] = 0
    plan.descriptors[0] = 1
    with assert_raises(contains="outside its matrix"):
        plan.validate_shape(scope, 1, 2, 1, 1)
    with assert_raises(contains="descriptor/query scope"):
        _ = MapPlan(target, [ColumnMap(Int(UInt32.MAX) + 1, 0, base.copy())])
    assert_false(scope.initialized())
    assert_false(foreign.initialized())


# requires: mojo_gpu
def test_dynamic_qcut_label_controls_match_cpu_literal_spelling() raises:
    from dataframe_metal.accelerator.gpu.qcut_labels import build

    var scope = Scope(99999)
    var edges: List[UInt64] = [0x3FE8000000000000, 0x4002000000000000]
    var right = build(scope, edges, 2, False, 0)
    var left = build(scope, edges, 2, True, 0)
    assert_equal(
        right[0][].dtype.dictionary()[].get(Int(right[1][0])), "(-inf, 0.75]"
    )
    assert_equal(
        right[0][].dtype.dictionary()[].get(Int(right[1][1])), "(0.75, 2.25]"
    )
    assert_equal(
        right[0][].dtype.dictionary()[].get(Int(right[1][2])), "(2.25, inf]"
    )
    assert_equal(
        left[0][].dtype.dictionary()[].get(Int(left[1][0])), "[-inf, 0.75)"
    )
    assert_equal(
        left[0][].dtype.dictionary()[].get(Int(left[1][2])), "[2.25, inf)"
    )
    assert_false(scope.initialized())
