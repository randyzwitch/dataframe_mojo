"""Remap private text columns on device; numeric/opaque words stay distinct."""
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true
from dataframe import DataType
from dataframe.dtype import CategoricalDictionary
from dataframe.accelerator.gpu.residency import Scope, TextDomain, TextColumn
from dataframe.accelerator.gpu.text_domains import (
    union_domain,
    ColumnMap,
    MapPlan,
    Submission,
    submit,
)
from tests.public_contracts.test_dynamic_text_domains import (
    test_dynamic_text_domain_union_maps_are_metadata_and_private,
    test_dynamic_text_domains_keep_original_categorical_and_hash_provenance,
)


def _dtype(values: List[String]) -> DataType:
    var dictionary = CategoricalDictionary()
    for value in values:
        dictionary.append(value)
    return DataType.categorical(dictionary^)


def main() raises:
    test_dynamic_text_domain_union_maps_are_metadata_and_private()
    test_dynamic_text_domains_keep_original_categorical_and_hash_provenance()
    var scope = Scope()
    var base_domain = ArcPointer(TextDomain(scope, _dtype(["a", "z"]), False))
    var dynamic = ArcPointer(TextDomain(scope, _dtype(["m"]), True))
    var fields: List[Optional[TextColumn]] = [
        TextColumn(DataType.STRING, base_domain.copy()),
        None,
        TextColumn(DataType.STRING, dynamic.copy()),
        None,
    ]
    var target = union_domain(scope, base_domain, fields)
    var maps: List[ColumnMap] = [
        ColumnMap(0, 3, base_domain.copy()),
        ColumnMap(2, 2, dynamic.copy()),
    ]
    var plan = MapPlan(target, maps)
    # Low/high planes have untouched padding guards. Slot1 is numeric UInt32;
    # slot3 is opaque byte codes. Neither is a logical text field.
    var words = List[UInt64](length=56, fill=0xBAD)
    var valid = List[UInt8](length=28, fill=1)
    var errors = List[UInt32](length=28, fill=0)
    words[0] = 1
    words[1] = 0xFFFFFFFFFFFFFFFF
    words[2] = 0
    valid[1] = 0
    words[7] = 1
    words[8] = 0
    words[14] = 0
    words[15] = 0
    words[21] = 1
    words[22] = 0
    var expected = words.copy()
    expected[0] = 2
    expected[14] = 1
    expected[15] = 1
    var context = scope.context()
    var dw = context.enqueue_create_buffer[DType.uint64](len(words))
    var dv = context.enqueue_create_buffer[DType.uint8](len(valid))
    var de = context.enqueue_create_buffer[DType.uint32](len(errors))
    var submission = Optional[Submission]()
    try:
        context.enqueue_copy(dw, words.unsafe_ptr())
        context.enqueue_copy(dv, valid.unsafe_ptr())
        context.enqueue_copy(de, errors.unsafe_ptr())
        scope.record_upload(len(words) * 8 + len(valid) + len(errors) * 4)
        submission = submit(scope, plan^, dw, dv, de, 7)
        context.enqueue_copy(words.unsafe_ptr(), dw)
        context.enqueue_copy(errors.unsafe_ptr(), de)
        scope.synchronize()
    except error:
        scope.synchronize()
        _ = submission^
        _ = words^
        _ = valid^
        _ = errors^
        _ = dw^
        _ = dv^
        _ = de^
        raise error^
    for at in range(len(words)):
        assert_equal(words[at], expected[at])
    for value in errors:
        assert_equal(value, UInt32(0))
    assert_equal(scope.transfers()[2], 1)
    # 3word map entries +2fourword Int64 descriptors, all metadata uploads.
    assert_equal(
        scope.transfers()[0],
        len(words) * 8 + len(valid) + len(errors) * 4 + 3 * 4 + 2 * 4 * 8,
    )
    _ = submission^
    _ = words^
    _ = valid^
    _ = errors^
    _ = dw^
    _ = dv^
    _ = de^
    print(
        "PASS private text GPU remap, immutable planes/null guards/numeric+opaque provenance"
    )
