"""Partition mapping preserves ordering, cardinality, nulls and Arrow schemas."""
# contract-group: kernel.window_mappings
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true
from dataframe import Column, DataType, ListColumn, Series, StructColumn
from dataframe.window_mapping import assemble_window_mapping


def test_window_mapping_rows_join_and_explode_partition_order() raises:
    var source = Series(
        "v", Column[Int64]([99, 7, 0, 8, 99], [True, True, False, True, True])
    )
    var parts: List[Series] = [source.slice(1, 2), source.slice(3, 1)]
    var groups = List[Int]([0, 1, 0])
    var joined = assemble_window_mapping(parts, groups, "join", False, source)
    assert_equal(joined.dtype(), DataType.list(DataType.INT64))
    assert_equal(len(joined), 3)
    for row in [0, 2]:
        var values = joined.get(row).list()
        assert_equal(len(values), 2)
        assert_equal(values.get(0).int64(), Int64(7))
        assert_true(values.get(1).is_null())
    assert_equal(joined.get(1).list().get(0).int64(), Int64(8))
    var exploded = assemble_window_mapping(
        parts, groups, "explode", False, source
    )
    assert_equal(len(exploded), 3)
    assert_equal(exploded.get(0).int64(), Int64(7))
    assert_true(exploded.get(1).is_null())
    assert_equal(exploded.get(2).int64(), Int64(8))
    var empty_parts: List[Series] = [source.slice(0, 0), source.slice(3, 1)]
    var with_empty = assemble_window_mapping(
        empty_parts, groups, "join", False, source
    )
    assert_equal(len(with_empty.get(0).list()), 0)
    assert_equal(len(with_empty.get(2).list()), 0)
    assert_equal(with_empty.get(1).list().get(0).int64(), Int64(8))


def _nested_mapping_prototype() raises -> Series:
    var child = Series("item", Column[Int32]([3, 4])).with_dtype(DataType.DATE)
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    child._field_metadata = ArcPointer(metadata^)
    var lists = Series("dates", ListColumn([Int64(0), 1, 2], child^))
    lists._field_metadata = lists.list_column().child()._field_metadata
    var result = Series("nested", StructColumn([lists^]))
    result._field_metadata = (
        result.struct_column().field("dates")._field_metadata
    )
    return result^


def _assert_nested_mapping_metadata(actual: Series, expected: Series) raises:
    assert_equal(actual.dtype(), expected.dtype())
    assert_true(
        actual._field_metadata.value().ptr()
        == expected._field_metadata.value().ptr()
    )
    var actual_list = actual.struct_column().field("dates")
    var expected_list = expected.struct_column().field("dates")
    assert_true(
        actual_list._field_metadata.value().ptr()
        == expected_list._field_metadata.value().ptr()
    )
    assert_equal(actual_list.list_column().child().dtype(), DataType.DATE)
    assert_true(
        actual_list.list_column().child()._field_metadata.value().ptr()
        == expected_list.list_column().child()._field_metadata.value().ptr()
    )


def test_window_mapping_scalar_nested_and_empty_recursive_metadata() raises:
    var source = _nested_mapping_prototype()
    var parts: List[Series] = [source.slice(0, 1), source.slice(1, 1)]
    var groups = List[Int]([0, 1, 0])
    var joined = assemble_window_mapping(parts, groups, "join", True, source)
    assert_equal(len(joined), 3)
    _assert_nested_mapping_metadata(joined, source)
    var exploded = assemble_window_mapping(
        parts, groups, "explode", True, source
    )
    assert_equal(len(exploded), 2)
    _assert_nested_mapping_metadata(exploded, source)
    var lists = source.struct_column().field("dates")
    var list_parts: List[Series] = [lists.slice(0, 1), lists.slice(1, 1)]
    var scalar_lists = assemble_window_mapping(
        list_parts, groups, "join", True, lists
    )
    assert_equal(scalar_lists.dtype(), lists.dtype())
    assert_equal(len(scalar_lists), 3)
    assert_equal(len(scalar_lists.get(2).list()), 1)
    var wrapped = assemble_window_mapping(parts, groups, "join", False, source)
    assert_equal(len(wrapped), 3)
    _assert_nested_mapping_metadata(wrapped.list_column().child(), source)
    var no_parts = List[Series]()
    var no_groups = List[Int]()
    for scalar in [False, True]:
        for strategy in [String("join"), String("explode")]:
            var empty = assemble_window_mapping(
                no_parts, no_groups, strategy, scalar, source
            )
            assert_equal(len(empty), 0)
            if strategy == "join" and not scalar:
                _assert_nested_mapping_metadata(
                    empty.list_column().child(), source
                )
            else:
                _assert_nested_mapping_metadata(empty, source)
