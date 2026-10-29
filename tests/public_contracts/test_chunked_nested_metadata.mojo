"""Nested metadata consumers read source windows without merging parents."""
# contract-group: expr.chunk_windows
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false, assert_raises
from dataframe import (
    Column,
    DataFrame,
    DataType,
    ListColumn,
    Series,
    StructColumn,
    col,
)
from dataframe.expr import LIST_LEN, STRUCT_FIELD, _node
from dataframe.kernels.list import nested_op
from dataframe.dtype import CategoricalDictionary


def _uneven(source: Series) raises -> Series:
    return Series._from_chunks(
        [source.slice(1, 2), source.slice(3, 1), source.slice(4, 3)]
    )


def _lists() raises -> Series:
    # A null row still has a physical child span. The child itself is chunked
    # and nullable: counting offsets must neither read nor merge that child.
    var child = Series._from_chunks(
        [
            Series("item", Column[Int64]([999, 10, 20])),
            Series("item", Column[Int64]([30, 40, 50], [False, True, True])),
            Series("item", Column[Int64]([60, 70, 80, 90])),
        ]
    )
    return Series(
        "xs",
        ListColumn([Int64(0), 1, 3, 3, 6, 7, 7, 10], child^, [UInt8(119)]),
    )


def _check_lengths(actual: Series) raises:
    assert_equal(actual.dtype(), DataType.INT64)
    assert_equal(len(actual), 6)
    var expected: List[Int64] = [2, 0, 0, 1, 0, 3]
    for row in range(6):
        assert_equal(actual.get(row).is_null(), row == 2)
        if row != 2:
            assert_equal(actual.get(row).int64(), expected[row])


def test_list_lengths_use_only_sliced_parent_metadata() raises:
    var source = _uneven(_lists())
    var actual = nested_op(_node(LIST_LEN), source, source.dtype())
    _check_lengths(actual)
    assert_equal(actual.name(), "")
    assert_false(actual.is_chunked())
    assert_equal(source._chunked.value()[].cache[].state.load(), 0)
    for part in source.chunks():
        var child = part.list_column().child()
        assert_equal(child._chunked.value()[].cache[].state.load(), 0)
    var frame = DataFrame([source^])
    var result = frame.select_exprs([col("xs").list().len().alias("n")])
    _check_lengths(result.column("n"))
    var baseline = DataFrame([_lists().slice(1, 6)]).select_exprs(
        [col("xs").list().len().alias("n")]
    )
    assert_true(result.equals(baseline))


def _record(nullable: Bool) raises -> Series:
    var days = Series(
        "day",
        Column[Int64](
            [99, 11, 12, 13, 14, 15, 16],
            [True, True, False, True, True, True, True],
        ),
    ).with_dtype(DataType.DATE)
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    days._field_metadata = ArcPointer(metadata^)
    var category = Series(
        "category", Column[String](["z", "a", "b", "a", "b", "a", "b"])
    ).cast("categorical")
    var unused = Series._from_chunks(
        [
            Series("unused", Column[Int64]([90, 91])),
            Series("unused", Column[Int64]([92, 93, 94, 95, 96])),
        ]
    )
    var bits = List[UInt8]([UInt8(119)]) if nullable else List[UInt8]()
    return Series("record", StructColumn([days^, category^, unused^], bits^))


def _check_days(actual: Series, nullable: Bool) raises:
    assert_equal(actual.dtype(), DataType.DATE)
    assert_equal(len(actual), 6)
    for row in range(6):
        var missing = row == 1 or (nullable and row == 2)
        assert_equal(actual.get(row).is_null(), missing)
        if not missing:
            assert_equal(
                actual.with_dtype(DataType.INT64).get(row).int64(),
                Int64(11 + row),
            )


def test_struct_projection_preserves_field_schema_and_parent_mask() raises:
    for nullable in [False, True]:
        var original = _record(nullable)
        var expected_field = original.struct_column().field("day")
        var source = _uneven(original)
        var actual = nested_op(
            _node(STRUCT_FIELD, text="day"), source, source.dtype()
        )
        _check_days(actual, nullable)
        assert_equal(actual.name(), "")
        assert_true(actual.is_chunked())
        assert_true(
            actual._field_metadata.value().ptr()
            == expected_field._field_metadata.value().ptr()
        )
        assert_equal(source._chunked.value()[].cache[].state.load(), 0)
        for part in source.chunks():
            var unused = part.struct_column().field("unused")
            if unused.is_chunked():
                assert_equal(unused._chunked.value()[].cache[].state.load(), 0)
        if not nullable:
            var first = source.chunks()[0].struct_column().field("day")
            assert_true(actual.chunks()[0]._shares_buffers_with(first))
        var frame = DataFrame([source^])
        var result = frame.select_exprs(
            [col("record").field("day").alias("day")]
        )
        _check_days(result.column("day"), nullable)
        var baseline = DataFrame([_record(nullable).slice(1, 6)]).select_exprs(
            [col("record").field("day").alias("day")]
        )
        assert_true(result.equals(baseline))


def _owned_category_projection() raises -> Series:
    var source = _uneven(_record(True))
    var field = source.chunks()[0].struct_column().field("category")
    var result = nested_op(
        _node(STRUCT_FIELD, text="category"), source, source.dtype()
    )
    assert_true(
        result.dtype().dictionary().ptr() == field.dtype().dictionary().ptr()
    )
    assert_equal(source._chunked.value()[].cache[].state.load(), 0)
    return result^


def test_struct_dictionary_projection_outlives_parent_and_keeps_domain() raises:
    var actual = _owned_category_projection()
    assert_true(actual.dtype().is_categorical())
    assert_equal(len(actual), 6)
    var labels: List[String] = ["a", "b", "a", "b", "a", "b"]
    for row in range(6):
        assert_equal(actual.get(row).is_null(), row == 2)
        if row != 2:
            assert_equal(actual.get(row).string(), labels[row])
    var original = _record(True).struct_column().field("category")
    assert_equal(actual.dtype(), original.dtype())


def test_empty_nested_metadata_and_missing_field_errors() raises:
    var lists = _lists().slice(2, 0)
    var lengths = nested_op(_node(LIST_LEN), lists, lists.dtype())
    assert_equal(lengths.dtype(), DataType.INT64)
    assert_equal(len(lengths), 0)
    var records = _record(True).slice(2, 0)
    var field = nested_op(
        _node(STRUCT_FIELD, text="day"), records, records.dtype()
    )
    assert_equal(field.dtype(), DataType.DATE)
    assert_equal(len(field), 0)
    var chunked = _uneven(_record(True))
    with assert_raises(contains="Struct has no field named absent"):
        _ = nested_op(
            _node(STRUCT_FIELD, text="absent"), chunked, chunked.dtype()
        )
    assert_equal(chunked._chunked.value()[].cache[].state.load(), 0)


def _owned_null_lengths() raises -> Series:
    var source = Series(
        "xs",
        ListColumn(
            [Int64(0), 1, 3, 4],
            Series("item", Column[Int64]([91, 92, 93, 94])),
            [UInt8(0)],
        ),
    )
    var chunked = Series._from_chunks([source.slice(0, 1), source.slice(1, 2)])
    var result = nested_op(_node(LIST_LEN), chunked, chunked.dtype())
    assert_equal(chunked._chunked.value()[].cache[].state.load(), 0)
    return result^


def test_all_null_lengths_outlive_parent_with_zero_payloads() raises:
    var actual = _owned_null_lengths()
    assert_equal(actual.dtype(), DataType.INT64)
    assert_equal(len(actual), 3)
    ref column = actual._data[Column[Int64]]
    for row in range(3):
        assert_true(actual.get(row).is_null())
        assert_equal(column._get(row), Int64(0))


def test_struct_projection_unifies_different_chunk_dictionaries_and_masks() raises:
    var left_domain = CategoricalDictionary()
    for label in ["red", "blue", "unused-left"]:
        left_domain.append(label)
    var right_domain = CategoricalDictionary()
    for label in ["blue", "red", "green", "unused-right"]:
        right_domain.append(label)
    var left_field = Series(
        "label",
        Column[UInt32]([0, 1, 2, 0], [True, True, False, True]),
    ).with_dtype(DataType.categorical(left_domain^, distinct=True))
    var right_field = Series(
        "label",
        Column[UInt32]([2, 0, 1, 2], [True, False, True, True]),
    ).with_dtype(DataType.categorical(right_domain^, distinct=True))
    var metadata: List[UInt8] = [1, 0, 0, 0, 1, 0, 0, 0, 107, 1, 0, 0, 0, 118]
    left_field._field_metadata = ArcPointer(metadata^)
    right_field._field_metadata = left_field._field_metadata
    var left_unused = Series._from_chunks(
        [
            Series("unused", Column[Int64]([90, 91])),
            Series("unused", Column[Int64]([92, 93])),
        ]
    )
    var right_unused = Series._from_chunks(
        [
            Series("unused", Column[Int64]([94])),
            Series("unused", Column[Int64]([95, 96, 97])),
        ]
    )
    var left = Series(
        "record", StructColumn([left_field.copy(), left_unused^], [UInt8(7)])
    )
    var right = Series(
        "record", StructColumn([right_field^, right_unused^], [UInt8(15)])
    )
    # Parent schemas match even though categorical codes have different
    # meanings. The nonzero slice also crosses a child storage boundary.
    assert_equal(left.dtype(), right.dtype())
    var source = Series._from_chunks([left.slice(1, 3), right.slice(0, 3)])
    var actual = nested_op(
        _node(STRUCT_FIELD, text="label"), source, source.dtype()
    )
    var domain = actual.dtype().dictionary()
    var labels: List[String] = [
        "red",
        "blue",
        "unused-left",
        "green",
        "unused-right",
    ]
    assert_equal(len(domain[]), len(labels))
    for code in range(len(labels)):
        assert_equal(String(domain[].get(code)), labels[code])
    assert_true(
        actual._field_metadata.value().ptr()
        == left_field._field_metadata.value().ptr()
    )
    assert_equal(source._chunked.value()[].cache[].state.load(), 0)
    for part in source.chunks():
        var unused = part.struct_column().field("unused")
        assert_true(unused.is_chunked())
        assert_equal(unused._chunked.value()[].cache[].state.load(), 0)
    var frame = DataFrame([source^])
    var public = frame.select_exprs(
        [col("record").field("label").alias("label")], batch_size=6
    )
    var results: List[Series] = [actual^, public.column("label")]
    var expected: List[String] = ["blue", "", "", "green", "", "red"]
    var valid: List[Bool] = [True, False, False, True, False, True]
    for result in results:
        assert_true(result.dtype().is_categorical())
        assert_equal(len(result), len(expected))
        for row in range(len(expected)):
            assert_equal(result.get(row).is_null(), not valid[row])
            if valid[row]:
                assert_equal(result.get(row).string(), expected[row])
