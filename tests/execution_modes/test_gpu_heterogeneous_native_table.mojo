"""Tagged native owners and one-map heterogeneous native Table operations."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_raises, assert_true
from std.memory import ArcPointer, bitcast
from dataframe import Column, DataFrame, DataType, Series
from dataframe.dtype import CategoricalDictionary
from dataframe.accelerator.code_domains import CategoryField, CategoryOwner
from dataframe.accelerator.gpu.query_scope import Scope
from dataframe.accelerator.gpu.residency import (
    Matrix,
    NativeJoinField,
    NativeRowMap,
    Table,
)
from dataframe.accelerator.gpu.runtime import materialize
from dataframe.accelerator.gpu.typed_storage import (
    NativeColumnAny,
    allocate_native_column,
    upload_native,
)
from tests.execution_modes.test_backend_metal import require_metal


def test_native_bool_descriptor_is_tagged_without_admitting_host_transport() raises:
    var scope = Scope()
    var owner = allocate_native_column[DType.bool](scope, DataType.BOOL, 0)
    var tagged = NativeColumnAny(owner)
    var view = tagged.view(0, 0)
    assert_equal(view.logical, DataType.BOOL)
    assert_equal(view.length, 0)
    assert_equal(view.owner_identity(), tagged.owner_identity())
    var field = Series("b", Column[Bool](List[Bool]()))
    with assert_raises(contains="Native Bool transport is not implemented"):
        _ = tagged.readback_series(field)
    var table = Table.native(
        DataFrame(List[Series]([field.copy()])),
        List[NativeColumnAny]([tagged.copy()]),
    )
    assert_equal(table.rows, 0)


# requires: metal
def test_native_any_view_preserves_temporal_logical_dtype_and_bitmap() raises:
    _ = require_metal()
    var source = (
        Series(
            "day",
            Column[Int64](
                [90, 10, 20, 30, 40, 99], [True, True, False, True, True, True]
            ),
        )
        .with_dtype(DataType.DATE)
        .slice(1, 4)
    )
    var scope = Scope()
    var typed = upload_native[DType.int64](scope, source)
    var any = NativeColumnAny(typed)
    var view = any.view(1, 2)
    assert_equal(any.logical, DataType.DATE)
    assert_equal(view.logical, DataType.DATE)
    assert_equal(view.value_offset, typed.value_offset + 1)
    assert_equal(view.validity_bit_offset, typed.validity_bit_offset + 1)
    assert_equal(any.owner_identity(), view.owner_identity())
    _ = any^
    _ = typed^
    assert_equal(scope.info()[0], 33)

    var field = Series("day", Column[Int64](List[Int64]())).with_dtype(
        DataType.DATE
    )
    field._field_metadata = ArcPointer(
        List[UInt8]([UInt8(1), UInt8(2), UInt8(3)])
    )
    var result = view.readback_series(field)
    assert_equal(result.dtype(), DataType.DATE)
    assert_equal(
        result._field_metadata.value()[], field._field_metadata.value()[]
    )
    assert_equal(result.int64()._get(0), Int64(20))
    assert_equal(result.int64()._get(1), Int64(30))
    assert_equal(result.int64().is_valid(0), False)
    assert_equal(result.int64().is_valid(1), True)
    _ = view^
    assert_equal(scope.info()[0], 0)


# requires: metal
def test_native_table_mixed_gather_reuses_one_index_map_and_materializes() raises:
    _ = require_metal()
    var scope = Scope()
    var floats = upload_native[DType.float32](
        scope, Series("f", Column[Float32]([1.5, 2.5, 3.5]))
    )
    var small = upload_native[DType.int16](
        scope,
        Series("i", Column[Int16]([7, 8, 9], [True, False, True])),
    )
    var schema = DataFrame(
        List[Series](
            [
                Series("f", Column[Float32](List[Float32]())),
                Series("i", Column[Int16](List[Int16]())),
            ]
        )
    )
    var columns = List[NativeColumnAny](
        [NativeColumnAny(floats), NativeColumnAny(small)]
    )
    var table = Table.native(schema.copy(), columns^)
    var cells = 6
    var context = scope.context()
    var matrix_words = context.enqueue_create_buffer[DType.uint64](2 * cells)
    var matrix_validity = context.enqueue_create_buffer[DType.uint8](cells)
    var matrix_errors = context.enqueue_create_buffer[DType.uint32](cells)
    var copied_bytes = table.copy_to(
        scope,
        matrix_words,
        matrix_validity,
        matrix_errors,
        3,
        2,
        List[Int64]([Int64(0), Int64(1)]),
    )
    assert_equal(copied_bytes, 3 * 2 * 21)
    var matrix_table = Table(
        Matrix(
            scope,
            3,
            2,
            matrix_words^,
            matrix_validity^,
            matrix_errors^,
        ),
        schema.copy(),
        List[Int64]([Int64(0), Int64(1)]),
        3,
    )
    var copied = materialize(matrix_table)
    assert_equal(copied.column("f").float32()._get(2), Float32(3.5))
    assert_equal(copied.column("i").int16()._get(1), Int16(0))
    assert_equal(copied.column("i").int16().is_valid(1), False)

    var before = scope.transfers()
    var selected = table.gather(scope, List[Int]([2, 1, 0]))
    var after = scope.transfers()
    assert_equal(after[3] - before[3], 1)
    assert_equal(after[2] - before[2], 2)
    assert_equal(after[0] - before[0], 12)
    assert_equal(selected.rows, 3)

    var materialized = materialize(selected)
    assert_equal(materialized.column("f").float32()._get(0), Float32(3.5))
    assert_equal(materialized.column("f").float32()._get(1), Float32(2.5))
    assert_equal(materialized.column("i").int16()._get(0), Int16(9))
    assert_equal(materialized.column("i").int16().is_valid(1), False)
    assert_equal(materialized.column("i").int16()._get(2), Int16(7))

    var with_missing = table.gather(
        scope, List[Int]([-1, 1]), allow_missing=True
    )
    var nullable = materialize(with_missing)
    assert_equal(nullable.column("f").float32().is_valid(0), False)
    assert_equal(nullable.column("f").float32()._get(1), Float32(2.5))
    assert_equal(nullable.column("i").int16().is_valid(0), False)
    assert_equal(nullable.column("i").int16().is_valid(1), False)


# requires: metal
def test_native_float64_gather_and_matrix_bridge_preserve_bits() raises:
    _ = require_metal()
    var scope = Scope()
    var bits = List[UInt64](
        [
            UInt64(0x7FF8000000000031),
            UInt64(0x8000000000000000),
            UInt64(1),
            UInt64(0x7FF0000000000000),
            UInt64(0xFFF0000000000000),
            UInt64(0x7FF8000000000099),
        ]
    )
    var values = List[Float64]()
    for value in bits:
        values.append(bitcast[DType.float64](value))
    var source = Series(
        "x", Column[Float64](values^, [True, True, True, True, True, False])
    )
    var owner = upload_native[DType.float64](scope, source)
    var schema = DataFrame(
        List[Series]([Series("x", Column[Float64](List[Float64]()))])
    )
    var table = Table.native(
        schema.copy(), List[NativeColumnAny]([NativeColumnAny(owner)])
    )
    var indices = List[Int]([2, 1, 0, 3, 4, 5, -1])
    var selected = table.gather(scope, indices, allow_missing=True)
    var direct = materialize(selected)
    var rows = len(indices)
    var context = scope.context()
    var words = context.enqueue_create_buffer[DType.uint64](2 * rows)
    var validity = context.enqueue_create_buffer[DType.uint8](rows)
    var errors = context.enqueue_create_buffer[DType.uint32](rows)
    assert_equal(
        selected.copy_to(
            scope, words, validity, errors, rows, 1, List[Int64]([Int64(0)])
        ),
        rows * 21,
    )
    var generic = Table(
        Matrix(scope, rows, 1, words^, validity^, errors^),
        schema.copy(),
        List[Int64]([Int64(0)]),
        rows,
    )
    var bridged = materialize(generic)
    for row in range(rows):
        var present = row < 5
        assert_equal(direct.column("x").float64().is_valid(row), present)
        assert_equal(bridged.column("x").float64().is_valid(row), present)
        var expected = bits[indices[row]] if present else UInt64(0)
        assert_equal(
            bitcast[DType.uint64](direct.column("x").float64()._get(row)),
            expected,
        )
        assert_equal(
            bitcast[DType.uint64](bridged.column("x").float64()._get(row)),
            expected,
        )


# requires: metal
def test_native_table_gather_budget_and_unsupported_storage_fail_before_transfer() raises:
    _ = require_metal()
    var scope = Scope(budget=19)
    var floats = upload_native[DType.float32](
        scope, Series("f", Column[Float32]([1, 2, 3]))
    )
    var small = upload_native[DType.int16](
        scope, Series("i", Column[Int16]([4, 5, 6], [True, False, True]))
    )
    var schema = DataFrame(
        List[Series](
            [
                Series("f", Column[Float32](List[Float32]())),
                Series("i", Column[Int16](List[Int16]())),
            ]
        )
    )
    var table = Table.native(
        schema.copy(),
        List[NativeColumnAny](
            [NativeColumnAny(floats), NativeColumnAny(small)]
        ),
    )
    var before = scope.transfers()
    with assert_raises(contains="memory budget"):
        _ = table.gather(scope, List[Int]([2, 0]))
    assert_equal(scope.info()[0], 19)
    assert_equal(scope.transfers()[0], before[0])
    assert_equal(scope.transfers()[1], before[1])
    assert_equal(scope.transfers()[2], before[2])
    assert_equal(scope.transfers()[3], before[3])

    var dictionary = CategoricalDictionary()
    dictionary.append("metadata-only")
    var category_owner = ArcPointer(
        CategoryOwner(DataType.categorical(dictionary^), List[Column[UInt32]]())
    )
    table.categories[0] = CategoryField(category_owner.copy())
    var metadata_before = scope.transfers()
    with assert_raises(contains="cannot carry text metadata"):
        _ = table.gather(scope, List[Int]([0]))
    var metadata_after = scope.transfers()
    assert_equal(metadata_after[0], metadata_before[0])
    assert_equal(metadata_after[1], metadata_before[1])
    assert_equal(metadata_after[2], metadata_before[2])
    assert_equal(metadata_after[3], metadata_before[3])

    var decimal = Series("d", Column[Int64]([1])).with_dtype(
        DataType.decimal(18, 0, 64)
    )
    with assert_raises(contains="supported fixed-width dtype"):
        _ = upload_native[DType.int64](scope, decimal)
    var boolean = Series("b", Column[Bool]([True, False]))
    with assert_raises(contains="Native Bool upload is not implemented"):
        _ = upload_native[DType.bool](scope, boolean)


# requires: metal
def test_native_gather_carries_deferred_error_sidecars() raises:
    _ = require_metal()
    var scope = Scope()
    var source = allocate_native_column[DType.int16](
        scope, DataType.INT16, 3, with_errors=True
    )
    var values = source.values()
    var errors = source.errors()
    var host_values = List[Int16]([Int16(10), Int16(20), Int16(30)])
    var host_errors = List[UInt32]([UInt32(5), UInt32(17), UInt32(29)])
    var context = scope.context()
    context.enqueue_copy(values, host_values.unsafe_ptr())
    context.enqueue_copy(errors.value(), host_errors.unsafe_ptr())
    scope.record_upload(3 * 2 + 3 * 4)
    scope.synchronize()

    var field = Series("value", Column[Int16](List[Int16]()))
    var table = Table.native(
        DataFrame(List[Series]([field.copy()])),
        List[NativeColumnAny]([NativeColumnAny(source)]),
    )
    var before = scope.transfers()
    var gathered = table.gather(scope, List[Int]([2, 0]))
    var after = scope.transfers()
    assert_equal(after[3] - before[3], 1)
    assert_true(gathered.native_columns[0].has_errors())
    var gathered_errors = gathered.native_columns[0].read_error_codes()
    assert_equal(gathered_errors[0], UInt32(29))
    assert_equal(gathered_errors[1], UInt32(5))


# requires: metal
def test_native_join_gather_coalesces_device_maps_with_one_drain() raises:
    _ = require_metal()
    var scope = Scope()
    var left_id = upload_native[DType.int64](
        scope, Series("id", Column[Int64]([99, 1, 2, 3, 4]))
    )
    var left_value = upload_native[DType.int16](
        scope,
        Series("left_value", Column[Int16]([99, 10, 20, 30, 40])),
    )
    var right_id = upload_native[DType.int64](
        scope, Series("id", Column[Int64]([88, 11, 12, 13, 14]))
    )
    var right_value = upload_native[DType.float32](
        scope,
        Series("right_value", Column[Float32]([8, 5, 6, 7, 8])),
    )
    var left_schema = DataFrame(
        List[Series](
            [
                Series("id", Column[Int64](List[Int64]())),
                Series("left_value", Column[Int16](List[Int16]())),
            ]
        )
    )
    var right_schema = DataFrame(
        List[Series](
            [
                Series("id", Column[Int64](List[Int64]())),
                Series("right_value", Column[Float32](List[Float32]())),
            ]
        )
    )
    var left = Table.native(
        left_schema.copy(),
        List[NativeColumnAny](
            [NativeColumnAny(left_id), NativeColumnAny(left_value)]
        ),
    )
    var right = Table.native(
        right_schema.copy(),
        List[NativeColumnAny](
            [NativeColumnAny(right_id), NativeColumnAny(right_value)]
        ),
    )
    # Both tables are non-owning row views; gathers must add each table offset.
    left.offset = 1
    left.rows = 4
    right.offset = 1
    right.rows = 4

    var context = scope.context()
    var left_device = context.enqueue_create_buffer[DType.uint32](4)
    var right_device = context.enqueue_create_buffer[DType.uint32](4)
    var left_host = List[UInt32]([UInt32(0), UInt32(1), UInt32.MAX, UInt32(3)])
    var right_host = List[UInt32]([UInt32(0), UInt32.MAX, UInt32(1), UInt32(3)])
    context.enqueue_copy(left_device, left_host.unsafe_ptr())
    context.enqueue_copy(right_device, right_host.unsafe_ptr())
    scope.record_upload(4 * 4 * 2)
    scope.synchronize()
    var left_map = NativeRowMap(scope, left_device, 4, left.rows, True)
    var right_map = NativeRowMap(scope, right_device, 4, right.rows, True)

    var output_id = Series("id", Column[Int64](List[Int64]()))
    output_id._field_metadata = ArcPointer(
        List[UInt8]([UInt8(4), UInt8(5), UInt8(6)])
    )
    var output_schema = DataFrame(
        List[Series](
            [
                output_id.copy(),
                Series("left_value", Column[Int16](List[Int16]())),
                Series("right_value", Column[Float32](List[Float32]())),
            ]
        )
    )
    var fields = List[NativeJoinField](
        [
            NativeJoinField(0, 0, True),
            NativeJoinField(1, -1, False),
            NativeJoinField(-1, 1, False),
        ]
    )
    var before = scope.transfers()
    var gathered = left.gather_join(
        scope, right, output_schema, fields, left_map, right_map
    )
    var after = scope.transfers()
    assert_equal(after[0], before[0])
    assert_equal(after[3] - before[3], 1)
    assert_equal(after[2] - before[2], 4)
    assert_equal(gathered.rows, 4)
    var materialized = materialize(gathered)
    assert_equal(materialized.column("id").int64()._get(0), Int64(11))
    assert_equal(materialized.column("id").int64()._get(1), Int64(2))
    assert_equal(materialized.column("id").int64()._get(2), Int64(12))
    assert_equal(materialized.column("id").int64()._get(3), Int64(14))
    assert_equal(materialized.column("left_value").int16()._get(0), Int16(10))
    assert_equal(materialized.column("left_value").int16().is_valid(2), False)
    assert_equal(
        materialized.column("right_value").float32()._get(1), Float32(0)
    )
    assert_equal(
        materialized.column("right_value").float32().is_valid(1), False
    )
    assert_equal(
        materialized.column("id")._field_metadata.value()[],
        output_id._field_metadata.value()[],
    )


# requires: metal
def test_native_join_gather_small_payloads_use_safe_unused_error_buffers() raises:
    _ = require_metal()
    var scope = Scope()
    var int8 = upload_native[DType.int8](
        scope, Series("i8", Column[Int8]([11, 12, 13]))
    )
    var uint8 = upload_native[DType.uint8](
        scope, Series("u8", Column[UInt8]([21, 22, 23]))
    )
    var int16 = upload_native[DType.int16](
        scope, Series("i16", Column[Int16]([31, 32, 33]))
    )
    var right_value = upload_native[DType.uint8](
        scope, Series("r", Column[UInt8]([99]))
    )
    var left_schema = DataFrame(
        List[Series](
            [
                Series("i8", Column[Int8](List[Int8]())),
                Series("u8", Column[UInt8](List[UInt8]())),
                Series("i16", Column[Int16](List[Int16]())),
            ]
        )
    )
    var right_schema = DataFrame(
        List[Series]([Series("r", Column[UInt8](List[UInt8]()))])
    )
    var left = Table.native(
        left_schema.copy(),
        List[NativeColumnAny](
            [
                NativeColumnAny(int8),
                NativeColumnAny(uint8),
                NativeColumnAny(int16),
            ]
        ),
    )
    var right = Table.native(
        right_schema.copy(),
        List[NativeColumnAny]([NativeColumnAny(right_value)]),
    )
    var context = scope.context()
    var left_device = context.enqueue_create_buffer[DType.uint32](3)
    var right_device = context.enqueue_create_buffer[DType.uint32](3)
    var left_host = List[UInt32]([UInt32(0), UInt32(1), UInt32(2)])
    var right_host = List[UInt32]([UInt32.MAX, UInt32.MAX, UInt32.MAX])
    context.enqueue_copy(left_device, left_host.unsafe_ptr())
    context.enqueue_copy(right_device, right_host.unsafe_ptr())
    scope.record_upload(24)
    scope.synchronize()
    var left_map = NativeRowMap(scope, left_device, 3, 3, False)
    var right_map = NativeRowMap(scope, right_device, 3, 1, True)
    var output_schema = DataFrame(
        List[Series](
            [
                Series("i8", Column[Int8](List[Int8]())),
                Series("u8", Column[UInt8](List[UInt8]())),
                Series("i16", Column[Int16](List[Int16]())),
            ]
        )
    )
    var fields = List[NativeJoinField](
        [
            NativeJoinField(0, -1, False),
            NativeJoinField(1, -1, False),
            NativeJoinField(2, -1, False),
        ]
    )

    var before = scope.transfers()
    var gathered = left.gather_join(
        scope, right, output_schema, fields, left_map, right_map
    )
    var after = scope.transfers()
    assert_equal(after[0], before[0])
    assert_equal(after[2] - before[2], 4)
    assert_equal(after[3] - before[3], 1)
    var result = materialize(gathered)
    assert_equal(result.column("i8").int8()._get(0), Int8(11))
    assert_equal(result.column("i8").int8()._get(2), Int8(13))
    assert_equal(result.column("u8").uint8()._get(1), UInt8(22))
    assert_equal(result.column("i16").int16()._get(2), Int16(33))


# requires: metal
def test_native_join_gather_bounds_errors_budget_and_empty() raises:
    _ = require_metal()
    var scope = Scope()
    var left_values = allocate_native_column[DType.int32](
        scope, DataType.INT32, 2, with_errors=True
    )
    var values = left_values.values()
    var errors = left_values.errors()
    var host_values = List[Int32]([Int32(17), Int32(29)])
    var host_errors = List[UInt32]([UInt32(5), UInt32(13)])
    var context = scope.context()
    context.enqueue_copy(values, host_values.unsafe_ptr())
    context.enqueue_copy(errors.value(), host_errors.unsafe_ptr())
    scope.record_upload(2 * 8)
    scope.synchronize()
    var right_values = upload_native[DType.int32](
        scope, Series("right", Column[Int32]([71, 72]))
    )
    var left_field = Series("left", Column[Int32](List[Int32]()))
    var right_field = Series("right", Column[Int32](List[Int32]()))
    var left = Table.native(
        DataFrame(List[Series]([left_field.copy()])),
        List[NativeColumnAny]([NativeColumnAny(left_values)]),
    )
    var right = Table.native(
        DataFrame(List[Series]([right_field.copy()])),
        List[NativeColumnAny]([NativeColumnAny(right_values)]),
    )
    var left_device = context.enqueue_create_buffer[DType.uint32](2)
    var right_device = context.enqueue_create_buffer[DType.uint32](2)
    var left_host = List[UInt32]([UInt32(1), UInt32(0)])
    var right_host = List[UInt32]([UInt32(0), UInt32.MAX])
    context.enqueue_copy(left_device, left_host.unsafe_ptr())
    context.enqueue_copy(right_device, right_host.unsafe_ptr())
    scope.record_upload(16)
    scope.synchronize()
    var left_map = NativeRowMap(scope, left_device, 2, 2, False)
    var right_map = NativeRowMap(scope, right_device, 2, 2, True)
    var output_schema = DataFrame(
        List[Series]([left_field.copy(), right_field.copy()])
    )
    var fields = List[NativeJoinField](
        [NativeJoinField(0, -1, False), NativeJoinField(-1, 0, False)]
    )
    var gathered = left.gather_join(
        scope, right, output_schema, fields, left_map, right_map
    )
    assert_true(gathered.native_columns[0].has_errors())
    var error_codes = gathered.native_columns[0].read_error_codes()
    assert_equal(error_codes[0], UInt32(13))
    assert_equal(error_codes[1], UInt32(5))

    var empty_left_device = context.enqueue_create_buffer[DType.uint32](1)
    var empty_right_device = context.enqueue_create_buffer[DType.uint32](1)
    var empty_left = NativeRowMap(scope, empty_left_device, 0, 2, False)
    var empty_right = NativeRowMap(scope, empty_right_device, 0, 2, False)
    var before_empty = scope.transfers()
    var empty = left.gather_join(
        scope, right, output_schema, fields, empty_left, empty_right
    )
    assert_equal(empty.rows, 0)
    assert_equal(scope.transfers()[3], before_empty[3])

    var bad_device = context.enqueue_create_buffer[DType.uint32](2)
    var bad_host = List[UInt32]([UInt32(2), UInt32(0)])
    context.enqueue_copy(bad_device, bad_host.unsafe_ptr())
    scope.record_upload(8)
    scope.synchronize()
    var bad_left = NativeRowMap(scope, bad_device, 2, 2, False)
    var before_bad = scope.transfers()
    with assert_raises(contains="row map is outside its bounds"):
        _ = left.gather_join(
            scope, right, output_schema, fields, bad_left, right_map
        )
    var after_bad = scope.transfers()
    assert_equal(after_bad[0], before_bad[0])
    assert_equal(after_bad[3] - before_bad[3], 1)

    var both_missing_left_device = context.enqueue_create_buffer[DType.uint32](
        1
    )
    var both_missing_right_device = context.enqueue_create_buffer[DType.uint32](
        1
    )
    var both_missing_host = List[UInt32]([UInt32.MAX])
    context.enqueue_copy(
        both_missing_left_device, both_missing_host.unsafe_ptr()
    )
    context.enqueue_copy(
        both_missing_right_device, both_missing_host.unsafe_ptr()
    )
    scope.record_upload(8)
    scope.synchronize()
    var both_missing_left = NativeRowMap(
        scope, both_missing_left_device, 1, 2, True
    )
    var both_missing_right = NativeRowMap(
        scope, both_missing_right_device, 1, 2, True
    )
    with assert_raises(contains="row map is outside its bounds"):
        _ = left.gather_join(
            scope,
            right,
            output_schema,
            fields,
            both_missing_left,
            both_missing_right,
        )

    var bool_owner = allocate_native_column[DType.bool](scope, DataType.BOOL, 2)
    var bool_field = Series("flag", Column[Bool](List[Bool]()))
    var bool_table = Table.native(
        DataFrame(List[Series]([bool_field.copy()])),
        List[NativeColumnAny]([NativeColumnAny(bool_owner)]),
    )
    var bool_schema = DataFrame(List[Series]([bool_field.copy()]))
    var bool_fields = List[NativeJoinField]([NativeJoinField(0, -1, False)])
    with assert_raises(contains="dtype differs or is unsupported"):
        _ = bool_table.gather_join(
            scope, right, bool_schema, bool_fields, left_map, right_map
        )

    var limited = Scope(budget=32)
    var limited_left = upload_native[DType.int32](
        limited, Series("left", Column[Int32]([1, 2]))
    )
    var limited_right = upload_native[DType.int32](
        limited, Series("right", Column[Int32]([3, 4]))
    )
    var limited_left_table = Table.native(
        DataFrame(List[Series]([left_field.copy()])),
        List[NativeColumnAny]([NativeColumnAny(limited_left)]),
    )
    var limited_right_table = Table.native(
        DataFrame(List[Series]([right_field.copy()])),
        List[NativeColumnAny]([NativeColumnAny(limited_right)]),
    )
    var limited_context = limited.context()
    var limited_left_map = limited_context.enqueue_create_buffer[DType.uint32](
        2
    )
    var limited_right_map = limited_context.enqueue_create_buffer[DType.uint32](
        2
    )
    var limited_left_host = List[UInt32]([UInt32(0), UInt32(1)])
    var limited_right_host = List[UInt32]([UInt32(0), UInt32(1)])
    limited_context.enqueue_copy(
        limited_left_map, limited_left_host.unsafe_ptr()
    )
    limited_context.enqueue_copy(
        limited_right_map, limited_right_host.unsafe_ptr()
    )
    limited.record_upload(16)
    limited.synchronize()
    var limited_left_rows = NativeRowMap(limited, limited_left_map, 2, 2, False)
    var limited_right_rows = NativeRowMap(
        limited, limited_right_map, 2, 2, False
    )
    var before_limited = limited.transfers()
    with assert_raises(contains="memory budget"):
        _ = limited_left_table.gather_join(
            limited,
            limited_right_table,
            output_schema,
            fields,
            limited_left_rows,
            limited_right_rows,
        )
    var after_limited = limited.transfers()
    assert_equal(after_limited[0], before_limited[0])
    assert_equal(after_limited[1], before_limited[1])
    assert_equal(after_limited[2], before_limited[2])
    assert_equal(after_limited[3], before_limited[3])
