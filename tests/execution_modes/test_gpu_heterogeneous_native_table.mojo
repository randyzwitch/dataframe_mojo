"""Tagged native owners and one-map heterogeneous native Table operations."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_raises, assert_true
from std.memory import ArcPointer
from dataframe import Column, DataFrame, DataType, Series
from dataframe.dtype import CategoricalDictionary
from dataframe.accelerator.code_domains import CategoryField, CategoryOwner
from dataframe.accelerator.gpu.query_scope import Scope
from dataframe.accelerator.gpu.residency import Matrix, Table
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
