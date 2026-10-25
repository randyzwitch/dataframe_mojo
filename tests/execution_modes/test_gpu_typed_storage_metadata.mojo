"""Typed-column footprint checks and native Float32 transfer roundtrips."""
# contract-group: execution.backends
from std.testing import assert_equal, assert_raises
from dataframe import Column, Series
from dataframe_metal.accelerator.gpu.query_scope import Scope
from dataframe_metal.accelerator.gpu.typed_storage import (
    native_column_bytes,
    check_native_column_capacity,
    upload_float32,
    readback_float32,
    _NativeReservation,
)
from tests.support.metal_outcomes import require_metal


def test_native_float32_column_capacity_includes_only_present_sidecars() raises:
    assert_equal(native_column_bytes[DType.float32](0), 0)
    assert_equal(native_column_bytes[DType.float32](5), 20)
    assert_equal(native_column_bytes[DType.float32](5, 2), 22)
    assert_equal(native_column_bytes[DType.float32](5, 2, 5), 42)
    with assert_raises(contains="allocation size overflow"):
        _ = native_column_bytes[DType.float32](Int.MAX)


def test_native_column_capacity_checks_budget_before_allocation() raises:
    var scope = Scope(budget=41)
    with assert_raises(contains="memory budget"):
        _ = check_native_column_capacity[DType.float32](scope, 5, 2, 5)
    assert_equal(scope.info()[0], 0)
    assert_equal(scope.info()[1], 0)
    assert_equal(
        check_native_column_capacity[DType.float32](Scope(), 5, 2, 5), 42
    )


def test_native_reservation_precedes_allocation_and_releases_shared_scope() raises:
    var scope = Scope(budget=8)
    var copied = scope.copy()
    var reservation = _NativeReservation(scope, 8)
    assert_equal(copied.info()[0], 8)
    with assert_raises(contains="memory budget"):
        _ = _NativeReservation(copied, 1)
    assert_equal(scope.info()[0], 8)
    assert_equal(scope.initialized(), False)
    _ = reservation^
    assert_equal(scope.info()[0], 0)
    assert_equal(scope.info()[1], 8)
    var healthy = _NativeReservation(copied, 8)
    assert_equal(scope.info()[0], 8)
    _ = healthy^
    assert_equal(scope.info()[0], 0)


# requires: metal
def test_native_float32_upload_roundtrips_slice_and_packed_nulls() raises:
    _ = require_metal()
    var source = Series(
        "x",
        Column[Float32](
            [99, 1, 2, 3, 4, 88],
            [True, False, True, False, True, True],
        ),
    ).slice(1, 4)
    var scope = Scope()
    var resident = upload_float32(scope, source)
    assert_equal(resident.length, 4)
    assert_equal(resident.validity_bit_offset, 1)
    assert_equal(scope.transfers()[0], 17)
    var sliced = resident.view(1, 2)
    var result = readback_float32(sliced)
    assert_equal(result._get(0), Float32(2))
    assert_equal(result._get(1), Float32(3))
    assert_equal(result.is_valid(0), True)
    assert_equal(result.is_valid(1), False)
    assert_equal(scope.transfers()[1], 9)
    assert_equal(scope.info()[0], 17)
    _ = sliced^
    _ = resident^
    assert_equal(scope.info()[0], 0)


# requires: metal
def test_native_float32_upload_rechunks_only_validity_for_chunked_series() raises:
    _ = require_metal()
    var first = Series(
        "x", Column[Float32]([90, 1, 2], [True, False, True])
    ).slice(1, 2)
    var second = Series(
        "x", Column[Float32]([3, 4, 5, 99], [False, True, True, True])
    ).slice(0, 3)
    var source = Series._from_chunks([first^, second^])
    var scope = Scope()
    var resident = upload_float32(scope, source)
    assert_equal(resident.length, 5)
    assert_equal(resident.validity_bit_offset, 0)
    assert_equal(scope.transfers()[0], 21)
    var result = readback_float32(resident)
    assert_equal(result._get(0), Float32(1))
    assert_equal(result._get(1), Float32(2))
    assert_equal(result._get(2), Float32(3))
    assert_equal(result._get(3), Float32(4))
    assert_equal(result._get(4), Float32(5))
    assert_equal(result.is_valid(0), False)
    assert_equal(result.is_valid(1), True)
    assert_equal(result.is_valid(2), False)
    assert_equal(result.is_valid(3), True)
    assert_equal(result.is_valid(4), True)
    assert_equal(scope.transfers()[1], 21)


# requires: metal
def test_native_float32_empty_and_budget_failure_allocate_nothing() raises:
    _ = require_metal()
    var empty = Series("x", Column[Float32](List[Float32]()))
    var scope = Scope()
    var resident = upload_float32(scope, empty)
    assert_equal(resident.length, 0)
    assert_equal(resident.has_validity(), False)
    assert_equal(resident.has_errors(), False)
    assert_equal(scope.transfers()[0], 0)
    assert_equal(scope.initialized(), False)
    var empty_result = readback_float32(resident)
    assert_equal(len(empty_result), 0)
    assert_equal(scope.initialized(), False)

    var exact = Scope(budget=13)
    var nullable = Series("x", Column[Float32]([1, 2, 3], [True, False, True]))
    var exact_owner = upload_float32(exact, nullable)
    assert_equal(exact_owner.retained_bytes(), 13)
    assert_equal(exact.info()[0], 13)
    _ = exact_owner^
    assert_equal(exact.info()[0], 0)

    var too_small = Scope(budget=12)
    with assert_raises(contains="memory budget"):
        _ = upload_float32(too_small, nullable)
    assert_equal(too_small.initialized(), False)
    assert_equal(too_small.info()[0], 0)


# requires: metal
def test_native_float32_views_retain_one_owner_and_readback_checks_budget() raises:
    _ = require_metal()
    var source = Series("x", Column[Float32]([1, 2, 3, 4]))
    var scope = Scope()
    var original = upload_float32(scope, source)
    var view = original.view(1, 2)
    _ = original^
    assert_equal(scope.info()[0], 16)
    var result = readback_float32(view)
    assert_equal(result._get(0), Float32(2))
    assert_equal(result._get(1), Float32(3))
    _ = view^
    assert_equal(scope.info()[0], 0)

    var limited = Scope(budget=16)
    var retained = upload_float32(limited, source)
    with assert_raises(contains="memory budget"):
        _ = readback_float32(retained)
    assert_equal(limited.transfers()[1], 0)
    assert_equal(limited.info()[0], 16)
    _ = retained^
    assert_equal(limited.info()[0], 0)

    var all_valid = Series("x", Column[Float32]([1, 2, 3], [True, True, True]))
    var infallible = Scope()
    var native = upload_float32(infallible, all_valid)
    assert_equal(native.has_validity(), False)
    assert_equal(native.has_errors(), False)
    assert_equal(native.retained_bytes(), 12)
    assert_equal(infallible.transfers()[0], 12)
