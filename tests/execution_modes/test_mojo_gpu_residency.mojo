"""Resident Mojo matrix ownership, budgets and boundary-only readback."""

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer
from std.memory import bitcast, Pointer
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import DataFrame, DataType, Column, Series
from dataframe_metal.accelerator.gpu.residency import (
    Scope,
    Matrix,
    Table,
    HostMatrix,
    CELL_BYTES,
    checked_mul,
    raise_value_error,
)


def _increment(words: Pointer[UInt64, MutAnyOrigin], rows: UInt32):
    var row = global_idx.x
    if row < Int(rows):
        words[unsafe_offset=Int(rows) + row] += 1


def _schema() raises -> DataFrame:
    return DataFrame(
        [
            Series("x", Column[Int64](List[Int64]())),
            Series("flag", Column[Bool](List[Bool]())),
            Series("wide", Column[Int128](List[Int128]())).with_dtype(
                DataType.decimal(38, 2)
            ),
        ]
    )


def _wide(row: Int) -> Int128:
    var value = (Int128(1) << 100) + Int128(row)
    return -value if row % 2 else value


def _source(scope: Scope, fail: Bool = False) raises -> Table:
    var capacity = 5
    var slots = 4
    var cells = capacity * slots
    _ = scope.check(cells * CELL_BYTES)
    var words = List[UInt64](length=cells * 2, fill=0)
    var validity = List[UInt8](length=cells, fill=1)
    var errors = List[UInt32](length=cells, fill=0)
    for row in range(capacity):
        words[row] = bitcast[DType.uint64](Int64(row * 13 - 20))
        words[capacity + row] = UInt64(row % 2)
        var value = _wide(row)
        words[2 * capacity + row] = UInt64(value)
        words[cells + 2 * capacity + row] = UInt64(value >> 64)
    validity[3] = 0
    if fail:
        errors[2] = 1
    var ctx = scope.context()
    var device_words = ctx.enqueue_create_buffer[DType.uint64](cells * 2)
    var device_validity = ctx.enqueue_create_buffer[DType.uint8](cells)
    var device_errors = ctx.enqueue_create_buffer[DType.uint32](cells)
    ctx.enqueue_copy(device_words, words.unsafe_ptr())
    ctx.enqueue_copy(device_validity, validity.unsafe_ptr())
    ctx.enqueue_copy(device_errors, errors.unsafe_ptr())
    scope.record_upload(cells * CELL_BYTES)
    scope.synchronize()
    _ = words^
    _ = validity^
    _ = errors^
    var matrix = Matrix(
        scope, capacity, slots, device_words^, device_validity^, device_errors^
    )
    var columns: List[Int64] = [0, 1, 2]
    return Table(matrix^, _schema(), columns^, 3, 1)


# requires: mojo_gpu
def test_mojo_gpu_residency_lazy_empty_identity_and_zero_column_cardinality() raises:
    var scope = Scope(999999, 0)
    assert_true(not scope.initialized())
    assert_equal(scope.check(0), 0)
    with assert_raises():
        _ = scope.check(1)
    assert_true(not scope.initialized())
    var empty = Table.empty(_schema())
    assert_equal(empty.readback().rows, 0)
    var rows = Table.rows_only(scope, 7)
    assert_equal(rows.rows, 7)
    assert_equal(rows.readback().rows, 7)
    assert_equal(rows.readback().schema.width(), 0)
    assert_equal(rows.head(1).readback().rows, 1)
    assert_equal(rows.head(0).readback().rows, 0)
    assert_equal(scope.info()[0], 0)
    assert_true(not scope.initialized())
    var identity_scope = Scope(999999, 1000)
    var schema = DataFrame(
        [
            Series("sum", Column[Int64](List[Int64]())),
            Series("mean", Column[Float64](List[Float64]())),
        ]
    )
    var words: List[UInt64] = [0, 0, 0, 0]
    var mask: List[UInt8] = [1, 0]
    var errors: List[UInt32] = [0, 0]
    var identity = Table.identity(
        identity_scope, HostMatrix(schema^, 1, words^, mask^, errors^)
    )
    var copy = identity.copy()
    assert_true(copy.same_storage(identity))
    assert_equal(identity_scope.info()[0], 42)
    var host = copy.readback()
    assert_equal(host.rows, 1)
    assert_equal(host.schema.height(), 0)
    assert_equal(host.validity[0], UInt8(1))
    assert_equal(host.validity[1], UInt8(0))
    assert_true(not identity_scope.initialized())
    assert_equal(identity_scope.transfers()[0], 0)
    assert_equal(identity_scope.transfers()[1], 0)


def _copy_branch(scope: Scope) raises -> HostMatrix:
    var source = _source(scope)
    var alias = source.copy()
    assert_true(alias.same_storage(source))
    assert_equal(scope.info()[0], 420)
    var head = alias.head(2)
    assert_true(head.same_storage(source))
    assert_equal(head.rows, 2)
    var capacity = 3
    var slots = 3
    var cells = capacity * slots
    _ = scope.check(cells * CELL_BYTES + 3 * 3 * 16)
    var ctx = scope.context()
    var words = ctx.enqueue_create_buffer[DType.uint64](cells * 2)
    var validity = ctx.enqueue_create_buffer[DType.uint8](cells)
    var errors = ctx.enqueue_create_buffer[DType.uint32](cells)
    var mapping: List[Int64] = [2, 0, 2]
    assert_equal(
        source.copy_to(
            scope, words, validity, errors, capacity, slots, mapping
        ),
        1,
    )
    assert_equal(scope.transfers()[1], 0)
    assert_equal(scope.transfers()[0], 420 + 72)
    ctx.enqueue_function[_increment](
        words.unsafe_ptr(), UInt32(capacity), grid_dim=1, block_dim=32
    )
    scope.record_launches()
    scope.synchronize()
    var schema = DataFrame(
        [
            Series("wide", Column[Int128](List[Int128]())).with_dtype(
                DataType.decimal(38, 2)
            ),
            Series("x", Column[Int64](List[Int64]())),
            Series("alias", Column[Int128](List[Int128]())).with_dtype(
                DataType.decimal(38, 2)
            ),
        ]
    )
    var matrix = Matrix(scope, capacity, slots, words^, validity^, errors^)
    var columns: List[Int64] = [0, 1, 2]
    var output = Table(matrix^, schema^, columns^, source.rows)
    assert_equal(scope.info()[0], 609)
    var host = output.readback()
    var original = source.readback()
    for row in range(3):
        var value = _wide(row + 1)
        assert_equal(host.words[row], UInt64(value))
        assert_equal(host.words[cells + row], UInt64(value >> 64))
        assert_equal(host.words[6 + row], UInt64(value))
        assert_equal(host.words[cells + 6 + row], UInt64(value >> 64))
        assert_equal(
            host.words[3 + row],
            bitcast[DType.uint64](Int64((row + 1) * 13 - 19)),
        )
        assert_equal(host.words[cells + 3 + row], UInt64(0))
        assert_equal(original.words[row] + 1, host.words[3 + row])
        assert_equal(host.validity[3 + row], UInt8(row != 2))
    assert_equal(scope.transfers()[1], 378)
    assert_equal(scope.transfers()[2], 2)
    var wrong = Scope(999999)
    var matrix_owner = output.matrix()
    with assert_raises():
        _ = source.copy_to(
            wrong,
            matrix_owner[].words,
            matrix_owner[].validity,
            matrix_owner[].errors,
            capacity,
            slots,
            mapping,
        )
    assert_true(not wrong.initialized())
    return host^


# requires: mojo_gpu
def test_mojo_gpu_residency_shared_paired_matrix_device_copy_and_lifetime() raises:
    var scope = Scope()
    var host = _copy_branch(scope)
    assert_equal(scope.info()[0], 0)
    assert_equal(host.rows, 3)
    assert_true(scope.info()[1] >= 609)
    var info = scope.device_info()
    assert_equal(info[0], 0)
    assert_equal(info[2], "metal")


# requires: mojo_gpu
def test_mojo_gpu_residency_export_budget_and_deferred_errors() raises:
    with assert_raises(contains="decimal multiplication overflow"):
        raise_value_error(4, "generic")
    with assert_raises(contains="decimal precision"):
        raise_value_error(8, "generic")
    with assert_raises(contains="integer overflow"):
        raise_value_error(16, "generic")
    for error in [UInt32(1), UInt32(2)]:
        with assert_raises(contains="strict cast failure"):
            raise_value_error(error, "strict cast failure")
    var scope = Scope(0, 698)
    var source = _source(scope)
    with assert_raises():
        _ = source.readback()
    assert_equal(scope.transfers()[1], 0)
    var exact_scope = Scope(0, 699)
    var exact = _source(exact_scope)
    assert_equal(exact.readback().rows, 3)
    assert_equal(exact_scope.info()[1], 699)
    var failed_scope = Scope()
    var failed = _source(failed_scope, True)
    with assert_raises():
        failed.observe_errors()
    assert_equal(failed_scope.transfers()[1], 36)
    failed.head(1).observe_errors()
    assert_equal(failed_scope.transfers()[1], 48)
    with assert_raises():
        _ = failed.readback()
    assert_equal(failed.head(1).readback().rows, 1)


# requires: mojo_gpu
def test_mojo_gpu_residency_structural_identity_lazy_upload_and_target_offset() raises:
    var scope = Scope()
    var schema = DataFrame(
        [
            Series("value", Column[Int64](List[Int64]())),
            Series("wide", Column[Int128](List[Int128]())).with_dtype(
                DataType.decimal(38, 2)
            ),
        ]
    )
    var value = _wide(1)
    var words: List[UInt64] = [7, UInt64(value), 0, UInt64(value >> 64)]
    var validity: List[UInt8] = [1, 1]
    var errors: List[UInt32] = [0, 0]
    var identity = Table.identity(
        scope, HostMatrix(schema.copy(), 1, words^, validity^, errors^)
    )
    identity.observe_errors()
    assert_true(not scope.initialized())
    var ctx = scope.context()
    var device_words = ctx.enqueue_create_buffer[DType.uint64](12)
    var device_validity = ctx.enqueue_create_buffer[DType.uint8](6)
    var device_errors = ctx.enqueue_create_buffer[DType.uint32](6)
    var mapping: List[Int64] = [0, 1]
    assert_equal(
        identity.copy_to(
            scope,
            device_words,
            device_validity,
            device_errors,
            2,
            3,
            mapping,
            1,
        ),
        0,
    )
    assert_equal(scope.transfers()[0], 42)
    assert_equal(scope.transfers()[1], 0)
    var matrix = Matrix(
        scope, 2, 3, device_words^, device_validity^, device_errors^
    )
    var columns: List[Int64] = [1, 2]
    var result = Table(matrix^, schema^, columns^, 1)
    var host = result.readback()
    assert_equal(host.words[0], UInt64(7))
    assert_equal(host.words[1], UInt64(value))
    assert_equal(host.words[3], UInt64(value >> 64))
    assert_equal(host.validity[1], UInt8(1))
    var pending = ctx.enqueue_create_buffer[DType.uint64](2)
    var waits_before = scope.transfers()[3]
    with assert_raises(contains="injected after asynchronous"):
        _injected_submission_error(scope, pending)
    assert_equal(scope.transfers()[3], waits_before + 1)
    var control = scope.read_control(pending)
    assert_equal(scope.transfers()[3], waits_before + 2)
    assert_equal(control[0], UInt64(0x123456789ABCDEF0))
    assert_equal(control[1], UInt64.MAX)


def _injected_submission_error(
    scope: Scope, buffer: DeviceBuffer[DType.uint64]
) raises:
    var source: List[UInt64] = [0x123456789ABCDEF0, UInt64.MAX]
    var ctx = scope.context()
    try:
        ctx.enqueue_copy(buffer, source.unsafe_ptr())
        raise Error("injected after asynchronous host transfer")
    except error:
        scope.synchronize()
        _ = source^
        raise error^
