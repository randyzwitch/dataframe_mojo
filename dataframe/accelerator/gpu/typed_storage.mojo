"""Typed, shared ownership for native-width device columns.

Values keep their physical dtype. Optional validity is a packed bitmap whose
bit offset is independent of the values view; optional errors contain one
UInt32 code per retained row. This module depends only on query_scope, so it
can be used by kernels and materializers without a residency cycle.
"""
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer, Pointer
from std.sys import size_of
from std.utils import Variant
from dataframe import Column, DataType, Series
from dataframe.column import _append_bits, _copy_bits
from dataframe.accelerator.faults import data_fault
from .query_scope import Scope, checked_add, checked_mul


def native_column_bytes[
    dtype: DType
](
    value_capacity: Int,
    validity_bytes: Int = 0,
    error_capacity: Int = 0,
) raises -> Int:
    """Return bytes retained by values and present sidecars, with overflow checks."""
    if value_capacity < 0 or validity_bytes < 0 or error_capacity < 0:
        raise Error("Mojo GPU typed column capacity is invalid")
    var total = checked_mul(value_capacity, size_of[Scalar[dtype]]())
    total = checked_add(total, validity_bytes)
    total = checked_add(total, checked_mul(error_capacity, size_of[UInt32]()))
    return total


def check_native_column_capacity[
    dtype: DType
](
    scope: Scope,
    value_capacity: Int,
    validity_bytes: Int = 0,
    error_capacity: Int = 0,
) raises -> Int:
    """Check the complete owner footprint before any device allocation."""
    return scope.check(
        native_column_bytes[dtype](
            value_capacity, validity_bytes, error_capacity
        )
    )


struct _NativeReservation(Movable):
    """Reserve query bytes before allocation and release on every exit."""

    var scope: Scope
    var bytes: Int

    def __init__(out self, scope: Scope, bytes: Int) raises:
        scope._retain(bytes)
        self.scope = scope.copy()
        self.bytes = bytes

    def __deinit__(deinit self):
        self.scope._release(self.bytes)


struct _NativeColumnOwner[dtype: DType](Movable):
    var scope: Scope
    var retained: Int
    var values: Optional[DeviceBuffer[Self.dtype]]
    var validity: Optional[DeviceBuffer[DType.uint8]]
    var errors: Optional[DeviceBuffer[DType.uint32]]
    var reservation: _NativeReservation

    def __init__(
        out self,
        scope: Scope,
        var values: Optional[DeviceBuffer[Self.dtype]] = None,
        var validity: Optional[DeviceBuffer[DType.uint8]] = None,
        var errors: Optional[DeviceBuffer[DType.uint32]] = None,
        var reservation: Optional[_NativeReservation] = None,
    ) raises:
        var validity_bytes = len(validity.value()) if validity else 0
        var error_capacity = len(errors.value()) if errors else 0
        var value_capacity = len(values.value()) if values else 0
        var retained = native_column_bytes[Self.dtype](
            value_capacity, validity_bytes, error_capacity
        )
        if reservation:
            if reservation.value().bytes != retained or not scope.same_query(
                reservation.value().scope
            ):
                raise Error("Mojo GPU typed column reservation mismatch")
            self.reservation = reservation.take()
        else:
            self.reservation = _NativeReservation(scope, retained)
        self.scope = scope.copy()
        self.retained = retained
        self.values = values^
        self.validity = validity^
        self.errors = errors^

    def __deinit__(deinit self):
        _ = self.values^
        _ = self.validity^
        _ = self.errors^
        _ = self.reservation^


def _native_logical_dtype_matches[dtype: DType](logical: DataType) -> Bool:
    comptime if dtype == DType.bool:
        return logical == DataType.BOOL
    else:
        return logical.physical() == DataType.of(dtype)


struct NativeColumn[dtype: DType](Copyable):
    """Immutable row view over an Arc-owned typed device allocation."""

    var _owner: ArcPointer[_NativeColumnOwner[Self.dtype]]
    var logical: DataType
    var length: Int
    var value_offset: Int
    var validity_bit_offset: Int

    def __init__(
        out self,
        scope: Scope,
        logical: DataType,
        length: Int,
        var values: DeviceBuffer[Self.dtype],
        var validity: Optional[DeviceBuffer[DType.uint8]] = None,
        var errors: Optional[DeviceBuffer[DType.uint32]] = None,
        value_offset: Int = 0,
        validity_bit_offset: Int = 0,
        var reservation: Optional[_NativeReservation] = None,
    ) raises:
        if not _native_logical_dtype_matches[Self.dtype](logical):
            raise Error(
                "Mojo GPU typed column logical and physical dtypes differ"
            )
        if (
            length < 0
            or value_offset < 0
            or validity_bit_offset < 0
            or value_offset > len(values)
            or length > len(values) - value_offset
        ):
            raise Error("Mojo GPU typed column view is outside its values")
        if validity:
            var bits = checked_add(validity_bit_offset, length)
            var needed_bytes = checked_add(bits, 7) // 8
            if needed_bytes > len(validity.value()):
                raise Error(
                    "Mojo GPU typed column view is outside its validity bitmap"
                )
        elif validity_bit_offset != 0:
            raise Error("Mojo GPU typed column has an absent validity bitmap")
        if errors and (
            value_offset > len(errors.value())
            or length > len(errors.value()) - value_offset
        ):
            raise Error("Mojo GPU typed column view is outside its errors")
        self._owner = ArcPointer(
            _NativeColumnOwner[Self.dtype](
                scope,
                Optional[DeviceBuffer[Self.dtype]](values^),
                validity^,
                errors^,
                reservation^,
            )
        )
        self.logical = logical
        self.length = length
        self.value_offset = value_offset
        self.validity_bit_offset = validity_bit_offset

    def __init__(out self, scope: Scope, logical: DataType, length: Int) raises:
        if not _native_logical_dtype_matches[Self.dtype](logical):
            raise Error(
                "Mojo GPU typed column logical and physical dtypes differ"
            )
        if length != 0:
            raise Error("Mojo GPU typed column without values must be empty")
        self._owner = ArcPointer(
            _NativeColumnOwner[Self.dtype](scope, None, None, None)
        )
        self.logical = logical
        self.length = 0
        self.value_offset = 0
        self.validity_bit_offset = 0

    def __init__(
        out self,
        owner: ArcPointer[_NativeColumnOwner[Self.dtype]],
        logical: DataType,
        length: Int,
        value_offset: Int,
        validity_bit_offset: Int,
    ) raises:
        if not _native_logical_dtype_matches[Self.dtype](logical):
            raise Error(
                "Mojo GPU typed column logical and physical dtypes differ"
            )
        if (
            length < 0
            or value_offset < 0
            or validity_bit_offset < 0
            or (
                owner[].values
                and (
                    value_offset > len(owner[].values.value())
                    or length > len(owner[].values.value()) - value_offset
                )
            )
            or (not owner[].values and (length != 0 or value_offset != 0))
        ):
            raise Error("Mojo GPU typed column view is outside its values")
        if owner[].validity:
            var bits = checked_add(validity_bit_offset, length)
            var needed_bytes = checked_add(bits, 7) // 8
            if needed_bytes > len(owner[].validity.value()):
                raise Error(
                    "Mojo GPU typed column view is outside its validity bitmap"
                )
        elif validity_bit_offset != 0:
            raise Error("Mojo GPU typed column has an absent validity bitmap")
        if owner[].errors and (
            value_offset > len(owner[].errors.value())
            or length > len(owner[].errors.value()) - value_offset
        ):
            raise Error("Mojo GPU typed column view is outside its errors")
        self._owner = owner.copy()
        self.logical = logical
        self.length = length
        self.value_offset = value_offset
        self.validity_bit_offset = validity_bit_offset

    def view(self, offset: Int, length: Int) raises -> Self:
        if offset < 0 or length < 0 or offset > self.length:
            raise Error("Mojo GPU typed column slice is invalid")
        if length > self.length - offset:
            raise Error("Mojo GPU typed column slice is outside its view")
        var next_bit_offset = (
            self.validity_bit_offset + offset if self._owner[].validity else 0
        )
        return Self(
            self._owner.copy(),
            self.logical,
            length,
            self.value_offset + offset,
            next_bit_offset,
        )

    def values(self) raises -> DeviceBuffer[Self.dtype]:
        """Copy the device handle; keep this NativeColumn alive while using it."""
        if not self._owner[].values:
            raise Error("Empty Mojo GPU typed column has no values buffer")
        return self._owner[].values.value().copy()

    def validity(self) -> Optional[DeviceBuffer[DType.uint8]]:
        if self._owner[].validity:
            return self._owner[].validity.value().copy()
        return None

    def errors(self) -> Optional[DeviceBuffer[DType.uint32]]:
        if self._owner[].errors:
            return self._owner[].errors.value().copy()
        return None

    def scope(self) -> Scope:
        return self._owner[].scope.copy()

    def has_validity(self) -> Bool:
        return Bool(self._owner[].validity)

    def has_errors(self) -> Bool:
        return Bool(self._owner[].errors)

    def read_error_codes(self) raises -> List[UInt32]:
        """Observe the retained error plane before publishing values.

        An absent plane explicitly denotes an infallible owner and needs no
        device read. Present planes are synchronized through the query scope.
        """
        if not self._owner[].errors:
            return List[UInt32]()
        var errors = self._owner[].errors.value().copy()
        _ = self._owner[].scope.check(
            checked_add(
                checked_mul(len(errors), size_of[UInt32]()),
                checked_mul(self.length, size_of[UInt32]()),
            )
        )
        self._owner[].scope.record_download(
            checked_mul(len(errors), size_of[UInt32]())
        )
        var codes = self._owner[].scope.read_control(errors)
        var observed = List[UInt32](capacity=self.length)
        for row in range(self.length):
            observed.append(codes[self.value_offset + row])
        return observed^

    def retained_bytes(self) -> Int:
        return self._owner[].retained


# A table cannot put `NativeColumn[dtype]` in one List for a runtime dtype.
# Keep each allocation in the existing generic owner and erase only the
# compile-time dtype at the resident-table boundary. This mirrors Series.Storage:
# one active Variant arm owns one column; the inactive arms own nothing.
comptime NativeStorage = Variant[
    NativeColumn[DType.int64],
    NativeColumn[DType.float64],
    NativeColumn[DType.bool],
    NativeColumn[DType.int8],
    NativeColumn[DType.int16],
    NativeColumn[DType.int32],
    NativeColumn[DType.uint8],
    NativeColumn[DType.uint16],
    NativeColumn[DType.uint32],
    NativeColumn[DType.uint64],
    NativeColumn[DType.float32],
]


struct NativeColumnAny(Copyable):
    """Runtime-tagged view that retains one typed native allocation."""

    var _storage: NativeStorage
    var logical: DataType
    var length: Int
    var value_offset: Int
    var validity_bit_offset: Int

    def __init__[dtype: DType](out self, column: NativeColumn[dtype]) raises:
        var logical = column.logical
        if (
            logical.is_decimal()
            or logical.is_categorical()
            or logical.is_nested()
            or not (
                logical.is_numeric()
                or logical == DataType.BOOL
                or logical.is_temporal()
            )
            or not _native_logical_dtype_matches[dtype](logical)
        ):
            raise Error(
                "Native table column logical or physical dtype is unsupported"
            )
        self._storage = NativeStorage(column.copy())
        self.logical = logical
        self.length = column.length
        self.value_offset = column.value_offset
        self.validity_bit_offset = column.validity_bit_offset

    def view(self, offset: Int, length: Int) raises -> Self:
        if offset < 0 or length < 0 or offset > self.length:
            raise Error("Mojo GPU typed column slice is invalid")
        if length > self.length - offset:
            raise Error("Mojo GPU typed column slice is outside its view")
        if self._storage.isa[NativeColumn[DType.int64]]():
            return Self(
                self._storage[NativeColumn[DType.int64]].view(offset, length)
            )
        if self._storage.isa[NativeColumn[DType.float64]]():
            return Self(
                self._storage[NativeColumn[DType.float64]].view(offset, length)
            )
        if self._storage.isa[NativeColumn[DType.bool]]():
            return Self(
                self._storage[NativeColumn[DType.bool]].view(offset, length)
            )
        if self._storage.isa[NativeColumn[DType.int8]]():
            return Self(
                self._storage[NativeColumn[DType.int8]].view(offset, length)
            )
        if self._storage.isa[NativeColumn[DType.int16]]():
            return Self(
                self._storage[NativeColumn[DType.int16]].view(offset, length)
            )
        if self._storage.isa[NativeColumn[DType.int32]]():
            return Self(
                self._storage[NativeColumn[DType.int32]].view(offset, length)
            )
        if self._storage.isa[NativeColumn[DType.uint8]]():
            return Self(
                self._storage[NativeColumn[DType.uint8]].view(offset, length)
            )
        if self._storage.isa[NativeColumn[DType.uint16]]():
            return Self(
                self._storage[NativeColumn[DType.uint16]].view(offset, length)
            )
        if self._storage.isa[NativeColumn[DType.uint32]]():
            return Self(
                self._storage[NativeColumn[DType.uint32]].view(offset, length)
            )
        if self._storage.isa[NativeColumn[DType.uint64]]():
            return Self(
                self._storage[NativeColumn[DType.uint64]].view(offset, length)
            )
        return Self(
            self._storage[NativeColumn[DType.float32]].view(offset, length)
        )

    def scope(self) -> Scope:
        if self._storage.isa[NativeColumn[DType.int64]]():
            return self._storage[NativeColumn[DType.int64]].scope()
        if self._storage.isa[NativeColumn[DType.float64]]():
            return self._storage[NativeColumn[DType.float64]].scope()
        if self._storage.isa[NativeColumn[DType.bool]]():
            return self._storage[NativeColumn[DType.bool]].scope()
        if self._storage.isa[NativeColumn[DType.int8]]():
            return self._storage[NativeColumn[DType.int8]].scope()
        if self._storage.isa[NativeColumn[DType.int16]]():
            return self._storage[NativeColumn[DType.int16]].scope()
        if self._storage.isa[NativeColumn[DType.int32]]():
            return self._storage[NativeColumn[DType.int32]].scope()
        if self._storage.isa[NativeColumn[DType.uint8]]():
            return self._storage[NativeColumn[DType.uint8]].scope()
        if self._storage.isa[NativeColumn[DType.uint16]]():
            return self._storage[NativeColumn[DType.uint16]].scope()
        if self._storage.isa[NativeColumn[DType.uint32]]():
            return self._storage[NativeColumn[DType.uint32]].scope()
        if self._storage.isa[NativeColumn[DType.uint64]]():
            return self._storage[NativeColumn[DType.uint64]].scope()
        return self._storage[NativeColumn[DType.float32]].scope()

    def has_validity(self) -> Bool:
        if self._storage.isa[NativeColumn[DType.int64]]():
            return self._storage[NativeColumn[DType.int64]].has_validity()
        if self._storage.isa[NativeColumn[DType.float64]]():
            return self._storage[NativeColumn[DType.float64]].has_validity()
        if self._storage.isa[NativeColumn[DType.bool]]():
            return self._storage[NativeColumn[DType.bool]].has_validity()
        if self._storage.isa[NativeColumn[DType.int8]]():
            return self._storage[NativeColumn[DType.int8]].has_validity()
        if self._storage.isa[NativeColumn[DType.int16]]():
            return self._storage[NativeColumn[DType.int16]].has_validity()
        if self._storage.isa[NativeColumn[DType.int32]]():
            return self._storage[NativeColumn[DType.int32]].has_validity()
        if self._storage.isa[NativeColumn[DType.uint8]]():
            return self._storage[NativeColumn[DType.uint8]].has_validity()
        if self._storage.isa[NativeColumn[DType.uint16]]():
            return self._storage[NativeColumn[DType.uint16]].has_validity()
        if self._storage.isa[NativeColumn[DType.uint32]]():
            return self._storage[NativeColumn[DType.uint32]].has_validity()
        if self._storage.isa[NativeColumn[DType.uint64]]():
            return self._storage[NativeColumn[DType.uint64]].has_validity()
        return self._storage[NativeColumn[DType.float32]].has_validity()

    def has_errors(self) -> Bool:
        if self._storage.isa[NativeColumn[DType.int64]]():
            return self._storage[NativeColumn[DType.int64]].has_errors()
        if self._storage.isa[NativeColumn[DType.float64]]():
            return self._storage[NativeColumn[DType.float64]].has_errors()
        if self._storage.isa[NativeColumn[DType.bool]]():
            return self._storage[NativeColumn[DType.bool]].has_errors()
        if self._storage.isa[NativeColumn[DType.int8]]():
            return self._storage[NativeColumn[DType.int8]].has_errors()
        if self._storage.isa[NativeColumn[DType.int16]]():
            return self._storage[NativeColumn[DType.int16]].has_errors()
        if self._storage.isa[NativeColumn[DType.int32]]():
            return self._storage[NativeColumn[DType.int32]].has_errors()
        if self._storage.isa[NativeColumn[DType.uint8]]():
            return self._storage[NativeColumn[DType.uint8]].has_errors()
        if self._storage.isa[NativeColumn[DType.uint16]]():
            return self._storage[NativeColumn[DType.uint16]].has_errors()
        if self._storage.isa[NativeColumn[DType.uint32]]():
            return self._storage[NativeColumn[DType.uint32]].has_errors()
        if self._storage.isa[NativeColumn[DType.uint64]]():
            return self._storage[NativeColumn[DType.uint64]].has_errors()
        return self._storage[NativeColumn[DType.float32]].has_errors()

    def read_error_codes(self) raises -> List[UInt32]:
        if self._storage.isa[NativeColumn[DType.int64]]():
            return self._storage[NativeColumn[DType.int64]].read_error_codes()
        if self._storage.isa[NativeColumn[DType.float64]]():
            return self._storage[NativeColumn[DType.float64]].read_error_codes()
        if self._storage.isa[NativeColumn[DType.bool]]():
            return self._storage[NativeColumn[DType.bool]].read_error_codes()
        if self._storage.isa[NativeColumn[DType.int8]]():
            return self._storage[NativeColumn[DType.int8]].read_error_codes()
        if self._storage.isa[NativeColumn[DType.int16]]():
            return self._storage[NativeColumn[DType.int16]].read_error_codes()
        if self._storage.isa[NativeColumn[DType.int32]]():
            return self._storage[NativeColumn[DType.int32]].read_error_codes()
        if self._storage.isa[NativeColumn[DType.uint8]]():
            return self._storage[NativeColumn[DType.uint8]].read_error_codes()
        if self._storage.isa[NativeColumn[DType.uint16]]():
            return self._storage[NativeColumn[DType.uint16]].read_error_codes()
        if self._storage.isa[NativeColumn[DType.uint32]]():
            return self._storage[NativeColumn[DType.uint32]].read_error_codes()
        if self._storage.isa[NativeColumn[DType.uint64]]():
            return self._storage[NativeColumn[DType.uint64]].read_error_codes()
        return self._storage[NativeColumn[DType.float32]].read_error_codes()

    def retained_bytes(self) -> Int:
        if self._storage.isa[NativeColumn[DType.int64]]():
            return self._storage[NativeColumn[DType.int64]].retained_bytes()
        if self._storage.isa[NativeColumn[DType.float64]]():
            return self._storage[NativeColumn[DType.float64]].retained_bytes()
        if self._storage.isa[NativeColumn[DType.bool]]():
            return self._storage[NativeColumn[DType.bool]].retained_bytes()
        if self._storage.isa[NativeColumn[DType.int8]]():
            return self._storage[NativeColumn[DType.int8]].retained_bytes()
        if self._storage.isa[NativeColumn[DType.int16]]():
            return self._storage[NativeColumn[DType.int16]].retained_bytes()
        if self._storage.isa[NativeColumn[DType.int32]]():
            return self._storage[NativeColumn[DType.int32]].retained_bytes()
        if self._storage.isa[NativeColumn[DType.uint8]]():
            return self._storage[NativeColumn[DType.uint8]].retained_bytes()
        if self._storage.isa[NativeColumn[DType.uint16]]():
            return self._storage[NativeColumn[DType.uint16]].retained_bytes()
        if self._storage.isa[NativeColumn[DType.uint32]]():
            return self._storage[NativeColumn[DType.uint32]].retained_bytes()
        if self._storage.isa[NativeColumn[DType.uint64]]():
            return self._storage[NativeColumn[DType.uint64]].retained_bytes()
        return self._storage[NativeColumn[DType.float32]].retained_bytes()

    def readback_bytes(self) raises -> Int:
        if self._storage.isa[NativeColumn[DType.int64]]():
            return _native_readback_bytes[DType.int64](
                self._storage[NativeColumn[DType.int64]]
            )
        if self._storage.isa[NativeColumn[DType.float64]]():
            return _native_readback_bytes[DType.float64](
                self._storage[NativeColumn[DType.float64]]
            )
        if self._storage.isa[NativeColumn[DType.bool]]():
            return _native_readback_bytes[DType.bool](
                self._storage[NativeColumn[DType.bool]]
            )
        if self._storage.isa[NativeColumn[DType.int8]]():
            return _native_readback_bytes[DType.int8](
                self._storage[NativeColumn[DType.int8]]
            )
        if self._storage.isa[NativeColumn[DType.int16]]():
            return _native_readback_bytes[DType.int16](
                self._storage[NativeColumn[DType.int16]]
            )
        if self._storage.isa[NativeColumn[DType.int32]]():
            return _native_readback_bytes[DType.int32](
                self._storage[NativeColumn[DType.int32]]
            )
        if self._storage.isa[NativeColumn[DType.uint8]]():
            return _native_readback_bytes[DType.uint8](
                self._storage[NativeColumn[DType.uint8]]
            )
        if self._storage.isa[NativeColumn[DType.uint16]]():
            return _native_readback_bytes[DType.uint16](
                self._storage[NativeColumn[DType.uint16]]
            )
        if self._storage.isa[NativeColumn[DType.uint32]]():
            return _native_readback_bytes[DType.uint32](
                self._storage[NativeColumn[DType.uint32]]
            )
        if self._storage.isa[NativeColumn[DType.uint64]]():
            return _native_readback_bytes[DType.uint64](
                self._storage[NativeColumn[DType.uint64]]
            )
        return _native_readback_bytes[DType.float32](
            self._storage[NativeColumn[DType.float32]]
        )

    def gather_bytes(self, rows: Int, allow_missing: Bool) raises -> Int:
        var nullable = self.has_validity() or allow_missing
        var error_rows = rows if self.has_errors() else 0
        if self._storage.isa[NativeColumn[DType.int64]]():
            return _native_gather_bytes[DType.int64](rows, nullable, error_rows)
        if self._storage.isa[NativeColumn[DType.float64]]():
            return _native_gather_bytes[DType.float64](
                rows, nullable, error_rows
            )
        if self._storage.isa[NativeColumn[DType.bool]]():
            return _native_gather_bytes[DType.bool](rows, nullable, error_rows)
        if self._storage.isa[NativeColumn[DType.int8]]():
            return _native_gather_bytes[DType.int8](rows, nullable, error_rows)
        if self._storage.isa[NativeColumn[DType.int16]]():
            return _native_gather_bytes[DType.int16](rows, nullable, error_rows)
        if self._storage.isa[NativeColumn[DType.int32]]():
            return _native_gather_bytes[DType.int32](rows, nullable, error_rows)
        if self._storage.isa[NativeColumn[DType.uint8]]():
            return _native_gather_bytes[DType.uint8](rows, nullable, error_rows)
        if self._storage.isa[NativeColumn[DType.uint16]]():
            return _native_gather_bytes[DType.uint16](
                rows, nullable, error_rows
            )
        if self._storage.isa[NativeColumn[DType.uint32]]():
            return _native_gather_bytes[DType.uint32](
                rows, nullable, error_rows
            )
        if self._storage.isa[NativeColumn[DType.uint64]]():
            return _native_gather_bytes[DType.uint64](
                rows, nullable, error_rows
            )
        return _native_gather_bytes[DType.float32](rows, nullable, error_rows)

    def owner_identity(self) -> Int:
        if self._storage.isa[NativeColumn[DType.int64]]():
            return Int(
                Pointer(to=self._storage[NativeColumn[DType.int64]]._owner[])
            )
        if self._storage.isa[NativeColumn[DType.float64]]():
            return Int(
                Pointer(to=self._storage[NativeColumn[DType.float64]]._owner[])
            )
        if self._storage.isa[NativeColumn[DType.bool]]():
            return Int(
                Pointer(to=self._storage[NativeColumn[DType.bool]]._owner[])
            )
        if self._storage.isa[NativeColumn[DType.int8]]():
            return Int(
                Pointer(to=self._storage[NativeColumn[DType.int8]]._owner[])
            )
        if self._storage.isa[NativeColumn[DType.int16]]():
            return Int(
                Pointer(to=self._storage[NativeColumn[DType.int16]]._owner[])
            )
        if self._storage.isa[NativeColumn[DType.int32]]():
            return Int(
                Pointer(to=self._storage[NativeColumn[DType.int32]]._owner[])
            )
        if self._storage.isa[NativeColumn[DType.uint8]]():
            return Int(
                Pointer(to=self._storage[NativeColumn[DType.uint8]]._owner[])
            )
        if self._storage.isa[NativeColumn[DType.uint16]]():
            return Int(
                Pointer(to=self._storage[NativeColumn[DType.uint16]]._owner[])
            )
        if self._storage.isa[NativeColumn[DType.uint32]]():
            return Int(
                Pointer(to=self._storage[NativeColumn[DType.uint32]]._owner[])
            )
        if self._storage.isa[NativeColumn[DType.uint64]]():
            return Int(
                Pointer(to=self._storage[NativeColumn[DType.uint64]]._owner[])
            )
        return Int(
            Pointer(to=self._storage[NativeColumn[DType.float32]]._owner[])
        )

    def readback_series(self, field: Series) raises -> Series:
        """Materialize this typed view with the schema's logical metadata."""
        if field.dtype() != self.logical:
            raise Error("Native readback schema dtype differs from its owner")
        # Bool values are bit-packed on the host. Until the bitmap-to-native
        # upload and native-to-bitmap readback kernels are present, refuse the
        # transport explicitly instead of expanding or repacking row by row.
        if self.logical == DataType.BOOL:
            raise Error("Native Bool transport is not implemented")
        var result: Series
        if self._storage.isa[NativeColumn[DType.int64]]():
            result = _readback_series[DType.int64](
                field, self._storage[NativeColumn[DType.int64]]
            )
        elif self._storage.isa[NativeColumn[DType.float64]]():
            result = _readback_series[DType.float64](
                field, self._storage[NativeColumn[DType.float64]]
            )
        elif self._storage.isa[NativeColumn[DType.int8]]():
            result = _readback_series[DType.int8](
                field, self._storage[NativeColumn[DType.int8]]
            )
        elif self._storage.isa[NativeColumn[DType.int16]]():
            result = _readback_series[DType.int16](
                field, self._storage[NativeColumn[DType.int16]]
            )
        elif self._storage.isa[NativeColumn[DType.int32]]():
            result = _readback_series[DType.int32](
                field, self._storage[NativeColumn[DType.int32]]
            )
        elif self._storage.isa[NativeColumn[DType.uint8]]():
            result = _readback_series[DType.uint8](
                field, self._storage[NativeColumn[DType.uint8]]
            )
        elif self._storage.isa[NativeColumn[DType.uint16]]():
            result = _readback_series[DType.uint16](
                field, self._storage[NativeColumn[DType.uint16]]
            )
        elif self._storage.isa[NativeColumn[DType.uint32]]():
            result = _readback_series[DType.uint32](
                field, self._storage[NativeColumn[DType.uint32]]
            )
        elif self._storage.isa[NativeColumn[DType.uint64]]():
            result = _readback_series[DType.uint64](
                field, self._storage[NativeColumn[DType.uint64]]
            )
        else:
            result = _readback_series[DType.float32](
                field, self._storage[NativeColumn[DType.float32]]
            )
        return result^


def allocate_native_column[
    dtype: DType
](
    scope: Scope,
    logical: DataType,
    length: Int,
    *,
    with_validity: Bool = False,
    with_errors: Bool = False,
) raises -> NativeColumn[dtype]:
    """Reserve a typed output owner after preflighting its whole footprint."""
    if length < 0 or length > Int(Int32.MAX):
        raise Error("Mojo GPU typed column length is invalid")
    if length == 0:
        return NativeColumn[dtype](scope, logical, 0)
    var validity_bytes = (
        checked_add(length, 7) // 8 if with_validity and length else 0
    )
    var error_capacity = length if with_errors and length else 0
    _ = check_native_column_capacity[dtype](
        scope, length, validity_bytes, error_capacity
    )
    var reservation = _NativeReservation(
        scope,
        native_column_bytes[dtype](length, validity_bytes, error_capacity),
    )
    var ctx = scope.context()
    var values = ctx.enqueue_create_buffer[dtype](length)
    var validity = Optional[DeviceBuffer[DType.uint8]]()
    var errors = Optional[DeviceBuffer[DType.uint32]]()
    if validity_bytes:
        validity = Optional[DeviceBuffer[DType.uint8]](
            ctx.enqueue_create_buffer[DType.uint8](validity_bytes)
        )
    if error_capacity:
        errors = Optional[DeviceBuffer[DType.uint32]](
            ctx.enqueue_create_buffer[DType.uint32](error_capacity)
        )
    return NativeColumn[dtype](
        scope,
        logical,
        length,
        values^,
        validity^,
        errors^,
        reservation=Optional[_NativeReservation](reservation^),
    )


def upload_native[
    dtype: DType
](scope: Scope, series: Series) raises -> NativeColumn[dtype]:
    """Upload supported primitive storage without instantiating packed Bool access."""
    comptime if dtype == DType.bool:
        raise Error("Native Bool upload is not implemented")
    else:
        return _upload_native_unpacked[dtype](scope, series)


def _upload_native_unpacked[
    dtype: DType
](scope: Scope, series: Series) raises -> NativeColumn[dtype]:
    """Bulk-upload a native-width primitive Series, preserving slices and packed validity.

    Values always move at their physical width. A single nullable chunk retains the
    source bitmap's bit offset; multiple chunks repack only their bitmap to
    concatenate row ranges. Source chunks remain alive through synchronization.
    """
    var logical = series.dtype()
    if (
        logical.is_decimal()
        or logical.is_categorical()
        or logical.is_nested()
        or not (
            logical.is_numeric()
            or logical == DataType.BOOL
            or logical.is_temporal()
        )
        or not _native_logical_dtype_matches[dtype](logical)
    ):
        raise Error("Native upload requires a supported fixed-width dtype")
    if logical == DataType.BOOL:
        raise Error("Native Bool upload is not implemented")
    var chunks = series.chunks()
    var rows = 0
    var has_validity = False
    for part in chunks:
        if not part._data.isa[Column[Scalar[dtype]]]():
            raise Error("Native upload requires the exact physical dtype")
        ref column = part._data[Column[Scalar[dtype]]]
        rows = checked_add(rows, len(column))
        if column.null_count() > 0:
            has_validity = True
    if rows != len(series):
        raise Error("Native upload chunk lengths differ from Series")

    if rows == 0:
        return NativeColumn[dtype](scope, logical, 0)

    var source_bit_offset = 0
    var validity_bytes = 0
    var packed_bits = List[UInt8]()
    if has_validity:
        if len(chunks) == 1:
            ref column = chunks[0]._data[Column[Scalar[dtype]]]
            source_bit_offset = column.validity_offset() % 8
            validity_bytes = (source_bit_offset + rows + 7) // 8
        else:
            validity_bytes = checked_add(rows, 7) // 8

    # A multi-chunk nullable upload temporarily holds the packed host bitmap
    # alongside the device values and bitmap. Preflight all of it before
    # allocating that host bitmap or any device storage.
    var preflight_bytes = native_column_bytes[dtype](rows, validity_bytes, 0)
    if len(chunks) > 1 and has_validity:
        preflight_bytes = checked_add(preflight_bytes, validity_bytes)
    _ = scope.check(preflight_bytes)
    var reservation = _NativeReservation(
        scope, native_column_bytes[dtype](rows, validity_bytes, 0)
    )
    var bitmap_reservation = Optional[_NativeReservation]()
    if len(chunks) > 1 and has_validity:
        bitmap_reservation = _NativeReservation(scope, validity_bytes)

    if has_validity and len(chunks) > 1:
        packed_bits = List[UInt8](capacity=validity_bytes)
        var appended = 0
        for part in chunks:
            ref column = part._data[Column[Scalar[dtype]]]
            ref source_bits = column._bits[]
            if len(source_bits) == 0:
                var needed = checked_add(appended + len(column), 7) // 8
                packed_bits.resize(needed, 0)
                for row in range(len(column)):
                    var bit = appended + row
                    packed_bits[bit // 8] |= UInt8(1) << UInt8(bit % 8)
            else:
                _append_bits(
                    packed_bits,
                    appended,
                    source_bits,
                    column.validity_offset(),
                    len(column),
                )
            appended += len(column)

    var ctx = scope.context()
    var values = ctx.enqueue_create_buffer[dtype](rows)
    var validity = Optional[DeviceBuffer[DType.uint8]]()
    if validity_bytes:
        validity = Optional[DeviceBuffer[DType.uint8]](
            ctx.enqueue_create_buffer[DType.uint8](validity_bytes)
        )

    var uploaded = 0
    try:
        var target = 0
        for part in chunks:
            ref column = part._data[Column[Scalar[dtype]]]
            var count = len(column)
            if count:
                var destination = values.create_sub_buffer[dtype](target, count)
                ctx.enqueue_copy(destination, column.unsafe_values())
                uploaded = checked_add(
                    uploaded, checked_mul(count, size_of[Scalar[dtype]]())
                )
            target += count
        if validity_bytes:
            if len(chunks) == 1:
                ref column = chunks[0]._data[Column[Scalar[dtype]]]
                var byte_offset = column.validity_offset() // 8
                ctx.enqueue_copy(
                    validity.value(),
                    column.unsafe_validity().unsafe_offset(byte_offset),
                )
            else:
                ctx.enqueue_copy(validity.value(), packed_bits.unsafe_ptr())
            uploaded = checked_add(uploaded, validity_bytes)
        if uploaded:
            scope.record_upload(uploaded)
            scope.synchronize()
    except error:
        scope.synchronize()
        _ = chunks^
        _ = packed_bits^
        _ = bitmap_reservation^
        raise error^
    _ = chunks^
    _ = packed_bits^
    _ = bitmap_reservation^

    return NativeColumn[dtype](
        scope,
        logical,
        rows,
        values^,
        validity^,
        None,
        0,
        source_bit_offset,
        Optional[_NativeReservation](reservation^),
    )


def upload_float32(
    scope: Scope, series: Series
) raises -> NativeColumn[DType.float32]:
    return upload_native[DType.float32](scope, series)


def readback_float32(
    column: NativeColumn[DType.float32],
) raises -> Column[Float32]:
    """Compatibility helper for callers with a statically Float32 column."""
    return readback_native[DType.float32](column)


def _native_readback_bytes[
    dtype: DType
](column: NativeColumn[dtype]) raises -> Int:
    var bytes = checked_mul(column.length, size_of[Scalar[dtype]]())
    var validity = column.validity()
    if validity:
        bytes = checked_add(bytes, len(validity.value()))
        var packed_bytes = checked_add(column.length, 7) // 8
        if (
            column.validity_bit_offset != 0
            or len(validity.value()) != packed_bytes
        ):
            bytes = checked_add(bytes, packed_bytes)
    var errors = column.errors()
    if errors:
        bytes = checked_add(
            bytes, checked_mul(len(errors.value()), size_of[UInt32]())
        )
        bytes = checked_add(
            bytes, checked_mul(column.length, size_of[UInt32]())
        )
    return bytes


def _native_gather_bytes[
    dtype: DType
](rows: Int, nullable: Bool, error_rows: Int) raises -> Int:
    return native_column_bytes[dtype](
        rows,
        checked_add(rows, 7) // 8 if nullable else 0,
        error_rows,
    )


def readback_native[
    dtype: DType
](column: NativeColumn[dtype]) raises -> Column[Scalar[dtype]]:
    """Bulk-read a native view and adopt values plus a compact validity bitmap."""
    var scope = column.scope()
    var validity = column.validity()
    var raw_validity_bytes = len(validity.value()) if validity else 0
    var packed_validity_bytes = checked_add(column.length, 7) // 8
    var needs_repacked_validity = validity and (
        column.validity_bit_offset != 0
        or raw_validity_bytes != packed_validity_bytes
    )
    var host_bytes = _native_readback_bytes[dtype](column)
    # Includes all host readback/repack temporaries before any of them exist.
    _ = scope.check(host_bytes)

    var error_codes = column.read_error_codes()
    for code in error_codes:
        if code:
            raise data_fault("Mojo GPU typed result reported a deferred error")

    var values = List[Scalar[dtype]](length=column.length, fill=0)
    var validity_bytes = List[UInt8]()
    if validity:
        validity_bytes = List[UInt8](length=raw_validity_bytes, fill=0)
    var transferred = 0
    if column.length or len(validity_bytes):
        var ctx = scope.context()
        try:
            if column.length:
                var device_values = column.values()
                var source = device_values.create_sub_buffer[dtype](
                    column.value_offset, column.length
                )
                ctx.enqueue_copy(values.unsafe_ptr(), source)
                transferred = checked_add(
                    transferred,
                    checked_mul(column.length, size_of[Scalar[dtype]]()),
                )
            if len(validity_bytes):
                ctx.enqueue_copy(validity_bytes.unsafe_ptr(), validity.value())
                transferred = checked_add(transferred, len(validity_bytes))
            if transferred:
                scope.record_download(transferred)
                scope.synchronize()
        except error:
            scope.synchronize()
            _ = values^
            _ = validity_bytes^
            raise error^
    if not validity:
        _ = error_codes^
        return Column[Scalar[dtype]](values^)
    var bits = validity_bytes^
    if needs_repacked_validity:
        bits = _copy_bits(bits, column.validity_bit_offset, column.length)
    _ = error_codes^
    return Column[Scalar[dtype]](values=values^, bits=bits^)


def _readback_series[
    dtype: DType
](field: Series, column: NativeColumn[dtype]) raises -> Series:
    var result = Series(field.name(), readback_native[dtype](column))
    if result.dtype() != field.dtype():
        result = result.with_dtype(field.dtype())
    result._field_metadata = field._field_metadata.copy()
    return result^
