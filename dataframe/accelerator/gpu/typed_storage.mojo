"""Typed, shared ownership for native-width device columns.

Values keep their physical dtype. Optional validity is a packed bitmap whose
bit offset is independent of the values view; optional errors contain one
UInt32 code per retained row. This module depends only on query_scope, so it
can be used by kernels and materializers without a residency cycle.
"""
from max.gpu.host import DeviceBuffer
from std.memory import ArcPointer
from std.sys import size_of
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
        if logical.physical() != DataType.of(Self.dtype):
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
        if logical.physical() != DataType.of(Self.dtype):
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
        if logical.physical() != DataType.of(Self.dtype):
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


def upload_float32(
    scope: Scope, series: Series
) raises -> NativeColumn[DType.float32]:
    """Bulk-upload a Float32 Series, preserving slices and packed validity.

    Values always move as Float32 bytes. A single nullable chunk retains the
    source bitmap's bit offset; multiple chunks repack only their bitmap to
    concatenate row ranges. Source chunks remain alive through synchronization.
    """
    if series.dtype() != DataType.FLOAT32:
        raise Error("Native Float32 upload requires a Float32 Series")
    var chunks = series.chunks()
    var rows = 0
    var has_validity = False
    for part in chunks:
        if not part._data.isa[Column[Float32]]():
            raise Error(
                "Native Float32 upload requires physical Float32 values"
            )
        ref column = part._data[Column[Float32]]
        rows = checked_add(rows, len(column))
        if column.null_count() > 0:
            has_validity = True
    if rows != len(series):
        raise Error("Native Float32 upload chunk lengths differ from Series")

    if rows == 0:
        return NativeColumn[DType.float32](scope, DataType.FLOAT32, 0)

    var source_bit_offset = 0
    var validity_bytes = 0
    var packed_bits = List[UInt8]()
    if has_validity:
        if len(chunks) == 1:
            ref column = chunks[0]._data[Column[Float32]]
            source_bit_offset = column.validity_offset() % 8
            validity_bytes = (source_bit_offset + rows + 7) // 8
        else:
            validity_bytes = checked_add(rows, 7) // 8

    # A multi-chunk nullable upload temporarily holds the packed host bitmap
    # alongside the device values and bitmap. Preflight all of it before
    # allocating that host bitmap or any device storage.
    var preflight_bytes = native_column_bytes[DType.float32](
        rows, validity_bytes, 0
    )
    if len(chunks) > 1 and has_validity:
        preflight_bytes = checked_add(preflight_bytes, validity_bytes)
    _ = scope.check(preflight_bytes)
    var reservation = _NativeReservation(
        scope, native_column_bytes[DType.float32](rows, validity_bytes, 0)
    )
    var bitmap_reservation = Optional[_NativeReservation]()
    if len(chunks) > 1 and has_validity:
        bitmap_reservation = _NativeReservation(scope, validity_bytes)

    if has_validity and len(chunks) > 1:
        packed_bits = List[UInt8](capacity=validity_bytes)
        var appended = 0
        for part in chunks:
            ref column = part._data[Column[Float32]]
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
    var values = ctx.enqueue_create_buffer[DType.float32](rows)
    var validity = Optional[DeviceBuffer[DType.uint8]]()
    if validity_bytes:
        validity = Optional[DeviceBuffer[DType.uint8]](
            ctx.enqueue_create_buffer[DType.uint8](validity_bytes)
        )

    var uploaded = 0
    try:
        var target = 0
        for part in chunks:
            ref column = part._data[Column[Float32]]
            var count = len(column)
            if count:
                var destination = values.create_sub_buffer[DType.float32](
                    target, count
                )
                ctx.enqueue_copy(destination, column.unsafe_values())
                uploaded = checked_add(uploaded, checked_mul(count, 4))
            target += count
        if validity_bytes:
            if len(chunks) == 1:
                ref column = chunks[0]._data[Column[Float32]]
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

    return NativeColumn[DType.float32](
        scope,
        DataType.FLOAT32,
        rows,
        values^,
        validity^,
        None,
        0,
        source_bit_offset,
        Optional[_NativeReservation](reservation^),
    )


def readback_float32(
    column: NativeColumn[DType.float32],
) raises -> Column[Float32]:
    """Bulk-read a Float32 view and adopt values plus packed validity bits."""
    var scope = column.scope()
    var validity = column.validity()
    var errors = column.errors()
    var raw_validity_bytes = len(validity.value()) if validity else 0
    var packed_validity_bytes = checked_add(column.length, 7) // 8
    var needs_repacked_validity = validity and (
        column.validity_bit_offset != 0
        or raw_validity_bytes != packed_validity_bytes
    )
    var host_bytes = checked_mul(column.length, size_of[Float32]())
    if validity:
        host_bytes = checked_add(host_bytes, raw_validity_bytes)
        if needs_repacked_validity:
            host_bytes = checked_add(host_bytes, packed_validity_bytes)
    if errors:
        host_bytes = checked_add(
            host_bytes,
            checked_mul(len(errors.value()), size_of[UInt32]()),
        )
        host_bytes = checked_add(
            host_bytes, checked_mul(column.length, size_of[UInt32]())
        )
    # Includes all host readback/repack temporaries before any of them exist.
    _ = scope.check(host_bytes)

    var error_codes = column.read_error_codes()
    for code in error_codes:
        if code:
            raise data_fault(
                "Mojo GPU typed Float32 result reported a deferred error"
            )

    var values = List[Float32](length=column.length, fill=0)
    var validity_bytes = List[UInt8]()
    if validity:
        validity_bytes = List[UInt8](length=raw_validity_bytes, fill=0)
    var transferred = 0
    if column.length or len(validity_bytes):
        var ctx = scope.context()
        try:
            if column.length:
                var device_values = column.values()
                var source = device_values.create_sub_buffer[DType.float32](
                    column.value_offset, column.length
                )
                ctx.enqueue_copy(values.unsafe_ptr(), source)
                transferred = checked_add(
                    transferred, checked_mul(column.length, size_of[Float32]())
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
        return Column[Float32](values^)
    var bits = validity_bytes^
    if needs_repacked_validity:
        bits = _copy_bits(bits, column.validity_bit_offset, column.length)
    _ = error_codes^
    return Column[Float32](values=values^, bits=bits^)
