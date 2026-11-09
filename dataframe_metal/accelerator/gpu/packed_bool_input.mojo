"""Map or upload CPU Boolean bitmaps without expanding their values."""
from std.ffi import external_call
from std.memory import Pointer
from max.gpu.host import DeviceBuffer
from dataframe import DataType, Series
from dataframe.storage.boolean import BoolColumn
from .query_scope import Scope, checked_add
from .typed_storage import (
    NativeColumn,
    NativeColumnAny,
    _NativeReservation,
    _append_native_bits,
)
from .numeric_view import NumericView


def coalesced_bool_chunks(series: Series) raises -> List[Series]:
    var result = List[Series]()
    for part in series.chunks():
        if len(part) == 0:
            continue
        if not part._data.isa[BoolColumn]():
            raise Error("Metal Boolean input requires packed Bool storage")
        ref current = part._data[BoolColumn]
        if result:
            var last = len(result) - 1
            ref previous = result[last]._data[BoolColumn]
            if (
                previous._shares_buffers_with(current)
                and previous._offset + len(previous) == current._offset
            ):
                var joined = BoolColumn._from_buffers(
                    previous._data,
                    previous._bits,
                    previous._offset,
                    checked_add(len(previous), len(current)),
                )
                var merged = Series(part.name(), joined^)
                merged._field_metadata = part._field_metadata
                result[last] = merged^
                continue
        result.append(part.copy())
    return result^


def _map_bitmap(
    scope: Scope, source: Pointer[UInt8, MutAnyOrigin], count: Int
) raises -> Tuple[DeviceBuffer[DType.uint8], Int]:
    var page = Int(external_call["getpagesize", Int32]())
    var address = Int(source)
    var base = address - address % page
    var length = (
        checked_add(checked_add(address - base, count), page - 1) // page * page
    )
    _ = scope.check(length)
    return (
        scope.context().wrap_host_memory[DType.uint8](
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base), length
        ),
        address - base,
    )


def _mapped_bool(
    part: Series, scope: Scope, mut owners: List[NativeColumnAny]
) raises -> NumericView:
    ref column = part._data[BoolColumn]
    var bit = column.validity_offset() % 8
    var byte = column.validity_offset() // 8
    var count = (bit + len(column) + 7) // 8
    var mapped = _map_bitmap(
        scope, column.unsafe_values().unsafe_offset(byte), count
    )
    var validity = Optional[DeviceBuffer[DType.uint8]]()
    var validity_offset = 0
    var nullable = column.null_count() > 0
    if nullable:
        var bits = _map_bitmap(
            scope, column.unsafe_validity().unsafe_offset(byte), count
        )
        validity_offset = bits[1] * 8 + bit
        validity = bits[0].copy()
    var values = mapped[0].create_sub_buffer[DType.uint8](mapped[1], count)
    var bits = validity.value().copy() if validity else values.copy()
    var owner = NativeColumn[DType.uint8](
        scope,
        DataType.UINT8,
        count,
        mapped[0].copy(),
        validity^,
        value_offset=mapped[1],
        validity_bit_offset=validity_offset,
    )
    owners.append(NativeColumnAny(owner))
    return NumericView(
        values^, bits^, validity_offset, len(column), 3, nullable, bit
    )


def _uploaded_bool(
    series: Series, scope: Scope, mut owners: List[NativeColumnAny]
) raises -> NumericView:
    var count = checked_add(len(series), 7) // 8
    _ = scope.check(count * 4)
    var host_reservation = _NativeReservation(scope, count * 2)
    var reservation = _NativeReservation(scope, count * 2)
    var values = List[UInt8]()
    var validity = List[UInt8]()
    var rows = 0
    for part in series.chunks():
        ref column = part._data[BoolColumn]
        _append_native_bits(
            values, rows, column._data[].span(), column._offset, len(column)
        )
        if len(column._bits[]):
            _append_native_bits(
                validity,
                rows,
                column._bits[].span(),
                column._offset,
                len(column),
            )
        else:
            var valid = List[UInt8](length=(len(column) + 7) // 8, fill=255)
            _append_native_bits(validity, rows, Span(valid), 0, len(column))
        rows = checked_add(rows, len(column))
    var ctx = scope.context()
    var data = ctx.enqueue_create_buffer[DType.uint8](count)
    var bits = ctx.enqueue_create_buffer[DType.uint8](count)
    try:
        ctx.enqueue_copy(data, values.unsafe_ptr())
        ctx.enqueue_copy(bits, validity.unsafe_ptr())
        scope.record_upload(count * 2)
        scope.synchronize()
    except error:
        scope.synchronize()
        raise error
    var owner = NativeColumn[DType.uint8](
        scope,
        DataType.UINT8,
        count,
        data.copy(),
        bits.copy(),
        reservation=reservation^,
    )
    owners.append(NativeColumnAny(owner))
    return NumericView(data^, bits^, 0, rows, 3, True)


def place_packed_bool(
    series: Series,
    scope: Scope,
    zero_copy: Bool,
    mut owners: List[NativeColumnAny],
    coalesce: Bool,
) raises -> List[NumericView]:
    var result = List[NumericView]()
    if len(series) == 0:
        return result^
    if not zero_copy:
        result.append(_uploaded_bool(series, scope, owners))
        return result^
    var chunks = coalesced_bool_chunks(series) if coalesce else series.chunks()
    for part in chunks:
        if len(part):
            result.append(_mapped_bool(part, scope, owners))
    return result^
