"""Native byte views for heterogeneous fused kernels; no payload conversion."""
from max.gpu.host import DeviceBuffer
from std.sys import size_of
from std.memory import Pointer
from dataframe.storage.boolean import BoolColumn
from dataframe import Column, DataFrame, DataType, Series
from .query_scope import Scope
from .reduction_inputs import _place_chunks
from .typed_storage import NativeColumnAny, coalesced_native_chunks


from .numeric_view import NumericView
from .packed_bool_input import (
    place_packed_bool,
    coalesced_bool_chunks,
    _map_bitmap,
)


def _place[
    D: DType
](
    frame: DataFrame,
    name: String,
    scope: Scope,
    zero_copy: Bool,
    empty: DeviceBuffer[DType.uint8],
    tag: Int,
    mut owners: List[NativeColumnAny],
    coalesce: Bool,
) raises -> List[NumericView]:
    var source = frame.column(name)
    var chunks = _place_chunks[D](scope, source, zero_copy, coalesce=coalesce)
    var views = List[NumericView]()
    for chunk in chunks:
        owners.append(NativeColumnAny(chunk))
        var values = chunk.values().create_sub_buffer[DType.uint8](
            chunk.value_offset * size_of[Scalar[D]](),
            chunk.length * size_of[Scalar[D]](),
        )
        var bits = (
            chunk.validity()
            .value()
            .copy() if chunk.has_validity() else empty.copy()
        )
        views.append(
            NumericView(
                values^,
                bits^,
                chunk.validity_bit_offset,
                chunk.length,
                tag,
                chunk.has_validity(),
            )
        )
    return views^


def numeric_views(
    frame: DataFrame,
    name: String,
    scope: Scope,
    zero_copy: Bool,
    empty: DeviceBuffer[DType.uint8],
    mut owners: List[NativeColumnAny],
    *,
    coalesce: Bool = False,
) raises -> List[NumericView]:
    var dtype = frame.column(name).dtype()
    if dtype == DataType.BOOL:
        return place_packed_bool(
            frame.column(name), scope, zero_copy, owners, coalesce
        )
    if dtype == DataType.INT32:
        return _place[DType.int32](
            frame, name, scope, zero_copy, empty, 0, owners, coalesce
        )
    if dtype == DataType.INT64:
        return _place[DType.int64](
            frame, name, scope, zero_copy, empty, 1, owners, coalesce
        )
    return _place[DType.float32](
        frame, name, scope, zero_copy, empty, 2, owners, coalesce
    )


def single_native_span(series: Series) raises -> Bool:
    """Whether all logical chunks can share one native value/bitmap mapping."""
    if series.dtype() == DataType.BOOL:
        return len(coalesced_bool_chunks(series)) == 1
    comptime for i in range(3):
        comptime D = DType.int32 if i == 0 else (
            DType.int64 if i == 1 else DType.float32
        )
        if series.dtype() == DataType.of(D):
            var groups = coalesced_native_chunks[D](series)
            return len(groups) == 1 and groups[0]._data.isa[Column[Scalar[D]]]()
    return False


def _validity_view(
    scope: Scope,
    empty: DeviceBuffer[DType.uint8],
    rows: Int,
    bits: Pointer[UInt8, MutAnyOrigin],
    offset: Int,
    nullable: Bool,
) raises -> NumericView:
    if not nullable:
        return NumericView(empty.copy(), empty.copy(), 0, rows, 0, False)
    var bit = offset % 8
    var mapped = _map_bitmap(
        scope, bits.unsafe_offset(offset // 8), (bit + rows + 7) // 8
    )
    return NumericView(
        empty.copy(), mapped[0].copy(), mapped[1] * 8 + bit, rows, 0, True
    )


def _numeric_validity_views[
    D: DType
](
    series: Series, scope: Scope, empty: DeviceBuffer[DType.uint8]
) raises -> List[NumericView]:
    var result = List[NumericView]()
    for part in coalesced_native_chunks[D](series):
        if not part._data.isa[Column[Scalar[D]]]():
            raise Error("Metal count input requires native primitive storage")
        ref column = part._data[Column[Scalar[D]]]
        result.append(
            _validity_view(
                scope,
                empty,
                len(part),
                column.unsafe_validity(),
                column.validity_offset(),
                len(column._bits[]) != 0,
            )
        )
    return result^


def validity_views(
    series: Series, scope: Scope, empty: DeviceBuffer[DType.uint8]
) raises -> List[NumericView]:
    """Map only validity; the caller retains the source through the GPU drain."""
    if series.dtype() == DataType.BOOL:
        var result = List[NumericView]()
        for part in coalesced_bool_chunks(series):
            ref column = part._data[BoolColumn]
            result.append(
                _validity_view(
                    scope,
                    empty,
                    len(part),
                    column.unsafe_validity(),
                    column.validity_offset(),
                    len(column._bits[]) != 0,
                )
            )
        return result^
    comptime for i in range(3):
        comptime D = DType.int32 if i == 0 else DType.int64 if i == 1 else DType.float32
        if series.dtype() == DataType.of(D):
            return _numeric_validity_views[D](series, scope, empty)
    raise Error("Metal count input requires a native dtype")
