"""Native byte views for heterogeneous fused kernels; no payload conversion."""
from max.gpu.host import DeviceBuffer
from std.sys import size_of
from dataframe import DataFrame, DataType
from .query_scope import Scope
from .reduction_inputs import _place_chunks
from .typed_storage import NativeColumnAny


@fieldwise_init
struct NumericView(Copyable, Movable):
    var values: DeviceBuffer[DType.uint8]
    var bits: DeviceBuffer[DType.uint8]
    var bit_offset: Int
    var rows: Int
    var tag: Int
    var nullable: Bool

    def view(self, start: Int, length: Int) raises -> Self:
        var width = 8 if self.tag == 1 else 4
        return Self(
            self.values.create_sub_buffer[DType.uint8](
                start * width, length * width
            ),
            self.bits.copy(),
            self.bit_offset + start if self.nullable else 0,
            length,
            self.tag,
            self.nullable,
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
) raises -> List[NumericView]:
    var source = frame.column(name)
    var chunks = _place_chunks[D](scope, source, zero_copy)
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
) raises -> List[NumericView]:
    var dtype = frame.column(name).dtype()
    if dtype == DataType.INT32:
        return _place[DType.int32](
            frame, name, scope, zero_copy, empty, 0, owners
        )
    if dtype == DataType.INT64:
        return _place[DType.int64](
            frame, name, scope, zero_copy, empty, 1, owners
        )
    return _place[DType.float32](
        frame, name, scope, zero_copy, empty, 2, owners
    )
