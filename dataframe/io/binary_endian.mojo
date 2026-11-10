"""Fixed-width scalar loads from byte spans, independent of alignment."""
from std.memory import bitcast
from std.sys import Endian, size_of


@always_inline
def read_scalar[
    D: DType, endian: Endian
](data: Span[UInt8, _], at: Int) -> Scalar[D]:
    comptime width = size_of[Scalar[D]]()
    var bytes = Array[UInt8, width](uninitialized=True)
    comptime for i in range(width):
        bytes[i] = data[at + i]
    comptime bits = DType.uint8 if width == 1 else (
        DType.uint16 if width
        == 2 else DType.uint32 if width
        == 4 else DType.uint64
    )
    return bitcast[D](Scalar[bits].from_bytes[endian=endian](bytes))
