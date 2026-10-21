"""Exact-capacity scratch reuse for one exclusively leased Metal context.

Only private Int64 and UInt32 root buffers participate. Native column handles
and sub-buffer views are deliberately excluded because their aliases can outlive
an individual operator. The caller returns scratch only after its final use has
completed, and serializes pool access through the owning service's lock.
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys import size_of
from .target import METAL_TARGET

comptime _Int64Buffer = DeviceBuffer[DType.int64] if METAL_TARGET else Int
comptime _UInt32Buffer = DeviceBuffer[DType.uint32] if METAL_TARGET else Int


struct ScratchPool(Movable):
    var int64_buffers: List[_Int64Buffer]
    var uint32_buffers: List[_UInt32Buffer]
    var byte_limit: Int
    var entry_limit: Int
    var idle_bytes: Int
    var idle_entries: Int
    var hits: Int
    var misses: Int

    def __init__(out self, byte_limit: Int, entry_limit: Int):
        self.int64_buffers = List[_Int64Buffer]()
        self.uint32_buffers = List[_UInt32Buffer]()
        self.byte_limit = byte_limit
        self.entry_limit = entry_limit
        self.idle_bytes = 0
        self.idle_entries = 0
        self.hits = 0
        self.misses = 0

    def clear(mut self):
        self.int64_buffers.clear()
        self.uint32_buffers.clear()
        self.idle_bytes = 0
        self.idle_entries = 0

    def take[
        dtype: DType
    ](mut self, context: DeviceContext, length: Int) raises -> Optional[
        DeviceBuffer[dtype]
    ]:
        comptime if METAL_TARGET:
            comptime if dtype == DType.int64:
                for index in range(len(self.int64_buffers)):
                    if (
                        len(
                            rebind[DeviceBuffer[DType.int64]](
                                self.int64_buffers[index]
                            )
                        )
                        == length
                    ):
                        if (
                            rebind[DeviceBuffer[DType.int64]](
                                self.int64_buffers[index]
                            ).context()
                            != context
                        ):
                            raise Error(
                                "Mojo GPU scratch pool context mismatch"
                            )
                        self.idle_bytes -= length * size_of[Int64]()
                        self.idle_entries -= 1
                        self.hits += 1
                        return rebind[DeviceBuffer[dtype]](
                            self.int64_buffers.pop(index)
                        )
            elif dtype == DType.uint32:
                for index in range(len(self.uint32_buffers)):
                    if (
                        len(
                            rebind[DeviceBuffer[DType.uint32]](
                                self.uint32_buffers[index]
                            )
                        )
                        == length
                    ):
                        if (
                            rebind[DeviceBuffer[DType.uint32]](
                                self.uint32_buffers[index]
                            ).context()
                            != context
                        ):
                            raise Error(
                                "Mojo GPU scratch pool context mismatch"
                            )
                        self.idle_bytes -= length * size_of[UInt32]()
                        self.idle_entries -= 1
                        self.hits += 1
                        return rebind[DeviceBuffer[dtype]](
                            self.uint32_buffers.pop(index)
                        )
        self.misses += 1
        return None

    def put[
        dtype: DType
    ](mut self, context: DeviceContext, var buffer: DeviceBuffer[dtype]) raises:
        comptime if METAL_TARGET and (
            dtype == DType.int64 or dtype == DType.uint32
        ):
            if buffer.context() != context:
                raise Error("Mojo GPU scratch pool context mismatch")
            var length = len(buffer)
            if length > Int.MAX // size_of[Scalar[dtype]]():
                raise Error("Mojo GPU scratch pool allocation size overflow")
            var bytes = length * size_of[Scalar[dtype]]()
            if (
                self.idle_entries >= self.entry_limit
                or bytes > self.byte_limit - self.idle_bytes
            ):
                return
            comptime if dtype == DType.int64:
                self.int64_buffers.append(rebind[_Int64Buffer](buffer))
            else:
                self.uint32_buffers.append(rebind[_UInt32Buffer](buffer))
            self.idle_bytes += bytes
            self.idle_entries += 1
