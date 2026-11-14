"""Capacity accounting for retained buffers, without reading their payloads.

Track one live set at a time: addresses can be reused after an owner is freed.
Owned allocations are counted by capacity, shared allocations once. Opaque
foreign owners expose only their declared buffer spans; allocator overhead and
undisclosed producer allocations are outside this count. This is a collection
threshold metric, not a query or process memory limit.
"""
from std.memory import ArcPointer, Pointer
from std.sys import size_of
from dataframe.storage.buffer import (
    _Buffer,
    _BufferData,
    _ReleaseForeign,
    _release_shared_buffer,
    _release_shared_list,
)


struct RetainedBuffers(Movable):
    var bytes: Int
    var _allocations: Dict[Int, Int]
    var _visited: Dict[Int, Bool]

    def __init__(out self):
        self.bytes = 0
        self._allocations = Dict[Int, Int]()
        self._visited = Dict[Int, Bool]()

    def visit(mut self, address: Int) -> Bool:
        if address in self._visited:
            return False
        self._visited[address] = True
        return True

    def add(mut self, address: Int, bytes: Int):
        if bytes <= 0:
            return
        var previous = self._allocations.get(address, 0)
        if bytes > previous:
            self.bytes += bytes - previous
            self._allocations[address] = bytes

    def list[T: Copyable & Deinitable](mut self, values: List[T]):
        self.add(Int(values.unsafe_ptr()), values.capacity() * size_of[T]())

    def buffer[T: Copyable & Deinitable](mut self, buffer: _Buffer[T]):
        self._data(buffer[])

    def _data[T: Copyable & Deinitable](mut self, data: _BufferData[T]):
        if not self.visit(Int(Pointer(to=data))):
            return
        if data.owner:
            ref owner = data.owner.value()[]
            var shared: _ReleaseForeign = _release_shared_buffer[T]
            var list: _ReleaseForeign = _release_shared_list[T]
            if owner.release == Pointer(to=shared).unsafe_bitcast[Int]()[]:
                ref held = Pointer[ArcPointer[_BufferData[T]], ImmutAnyOrigin](
                    unsafe_from_address=owner.context
                )[]
                self._data(held[])
            elif owner.release == Pointer(to=list).unsafe_bitcast[Int]()[]:
                ref held = Pointer[ArcPointer[List[T]], ImmutAnyOrigin](
                    unsafe_from_address=owner.context
                )[]
                self.list(held[])
            elif not data.residence and not data.repeat:
                self.add(data.address, data.length * size_of[T]())
        else:
            self.list(data.values)
        if data.repeat:
            # A constant holds one value until a reader writes it out.
            ref repeat = data.repeat.value()[]
            self.add(Int(Pointer(to=repeat)), size_of[T]())
            if repeat.is_filled():
                self.list(repeat._filled)
        if data.residence:
            ref residence = data.residence.value()[]
            # Use the residence identity: inspecting usage must never trigger
            # a device-to-host copy. Unified buffers have only one allocation.
            self.add(Int(Pointer(to=residence)), residence.bytes)
            if not residence._unified and residence.on_host():
                self.add(Int(residence._host.load()), residence.bytes)
        if data.selection:
            self.list(data.selection.value()[].rows[])
        var stats = data.statistics()
        if stats:
            ref record = stats.value()[]
            self.list(record.vectors)
            self.list(record.groups)
            if record.pin:
                self.buffer(record.pin.value())
