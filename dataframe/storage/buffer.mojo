"""Shared owned or read-only foreign buffers, with explicit writable access."""
from std.collections import Optional
from std.sys import size_of
from std.memory import ArcPointer, Pointer, Layout, Allocation, alloc, dealloc
from dataframe.storage.residence import (
    _DeviceResidence,
    _host_address_or_abort,
)

comptime _ReleaseForeign = def(Int) thin abi("C") -> None


struct _ForeignOwner(Movable):
    """An opaque producer owner released once after the last buffer view."""

    var context: Int
    var release: Int

    def __init__(out self, context: Int, release: Int):
        self.context = context
        self.release = release

    def __deinit__(deinit self):
        if self.release != 0:
            var callback = self.release
            Pointer(to=callback).unsafe_bitcast[_ReleaseForeign]()[](
                self.context
            )


struct _BufferData[T: Copyable & Deinitable](Movable, Sized):
    var values: List[Self.T]
    var address: Int
    var length: Int
    var owner: Optional[ArcPointer[_ForeignOwner]]
    # Set when the data lives in an accelerator's memory; `address` is then
    # 0 and host reads go through the residence's one-time copy.
    var residence: Optional[ArcPointer[_DeviceResidence]]

    def __init__(out self, var values: List[Self.T]):
        self.values = values^
        self.address = 0
        self.length = 0
        self.owner = None
        self.residence = None

    def __init__(
        out self, address: Int, length: Int, owner: ArcPointer[_ForeignOwner]
    ):
        self.values = List[Self.T]()
        self.address = address
        self.length = length
        self.owner = owner
        self.residence = None

    def __init__(
        out self, length: Int, residence: ArcPointer[_DeviceResidence]
    ):
        # Device buffers take the foreign branch of every read, so buffers
        # that own a host list pay nothing for them.
        self.values = List[Self.T]()
        self.address = 0
        self.length = length
        self.owner = ArcPointer(_ForeignOwner(0, 0))
        self.residence = residence

    def __len__(self) -> Int:
        return self.length if self.owner else len(self.values)

    def unsafe_ptr(self) -> Pointer[Self.T, MutAnyOrigin]:
        """A read-only pointer; the buffer/producer must remain alive."""
        if self.owner:
            if self.address == 0 and self.residence:
                return Pointer[Self.T, MutAnyOrigin](
                    unsafe_from_address=_host_address_or_abort(
                        self.residence.value()[]
                    )
                )
            return Pointer[Self.T, MutAnyOrigin](
                unsafe_from_address=self.address
            )
        return Pointer[Self.T, MutAnyOrigin](
            unsafe_from_address=Int(self.values.unsafe_ptr())
        )

    def __getitem__(self, index: Int) -> ref[ImmutAnyOrigin] Self.T:
        return Pointer[Self.T, ImmutAnyOrigin](
            unsafe_from_address=Int(self.unsafe_ptr())
        )[unsafe_offset=index]

    def span(self) -> Span[Self.T, ImmutAnyOrigin]:
        return Span[Self.T, ImmutAnyOrigin](
            unsafe_ptr=Pointer[Self.T, ImmutAnyOrigin](
                unsafe_from_address=Int(self.unsafe_ptr())
            ),
            length=len(self),
        )

    def copy(self) -> List[Self.T]:
        """An owned payload copy, not another view of foreign memory."""
        var result = List[Self.T](capacity=len(self))
        result.extend(self.span())
        return result^


struct _Buffer[T: Copyable & Deinitable](Copyable, Sized):
    """Shared immutable storage; writable() detaches shared/foreign buffers."""

    var _state: ArcPointer[_BufferData[Self.T]]

    def __init__(out self, var values: List[Self.T]):
        self._state = ArcPointer(_BufferData(values^))

    def __init__(out self, values: ArcPointer[List[Self.T]]):
        var pointer = alloc(
            Layout[ArcPointer[List[Self.T]]](length=1)
        ).unsafe_leak()
        pointer.unsafe_write(values.copy())
        var callback: _ReleaseForeign = _release_shared_list[Self.T]
        var owner = ArcPointer(
            _ForeignOwner(
                Int(pointer),
                Pointer(to=callback).unsafe_bitcast[Int]()[],
            )
        )
        self._state = ArcPointer(
            _BufferData[Self.T](
                Int(values[].unsafe_ptr()),
                len(values[]),
                owner,
            )
        )

    def __init__(out self, var state: _BufferData[Self.T]):
        self._state = ArcPointer(state^)

    @staticmethod
    def borrowed(
        address: Int, length: Int, owner: ArcPointer[_ForeignOwner]
    ) -> Self:
        return Self(_BufferData[Self.T](address, length, owner))

    @staticmethod
    def on_device(
        length: Int, residence: ArcPointer[_DeviceResidence]
    ) raises -> Self:
        """`length` elements in an accelerator's memory. The element type
        must be plain bytes (numbers, validity words): the host copy is a
        byte copy."""
        if residence[].bytes != length * size_of[Self.T]():
            raise Error("device buffer size does not match its length")
        return Self(_BufferData[Self.T](length, residence))

    def residence(self) -> Optional[ArcPointer[_DeviceResidence]]:
        """The device copy, for a backend that can read it in place."""
        return self._state[].residence.copy()

    def on_host(self) -> Bool:
        """Whether host reads need no copy."""
        var residence = self._state[].residence.copy()
        return not residence or residence.value()[].on_host()

    def copy_to_host(self) raises:
        """Copy device data to the host now, raising on failure. Call before
        handing the buffer to code that reads it without raising."""
        var residence = self._state[].residence.copy()
        if residence:
            _ = residence.value()[].host_address()

    def __getitem__(self) -> ref[ImmutAnyOrigin] _BufferData[Self.T]:
        return self._state[]

    def __len__(self) -> Int:
        return len(self._state[])

    def __getitem__(self, index: Int) -> ref[ImmutAnyOrigin] Self.T:
        return self._state[][index]

    def unsafe_ptr(self) -> Pointer[Self.T, MutAnyOrigin]:
        return self._state[].unsafe_ptr()

    def _window(self, offset: Int, length: Int) raises -> Self:
        if offset < 0 or length < 0 or offset > len(self) - length:
            raise Error("Invalid buffer window")
        if offset == 0 and length == len(self):
            return self.copy()
        var pointer = alloc(
            Layout[ArcPointer[_BufferData[Self.T]]](length=1)
        ).unsafe_leak()
        pointer.unsafe_write(self._state.copy())
        var callback: _ReleaseForeign = _release_shared_buffer[Self.T]
        var owner = ArcPointer(
            _ForeignOwner(
                Int(pointer), Pointer(to=callback).unsafe_bitcast[Int]()[]
            )
        )
        return Self.borrowed(
            Int(self.unsafe_ptr().unsafe_offset(offset)), length, owner
        )

    def is_borrowed(self) -> Bool:
        return Bool(self._state[].owner)

    def count(self) -> Int:
        return Int(self._state.count())

    def ptr(self) -> Pointer[_BufferData[Self.T], ImmutAnyOrigin]:
        return Pointer[_BufferData[Self.T], ImmutAnyOrigin](
            unsafe_from_address=Int(self._state.ptr())
        )

    def writable(mut self) -> ref[MutAnyOrigin] List[Self.T]:
        if self._state.count() != 1 or self.is_borrowed():
            self._state = ArcPointer(_BufferData(self._state[].copy()))
        return Pointer(to=self._state[].values).unsafe_origin_cast[
            MutAnyOrigin
        ]()[]


def _release_shared_list[T: Copyable & Deinitable](context: Int) abi("C"):
    var pointer = Pointer[ArcPointer[List[T]], MutUntrackedOrigin](
        unsafe_from_address=context
    )
    var held = pointer.unsafe_take_pointee()
    dealloc(
        Allocation[ArcPointer[List[T]]](
            unsafe_owned_ptr=pointer,
            layout=Layout[ArcPointer[List[T]]](length=1),
        )
    )
    _ = held^


def _release_shared_buffer[T: Copyable & Deinitable](context: Int) abi("C"):
    var pointer = Pointer[ArcPointer[_BufferData[T]], MutUntrackedOrigin](
        unsafe_from_address=context
    )
    var held = pointer.unsafe_take_pointee()
    dealloc(
        Allocation[ArcPointer[_BufferData[T]]](
            unsafe_owned_ptr=pointer,
            layout=Layout[ArcPointer[_BufferData[T]]](length=1),
        )
    )
    _ = held^
