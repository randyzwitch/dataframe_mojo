"""Shared owned or read-only foreign buffers, with explicit writable access."""
from std.collections import Optional
from std.memory import ArcPointer, Pointer, Layout, Allocation, alloc, dealloc

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

    def __init__(out self, var values: List[Self.T]):
        self.values = values^
        self.address = 0
        self.length = 0
        self.owner = None

    def __init__(
        out self, address: Int, length: Int, owner: ArcPointer[_ForeignOwner]
    ):
        self.values = List[Self.T]()
        self.address = address
        self.length = length
        self.owner = owner

    def __len__(self) -> Int:
        return self.length if self.owner else len(self.values)

    def unsafe_ptr(self) -> Pointer[Self.T, MutAnyOrigin]:
        """A read-only pointer; the buffer/producer must remain alive."""
        if self.owner:
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


struct _Buffer[T: Copyable & Deinitable](Copyable):
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

    @staticmethod
    def borrowed(
        address: Int, length: Int, owner: ArcPointer[_ForeignOwner]
    ) -> Self:
        var result = Self(List[Self.T]())
        result._state = ArcPointer(_BufferData[Self.T](address, length, owner))
        return result^

    def __getitem__(self) -> ref[ImmutAnyOrigin] _BufferData[Self.T]:
        return self._state[]

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
