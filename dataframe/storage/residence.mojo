"""Buffers whose data lives in an accelerator's memory.

A backend that leaves a result on its device hands the core a
`_DeviceResidence`: an opaque handle, which device holds the bytes, and
C-ABI callbacks to copy them to host memory and to free them. The core never
imports a backend. The first host read copies the bytes once, however many
threads ask at the same time; the device copy stays until the last buffer
referring to it is gone, so a later device consumer can read it without
another upload.

On unified memory (Apple silicon, AMD APUs, NVIDIA Grace-class parts) the
device memory may be directly readable by the CPU. A backend then passes
that address as `host_address`, and no copy is made.
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.memory import Pointer, Layout, Allocation, alloc, dealloc
from std.os import abort

# (context, host destination address, bytes) -> 0 on success.
comptime _CopyToHost = def(Int, Int, Int) thin abi("C") -> Int
comptime _ReleaseDevice = def(Int) thin abi("C") -> None

comptime _NOT_COPIED = 0
comptime _COPYING = 1
comptime _COPIED = 2
comptime _FAILED = 3

# Host copies are aligned for SIMD loads of any element type.
comptime _HostBlock = SIMD[DType.uint8, 64]


struct _DeviceResidence(Movable):
    """One device allocation and, once read on the host, its host copy."""

    var backend: String
    var device: Int
    var context: Int
    var bytes: Int
    var to_host: Int
    var release: Int
    var _host: Atomic[Int64]
    var _state: Atomic[Int64]
    var _unified: Bool

    def __init__(
        out self,
        *,
        backend: String,
        device: Int,
        context: Int,
        bytes: Int,
        to_host: Int,
        release: Int,
        host_address: Int = 0,
    ):
        """`backend` names the provider ("nvidia", "amd", "metal"), so
        devices of different vendors in one machine stay distinct; `device`
        is that backend's ordinal. `to_host` is a `_CopyToHost` and
        `release` a `_ReleaseDevice`, as integer addresses. A nonzero
        `host_address` is device memory the CPU can read in place."""
        self.backend = backend
        self.device = device
        self.context = context
        self.bytes = bytes
        self.to_host = to_host
        self.release = release
        self._host = Atomic[Int64](Int64(host_address))
        self._state = Atomic[Int64](
            Int64(_COPIED if host_address != 0 else _NOT_COPIED)
        )
        self._unified = host_address != 0

    def __deinit__(deinit self):
        if (
            not self._unified
            and self.bytes > 0
            and self._state.load() == _COPIED
        ):
            var address = Int(self._host.load())
            var pointer = Pointer[_HostBlock, MutUntrackedOrigin](
                unsafe_from_address=address
            )
            dealloc(
                Allocation[_HostBlock](
                    unsafe_owned_ptr=pointer,
                    layout=Layout[_HostBlock](length=_blocks(self.bytes)),
                )
            )
        if self.release != 0:
            var callback = self.release
            Pointer(to=callback).unsafe_bitcast[_ReleaseDevice]()[](
                self.context
            )

    def on_host(self) -> Bool:
        """Whether host reads need no copy."""
        return self._state.load() == _COPIED

    def host_address(self) raises -> Int:
        """The host copy's address, copying the device bytes the first time.
        Concurrent callers wait for the one copy; a failed copy raises for
        every caller."""
        # Field pointers give interior mutability, as `_BufferFacts` does;
        # the residence is shared by every buffer that refers to it.
        var cell = Pointer(to=self._state).unsafe_mut_cast[True]()
        var host = Pointer(to=self._host).unsafe_mut_cast[True]()
        while True:
            var state = cell[].load()
            if state == _COPIED:
                return Int(host[].load())
            if state == _FAILED:
                raise Error(
                    "copying a "
                    + self.backend
                    + " device buffer to the host failed"
                )
            var expected = Int64(_NOT_COPIED)
            if cell[].compare_exchange(expected, Int64(_COPYING)):
                var address = self._copy()
                if address == 0 and self.bytes > 0:
                    cell[].store(_FAILED)
                    continue
                host[].store(Int64(address))
                cell[].store(_COPIED)
                return address
            _ = external_call["sched_yield", Int32]()

    def _copy(self) -> Int:
        if self.bytes == 0:
            return 0
        var pointer = alloc(
            Layout[_HostBlock](length=_blocks(self.bytes))
        ).unsafe_leak()
        var callback = self.to_host
        var status = Pointer(to=callback).unsafe_bitcast[_CopyToHost]()[](
            self.context, Int(pointer), self.bytes
        )
        if status != 0:
            dealloc(
                Allocation[_HostBlock](
                    unsafe_owned_ptr=pointer,
                    layout=Layout[_HostBlock](length=_blocks(self.bytes)),
                )
            )
            return 0
        return Int(pointer)


def _blocks(bytes: Int) -> Int:
    return (bytes + 63) // 64


def _host_address_or_abort(residence: _DeviceResidence) -> Int:
    """For readers that cannot raise (`_BufferData.unsafe_ptr`). Copy the
    buffer with `_Buffer.copy_to_host`, which raises, before handing it to
    such readers; this path is the safety net."""
    try:
        return residence.host_address()
    except error:
        abort(String(error))
