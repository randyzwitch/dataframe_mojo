"""Observe actual native IPC buffer addresses and producer release callbacks."""
from std.memory import ArcPointer, Pointer
from dataframe import ArrowArray, ArrowSchema
from dataframe.io.arrow import (
    _at,
    _buffer,
    _leak,
    _reclaim,
    _call_array_release,
)
from dataframe.io.parquet import (
    _ArrowArrayStream,
    _release_stream,
    _stream_next,
    _stream_schema,
)
from tests.support.lifecycle_stream import (
    ArrayRelease,
    StreamGet,
    StreamRelease,
    StreamError,
)


struct IpcObservation(Movable):
    var addresses: List[Int]
    var stream_releases: Int
    var array_releases: Int

    def __init__(out self):
        self.addresses = List[Int]()
        self.stream_releases = 0
        self.array_releases = 0


struct _ArrayWatch(Movable):
    var release: Int
    var private_data: Int
    var counts: ArcPointer[IpcObservation]

    def __init__(
        out self,
        release: Int,
        private_data: Int,
        counts: ArcPointer[IpcObservation],
    ):
        self.release = release
        self.private_data = private_data
        self.counts = counts.copy()


struct _StreamWatch(Movable):
    var original: _ArrowArrayStream
    var counts: ArcPointer[IpcObservation]

    def __init__(
        out self,
        var original: _ArrowArrayStream,
        counts: ArcPointer[IpcObservation],
    ):
        self.original = original^
        self.counts = counts.copy()

    def __deinit__(deinit self):
        _release_stream(self.original)
        self.counts[].stream_releases += 1


def release_observed_array(array: Pointer[ArrowArray, MutAnyOrigin]) abi("C"):
    var state = _reclaim[_ArrayWatch](array[].private_data)
    array[].release = state.release
    array[].private_data = state.private_data
    _call_array_release(array)
    state.counts[].array_releases += 1


def observed_next(stream_address: Int, output: Int) abi("C") -> Int32:
    ref stream = _at[_ArrowArrayStream](stream_address)[]
    ref state = _at[_StreamWatch](stream.private_data)[]
    ref array = _at[ArrowArray](output)[]
    var status = _stream_next(state.original, array)
    if array.release != 0:
        if array.n_children > 0:
            ref child = _at[ArrowArray](_at[Int](array.children)[])[]
            state.counts[].addresses.append(
                _buffer(child, 1) + (Int(child.offset) + Int(array.offset)) * 8
            )
        var context = _leak(
            _ArrayWatch(array.release, array.private_data, state.counts)
        )
        var callback: ArrayRelease = release_observed_array
        array.private_data = context
        array.release = Pointer(to=callback).unsafe_bitcast[Int]()[]
    return status


def observed_schema(stream_address: Int, output: Int) abi("C") -> Int32:
    ref stream = _at[_ArrowArrayStream](stream_address)[]
    ref state = _at[_StreamWatch](stream.private_data)[]
    return _stream_schema(state.original, _at[ArrowSchema](output)[])


def observed_error(stream_address: Int) abi("C") -> Int:
    ref stream = _at[_ArrowArrayStream](stream_address)[]
    ref state = _at[_StreamWatch](stream.private_data)[]
    var callback = state.original.get_last_error
    return Pointer(to=callback).unsafe_bitcast[StreamError]()[](
        Int(Pointer(to=state.original))
    )


def observed_release(stream_address: Int) abi("C"):
    ref stream = _at[_ArrowArrayStream](stream_address)[]
    var context = stream.private_data
    stream.release = 0
    stream.private_data = 0
    _ = _reclaim[_StreamWatch](context)


def observe_ipc(
    var original: _ArrowArrayStream, counts: ArcPointer[IpcObservation]
) -> _ArrowArrayStream:
    var stream = _ArrowArrayStream()
    stream.private_data = _leak(_StreamWatch(original^, counts))
    var next: StreamGet = observed_next
    var schema: StreamGet = observed_schema
    var error: StreamError = observed_error
    var release: StreamRelease = observed_release
    stream.get_next = Pointer(to=next).unsafe_bitcast[Int]()[]
    stream.get_schema = Pointer(to=schema).unsafe_bitcast[Int]()[]
    stream.get_last_error = Pointer(to=error).unsafe_bitcast[Int]()[]
    stream.release = Pointer(to=release).unsafe_bitcast[Int]()[]
    return stream^
