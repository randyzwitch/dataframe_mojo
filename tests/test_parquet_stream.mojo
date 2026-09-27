"""Stream callback ownership, including failures after partial consumption."""
from std.memory import Pointer
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import Column, DataFrame, Series
from dataframe.arrow import (
    ArrowArray,
    ArrowSchema,
    _at,
    _c_string,
    _export_frame,
    _release_imported,
)
from dataframe.parquet import (
    _ArrowArrayStream,
    _StreamGet,
    _StreamError,
    _StreamRelease,
    _collect_stream,
    _ParquetBatches,
)


struct State(Movable):
    var array: ArrowArray
    var schema: ArrowSchema
    var error: List[UInt8]
    var mode: Int
    var calls: Int
    var array_releases: Int
    var stream_releases: Int

    def __init__(out self, mode: Int):
        self.array = ArrowArray()
        self.schema = ArrowSchema()
        self.error = _c_string("injected stream failure")
        self.mode = mode
        self.calls = 0
        self.array_releases = 0
        self.stream_releases = 0


def next_batch(address: Int, output: Int) abi("C") -> Int32:
    ref stream = _at[_ArrowArrayStream](address)[]
    ref state = _at[State](stream.private_data)[]
    state.calls += 1
    if state.calls > 1:
        return 5 if state.mode == 1 else 0
    _ = external_call["memcpy", Int](output, Int(Pointer(to=state.array)), 80)
    state.array.release = 0
    return 0


def get_schema(address: Int, output: Int) abi("C") -> Int32:
    ref stream = _at[_ArrowArrayStream](address)[]
    ref state = _at[State](stream.private_data)[]
    if state.mode == 2:
        return 5
    _ = external_call["memcpy", Int](output, Int(Pointer(to=state.schema)), 72)
    state.schema.release = 0
    if state.mode == 3:
        _at[ArrowSchema](output)[].n_children += 1
    return 0


def last_error(address: Int) abi("C") -> Int:
    ref stream = _at[_ArrowArrayStream](address)[]
    return Int(_at[State](stream.private_data)[].error.unsafe_ptr())


def release_stream(address: Int) abi("C"):
    ref stream = _at[_ArrowArrayStream](address)[]
    ref state = _at[State](stream.private_data)[]
    state.stream_releases += 1
    _release_imported(state.array, state.schema)
    stream.release = 0


def exercise(
    mut state: State, cursor: Bool = False, early: Bool = False
) raises:
    var frame = DataFrame([Series("k", Column[Int64]([1, 2]))])
    _export_frame(
        frame, state.array, state.schema, Int(Pointer(to=state.array_releases))
    )
    var stream = _ArrowArrayStream()
    var next_fn: _StreamGet = next_batch
    var schema_fn: _StreamGet = get_schema
    var error_fn: _StreamError = last_error
    var release_fn: _StreamRelease = release_stream
    stream.get_next = Pointer(to=next_fn).unsafe_bitcast[Int]()[]
    stream.get_schema = Pointer(to=schema_fn).unsafe_bitcast[Int]()[]
    stream.get_last_error = Pointer(to=error_fn).unsafe_bitcast[Int]()[]
    stream.release = Pointer(to=release_fn).unsafe_bitcast[Int]()[]
    stream.private_data = Int(Pointer(to=state))
    var failed = False
    try:
        if cursor:
            var batches = _ParquetBatches(stream^, 1)
            var first = batches.next()
            assert_true(first.value().equals(frame.head(1)))
            if not early:
                var second = batches.next()
                assert_true(second.value().equals(frame.slice(1, 1)))
                assert_true(not batches.next())
            _ = batches^
        else:
            var result = _collect_stream(stream)
            assert_true(result.equals(frame))
            assert_equal(stream.release, 0)
    except e:
        failed = True
        if state.mode == 1 or state.mode == 2:
            assert_true("injected stream failure" in String(e))
    assert_equal(failed, state.mode != 0)
    assert_equal(state.stream_releases, 1)
    assert_equal(state.array_releases, 2)


def test_stream_releases_once_on_eof_next_schema_and_import_errors() raises:
    for mode in range(4):
        var state = State(mode)
        exercise(state)


def test_batch_cursor_releases_on_eof_error_and_early_stop() raises:
    for mode in range(4):
        var state = State(mode)
        exercise(state, cursor=True)
    var state = State(0)
    exercise(state, cursor=True, early=True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
