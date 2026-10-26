"""Default IPC reads share real mapped/decoded buffers and retain their owner."""
# contract-group: io.ipc
from std.ffi import external_call
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    StringColumn,
    Series,
    read_ipc,
    read_ipc_stream,
    scan_ipc,
)
from dataframe.io.ipc import _collect, _open_scan, _IpcBatches
from tests.support.ipc_buffer_observer import IpcObservation, observe_ipc
from tests.support.mapped_pages import assert_mapped_page, assert_unmapped_page


def assert_borrowed_chunks(series: Series) raises:
    for chunk in series.chunks():
        if chunk._data.isa[Column[Int64]]():
            assert_true(chunk._data[Column[Int64]]._data.is_borrowed())
        elif chunk._data.isa[Column[Float64]]():
            assert_true(chunk._data[Column[Float64]]._data.is_borrowed())
        else:
            assert_true(chunk._data[StringColumn]._bytes.is_borrowed())


def test_ipc_default_eager_stream_and_lazy_reads_borrow_mapped_or_decoded_chunks() raises:
    for codec in ["uncompressed", "lz4", "zstd"]:
        for mapped in [False, True]:
            for stream in [False, True]:
                var path = (
                    "tests/fixtures/ipc/core-"
                    + codec
                    + (".stream" if stream else ".arrow")
                )
                var frame = read_ipc_stream(
                    path, memory_map=mapped
                ) if stream else read_ipc(path, memory_map=mapped)
                assert_equal(frame.height(), 9)
                for column in frame._columns:
                    assert_borrowed_chunks(column)
            var lazy = (
                scan_ipc(
                    "tests/fixtures/ipc/core-" + codec + ".arrow",
                    memory_map=mapped,
                )
                .slice(1, 5)
                .collect()
            )
            assert_equal(lazy.height(), 5)
            assert_equal(lazy.item(0, "x").int64(), Int64(1))
            assert_equal(lazy.item(4, "x").int64(), Int64(101))
            assert_borrowed_chunks(lazy["x"])
            assert_borrowed_chunks(lazy["s"])


def escaped_eager_mapping(counts: ArcPointer[IpcObservation]) raises -> Series:
    var original = _open_scan(
        "tests/fixtures/ipc/core-uncompressed.arrow", ["x"], True, True, 0, -1
    )
    var stream = observe_ipc(original^, counts)
    var frame = _collect(stream)
    assert_equal(counts[].stream_releases, 1)
    assert_equal(counts[].array_releases, 0)
    var chunks = frame["x"].chunks()
    assert_equal(len(chunks), 3)
    for k in range(len(chunks)):
        assert_equal(
            Int(chunks[k].int64().unsafe_values()), counts[].addresses[k]
        )
    return frame["x"].slice(0, 9)


def escaped_lazy_mapping(counts: ArcPointer[IpcObservation]) raises -> Series:
    var batches = _IpcBatches(
        "tests/fixtures/ipc/core-uncompressed.arrow", ["x"], True, True, 1, 2, 1
    )
    var original = batches.stream^
    batches.stream = observe_ipc(original^, counts)
    var next = batches.next()
    var frame = next.take()
    assert_equal(Int(frame["x"].int64().unsafe_values()), counts[].addresses[0])
    _ = batches^
    assert_equal(counts[].stream_releases, 1)
    assert_equal(counts[].array_releases, 0)
    return frame["x"].slice(0, 1)


def exercise_escaped_mapping(
    counts: ArcPointer[IpcObservation], lazy: Bool
) raises:
    var series = escaped_lazy_mapping(
        counts
    ) if lazy else escaped_eager_mapping(counts)
    assert_equal(series.get(0).int64(), Int64(1) if lazy else Int64(0))
    var page = Int(external_call["getpagesize", Int32]())
    for address in counts[].addresses:
        assert_mapped_page(address - address % page)
    assert_equal(counts[].array_releases, 0)
    _ = series^


def test_ipc_default_eager_and_incremental_buffers_keep_exact_native_addresses_until_last_view() raises:
    for lazy in [False, True]:
        var counts = ArcPointer(IpcObservation())
        exercise_escaped_mapping(counts, lazy)
        assert_equal(counts[].stream_releases, 1)
        assert_equal(counts[].array_releases, 1 if lazy else 3)
        var page = Int(external_call["getpagesize", Int32]())
        for address in counts[].addresses:
            assert_unmapped_page(address - address % page)
