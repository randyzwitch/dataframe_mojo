"""A native IPC mapping remains owned when only an Arrow export survives."""
# contract-group: io.ipc
from std.ffi import external_call
from std.memory import ArcPointer
from std.testing import assert_equal
from dataframe import ArrowArray, ArrowSchema, export_arrow_series
from dataframe.io.arrow import _at, _buffer, _release_imported
from tests.interchange.test_ipc_ownership import escaped_lazy_mapping
from tests.support.ipc_buffer_observer import IpcObservation
from tests.support.mapped_pages import assert_mapped_page, assert_unmapped_page


def export_after_reader_and_column_scope(
    counts: ArcPointer[IpcObservation],
    mut array: ArrowArray,
    mut schema: ArrowSchema,
) raises:
    var series = escaped_lazy_mapping(counts)
    export_arrow_series(series, array, schema)
    assert_equal(
        _buffer(array, 1) + Int(array.offset) * 8, counts[].addresses[0]
    )
    _ = series^


def test_ipc_native_mapping_outlives_reader_and_column_until_sole_arrow_export_is_released() raises:
    var counts = ArcPointer(IpcObservation())
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_after_reader_and_column_scope(counts, array, schema)
    assert_equal(counts[].stream_releases, 1)
    assert_equal(counts[].array_releases, 0)
    var address = counts[].addresses[0]
    var page = Int(external_call["getpagesize", Int32]())
    assert_mapped_page(address - address % page)
    assert_equal(_at[Int64](address)[], Int64(1))
    _release_imported(array, schema)
    assert_equal(counts[].array_releases, 1)
    assert_unmapped_page(address - address % page)
