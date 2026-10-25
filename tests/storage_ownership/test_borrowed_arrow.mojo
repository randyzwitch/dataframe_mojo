"""Foreign fixed-width payload ownership, alias isolation and Arrow release."""
# contract-group: storage.borrowed_arrow
from std.collections import Optional
from std.memory import ArcPointer, Pointer
from std.testing import assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Series,
    ArrowArray,
    ArrowSchema,
    import_arrow,
    import_arrow_series,
    export_arrow_series,
    export_arrow,
    col,
)
from dataframe.io.arrow import (
    _at,
    _buffer,
    _export_frame,
    _export_series,
    _c_string,
    _release_imported,
    _import_arrow_with_pool,
)
from dataframe.io.ipc import _open_scan
from dataframe.io.parquet import _stream_next, _stream_schema, _release_stream
from dataframe.parallel import Pool
from tests.support.lifecycle_jobs import WorkerSetting
from tests.support.arrow_foreign import integers


def imported_after_producer_scope(counter: Int) raises -> Series:
    var source = Series("x", Column[Int64]([10, 20, 30, 40]))
    var array = ArrowArray()
    var schema = ArrowSchema()
    _export_series(source, array, schema, counter)
    var original = _buffer(array, 1)
    var imported = import_arrow_series(array, schema, copy=False)
    assert_equal(array.release, 0)
    assert_equal(schema.release, 0)
    assert_equal(_at[Int](counter)[], 0)
    assert_equal(Int(imported.int64().unsafe_values()), original)
    return imported.slice(1, 2)


def exercise_last_view(counter: Int) raises:
    var view = imported_after_producer_scope(counter)
    assert_equal(view.get(0).int64(), Int64(20))
    assert_equal(view.get(1).int64(), Int64(30))
    assert_equal(_at[Int](counter)[], 0)
    var grown = view.int64()
    grown._append_column(Column[Int64]([99]))
    assert_equal(
        grown.to_list(), List[Int64]([Int64(20), Int64(30), Int64(99)])
    )
    assert_equal(view.get(1).int64(), Int64(30))
    assert_equal(_at[Int](counter)[], 0)


def test_borrowed_arrow_last_view_releases_once_after_source_and_mutation_copy() raises:
    var counter = ArcPointer(Int(0))
    exercise_last_view(Int(counter.ptr()))
    assert_equal(counter[], 1)


def exercise_frame(counter: Int, copy: Bool) raises:
    var source = DataFrame(
        [
            Series("x", Column[Int64]([1, 2, 3], [True, False, True])),
            Series("y", Column[Float64]([1.5, 2.5, 3.5])),
        ]
    )
    var array = ArrowArray()
    var schema = ArrowSchema()
    _export_frame(source, array, schema, counter)
    ref child = _at[ArrowArray](_at[Int](array.children)[])[]
    var address = _buffer(child, 1)
    var frame = import_arrow(array, schema, copy=copy)
    assert_equal(Int(frame["x"].int64().unsafe_values()) == address, not copy)
    assert_equal(_at[Int](counter)[], 3 if copy else 0)
    assert_true(frame.item(1, "x").is_null())
    assert_equal(frame.item(2, "y").float64(), Float64(3.5))
    var selected = frame.filter(col("x") > 1)
    assert_equal(selected.item(0, "x").int64(), Int64(3))
    var output = ArrowArray()
    var output_schema = ArrowSchema()
    export_arrow_series(frame["x"], output, output_schema)
    if not copy:
        assert_equal(_buffer(output, 1), address)
    _release_imported(output, output_schema)


def test_borrowed_arrow_frame_default_copy_and_retained_export_obey_release_counts() raises:
    for copy in [False, True]:
        var counter = ArcPointer(Int(0))
        exercise_frame(Int(counter.ptr()), copy)
        assert_equal(counter[], 3)


def test_borrowed_arrow_mapped_payload_outlives_reader_and_keeps_buffer_address() raises:
    var stream = _open_scan(
        "tests/fixtures/ipc/core-uncompressed.arrow", ["x"], True, True, 0, 3
    )
    var array = ArrowArray()
    var schema = ArrowSchema()
    assert_equal(_stream_next(stream, array), Int32(0))
    assert_equal(_stream_schema(stream, schema), Int32(0))
    ref child = _at[ArrowArray](_at[Int](array.children)[])[]
    var address = _buffer(child, 1)
    var frame = import_arrow(array, schema, copy=False)
    _release_stream(stream)
    assert_equal(Int(frame["x"].int64().unsafe_values()), address)
    assert_equal(frame.item(2, "x").int64(), Int64(2))
    var tail = frame.slice(1, 2)
    assert_equal(Int(tail["x"].int64().unsafe_values()), address + 8)
    assert_equal(tail.item(1, "x").int64(), Int64(2))


def exercise_error(counter: Int, parallel: Bool) raises:
    var rows = 131072 if parallel else 4
    var source = DataFrame(
        [
            Series("x", Column[Int64](List[Int64](length=rows, fill=7))),
            Series("y", Column[Int64](List[Int64](length=rows, fill=8))),
        ]
    )
    var array = ArrowArray()
    var schema = ArrowSchema()
    _export_frame(source, array, schema, counter)
    var invalid = _c_string("invalid-foreign-format")
    _at[ArrowSchema](_at[Int](schema.children + 8)[])[].format = Int(
        invalid.unsafe_ptr()
    )
    var pool = Pool(2)
    var failed = False
    try:
        _ = _import_arrow_with_pool(array, schema, pool, copy=False)
    except error:
        assert_true("Unsupported Arrow format" in String(error))
        failed = True
    assert_true(failed)
    assert_equal(array.release, 0)
    assert_equal(schema.release, 0)
    _ = invalid^


def test_borrowed_arrow_partial_and_parallel_import_errors_release_parent_once() raises:
    var setting = WorkerSetting(2)
    for parallel in [False, True]:
        var counter = ArcPointer(Int(0))
        exercise_error(Int(counter.ptr()), parallel)
        assert_equal(counter[], 3)
    _ = setting^


def test_borrowed_arrow_fixed_width_temporal_decimal_and_slice_values_match_copied_import() raises:
    var path = "tests/fixtures/ipc/typed.arrow"
    var stream = _open_scan(path, List[String](), False, True, 0, -1)
    while True:
        var array = ArrowArray()
        var schema = ArrowSchema()
        assert_equal(_stream_next(stream, array), Int32(0))
        if array.release == 0:
            break
        assert_equal(_stream_schema(stream, schema), Int32(0))
        var imported = import_arrow(array, schema, copy=False)
        assert_equal(imported["i"].dtype(), DataType.INT8)
        assert_equal(
            imported["narrow_precision_wide_storage"].dtype(),
            DataType.decimal(9, 2, 128),
        )
        var exported = ArrowArray()
        var exported_schema = ArrowSchema()

        export_arrow(imported, exported, exported_schema)
        var copied = import_arrow(exported, exported_schema)
        assert_true(copied.equals(imported))
    _release_stream(stream)


def test_borrowed_arrow_unaligned_payload_uses_owned_copy_without_mutating_producer() raises:
    var counter = ArcPointer(Int(0))
    var source = Series("x", Column[Int64]([0, 0]))
    var array = ArrowArray()
    var schema = ArrowSchema()
    _export_series(source, array, schema, Int(counter.ptr()))
    var raw: List[UInt8] = [0]
    raw.extend(Span(integers([-7, 99], 8)))
    var address = Int(raw.unsafe_ptr()) + 1
    _at[Int](array.buffers)[unsafe_offset=1] = address
    var imported = import_arrow_series(array, schema, copy=False)
    assert_equal(imported.get(0).int64(), Int64(-7))
    assert_equal(imported.get(1).int64(), Int64(99))
    assert_true(Int(imported.int64().unsafe_values()) != address)
    assert_true(not imported.int64()._data.is_borrowed())
    assert_equal(counter[], 1)
    assert_equal(raw[1], UInt8(249))
    _ = raw^
