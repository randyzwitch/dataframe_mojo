"""Resource ownership verified with real foreign callbacks and production cursors."""
from std.memory import Pointer
from std.testing import assert_equal, assert_true, assert_raises
from tests.support.mapped_pages import assert_mapped_page, assert_unmapped_page
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    Series,
    import_arrow,
)
from dataframe.arrow import _export_frame
from dataframe.parquet import _ParquetBatches, _collect_stream
from tests.support.lifecycle_stream import (
    StreamState,
    counted_stream,
    array_release,
    schema_release,
    at,
)


def assert_releases(state: StreamState, arrays: Int, schemas: Int) raises:
    assert_equal(state.stream_releases, 1)
    assert_equal(state.root_array_releases, arrays)
    assert_equal(state.child_array_releases, arrays)
    assert_equal(state.root_schema_releases, schemas)
    assert_equal(state.child_schema_releases, schemas)


def consume_all(mut state: StreamState, size: Int) raises -> List[DataFrame]:
    var cursor = _ParquetBatches(counted_stream(state), size)
    var frames = List[DataFrame]()
    while True:
        var batch = cursor.next()
        if not batch:
            break
        frames.append(batch.value().copy())
    # EOF is terminal: neither callbacks nor cleanup run a second time.
    var calls = state.next_calls
    assert_true(not cursor.next())
    assert_equal(state.next_calls, calls)
    return frames^


def consume_one(
    mut state: StreamState, fail_consumer: Bool
) raises -> DataFrame:
    var cursor = _ParquetBatches(counted_stream(state), 1)
    var first = cursor.next().value().copy()
    if fail_consumer:
        raise Error("deliberate consumer cancellation")
    return first^


def consume_until_error(mut state: StreamState) raises:
    var cursor = _ParquetBatches(counted_stream(state), 2)
    while cursor.next():
        pass


def test_stream_eof_releases_each_array_schema_and_stream_once() raises:
    for size in [1, 2, 3, 17]:
        var state = StreamState()
        var frames = consume_all(state, size)
        assert_releases(state, 2, 2)
        assert_equal(state.next_calls, 3)
        assert_equal(state.schema_calls, 2)
        assert_equal(state.emitted, 2)
        var seen = 0
        state.values[0] = -777
        state.values[1] = -888
        for frame in frames:
            assert_equal(frame.columns(), List[String](["x"]))
            for row in range(frame.height()):
                seen += 1
                assert_equal(frame.item(row, "x").int64(), Int64(seen))
        assert_equal(seen, 4)
    var state = StreamState()
    var stream = counted_stream(state)
    var eager = _collect_stream(stream)
    assert_releases(state, 2, 2)
    assert_equal(stream.release, 0)
    assert_equal(eager.height(), 4)
    for row in range(4):
        assert_equal(eager.item(row, "x").int64(), Int64(row + 1))


def test_stream_early_stop_and_consumer_exception_release_pending_buffers() raises:
    var state = StreamState()
    var first = consume_one(state, False)
    assert_releases(state, 1, 1)
    assert_equal(state.next_calls, 1)
    state.values[0] = -999
    assert_equal(first.item(0, "x").int64(), Int64(1))
    var cancelled = StreamState()
    with assert_raises():
        _ = consume_one(cancelled, True)
    assert_releases(cancelled, 1, 1)
    assert_equal(cancelled.next_calls, 1)
    # A fresh cursor after cancellation owns a fresh set of resources.
    var retry = StreamState()
    var frames = consume_all(retry, 1)
    assert_releases(retry, 2, 2)
    assert_equal(len(frames), 4)


def test_stream_schema_import_and_late_errors_release_partial_exports() raises:
    for mode in [1, 2, 3, 4]:
        var state = StreamState(mode)
        var rejected = False
        try:
            consume_until_error(state)
        except error:
            rejected = True
            if mode != 2:
                assert_true(
                    String(error).find("deliberate lifecycle producer failure")
                    >= 0
                )
        assert_true(rejected)
        assert_releases(state, 1, 0 if mode == 4 else 1)
        assert_equal(state.next_calls, 2 if mode == 3 else 1)
        # The eager consumer must obey the same foreign ownership protocol.
        var eager_state = StreamState(mode)
        var stream = counted_stream(eager_state)
        with assert_raises():
            _ = _collect_stream(stream)
        assert_releases(eager_state, 1, 0 if mode == 4 else 1)
        assert_equal(stream.release, 0)


def export_after_source_destruction(
    mut array: ArrowArray, mut schema: ArrowSchema, counter: Int
) raises:
    var source = DataFrame(
        [
            Series("i", Column[Int64]([4, 5, 6])),
            Series("s", Column[String](["alpha", "beta", "gamma"])),
            Series("f", Column[Float64]([1.25, -2.5, 3.75])),
        ]
    )
    _export_frame(source, array, schema, counter)


def move_array(mut target: ArrowArray, mut source: ArrowArray):
    # C Data Interface move protocol: copy the struct, null the old release.
    target.length = source.length
    target.null_count = source.null_count
    target.offset = source.offset
    target.n_buffers = source.n_buffers
    target.n_children = source.n_children
    target.buffers = source.buffers
    target.children = source.children
    target.dictionary = source.dictionary
    target.release = source.release
    target.private_data = source.private_data
    source.release = 0


def move_schema(mut target: ArrowSchema, mut source: ArrowSchema):
    target.format = source.format
    target.name = source.name
    target.metadata = source.metadata
    target.flags = source.flags
    target.n_children = source.n_children
    target.children = source.children
    target.dictionary = source.dictionary
    target.release = source.release
    target.private_data = source.private_data
    source.release = 0


def test_exported_children_outlive_source_and_parent_with_exact_release_counts() raises:
    var releases = 0
    var array = ArrowArray()
    var schema = ArrowSchema()
    export_after_source_destruction(array, schema, Int(Pointer(to=releases)))
    assert_equal(releases, 0)
    var child = ArrowArray()
    var child_schema = ArrowSchema()
    var child_address = at[Int](array.children)[]
    var schema_address = at[Int](schema.children)[]
    move_array(child, at[ArrowArray](child_address)[])
    move_schema(child_schema, at[ArrowSchema](schema_address)[])
    array_release(array)
    schema_release(schema)
    assert_equal(releases, 3)  # parent and the two siblings
    assert_equal(array.release, 0)
    assert_equal(schema.release, 0)
    var values_address = at[Int](child.buffers)[unsafe_offset=1]
    for row in range(3):
        assert_equal(
            at[Int64](values_address)[unsafe_offset=row], Int64(row + 4)
        )
    # Import consumes the detached child. Its callback owns the final buffer.
    from dataframe import import_arrow_series

    var retained = import_arrow_series(child, child_schema)
    assert_equal(releases, 4)
    assert_equal(child.release, 0)
    assert_equal(child_schema.release, 0)
    assert_equal(retained.get(2).int64(), Int64(6))
    array_release(array)
    schema_release(schema)
    array_release(child)
    schema_release(child_schema)
    assert_equal(releases, 4)


def test_arrow_import_errors_and_released_inputs_consume_each_live_export_once() raises:
    for failure in range(3):
        var releases = 0
        var array = ArrowArray()
        var schema = ArrowSchema()
        export_after_source_destruction(
            array, schema, Int(Pointer(to=releases))
        )
        if failure == 0:
            array_release(array)
        elif failure == 1:
            schema_release(schema)
        else:
            array.n_children = 2  # exported schema still declares three
        with assert_raises():
            _ = import_arrow(array, schema)
        assert_equal(array.release, 0)
        assert_equal(schema.release, 0)
        assert_equal(releases, 4)
        with assert_raises():
            _ = import_arrow(array, schema)
        assert_equal(releases, 4)


def consume_discarded_csv_pages(path: String) raises -> List[DataFrame]:
    from std.ffi import external_call
    from dataframe import CsvField, CsvSchema
    from dataframe.csv_types import _map_file
    from dataframe.csv_reader import _CsvBatches

    var mapping = _map_file(path)
    assert_true(mapping.address != 0)
    var first_page = mapping.address
    assert_mapped_page(first_page)
    var schema = CsvSchema([CsvField.int64("x"), CsvField.string("s")])
    var cursor = _CsvBatches(mapping^, schema^, [], -1, 37)
    var frames = List[DataFrame]()
    var proved_unmapped = False
    while True:
        var next = cursor.next()
        if not next:
            break
        var job = next.take()
        job.run()
        frames.append(job^.into_frame())
        cursor.discard()
        if not proved_unmapped and cursor.mapping.address != first_page:
            # Probe the actual mapping; a counter or a copied result alone
            # would not prove that the source pages ceased to exist.
            assert_unmapped_page(first_page)
            proved_unmapped = True
    assert_true(proved_unmapped)
    return frames^


def test_csv_strings_remain_owned_after_consumed_source_pages_are_unmapped() raises:
    from std.os import getenv

    var path = getenv("COVERAGE_TEST_TMP") + "/lifecycle-owned-strings.csv"
    var suffix = String(
        ' café "quoted" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\ncontinued'
    )
    with open(path, "w") as file:
        file.write("x,s\n")
        for row in range(512):
            file.write(
                String(row)
                + ',"name-'
                + String(row)
                + ' café ""quoted"" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\ncontinued"\n'
            )
    var frames = consume_discarded_csv_pages(path)
    var index = 0
    for frame in frames:
        for row in range(frame.height()):
            assert_equal(frame.item(row, "x").int64(), Int64(index))
            assert_equal(
                frame.item(row, "s").string(), "name-" + String(index) + suffix
            )
            index += 1
    assert_equal(index, 512)
