"""Budget accounting follows retained allocations, not the visible row count."""
# contract-group: storage.views
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true
from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    DataType,
    ListColumn,
    Series,
    StringColumn,
    StructColumn,
)
from dataframe.storage.buffer import _Buffer
from dataframe.storage.retained import RetainedBuffers
from dataframe.storage.string_view import StringViewBuilder
from dataframe.planning.cost.sampling import _frame_bytes, _retain_frame
from tests.storage_ownership.test_device_residence import (
    FakeDevice,
    device_column,
    residence,
    values,
)


def test_retained_numeric_capacity_validity_and_shared_windows() raises:
    var values = List[Int64](capacity=1024)
    values.extend(List[Int64]([Int64(1), Int64(2), Int64(3)]))
    var expected = values.capacity() * 8
    var frame = DataFrame([Series("x", Column[Int64](values^))])
    assert_equal(_frame_bytes(frame), expected)
    assert_equal(_frame_bytes(frame.slice(1, 1)), expected)
    assert_equal(_frame_bytes(frame.slice(0, 0)), expected)
    var usage = RetainedBuffers()
    _retain_frame(frame, usage)
    _retain_frame(frame.slice(1, 1), usage)
    assert_equal(usage.bytes, expected)
    var boolean = DataFrame(
        [Series("b", BoolColumn(List[Bool]([True, False]), [True, False]))]
    )
    assert_true(_frame_bytes(boolean) >= 2)


def test_retained_buffer_windows_and_selection_keep_complete_owner() raises:
    var values = List[Int64](capacity=2048)
    values.extend(List[Int64]([Int64(1), Int64(2), Int64(3), Int64(4)]))
    var expected = values.capacity() * 8
    var source = _Buffer(values^)
    var window = source._window(1, 1)
    var usage = RetainedBuffers()
    usage.buffer(window)
    assert_equal(usage.bytes, expected)
    usage.buffer(source)
    assert_equal(usage.bytes, expected)
    var rows = List[Int](capacity=16)
    rows.append(1)
    var row_bytes = rows.capacity() * 8
    var selected = _Buffer.selecting(source, ArcPointer(rows^), 0, 0)
    usage.buffer(selected)
    assert_equal(usage.bytes, expected + row_bytes)
    var shared = List[UInt8](capacity=4096)
    shared.append(1)
    var shared_bytes = shared.capacity()
    var arc = ArcPointer(shared^)
    var wrapped = _Buffer(arc)
    usage.buffer(wrapped)
    usage.buffer(_Buffer(arc))
    assert_equal(usage.bytes, expected + row_bytes + shared_bytes)
    _ = window^
    _ = selected^
    _ = wrapped^
    _ = source^
    _ = arc^


def test_retained_long_text_and_views_include_payload_and_unused_backing_rows() raises:
    var long = String("é") * 10000
    var flat = DataFrame(
        [Series("s", StringColumn(List[String]([long, "tiny"])))]
    )
    assert_true(_frame_bytes(flat) >= 20000 + 4 + 24)
    assert_equal(_frame_bytes(flat.slice(1, 1)), _frame_bytes(flat))
    var builder = StringViewBuilder(2)
    builder.append(StringSlice(long))
    builder.append(StringSlice("tiny"))
    var viewed = DataFrame([Series("s", StringColumn(builder^.finish()))])
    assert_true(_frame_bytes(viewed) >= 20000 + 2 * 16)
    assert_equal(_frame_bytes(viewed.slice(1, 1)), _frame_bytes(viewed))
    var usage = RetainedBuffers()
    _retain_frame(viewed, usage)
    var expected = usage.bytes
    _retain_frame(viewed.slice(1, 1), usage)
    assert_equal(usage.bytes, expected)


def test_retained_nested_children_and_categorical_dictionary_are_counted() raises:
    var child = Series(
        "value", StringColumn(List[String]([String("x") * 80000, "small"]))
    )
    var lists = Series("items", ListColumn([0, 1, 2], child.copy()))
    var frame = DataFrame([Series("outer", StructColumn([lists^]))])
    assert_true(_frame_bytes(frame) >= 80000)
    assert_equal(_frame_bytes(frame.slice(1, 1)), _frame_bytes(frame))
    var category = child.cast(DataType.CATEGORICAL)
    var coded = DataFrame([category^])
    assert_true(_frame_bytes(coded) >= 80000)
    assert_equal(_frame_bytes(coded.slice(1, 1)), _frame_bytes(coded))


def test_retained_chunks_include_published_merge_cache() raises:
    var one = Series("x", Column[Int64]([1, 2]))
    var two = Series("x", Column[Int64]([3, 4]))
    var chunks = Series._from_chunks([one.copy(), two.copy()])
    var cold = _frame_bytes(DataFrame([chunks.copy()]))
    var merged = chunks.int64()
    assert_equal(
        merged.to_list(), List[Int64]([Int64(1), Int64(2), Int64(3), Int64(4)])
    )
    var warm = _frame_bytes(DataFrame([chunks.copy()]))
    assert_true(warm >= cold + 4 * 8)
    var usage = RetainedBuffers()
    _retain_frame(DataFrame([chunks.copy()]), usage)
    _retain_frame(DataFrame([one.copy()]), usage)
    _retain_frame(DataFrame([two.copy()]), usage)
    assert_equal(usage.bytes, warm)


def test_retained_device_accounting_never_materializes_host_data() raises:
    var device = ArcPointer(FakeDevice(values(1000)))
    var column = device_column(device)
    var frame = DataFrame([Series("x", column.copy())])
    assert_equal(_frame_bytes(frame), 8000)
    assert_equal(device[].copies.load(), Int64(0))
    _ = column._get(0)
    assert_equal(_frame_bytes(frame), 16000)
    assert_equal(device[].copies.load(), Int64(1))
    var unified = residence(device, Int(device[].data.unsafe_ptr()))
    var shared = _Buffer[Int64].on_device(1000, unified)
    var usage = RetainedBuffers()
    usage.buffer(shared)
    assert_equal(usage.bytes, 8000)
    assert_equal(device[].copies.load(), Int64(1))
