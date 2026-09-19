"""Arrow C Data Interface interoperability with pyarrow (development only).

Run with `pixi run -e oracle oracle-arrow`. pyarrow is the reference
consumer and producer; the library itself never imports Python.

C structs are heap-allocated and passed to Python by address, as a C or
Python consumer would do.
"""
from std.memory import Pointer
from std.python import Python, PythonObject
from std.testing import assert_equal, assert_true
from dataframe import (
    ArrowArray,
    ArrowSchema,
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
    import_arrow,
    import_arrow_series,
)
from dataframe.arrow import _at, _export_frame, _leak, _reclaim


struct CStructs:
    """Heap ArrowArray/ArrowSchema pair addressed like foreign memory."""

    var array: Int
    var schema: Int

    def __init__(out self):
        self.array = _leak(ArrowArray())
        self.schema = _leak(ArrowSchema())

    def free(self):
        _ = _reclaim[ArrowArray](self.array)
        _ = _reclaim[ArrowSchema](self.schema)


def sample() raises -> DataFrame:
    var valid = List[Bool]()
    var ints = List[Int64]()
    var floats = List[Float64]()
    var bools = List[Bool]()
    var texts = List[String]()
    var times = List[Int64]()
    for i in range(37):
        times.append(Int64(i) * 2_000_000_000_000)  # within [0, 24h) in ns
        valid.append(i % 5 != 3)
        ints.append(Int64(i * 86_400 - 1000))
        floats.append(Float64(i) * 0.5 - 3.0)
        bools.append(i % 3 == 1)
        texts.append("" if i % 7 == 0 else String("ñ🔥") * (i % 3) + String(i))
    return DataFrame(
        [
            Series("i", Column[Int64](ints.copy(), valid)),
            Series("f", Column[Float64](floats^, valid)),
            Series("b", Column[Bool](bools^, valid)),
            Series("s", StringColumn(texts, valid)),
            Series("d", Column[Int64](ints.copy(), valid)).with_dtype(
                DataType.DATE
            ),
            Series("ts", Column[Int64](ints.copy(), valid)).with_dtype(
                DataType.datetime("ms")
            ),
            Series("du", Column[Int64](ints.copy(), valid)).with_dtype(
                DataType.duration("us")
            ),
            Series("t", Column[Int64](times^, valid)).with_dtype(DataType.TIME),
        ]
    )


def to_pyarrow(frame: DataFrame, releases: Int) raises -> PythonObject:
    var pa = Python.import_module("pyarrow")
    var c = CStructs()
    _export_frame(
        frame,
        _at[ArrowArray](c.array)[],
        _at[ArrowSchema](c.schema)[],
        releases,
    )
    # pyarrow moves the structs' contents out and releases them later.
    var batch = pa.RecordBatch._import_from_c(c.array, c.schema)
    c.free()
    return batch


def from_pyarrow(batch: PythonObject) raises -> DataFrame:
    var c = CStructs()
    batch._export_to_c(c.array, c.schema)
    var frame = import_arrow(c.array, c.schema)
    c.free()
    return frame^


def assert_frames_equal(a: DataFrame, b: DataFrame) raises:
    assert_equal(a.height(), b.height())
    assert_equal(a.width(), b.width())
    for k in range(a.width()):
        assert_equal(a._columns[k].name(), b._columns[k].name())
        assert_true(a._columns[k].dtype() == b._columns[k].dtype())
        assert_true(a._columns[k].equals(b._columns[k]), a._columns[k].name())


def check_export_is_valid_arrow() raises:
    var frame = sample()
    var releases = 0
    var batch = to_pyarrow(frame, Int(Pointer(to=releases)))
    batch.validate(full=True)
    var types = List[String]()
    for dtype in batch.schema.types:
        types.append(String(dtype))
    assert_equal(
        types,
        [
            "int64",
            "double",
            "bool",
            "large_string",
            "date32[day]",
            "timestamp[ms]",
            "duration[us]",
            "time64[ns]",
        ],
    )
    assert_equal(Int(py=batch.num_rows), 37)
    assert_equal(
        Int(py=batch.column(0).null_count), frame._columns[0].null_count()
    )
    # Python sees the same values we hold.
    var row = batch.slice(10, 1).to_pylist()[0]
    assert_equal(Int(py=row["i"]), 10 * 86_400 - 1000)
    assert_equal(String(row["s"]), "ñ🔥10")
    assert_true(Bool(py=row["b"]))
    assert_true(batch.slice(8, 1).to_pylist()[0]["i"] is Python.none())
    # pyarrow may move the children out and release the parent early
    # (the spec allows it), but children live as long as the batch.
    assert_true(releases <= 1)
    # Zero-copy: pyarrow reads our Int64 and UTF-8 buffers in place.
    ref ints = frame._columns[0]._data[Column[Int64]]
    assert_equal(
        Int(py=batch.column(0).buffers()[1].address),
        Int(ints._data[].unsafe_ptr()),
    )
    ref texts = frame._columns[3]._data[StringColumn]
    assert_equal(
        Int(py=batch.column(3).buffers()[2].address),
        Int(texts._bytes[].unsafe_ptr()),
    )
    # Mojo destroys `batch` after its last use above, dropping pyarrow's
    # reference; each of the 1 + 8 exported structs is released once.
    Python.import_module("gc").collect()
    assert_equal(releases, 9)


def check_round_trips_with_offsets() raises:
    var frame = sample()
    for start in [0, 1, 5, 8, 13]:
        for length in [0, 1, 7, 20]:
            var window = frame.slice(start, min(length, 37 - start))
            assert_frames_equal(from_pyarrow(to_pyarrow(window, 0)), window)
    # pyarrow-side slicing: a nonzero offset on the struct array itself.
    var batch = to_pyarrow(frame, 0).slice(9, 17)
    assert_frames_equal(from_pyarrow(batch), frame.slice(9, 17))


def check_pyarrow_produced_types() raises:
    var pa = Python.import_module("pyarrow")
    var datetime = Python.import_module("datetime")
    var none = Python.none()
    var batch = pa.record_batch(
        Python.dict(
            i8=pa.array(Python.list(1, none, -3, 4), type=pa.int8()),
            u32=pa.array(Python.list(4000000000, 1, none, 0), type=pa.uint32()),
            f32=pa.array(Python.list(1.5, none, -2.25, 0.0), type=pa.float32()),
            s=pa.array(Python.list("a", "", none, "ß"), type=pa.string()),
            ls=pa.array(
                Python.list("x", "yy", "zzz", none), type=pa.large_string()
            ),
            b=pa.array(Python.list(True, False, none, True)),
            d64=pa.array(
                Python.list(
                    datetime.date(2024, 2, 29),
                    none,
                    datetime.date(1969, 12, 31),
                    datetime.date(1970, 1, 1),
                ),
                type=pa.date64(),
            ),
            t32=pa.array(
                Python.list(0, 1000, none, 86399000), type=pa.time32("ms")
            ),
            ts=pa.array(Python.list(1, 2, 3, none), type=pa.timestamp("ns")),
        )
    ).slice(1, 3)
    var frame = from_pyarrow(batch)
    assert_equal(frame.height(), 3)
    assert_true(frame.item(0, "i8").is_null())
    # Narrow types import natively (no widening).
    assert_equal(frame.item(1, "i8").int8(), -3)
    assert_equal(frame.item(0, "u32").uint32(), 1)
    assert_equal(frame.item(1, "f32").float32(), -2.25)
    assert_equal(frame.item(0, "s").string(), "")
    assert_true(frame.item(1, "s").is_null())
    assert_equal(frame.item(2, "s").string(), "ß")
    assert_equal(frame.item(1, "ls").string(), "zzz")
    assert_true(frame.item(2, "ls").is_null())
    assert_equal(frame.item(0, "b").bool(), False)
    assert_true(frame.column("d64").dtype() == DataType.DATE)
    assert_equal(frame.item(1, "d64").to_physical(), -1)
    assert_true(frame.column("t32").dtype() == DataType.TIME)
    assert_equal(frame.item(0, "t32").to_physical(), 1_000_000_000)
    assert_true(frame.column("ts").dtype() == DataType.datetime("ns"))
    assert_equal(frame.item(1, "ts").to_physical(), 3)
    assert_true(frame.item(2, "ts").is_null())


def check_unsupported_types_are_rejected() raises:
    var pa = Python.import_module("pyarrow")
    var cases = Python.list(
        pa.array(Python.list(1, 2), type=pa.timestamp("us", tz="UTC")),
        pa.array(Python.list("a", "b")).dictionary_encode(),
        pa.array(Python.list(Python.list(1), Python.list(2))),
    )
    for array in cases:
        var c = CStructs()
        array._export_to_c(c.array, c.schema)
        var failed = False
        try:
            _ = import_arrow_series(c.array, c.schema)
        except e:
            failed = "not supported" in String(e) or "Unsupported" in String(e)
        assert_true(failed, String(array.type))
        # Released on failure too: the structs are consumed.
        assert_equal(_at[ArrowArray](c.array)[].release, 0)
        c.free()


def main() raises:
    check_export_is_valid_arrow()
    check_round_trips_with_offsets()
    check_pyarrow_produced_types()
    check_unsupported_types_are_rejected()
    print("arrow C data interface: pyarrow interop ok")
