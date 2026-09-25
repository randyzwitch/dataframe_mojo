"""Tests for read_parquet through the libdfparquet backend.

Fixtures in tests/fixtures were written by pyarrow (types.parquet: one row
group, snappy; types_plain: uncompressed; row_groups: three zstd row groups;
empty: a schema with no rows). The module skips, with a notice, when the
reader library cannot be found, so the suite passes on a machine without a
C++ toolchain; set DATAFRAME_PARQUET_LIBRARY or run
`pixi run -e native build-dfparquet` to make it run.
"""
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from dataframe import (
    DataFrame,
    DataType,
    col,
    parquet_backend_version,
    parquet_library_candidates,
    read_parquet,
)

comptime FIXTURES = "tests/fixtures/"


def check_types_frame(frame: DataFrame) raises:
    assert_equal(frame.height(), 4)
    assert_equal(frame.width(), 8)
    var names = frame.columns()
    assert_equal(names[0], "i")
    assert_equal(names[7], "f32")
    assert_equal(frame.column("i").dtype(), DataType.INT64)
    assert_equal(frame.column("f").dtype(), DataType.FLOAT64)
    assert_equal(frame.column("s").dtype(), DataType.STRING)
    assert_equal(frame.column("b").dtype(), DataType.BOOL)
    assert_equal(frame.column("d").dtype(), DataType.DATE)
    assert_equal(frame.column("ts").dtype(), DataType.datetime("us"))
    assert_equal(frame.column("i32").dtype(), DataType.INT32)
    assert_equal(frame.column("f32").dtype(), DataType.FLOAT32)

    var i = frame.column("i")
    assert_equal(i.get(0).int64(), 1)
    assert_true(i.get(1).is_null())
    assert_equal(i.get(3).int64(), 4)
    var f = frame.column("f")
    assert_equal(f.get(1).float64(), 2.5)
    assert_true(f.get(2).is_null())
    var s = frame.column("s")
    assert_equal(s.get(1).string(), "bb")
    assert_true(s.get(2).is_null())
    assert_equal(s.get(3).string(), "dddd")
    var b = frame.column("b")
    assert_equal(b.get(0).bool(), True)
    assert_equal(b.get(1).bool(), False)
    assert_true(b.get(2).is_null())
    # Typed cell reads are strict, so temporal columns are checked through
    # their Int64 storage: 2024-01-01 is 19723 days after the epoch and
    # 1999-12-31 is 10956; 2024-01-01 12:00 is 1704110400000000 us.
    var d = frame.column("d").with_dtype(DataType.INT64)
    assert_equal(d.get(0).int64(), 19723)
    assert_true(d.get(1).is_null())
    assert_equal(d.get(3).int64(), 10956)
    var ts = frame.column("ts").with_dtype(DataType.INT64)
    assert_equal(ts.get(0).int64(), 1704110400000000)
    assert_true(ts.get(2).is_null())
    assert_equal(frame.column("i32").get(3).int32(), Int32(10))
    assert_true(frame.column("i32").get(2).is_null())
    assert_equal(frame.column("f32").get(2).float32(), Float32(2.5))
    assert_true(frame.column("f32").get(1).is_null())


def test_reads_every_supported_type() raises:
    check_types_frame(read_parquet(FIXTURES + "types.parquet"))


def test_uncompressed_file_reads_the_same() raises:
    check_types_frame(read_parquet(FIXTURES + "types_plain.parquet"))


def test_columns_select_and_order() raises:
    var frame = read_parquet(
        FIXTURES + "types.parquet", columns=["f", "i"], use_threads=False
    )
    assert_equal(frame.width(), 2)
    var names = frame.columns()
    assert_equal(names[0], "f")
    assert_equal(names[1], "i")
    assert_equal(frame.height(), 4)
    assert_equal(frame.column("i").get(0).int64(), 1)


def test_row_groups_and_zstd() raises:
    var frame = read_parquet(FIXTURES + "row_groups.parquet")
    assert_equal(frame.height(), 1000)
    assert_equal(frame.width(), 3)
    # k is null at every seventh row (143 of them), so the sum skips those.
    var expected = 0
    for row in range(1000):
        if row % 7 != 0:
            expected += row
    assert_equal(frame.select(col("k").sum()).item().int64(), Int64(expected))
    assert_equal(frame.select(col("k").null_count()).item().int64(), 143)
    assert_equal(frame.column("s").get(999).string(), "row-999")
    assert_equal(frame.column("v").get(4).float64(), 1.0)


def test_empty_file_keeps_schema() raises:
    var frame = read_parquet(FIXTURES + "empty.parquet")
    assert_equal(frame.height(), 0)
    assert_equal(frame.width(), 8)
    assert_equal(frame.column("ts").dtype(), DataType.datetime("us"))


def test_errors_are_reported() raises:
    var raised = False
    try:
        _ = read_parquet(FIXTURES + "does_not_exist.parquet")
    except e:
        raised = True
        assert_true(String(e).startswith("read_parquet: "))
    assert_true(raised)

    raised = False
    try:
        _ = read_parquet(FIXTURES + "types.parquet", columns=["nope"])
    except e:
        raised = True
        assert_true("nope" in String(e))
    assert_true(raised)

    raised = False
    try:
        _ = read_parquet(FIXTURES + "types.parquet", columns=["i", "i"])
    except e:
        raised = True
    assert_true(raised)

    raised = False
    try:
        # A CSV is not a Parquet file; the reader must say so, not crash.
        _ = read_parquet("tests/test_parquet.mojo")
    except e:
        raised = True
    assert_true(raised)


def main() raises:
    try:
        print("libdfparquet: arrow", parquet_backend_version())
    except e:
        print("skipped: ", e)
        print("looked for:")
        for candidate in parquet_library_candidates():
            print("  ", candidate)
        return
    TestSuite.discover_tests[__functions_in_module()]().run()
