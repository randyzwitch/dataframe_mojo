"""Tests for read_parquet through the libdfparquet backend.

Fixtures in tests/fixtures were written by pyarrow (types.parquet: one row
group, snappy; types_plain: uncompressed; row_groups: three zstd row groups;
empty: a schema with no rows). The module skips, with a notice, when the
reader library cannot be found, so the suite passes on a machine without a
C++ toolchain; set DATAFRAME_PARQUET_LIBRARY or run
`pixi run -e native build-dfparquet` to make it run.
"""
from std.collections import Optional
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from dataframe import (
    DataFrame,
    DataType,
    col,
    lit,
    parquet_backend_version,
    parquet_library_candidates,
    parquet_row_group_statistics,
    read_parquet,
    scan_parquet,
)
from dataframe.parquet import _pruned_row_groups

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


def test_time_zone_timestamps_import_as_utc_instants() raises:
    for name in [String("timestamp_utc"), "timestamp_tz"]:
        var frame = read_parquet(FIXTURES + name + ".parquet")
        assert_equal(frame.column("c").dtype(), DataType.datetime("us"))
        var c = frame.column("c").with_dtype(DataType.INT64)
        assert_equal(c.get(0).int64(), 1)
        assert_true(c.get(1).is_null())
        assert_equal(c.get(2).int64(), 3)


def test_dictionary_float16_and_views_are_coerced() raises:
    var words = read_parquet(FIXTURES + "dictionary_string.parquet").column("c")
    assert_equal(words.dtype(), DataType.STRING)
    assert_equal(words.get(0).string(), "x")
    assert_equal(words.get(1).string(), "y")
    assert_true(words.get(2).is_null())
    assert_equal(words.get(3).string(), "x")
    var halves = read_parquet(FIXTURES + "float16.parquet").column("c")
    assert_equal(halves.dtype(), DataType.FLOAT32)
    assert_equal(halves.get(0).float32(), Float32(1.5))
    assert_true(halves.get(1).is_null())
    assert_equal(halves.get(2).float32(), Float32(2.5))
    var views = read_parquet(FIXTURES + "string_view.parquet").column("c")
    assert_equal(views.dtype(), DataType.STRING)
    assert_equal(views.get(2).string(), "b")
    assert_true(views.get(1).is_null())


def test_row_groups_can_be_selected() raises:
    var one: List[Int] = [1]
    var middle = read_parquet(FIXTURES + "row_groups.parquet", row_groups=one^)
    assert_equal(middle.height(), 400)
    assert_equal(middle.column("s").get(0).string(), "row-400")
    var none = read_parquet(
        FIXTURES + "row_groups.parquet", row_groups=List[Int]()
    )
    assert_equal(none.height(), 0)
    assert_equal(none.width(), 3)
    var raised = False
    try:
        var seven: List[Int] = [7]
        _ = read_parquet(FIXTURES + "row_groups.parquet", row_groups=seven^)
    except e:
        raised = True
        assert_true("out of range" in String(e))
    assert_true(raised)


def test_row_group_statistics() raises:
    var stats = parquet_row_group_statistics(FIXTURES + "row_groups.parquet")
    assert_equal(stats.height(), 3)
    var names = stats.columns()
    assert_equal(names[0], "row_group")
    assert_equal(names[1], "rows")
    assert_equal(names[2], "min:k")
    assert_equal(names[3], "max:k")
    assert_equal(names[4], "nulls:k")
    assert_equal(stats.column("rows").get(2).int64(), 200)
    # k is i except at every seventh row, which is null (0 and 399 are).
    assert_equal(stats.column("min:k").get(0).int64(), 1)
    assert_equal(stats.column("max:k").get(0).int64(), 398)
    assert_equal(stats.column("min:k").get(2).int64(), 800)
    assert_equal(stats.column("max:k").get(2).int64(), 999)
    assert_equal(stats.column("nulls:k").get(0).int64(), 58)
    assert_equal(stats.column("nulls:k").get(1).int64(), 57)
    assert_equal(stats.column("nulls:k").get(2).int64(), 28)
    assert_equal(stats.column("min:v").dtype(), DataType.FLOAT64)
    assert_equal(stats.column("max:v").get(1).float64(), 799.0 / 4.0)
    assert_equal(stats.column("min:s").dtype(), DataType.STRING)
    assert_equal(stats.column("min:s").get(1).string(), "row-400")


def groups(mask: Optional[List[Int]]) -> String:
    if not mask:
        return "all"
    var out = String()
    for g in mask.value():
        out += String(g) + ","
    return out


def test_pruning_from_bounds() raises:
    var stats = parquet_row_group_statistics(FIXTURES + "row_groups.parquet")
    assert_equal(
        groups(_pruned_row_groups(stats, col("k") > lit(Int64(850)))), "2,"
    )
    assert_equal(
        groups(_pruned_row_groups(stats, lit(Int64(850)) < col("k"))), "2,"
    )
    assert_equal(
        groups(_pruned_row_groups(stats, col("k") >= lit(Int64(799)))), "1,2,"
    )
    assert_equal(groups(_pruned_row_groups(stats, col("v") < lit(10.0))), "0,")
    assert_equal(
        groups(_pruned_row_groups(stats, col("k") == lit(Int64(450)))), "1,"
    )
    # Lexicographic string bounds: "row-0" <= "row-500" <= "row-99" too.
    assert_equal(
        groups(_pruned_row_groups(stats, col("s") == lit("row-500"))), "0,1,"
    )
    assert_equal(
        groups(
            _pruned_row_groups(
                stats, (col("k") > lit(Int64(850))) & (col("v") < lit(10.0))
            )
        ),
        "",
    )
    assert_equal(
        groups(
            _pruned_row_groups(
                stats, (col("k") > lit(Int64(850))) | (col("v") < lit(10.0))
            )
        ),
        "0,2,",
    )
    # No usable bound: every group is read.
    assert_equal(
        groups(_pruned_row_groups(stats, col("k") != lit(Int64(5)))), "all"
    )
    assert_equal(
        groups(_pruned_row_groups(stats, col("k") > lit(Int64(-1)))), "all"
    )
    assert_equal(
        groups(_pruned_row_groups(stats, col("missing") > lit(Int64(1)))), "all"
    )
    assert_equal(groups(_pruned_row_groups(stats, col("k").is_null())), "all")


def test_scan_parquet_matches_eager_and_prunes() raises:
    var path = FIXTURES + "row_groups.parquet"
    var eager = read_parquet(path).filter(col("k") > lit(Int64(850)))
    var lazy = scan_parquet(path).filter(col("k") > lit(Int64(850))).collect()
    assert_equal(lazy.height(), eager.height())
    assert_equal(lazy.height(), 128)
    assert_equal(
        lazy.select(col("k").sum()).item().int64(),
        eager.select(col("k").sum()).item().int64(),
    )
    # A filter no group can satisfy reads nothing and keeps the schema.
    var empty = (
        scan_parquet(path)
        .filter((col("k") > lit(Int64(850))) & (col("v") < lit(10.0)))
        .collect()
    )
    assert_equal(empty.height(), 0)
    assert_equal(empty.width(), 3)
    # Projection reaches the reader; the filter column is still read.
    var narrow = (
        scan_parquet(path).filter(col("k") > lit(Int64(850))).select(["s"])
    )
    var plan = narrow.explain()
    assert_true("SCAN PARQUET" in plan)
    assert_true("[project k, s]" in plan or "[project s, k]" in plan)
    var out = narrow.collect()
    assert_equal(out.width(), 1)
    assert_equal(out.height(), 128)
    var schema = scan_parquet(path).select(["v"]).collect_schema()
    assert_equal(len(schema), 1)
    assert_equal(schema[0], "v: float64")


def test_nested_parquet_columns() raises:
    var lists = read_parquet(FIXTURES + "list_int.parquet").column("c")
    assert_equal(lists.dtype(), DataType.list(DataType.INT64))
    assert_equal(String(lists.get(0)), "[1, 2]")
    assert_true(lists.get(1).is_null())
    assert_equal(len(lists.get(2).list()), 0)
    var structs = read_parquet(FIXTURES + "struct.parquet").column("c")
    assert_equal(structs.dtype().name(), "struct[a: int64, b: string]")
    assert_equal(structs.get(0).struct_field("b").string(), "s")
    assert_true(structs.get(1).is_null())
    # Naming a nested column must read every leaf behind it.
    var projected = read_parquet(FIXTURES + "struct.parquet", columns=["c"])
    assert_equal(
        projected.column("c").dtype().name(), "struct[a: int64, b: string]"
    )
    assert_equal(projected.column("c").get(0).struct_field("a").int64(), 1)


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
