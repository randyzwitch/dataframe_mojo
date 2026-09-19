"""Row-parallel expressions (#5) and parallel filter compaction (#7) produce
exactly the serial result: same rows, order, values, nulls, and dtypes."""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
    lit,
    when,
)

comptime ROWS = 200_003  # ragged partitions and bitmap tails


def c_string(text: String) -> List[UInt8]:
    var bytes = List[UInt8]()
    bytes.extend(text.as_bytes())
    bytes.append(0)
    return bytes^


def set_threads(n: Int):
    var name = c_string("DATAFRAME_THREADS")
    var value = c_string(String(n))
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


def frame() raises -> DataFrame:
    var x = List[Float64](capacity=ROWS)
    var i64 = List[Int64](capacity=ROWS)
    var u8 = List[UInt8](capacity=ROWS)
    var flags = List[Bool](capacity=ROWS)
    var texts = List[String](capacity=ROWS)
    var valid = List[Bool](capacity=ROWS)
    var predicate_valid = List[Bool](capacity=ROWS)
    for i in range(ROWS):
        var h = (i * 2654435761) % 1_000_003
        x.append(Float64(h % 2001) / 10.0 - 100.0)
        i64.append(Int64(h) - 500_000)
        u8.append(UInt8(h % 256))
        flags.append(h % 3 == 0)
        texts.append("row" + String(h % 1009))
        valid.append(i % 13 != 5)
        predicate_valid.append(i % 17 != 2)
    return DataFrame(
        [
            Series("x", Column[Float64](x^, valid)),
            Series("i", Column[Int64](i64.copy(), valid)),
            Series("u8", Column[UInt8](u8^, valid)),
            Series("b", Column[Bool](flags.copy(), predicate_valid)),
            Series("s", StringColumn(texts, valid)),
            Series("d", Column[Int64](i64^, valid)).with_dtype(DataType.DATE),
        ]
    )


def exprs() -> List[Expr]:
    return [
        ((col("x") * 2.0 + 1.0) / 3.0).alias("fused"),
        (col("x") % 7.0).alias("mod"),
        (col("i") * 3 - 1).alias("checked"),
        (col("u8").cast(DataType.INT64) + 1).alias("widened"),
        col("s").str().to_uppercase().alias("upper"),
        (col("x") > 0).alias("positive"),
        when(col("b")).then(col("i")).otherwise(0).alias("pick"),
        (col("i") - col("i").mean().cast(DataType.INT64)).alias("centered"),
        col("x").cum_sum().alias("running"),
    ]


def assert_frames_equal(a: DataFrame, b: DataFrame, what: String) raises:
    assert_equal(a.height(), b.height(), what)
    assert_equal(a.width(), b.width(), what)
    for k in range(a.width()):
        assert_true(a._columns[k].dtype() == b._columns[k].dtype(), what)
        assert_true(
            a._columns[k].equals(b._columns[k]),
            what + " column " + a._columns[k].name(),
        )


def test_expressions_match_serial() raises:
    var df = frame()
    set_threads(1)
    var serial = df.select_exprs(exprs())
    for threads in [2, 3, 8, 32]:
        set_threads(threads)
        for batch in [1024, 777, 65536]:
            var parallel = df.select_exprs(exprs(), batch_size=batch)
            assert_frames_equal(
                serial,
                parallel,
                String(threads) + " threads, batch " + String(batch),
            )
    set_threads(1)


def test_filters_match_serial_at_every_retention() raises:
    var df = frame()
    var predicates: List[Expr] = [
        col("x") > 1000.0,  # none
        col("x") > 90.0,  # ~5%
        col("x") > 0.0,  # ~half, with nulls in the predicate
        col("b"),  # nullable Bool predicate
        col("x") > -1000.0,  # every valid row
        lit(True),  # scalar: all rows
        col("i") > col("i").mean().cast(DataType.INT64),  # aggregate-derived
    ]
    for p in range(len(predicates)):
        set_threads(1)
        var serial = df.filter(predicates[p])
        for threads in [2, 5, 16]:
            set_threads(threads)
            assert_frames_equal(
                serial,
                df.filter(predicates[p]),
                "predicate " + String(p) + ", " + String(threads) + " threads",
            )
    set_threads(1)


def test_small_empty_and_zero_column_frames() raises:
    set_threads(8)
    var df = frame()
    assert_equal(df.slice(0, 0).filter(col("x") > 0.0).height(), 0)
    assert_equal(
        df.slice(10, 50).filter(col("x") > 0.0).height(),
        df.slice(10, 50).filter(col("x") > 0.0).height(),
    )
    var no_columns = DataFrame([], height=ROWS)
    var mask = List[Bool](capacity=ROWS)
    for i in range(ROWS):
        mask.append(i % 3 == 0)
    assert_equal(no_columns.filter(Column[Bool](mask^)).height(), 66_668)
    set_threads(1)


def test_worker_errors_leave_no_partial_result() raises:
    var values = List[Int32](length=ROWS, fill=1)
    values[ROWS - 7] = Int32.MAX
    var df = DataFrame([Series("x", Column[Int32](values^))])
    set_threads(8)
    var message = String()
    try:
        _ = df.select(col("x") + 1)
    except e:
        message = String(e)
    set_threads(1)
    assert_true("int32 addition overflow" in message, message)


def test_repeated_runs_are_stable() raises:
    # Stress the output and validity ownership boundaries.
    var df = frame()
    set_threads(1)
    var serial = df.filter(col("x") > 0.0)
    set_threads(16)
    for _ in range(20):
        assert_frames_equal(serial, df.filter(col("x") > 0.0), "stress")
    set_threads(1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
