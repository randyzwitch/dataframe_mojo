"""Partitioned (multi-threaded) reductions agree with a single-threaded pass."""
from std.ffi import external_call
from std.math import isnan
from std.testing import TestSuite, assert_equal, assert_true
from dataframe import (
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    StringColumn,
    col,
)
from dataframe.parallel import (
    Job,
    MIN_ROWS_PER_WORKER,
    partitions,
    run_jobs,
    worker_count,
)

comptime ROWS = 300_001  # several workers, and a ragged final partition


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
    # Keep both buffers alive until setenv has copied them.
    _ = name^
    _ = value^


def frame() raises -> DataFrame:
    var ints = List[Int64](capacity=ROWS)
    var floats = List[Float64](capacity=ROWS)
    var bools = List[Bool](capacity=ROWS)
    var texts = List[String](capacity=ROWS)
    var small = List[Int32](capacity=ROWS)
    var big = List[UInt64](capacity=ROWS)
    var keys = List[Int64](capacity=ROWS)
    var valid = List[Bool](capacity=ROWS)
    for i in range(ROWS):
        var x = (i * 7919) % 100_003
        ints.append(Int64(x) - 50_000)
        floats.append(
            Float64(0) / Float64(0) if i % 9973 == 0 else Float64(x) / 3.0
        )
        bools.append(x % 3 == 0)
        texts.append("k" + String(x % 997))
        small.append(Int32(x % 1000) - 500)
        big.append(UInt64(x) * 184_467_440_737 + (UInt64(1) << 63))
        keys.append(Int64(i % 13))
        valid.append(i % 11 != 4)
    return DataFrame(
        [
            Series("i", Column[Int64](ints^, valid)),
            Series("f", Column[Float64](floats^, valid)),
            Series("b", Column[Bool](bools^, valid)),
            Series("s", StringColumn(texts, valid)),
            Series("i32", Column[Int32](small^, valid)),
            Series("u64", Column[UInt64](big^, valid)),
            Series("g", Column[Int64](keys^)),
        ]
    )


def exprs() -> List[Expr]:
    var out = List[Expr]()
    for name in ["i", "f", "b", "s", "i32", "u64"]:
        out.append(col(name).count().alias(name + "_count"))
        out.append(col(name).null_count().alias(name + "_nulls"))
        out.append(col(name).min().alias(name + "_min"))
        out.append(col(name).max().alias(name + "_max"))
        out.append(col(name).first().alias(name + "_first"))
        out.append(col(name).last().alias(name + "_last"))
        out.append(col(name).n_unique().alias(name + "_n_unique"))
    for name in ["i", "f", "i32"]:
        out.append(col(name).sum().alias(name + "_sum"))
        out.append(col(name).mean().alias(name + "_mean"))
        out.append(col(name).std().alias(name + "_std"))
        out.append(col(name).median().alias(name + "_median"))
    out.append(col("b").any().alias("b_any"))
    out.append(col("b").all().alias("b_all"))
    out.append(col("f").len().alias("len"))
    return out^


def close(a: Float64, b: Float64) -> Bool:
    if isnan(a) or isnan(b):
        return isnan(a) and isnan(b)
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def assert_same(serial: DataFrame, parallel: DataFrame, what: String) raises:
    assert_equal(serial.height(), parallel.height())
    assert_equal(serial.width(), parallel.width())
    for k in range(serial.width()):
        ref a = serial._columns[k]
        ref b = parallel._columns[k]
        var label = what + " " + a.name()
        assert_true(a.dtype() == b.dtype(), label)
        if a.dtype() == DataType.FLOAT64:
            # Float sums and moments may reassociate; everything else is exact.
            for row in range(len(a)):
                var x = a.get(row)
                var y = b.get(row)
                assert_equal(x.is_null(), y.is_null(), label)
                if not x.is_null():
                    assert_true(close(x.float64(), y.float64()), label)
        else:
            assert_true(a.equals(b), label)


def test_global_reductions_match_serial() raises:
    var df = frame()
    set_threads(1)
    var serial = df.select_exprs(exprs())
    for threads in [2, 3, 8, 32]:
        set_threads(threads)
        assert_same(serial, df.select_exprs(exprs()), String(threads))
    set_threads(1)


def test_grouped_reductions_match_serial() raises:
    var df = frame()
    set_threads(1)
    var serial = df.group_by("g", maintain_order=True).agg(exprs())
    for threads in [2, 8]:
        set_threads(threads)
        var parallel = df.group_by("g", maintain_order=True).agg(exprs())
        assert_same(serial, parallel, "grouped " + String(threads))
    set_threads(1)


def test_errors_raise_from_workers() raises:
    var df = DataFrame(
        [Series("x", Column[Int32](List[Int32](length=ROWS, fill=Int32.MAX)))]
    )
    set_threads(8)
    var message = String()
    try:
        _ = df.select(col("x").sum())
    except e:
        message = String(e)
    set_threads(1)
    assert_true("int32 expression sum overflow" in message, message)


def test_worker_count_and_partitions() raises:
    set_threads(8)
    assert_equal(worker_count(10), 1)
    assert_equal(worker_count(MIN_ROWS_PER_WORKER * 3), 3)
    assert_equal(worker_count(MIN_ROWS_PER_WORKER * 100), 8)
    set_threads(1)
    assert_equal(worker_count(MIN_ROWS_PER_WORKER * 100), 1)
    var bounds = partitions(10_000, 3, 1024)
    assert_equal(bounds, [0, 4096, 8192, 10_000])
    assert_equal(partitions(100, 4, 64), [0, 64, 100, 100, 100])


struct Square(Job):
    var input: Int
    var output: Int

    def __init__(out self, input: Int):
        self.input = input
        self.output = 0

    def run(mut self) raises:
        if self.input < 0:
            raise Error("negative input " + String(self.input))
        self.output = self.input * self.input


def test_run_jobs_returns_results_in_order() raises:
    var jobs = List[Square]()
    for i in range(20):
        jobs.append(Square(i))
    run_jobs(jobs)
    assert_equal(len(jobs), 20)
    for i in range(20):
        assert_equal(jobs[i].output, i * i)
    var bad = List[Square]()
    bad.append(Square(2))
    bad.append(Square(-3))
    var message = String()
    try:
        run_jobs(bad)
    except e:
        message = String(e)
    assert_equal(message, "negative input -3")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
