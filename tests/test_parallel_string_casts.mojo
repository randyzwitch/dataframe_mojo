"""String-to-number casts in row ranges on every worker (#149).

Results must not depend on how rows are split: a cast on many workers
equals the same cast on one, for every number type, with nulls, text that
fails to parse, a filter mask (when/then), and view-storage strings. A
strict cast with failures in several ranges reports the earliest row, as a
serial loop did.
"""
from std.ffi import external_call
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import (
    Column,
    DataFrame,
    Expr,
    Series,
    StringColumn,
    col,
    lit,
    when,
)
from dataframe.string_view import StringViewBuilder


def set_threads(n: Int):
    var name = String("DATAFRAME_THREADS")
    var value = String(n)
    _ = external_call["setenv", Int32](
        Int(name.unsafe_ptr()), Int(value.unsafe_ptr()), Int32(1)
    )
    _ = name^
    _ = value^


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self, bound: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(bound))


def texts(rows: Int, seed: UInt64) -> List[String]:
    var rng = Lcg(seed)
    var out = List[String](capacity=rows)
    var bad: List[String] = ["x", "", " 1", "1e", "--2", "0x10", "nan?"]
    for _ in range(rows):
        var pick = rng.next(20)
        if pick == 0:
            out.append(bad[rng.next(len(bad))])
        elif pick < 8:
            out.append(String(rng.next(300) - 150))
        elif pick < 12:
            out.append(String(rng.next(1 << 40)))
        elif pick < 16:
            out.append(String(Float64(rng.next(100000)) / 7))
        else:
            out.append(String(Float64(rng.next(1000)) * 1e-5) + "e3")
    return out^


def frame(rows: Int, seed: UInt64) raises -> DataFrame:
    var values = texts(rows, seed)
    var valid = List[Bool](capacity=rows)
    var rng = Lcg(seed + 1)
    for _ in range(rows):
        valid.append(rng.next(11) != 0)
    return DataFrame(
        [
            Series("s", StringColumn(values, valid)),
            Series("k", Column[Int64]([Int64(i % 3) for i in range(rows)])),
        ]
    )


def casts(data: DataFrame) raises -> DataFrame:
    var targets: List[String] = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ]
    var exprs = List[Expr]()
    for t in targets:
        exprs.append(col("s").cast(t, strict=False).alias(t))
    # A mask: rows where k is 0 are cast, the rest are null.
    exprs.append(
        when(col("k") == lit(Int64(0)))
        .then(col("s").cast("float64", strict=False))
        .otherwise(lit(Float64(-1)))
        .alias("masked")
    )
    return data.select_exprs(exprs)


def test_results_do_not_depend_on_workers() raises:
    var data = frame(200_000, 3)
    set_threads(1)
    var serial = casts(data)
    set_threads(8)
    var parallel = casts(data)
    assert_true(parallel.equals(serial))
    # View-storage strings take the same values.
    var builder = StringViewBuilder()
    var values = texts(100_000, 9)
    for v in values:
        builder.append(StringSlice(v))
    var views = DataFrame([Series("s", StringColumn(builder^.finish()))])
    var plain = DataFrame([Series("s", StringColumn(values))])
    assert_true(
        views.select_exprs([col("s").cast("float64", strict=False)]).equals(
            plain.select_exprs([col("s").cast("float64", strict=False)])
        )
    )


def test_strict_reports_the_earliest_failure() raises:
    set_threads(8)
    var values = List[String](length=300_000, fill="7")
    values[250_000] = "late"
    values[120_000] = "early"
    var data = DataFrame([Series("s", StringColumn(values))])
    with assert_raises(contains="failed at row 120000 for value 'early'"):
        _ = data.select_exprs([col("s").cast("int64")])
    values[120_000] = "7"
    data = DataFrame([Series("s", StringColumn(values))])
    with assert_raises(contains="failed at row 250000 for value 'late'"):
        _ = data.select_exprs([col("s").cast("int32")])
    values[250_000] = "300"
    data = DataFrame([Series("s", StringColumn(values))])
    with assert_raises(contains="out of int8 range"):
        _ = data.select_exprs([col("s").cast("int8")])
    var ok = data.select_exprs([col("s").cast("int16")])
    assert_equal(ok.item(250_000, "s").int16(), 300)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
