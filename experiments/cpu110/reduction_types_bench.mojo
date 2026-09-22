"""Bare-column reductions on every numeric dtype, with sliced nullable input."""
from std.sys import argv
from std.time import monotonic
from dataframe import Column, DataFrame, Expr, Series, col


def measure(frame: DataFrame, dtype: String, name: String, expr: Expr) raises:
    var expected = frame.select(expr).item()
    var best = Int.MAX
    for _ in range(5):
        var start = monotonic()
        var actual = frame.select(expr).item()
        var elapsed = monotonic() - start
        if actual != expected:
            raise Error("reduction changed across repetitions")
        best = min(best, elapsed)
    print(dtype, name, best, expected, sep=",")


def run[D: DType](label: String, rows: Int) raises:
    var values = List[Scalar[D]](capacity=rows + 3)
    var valid = List[Bool](capacity=rows + 3)
    for i in range(rows + 3):
        values.append(Int64(i % 100).cast[D]())
        valid.append(i % 7 != 0)
    var frame = DataFrame(
        [Series("x", Column[Scalar[D]](values^, valid^).slice(3, rows))]
    )
    measure(frame, label, "sum", col("x").sum())
    measure(frame, label, "mean", col("x").mean())
    measure(frame, label, "count", col("x").count())
    measure(frame, label, "min", col("x").min())
    measure(frame, label, "max", col("x").max())


def main() raises:
    var args = argv()
    var rows = Int(String(args[1]))
    run[DType.int8]("i8", rows)
    run[DType.uint8]("u8", rows)
    run[DType.int16]("i16", rows)
    run[DType.uint16]("u16", rows)
    run[DType.int32]("i32", rows)
    run[DType.uint32]("u32", rows)
    run[DType.int64]("i64", rows)
    run[DType.uint64]("u64", rows)
    run[DType.float32]("f32", rows)
    run[DType.float64]("f64", rows)
