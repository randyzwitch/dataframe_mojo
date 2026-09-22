"""Deterministic baseline/candidate dump for direct numeric reductions.

Compile this file with either source tree on `-I`; compare stdout byte for
byte. Values use small binary integers to avoid harmless Float64 sum-order
differences. The explicit IEEE and overflow cases follow afterwards.
"""
from dataframe import Column, DataFrame, Expr, Series, col


def emit(label: String, name: String, frame: DataFrame, expr: Expr) raises:
    try:
        print(label, ",", name, ",", frame.select(expr).item(), sep="")
    except e:
        print(label, ",", name, ",ERR,", String(e), sep="")


def dump[D: DType](label: String, offset: Int, tail: Int) raises:
    var values = List[Scalar[D]](capacity=25)
    var valid = List[Bool](capacity=25)
    for i in range(25):
        values.append(Int64((i % 5) - 2).cast[D]())
        valid.append((i % 3) != 0)
    var frame = DataFrame([Series("x", Column[Scalar[D]](values^, valid^))])
    var sliced = frame.slice(offset, tail)
    var tag = label + ":o" + String(offset) + ":n" + String(tail)
    emit(tag, "sum", sliced, col("x").sum())
    emit(tag, "mean", sliced, col("x").mean())
    emit(tag, "count", sliced, col("x").count())
    emit(tag, "min", sliced, col("x").min())
    emit(tag, "max", sliced, col("x").max())


def dump_all_slices[D: DType](label: String) raises:
    for offset in range(8):
        for tail in range(18):
            dump[D](label, offset, tail)


def dump_float_ieee[D: DType](label: String) raises:
    var nan = Float64(0) / Float64(0)
    var values = List[Scalar[D]](
        [
            Float64(-0.0).cast[D](),
            Float64(0.0).cast[D](),
            nan.cast[D](),
            Float64(-3).cast[D](),
            Float64(3).cast[D](),
            (Float64(1) / Float64(0)).cast[D](),
            (-Float64(1) / Float64(0)).cast[D](),
        ]
    )
    var valid = List[Bool]([True, True, True, True, False, True, True])
    var frame = DataFrame([Series("x", Column[Scalar[D]](values^, valid^))])
    emit(label, "sum", frame, col("x").sum())
    emit(label, "mean", frame, col("x").mean())
    emit(label, "count", frame, col("x").count())
    emit(label, "min", frame, col("x").min())
    emit(label, "max", frame, col("x").max())


def dump_overflow() raises:
    var i64 = DataFrame(
        [Series("x", Column[Int64]([Int64.MAX, Int64.MAX, Int64.MIN]))]
    )
    emit("i64-overflow", "sum", i64, col("x").sum())
    emit("i64-overflow", "mean", i64, col("x").mean())
    var u64 = DataFrame(
        [Series("x", Column[UInt64]([UInt64.MAX, UInt64.MAX, UInt64(1)]))]
    )
    emit("u64-overflow", "sum", u64, col("x").sum())
    emit("u64-overflow", "mean", u64, col("x").mean())
    emit("u64-overflow", "min", u64, col("x").min())
    emit("u64-overflow", "max", u64, col("x").max())


def main() raises:
    dump_all_slices[DType.int8]("i8")
    dump_all_slices[DType.uint8]("u8")
    dump_all_slices[DType.int16]("i16")
    dump_all_slices[DType.uint16]("u16")
    dump_all_slices[DType.int32]("i32")
    dump_all_slices[DType.uint32]("u32")
    dump_all_slices[DType.int64]("i64")
    dump_all_slices[DType.uint64]("u64")
    dump_all_slices[DType.float32]("f32")
    dump_all_slices[DType.float64]("f64")
    dump_float_ieee[DType.float32]("f32-ieee")
    dump_float_ieee[DType.float64]("f64-ieee")
    dump_overflow()
