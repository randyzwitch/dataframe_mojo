"""Capture identified, independently checked placement samples as JSONL.

Run with the packaged provider being calibrated. Optional arguments: number
of repetitions (>=3), then three or more row counts. Defaults: five repeats
at 50M, 75M and 100M rows. Progress/errors go to stderr via the runtime;
stdout contains only capture records for scripts/metal_calibration.py.
"""
from std.sys import argv
from dataframe import Column, DataFrame, Expr, Series, col
from dataframe_metal.calibration_capture import capture


def _table(rows: Int, nulls: Bool) raises -> DataFrame:
    var a = List[Float32](capacity=rows)
    var b = List[Float32](capacity=rows)
    var d = List[Float32](capacity=rows)
    var i = List[Int32](capacity=rows)
    var p = List[Int32](capacity=rows)
    var av = List[Bool](capacity=rows if nulls else 0)
    var bv = List[Bool](capacity=rows if nulls else 0)
    var dv = List[Bool](capacity=rows if nulls else 0)
    var pv = List[Bool](capacity=rows if nulls else 0)
    for row in range(rows):
        a.append(Float32(row % 16 - 8) / 4)
        b.append(Float32((row * 3) % 8 - 4) / 2)
        d.append(Float32((row * 5) % 32 - 16) / 8)
        i.append(Int32(row % 16 - 8))
        p.append(Int32(row % 4))
        if nulls:
            av.append(row % 8 != 1)
            bv.append(row % 16 != 3)
            dv.append(row % 32 != 7)
            pv.append(row % 16 != 5)
    if nulls:
        return DataFrame(
            [
                Series("a", Column[Float32](a^, av)),
                Series("b", Column[Float32](b^, bv)),
                Series("d", Column[Float32](d^, dv)),
                Series("i", Column[Int32](i^, av)),
                Series("p", Column[Int32](p^, pv)),
            ]
        )
    return DataFrame(
        [
            Series("a", Column[Float32](a^)),
            Series("b", Column[Float32](b^)),
            Series("d", Column[Float32](d^)),
            Series("i", Column[Int32](i^)),
            Series("p", Column[Int32](p^)),
        ]
    )


def _chunks(frame: DataFrame, count: Int) raises -> DataFrame:
    if count == 1:
        return frame.copy()
    var columns = List[Series]()
    for name in frame.columns():
        var parts = List[Series]()
        for i in range(count):
            var first = frame.height() * i // count
            var last = frame.height() * (i + 1) // count
            parts.append(frame.column(name).slice(first, last - first))
        columns.append(Series._from_chunks(parts))
    return DataFrame(columns^)


def _reference(rows: Int, nulls: Bool, mode: Int) raises -> DataFrame:
    # A period covers all values, predicates and independent null masks.
    var integer = Int64(0)
    var count = Int64(0)
    var sales = Float64(0)
    var discount = Float64(0)
    var quantity_count = Int64(0)
    var sales_count = Int64(0)
    var low = Int32.MAX
    var high = Int32.MIN
    for row in range(32):
        var weight = Int64(rows // 32 + Int(row < rows % 32))
        if row % 4 == 0 or (nulls and row % 16 == 5):
            continue
        if not nulls or row % 8 != 1:
            if weight:
                low = min(low, Int32(row % 16 - 8))
                high = max(high, Int32(row % 16 - 8))
            integer += Int64(row % 16 - 8) * weight
            count += weight
            if not nulls or row % 16 != 3:
                sales_count += weight
                sales += (
                    (Float64(row % 16 - 8) / 4)
                    * (Float64((row * 3) % 8 - 4) / 2)
                    * Float64(weight)
                )
        if not nulls or row % 32 != 7:
            discount += Float64((row * 5) % 32 - 16) / 8 * Float64(weight)
        if not nulls or row % 16 != 3:
            quantity_count += weight
    if mode == 5:
        return DataFrame(
            [
                Series("sum", Column[Int32]([Int32(integer)])),
                Series("count", Column[Int64]([count])),
                Series(
                    "mean", Column[Float64]([Float64(integer) / Float64(count)])
                ),
                Series("min", Column[Int32]([low])),
                Series("max", Column[Int32]([high])),
            ]
        )
    if mode == 6:
        return DataFrame(
            [
                Series("sum", Column[Float32]([Float32(Float64(integer) / 4)])),
                Series("count", Column[Int64]([count])),
                Series(
                    "mean",
                    Column[Float64]([Float64(integer) / Float64(4 * count)]),
                ),
                Series("min", Column[Float32]([Float32(low) / 4])),
                Series("max", Column[Float32]([Float32(high) / 4])),
            ]
        )
    if mode == 7:
        return DataFrame(
            [
                Series("min", Column[Int32]([low])),
                Series("max", Column[Float32]([Float32(high) / 4])),
                Series("sales", Column[Float32]([Float32(sales)])),
                Series("count", Column[Int64]([sales_count])),
                Series("mean", Column[Float64]([sales / Float64(sales_count)])),
            ]
        )
    if mode == 0:
        return DataFrame([Series("sum", Column[Int32]([Int32(integer)]))])
    if mode == 1:
        return DataFrame(
            [
                Series("sum", Column[Int32]([Int32(integer)])),
                Series("count", Column[Int64]([count])),
                Series(
                    "mean", Column[Float64]([Float64(integer) / Float64(count)])
                ),
            ]
        )
    if mode == 3:
        return DataFrame(
            [
                Series("integer", Column[Int32]([Int32(integer)])),
                Series("sum", Column[Float32]([Float32(Float64(integer) / 4)])),
                Series("count", Column[Int64]([count])),
                Series(
                    "mean",
                    Column[Float64]([Float64(integer) / Float64(4 * count)]),
                ),
            ]
        )
    if mode == 4:
        return DataFrame(
            [
                Series("sales", Column[Float32]([Float32(sales)])),
                Series("count", Column[Int64]([sales_count])),
                Series("mean", Column[Float64]([sales / Float64(sales_count)])),
                Series("discount", Column[Float32]([Float32(discount)])),
            ]
        )
    return DataFrame(
        [
            Series("sales", Column[Float32]([Float32(sales)])),
            Series("discount", Column[Float32]([Float32(discount)])),
            Series("count", Column[Int64]([quantity_count])),
        ]
    )


def _expressions(mode: Int) -> List[Expr]:
    if mode == 5 or mode == 6:
        var value = col("i" if mode == 5 else "a")
        return [
            value.sum().alias("sum"),
            value.count().alias("count"),
            value.mean().alias("mean"),
            value.min().alias("min"),
            value.max().alias("max"),
        ]
    if mode == 7:
        var value = col("a") * col("b")
        return [
            col("i").min().alias("min"),
            col("a").max().alias("max"),
            value.sum().alias("sales"),
            value.count().alias("count"),
            value.mean().alias("mean"),
        ]
    if mode == 0:
        return [col("i").sum().alias("sum")]
    if mode == 1:
        return [
            col("i").sum().alias("sum"),
            col("i").count().alias("count"),
            col("i").mean().alias("mean"),
        ]
    if mode == 3:
        return [
            col("i").sum().alias("integer"),
            col("a").sum().alias("sum"),
            col("a").count().alias("count"),
            col("a").mean().alias("mean"),
        ]
    if mode == 4:
        var value = col("a") * col("b")
        return [
            value.sum().alias("sales"),
            value.count().alias("count"),
            value.mean().alias("mean"),
            col("d").sum().alias("discount"),
        ]
    return [
        (col("a") * col("b")).sum().alias("sales"),
        col("d").sum().alias("discount"),
        col("b").count().alias("count"),
    ]


def main() raises:
    var args = argv()
    var repetitions = Int(String(args[1])) if len(args) > 1 else 5
    var sizes = List[Int]()
    for i in range(2, len(args)):
        sizes.append(Int(String(args[i])))
    if not sizes:
        sizes = [50_000_000, 75_000_000, 100_000_000]
    if len(sizes) < 3 or repetitions < 3:
        raise Error("Measure at least three sizes and three repetitions")
    for rows in sizes:
        if rows < 32 or rows > 100_000_000:
            raise Error(
                "Built-in calibration corpus supports 32 through 100M rows; use capture() for other workloads"
            )
        for nulls in [False, True]:
            var base = _table(rows, nulls)
            for chunks in [1, 8]:
                var frame = _chunks(base, chunks)
                for mode in range(8):
                    var expressions = _expressions(mode)
                    var plan = (
                        frame.lazy()
                        .filter(col("p") > 0)
                        .select_exprs(expressions)
                    )
                    print(
                        capture(
                            plan,
                            _reference(rows, nulls, mode),
                            repetitions,
                            Bool((mode + chunks) % 2),
                        )
                    )
