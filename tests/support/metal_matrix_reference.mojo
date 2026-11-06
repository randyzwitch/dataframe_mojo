"""Independent small scalar oracle for shared numeric expression reductions."""
from dataframe import Column, DataFrame, Expr, Series, col


def _value(column: Int, row: Int) -> Int64:
    if column == 0:
        return Int64(row % 5 + 1)
    if column == 1:
        return Int64(row % 3 + 2)
    if column == 2:
        return (Int64(1) << 32) + Int64(row * 2 + 5)
    if column == 3:
        return (Int64(1) << 32) + Int64(row + 1)
    if column == 4:
        return Int64(row % 4 + 1)
    if column == 5:
        return Int64(row % 3 + 1)
    if column == 6:
        return Int64(row % 2 + 1)
    if column == 7:
        return Int64(row % 5 + 2)
    return Int64(row % 3 - 1) if column == 8 else Int64(row % 4)


def _present(column: Int, row: Int) -> Bool:
    if column == 0:
        return row % 7 != 1
    if column == 1:
        return row % 5 != 2
    if column == 2:
        return row % 11 != 1
    if column == 3:
        return row % 3 != 0
    if column == 4:
        return row % 4 != 0
    if column == 5:
        return row % 6 != 1
    if column == 6:
        return row % 9 != 3
    if column == 7:
        return row % 5 != 1
    return row % 5 != 0 if column == 8 else row % 7 != 0


def matrix_frame(fragmented: Bool) raises -> DataFrame:
    var names = List[String](["a", "b", "l", "m", "f", "g", "h", "j", "p", "k"])
    var columns = List[Series]()
    for column in range(10):
        var parts = List[Series]()
        for part in range(2 if fragmented else 1):
            var split = 3 + column * 2
            var first = split if part == 1 else 0
            var last = split if fragmented and part == 0 else 34
            var values = List[Int64]()
            var valid = List[Bool]()
            for physical in range(first, last):
                values.append(_value(column, physical - 1))
                valid.append(_present(column, physical - 1))
            if column < 2 or column == 8:
                var narrow = List[Int32]()
                for value in values:
                    narrow.append(Int32(value))
                parts.append(
                    Series(names[column], Column[Int32](narrow^, valid))
                )
            elif column < 4 or column == 9:
                parts.append(
                    Series(names[column], Column[Int64](values^, valid))
                )
            else:
                var narrow = List[Float32]()
                for value in values:
                    narrow.append(Float32(value))
                parts.append(
                    Series(names[column], Column[Float32](narrow^, valid))
                )
        columns.append(Series._from_chunks(parts))
    return DataFrame(columns^).slice(1, 32)


def matrix_expressions() -> List[Expr]:
    var values = List[Expr](
        [
            col("a") * col("b") + col("a"),
            col("l") - col("m") + col("l"),
            col("f") * col("g") + 1,
            (col("h") + col("j")) * col("h"),
        ]
    )
    var expressions = List[Expr]()
    for i in range(4):
        var suffix = String(i)
        expressions.append(
            values[i].sum(min_count=2 if i == 3 else 0).alias("s" + suffix)
        )
        expressions.append(values[i].count().alias("n" + suffix))
        expressions.append(values[i].mean().alias("avg" + suffix))
        expressions.append(values[i].min().alias("lo" + suffix))
        expressions.append(values[i].max().alias("hi" + suffix))
    return expressions^


def _typed(
    field: Int, name: String, values: List[Float64], valid: List[Bool]
) raises -> Series:
    if field == 0:
        var narrow = List[Int32]()
        for value in values:
            narrow.append(Int32(value))
        return Series(name, Column[Int32](narrow^, valid))
    if field == 1:
        var narrow = List[Int64]()
        for value in values:
            narrow.append(Int64(value))
        return Series(name, Column[Int64](narrow^, valid))
    var narrow = List[Float32]()
    for value in values:
        narrow.append(Float32(value))
    return Series(name, Column[Float32](narrow^, valid))


def matrix_reference(grouped: Bool) raises -> DataFrame:
    var sums = List[Float64](length=20, fill=0)
    var counts = List[Int64](length=20, fill=0)
    var low = List[Float64](length=20, fill=Float64.MAX_FINITE)
    var high = List[Float64](length=20, fill=-Float64.MAX_FINITE)
    var seen = List[Bool](length=5, fill=False)
    var order = List[Int]()
    for row in range(32):
        # Independent scalar selection, including a null predicate branch.
        if _present(8, row) and _value(8, row) <= 0:
            continue
        var group = (row % 4 if _present(9, row) else 4) if grouped else 0
        if not seen[group]:
            seen[group] = True
            order.append(group)
        for field in range(4):
            if not _present(field * 2, row) or not _present(field * 2 + 1, row):
                continue
            var a = _value(field * 2, row)
            var b = _value(field * 2 + 1, row)
            # All fixture values/results are exactly representable in their
            # native dtype and Float64, independently of GPU reassociation.
            var value = Float64(
                a * b + a if field
                == 0 else a - b + a if field
                == 1 else a * b + 1 if field
                == 2 else (a + b) * a
            )
            var at = group * 4 + field
            sums[at] += value
            counts[at] += 1
            low[at] = min(low[at], value)
            high[at] = max(high[at], value)
    var columns = List[Series]()
    if grouped:
        var keys = List[Int64]()
        var valid = List[Bool]()
        for group in order:
            keys.append(Int64(0 if group == 4 else group))
            valid.append(group != 4)
        columns.append(Series("k", Column[Int64](keys^, valid)))
    for field in range(4):
        var totals = List[Float64]()
        var ns = List[Int64]()
        var means = List[Float64]()
        var minimum = List[Float64]()
        var maximum = List[Float64]()
        var sum_valid = List[Bool]()
        var valid = List[Bool]()
        for group in order:
            var at = group * 4 + field
            totals.append(sums[at])
            ns.append(counts[at])
            means.append(
                sums[at] / Float64(counts[at]) if counts[at] else Float64(0)
            )
            minimum.append(low[at] if counts[at] else Float64(0))
            maximum.append(high[at] if counts[at] else Float64(0))
            sum_valid.append(counts[at] >= 2 if field == 3 else True)
            valid.append(counts[at] > 0)
        var suffix = String(field)
        columns.append(_typed(field, "s" + suffix, totals, sum_valid))
        columns.append(Series("n" + suffix, Column[Int64](ns^)))
        columns.append(Series("avg" + suffix, Column[Float64](means^, valid)))
        columns.append(_typed(field, "lo" + suffix, minimum, valid))
        columns.append(_typed(field, "hi" + suffix, maximum, valid))
    return DataFrame(columns^)
