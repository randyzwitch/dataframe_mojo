"""Typed fixtures and scalar row-group expectations for public reductions.

Codes are independently chosen numeric values and ordering/equality ranks. No
production aggregate, sorting, grouping, or dataframe equality forms an oracle.
"""
from std.math import abs, sqrt, isnan
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Expr, Series, AnyValue, col
from tests.support.numeric_model import INTEGER_DTYPES, FLOAT_DTYPES
from tests.support.coverage_compare import assert_value


def typed_reduction_columns(valid: List[Bool]) raises -> List[Series]:
    var result = List[Series]()
    comptime for i in range(len(INTEGER_DTYPES)):
        comptime D = INTEGER_DTYPES[i]
        result.append(
            Series("x", Column[Scalar[D]]([3, 1, 3, 2, 1, 9], valid.copy()))
        )
    comptime for i in range(len(FLOAT_DTYPES)):
        comptime D = FLOAT_DTYPES[i]
        result.append(
            Series("x", Column[Scalar[D]]([3, 1, 3, 2, 1, 9], valid.copy()))
        )
    result.append(
        Series(
            "x",
            Column[Bool]([True, False, True, True, False, True], valid.copy()),
        )
    )
    var strings = Series(
        "x", Column[String](["c", "a", "c", "b", "a", "z"], valid.copy())
    )
    result.append(strings.copy())
    result.append(strings.cast(DataType.CATEGORICAL))
    var binary = Series.binary(
        "x", [[255], [0], [255], [128], [0], [255, 0]], valid
    )
    result.append(binary.copy())
    result.append(binary.with_dtype(DataType.geometry('{"crs":"EPSG:4326"}')))
    var ticks = Series("x", Column[Int64]([3, 1, 3, 2, 1, 9], valid.copy()))
    for dtype in [
        DataType.DATE,
        DataType.TIME,
        DataType.datetime("ns", "UTC"),
        DataType.datetime("us"),
        DataType.datetime("ms", "America/New_York"),
        DataType.duration("ns"),
        DataType.duration("us"),
        DataType.duration("ms"),
    ]:
        result.append(ticks.with_dtype(dtype))
    for width in [32, 64, 128]:
        result.append(
            Series("x", Column[Int128]([3, 1, 3, 2, 1, 9], valid.copy()))
            .with_dtype(DataType.decimal(8, 2))
            .cast(DataType.decimal(8, 2, width))
        )
    return result^


def fixture_codes(dtype: DataType) -> List[Int64]:
    return [1, 0, 1, 1, 0, 1] if dtype == DataType.BOOL else [3, 1, 3, 2, 1, 9]


def reduction_validity(mode: Int) -> List[Bool]:
    return [True, False, True, True, True, False] if mode == 1 else List[Bool](
        length=6, fill=mode == 0
    )


def typed_scalar_reductions(
    dtype: DataType, include_count: Bool = True
) -> List[Expr]:
    var result: List[Expr] = [
        col("x").null_count().alias("nulls"),
        col("x").len().alias("len"),
        col("x").min().alias("min"),
        col("x").max().alias("max"),
        col("x").n_unique().alias("unique"),
    ]
    if include_count:
        result.append(col("x").count().alias("count"))
    if not dtype.is_decimal():
        result.append(col("x").arg_min().alias("arg_min"))
        result.append(col("x").arg_max().alias("arg_max"))
    if dtype.is_numeric() or dtype.is_duration():
        result.append(col("x").sum().alias("sum"))
        result.append(col("x").sum(min_count=3).alias("sum3"))
    if dtype.is_numeric():
        result.append(col("x").mean().alias("mean"))
    # Decimal mean/sum/extrema are implemented; the binder also advertises
    # other numeric statistics, whose Int128 runtime path currently aborts.
    # Fatal obligations are verified by isolated drivers, outside this matrix.
    if dtype.is_numeric() and not dtype.is_decimal():
        for ddof in [0, 1, 6]:
            result.append(col("x").var(ddof).alias("var" + String(ddof)))
            result.append(col("x").std(ddof).alias("std" + String(ddof)))
        if dtype != DataType.UINT64:
            result.append(col("x").median().alias("median"))
            for method in ["linear", "lower", "higher", "nearest", "midpoint"]:
                result.append(col("x").quantile(0.25, method).alias(method))
        for bias in [False, True]:
            result.append(col("x").skew(bias).alias("skew" + String(Int(bias))))
            for fisher in [False, True]:
                result.append(
                    col("x")
                    .kurtosis(fisher, bias)
                    .alias("kurt" + String(Int(bias)) + String(Int(fisher)))
                )
    if dtype == DataType.BOOL:
        for ignore in [False, True]:
            result.append(
                col("x").any(ignore).alias("any" + String(Int(ignore)))
            )
            result.append(
                col("x").all(ignore).alias("all" + String(Int(ignore)))
            )
    return result^


def near(actual: Float64, expected: Float64, context: String = "") raises:
    # Fixtures are bounded by 9 and n<=6: independent two-pass centered
    # moments vs online algorithms receive 2e-12 absolute+relative budget.
    assert_true(
        abs(actual - expected) <= 2e-12 * (1 + abs(expected)),
        "typed reduction numeric mismatch "
        + context
        + ": observed "
        + String(actual)
        + ", expected "
        + String(expected),
    )


def assert_typed_reductions(
    actual: DataFrame,
    output: Int,
    source: Series,
    valid: List[Bool],
    rows: List[Int],
    include_count: Bool = True,
) raises:
    var codes = fixture_codes(source.dtype())
    var sorted = List[Int64]()
    var count = 0
    var total = Int128(0)
    var min_row = -1
    var max_row = -1
    var min_index = -1
    var max_index = -1
    var seen = List[Int64]()
    var saw_null = False
    for local in range(len(rows)):
        var row = rows[local]
        if not valid[row]:
            saw_null = True
            continue
        count += 1
        total += Int128(codes[row])
        var position = len(sorted)
        sorted.append(codes[row])
        while position > 0 and sorted[position - 1] > sorted[position]:
            var old = sorted[position - 1]
            sorted[position - 1] = sorted[position]
            sorted[position] = old
            position -= 1
        var duplicate = False
        for code in seen:
            duplicate = duplicate or code == codes[row]
        if not duplicate:
            seen.append(codes[row])
        if min_row < 0 or codes[row] < codes[min_row]:
            min_row = row
            min_index = local
        if max_row < 0 or codes[row] > codes[max_row]:
            max_row = row
            max_index = local
    for name in ["count", "nulls", "len", "unique"]:
        if name == "count" and not include_count:
            continue
        assert_true(actual.column(name).dtype() == DataType.INT64)
    if include_count:
        assert_equal(actual.item(output, "count").int64(), Int64(count))
    assert_equal(actual.item(output, "nulls").int64(), Int64(len(rows) - count))
    assert_equal(actual.item(output, "len").int64(), Int64(len(rows)))
    assert_equal(
        actual.item(output, "unique").int64(), Int64(len(seen) + Int(saw_null))
    )
    for high in [False, True]:
        var name = "max" if high else "min"
        var chosen = max_row if high else min_row
        assert_true(actual.column(name).dtype() == source.dtype())
        assert_equal(actual.item(output, name).is_null(), chosen < 0)
        if chosen >= 0:
            assert_value(actual.item(output, name), source.get(chosen))
        if source.dtype().is_decimal():
            continue
        var arg = "arg_max" if high else "arg_min"
        assert_true(actual.column(arg).dtype() == DataType.UINT32)
        assert_equal(actual.item(output, arg).is_null(), chosen < 0)
        if chosen >= 0:
            assert_equal(
                actual.item(output, arg).numeric[DType.uint32](),
                UInt32(max_index if high else min_index),
            )
    var divisor = 100.0 if source.dtype().is_decimal() else 1.0
    if source.dtype().is_numeric() or source.dtype().is_duration():
        var expected_dtype = source.dtype()
        if expected_dtype.is_integer() and expected_dtype.bit_width() <= 16:
            expected_dtype = DataType.INT64
        if expected_dtype.is_decimal():
            # Any decimal sums to decimal(38) at its scale, as in DuckDB.
            expected_dtype = DataType.decimal(38, expected_dtype.scale())
        for threshold in [0, 3]:
            var name = "sum" if threshold == 0 else "sum3"
            assert_true(actual.column(name).dtype() == expected_dtype)
            assert_equal(actual.item(output, name).is_null(), count < threshold)
            if count >= threshold:
                var expected: Series
                if expected_dtype.is_decimal():
                    expected = (
                        Series("", Column[Int128]([total]))
                        .with_dtype(
                            DataType.decimal(
                                expected_dtype.precision(),
                                expected_dtype.scale(),
                            )
                        )
                        .cast(expected_dtype)
                    )
                else:
                    expected = Series("", Column[Int64]([Int64(total)])).cast(
                        expected_dtype
                    )
                assert_value(actual.item(output, name), expected.get(0))
    if source.dtype() == DataType.BOOL:
        var yes = total > 0
        var no = Int128(count) > total
        for ignore in [False, True]:
            for all in [False, True]:
                var name = ("all" if all else "any") + String(Int(ignore))
                var null = (
                    not ignore and saw_null and (not no if all else not yes)
                )
                assert_true(actual.column(name).dtype() == DataType.BOOL)
                assert_equal(actual.item(output, name).is_null(), null)
                if not null:
                    assert_equal(
                        actual.item(output, name).bool(), not no if all else yes
                    )
    if not source.dtype().is_numeric():
        return
    var mean = Float64(total) / Float64(max(1, count)) / divisor
    var m2 = 0.0
    var m3 = 0.0
    var m4 = 0.0
    for code in sorted:
        var delta = Float64(code) / divisor - mean
        m2 += delta * delta
        m3 += delta * delta * delta
        m4 += delta * delta * delta * delta
    for name in [
        "mean",
        "median",
        "linear",
        "lower",
        "higher",
        "nearest",
        "midpoint",
    ]:
        if name != "mean" and (
            source.dtype().is_decimal() or source.dtype() == DataType.UINT64
        ):
            continue
        assert_true(actual.column(name).dtype() == DataType.FLOAT64)
        assert_equal(actual.item(output, name).is_null(), count == 0)
        if not count:
            continue
        var expected = mean
        if name != "mean":
            var position = (0.5 if name == "median" else 0.25) * Float64(
                count - 1
            )
            var low = Int(position)
            var high = low if position == Float64(low) else low + 1
            var a = Float64(sorted[low]) / divisor
            var b = Float64(sorted[high]) / divisor
            expected = a + (b - a) * (position - Float64(low))
            if name == "lower":
                expected = a
            if name == "higher":
                expected = b
            if name == "nearest":
                expected = Float64(sorted[Int(position + 0.5)]) / divisor
            if name == "midpoint":
                expected = (a + b) / 2
        near(
            actual.item(output, name).float64(),
            expected,
            source.dtype().name() + ":" + name,
        )
    if source.dtype().is_decimal():
        return
    for ddof in [0, 1, 6]:
        for std in [False, True]:
            var name = ("std" if std else "var") + String(ddof)
            assert_true(actual.column(name).dtype() == DataType.FLOAT64)
            assert_equal(actual.item(output, name).is_null(), count <= ddof)
            if count > ddof:
                var variance = m2 / Float64(count - ddof)
                near(
                    actual.item(output, name).float64(),
                    sqrt(variance) if std else variance,
                    source.dtype().name() + ":" + name,
                )
    for bias in [False, True]:
        var name = "skew" + String(Int(bias))
        var null = count == 0 or (not bias and count < 3)
        assert_equal(actual.item(output, name).is_null(), null)
        if not null and m2 == 0:
            assert_true(isnan(actual.item(output, name).float64()))
        if not null and m2 > 0:
            var value = sqrt(Float64(count)) * m3 / (m2 * sqrt(m2))
            if not bias:
                value *= sqrt(Float64(count * (count - 1))) / Float64(count - 2)
            near(
                actual.item(output, name).float64(),
                value,
                source.dtype().name() + ":" + name,
            )
        for fisher in [False, True]:
            name = "kurt" + String(Int(bias)) + String(Int(fisher))
            null = count == 0 or (not bias and count < 4)
            assert_equal(actual.item(output, name).is_null(), null)
            if not null and m2 == 0:
                assert_true(isnan(actual.item(output, name).float64()))
            if not null and m2 > 0:
                var excess = Float64(count) * m4 / (m2 * m2) - 3
                if not bias:
                    excess = (
                        Float64(count - 1)
                        * (Float64(count + 1) * excess + 6)
                        / Float64((count - 2) * (count - 3))
                    )
                near(
                    actual.item(output, name).float64(),
                    excess if fisher else excess + 3,
                    source.dtype().name() + ":" + name,
                )


def fixture_key_matches(
    value: AnyValue, dtype: DataType, code: Int64
) raises -> Bool:
    if dtype.is_temporal():
        return value.to_physical() == code
    if dtype == DataType.BOOL:
        return value.bool() == (code != 0)
    if dtype == DataType.STRING or dtype.is_categorical():
        var text = (
            "a" if code
            == 1 else "b" if code
            == 2 else "c" if code
            == 3 else "z"
        )
        return value.string() == text
    if dtype.is_binary() or dtype.is_geometry():
        var bytes = List[UInt8]()
        if code == 1:
            bytes.append(0)
        elif code == 2:
            bytes.append(128)
        else:
            bytes.append(255)
            if code != 3:
                bytes.append(0)
        return value.bytes() == bytes
    if dtype.is_decimal():
        return String(value) == "0.0" + String(code)
    comptime numeric = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float32,
        DType.float64,
    )
    comptime for i in range(len(numeric)):
        comptime D = numeric[i]
        if dtype == DataType.of(D):
            return value.numeric[D]() == Scalar[D](code)
    raise Error("Unmodeled fixture key dtype")


def histogram(
    source: Series, valid: List[Bool], rows: List[Int]
) -> Tuple[List[Int], List[Int]]:
    var codes = fixture_codes(source.dtype())
    var positions = List[Int]()
    var counts = List[Int]()
    for row in rows:
        var found = -1
        for i in range(len(positions)):
            var first = positions[i]
            if valid[row] == valid[first] and (
                not valid[row] or codes[row] == codes[first]
            ):
                found = i
        if found >= 0:
            counts[found] += 1
            continue
        var place = len(positions)
        positions.append(row)
        counts.append(1)
        while place > 0:
            var previous = positions[place - 1]
            if not valid[row] or (
                valid[previous] and codes[previous] <= codes[row]
            ):
                break
            positions[place] = previous
            counts[place] = counts[place - 1]
            positions[place - 1] = row
            counts[place - 1] = 1
            place -= 1
    return (positions^, counts^)


def assert_distribution(
    mode: Series,
    counts: Series,
    source: Series,
    valid: List[Bool],
    rows: List[Int],
    sorted_counts: Bool,
    normalized: Bool,
) raises:
    var model = histogram(source, valid, rows)
    var positions = model[0].copy()
    var frequencies = model[1].copy()
    var maximum = 0
    for frequency in frequencies:
        maximum = max(maximum, frequency)
    var mode_positions = List[Int]()
    for i in range(len(positions)):
        if frequencies[i] == maximum:
            mode_positions.append(positions[i])
    assert_true(mode.dtype() == source.dtype())
    assert_equal(len(mode), len(mode_positions))
    for row in range(len(mode_positions)):
        assert_value(mode.get(row), source.get(mode_positions[row]))
    if sorted_counts:
        for i in range(1, len(positions)):
            var place = i
            while place > 0 and frequencies[place] > frequencies[place - 1]:
                var p = positions[place - 1]
                var f = frequencies[place - 1]
                positions[place - 1] = positions[place]
                frequencies[place - 1] = frequencies[place]
                positions[place] = p
                frequencies[place] = f
                place -= 1
    var count_name = "proportion" if normalized else "count"
    var dtype = DataType.struct(
        ["x", count_name],
        [source.dtype(), DataType.FLOAT64 if normalized else DataType.UINT32],
    )
    assert_true(counts.dtype() == dtype)
    assert_equal(len(counts), len(positions))
    for row in range(len(positions)):
        var record = counts.get(row)
        assert_true(not record.is_null())
        assert_value(record.struct_field("x"), source.get(positions[row]))
        var frequency = record.struct_field(count_name)
        assert_true(not frequency.is_null())
        if normalized:
            near(
                frequency.float64(),
                Float64(frequencies[row]) / Float64(len(rows)),
            )
        else:
            assert_equal(
                frequency.numeric[DType.uint32](), UInt32(frequencies[row])
            )
