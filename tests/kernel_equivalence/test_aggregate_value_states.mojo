"""The aggregate table's value states against a scalar model of each group:
first and last (by row order) of strings, integers, floats, Booleans and
decimals, nulls included; string, Boolean and decimal minima and maxima;
any and all; and counts of string and Boolean values. Both routes that
reach the table are checked (the eager group-by, and a pipeline's grouped
sink after a filter), with tables that clear (eight workers), grow (two)
and a single worker."""
# contract-group: reduce.grouped
# contract-group: kernel.reduce
from std.testing import assert_equal, assert_true

from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    lit,
)
from dataframe.value import AnyValue
from tests.support.group_strategy_reference import (
    ScopedEnvironment,
    StrategyTrace,
)


def group_of(row: Int, groups: Int) -> Int:
    return (row * 7919) % groups


def text_valid(row: Int) -> Bool:
    return row % 11 != 3


def text(row: Int) -> String:
    if row % 3 == 0:
        return "v" + String(row % 97)
    if row % 3 == 1:
        return "a value longer than twelve bytes " + String(row % 89)
    return "é" + String(row % 53)


def int_valid(row: Int) -> Bool:
    return row % 7 != 5


def int_value(row: Int) -> Int32:
    return Int32((row * 104729) % 1999 - 1000)


def real_valid(row: Int) -> Bool:
    return row % 13 != 1


def real(row: Int) -> Float64:
    if row % 17 == 4:
        return -0.0
    return Float64(row % 211) / 8 - 13


def flag_valid(row: Int) -> Bool:
    return row % 5 != 2


def flag(row: Int) -> Bool:
    return (row * 31) % 7 < 3


def decimal_valid(row: Int) -> Bool:
    return row % 9 != 6


def decimal(row: Int) -> Int64:
    return Int64((row * 7907) % 200_001 - 100_000)


def frame(rows: Int, groups: Int) raises -> DataFrame:
    var k = List[Int64]()
    var t = List[String]()
    var tv = List[Bool]()
    var x = List[Int32]()
    var xv = List[Bool]()
    var y = List[Float64]()
    var yv = List[Bool]()
    var b = List[Bool]()
    var bv = List[Bool]()
    var d = List[Int64]()
    var dv = List[Bool]()
    var w = List[Int128]()
    var r = List[Int64]()
    for row in range(rows):
        k.append(Int64(group_of(row, groups)) * 1_000_003)
        t.append(text(row) if text_valid(row) else String(""))
        tv.append(text_valid(row))
        x.append(int_value(row))
        xv.append(int_valid(row))
        y.append(real(row))
        yv.append(real_valid(row))
        b.append(flag(row))
        bv.append(flag_valid(row))
        d.append(decimal(row))
        dv.append(decimal_valid(row))
        w.append(Int128(decimal(row)) * 1_000_000_000_000)
        r.append(Int64(row))
    return DataFrame(
        [
            Series("k", Column[Int64](k^)),
            Series("t", Column[String](t^, tv)),
            Series("x", Column[Int32](x^, xv)),
            Series("y", Column[Float64](y^, yv)),
            Series("b", BoolColumn(b, bv)),
            Series("d", Column[Int64](d^, dv)).with_dtype(
                DataType.decimal(15, 2, 64)
            ),
            Series("w", Column[Int128](w^, dv)).with_dtype(
                DataType.decimal(30, 4)
            ),
            Series("r", Column[Int64](r^)),
        ]
    )


def reductions() -> List[Expr]:
    return [
        col("t").first().alias("t_first"),
        col("t").last().alias("t_last"),
        col("t").min().alias("t_min"),
        col("t").max().alias("t_max"),
        col("t").count().alias("t_count"),
        col("x").first().alias("x_first"),
        col("x").last().alias("x_last"),
        col("y").first().alias("y_first"),
        col("y").last().alias("y_last"),
        col("b").first().alias("b_first"),
        col("b").last().alias("b_last"),
        col("b").min().alias("b_min"),
        col("b").max().alias("b_max"),
        col("b").any().alias("b_any"),
        col("b").all(ignore_nulls=False).alias("b_all"),
        col("b").null_count().alias("b_nulls"),
        col("d").min().alias("d_min"),
        col("d").max().alias("d_max"),
        col("d").first().alias("d_first"),
        col("w").min().alias("w_min"),
        col("w").max().alias("w_max"),
    ]


struct Model(Copyable, Movable):
    var rows: List[Int]

    def __init__(out self):
        self.rows = List[Int]()


def models(rows: Int, groups: Int, start: Int) -> List[Model]:
    var out = List[Model](length=groups, fill=Model())
    for row in range(start, rows):
        out[group_of(row, groups)].rows.append(row)
    return out^


def assert_text(value: AnyValue, valid: Bool, expected: String, what: String) raises:
    if not valid:
        assert_true(value.is_null(), what + " should be null")
        return
    assert_true(not value.is_null(), what + " should not be null")
    assert_equal(value.string(), expected, what)


def assert_bool(value: AnyValue, valid: Bool, expected: Bool, what: String) raises:
    if not valid:
        assert_true(value.is_null(), what + " should be null")
        return
    assert_true(not value.is_null(), what + " should not be null")
    assert_equal(value.bool(), expected, what)


def unscaled(column: Series, row: Int) raises -> Int128:
    var flat = column.rechunk() if column.is_chunked() else column.copy()
    if flat._data.isa[Column[Int64]]():
        return Int128(flat._data[Column[Int64]]._get(row))
    return flat._data[Column[Int128]]._get(row)


def check(given: DataFrame, rows: Int, groups: Int, start: Int) raises:
    # One chunk, so reading a cell does not walk every chunk.
    var actual = given.rechunk()
    var expected = models(rows, groups, start)
    var present = 0
    for g in range(groups):
        if len(expected[g].rows) > 0:
            present += 1
    assert_equal(actual.height(), present)
    var d_min = actual.column("d_min")
    var d_max = actual.column("d_max")
    var d_first = actual.column("d_first")
    var w_min = actual.column("w_min")
    var w_max = actual.column("w_max")
    assert_true(d_min.dtype() == DataType.decimal(15, 2, 64), "d_min type")
    assert_true(w_max.dtype() == DataType.decimal(30, 4), "w_max type")
    for r in range(actual.height()):
        var g = Int(actual.item(r, "k").int64() // 1_000_003)
        ref members = expected[g].rows
        assert_true(len(members) > 0, "unexpected group")
        var first = members[0]
        var last = members[len(members) - 1]
        var name = " of group " + String(g)
        assert_text(
            actual.item(r, "t_first"), text_valid(first), text(first), "t_first" + name
        )
        assert_text(
            actual.item(r, "t_last"), text_valid(last), text(last), "t_last" + name
        )
        var low = String("")
        var high = String("")
        var texts = 0
        var bools = 0
        var any_true = False
        var any_false = False
        var nulls = 0
        var d_low = Int128.MAX
        var d_high = Int128.MIN
        var decimals = 0
        for row in members:
            if text_valid(row):
                var value = text(row)
                if texts == 0 or value < low:
                    low = value
                if texts == 0 or value > high:
                    high = value
                texts += 1
            if flag_valid(row):
                bools += 1
                if flag(row):
                    any_true = True
                else:
                    any_false = True
            else:
                nulls += 1
            if decimal_valid(row):
                decimals += 1
                d_low = min(d_low, Int128(decimal(row)))
                d_high = max(d_high, Int128(decimal(row)))
        assert_text(actual.item(r, "t_min"), texts > 0, low, "t_min" + name)
        assert_text(actual.item(r, "t_max"), texts > 0, high, "t_max" + name)
        assert_equal(actual.item(r, "t_count").int64(), Int64(texts))
        var xf = actual.item(r, "x_first")
        if int_valid(first):
            assert_equal(xf.int32(), int_value(first), "x_first" + name)
        else:
            assert_true(xf.is_null(), "x_first null" + name)
        var xl = actual.item(r, "x_last")
        if int_valid(last):
            assert_equal(xl.int32(), int_value(last), "x_last" + name)
        else:
            assert_true(xl.is_null(), "x_last null" + name)
        var yf = actual.item(r, "y_first")
        if real_valid(first):
            assert_equal(yf.float64(), real(first), "y_first" + name)
        else:
            assert_true(yf.is_null(), "y_first null" + name)
        var yl = actual.item(r, "y_last")
        if real_valid(last):
            assert_equal(yl.float64(), real(last), "y_last" + name)
        else:
            assert_true(yl.is_null(), "y_last null" + name)
        assert_bool(
            actual.item(r, "b_first"), flag_valid(first), flag(first), "b_first" + name
        )
        assert_bool(
            actual.item(r, "b_last"), flag_valid(last), flag(last), "b_last" + name
        )
        assert_bool(actual.item(r, "b_min"), bools > 0, not any_false, "b_min" + name)
        assert_bool(actual.item(r, "b_max"), bools > 0, any_true, "b_max" + name)
        assert_bool(actual.item(r, "b_any"), True, any_true, "b_any" + name)
        assert_bool(
            actual.item(r, "b_all"),
            any_false or nulls == 0,
            not any_false,
            "b_all" + name,
        )
        assert_equal(actual.item(r, "b_nulls").int64(), Int64(nulls))
        if decimals == 0:
            assert_true(d_min.get(r).is_null(), "d_min null" + name)
            assert_true(w_max.get(r).is_null(), "w_max null" + name)
        else:
            assert_equal(unscaled(d_min, r), d_low, "d_min" + name)
            assert_equal(unscaled(d_max, r), d_high, "d_max" + name)
            assert_equal(
                unscaled(w_min, r), d_low * 1_000_000_000_000, "w_min" + name
            )
            assert_equal(
                unscaled(w_max, r), d_high * 1_000_000_000_000, "w_max" + name
            )
        if decimal_valid(first):
            assert_equal(unscaled(d_first, r), Int128(decimal(first)))
        else:
            assert_true(d_first.get(r).is_null(), "d_first null" + name)


def eager(threads: Int, rows: Int, groups: Int) raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", String(threads))
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var observed = StrategyTrace()
    var actual = frame(rows, groups).group_by(["k"]).agg(reductions())
    assert_true(
        "dataframe-path: group_by.aggregate_table\n" in observed.text(),
        "value states did not reach the aggregate table",
    )
    _ = observed^
    check(actual, rows, groups, 0)
    _ = trace^
    _ = workers^


def test_eager_value_states_with_tables_that_clear() raises:
    eager(8, 60_000, 8_000)


def test_eager_value_states_with_tables_that_grow() raises:
    eager(2, 60_000, 3_000)


def test_eager_value_states_on_one_worker() raises:
    eager(1, 20_000, 700)


def test_pipeline_value_states_after_a_filter() raises:
    var workers = ScopedEnvironment("DATAFRAME_THREADS", "8")
    var trace = ScopedEnvironment("DATAFRAME_TRACE_PATHS", "1")
    var rows = 120_000
    var groups = 9_000
    var start = 1_234
    var observed = StrategyTrace()
    var actual = (
        frame(rows, groups)
        .lazy()
        .filter(col("r") >= lit(Int64(start)))
        .group_by(["k"])
        .agg(reductions())
        .collect()
    )
    assert_true(
        "dataframe-path: pipeline.group_table\n" in observed.text(),
        "value states did not reach the aggregate table sink",
    )
    _ = observed^
    check(actual, rows, groups, start)
    _ = trace^
    _ = workers^
