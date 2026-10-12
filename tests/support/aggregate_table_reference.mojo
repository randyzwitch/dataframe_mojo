"""Logical rows for grouped reductions through the aggregate table, an
independent scalar model of their groups, and a dictionary-coded string
layout.

Group g's keys are a function of g alone and distinct across groups: a string
key (short, longer than twelve bytes with Unicode, empty, or null), an Int64
key (null only where the string is not), a Float64 key whose zero arrives as
-0.0 or 0.0 and which may be NaN, and a nullable Boolean. Row r belongs to
group (r * 7919) % groups, so a group's rows are scattered."""
from std.math import isnan, sqrt
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true

from dataframe import (
    BoolColumn,
    Column,
    DataFrame,
    DataType,
    Series,
    StringColumn,
)
from dataframe.categorical import decode_contiguous, encode
from dataframe.string_column import StringCodes
from dataframe.value import AnyValue


def group_of(row: Int, groups: Int) -> Int:
    return (row * 7919) % groups


def string_key_valid(g: Int) -> Bool:
    return g % 29 != 5


def string_key(g: Int) -> String:
    if g % 4 == 0:
        return "k" + String(g)
    if g % 4 == 1:
        return "a longer group key 東京 " + String(g)
    if g % 4 == 2:
        return String(g)
    return "é" + String(g) + " with a tail past twelve bytes"


def int_key_valid(g: Int) -> Bool:
    # Null only where the string key identifies the group.
    return not (g % 31 == 7 and string_key_valid(g))


def int_key(g: Int) -> Int64:
    return Int64(g) * 1_000_003 - 5_000_000


def float_key(g: Int) -> Float64:
    if g % 7 == 3:
        return Float64(0) / Float64(0)
    if g % 6 == 1:
        return -0.0
    if g % 6 == 4:
        return 0.0
    return Float64(g % 5) - 2.5


def bool_key_valid(g: Int) -> Bool:
    return g % 13 != 2


def narrow_valid(row: Int) -> Bool:
    return row % 17 != 3


def narrow(row: Int) -> Int32:
    return Int32((row * 104729) % 1999 - 1000)


def wide_valid(row: Int) -> Bool:
    return row % 13 != 4


def wide(row: Int) -> Int64:
    return Int64(row % 977) * 1_000_000_007 - 400_000_000_000


def real_valid(row: Int) -> Bool:
    return row % 19 != 9


def real(row: Int) -> Float64:
    return Float64(row % 211) / 8 - 13


def aggregate_frame(rows: Int, groups: Int) raises -> DataFrame:
    """Columns s, i, f, b (keys of each row's group), then x (Int32), w
    (Int64) and y (Float64) values, and r, the row's index."""
    var s = List[String](capacity=rows)
    var sv = List[Bool](capacity=rows)
    var i = List[Int64](capacity=rows)
    var iv = List[Bool](capacity=rows)
    var f = List[Float64](capacity=rows)
    var b = List[Bool](capacity=rows)
    var bv = List[Bool](capacity=rows)
    var x = List[Int32](capacity=rows)
    var xv = List[Bool](capacity=rows)
    var w = List[Int64](capacity=rows)
    var wv = List[Bool](capacity=rows)
    var y = List[Float64](capacity=rows)
    var yv = List[Bool](capacity=rows)
    var r = List[Int64](capacity=rows)
    for row in range(rows):
        r.append(Int64(row))
        var g = group_of(row, groups)
        s.append(string_key(g) if string_key_valid(g) else String("hidden"))
        sv.append(string_key_valid(g))
        i.append(int_key(g))
        iv.append(int_key_valid(g))
        f.append(float_key(g))
        b.append(g % 2 == 0)
        bv.append(bool_key_valid(g))
        x.append(narrow(row))
        xv.append(narrow_valid(row))
        w.append(wide(row))
        wv.append(wide_valid(row))
        y.append(real(row))
        yv.append(real_valid(row))
    return DataFrame(
        [
            Series("s", Column[String](s^, sv^)),
            Series("i", Column[Int64](i^, iv^)),
            Series("f", Column[Float64](f^)),
            Series("b", BoolColumn(b^, bv^)),
            Series("x", Column[Int32](x^, xv^)),
            Series("w", Column[Int64](w^, wv^)),
            Series("y", Column[Float64](y^, yv^)),
            Series("r", Column[Int64](r^)),
        ]
    )


struct GroupModel(Copyable, Movable):
    """One group's expected states, from scalar scans."""

    var rows: Int
    var x_count: Int
    var x_total: Int128
    var x_min: Int32
    var x_max: Int32
    var w_count: Int
    var w_nulls: Int
    var w_total: Int128
    var ys: List[Float64]

    def __init__(out self):
        self.rows = 0
        self.x_count = 0
        self.x_total = 0
        self.x_min = Int32.MAX
        self.x_max = Int32.MIN
        self.w_count = 0
        self.w_nulls = 0
        self.w_total = 0
        self.ys = List[Float64]()

    def add(mut self, row: Int):
        self.rows += 1
        if narrow_valid(row):
            var value = narrow(row)
            self.x_count += 1
            self.x_total += Int128(value)
            self.x_min = min(self.x_min, value)
            self.x_max = max(self.x_max, value)
        if wide_valid(row):
            self.w_count += 1
            self.w_total += Int128(wide(row))
        else:
            self.w_nulls += 1
        if real_valid(row):
            self.ys.append(real(row))

    def y_mean(self) -> Float64:
        var total = Float64(0)
        for value in self.ys:
            total += value
        return total / Float64(len(self.ys))

    def y_std(self) -> Float64:
        """Two-pass sample standard deviation (ddof 1)."""
        var mean = self.y_mean()
        var squares = Float64(0)
        for value in self.ys:
            squares += (value - mean) * (value - mean)
        return sqrt(squares / Float64(len(self.ys) - 1))


def group_models(rows: Int, groups: Int, start: Int = 0) -> List[GroupModel]:
    """Each group's model over rows `start` on (what a filter `r >= start`
    keeps)."""
    var models = List[GroupModel](length=groups, fill=GroupModel())
    for row in range(start, rows):
        models[group_of(row, groups)].add(row)
    return models^


def integer(value: AnyValue) raises -> Int128:
    """An integer result cell at whatever width the reduction returns."""
    var dtype = value.dtype()
    if dtype == DataType.INT8:
        return Int128(value.int8())
    if dtype == DataType.INT16:
        return Int128(value.int16())
    if dtype == DataType.INT32:
        return Int128(value.int32())
    if dtype == DataType.UINT8:
        return Int128(value.uint8())
    if dtype == DataType.UINT16:
        return Int128(value.uint16())
    if dtype == DataType.UINT32:
        return Int128(value.uint32())
    if dtype == DataType.UINT64:
        return Int128(value.uint64())
    return Int128(value.int64())


def key_text(s: AnyValue, i: AnyValue) raises -> String:
    """A result row's string and integer keys as one lookup text."""
    var text = String("null") if s.is_null() else "s:" + s.string()
    text += "|"
    text += String("null") if i.is_null() else String(i.int64())
    return text


def replaced_key(g: Int) -> String:
    """String key g with its first "group key" written "GROUPED KEY"."""
    if g % 4 == 1:
        return "a longer GROUPED KEY 東京 " + String(g)
    return string_key(g)


def model_key_text(g: Int, replaced: Bool = False) -> String:
    var text = String("s:") + (
        replaced_key(g) if replaced else string_key(g)
    ) if string_key_valid(g) else String("null")
    text += "|"
    text += String(int_key(g)) if int_key_valid(g) else String("null")
    return text


def assert_close(actual: Float64, expected: Float64, what: String) raises:
    var scale = max(Float64(1), abs(expected))
    assert_true(
        abs(actual - expected) <= 1e-12 * scale,
        what + ": " + String(actual) + " against " + String(expected),
    )


def assert_groups(
    actual: DataFrame,
    models: List[GroupModel],
    with_float_and_bool_keys: Bool,
    replaced: Bool = False,
) raises:
    """Every model group with rows appears once with its keys and states
    (sum, mean, min, max and count of x; sum and null count of w; row
    count; mean and standard deviation of y), in any order. `replaced`:
    the string keys are `replaced_key`'s."""
    var groups = len(models)
    var lookup = Dict[String, Int]()
    var present = 0
    for g in range(groups):
        if models[g].rows > 0:
            lookup[model_key_text(g, replaced)] = g
            present += 1
    assert_equal(actual.height(), present)
    var seen = List[Bool](length=groups, fill=False)
    for r in range(actual.height()):
        var text = key_text(actual.item(r, "s"), actual.item(r, "i"))
        assert_true(text in lookup, "unexpected group " + text)
        var g = lookup[text]
        assert_true(not seen[g], "group twice: " + text)
        seen[g] = True
        ref model = models[g]
        if with_float_and_bool_keys:
            var f = actual.item(r, "f").float64()
            var expected = float_key(g)
            if isnan(expected):
                assert_true(isnan(f), "NaN key of " + text)
            else:
                # -0.0 and 0.0 are one key; either sign may come back.
                assert_equal(f, expected)
            var bcell = actual.item(r, "b")
            assert_equal(bcell.is_null(), not bool_key_valid(g))
            if bool_key_valid(g):
                assert_equal(bcell.bool(), g % 2 == 0)
        assert_equal(integer(actual.item(r, "rows")), Int128(model.rows))
        assert_equal(integer(actual.item(r, "x_count")), Int128(model.x_count))
        var x_sum = actual.item(r, "x_sum")
        var w_sum = actual.item(r, "w_sum")
        if model.x_count == 0:
            assert_true(x_sum.is_null(), "empty x sum of " + text)
            assert_true(actual.item(r, "x_min").is_null())
        else:
            assert_equal(integer(x_sum), model.x_total)
            assert_equal(integer(actual.item(r, "x_min")), Int128(model.x_min))
            assert_equal(integer(actual.item(r, "x_max")), Int128(model.x_max))
            assert_close(
                actual.item(r, "x_mean").float64(),
                Float64(model.x_total) / Float64(model.x_count),
                "x mean of " + text,
            )
        if model.w_count == 0:
            assert_true(w_sum.is_null(), "empty w sum of " + text)
        else:
            assert_equal(integer(w_sum), model.w_total)
        assert_equal(integer(actual.item(r, "w_nulls")), Int128(model.w_nulls))
        if len(model.ys) >= 2:
            assert_close(
                actual.item(r, "y_mean").float64(),
                model.y_mean(),
                "y mean of " + text,
            )
            assert_close(
                actual.item(r, "y_std").float64(),
                model.y_std(),
                "y std of " + text,
            )


def coded_strings(values: Series) raises -> Series:
    """String column `values` as a Parquet column with a dictionary page
    loads: the same strings, carrying their dictionary's codes."""
    var categorical = encode(values)
    var codes = categorical.rechunk()
    ref column = codes._data[Column[UInt32]]
    var copied = List[UInt32](capacity=len(column))
    for k in range(len(column)):
        copied.append(column._get(k))
    var strings = decode_contiguous(categorical)
    if strings.is_chunked():
        strings = strings.rechunk()
    strings._data[StringColumn]._codes = ArcPointer(
        StringCodes(ArcPointer(copied^), categorical.dtype())
    )
    return strings^
