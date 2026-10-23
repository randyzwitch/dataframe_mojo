"""Equal labels are equal group keys on every grouping route (#943)."""
# contract-group: reduce.grouped
from std.memory import ArcPointer
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series, StringColumn, col
from dataframe.categorical import decode
from dataframe.dtype import CategoricalDictionary
from dataframe.string_column import StringCodes


def _dictionary() -> CategoricalDictionary:
    # Codes 0 and 1 name one label; 3 is never used.
    var dictionary = CategoricalDictionary()
    for label in ["same", "same", "other", "unused"]:
        dictionary.append(label)
    return dictionary^


def _categorical(codes: List[UInt32], valid: List[Bool]) raises -> Series:
    return Series("k", Column[UInt32](codes.copy(), valid.copy())).with_dtype(
        DataType.categorical(_dictionary())
    )


def _coded(source: Series) raises -> Series:
    # Parquet-style: text beside the dictionary codes it was read from.
    var result = decode(source)
    var codes = source.uint32()
    var values = List[UInt32](capacity=len(source))
    for row in range(len(source)):
        values.append(codes._get(row))
    result._data[StringColumn]._codes = ArcPointer(
        StringCodes(ArcPointer(values^), source.dtype())
    )
    return result^


def _summary(result: DataFrame) raises -> List[String]:
    """One line a group, label and aggregates, sorted: group order is
    unspecified without maintain_order."""
    var keys = result.column("k")
    if keys.dtype().is_categorical():
        keys = decode(keys)
    var text = keys.string()
    var lines = List[String]()
    for row in range(result.height()):
        var label = String(text._get(row)) if text._valid(row) else "<null>"
        var line = label + ":" + String(result.item(row, "s").int64())
        if result.width() > 2:
            line += ":" + String(result.item(row, "m").int64())
        lines.append(line)
    sort(lines)
    return lines^


def _check(frame: DataFrame, expected: List[String]) raises:
    for ordered in [False, True]:
        var eager = frame.group_by("k", maintain_order=ordered).agg(
            [col("x").sum().alias("s"), col("x").max().alias("m")]
        )
        assert_equal(_summary(eager), expected)
        var lazy = (
            frame.lazy()
            .group_by("k", maintain_order=ordered)
            .agg([col("x").sum().alias("s"), col("x").max().alias("m")])
            .collect()
        )
        assert_equal(_summary(lazy), expected)
    var indices = frame.group_indices("k")
    assert_equal(indices.count(), len(expected))
    # A window partition is a grouping too.
    var totals = frame.with_columns(col("x").sum().over("k").alias("t"))
    var seen = List[String]()
    for row in range(totals.height()):
        var key = frame.column("k")
        if key.dtype().is_categorical():
            key = decode(key)
        var text = key.string()
        var label = String(text._get(row)) if text._valid(row) else "<null>"
        var line = label + ":" + String(totals.item(row, "t").int64())
        if line not in seen:
            seen.append(line)
    assert_equal(len(seen), len(expected))


def _frame(key: Series, x: List[Int64]) raises -> DataFrame:
    return DataFrame([key.copy(), Series("x", Column[Int64](x.copy()))])


def test_two_codes_one_label_are_one_group_on_every_cpu_route() raises:
    var codes: List[UInt32] = [0, 1, 0, 1, 2, 0, 0, 1]
    var valid: List[Bool] = [True, True, True, True, True, False, False, True]
    var x: List[Int64] = [1, 2, 3, 4, 5, 6, 7, 8]
    # same: 1 + 2 + 3 + 4 + 8, other: 5, null: 6 + 7.
    var expected: List[String] = ["<null>:13:7", "other:5:5", "same:18:8"]
    var key = _categorical(codes, valid)
    _check(_frame(key, x), expected)
    _check(_frame(_coded(key), x), expected)
    _check(_frame(decode(key), x), expected)


def test_two_codes_one_label_on_enough_rows_for_parallel_routes() raises:
    var n = 100000
    var codes = List[UInt32](capacity=n)
    var valid = List[Bool](capacity=n)
    var x = List[Int64](capacity=n)
    var same = Int64(0)
    var other = Int64(0)
    var nulls = Int64(0)
    var last_null = 0
    var last_other = 0
    var last_same = 0
    for row in range(n):
        codes.append(UInt32(row % 3))
        valid.append(row % 11 != 0)
        x.append(Int64(row))
        # Rows ascend, so the last row of a group is its maximum.
        if row % 11 == 0:
            nulls += Int64(row)
            last_null = row
        elif row % 3 == 2:
            other += Int64(row)
            last_other = row
        else:
            same += Int64(row)
            last_same = row
    var expected: List[String] = [
        "<null>:" + String(nulls) + ":" + String(last_null),
        "other:" + String(other) + ":" + String(last_other),
        "same:" + String(same) + ":" + String(last_same),
    ]
    sort(expected)
    var key = _categorical(codes, valid)
    _check(_frame(key, x), expected)
    _check(_frame(_coded(key), x), expected)
