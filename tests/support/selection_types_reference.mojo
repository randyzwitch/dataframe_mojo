"""Positional typed cells and literal byte order for public row selection.

Production columns only adapt these inputs. Neither their getters nor their
sorting, equality, casting, or filtering determine an expected result.
"""
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe import (
    AnyValue,
    Column,
    DataFrame,
    DataType,
    ListColumn,
    Series,
    StructColumn,
    from_wkb,
)
from dataframe.dtype import CategoricalDictionary
from tests.support.key_kernel_reference import KeyCell
from tests.support.coverage_generators import chunk_layout

comptime SELECTION_NUMERIC = (
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
comptime SELECTION_TYPES = 29


def selection_rank(row: Int) -> Int:
    var values: List[Int] = [3, 1, 3, 2, 0, 2, 1, 4, 3]
    return values[row % len(values)]


def selection_valid(row: Int, mode: Int) -> Bool:
    return mode == 0 or mode == 1 and row % 5 != 2


def selection_text(row: Int) -> String:
    var values: List[String] = [
        "",
        "a\x00",
        "abcdefghijkl",
        "abcdefghijklm\x00é",
        "é",
    ]
    return values[selection_rank(row)].copy()


def selection_bytes(row: Int) -> List[UInt8]:
    var rank = selection_rank(row)
    if rank == 0:
        return []
    if rank == 1:
        return [0]
    if rank == 2:
        return [0, 0]
    if rank == 3:
        var result = List[UInt8](length=32, fill=255)
        result[0] = 128
        return result^
    return [255, 0, 128]


def selection_wkb(row: Int) -> List[UInt8]:
    # Valid little-endian XY point. Exact bytes, including signed coordinates,
    # are the key contract; numerical coordinate order is deliberately different.
    var result: List[UInt8] = [1, 1, 0, 0, 0]
    var x = bitcast[DType.uint64](Float64(selection_rank(row) - 2))
    var y = bitcast[DType.uint64](Float64(-7))
    for value in [x, y]:
        for byte in range(8):
            result.append(UInt8((value >> UInt64(8 * byte)) & 255))
    return result^


def bytes_less(a: List[UInt8], b: List[UInt8]) -> Bool:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return a[i] < b[i]
    return len(a) < len(b)


def selection_decimal(kind: Int, row: Int) -> Int128:
    var base = (
        Int128(1000000000000000000) * 1000000000000 if kind
        == 16 else Int128(9007199254740993) if kind
        == 15 else Int128(1234567)
    )
    return (base + Int128(selection_rank(row))) * Int128(1 if row % 3 else -1)


def decimal_text(value: Int128) -> String:
    var magnitude = -value if value < 0 else value
    var cents = Int(magnitude % 100)
    return (
        ("-" if value < 0 else "")
        + String(magnitude // 100)
        + "."
        + ("0" if cents < 10 else "")
        + String(cents)
    )


def selection_tick(kind: Int, row: Int) -> Int64:
    var rank = Int64(selection_rank(row))
    return (
        rank + 1
    ) * 123456789 if kind == 18 else rank * 1000000007 - 2000000014


def selection_float(row: Int) -> Float64:
    if row % 9 == 0:
        return Float64(0) / Float64(0)
    if row % 9 == 1:
        return -0.0
    if row % 9 == 2:
        return 0.0
    if row % 9 == 3:
        return Float64(1) / Float64(0)
    if row % 9 == 4:
        return -Float64(1) / Float64(0)
    return Float64(selection_rank(row)) + 0.25


def selection_numeric[D: DType](row: Int, valid: Bool) -> Scalar[D]:
    if not valid:
        return Scalar[D].MAX
    comptime if D.is_floating_point():
        return Scalar[D](selection_float(row))
    elif D == DType.uint64:
        return Scalar[D](UInt64(1) << 63) + Scalar[D](selection_rank(row))
    elif D.is_unsigned():
        return Scalar[D](selection_rank(row) + 3)
    else:
        return Scalar[D](selection_rank(row) - 2)


def selection_dtype(kind: Int) raises -> DataType:
    comptime for i in range(len(SELECTION_NUMERIC)):
        if kind == i:
            return DataType.of(SELECTION_NUMERIC[i])
    if kind == 10:
        return DataType.BOOL
    if kind == 11:
        return DataType.STRING
    if kind == 12:
        return DataType.BINARY
    if kind == 13:
        var dictionary = CategoricalDictionary()
        # Code order differs from lexical order.
        for row in [0, 4, 1, 3, 7]:
            var label = selection_text(row)
            dictionary.append(label)
        return DataType.categorical(dictionary^)
    if kind >= 14 and kind <= 16:
        return DataType.decimal(
            9 if kind == 14 else 18 if kind == 15 else 38,
            2,
            32 if kind == 14 else 64 if kind == 15 else 128,
        )
    if kind == 17:
        return DataType.DATE
    if kind == 18:
        return DataType.TIME
    if kind >= 19 and kind <= 22:
        return DataType.datetime(
            "ns" if kind == 19 or kind == 22 else "us" if kind == 20 else "ms",
            "America/New_York" if kind == 22 else "",
        )
    if kind >= 23 and kind <= 25:
        return DataType.duration(
            "ns" if kind == 23 else "us" if kind == 24 else "ms"
        )
    if kind == 26:
        return DataType.list(DataType.INT64)
    if kind == 27:
        return DataType.struct(
            ["number", "label"], [DataType.INT64, DataType.STRING]
        )
    return DataType.geometry('{"crs":"EPSG:4326","edges":"spherical"}')


def selection_column(
    kind: Int, n: Int, offset: Int, mode: Int
) raises -> Series:
    var count = offset + n
    var valid = List[Bool]()
    for row in range(count):
        valid.append(selection_valid(row, mode))
    var result: Series
    comptime for i in range(len(SELECTION_NUMERIC)):
        comptime D = SELECTION_NUMERIC[i]
        if kind == i:
            var values = List[Scalar[D]]()
            for row in range(count):
                values.append(selection_numeric[D](row, valid[row]))
            return Series("x", Column[Scalar[D]](values^, valid.copy())).slice(
                offset, n
            )
    if kind == 10:
        var values = List[Bool]()
        for row in range(count):
            values.append(row % 3 != 0)
        result = Series("x", Column[Bool](values^, valid.copy()))
    elif kind == 11:
        var values = List[String]()
        for row in range(count):
            values.append(selection_text(row))
        result = Series("x", Column[String](values^, valid.copy()))
    elif kind == 12 or kind == 28:
        var values = List[List[UInt8]]()
        for row in range(count):
            values.append(
                selection_bytes(row) if kind == 12 else selection_wkb(row)
            )
        result = Series.binary("x", values, valid)
        if kind == 28:
            result = from_wkb(result, '{"crs":"EPSG:4326","edges":"spherical"}')
    elif kind == 13:
        var values = List[UInt32]()
        var codes: List[UInt32] = [1, 2, 3, 0, 4]
        for row in range(count):
            values.append(
                codes[selection_rank(row)] if valid[row] else UInt32.MAX
            )
        result = Series("x", Column[UInt32](values^, valid.copy())).with_dtype(
            selection_dtype(kind)
        )
    elif kind >= 14 and kind <= 16:
        if kind == 14:
            var values = List[Int32]()
            for row in range(count):
                values.append(
                    Int32(selection_decimal(kind, row)) if valid[
                        row
                    ] else Int32.MAX
                )
            result = Series(
                "x", Column[Int32](values^, valid.copy())
            ).with_dtype(selection_dtype(kind))
        elif kind == 15:
            var values = List[Int64]()
            for row in range(count):
                values.append(
                    Int64(selection_decimal(kind, row)) if valid[
                        row
                    ] else Int64.MAX
                )
            result = Series(
                "x", Column[Int64](values^, valid.copy())
            ).with_dtype(selection_dtype(kind))
        else:
            var values = List[Int128]()
            for row in range(count):
                values.append(
                    selection_decimal(kind, row) if valid[row] else Int128.MAX
                )
            result = Series(
                "x", Column[Int128](values^, valid.copy())
            ).with_dtype(selection_dtype(kind))
    elif kind >= 17 and kind <= 25:
        var values = List[Int64]()
        for row in range(count):
            values.append(
                selection_tick(kind, row) if valid[row] else Int64.MAX
            )
        result = Series("x", Column[Int64](values^, valid.copy())).with_dtype(
            selection_dtype(kind)
        )
    elif kind == 26:
        var rows = List[Series]()
        for row in range(count):
            if row % 4 == 0:
                rows.append(Series("item", Column[Int64]([])))
            else:
                rows.append(
                    Series(
                        "item",
                        Column[Int64](
                            [Int64(row), 999999, Int64(-row)],
                            [True, False, True],
                        ),
                    )
                )
        result = Series("x", ListColumn.from_lists(rows, DataType.INT64, valid))
    else:
        var numbers = List[Int64]()
        var labels = List[String]()
        var child_valid = List[Bool]()
        var bits = List[UInt8](length=(count + 7) // 8, fill=0)
        for row in range(count):
            numbers.append(Int64(row))
            labels.append(selection_text(row))
            child_valid.append(row % 3 != 1)
            if valid[row]:
                bits[row // 8] |= UInt8(1 << (row % 8))
        result = Series(
            "x",
            StructColumn(
                [
                    Series(
                        "number", Column[Int64](numbers^, child_valid.copy())
                    ),
                    Series(
                        "label", Column[String](labels^, child_valid.copy())
                    ),
                ],
                bits^,
            ),
        )
    return result.slice(offset, n)


def selection_frame(
    n: Int, offset: Int, mode: Int, chunked: Bool = False
) raises -> DataFrame:
    var columns = List[Series]()
    var ids = List[Int64]()
    var predicates = List[Bool]()
    var predicate_valid = List[Bool]()
    for row in range(offset + n):
        ids.append(Int64(row))
        predicates.append(row % 3 == 1)
        predicate_valid.append(row % 5 != 2)
    columns.append(Series("id", Column[Int64](ids^)).slice(offset, n))
    columns.append(
        Series("predicate", Column[Bool](predicates^, predicate_valid^)).slice(
            offset, n
        )
    )
    for kind in range(SELECTION_TYPES):
        columns.append(
            selection_column(kind, n, offset, mode).renamed("v" + String(kind))
        )
    var frame = DataFrame(columns^)
    return chunk_layout(frame) if chunked else frame^


def assert_selection_cell(
    actual: AnyValue, kind: Int, row: Int, mode: Int
) raises:
    assert_true(
        actual.dtype()
        == (DataType.STRING if kind == 13 else selection_dtype(kind))
    )
    var valid = selection_valid(row, mode)
    assert_equal(actual.is_null(), not valid)
    if not valid:
        return
    comptime for i in range(len(SELECTION_NUMERIC)):
        comptime D = SELECTION_NUMERIC[i]
        if kind == i:
            var expected = selection_numeric[D](row, True)
            var value = actual.numeric[D]()
            comptime if D.is_floating_point():
                if isnan(expected):
                    assert_true(isnan(value))
                else:
                    assert_equal(value, expected)
                    if value == 0:
                        assert_equal(
                            bitcast[DType.uint64](Float64(value)),
                            bitcast[DType.uint64](Float64(expected)),
                        )
            else:
                assert_equal(value, expected)
            return
    if kind == 10:
        assert_equal(actual.bool(), row % 3 != 0)
    elif kind == 11 or kind == 13:
        assert_equal(actual.string(), selection_text(row))
    elif kind == 12 or kind == 28:
        assert_equal(
            actual.bytes(),
            selection_bytes(row) if kind == 12 else selection_wkb(row),
        )
    elif kind >= 14 and kind <= 16:
        assert_equal(String(actual), decimal_text(selection_decimal(kind, row)))
    elif kind >= 17 and kind <= 25:
        assert_equal(actual.to_physical(), selection_tick(kind, row))
    elif kind == 26:
        var items = actual.list()
        assert_true(items.dtype() == DataType.INT64)
        assert_equal(len(items), 0 if row % 4 == 0 else 3)
        if row % 4:
            assert_equal(items.get(0).int64(), Int64(row))
            assert_true(items.get(1).is_null())
            assert_equal(items.get(2).int64(), Int64(-row))
    else:
        var number = actual.struct_field("number")
        var label = actual.struct_field("label")
        assert_equal(number.is_null(), row % 3 == 1)
        assert_equal(label.is_null(), row % 3 == 1)
        if row % 3 != 1:
            assert_equal(number.int64(), Int64(row))
            assert_equal(label.string(), selection_text(row))


def selection_key(kind: Int, row: Int, mode: Int) raises -> KeyCell:
    var valid = selection_valid(row, mode)
    if kind == 8 or kind == 9:
        return KeyCell(1, valid, floating=selection_float(row))
    if kind == 11 or kind == 13:
        return KeyCell(2, valid, text=selection_text(row))
    var integer = Int128(selection_rank(row) - 2)
    if kind >= 4 and kind <= 6:
        integer = Int128(selection_rank(row) + 3)
    if kind == 7:
        integer = (Int128(1) << 63) + Int128(selection_rank(row))
    if kind == 10:
        integer = Int128(row % 3 != 0)
    if kind == 12:
        integer = Int128(selection_rank(row))
    if kind == 28:
        integer = 0
        # Rank all five distinct WKB patterns with a scalar byte comparator.
        for representative in [4, 1, 3, 0, 7]:
            integer += Int128(
                bytes_less(selection_wkb(representative), selection_wkb(row))
            )
    if kind >= 14 and kind <= 16:
        integer = selection_decimal(kind, row)
    if kind >= 17 and kind <= 25:
        integer = Int128(selection_tick(kind, row))
    return KeyCell(0, valid, integer=integer)


def assert_selected_frame(actual: DataFrame, rows: List[Int], mode: Int) raises:
    assert_equal(actual.height(), len(rows))
    var names: List[String] = ["id", "predicate"]
    for kind in range(SELECTION_TYPES):
        names.append("v" + String(kind))
        assert_true(
            actual.column(names[len(names) - 1]).dtype()
            == selection_dtype(kind)
        )
    assert_equal(actual.columns(), names)
    for at in range(len(rows)):
        var row = rows[at]
        assert_equal(actual.item(at, "id").int64(), Int64(row))
        assert_equal(actual.item(at, "predicate").is_null(), row % 5 == 2)
        if row % 5 != 2:
            assert_equal(actual.item(at, "predicate").bool(), row % 3 == 1)
        for kind in range(SELECTION_TYPES):
            assert_selection_cell(
                actual.item(at, "v" + String(kind)), kind, row, mode
            )
