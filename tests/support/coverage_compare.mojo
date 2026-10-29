"""Explicit complete typed comparison; no production equality or aggregate checks."""
from std.collections import Dict
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from dataframe import AnyValue, DataFrame, DataType
from tests.support.reference_model import (
    ReferenceRow,
    ReferenceTable,
    RowPair,
    field_valid,
    reference_join,
)


def assert_float(actual: Float64, expected: Float64) raises:
    if isnan(expected):
        assert_true(isnan(actual))
    else:
        assert_equal(actual, expected)
        if expected == 0:
            assert_equal(
                bitcast[DType.uint64](actual), bitcast[DType.uint64](expected)
            )


def assert_value(actual: AnyValue, expected: AnyValue) raises:
    """Typed metamorphic equality, including nested children and float bit policy."""
    assert_true(actual.dtype() == expected.dtype())
    assert_equal(actual.is_null(), expected.is_null())
    if expected.is_null():
        return
    var dtype = expected.dtype()
    comptime numeric_types = (
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
    comptime for index in range(len(numeric_types)):
        comptime D = numeric_types[index]
        if dtype == DataType.of(D):
            comptime if D.is_floating_point():
                assert_float(
                    Float64(actual.numeric[D]()), Float64(expected.numeric[D]())
                )
            else:
                assert_equal(actual.numeric[D](), expected.numeric[D]())
            return
    if dtype.is_temporal():
        assert_equal(actual.to_physical(), expected.to_physical())
    elif dtype == DataType.STRING:
        assert_equal(actual.string(), expected.string())
    elif dtype == DataType.BOOL:
        assert_equal(actual.bool(), expected.bool())
    elif dtype.is_binary() or dtype.is_geometry():
        assert_equal(actual.bytes(), expected.bytes())
    elif dtype.is_decimal():
        # AnyValue exposes decimal values as exact fixed-point text.
        assert_equal(String(actual), String(expected))
    elif dtype.is_list():
        var left = actual.list()
        var right = expected.list()
        assert_true(left.dtype() == right.dtype())
        assert_equal(len(left), len(right))
        for index in range(len(right)):
            assert_value(left.get(index), right.get(index))
    elif dtype.is_struct():
        for name in dtype.field_names():
            assert_value(actual.struct_field(name), expected.struct_field(name))
    else:
        raise Error("No declared comparison policy for this dtype")


def assert_frame(actual: DataFrame, expected: DataFrame) raises:
    """Metamorphic comparison. Independent expectations use scalar models/literals."""
    assert_equal(actual.height(), expected.height())
    assert_equal(actual.columns(), expected.columns())
    for name in expected.columns():
        assert_true(
            actual.column(name).dtype() == expected.column(name).dtype()
        )
        for row in range(expected.height()):
            assert_value(actual.item(row, name), expected.item(row, name))


def _row_text(frame: DataFrame, row: Int) raises -> String:
    var text = String()
    for name in frame.columns():
        var cell = frame.item(row, name)
        text += "null" if cell.is_null() else "v:" + String(cell)
        text += "\x1f"
    return text^


def assert_frame_unordered(actual: DataFrame, expected: DataFrame) raises:
    """`assert_frame` up to row order: a join promises none. Every actual
    row is paired with an unused expected row that prints alike, then
    compared cell by cell under the same typed policy."""
    assert_equal(actual.height(), expected.height())
    assert_equal(actual.columns(), expected.columns())
    for name in expected.columns():
        assert_true(
            actual.column(name).dtype() == expected.column(name).dtype()
        )
    var waiting = Dict[String, List[Int]]()
    for row in range(expected.height()):
        var text = _row_text(expected, row)
        if text not in waiting:
            waiting[text] = List[Int]()
        waiting[text].append(row)
    for row in range(actual.height()):
        var text = _row_text(actual, row)
        assert_true(
            text in waiting and len(waiting[text]) > 0,
            "No unused expected row matches output row " + String(row),
        )
        var match = waiting[text].pop()
        for name in expected.columns():
            assert_value(actual.item(row, name), expected.item(match, name))


def assert_model(actual: DataFrame, expected: ReferenceTable) raises:
    """Compare directly to reference rows, without materializing expected data."""
    assert_equal(actual.height(), len(expected.rows))
    assert_equal(actual.columns(), expected.names)
    for name in expected.names:
        if name != "i" and name != "f" and name != "s" and name != "b":
            raise Error("Unknown reference field")
        var dtype = (
            DataType.INT64 if name
            == "i" else DataType.FLOAT64 if name
            == "f" else DataType.STRING if name
            == "s" else DataType.BOOL
        )
        assert_true(actual.column(name).dtype() == dtype)
        for index in range(len(expected.rows)):
            ref row = expected.rows[index]
            var valid = (
                row.iv if name
                == "i" else row.fv if name
                == "f" else row.sv if name
                == "s" else row.bv
            )
            var cell = actual.item(index, name)
            assert_equal(cell.is_null(), not valid)
            if not valid:
                continue
            if name == "i":
                assert_equal(cell.int64(), row.i)
            elif name == "f":
                assert_float(cell.float64(), row.f)
            elif name == "s":
                assert_equal(cell.string(), row.s)
            elif name == "b":
                assert_equal(cell.bool(), row.b)
            else:
                raise Error("Unknown reference field")


def reference_dtype(name: String) raises -> DataType:
    if name == "i":
        return DataType.INT64
    if name == "f":
        return DataType.FLOAT64
    if name == "s":
        return DataType.STRING
    if name == "b":
        return DataType.BOOL
    raise Error("Unknown reference field")


def assert_model_unordered(actual: DataFrame, expected: ReferenceTable) raises:
    """Complete row multiset comparison, consuming each expected occurrence once."""
    assert_equal(actual.height(), len(expected.rows))
    assert_model(actual.clear(), expected.slice(0, 0))
    var used = List[Bool](length=len(expected.rows), fill=False)
    for row in range(actual.height()):
        var found = False
        for candidate in range(len(expected.rows)):
            if used[candidate]:
                continue
            try:
                assert_model(actual.slice(row, 1), expected.slice(candidate, 1))
                used[candidate] = True
                found = True
                break
            except:
                pass
        assert_true(
            found, "No unused expected row matches output row " + String(row)
        )


def assert_reference_cell(
    actual: AnyValue, row: ReferenceRow, name: String
) raises:
    assert_equal(actual.is_null(), not field_valid(row, name))
    if actual.is_null():
        return
    if name == "i":
        assert_equal(actual.int64(), row.i)
    elif name == "f":
        assert_float(actual.float64(), row.f)
    elif name == "s":
        assert_equal(actual.string(), row.s)
    elif name == "b":
        assert_equal(actual.bool(), row.b)
    else:
        raise Error("Unknown reference field")


def assert_join_model(
    actual: DataFrame,
    left: ReferenceTable,
    right: ReferenceTable,
    keys: List[String],
    how: String,
    coalesce: Bool = True,
    suffix: String = "_right",
) raises:
    var pairs = reference_join(left, right, keys, how)
    assert_join_pairs(actual, left, right, keys, how, pairs, coalesce, suffix)


def assert_join_pairs(
    actual: DataFrame,
    left: ReferenceTable,
    right: ReferenceTable,
    keys: List[String],
    how: String,
    pairs: List[RowPair],
    coalesce: Bool = True,
    suffix: String = "_right",
) raises:
    """Check complete join cells for an independently transformed pair relation."""
    var names = left.names.copy()
    var right_fields = List[String]()
    if how != "semi" and how != "anti":
        for name in right.names:
            var is_key = False
            for key in keys:
                is_key = is_key or key == name
            if how != "cross" and coalesce and is_key:
                continue
            var collision = False
            for existing in left.names:
                collision = collision or existing == name
            names.append(name + suffix if collision else name.copy())
            right_fields.append(name.copy())
    assert_equal(actual.columns(), names)
    assert_equal(actual.height(), len(pairs))
    for name in left.names:
        var dtype = (
            DataType.INT64 if name
            == "i" else DataType.FLOAT64 if name
            == "f" else DataType.STRING if name
            == "s" else DataType.BOOL
        )
        assert_true(actual.column(name).dtype() == dtype)
    for c in range(len(right_fields)):
        var name = right_fields[c]
        var dtype = (
            DataType.INT64 if name
            == "i" else DataType.FLOAT64 if name
            == "f" else DataType.STRING if name
            == "s" else DataType.BOOL
        )
        assert_true(actual.column(names[len(left.names) + c]).dtype() == dtype)
    for index in range(len(pairs)):
        var pair = pairs[index].copy()
        for name in left.names:
            var is_key = False
            for key in keys:
                is_key = is_key or key == name
            var cell = actual.item(index, name)
            if how == "right" and coalesce and is_key:
                # A right join preserves its right input's keys, with left names.
                assert_reference_cell(cell, right.rows[pair.right], name)
            elif pair.left >= 0:
                assert_reference_cell(cell, left.rows[pair.left], name)
            elif coalesce and is_key:
                assert_reference_cell(cell, right.rows[pair.right], name)
            else:
                assert_true(cell.is_null())
        for c in range(len(right_fields)):
            var cell = actual.item(index, names[len(left.names) + c])
            if pair.right >= 0:
                assert_reference_cell(
                    cell, right.rows[pair.right], right_fields[c]
                )
            else:
                assert_true(cell.is_null())
