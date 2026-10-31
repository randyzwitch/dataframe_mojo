"""Ordinary Boolean comparisons read independent owned bitmap windows."""
# contract-group: expr.comparison_chunks
from std.testing import assert_equal, assert_false, assert_raises
from dataframe import BoolColumn, Column, DataFrame, DataType, Series
from dataframe.expr import EQ, NE, LT, LE, GT, GE
from dataframe.kernels.binary import binary
from tests.support.coverage_compare import assert_frame


def _source(right: Bool, offset: Int, dense: Bool) raises -> Series:
    var values = List[Bool]()
    var valid = List[Bool]()
    for row in range(300):
        values.append(row % 3 != 0 if right else row % 5 != 0)
        valid.append(dense or row % (5 if right else 7) != 0)
    var source = Series("x", BoolColumn(values, valid)).slice(offset, 257)
    var parts = List[Series]()
    var at = 0
    var sizes = [7, 1, 0, 8, 19, 222] if right else [3, 0, 17, 1, 0, 236]
    for size in sizes:
        parts.append(source.slice(at, size))
        at += size
    return Series._from_chunks(parts)


def _check[op: Int](offset: Int, dense: Bool, shape: Int, cache: Int) raises:
    var left = _source(False, offset, dense)
    var right = _source(True, offset, dense)
    if shape == 1:
        left = Series("x", BoolColumn([True]))
    elif shape == 2:
        right = Series("x", BoolColumn([False]))
    elif shape == 3:
        left = Series("x", BoolColumn([True], [False]))
    elif shape == 4:
        right = Series("x", BoolColumn([True], [False]))
    if cache & 1 != 0:
        var ready = left.rechunk()
    if cache & 2 != 0:
        var ready = right.rechunk()
    var expected = List[Bool]()
    var valid = List[Bool]()
    for row in range(257):
        var i = offset + row
        var a = True if shape == 1 or shape == 3 else i % 5 != 0
        var b = False if shape == 2 else (True if shape == 4 else i % 3 != 0)
        var av = shape != 3 and (shape == 1 or dense or i % 7 != 0)
        var bv = shape != 4 and (shape == 2 or dense or i % 5 != 0)
        var hit: Bool
        comptime if op == EQ:
            hit = a == b
        elif op == NE:
            hit = a != b
        elif op == LT:
            hit = not a and b
        elif op == LE:
            hit = not a or b
        elif op == GT:
            hit = a and not b
        else:
            hit = a or not b
        valid.append(av and bv)
        expected.append(av and bv and hit)
    # Private Bool comparison masks are ignored, including a short mask.
    var actual = binary[op](left, right, [False])
    assert_frame(
        DataFrame([actual.renamed("x")]),
        DataFrame([Series("x", BoolColumn(expected, valid))]),
    )
    ref output = actual._data[BoolColumn]
    for row in range(len(output)):
        if not output._valid(row):
            assert_false(output._get(row))
    if left.is_chunked():
        assert_equal(
            left._chunked.value()[].cache[].state.load(),
            Int64(2 if cache & 1 != 0 else 0),
        )
    if right.is_chunked():
        assert_equal(
            right._chunked.value()[].cache[].state.load(),
            Int64(2 if cache & 2 != 0 else 0),
        )


def test_boolean_comparisons_own_unequal_windows_and_preserve_cached_sides() raises:
    # EQ belongs to the preceding membership PR; this subset owns the other five.
    comptime ops = (NE, LT, LE, GT, GE)
    comptime for k in range(len(ops)):
        for offset in range(8):
            _check[ops[k]](offset, offset % 2 == 0, 0, offset % 4)
        for shape in range(1, 5):
            _check[ops[k]](13, False, shape, shape % 4)


def test_boolean_comparison_empty_broadcast_and_length_error() raises:
    var source = _source(False, 0, False)
    var empty = source.slice(0, 0)
    var scalar = Series("x", BoolColumn([True]))
    assert_equal(len(binary[LT](empty, scalar)), 0)
    assert_equal(len(binary[GE](scalar, empty)), 0)
    with assert_raises(contains="Incompatible expression lengths"):
        _ = binary[NE](source, source.slice(0, 2))


def test_private_bool_tags_keep_numeric_storage_and_mask_dispatch() raises:
    var source = Series(
        "x", Column[Int64]([1, 2, 3, 99], [True, True, True, False])
    )
    var tagged = Series("x", source._data.copy(), DataType.BOOL)
    var chunks = Series._from_chunks([tagged.slice(0, 1), tagged.slice(1, 3)])
    var other = Series("x", Column[Int64]([2, 2, 1, 99]))
    var rhs = Series("x", other._data.copy(), DataType.BOOL)
    var expected = Series(
        "x", BoolColumn([True, False, False, False], [True, True, True, False])
    )
    assert_frame(
        DataFrame([binary[LT](chunks, rhs).renamed("x")]),
        DataFrame([expected^]),
    )
    var masked = binary[LT](chunks, rhs, List[Bool](length=4, fill=False))
    assert_frame(
        DataFrame([masked.renamed("x")]),
        DataFrame(
            [
                Series(
                    "x",
                    BoolColumn(
                        [False, False, False, False],
                        [False, False, False, False],
                    ),
                )
            ]
        ),
    )
