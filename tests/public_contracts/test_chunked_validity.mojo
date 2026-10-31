"""Validity operations keep literal null rules across physical boundaries."""
# contract-group: expr.chunk_windows
from std.math import isnan
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    BoolColumn,
    StringColumn,
    ListColumn,
    StructColumn,
    DataFrame,
    DataType,
    Expr,
    Series,
    col,
    lit,
    when,
)
from dataframe.dtype import NUMERIC_DTYPES
from dataframe.expr import (
    AND,
    OR,
    XOR,
    NOT,
    IS_NULL,
    IS_NOT_NULL,
    IS_NAN,
    IS_NOT_NAN,
    IS_FINITE,
    IS_INFINITE,
    FILL_NULL,
    FILL_NAN,
    KEEP_NULLS,
    EQ_MISSING,
    NE_MISSING,
)
from dataframe.kernels.binary import binary, unary
from tests.support.coverage_compare import assert_frame


def _pieces(source: Series, sizes: List[Int]) raises -> Series:
    var parts = List[Series]()
    var at = 0
    for size in sizes:
        parts.append(source.slice(at, size))
        at += size
    assert_equal(at, len(source))
    return Series._from_chunks(parts)


def _cold(source: Series) raises:
    if source.is_chunked():
        assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))


def _same(actual: Series, expected: Series) raises:
    assert_frame(
        DataFrame([actual.renamed("x")]),
        DataFrame([expected.renamed("x")]),
    )


def _bool_result(actual: Series, expected: Series) raises:
    _same(actual, expected)
    var column = actual.bool()
    for row in range(len(column)):
        if not column._valid(row):
            assert_equal(column._get(row), False)


def _repeat_bools(values: List[Bool], offset: Int, n: Int) -> List[Bool]:
    var result = List[Bool](capacity=n)
    for row in range(n):
        result.append(values[(offset + row) % len(values)])
    return result^


def _numeric_source[D: DType](right: Bool) raises -> Series:
    var seed = List[Int]()
    if right:
        seed = [9, 4, 88, 87, 7, 86, 5, 85]
    else:
        seed = [2, 99, 0, 98, 7, 9, 97, 96]
    var valid = List[Bool]()
    if right:
        valid = [True, True, False, False, True, False, True, False]
    else:
        valid = [True, False, True, False, True, True, False, False]
    var values = List[Scalar[D]](capacity=60)
    for row in range(60):
        values.append(Scalar[D](seed[row % 8]))
    return Series("x", Column[Scalar[D]](values^, _repeat_bools(valid, 0, 60)))


def _numeric_windows[
    D: DType
](offset: Int, mask: List[Bool] = List[Bool]()) raises:
    var flat_a = _numeric_source[D](False).slice(offset, 41)
    var flat_b = _numeric_source[D](True).slice(offset, 41)
    var a = _pieces(flat_a, [3, 17, 0, 21])
    var b = _pieces(flat_b, [4, 1, 15, 0, 21])
    comptime ops = (FILL_NULL, KEEP_NULLS, EQ_MISSING, NE_MISSING)
    comptime for k in range(len(ops)):
        comptime op = ops[k]
        var actual = binary[op](a, b, mask)
        comptime if op == FILL_NULL or op == KEEP_NULLS:
            var payload = List[Int]()
            comptime if op == FILL_NULL:
                payload = [2, 4, 0, 87, 7, 9, 5, 85]
            else:
                payload = [9, 4, 88, 87, 7, 86, 5, 85]
            var valid = List[Bool]()
            comptime if op == FILL_NULL:
                valid = [True, True, True, False, True, True, True, False]
            else:
                valid = [True, False, False, False, True, False, False, False]
            var values = List[Scalar[D]](capacity=41)
            for row in range(41):
                values.append(Scalar[D](payload[(offset + row) % 8]))
            _same(
                actual,
                Series(
                    "x",
                    Column[Scalar[D]](
                        values^, _repeat_bools(valid, offset, 41)
                    ),
                ),
            )
            # These operators copy selected raw values even under a null;
            # unlike arithmetic, they have no zero-null-payload rule.
            var raw = actual.numeric[D]()
            for row in range(41):
                assert_equal(
                    raw._get(row), Scalar[D](payload[(offset + row) % 8])
                )
        else:
            var flags = List[Bool]()
            comptime if op == EQ_MISSING:
                flags = [False, False, False, True, True, False, False, True]
            else:
                flags = [True, True, True, False, False, True, True, False]
            _bool_result(
                actual,
                Series("x", BoolColumn(_repeat_bools(flags, offset, 41))),
            )
            assert_equal(actual.null_count(), 0)
        _same(actual, binary[op](flat_a, flat_b, mask))
    var valid: List[Bool] = [True, False, True, False, True, True, False, False]
    var missing: List[Bool] = [
        False,
        True,
        False,
        True,
        False,
        False,
        True,
        True,
    ]
    _bool_result(
        unary[IS_NULL](a, 0, mask),
        Series("x", BoolColumn(_repeat_bools(missing, offset, 41))),
    )
    _bool_result(
        unary[IS_NOT_NULL](a, 0, mask),
        Series("x", BoolColumn(_repeat_bools(valid, offset, 41))),
    )
    _same(unary[IS_NULL](a, 0, mask), unary[IS_NULL](flat_a, 0, mask))
    _same(unary[IS_NOT_NULL](a, 0, mask), unary[IS_NOT_NULL](flat_a, 0, mask))
    _cold(a)
    _cold(b)


def test_numeric_fill_keep_and_missing_cover_all_widths_and_bitmap_offsets() raises:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        for offset in range(8):
            _numeric_windows[D](offset)


def _numeric_scalars[D: DType]() raises:
    var flat = _numeric_source[D](False).slice(0, 8)
    var rows = _pieces(flat, [1, 1, 1, 1, 1, 1, 1, 1])
    var scalar = Series("x", Column[Scalar[D]]([Scalar[D](7)]))
    var null_scalar = Series("x", Column[Scalar[D]]([Scalar[D](77)], [False]))
    var valid: List[Bool] = [True, False, True, False, True, True, False, False]
    var missing: List[Bool] = [
        False,
        True,
        False,
        True,
        False,
        False,
        True,
        True,
    ]
    var values = List[Scalar[D]]()
    for value in [2, 7, 0, 7, 7, 9, 7, 7]:
        values.append(Scalar[D](value))
    _same(
        binary[FILL_NULL](rows, scalar), Series("x", Column[Scalar[D]](values^))
    )
    values = List[Scalar[D]]()
    for value in [2, 77, 0, 77, 7, 9, 77, 77]:
        values.append(Scalar[D](value))
    _same(
        binary[FILL_NULL](rows, null_scalar),
        Series("x", Column[Scalar[D]](values^, valid)),
    )
    _same(binary[FILL_NULL](null_scalar, rows), flat)
    _same(binary[KEEP_NULLS](scalar, rows), flat)
    var hidden = binary[KEEP_NULLS](rows, null_scalar)
    assert_equal(hidden.null_count(), 8)
    var raw = hidden.numeric[D]()
    for row in range(8):
        assert_equal(raw._get(row), Scalar[D](77))
    _bool_result(
        binary[EQ_MISSING](rows, null_scalar), Series("x", BoolColumn(missing))
    )
    _bool_result(
        binary[NE_MISSING](rows, null_scalar), Series("x", BoolColumn(valid))
    )
    _bool_result(
        binary[EQ_MISSING](scalar, rows),
        Series(
            "x",
            BoolColumn([False, False, False, False, True, False, False, False]),
        ),
    )
    _cold(rows)


def test_numeric_scalars_broadcast_whole_operands_not_single_row_chunks() raises:
    comptime for k in range(len(NUMERIC_DTYPES)):
        _numeric_scalars[NUMERIC_DTYPES[k]]()


def _boolean_source(right: Bool) raises -> Series:
    # Invalid rows deliberately contain a true value bit.
    var values = List[Bool]()
    if right:
        values = [True, False, True, True, False, True, True, False, True]
    else:
        values = [True, True, True, False, False, False, True, True, True]
    var valid = List[Bool]()
    if right:
        valid = [True, True, False, True, True, False, True, True, False]
    else:
        valid = [True, True, True, True, True, True, False, False, False]
    return Series(
        "x",
        BoolColumn(_repeat_bools(values, 0, 60), _repeat_bools(valid, 0, 60)),
    )


def _boolean_expected(op: Int, offset: Int, n: Int) raises -> Series:
    var values = List[Bool]()
    var valid = List[Bool](length=9, fill=True)
    if op == AND:
        values = [True, False, False, False, False, False, False, False, False]
        valid = [True, True, False, True, True, True, False, True, False]
    elif op == OR:
        values = [True, True, True, True, False, False, True, False, False]
        valid = [True, True, True, True, True, False, True, False, False]
    elif op == XOR:
        values = [False, True, False, True, False, False, False, False, False]
        valid = [True, True, False, True, True, False, False, False, False]
    elif op == FILL_NULL:
        values = [True, True, True, False, False, False, True, False, False]
        valid = [True, True, True, True, True, True, True, True, False]
    elif op == KEEP_NULLS:
        values = [True, False, False, True, False, False, False, False, False]
        valid = [True, True, False, True, True, False, False, False, False]
    elif op == EQ_MISSING:
        values = [True, False, False, False, True, False, False, False, True]
    elif op == NE_MISSING:
        values = [False, True, True, True, False, True, True, True, False]
    elif op == NOT:
        values = [False, False, False, True, True, True, False, False, False]
        valid = [True, True, True, True, True, True, False, False, False]
    elif op == IS_NULL:
        values = [False, False, False, False, False, False, True, True, True]
    else:
        assert_equal(op, IS_NOT_NULL)
        values = [True, True, True, True, True, True, False, False, False]
    return Series(
        "x",
        BoolColumn(
            _repeat_bools(values, offset, n), _repeat_bools(valid, offset, n)
        ),
    )


def _boolean_windows(offset: Int, mask: List[Bool] = List[Bool]()) raises:
    var flat_a = _boolean_source(False).slice(offset, 41)
    var flat_b = _boolean_source(True).slice(offset, 41)
    var a = _pieces(flat_a, [3, 17, 0, 21])
    var b = _pieces(flat_b, [4, 1, 15, 0, 21])
    comptime ops = (AND, OR, XOR, FILL_NULL, KEEP_NULLS, EQ_MISSING, NE_MISSING)
    comptime for k in range(len(ops)):
        comptime op = ops[k]
        var actual = binary[op](a, b, mask)
        _bool_result(actual, _boolean_expected(op, offset, 41))
        _same(actual, binary[op](flat_a, flat_b, mask))
    comptime predicates = (NOT, IS_NULL, IS_NOT_NULL)
    comptime for k in range(len(predicates)):
        comptime op = predicates[k]
        var actual = unary[op](a, 0, mask)
        _bool_result(actual, _boolean_expected(op, offset, 41))
        _same(actual, unary[op](flat_a, 0, mask))
    _cold(a)
    _cold(b)


def test_boolean_truth_tables_cross_packed_boundaries_with_invalid_true_bits() raises:
    for offset in range(8):
        _boolean_windows(offset)


def test_boolean_null_scalars_keep_dominance_and_physical_row_order() raises:
    var flat = _boolean_source(False).slice(0, 9)
    var rows = _pieces(flat, [1, 1, 1, 1, 1, 1, 1, 1, 1])
    var yes = Series("x", BoolColumn([True]))
    var no = Series("x", BoolColumn([False]))
    var unknown = Series("x", BoolColumn([True], [False]))
    _bool_result(
        binary[AND](rows, no),
        Series("x", BoolColumn(List[Bool](length=9, fill=False))),
    )
    _bool_result(
        binary[OR](yes, rows),
        Series("x", BoolColumn(List[Bool](length=9, fill=True))),
    )
    _bool_result(
        binary[XOR](unknown, rows),
        Series(
            "x",
            BoolColumn(
                List[Bool](length=9, fill=False),
                List[Bool](length=9, fill=False),
            ),
        ),
    )
    _bool_result(
        binary[FILL_NULL](rows, no),
        Series(
            "x",
            BoolColumn(
                [True, True, True, False, False, False, False, False, False]
            ),
        ),
    )
    _same(binary[FILL_NULL](unknown, rows), flat)
    _bool_result(
        binary[EQ_MISSING](rows, unknown),
        Series(
            "x",
            BoolColumn(
                [False, False, False, False, False, False, True, True, True]
            ),
        ),
    )
    _cold(rows)


def _float_windows[
    D: DType
](offset: Int, mask: List[Bool] = List[Bool]()) raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var av: List[Float64] = [nan, nan, 2, inf, -0.0, nan, nan, -inf, 0.0, inf]
    var bv: List[Float64] = [5, 7, 9, 1, 0.0, 88, nan, -inf, -0.0, -inf]
    var a_valid: List[Bool] = [
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    var b_valid: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    ]
    var a_values = List[Scalar[D]]()
    var b_values = List[Scalar[D]]()
    for row in range(60):
        a_values.append(av[row % 10].cast[D]())
        b_values.append(bv[row % 10].cast[D]())
    var flat_a = Series(
        "x", Column[Scalar[D]](a_values^, _repeat_bools(a_valid, 0, 60))
    ).slice(offset, 41)
    var flat_b = Series(
        "x", Column[Scalar[D]](b_values^, _repeat_bools(b_valid, 0, 60))
    ).slice(offset, 41)
    var a = _pieces(flat_a, [3, 17, 0, 21])
    var b = _pieces(flat_b, [4, 1, 15, 0, 21])
    comptime predicates = (IS_NAN, IS_NOT_NAN, IS_FINITE, IS_INFINITE)
    comptime for k in range(len(predicates)):
        comptime op = predicates[k]
        var flags = List[Bool]()
        comptime if op == IS_NAN:
            flags = [
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
            ]
        elif op == IS_NOT_NAN:
            flags = [
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
            ]
        elif op == IS_FINITE:
            flags = [
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
            ]
        else:
            flags = [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                True,
            ]
        var actual = unary[op](a, 0, mask)
        _bool_result(
            actual,
            Series(
                "x",
                BoolColumn(
                    _repeat_bools(flags, offset, 41),
                    _repeat_bools(a_valid, offset, 41),
                ),
            ),
        )
        _same(actual, unary[op](flat_a, 0, mask))
    var fill: List[Float64] = [5, nan, 2, inf, -0.0, 88, nan, -inf, 0.0, inf]
    var fill_valid: List[Bool] = [
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    ]
    var want = List[Scalar[D]]()
    for row in range(41):
        want.append(fill[(offset + row) % 10].cast[D]())
    var actual = binary[FILL_NAN](a, b, mask)
    _same(
        actual,
        Series(
            "x", Column[Scalar[D]](want^, _repeat_bools(fill_valid, offset, 41))
        ),
    )
    _same(actual, binary[FILL_NAN](flat_a, flat_b, mask))
    var raw = actual.numeric[D]()
    for row in range(41):
        if (offset + row) % 10 == 1:
            assert_true(isnan(raw._get(row)))
        elif (offset + row) % 10 == 5:
            assert_equal(raw._get(row), Scalar[D](88))
    comptime missing_ops = (EQ_MISSING, NE_MISSING)
    comptime for k in range(len(missing_ops)):
        comptime op = missing_ops[k]
        var flags = List[Bool]()
        comptime if op == EQ_MISSING:
            flags = [
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                True,
                True,
                False,
            ]
        else:
            flags = [
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                False,
                False,
                True,
            ]
        _bool_result(
            binary[op](a, b, mask),
            Series("x", BoolColumn(_repeat_bools(flags, offset, 41))),
        )
        _same(binary[op](a, b, mask), binary[op](flat_a, flat_b, mask))
    _cold(a)
    _cold(b)


def _float_scalars[D: DType]() raises:
    var nan = (Float64(0) / Float64(0)).cast[D]()
    var flat = Series(
        "x", Column[Scalar[D]]([nan, 2, nan, -0.0], [True, True, False, True])
    )
    var rows = _pieces(flat, [1, 1, 1, 1])
    var scalar = Series("x", Column[Scalar[D]]([7]))
    var hidden = Series("x", Column[Scalar[D]]([88], [False]))
    _same(
        binary[FILL_NAN](rows, scalar),
        Series(
            "x", Column[Scalar[D]]([7, 2, nan, -0.0], [True, True, False, True])
        ),
    )
    _same(
        binary[FILL_NAN](rows, hidden),
        Series(
            "x",
            Column[Scalar[D]]([88, 2, nan, -0.0], [False, True, False, True]),
        ),
    )
    _same(binary[FILL_NAN](Series("x", Column[Scalar[D]]([nan])), rows), flat)
    _same(
        binary[FILL_NAN](hidden, rows),
        Series(
            "x",
            Column[Scalar[D]]([88, 88, 88, 88], [False, False, False, False]),
        ),
    )
    _cold(rows)


def test_float_predicates_fill_nan_and_missing_preserve_ieee_edges() raises:
    _float_scalars[DType.float32]()
    _float_scalars[DType.float64]()
    for offset in range(8):
        _float_windows[DType.float32](offset)
        _float_windows[DType.float64](offset)


def _null_windows(flat: Series, seed: List[Bool]) raises:
    for offset in range(8):
        var control = flat.slice(offset, 41)
        var source = _pieces(control, [3, 17, 0, 21])
        var present = _repeat_bools(seed, offset, 41)
        var absent = List[Bool]()
        for row in range(41):
            absent.append(not present[row])
        _bool_result(
            unary[IS_NULL](source, 0), Series("x", BoolColumn(absent^))
        )
        _bool_result(
            unary[IS_NOT_NULL](source, 0), Series("x", BoolColumn(present^))
        )
        _same(unary[IS_NULL](source, 0), unary[IS_NULL](control, 0))
        _same(unary[IS_NOT_NULL](source, 0), unary[IS_NOT_NULL](control, 0))
        _cold(source)


def test_null_predicates_read_parent_validity_across_storage_and_logical_tags() raises:
    from dataframe.storage.column import _pack_bits

    var seed: List[Bool] = [True, False, True, False, True, True, False, False]
    var valid = _repeat_bools(seed, 0, 60)
    var strings = List[String]()
    var bytes = List[List[UInt8]]()
    for row in range(60):
        strings.append("red" if row % 2 == 0 else "blue")
        bytes.append([UInt8(255), 0, UInt8(row)])
    var text = Series("x", StringColumn(strings^, valid))
    _null_windows(text, seed)
    _null_windows(Series.binary("x", bytes^, valid), seed)
    _null_windows(text.cast("categorical"), seed)
    var numeric = _numeric_source[DType.int64](False)
    _null_windows(numeric.with_dtype(DataType.DATE), seed)
    _null_windows(numeric.with_dtype(DataType.TIME), seed)
    _null_windows(numeric.with_dtype(DataType.datetime("us", "UTC")), seed)
    _null_windows(numeric.with_dtype(DataType.duration("ms")), seed)
    _null_windows(numeric.cast(DataType.decimal(9, 2, 32)), seed)
    _null_windows(numeric.cast(DataType.decimal(18, 2, 64)), seed)
    _null_windows(numeric.cast(DataType.decimal(30, 2, 128)), seed)
    var nulls = Series(
        "x",
        Column[Int64](
            List[Int64](length=60, fill=0), List[Bool](length=60, fill=False)
        ),
    ).with_dtype(DataType.NULL)
    _null_windows(nulls, [False])
    var child_flat = Series(
        "item", Column[Int64](List[Int64](length=61440, fill=123))
    )
    var child = _pieces(child_flat, [1000, 30000, 30440])
    var offsets = List[Int64]()
    for row in range(61):
        offsets.append(Int64(row * 1024))
    var nested = Series(
        "x", ListColumn(offsets^, child.copy(), _pack_bits(valid))
    )
    _null_windows(nested, seed)
    var field = _pieces(numeric.renamed("number"), [7, 22, 31])
    var record = Series(
        "x",
        StructColumn(
            [field.copy(), nested.renamed("large")], _pack_bits(valid)
        ),
    )
    _null_windows(record, seed)
    _cold(child)
    _cold(field)


def test_keep_nulls_generic_left_uses_validity_for_boolean_output() raises:
    var flat = _numeric_source[DType.int64](False).slice(0, 8)
    var strings = Series(
        "x",
        StringColumn(
            ["a", "b", "a", "b", "a", "b", "a", "b"],
            [True, False, True, False, True, True, False, False],
        ),
    )
    var sources = List[Series]()
    sources.append(flat.copy())
    sources.append(flat.with_dtype(DataType.DATE))
    sources.append(strings.copy())
    sources.append(strings.cast("categorical"))
    sources.append(
        Series.binary(
            "x",
            [
                [UInt8(255)],
                [UInt8(0)],
                [UInt8(254)],
                [UInt8(0)],
                [UInt8(253)],
                [UInt8(0)],
                [UInt8(252)],
                [UInt8(0)],
            ],
            [True, False, True, False, True, True, False, False],
        )
    )
    var right_flat = Series(
        "b",
        BoolColumn(
            [True, False, True, True, False, True, False, True],
            [True, True, False, True, True, False, True, False],
        ),
    )
    var right = _pieces(right_flat, [1, 3, 4])
    var expected = Series(
        "x",
        BoolColumn(
            [True, False, False, False, False, False, False, False],
            [True, False, False, False, True, False, False, False],
        ),
    )
    for source in sources:
        var left = _pieces(source, [3, 1, 4])
        _bool_result(binary[KEEP_NULLS](left, right), expected)
        _same(
            binary[KEEP_NULLS](left, right),
            binary[KEEP_NULLS](source, right_flat),
        )
        _bool_result(binary[KEEP_NULLS](source.slice(0, 1), right), right_flat)
        _bool_result(
            binary[KEEP_NULLS](source.slice(1, 1), right),
            Series(
                "x",
                BoolColumn(
                    List[Bool](length=8, fill=False),
                    List[Bool](length=8, fill=False),
                ),
            ),
        )
        _cold(left)
    _cold(right)
    var null_flat = Series(
        "x",
        Column[Int64](
            List[Int64](length=8, fill=0), List[Bool](length=8, fill=False)
        ),
    ).with_dtype(DataType.NULL)
    var nulls = _pieces(null_flat, [3, 5])
    var absent = Series(
        "x",
        BoolColumn(
            List[Bool](length=8, fill=False), List[Bool](length=8, fill=False)
        ),
    )
    _bool_result(binary[KEEP_NULLS](nulls, right), absent)
    _bool_result(binary[KEEP_NULLS](null_flat.slice(0, 1), right), absent)
    _cold(nulls)
    _cold(right)


def test_observation_masks_do_not_change_validity_kernel_semantics() raises:
    var masks = List[List[Bool]]()
    masks.append(List[Bool](length=41, fill=False))
    masks.append([False, True])
    for mask in masks:
        _numeric_windows[DType.int64](0, mask)
        _numeric_windows[DType.float64](3, mask)
        _boolean_windows(1, mask)
        _float_windows[DType.float64](5, mask)
        var scalar = Series("x", BoolColumn([True], [False]))
        _bool_result(
            unary[IS_NULL](scalar, 0, mask), Series("x", BoolColumn([True]))
        )
        _bool_result(
            unary[NOT](scalar, 0, mask),
            Series("x", BoolColumn([False], [False])),
        )
        _bool_result(
            binary[OR](scalar, Series("x", BoolColumn([True])), mask),
            Series("x", BoolColumn([True])),
        )


def test_empty_shapes_validate_lengths_before_returning_empty_results() raises:
    var empty = Series("x", Column[Int64]([]))
    var one = Series("x", Column[Int64]([7]))
    var two = Series("x", Column[Int64]([7, 8]))
    comptime ops = (FILL_NULL, KEEP_NULLS, EQ_MISSING, NE_MISSING)
    comptime for k in range(len(ops)):
        comptime op = ops[k]
        assert_equal(len(binary[op](empty, one)), 0)
        assert_equal(len(binary[op](one, empty)), 0)
        assert_equal(len(binary[op](empty, empty)), 0)
        with assert_raises(contains="Incompatible"):
            _ = binary[op](empty, two)
        with assert_raises(contains="Incompatible"):
            _ = binary[op](two, empty)
    var be = Series("x", BoolColumn([]))
    var bo = Series("x", BoolColumn([True]))
    var bt = Series("x", BoolColumn([True, False]))
    comptime bops = (
        AND,
        OR,
        XOR,
        FILL_NULL,
        KEEP_NULLS,
        EQ_MISSING,
        NE_MISSING,
    )
    comptime for k in range(len(bops)):
        comptime op = bops[k]
        assert_equal(len(binary[op](be, bo)), 0)
        assert_equal(len(binary[op](bo, be)), 0)
        with assert_raises(contains="Incompatible"):
            _ = binary[op](be, bt)
    assert_equal(unary[IS_NULL](empty, 0).dtype(), DataType.BOOL)
    assert_equal(len(unary[IS_NOT_NULL](empty, 0)), 0)
    assert_equal(len(unary[NOT](be, 0)), 0)
    var fe = Series("x", Column[Float64]([]))
    var fo = Series("x", Column[Float64]([1]))
    assert_equal(len(binary[FILL_NAN](fe, fo)), 0)
    comptime predicates = (IS_NAN, IS_NOT_NAN, IS_FINITE, IS_INFINITE)
    comptime for k in range(len(predicates)):
        comptime op = predicates[k]
        assert_equal(unary[op](fe, 0).dtype(), DataType.BOOL)
        assert_equal(len(unary[op](fe, 0)), 0)


def test_eager_and_lazy_complete_validity_expressions_have_literal_outputs() raises:
    var a = _pieces(
        _numeric_source[DType.int64](False).slice(0, 8).renamed("a"), [3, 1, 4]
    )
    var b = _pieces(
        _numeric_source[DType.int64](True).slice(0, 8).renamed("b"), [1, 5, 2]
    )
    var p = _pieces(_boolean_source(False).slice(0, 8).renamed("p"), [2, 1, 5])
    var q = _pieces(_boolean_source(True).slice(0, 8).renamed("q"), [4, 4])
    var frame = DataFrame([a.copy(), b.copy(), p.copy(), q.copy()])
    var expressions: List[Expr] = [
        col("a").is_null().alias("null"),
        col("a").fill_null(col("b")).alias("fill"),
        col("a").eq_missing(col("b")).alias("equal"),
        (~col("p")).alias("not"),
        (col("p") & col("q")).alias("and"),
        (col("p") | col("q")).alias("or"),
        (col("p") ^ col("q")).alias("xor"),
    ]
    var expected = DataFrame(
        [
            Series(
                "null",
                BoolColumn(
                    [False, True, False, True, False, False, True, True]
                ),
            ),
            Series(
                "fill",
                Column[Int64](
                    [2, 4, 0, 87, 7, 9, 5, 85],
                    [True, True, True, False, True, True, True, False],
                ),
            ),
            Series(
                "equal",
                BoolColumn(
                    [False, False, False, True, True, False, False, True]
                ),
            ),
            _boolean_expected(NOT, 0, 8).renamed("not"),
            _boolean_expected(AND, 0, 8).renamed("and"),
            _boolean_expected(OR, 0, 8).renamed("or"),
            _boolean_expected(XOR, 0, 8).renamed("xor"),
        ]
    )
    assert_frame(frame.select_exprs(expressions, batch_size=3), expected)
    assert_frame(
        frame.lazy()
        .select_exprs(expressions)
        .collect(engine="cpu", batch_size=3),
        expected,
    )
    _cold(a)
    _cold(b)
    _cold(p)
    _cold(q)


def test_public_schema_errors_precede_empty_and_inactive_evaluation() raises:
    for n in range(2):
        var frame = DataFrame(
            [
                Series("i", Column[Int64](List[Int64](length=n, fill=1))),
                Series("f", Column[Float64](List[Float64](length=n, fill=1))),
                Series("b", BoolColumn(List[Bool](length=n, fill=True))),
            ]
        )
        with assert_raises(contains="bool operand"):
            _ = frame.select_exprs([~col("i")])
        with assert_raises(contains="bool operands"):
            _ = frame.select_exprs([col("i") & col("i")])
        with assert_raises(contains="float operand"):
            _ = frame.select_exprs([col("i").is_finite()])
        with assert_raises(contains="float operands"):
            _ = frame.select_exprs([col("i").fill_nan(col("i"))])
        with assert_raises(contains="matching dtypes"):
            _ = frame.select_exprs([col("i").fill_null(col("f"))])
        with assert_raises(contains="float operand"):
            _ = frame.select_exprs(
                [when(lit(False)).then(col("i").is_nan()).otherwise(lit(False))]
            )
        var offsets = List[Int64](length=n + 1, fill=0)
        var nested = Series(
            "nested", ListColumn(offsets^, Series("item", Column[Int64]([])))
        )
        var structs = Series(
            "record",
            StructColumn(
                [Series("field", Column[Int64](List[Int64](length=n, fill=1)))]
            ),
        )
        var nested_frame = DataFrame(
            [
                nested^,
                structs^,
                Series("b", BoolColumn(List[Bool](length=n, fill=True))),
            ]
        )
        with assert_raises(contains="does not support list or struct"):
            _ = nested_frame.select_exprs(
                [col("nested")._binary(col("b"), KEEP_NULLS)]
            )
        with assert_raises(contains="does not support list or struct"):
            _ = nested_frame.select_exprs(
                [col("record")._binary(col("b"), KEEP_NULLS)]
            )


def _owned_outputs() raises -> List[Series]:
    var a = _pieces(
        Series(
            "x",
            Column[Int64]([2, 99, 4, 98, 6], [True, False, True, False, True]),
        ),
        [2, 1, 2],
    )
    var b = _pieces(
        Series(
            "x",
            Column[Int64]([7, 8, 77, 66, 9], [True, True, False, False, True]),
        ),
        [1, 3, 1],
    )
    var p = _pieces(
        Series(
            "x",
            BoolColumn(
                [True, False, True, False, True],
                [True, False, True, True, False],
            ),
        ),
        [1, 4],
    )
    var q = _pieces(
        Series(
            "x",
            BoolColumn(
                [False, True, True, True, False],
                [True, True, False, True, True],
            ),
        ),
        [3, 2],
    )
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var f = _pieces(
        Series(
            "x",
            Column[Float64](
                [nan, inf, nan, -0.0, 2], [True, True, False, True, True]
            ),
        ),
        [2, 3],
    )
    var results = List[Series]()
    results.append(binary[FILL_NULL](a, b))
    results.append(binary[KEEP_NULLS](a, b))
    results.append(unary[IS_NULL](a, 0))
    results.append(unary[NOT](p, 0))
    results.append(binary[AND](p, q))
    results.append(unary[IS_FINITE](f, 0))
    _cold(a)
    _cold(b)
    _cold(p)
    _cold(q)
    _cold(f)
    return results^


def test_validity_results_own_buffers_after_sources_are_destroyed() raises:
    var outputs = _owned_outputs()
    _same(
        outputs[0],
        Series(
            "x",
            Column[Int64]([2, 8, 4, 66, 6], [True, True, True, False, True]),
        ),
    )
    _same(
        outputs[1],
        Series(
            "x",
            Column[Int64]([7, 8, 77, 66, 9], [True, False, False, False, True]),
        ),
    )
    _bool_result(
        outputs[2], Series("x", BoolColumn([False, True, False, True, False]))
    )
    _bool_result(
        outputs[3],
        Series(
            "x",
            BoolColumn(
                [False, False, False, True, False],
                [True, False, True, True, False],
            ),
        ),
    )
    _bool_result(
        outputs[4],
        Series(
            "x",
            BoolColumn(
                [False, False, False, False, False],
                [True, False, False, True, True],
            ),
        ),
    )
    _bool_result(
        outputs[5],
        Series(
            "x",
            BoolColumn(
                [False, False, False, True, True],
                [True, True, False, True, True],
            ),
        ),
    )


def _integer_extremes[D: DType]() raises:
    var low = Scalar[D].MIN
    var high = Scalar[D].MAX
    var av: List[Bool] = [True, True, True, False, False]
    var bv: List[Bool] = [True, True, True, True, False]
    var flat_a = Series("x", Column[Scalar[D]]([low, high, 1, high, low], av))
    var flat_b = Series("x", Column[Scalar[D]]([low, low, 1, high, low], bv))
    var a = _pieces(flat_a, [1, 3, 1])
    var b = _pieces(flat_b, [3, 2])
    _bool_result(
        binary[EQ_MISSING](a, b),
        Series("x", BoolColumn([True, False, True, False, True])),
    )
    _bool_result(
        binary[NE_MISSING](a, b),
        Series("x", BoolColumn([False, True, False, True, False])),
    )
    _same(
        binary[FILL_NULL](a, b),
        Series(
            "x",
            Column[Scalar[D]](
                [low, high, 1, high, low], [True, True, True, True, False]
            ),
        ),
    )
    _same(binary[EQ_MISSING](a, b), binary[EQ_MISSING](flat_a, flat_b))
    _cold(a)
    _cold(b)


def test_missing_comparisons_preserve_full_integer_precision() raises:
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime D = NUMERIC_DTYPES[k]
        comptime if D.is_integral():
            _integer_extremes[D]()
    var a = _pieces(
        Series("x", Column[UInt64]([UInt64(9007199254740992), UInt64.MAX, 7])),
        [1, 2],
    )
    var b = _pieces(
        Series(
            "x", Column[UInt64]([UInt64(9007199254740993), UInt64.MAX - 1, 7])
        ),
        [2, 1],
    )
    _bool_result(
        binary[EQ_MISSING](a, b), Series("x", BoolColumn([False, False, True]))
    )
    _bool_result(
        binary[NE_MISSING](a, b), Series("x", BoolColumn([True, True, False]))
    )
    _cold(a)
    _cold(b)
