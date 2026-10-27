"""Numeric kernels preserve sliced bitmaps, broadcasts and observed errors."""
# contract-group: expr.arithmetic
# contract-group: expr.horizontal
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import (
    Column,
    DataFrame,
    Expr,
    Series,
    col,
    lit,
    when,
    mean_horizontal,
)
from dataframe.expr import ADD, SUB, MUL, DIV, NEG, ABS
from dataframe.expr_kernels import binary, unary
from dataframe.horizontal import _float_fold, _extreme
from tests.support.coverage_generators import chunk_layout


def _column[
    D: DType
](n: Int, offset: Int, mode: Int, side: Int) raises -> Column[Scalar[D]]:
    var values = List[Scalar[D]](capacity=n + offset)
    var valid = List[Bool](capacity=n + offset)
    for i in range(n + offset):
        values.append(Scalar[D]((i * 7 + side * 3) % 19 - 9))
        valid.append(
            mode == 0 or mode == 3 or (mode == 2 and (i + side) % 3 != 0)
        )
    var column = Column[Scalar[D]](values^) if mode == 0 else Column[Scalar[D]](
        values^, valid
    )
    return column.slice(offset, n)


def _check_binary[
    op: Int, D: DType, W: Int
](a: Column[Scalar[D]], b: Column[Scalar[D]], n: Int) raises:
    var result = binary[op, W](Series("a", a.copy()), Series("b", b.copy()))
    ref out = result._data[Column[Scalar[D]]]
    assert_equal(len(out), n)
    for i in range(n):
        var x = 0 if len(a) == 1 else i
        var y = 0 if len(b) == 1 else i
        var valid = a._valid(x) and b._valid(y)
        assert_equal(out._valid(i), valid)
        if not valid:
            assert_equal(out._get(i), Scalar[D](0))
            continue
        var expected: Scalar[D]
        comptime if op == ADD:
            expected = a._get(x) + b._get(y)
        elif op == SUB:
            expected = a._get(x) - b._get(y)
        elif op == MUL:
            expected = a._get(x) * b._get(y)
        else:
            expected = a._get(x) / b._get(y)
        if isnan(expected):
            assert_true(isnan(out._get(i)))
        else:
            comptime if D == DType.float32:
                assert_equal(
                    bitcast[DType.uint32](out._get(i)),
                    bitcast[DType.uint32](expected),
                )
            else:
                assert_equal(
                    bitcast[DType.uint64](out._get(i)),
                    bitcast[DType.uint64](expected),
                )


def _check_layouts[D: DType]() raises:
    for n in [0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 63, 64, 65, 129]:
        for lo in range(8):
            for ro in [0, 3, 7]:
                for lm in range(4):
                    for rm in range(4):
                        for broadcast in range(3):
                            if n == 0 and broadcast != 0:
                                continue
                            var a = _column[D](
                                1 if broadcast == 1 else n, lo, lm, 0
                            )
                            var b = _column[D](
                                1 if broadcast == 2 else n, ro, rm, 1
                            )
                            _check_binary[ADD, D, 4](a, b, n)
                            _check_binary[SUB, D, 8](a, b, n)
                            _check_binary[MUL, D, 16](a, b, n)
                            _check_binary[DIV, D, 1](a, b, n)
                            var positive = unary[ABS](Series("a", a.copy()), 0)
                            var negative = unary[NEG, 8](
                                Series("a", a.copy()), 0
                            )
                            for i in range(len(a)):
                                assert_equal(
                                    positive._data[Column[Scalar[D]]]._valid(i),
                                    a._valid(i),
                                )
                                assert_equal(
                                    negative._data[Column[Scalar[D]]]._valid(i),
                                    a._valid(i),
                                )
                                if a._valid(i):
                                    assert_equal(
                                        positive._data[Column[Scalar[D]]]._get(
                                            i
                                        ),
                                        abs(a._get(i)),
                                    )
                                    assert_equal(
                                        negative._data[Column[Scalar[D]]]._get(
                                            i
                                        ),
                                        -a._get(i),
                                    )


def test_float_validity_all_bit_offsets_widths_null_patterns_and_broadcasts() raises:
    _check_layouts[DType.float32]()
    _check_layouts[DType.float64]()


def _edges[D: DType]() raises:
    var zero = Scalar[D](0)
    var nan = zero / zero
    var inf = Scalar[D](1) / zero
    var values: List[Scalar[D]] = [
        nan,
        inf,
        -inf,
        zero,
        -zero,
        Scalar[D](1),
        Scalar[D](-1),
        nan,
        -zero,
    ]
    var valid: List[Bool] = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    var a = Column[Scalar[D]](values.copy(), valid)
    for value in [zero, -zero, inf, -inf, nan, Scalar[D](2)]:
        var b = Column[Scalar[D]]([value])
        _check_binary[ADD, D, 4](a, b, len(a))
        _check_binary[SUB, D, 8](a, b, len(a))
        _check_binary[MUL, D, 16](a, b, len(a))
        _check_binary[DIV, D, 1](a, b, len(a))


def test_packed_float_validity_preserves_nan_infinity_and_signed_zero() raises:
    _edges[DType.float32]()
    _edges[DType.float64]()


def _horizontal[D: DType]() raises:
    for n in [0, 1, 7, 8, 9, 16, 17, 65]:
        for lo in range(8):
            for lm in range(4):
                for rm in range(4):
                    for scalar in [False, True]:
                        var a = _column[D](n, lo, lm, 0)
                        var b = _column[D](1 if scalar else n, 7, rm, 1)
                        var inputs: List[Series] = [
                            Series("a", a.copy()),
                            Series("b", b.copy()),
                        ]
                        var least = _extreme[D, False](inputs, n)
                        var most = _extreme[D, True](inputs, n)
                        for ignore in [False, True]:
                            var sum = _float_fold[D, False](inputs, n, ignore)
                            var mean = _float_fold[D, True](inputs, n, ignore)
                            for row in range(n):
                                var other = 0 if scalar else row
                                var av = a._valid(row)
                                var bv = b._valid(other)
                                var count = Int(av) + Int(bv)
                                var total = (
                                    a._get(row) if av else Scalar[D](0)
                                ) + (b._get(other) if bv else Scalar[D](0))
                                assert_equal(
                                    sum._data[Column[Scalar[D]]]._valid(row),
                                    ignore or count == 2,
                                )
                                assert_equal(
                                    mean._data[Column[Scalar[D]]]._valid(row),
                                    count > 0 if ignore else count == 2,
                                )
                                assert_equal(
                                    least._data[Column[Scalar[D]]]._valid(row),
                                    count > 0,
                                )
                                assert_equal(
                                    most._data[Column[Scalar[D]]]._valid(row),
                                    count > 0,
                                )
                                if ignore or count == 2:
                                    assert_equal(
                                        sum._data[Column[Scalar[D]]]._get(row),
                                        total,
                                    )
                                if count > 0 and (ignore or count == 2):
                                    assert_equal(
                                        mean._data[Column[Scalar[D]]]._get(row),
                                        total / Scalar[D](count),
                                    )
                                if count > 0:
                                    var low = min(
                                        a._get(row), b._get(other)
                                    ) if av and bv else (
                                        a._get(row) if av else b._get(other)
                                    )
                                    var high = max(
                                        a._get(row), b._get(other)
                                    ) if av and bv else (
                                        a._get(row) if av else b._get(other)
                                    )
                                    assert_equal(
                                        least._data[Column[Scalar[D]]]._get(
                                            row
                                        ),
                                        low,
                                    )
                                    assert_equal(
                                        most._data[Column[Scalar[D]]]._get(row),
                                        high,
                                    )


def test_horizontal_validity_uses_each_inputs_bit_offset_and_broadcast() raises:
    _horizontal[DType.float32]()
    _horizontal[DType.float64]()


def _plans[D: DType]() raises:
    var a = _column[D](129, 3, 2, 0)
    var b = _column[D](129, 7, 2, 1)
    var frame = DataFrame([Series("a", a.copy()), Series("b", b.copy())])
    var expressions: List[Expr] = [
        (col("a") + col("b")).alias("add"),
        col("a").abs().alias("abs"),
        mean_horizontal([col("a"), col("b")]).alias("mean"),
    ]
    for source in [frame.copy(), chunk_layout(frame)]:
        for result in [
            source.select_exprs(expressions),
            source.lazy().select_exprs(expressions).collect(engine="cpu"),
        ]:
            var addition_series = result.column("add").rechunk()
            var absolute_series = result.column("abs").rechunk()
            var mean_series = result.column("mean").rechunk()
            ref addition = addition_series._data[Column[Scalar[D]]]
            ref absolute = absolute_series._data[Column[Scalar[D]]]
            ref mean = mean_series._data[Column[Scalar[D]]]
            for row in range(129):
                var av = a._valid(row)
                var bv = b._valid(row)
                var count = Int(av) + Int(bv)
                assert_equal(addition._valid(row), av and bv)
                assert_equal(absolute._valid(row), av)
                assert_equal(mean._valid(row), count > 0)
                if av and bv:
                    assert_equal(addition._get(row), a._get(row) + b._get(row))
                if av:
                    assert_equal(absolute._get(row), abs(a._get(row)))
                if count > 0:
                    var total = (a._get(row) if av else Scalar[D](0)) + (
                        b._get(row) if bv else Scalar[D](0)
                    )
                    assert_equal(mean._get(row), total / Scalar[D](count))


def test_float_packed_validity_eager_and_lazy_plans_preserve_differing_chunks() raises:
    _plans[DType.float32]()
    _plans[DType.float64]()


def test_nullable_checked_arithmetic_observes_only_selected_rows_with_differing_chunks() raises:
    var frame = DataFrame(
        [
            Series(
                "a",
                Column[Int32](
                    [Int32.MAX, 4, Int32.MAX, 7, 9],
                    [True, True, False, True, True],
                ),
            ),
            Series("b", Column[Int32]([1, 2, 1, 3, 4])),
            Series("take", Column[Bool]([False, True, True, True, False])),
        ]
    )
    var expression = (
        when(col("take"))
        .then(col("a") + col("b"))
        .otherwise(lit(Int32(0)))
        .alias("out")
    )
    for source in [frame.copy(), chunk_layout(frame)]:
        for result in [
            source.select(expression),
            source.lazy().select(expression).collect(),
        ]:
            assert_equal(result.item(0, "out").int32(), Int32(0))
            assert_equal(result.item(1, "out").int32(), Int32(6))
            assert_true(result.item(2, "out").is_null())
            assert_equal(result.item(3, "out").int32(), Int32(10))
            assert_equal(result.item(4, "out").int32(), Int32(0))
        with assert_raises(contains="overflow"):
            _ = source.select(col("a") + col("b"))
