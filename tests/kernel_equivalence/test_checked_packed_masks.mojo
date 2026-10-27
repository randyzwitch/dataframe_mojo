"""Checked integers observe only the intersection of validity and selection."""
# contract-group: expr.arithmetic
from std.testing import assert_equal, assert_raises
from dataframe import Column, Series
from dataframe.expr import ADD, SUB, MUL, ABS, NEG
from dataframe.expr_kernels import binary, _unary_int


def _input[
    D: DType
](n: Int, offset: Int, mode: Int, side: Int) raises -> Column[Scalar[D]]:
    var values = List[Scalar[D]](capacity=n + offset)
    var flags = List[Bool](capacity=n + offset)
    for row in range(n + offset):
        var present = (
            mode == 0 or mode == 3 or (mode == 2 and (row + side) % 3 != 0)
        )
        var value = row % 7 + 10 if side == 0 else row % 7 + 1
        values.append(Scalar[D](value) if present else Scalar[D].MAX)
        flags.append(present)
    var column = Column[Scalar[D]](values^) if mode == 0 else Column[Scalar[D]](
        values^, flags
    )
    return column.slice(offset, n)


def _check[
    op: Int, D: DType
](a: Column[Scalar[D]], b: Column[Scalar[D]], n: Int, mask: List[Bool]) raises:
    var result = binary[op](Series("a", a.copy()), Series("b", b.copy()), mask)
    ref out = result._data[Column[Scalar[D]]]
    assert_equal(len(out), n)
    for row in range(n):
        var active = True
        if len(mask) == n:
            active = mask[row]
        elif len(mask) > 0:
            active = mask[0]
            if n == 1:
                active = False
                for flag in mask:
                    active |= flag
        var ai = 0 if len(a) == 1 else row
        var bi = 0 if len(b) == 1 else row
        var present = active and a._valid(ai) and b._valid(bi)
        assert_equal(out._valid(row), present)
        if not present:
            assert_equal(out._get(row), Scalar[D](0))
            continue
        var x = Int128(a._get(ai))
        var y = Int128(b._get(bi))
        comptime if op == ADD:
            assert_equal(Int128(out._get(row)), x + y)
        elif op == SUB:
            assert_equal(Int128(out._get(row)), x - y)
        else:
            assert_equal(Int128(out._get(row)), x * y)


def _layouts[D: DType]() raises:
    for n in [0, 1, 7, 15, 16, 17, 31, 32, 33, 65]:
        for lo in range(8):
            for ro in [0, 3, 7]:
                for lm in range(4):
                    for rm in range(4):
                        for broadcast in range(3):
                            if n == 0 and broadcast != 0:
                                continue
                            var a = _input[D](
                                1 if broadcast == 1 else n, lo, lm, 0
                            )
                            var b = _input[D](
                                1 if broadcast == 2 else n, ro, rm, 1
                            )
                            for selection in range(5):
                                var mask = List[Bool]()
                                if selection == 1:
                                    for row in range(n):
                                        mask.append(row % 3 != 0)
                                elif selection == 2:
                                    mask = [True]
                                elif selection == 3:
                                    mask = [False]
                                elif selection == 4:
                                    mask = [False, True, False]
                                _check[ADD, D](a, b, n, mask)
                                _check[SUB, D](a, b, n, mask)
                                _check[MUL, D](a, b, n, mask)


def test_checked_packed_masks_match_wide_scalar_results_for_slices_broadcasts_and_simd_tails() raises:
    _layouts[DType.int8]()
    _layouts[DType.int16]()
    _layouts[DType.int32]()
    _layouts[DType.int64]()
    _layouts[DType.uint64]()


def _overflow[op: Int, D: DType]() raises:
    for bad in [0, 15, 16, 17, 31, 32]:
        var xs = List[Scalar[D]](length=33, fill=2)
        var ys = List[Scalar[D]](length=33, fill=1)
        xs[bad] = Scalar[D].MIN if op == SUB else Scalar[D].MAX
        ys[bad] = Scalar[D](2 if op == MUL else 1)
        var mask = List[Bool](length=33, fill=True)
        mask[bad] = False
        var flags = List[Bool](length=33, fill=True)
        flags[bad] = False
        var a = Column[Scalar[D]](xs.copy())
        var b = Column[Scalar[D]](ys.copy())
        _check[op, D](a, b, 33, mask)
        _check[op, D](Column[Scalar[D]](xs.copy(), flags), b, 33, List[Bool]())
        for observed in [List[Bool](), List[Bool](length=33, fill=True)]:
            with assert_raises(contains="overflow"):
                _ = binary[op](
                    Series("a", a.copy()), Series("b", b.copy()), observed
                )


def test_nullable_simd_overflow_checks_mask_poison_inactive_lanes_and_preserve_observed_errors() raises:
    _overflow[ADD, DType.int8]()
    _overflow[SUB, DType.int8]()
    _overflow[MUL, DType.int8]()
    _overflow[ADD, DType.int16]()
    _overflow[SUB, DType.int16]()
    _overflow[MUL, DType.int16]()
    _overflow[ADD, DType.int32]()
    _overflow[SUB, DType.int32]()
    _overflow[MUL, DType.int32]()


def _unary_masks[op: Int, D: DType]() raises:
    var input = Column[Scalar[D]](
        [
            Scalar[D].MIN,
            Scalar[D].MIN,
            Scalar[D](-3),
            Scalar[D](5),
            Scalar[D](5),
        ],
        [True, False, True, True, True],
    )
    var result = _unary_int[op, D](input, [False, True, True, False, True])
    ref out = result._data[Column[Scalar[D]]]
    for row in [0, 1, 3]:
        assert_equal(out._valid(row), False)
        assert_equal(out._get(row), Scalar[D](0))
    assert_equal(out._valid(2), True)
    assert_equal(out._get(2), Scalar[D](3))
    assert_equal(out._valid(4), True)
    assert_equal(out._get(4), Scalar[D](5 if op == ABS else -5))
    var scalar = Column[Scalar[D]]([Scalar[D].MIN])
    var unseen = _unary_int[op, D](scalar, [False, False, False])
    assert_equal(unseen._data[Column[Scalar[D]]]._valid(0), False)
    assert_equal(unseen._data[Column[Scalar[D]]]._get(0), Scalar[D](0))
    with assert_raises(contains="overflow"):
        _ = _unary_int[op, D](scalar, [False, True, False])


def test_integer_unary_masks_preserve_nulls_and_scalar_error_observation() raises:
    _unary_masks[ABS, DType.int8]()
    _unary_masks[NEG, DType.int8]()
    _unary_masks[ABS, DType.int16]()
    _unary_masks[NEG, DType.int16]()
    _unary_masks[ABS, DType.int32]()
    _unary_masks[NEG, DType.int32]()
    _unary_masks[ABS, DType.int64]()
    _unary_masks[NEG, DType.int64]()
    _unary_masks[ABS, DType.int128]()
    _unary_masks[NEG, DType.int128]()
