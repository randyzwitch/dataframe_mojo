"""Numeric inputs keep independent chunk boundaries and scalar broadcasts."""
# contract-group: expr.chunk_windows
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, BoolColumn, DataFrame, DataType, Series
from dataframe.kernels.binary import choose, binary
from dataframe.expr import (
    H_SUM,
    H_MEAN,
    H_MIN,
    H_MAX,
    ADD,
    SUB,
    MUL,
    DIV,
    FLOORDIV,
    MOD,
    POW,
    CLIP_LOW,
    CLIP_HIGH,
    GT,
    LT,
    GE,
    LE,
    EQ,
    NE,
)
from dataframe.kernels.horizontal import horizontal
from tests.support.coverage_compare import assert_frame


def _pieces(input: Series, sizes: List[Int]) raises -> Series:
    var parts = List[Series]()
    var start = 0
    for size in sizes:
        parts.append(input.slice(start, size))
        start += size
    assert_equal(start, len(input))
    return Series._from_chunks(parts)


def _numeric_choice[D: DType](offset: Int = 13) raises:
    var a = List[Scalar[D]]()
    var b = List[Scalar[D]]()
    var a_valid = List[Bool]()
    var b_valid = List[Bool]()
    for row in range(60):
        a.append(Scalar[D](row % 61 + 1))
        b.append(Scalar[D](100 - row % 17))
        a_valid.append(row % 7 != 0)
        b_valid.append(row % 5 != 0)
    var left = _pieces(
        Series("x", Column[Scalar[D]](a^, a_valid)).slice(offset, 41),
        [3, 0, 17, 1, 0, 20],
    )
    var right = _pieces(
        Series("x", Column[Scalar[D]](b^, b_valid)).slice(offset, 41),
        [7, 1, 0, 8, 19, 6],
    )
    var selected = List[Bool]()
    var want = List[Scalar[D]]()
    var valid = List[Bool]()
    for row in range(41):
        var take = row % 3 == 1
        var source = offset + row
        selected.append(take)
        want.append(Scalar[D](source % 61 + 1 if take else 100 - source % 17))
        valid.append(source % 7 != 0 if take else source % 5 != 0)
    var actual = choose(selected, left, right)
    assert_equal(left._chunked.value()[].cache[].state.load(), Int64(0))
    assert_equal(right._chunked.value()[].cache[].state.load(), Int64(0))
    assert_frame(
        DataFrame([actual.renamed("x")]),
        DataFrame([Series("x", Column[Scalar[D]](want^, valid))]),
    )


def test_numeric_choice_preserves_unequal_sliced_chunks_and_selected_nulls() raises:
    for offset in range(8):
        _numeric_choice[DType.int8](offset)
        _numeric_choice[DType.int16](offset)
        _numeric_choice[DType.int32](offset)
        _numeric_choice[DType.int64](offset)
        _numeric_choice[DType.uint8](offset)
        _numeric_choice[DType.uint16](offset)
        _numeric_choice[DType.uint32](offset)
        _numeric_choice[DType.uint64](offset)
        _numeric_choice[DType.float32](offset)
        _numeric_choice[DType.float64](offset)


def _owned_choice() raises -> Series:
    var input = _pieces(
        Series(
            "x", Column[Int64]([1, 2, 3, 4, 5], [True, True, False, True, True])
        ),
        [1, 1, 0, 3],
    )
    var null = Series("x", Column[Int64]([Int64.MIN], [False]))
    var actual = choose([True, False, True, False, False], null, input)
    assert_equal(input._chunked.value()[].cache[].state.load(), Int64(0))
    return actual^


def test_numeric_choice_broadcasts_whole_operands_and_owns_output() raises:
    var actual = _owned_choice()
    assert_frame(
        DataFrame([actual.renamed("x")]),
        DataFrame(
            [
                Series(
                    "x",
                    Column[Int64](
                        [0, 2, 0, 4, 5], [False, True, False, True, True]
                    ),
                )
            ]
        ),
    )
    var input = _pieces(Series("x", Column[Float32]([1, 2, 3])), [1, 0, 1, 1])
    var empty = choose(List[Bool](), input, Series("x", Column[Float32]([9])))
    assert_equal(len(empty), 0)
    assert_equal(empty.dtype(), DataType.FLOAT32)
    assert_equal(input._chunked.value()[].cache[].state.load(), Int64(0))


def test_horizontal_numeric_windows_preserve_global_bitmap_neighbors() raises:
    var a = List[Float64]()
    var b = List[Float64]()
    var av = List[Bool]()
    var bv = List[Bool]()
    for row in range(50):
        a.append(Float64(row % 13 - 6))
        b.append(Float64(20 - row % 11))
        av.append(row % 3 != 0)
        bv.append(row % 5 != 0)
    var left = _pieces(
        Series("x", Column[Float64](a^, av)).slice(13, 25), [1, 17, 0, 7]
    )
    var right = _pieces(
        Series("x", Column[Float64](b^, bv)).slice(13, 25), [3, 1, 1, 0, 20]
    )
    for scalar in [False, True]:
        for kind in [H_SUM, H_MEAN, H_MIN, H_MAX]:
            for ignore in [False, True]:
                var inputs = List[Series]()
                inputs.append(left.copy())
                inputs.append(right.copy())
                if scalar:
                    inputs.append(Series("x", Column[Float64]([5])))
                var actual = horizontal(
                    kind,
                    ignore,
                    inputs^,
                    DataType.FLOAT64,
                    DataType.FLOAT64,
                    0,
                    25,
                    List[Bool](),
                )
                var want = List[Float64]()
                var valid = List[Bool]()
                for row in range(25):
                    var source = row + 13
                    var total = Float64(5) if scalar else Float64(0)
                    var count = 1 if scalar else 0
                    var low = Float64(5) if scalar else Float64(0)
                    var high = low
                    for side in range(2):
                        var present = (
                            source % 3 != 0 if side == 0 else source % 5 != 0
                        )
                        if not present:
                            continue
                        var value = Float64(
                            source % 13 - 6
                        ) if side == 0 else Float64(20 - source % 11)
                        total += value
                        low = value if count == 0 else min(low, value)
                        high = value if count == 0 else max(high, value)
                        count += 1
                    var present = count > 0
                    if kind == H_SUM or kind == H_MEAN:
                        present = (
                            kind == H_SUM or count > 0
                        ) if ignore else count == (3 if scalar else 2)
                    var value = (
                        total if kind
                        == H_SUM else total / Float64(count) if kind
                        == H_MEAN
                        and count > 0 else low if kind == H_MIN else high
                    )
                    want.append(value)
                    valid.append(present)
                assert_frame(
                    DataFrame([actual.renamed("x")]),
                    DataFrame([Series("x", Column[Float64](want^, valid))]),
                )
    assert_equal(left._chunked.value()[].cache[].state.load(), Int64(0))
    assert_equal(right._chunked.value()[].cache[].state.load(), Int64(0))


def _binary_numeric_width[D: DType](offset: Int = 13) raises:
    var a = List[Scalar[D]]()
    var b = List[Scalar[D]]()
    var av = List[Bool]()
    var bv = List[Bool]()
    for row in range(60):
        a.append(Scalar[D](row % 9 + 12))
        b.append(Scalar[D](row % 3 + 1))
        av.append(row % 7 != 0)
        bv.append(row % 5 != 0)
    var left = _pieces(
        Series("x", Column[Scalar[D]](a^, av)).slice(offset, 41), [3, 17, 0, 21]
    )
    var right = _pieces(
        Series("x", Column[Scalar[D]](b^, bv)).slice(offset, 41), [4, 16, 0, 21]
    )
    var observed = List[Bool]()
    for row in range(41):
        observed.append(row % 4 != 1)
    comptime ops = (ADD, SUB, MUL, DIV, FLOORDIV, MOD, CLIP_LOW, CLIP_HIGH)
    comptime for k in range(len(ops)):
        comptime op = ops[k]
        var actual = binary[op](left, right, observed)
        var want = List[Scalar[D]]()
        var floating = List[Float64]()
        var valid = List[Bool]()
        for row in range(41):
            var x = Int64((offset + row) % 9 + 12)
            var y = Int64((offset + row) % 3 + 1)
            var present = (offset + row) % 7 != 0 and (offset + row) % 5 != 0
            comptime if not D.is_floating_point():
                present = present and observed[row]
            valid.append(present)
            comptime if op == DIV:
                floating.append(Float64(x) / Float64(y))
            else:
                var value = (
                    x + y if op
                    == ADD else x - y if op
                    == SUB else x * y if op
                    == MUL else x // y if op
                    == FLOORDIV else x % y if op
                    == MOD else max(x, y) if op
                    == CLIP_LOW else min(x, y)
                )
                want.append(Scalar[D](value))
        comptime if op == DIV:
            comptime if D == DType.float32:
                var singles = List[Float32]()
                for value in floating:
                    singles.append(value.cast[DType.float32]())
                assert_frame(
                    DataFrame([actual.renamed("x")]),
                    DataFrame([Series("x", Column[Float32](singles^, valid))]),
                )
            else:
                assert_frame(
                    DataFrame([actual.renamed("x")]),
                    DataFrame([Series("x", Column[Float64](floating^, valid))]),
                )
        else:
            assert_frame(
                DataFrame([actual.renamed("x")]),
                DataFrame([Series("x", Column[Scalar[D]](want^, valid))]),
            )
    comptime predicates = (GT, LT, GE, LE, EQ, NE)
    var scalar = Series("x", Column[Scalar[D]]([Scalar[D](15)]))
    comptime for k in range(len(predicates)):
        comptime op = predicates[k]
        var actual = binary[op](left, scalar, observed)
        var want = List[Bool]()
        var valid = List[Bool]()
        for row in range(41):
            var x = (offset + row) % 9 + 12
            want.append(
                x
                > 15 if op
                == GT else x
                < 15 if op
                == LT else x
                >= 15 if op
                == GE else x
                <= 15 if op
                == LE else x
                == 15 if op
                == EQ else x
                != 15
            )
            var present = (offset + row) % 7 != 0
            comptime if not D.is_floating_point():
                present = present and observed[row]
            valid.append(present)
        assert_frame(
            DataFrame([actual.renamed("x")]),
            DataFrame([Series("x", BoolColumn(want, valid))]),
        )
    assert_equal(left._chunked.value()[].cache[].state.load(), Int64(0))
    assert_equal(right._chunked.value()[].cache[].state.load(), Int64(0))


def test_chunked_binary_arithmetic_comparisons_and_masks_at_every_numeric_width() raises:
    for offset in range(8):
        _binary_numeric_width[DType.int8](offset)
        _binary_numeric_width[DType.int16](offset)
        _binary_numeric_width[DType.int32](offset)
        _binary_numeric_width[DType.int64](offset)
        _binary_numeric_width[DType.uint8](offset)
        _binary_numeric_width[DType.uint16](offset)
        _binary_numeric_width[DType.uint32](offset)
        _binary_numeric_width[DType.uint64](offset)
        _binary_numeric_width[DType.float32](offset)
        _binary_numeric_width[DType.float64](offset)


def _binary_poison[D: DType]() raises:
    var values = List[Scalar[D]](length=41, fill=3)
    values[17] = Scalar[D].MAX
    values[23] = Scalar[D].MAX
    var valid = List[Bool](length=41, fill=True)
    valid[23] = False
    var source = _pieces(
        Series("x", Column[Scalar[D]](values^, valid)), [3, 17, 0, 21]
    )
    var observed = List[Bool](length=41, fill=True)
    observed[17] = False
    var scalar = Series("x", Column[Scalar[D]]([Scalar[D](1)]))
    var output = binary[ADD](source, scalar, observed)
    assert_equal(output._data[Column[Scalar[D]]]._get(17), Scalar[D](0))
    assert_equal(output._data[Column[Scalar[D]]]._get(23), Scalar[D](0))
    for row in range(41):
        assert_equal(output.get(row).is_null(), row == 17 or row == 23)
        if row != 17 and row != 23:
            assert_equal(output.get(row).numeric[D](), Scalar[D](4))
    with assert_raises():
        _ = binary[ADD](source, scalar)
    assert_equal(source._chunked.value()[].cache[].state.load(), Int64(0))


def test_chunked_binary_masks_hide_only_unobserved_overflow_after_boundaries() raises:
    _binary_poison[DType.int8]()
    _binary_poison[DType.int16]()
    _binary_poison[DType.int32]()
    _binary_poison[DType.int64]()
    _binary_poison[DType.uint8]()
    _binary_poison[DType.uint16]()
    _binary_poison[DType.uint32]()
    _binary_poison[DType.uint64]()


def test_chunked_binary_zero_divisors_powers_and_one_row_physical_chunks() raises:
    var left = _pieces(
        Series("x", Column[Int64]([10, 12, 14, 16, 18])), [1, 1, 1, 1, 1]
    )
    var right = _pieces(
        Series("x", Column[Int64]([2, 0, 4, 0, 3])), [2, 1, 0, 2]
    )
    var quotient = binary[FLOORDIV](left, right)
    var remainder = binary[MOD](left, right)
    var valid: List[Bool] = [True, False, True, False, True]
    assert_frame(
        DataFrame([quotient.renamed("x")]),
        DataFrame([Series("x", Column[Int64]([5, 0, 3, 0, 6], valid))]),
    )
    assert_frame(
        DataFrame([remainder.renamed("x")]),
        DataFrame([Series("x", Column[Int64]([0, 0, 2, 0, 0], valid))]),
    )
    var base = _pieces(Series("x", Column[Int64]([2, -3, 4, 5])), [1, 0, 3])
    var exponent = _pieces(Series("x", Column[Int64]([3, 2, 0, 1])), [2, 2])
    assert_frame(
        DataFrame([binary[POW](base, exponent).renamed("x")]),
        DataFrame([Series("x", Column[Int64]([8, 9, 1, 5]))]),
    )
    var poison = _pieces(Series("x", Column[Int64]([3, Int64.MIN, 5])), [1, 2])
    var minus_one = Series("x", Column[Int64]([-1]))
    var safe = binary[FLOORDIV](poison, minus_one, [True, False, True])
    assert_frame(
        DataFrame([safe.renamed("x")]),
        DataFrame(
            [Series("x", Column[Int64]([-3, 0, -5], [True, False, True]))]
        ),
    )
    with assert_raises():
        _ = binary[FLOORDIV](poison, minus_one)
    assert_equal(left._chunked.value()[].cache[].state.load(), Int64(0))
    assert_equal(right._chunked.value()[].cache[].state.load(), Int64(0))
    assert_equal(base._chunked.value()[].cache[].state.load(), Int64(0))
    assert_equal(exponent._chunked.value()[].cache[].state.load(), Int64(0))
    assert_equal(poison._chunked.value()[].cache[].state.load(), Int64(0))


def _dense_numeric_windows[D: DType]() raises:
    var av = List[Scalar[D]]()
    var bv = List[Scalar[D]]()
    for row in range(60):
        av.append(Scalar[D](3 + row % 5))
        bv.append(Scalar[D](2 + row % 3))
    var flat_a = Series("x", Column[Scalar[D]](av^)).slice(13, 41)
    var flat_b = Series("x", Column[Scalar[D]](bv^)).slice(13, 41)
    var a = _pieces(flat_a, [3, 21, 17])
    var b = _pieces(flat_b, [5, 17, 19])
    comptime if D == DType.int8 or D == DType.int16 or D == DType.int32:
        comptime ops = (ADD, SUB, MUL)
        comptime for k in range(len(ops)):
            comptime op = ops[k]
            var want = List[Scalar[D]]()
            for row in range(41):
                var x = 3 + (row + 13) % 5
                var y = 2 + (row + 13) % 3
                want.append(
                    Scalar[D](
                        x + y if op == ADD else x - y if op == SUB else x * y
                    )
                )
            assert_frame(
                DataFrame([binary[op](a, b).renamed("x")]),
                DataFrame([Series("x", Column[Scalar[D]](want^))]),
            )
    var comparisons = List[Bool]()
    for row in range(41):
        comparisons.append(3 + (row + 13) % 5 > 2 + (row + 13) % 3)
    assert_frame(
        DataFrame([binary[GT](a, b).renamed("x")]),
        DataFrame([Series("x", BoolColumn(comparisons))]),
    )
    for chunk_a in [False, True]:
        for chunk_b in [False, True]:
            for kind in [H_MIN, H_MAX]:
                var inputs = List[Series]()
                inputs.append(a.copy() if chunk_a else flat_a.copy())
                inputs.append(b.copy() if chunk_b else flat_b.copy())
                var actual = horizontal(
                    kind,
                    True,
                    inputs^,
                    DataType.of(D),
                    DataType.of(D),
                    0,
                    41,
                    List[Bool](),
                )
                var want = List[Scalar[D]]()
                for row in range(41):
                    var x = 3 + (row + 13) % 5
                    var y = 2 + (row + 13) % 3
                    want.append(
                        Scalar[D](min(x, y) if kind == H_MIN else max(x, y))
                    )
                assert_frame(
                    DataFrame([actual.renamed("x")]),
                    DataFrame([Series("x", Column[Scalar[D]](want^))]),
                )
    assert_equal(a._chunked.value()[].cache[].state.load(), Int64(0))
    assert_equal(b._chunked.value()[].cache[].state.load(), Int64(0))


def test_dense_multi_input_simd_and_extrema_keep_contiguous_and_chunked_inputs() raises:
    _dense_numeric_windows[DType.int8]()
    _dense_numeric_windows[DType.int16]()
    _dense_numeric_windows[DType.int32]()
    _dense_numeric_windows[DType.int64]()
    _dense_numeric_windows[DType.uint8]()
    _dense_numeric_windows[DType.uint16]()
    _dense_numeric_windows[DType.uint32]()
    _dense_numeric_windows[DType.uint64]()
    _dense_numeric_windows[DType.float32]()
    _dense_numeric_windows[DType.float64]()


def _mixed_bitmap_source[D: DType](left: Bool) raises -> Series:
    var parts = List[Series]()
    var base = 0
    var sizes: List[Int] = [3, 17, 21] if left else [5, 9, 27]
    for chunk in range(3):
        var values = List[Scalar[D]]()
        var valid = List[Bool]()
        for relative in range(sizes[chunk] + 2):
            var row = base + relative - 1
            values.append(Scalar[D](row % 9 + 12 if left else row % 3 + 1))
            valid.append(row % 7 != 0 if left else row % 5 != 0)
        var part = Series("x", Column[Scalar[D]](values^, valid)).slice(
            1, sizes[chunk]
        ) if chunk == 1 else Series("x", Column[Scalar[D]](values^)).slice(
            1, sizes[chunk]
        )
        parts.append(part^)
        base += sizes[chunk]
    return Series._from_chunks(parts)


def _mixed_bitmap_inputs[D: DType]() raises:
    var a = _mixed_bitmap_source[D](True)
    var b = _mixed_bitmap_source[D](False)
    var selected = List[Bool]()
    var sum_values = List[Scalar[D]]()
    var sum_valid = List[Bool]()
    var picked = List[Scalar[D]]()
    var pick_valid = List[Bool]()
    for row in range(41):
        var av = row < 3 or row >= 20 or row % 7 != 0
        var bv = row < 5 or row >= 14 or row % 5 != 0
        var x = row % 9 + 12
        var y = row % 3 + 1
        var take = row % 2 == 0
        selected.append(take)
        sum_values.append(Scalar[D](x + y))
        sum_valid.append(av and bv)
        picked.append(Scalar[D](x if take else y))
        pick_valid.append(av if take else bv)
    assert_frame(
        DataFrame([binary[ADD](a, b).renamed("x")]),
        DataFrame([Series("x", Column[Scalar[D]](sum_values^, sum_valid))]),
    )
    assert_frame(
        DataFrame([choose(selected, a, b).renamed("x")]),
        DataFrame([Series("x", Column[Scalar[D]](picked^, pick_valid))]),
    )
    var null = Series("x", Column[Scalar[D]]([Scalar[D](99)], [False]))
    var all_null = binary[ADD](a, null)
    for row in range(41):
        assert_true(all_null.get(row).is_null())
        assert_equal(all_null._data[Column[Scalar[D]]]._get(row), Scalar[D](0))
    assert_equal(a._chunked.value()[].cache[].state.load(), Int64(0))
    assert_equal(b._chunked.value()[].cache[].state.load(), Int64(0))


def test_multi_input_bitless_nullable_bitless_windows_and_null_scalar() raises:
    _mixed_bitmap_inputs[DType.int64]()
    _mixed_bitmap_inputs[DType.float32]()
    _mixed_bitmap_inputs[DType.float64]()


def _floating_multi_edges[D: DType]() raises:
    var nan = Scalar[D](0) / Scalar[D](0)
    var inf = Scalar[D](1) / Scalar[D](0)
    var xs: List[Scalar[D]] = [nan, inf, -inf, -0.0, 0.0, -2, 4, 1]
    var ys: List[Scalar[D]] = [nan, 0.5, 2, 0.0, -0.0, 0.5, 0, nan]
    var av = List[Scalar[D]]()
    var bv = List[Scalar[D]]()
    for row in range(50):
        av.append(xs[row % 8])
        bv.append(ys[row % 8])
    var flat_a = Series("x", Column[Scalar[D]](av^)).slice(1, 33)
    var flat_b = Series("x", Column[Scalar[D]](bv^)).slice(1, 33)
    var a = _pieces(flat_a, [2, 17, 14])
    var b = _pieces(flat_b, [3, 16, 14])
    comptime ops = (
        ADD,
        SUB,
        MUL,
        DIV,
        FLOORDIV,
        MOD,
        POW,
        CLIP_LOW,
        CLIP_HIGH,
        GT,
        LT,
        GE,
        LE,
        EQ,
        NE,
    )
    comptime for k in range(len(ops)):
        comptime op = ops[k]
        assert_frame(
            DataFrame([binary[op](a, b).renamed("x")]),
            DataFrame([binary[op](flat_a, flat_b).renamed("x")]),
        )
    var neq = List[Bool]()
    for row in range(33):
        var slot = (row + 1) % 8
        neq.append(slot != 3 and slot != 4)
    assert_frame(
        DataFrame([binary[NE](a, b).renamed("x")]),
        DataFrame([Series("x", BoolColumn(neq))]),
    )
    var selected = List[Bool]()
    var picked = List[Scalar[D]]()
    for row in range(33):
        var take = row % 2 == 0
        selected.append(take)
        picked.append(xs[(row + 1) % 8] if take else ys[(row + 1) % 8])
    assert_frame(
        DataFrame([choose(selected, a, b).renamed("x")]),
        DataFrame([Series("x", Column[Scalar[D]](picked^))]),
    )
    var mins: List[Scalar[D]] = [nan, 0.5, -inf, -0.0, 0.0, -2, 0, 1]
    var maxs: List[Scalar[D]] = [nan, inf, 2, -0.0, 0.0, 0.5, 4, 1]
    for kind in [H_MIN, H_MAX]:
        var want = List[Scalar[D]]()
        for row in range(33):
            want.append(
                mins[(row + 1) % 8] if kind == H_MIN else maxs[(row + 1) % 8]
            )
        var actual = horizontal(
            kind,
            True,
            [a.copy(), b.copy()],
            DataType.of(D),
            DataType.of(D),
            0,
            33,
            List[Bool](),
        )
        assert_frame(
            DataFrame([actual.renamed("x")]),
            DataFrame([Series("x", Column[Scalar[D]](want^))]),
        )
    assert_equal(a._chunked.value()[].cache[].state.load(), Int64(0))
    assert_equal(b._chunked.value()[].cache[].state.load(), Int64(0))


def test_multi_input_float_nan_infinity_signed_zero_and_power_edges() raises:
    _floating_multi_edges[DType.float32]()
    _floating_multi_edges[DType.float64]()
