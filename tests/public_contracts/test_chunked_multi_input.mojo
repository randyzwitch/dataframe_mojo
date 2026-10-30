"""Numeric inputs keep independent chunk boundaries and scalar broadcasts."""
# contract-group: expr.chunk_windows
from std.testing import assert_equal, assert_true
from dataframe import Column, DataFrame, DataType, Series
from dataframe.kernels.binary import choose
from dataframe.expr import H_SUM, H_MEAN, H_MIN, H_MAX
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


def _numeric_choice[D: DType]() raises:
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
        Series("x", Column[Scalar[D]](a^, a_valid)).slice(13, 41),
        [3, 0, 17, 1, 0, 20],
    )
    var right = _pieces(
        Series("x", Column[Scalar[D]](b^, b_valid)).slice(13, 41),
        [7, 1, 0, 8, 19, 6],
    )
    var selected = List[Bool]()
    var want = List[Scalar[D]]()
    var valid = List[Bool]()
    for row in range(41):
        var take = row % 3 == 1
        var source = 13 + row
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
    _numeric_choice[DType.int8]()
    _numeric_choice[DType.int16]()
    _numeric_choice[DType.int32]()
    _numeric_choice[DType.int64]()
    _numeric_choice[DType.uint8]()
    _numeric_choice[DType.uint16]()
    _numeric_choice[DType.uint32]()
    _numeric_choice[DType.uint64]()
    _numeric_choice[DType.float32]()
    _numeric_choice[DType.float64]()


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
