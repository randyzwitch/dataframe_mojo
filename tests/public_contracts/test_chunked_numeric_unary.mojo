"""Unary numeric kernels preserve sliced validity, masks and owned results."""
# contract-group: expr.chunk_windows
from std.math import isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series
from dataframe.expr import ABS, NEG, SQRT, SIN, LOG1P, SIGN, TRUNCATE
from dataframe.kernels.binary import math_unary, unary
from tests.support.coverage_compare import assert_frame


def _fragment(input: Series) raises -> Series:
    var parts = List[Series]()
    var first = 3
    for size in [0, 1, 2, 0, 1, 5, 0]:
        parts.append(input.slice(first, size))
        first += size
    return Series._from_chunks(parts)


def _assert_unmerged(input: Series) raises:
    assert_true(input.is_chunked())
    assert_equal(input._chunked.value()[].cache[].state.load(), Int64(0))


def test_chunked_integer_unary_masks_use_global_rows_and_hide_overflow() raises:
    var source = Series(
        "x",
        Column[Int64](
            [9, 9, 9, -1, 2, Int64.MIN, -4, 5, -6, 7, -8, 9, 9],
            [
                True,
                True,
                True,
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
            ],
        ),
    )
    var input = _fragment(source)
    var observed: List[Bool] = [
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    var output = unary[ABS](input, 0, observed)
    _assert_unmerged(input)
    assert_frame(
        DataFrame([output.renamed("x")]),
        DataFrame(
            [
                Series(
                    "x",
                    Column[Int64](
                        [1, 2, 0, 4, 5, 0, 7, 8, 9],
                        [
                            True,
                            True,
                            False,
                            True,
                            True,
                            False,
                            True,
                            True,
                            True,
                        ],
                    ),
                )
            ]
        ),
    )
    with assert_raises():
        _ = unary[ABS](input, 0)
    _assert_unmerged(input)
    with assert_raises():
        _ = unary[NEG](input, 0)
    _assert_unmerged(input)


def _integer_width[D: DType]() raises:
    var source = Series(
        "x",
        Column[Scalar[D]](
            [9, 9, 9, 1, 4, 9, 16, 25, 36, 49, 64, 81, 9],
            [
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
                False,
                True,
                True,
            ],
        ),
    )
    var input = _fragment(source)
    var root = unary[SQRT](input, 0)
    _assert_unmerged(input)
    assert_frame(
        DataFrame([root.renamed("x")]),
        DataFrame(
            [
                Series(
                    "x",
                    Column[Float64](
                        [1, 2, 0, 4, 5, 6, 7, 0, 9],
                        [
                            True,
                            True,
                            False,
                            True,
                            True,
                            True,
                            True,
                            False,
                            True,
                        ],
                    ),
                )
            ]
        ),
    )
    var sign = math_unary[4](SIGN, input, 0)
    _assert_unmerged(input)
    assert_frame(
        DataFrame([sign.renamed("x")]),
        DataFrame(
            [
                Series(
                    "x",
                    Column[Scalar[D]](
                        [1, 1, 0, 1, 1, 1, 1, 0, 1],
                        [
                            True,
                            True,
                            False,
                            True,
                            True,
                            True,
                            True,
                            False,
                            True,
                        ],
                    ),
                )
            ]
        ),
    )
    var truncated = math_unary[4](TRUNCATE, input, 0)
    _assert_unmerged(input)
    assert_frame(
        DataFrame([truncated.renamed("x")]), DataFrame([source.slice(3, 9)])
    )
    var sine = math_unary[4](SIN, input, 0)
    assert_equal(sine.dtype(), DataType.FLOAT64)
    _assert_unmerged(input)
    var absolute = unary[ABS](input, 0)
    assert_equal(absolute.dtype(), DataType.of(D))
    _assert_unmerged(input)
    assert_frame(
        DataFrame([absolute.renamed("x")]), DataFrame([source.slice(3, 9)])
    )


def test_chunked_integer_unary_preserves_every_physical_width() raises:
    _integer_width[DType.int8]()
    _integer_width[DType.int16]()
    _integer_width[DType.int32]()
    _integer_width[DType.int64]()
    _integer_width[DType.uint8]()
    _integer_width[DType.uint16]()
    _integer_width[DType.uint32]()
    _integer_width[DType.uint64]()


def _owned_float[D: DType]() raises -> Series:
    var source = Series(
        "x",
        Column[Scalar[D]](
            [9, 9, 9, -1, 2, 3, -4, 5, -6, 7, -8, 9, 9],
            [
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
                False,
                True,
                True,
            ],
        ),
    )
    var input = _fragment(source)
    var output = unary[ABS, 4](input, 0)
    _assert_unmerged(input)
    return output^


def test_chunked_float_unary_owns_values_and_validity_after_sources_drop() raises:
    comptime for k in range(2):
        comptime D = DType.float32 if k == 0 else DType.float64
        var output = _owned_float[D]()
        assert_frame(
            DataFrame([output.renamed("x")]),
            DataFrame(
                [
                    Series(
                        "x",
                        Column[Scalar[D]](
                            [1, 2, 0, 4, 5, 6, 7, 0, 9],
                            [
                                True,
                                True,
                                False,
                                True,
                                True,
                                True,
                                True,
                                False,
                                True,
                            ],
                        ),
                    )
                ]
            ),
        )


def test_chunked_math_preserves_signed_zero_nan_infinity_and_warm_sources() raises:
    var inf = Float64(1) / Float64(0)
    var nan = Float64(0) / Float64(0)
    var source = Series(
        "x",
        Column[Float64](
            [9, 9, 9, -0.0, 0.0, -inf, inf, nan, -2, 3, 0, 9, 9],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
            ],
        ),
    )
    var input = _fragment(source)
    var sign = math_unary[4](SIGN, input, 0)
    _assert_unmerged(input)
    var expected: List[Float64] = [-0.0, 0.0, -1, 1, nan, -1, 1, 0, 1]
    for row in range(9):
        var value = sign.get(row)
        assert_equal(value.is_null(), row == 7)
        if row == 7:
            continue
        if row == 4:
            assert_true(isnan(value.float64()))
        else:
            assert_equal(
                bitcast[DType.uint64](value.float64()),
                bitcast[DType.uint64](expected[row]),
            )
    var cold = math_unary[4](SIN, input, 0)
    _assert_unmerged(input)
    var flat = input.rechunk()
    var warm = math_unary[4](SIN, input, 0)
    assert_frame(DataFrame([cold.renamed("x")]), DataFrame([warm.renamed("x")]))
    assert_frame(
        DataFrame([math_unary[4](TRUNCATE, input, 0).renamed("x")]),
        DataFrame([math_unary[4](TRUNCATE, flat, 0).renamed("x")]),
    )
