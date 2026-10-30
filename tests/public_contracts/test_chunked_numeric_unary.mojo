"""Unary numeric kernels preserve sliced validity, masks and owned results."""
# contract-group: expr.chunk_windows
from std.math import isnan
from std.memory import bitcast
from std.ffi import external_call
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series
from dataframe.expr import (
    ABS,
    NEG,
    SQRT,
    EXP,
    LOG,
    FLOOR,
    CEIL,
    ROUND,
    SIN,
    SIGN,
    TRUNCATE,
)
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


def _mixed_validity[D: DType]() raises:
    var input = Series._from_chunks(
        [
            Series("x", Column[Scalar[D]]([9, 1, 2, 3])).slice(1, 3),
            Series(
                "x",
                Column[Scalar[D]](
                    [9, 4, 5, 6, 7], [True, True, False, True, True]
                ),
            ).slice(1, 4),
            Series("x", Column[Scalar[D]]([9, 8, 9, 10, 11])).slice(1, 4),
        ]
    )
    var valid: List[Bool] = [
        True,
        True,
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
    assert_frame(
        DataFrame([unary[ABS](input, 0).renamed("x")]),
        DataFrame(
            [
                Series(
                    "x",
                    Column[Scalar[D]](
                        [1, 2, 3, 4, 0, 6, 7, 8, 9, 10, 11], valid
                    ),
                )
            ]
        ),
    )
    assert_frame(
        DataFrame([math_unary[4](SIGN, input, 0).renamed("x")]),
        DataFrame(
            [
                Series(
                    "x",
                    Column[Scalar[D]]([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], valid),
                )
            ]
        ),
    )
    _assert_unmerged(input)


def test_chunked_unary_mixed_bitmap_presence_crosses_output_bytes() raises:
    _mixed_validity[DType.int64]()
    _mixed_validity[DType.float32]()
    _mixed_validity[DType.float64]()


def _assert_float_result(
    actual: Series, expected: Series, tolerance: Float64
) raises:
    assert_equal(actual.dtype(), expected.dtype())
    assert_equal(len(actual), len(expected))
    var a = actual.cast(DataType.FLOAT64)
    var b = expected.cast(DataType.FLOAT64)
    for row in range(len(a)):
        var x = a.get(row)
        var y = b.get(row)
        assert_equal(x.is_null(), y.is_null())
        if y.is_null():
            continue
        var got = x.float64()
        var want = y.float64()
        if isnan(want):
            assert_true(isnan(got))
        elif want == 0 or abs(want) == Float64(1) / Float64(0):
            assert_equal(
                bitcast[DType.uint64](got), bitcast[DType.uint64](want)
            )
        else:
            assert_true(
                abs(got - want) <= tolerance * max(abs(want), Float64(1e-300))
            )


def _float_boundaries[D: DType]() raises:
    var inf = Scalar[D](1) / Scalar[D](0)
    var nan = Scalar[D](0) / Scalar[D](0)
    var source = Series(
        "x",
        Column[Scalar[D]](
            [9, 9, 9, -0.0, 0.0, -inf, inf, nan, -700, -2.5, 0.5, 700, 9],
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
                True,
                True,
                True,
            ],
        ),
    )
    var input = _fragment(source)
    var flat = source.slice(3, 9)
    var tolerance = Float64(2e-6) if D == DType.float32 else Float64(1e-13)
    comptime ops = (NEG, ABS, SQRT, EXP, LOG, FLOOR, CEIL, ROUND)
    comptime for k in range(len(ops)):
        comptime op = ops[k]
        var actual = unary[op, 4](input, 2)
        _assert_float_result(actual, unary[op, 4](flat, 2), tolerance)
    var sine = math_unary[4](SIN, input, 0)
    _assert_float_result(sine, math_unary[4](SIN, flat, 0), tolerance)
    _assert_unmerged(input)


def test_chunked_float_simd_tails_preserve_unary_edges() raises:
    _float_boundaries[DType.float32]()
    _float_boundaries[DType.float64]()


def _integer_errors[D: DType]() raises:
    var poison = Scalar[D].MIN if D.is_signed() else Scalar[D](1)
    var input = Series._from_chunks(
        [
            Series("x", Column[Scalar[D]]([0, 0, 0])),
            Series("x", Column[Scalar[D]]([0, poison, 0])),
        ]
    )
    var mask: List[Bool] = [True, True, True, True, False, True]
    var safe = unary[NEG](input, 0, mask)
    assert_true(safe.get(4).is_null())
    with assert_raises():
        _ = unary[NEG](input, 0)
    comptime if D.is_signed():
        _ = unary[ABS](input, 0, mask)
        with assert_raises():
            _ = unary[ABS](input, 0)
    _assert_unmerged(input)


def test_chunked_integer_overflow_guards_cover_signed_and_unsigned_widths() raises:
    _integer_errors[DType.int8]()
    _integer_errors[DType.int16]()
    _integer_errors[DType.int32]()
    _integer_errors[DType.int64]()
    _integer_errors[DType.uint8]()
    _integer_errors[DType.uint16]()
    _integer_errors[DType.uint32]()
    _integer_errors[DType.uint64]()


def test_chunked_integer_math_converts_signed_and_large_unsigned_values() raises:
    var signed = Series._from_chunks(
        [
            Series("x", Column[Int64]([-9, -1])),
            Series("x", Column[Int64]([0, 1, 9])),
        ]
    )
    var unsigned = Series._from_chunks(
        [
            Series("x", Column[UInt64]([0, 1])),
            Series("x", Column[UInt64]([UInt64.MAX, UInt64.MAX - 1])),
        ]
    )
    for input in [signed^, unsigned^]:
        var actual = math_unary[4](SIN, input, 0)
        var converted = input.cast(DataType.FLOAT64)
        for row in range(len(input)):
            var expected = external_call["sin", Float64](
                converted.get(row).float64()
            )
            assert_equal(actual.get(row).float64(), expected)
        _assert_unmerged(input)
