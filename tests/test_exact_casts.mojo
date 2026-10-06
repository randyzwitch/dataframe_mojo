"""Numeric casts take one typed loop when no value can fail to fit, or
none does. For every pair of numeric types, and from Bool, the result
must equal the general path's: the same values, nulls and type, through
slices and chunks, and the same error or null when a value does not fit.
"""
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from dataframe import BoolColumn, Column, DataType, Series
from dataframe.cast import _cast_exact, cast_series
from dataframe.dtype import NUMERIC_DTYPES


def general(input: Series, target: DataType, strict: Bool) raises -> Series:
    """The general path: a mask observing every row turns the typed loop
    off."""
    return cast_series(
        input, target, strict, 0, List[Bool](length=len(input), fill=True)
    )


def small[D: DType](rows: Int) raises -> Series:
    """Values every numeric type can hold (0 to 100), with nulls."""
    var values = List[Scalar[D]](capacity=rows)
    var valid = List[Bool](capacity=rows)
    for i in range(rows):
        values.append(Scalar[D]((i * 37) % 101))
        valid.append(i % 7 != 3)
    return Series("v", Column[Scalar[D]](values^, valid^))


def extremes[D: DType]() raises -> Series:
    """A type's own limits, zero and a null."""
    comptime if D.is_floating_point():
        var values: List[Scalar[D]] = [0, 1.5, -2.25, 1e30, -1e30, 7]
        return Series(
            "v",
            Column[Scalar[D]](values^, [True, True, True, True, True, False]),
        )
    else:
        var values: List[Scalar[D]] = [
            Scalar[D].MIN,
            Scalar[D].MAX,
            0,
            1,
            Scalar[D].MAX,
            5,
        ]
        return Series(
            "v",
            Column[Scalar[D]](values^, [True, True, True, True, False, True]),
        )


def test_every_pair_matches_the_general_path_on_values_that_fit() raises:
    comptime for j in range(len(NUMERIC_DTYPES)):
        comptime S = NUMERIC_DTYPES[j]
        var source = small[S](2_000)
        var chunked = Series._from_chunks(
            [source.slice(0, 13), source.slice(13, 1_987)]
        )
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime T = NUMERIC_DTYPES[k]
            var target = DataType.of(T)
            for input in [
                source.copy(),
                source.slice(5, 1_500),
                chunked.copy(),
            ]:
                var got = input.cast(target)
                var want = general(input, target, True)
                assert_true(got.dtype() == target)
                assert_true(got.equals(want), String(S) + " to " + String(T))
                assert_equal(got.null_count(), want.null_count())


def test_the_typed_loop_serves_the_pairs_it_should() raises:
    # Integers to anything; floats only to a float at least as wide.
    comptime for j in range(len(NUMERIC_DTYPES)):
        comptime S = NUMERIC_DTYPES[j]
        var source = small[S](64)
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime T = NUMERIC_DTYPES[k]
            comptime if S != T:
                var taken = Bool(_cast_exact(source, DataType.of(T)))
                comptime if S.is_floating_point():
                    assert_equal(
                        taken,
                        T == DType.float64 and S == DType.float32,
                        String(S) + " to " + String(T),
                    )
                else:
                    assert_true(taken, String(S) + " to " + String(T))


def test_limits_widen_exactly_and_overflow_falls_back() raises:
    comptime for j in range(len(NUMERIC_DTYPES)):
        comptime S = NUMERIC_DTYPES[j]
        var source = extremes[S]()
        comptime for k in range(len(NUMERIC_DTYPES)):
            comptime T = NUMERIC_DTYPES[k]
            var target = DataType.of(T)
            # Not strict: a value that does not fit becomes null on both
            # paths, wherever the typed loop gave up.
            var got = source.cast(target, strict=False)
            var want = general(source, target, False)
            assert_true(got.equals(want), String(S) + " to " + String(T))
            assert_equal(got.null_count(), want.null_count())
    var wide = Series("v", Column[Int64]([Int64(1), 40_000, -3]))
    with assert_raises(contains="out of int16 range"):
        _ = wide.cast(DataType.INT16)
    with assert_raises(contains="out of uint8 range"):
        _ = wide.cast(DataType.UINT8)
    # A value that does not fit under a null is not read.
    var hidden = Series(
        "v", Column[Int64]([Int64(1), 40_000, -3], [True, False, True])
    )
    assert_true(
        hidden.cast(DataType.INT16).equals(
            Series("v", Column[Int16]([Int16(1), 0, -3], [True, False, True]))
        )
    )


def test_bool_to_every_numeric_type() raises:
    var flags = List[Bool]()
    var valid = List[Bool]()
    for i in range(1_003):
        flags.append(i % 3 == 0)
        valid.append(i % 5 != 2)
    var plain = Series("b", BoolColumn(flags.copy()))
    var nullable = Series("b", BoolColumn(flags^, valid^))
    comptime for k in range(len(NUMERIC_DTYPES)):
        comptime T = NUMERIC_DTYPES[k]
        var target = DataType.of(T)
        for input in [plain.copy(), nullable.copy(), nullable.slice(9, 700)]:
            assert_true(Bool(_cast_exact(input, target)))
            var got = input.cast(target)
            assert_true(got.equals(general(input, target, True)), String(T))
    # A logical type over integer storage keeps the general path.
    var dates = Series("d", Column[Int64]([Int64(1), 2])).with_dtype(
        DataType.DATE
    )
    assert_true(
        dates.cast(DataType.INT64).equals(
            Series("d", Column[Int64]([Int64(1), 2]))
        )
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
