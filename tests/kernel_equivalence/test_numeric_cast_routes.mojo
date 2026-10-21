"""Numeric cast dispatch and results against independently bounded scalars."""
# contract-group: kernel.row_ops
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import BoolColumn, Column, DataType, Series
from dataframe.cast import _cast_exact, cast_series

comptime CAST_NUMBERS = (
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


def cast_present(row: Int, mode: Int) -> Bool:
    return mode == 0 or (mode == 1 and row % 5 != 2)


def assert_small_cast[
    T: DType
](
    actual: Series,
    n: Int,
    offset: Int,
    mode: Int,
    masked: Bool = False,
    boolean: Bool = False,
) raises:
    assert_true(actual.dtype() == DataType.of(T))
    assert_equal(len(actual), n)
    ref values = actual._data[Column[Scalar[T]]]
    for i in range(n):
        var row = offset + i
        var valid = cast_present(row, mode) and (not masked or i % 4 != 1)
        assert_equal(actual.get(i).is_null(), not valid)
        if valid:
            var expected = Int(row % 3 == 0) if boolean else (row * 37) % 101
            assert_equal(values.value(i), Scalar[T](expected))


def test_exact_numeric_cast_eligibility_and_masked_fallback_match_independent_typed_cells() raises:
    comptime for source_index in range(len(CAST_NUMBERS)):
        comptime S = CAST_NUMBERS[source_index]
        for n in [0, 1, 9, 65]:
            for offset in [0, 1, 7]:
                for mode in range(3):
                    var values = List[Scalar[S]]()
                    var valid = List[Bool]()
                    for row in range(n + offset):
                        var present = cast_present(row, mode)
                        values.append(
                            Scalar[S]((row * 37) % 101) if present else Scalar[
                                S
                            ].MAX
                        )
                        valid.append(present)
                    var source = Series(
                        "number", Column[Scalar[S]](values^, valid)
                    ).slice(offset, n)
                    comptime for target_index in range(len(CAST_NUMBERS)):
                        comptime T = CAST_NUMBERS[target_index]
                        var target = DataType.of(T)
                        var exact = _cast_exact(source, target)
                        comptime eligible = S != T and (
                            not S.is_floating_point()
                            or (S == DType.float32 and T == DType.float64)
                        )
                        assert_equal(Bool(exact), eligible)
                        if exact:
                            assert_small_cast[T](exact.value(), n, offset, mode)
                            ref cells = exact.value()._data[Column[Scalar[T]]]
                            for i in range(n):
                                if not cast_present(offset + i, mode):
                                    assert_equal(
                                        cells._ptr()[unsafe_offset=i],
                                        Scalar[T](0),
                                    )
                        assert_small_cast[T](
                            source.cast(target), n, offset, mode
                        )
                        # A nonempty observation mask explicitly disables the
                        # typed route; every selected cell still has its own
                        # literal scalar expectation. Equal-type casts retain
                        # the source by contract, before mask evaluation.
                        comptime if S != T:
                            var observed = List[Bool]()
                            for i in range(n):
                                observed.append(i % 4 != 1)
                            assert_small_cast[T](
                                cast_series(source, target, True, 0, observed),
                                n,
                                offset,
                                mode,
                                True,
                            )


def test_bool_cast_exact_routes_keep_unaligned_validity_and_literal_zero_one_values() raises:
    for n in [0, 1, 9, 65]:
        for offset in range(8):
            for mode in range(3):
                var flags = List[Bool]()
                var valid = List[Bool]()
                for row in range(n + offset):
                    flags.append(row % 3 == 0 or not cast_present(row, mode))
                    valid.append(cast_present(row, mode))
                var source = Series("flag", BoolColumn(flags, valid)).slice(
                    offset, n
                )
                comptime for k in range(len(CAST_NUMBERS)):
                    comptime T = CAST_NUMBERS[k]
                    var exact = _cast_exact(source, DataType.of(T))
                    assert_true(Bool(exact))
                    assert_small_cast[T](
                        exact.value(), n, offset, mode, boolean=True
                    )
                    assert_small_cast[T](
                        source.cast(DataType.of(T)),
                        n,
                        offset,
                        mode,
                        boolean=True,
                    )
                    var observed = List[Bool](length=n, fill=True)
                    assert_small_cast[T](
                        cast_series(source, DataType.of(T), True, 0, observed),
                        n,
                        offset,
                        mode,
                        boolean=True,
                    )


def test_integer_cast_boundaries_decline_only_on_visible_out_of_range_values_and_report_strict_errors() raises:
    comptime for s in range(8):
        comptime S = CAST_NUMBERS[s]
        comptime for t in range(8):
            comptime T = CAST_NUMBERS[t]
            for offset in [0, 7]:
                for hidden in [False, True]:
                    var raw: List[Scalar[S]] = [
                        Scalar[S].MIN,
                        Scalar[S].MAX,
                        0,
                        1,
                    ]
                    var values = List[Scalar[S]](length=offset, fill=0)
                    var valid = List[Bool](length=offset, fill=True)
                    var expected_valid = List[Bool]()
                    var fits = True
                    for i in range(4):
                        var observed = not hidden or i >= 2
                        var wide = Int128(raw[i])
                        var in_range = wide >= Int128(
                            Scalar[T].MIN
                        ) and wide <= Int128(Scalar[T].MAX)
                        fits = fits and (not observed or in_range)
                        expected_valid.append(observed and in_range)
                        values.append(raw[i])
                        valid.append(observed)
                    var source = Series(
                        "bounded", Column[Scalar[S]](values^, valid)
                    ).slice(offset, 4)
                    var exact = _cast_exact(source, DataType.of(T))
                    assert_equal(Bool(exact), S != T and fits)
                    var actual = source.cast(DataType.of(T), strict=False)
                    ref cells = actual._data[Column[Scalar[T]]]
                    for i in range(4):
                        assert_equal(
                            actual.get(i).is_null(), not expected_valid[i]
                        )
                        if expected_valid[i]:
                            assert_equal(
                                cells.value(i), Scalar[T](Int128(raw[i]))
                            )
                    if not fits:
                        with assert_raises(contains="range"):
                            _ = source.cast(DataType.of(T))
