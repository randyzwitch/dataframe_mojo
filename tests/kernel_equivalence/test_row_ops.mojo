"""Row kernels against scalar models, with explicit specialized/fallback calls."""
# contract-group: kernel.row_ops
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import BoolColumn, Column, DataType, Series
from dataframe.column import gather_scalars
from dataframe.gather import (
    true_rows,
    _MaskJob,
    float_compare_rows,
    take_parallel,
    take_sorted_chunked,
    can_filter_aligned_chunks,
)
from dataframe.expr import ADD, SUB, MUL, DIV, FLOORDIV, MOD, POW, EQ, LT, GT
from dataframe.expr_kernels import (
    _numeric_int,
    _vector_int,
    _numeric_float,
    _choose,
    _choose_numeric,
    choose,
    binary,
    _compare_bits,
    _dense_decimal,
)
from tests.support.coverage_compare import assert_float
from tests.support.row_kernel_reference import (
    row_valid,
    row_selected,
    expected_selection,
    row_payload,
    integer_bounds,
    checked_row_integer,
    decimal_power,
    half_even_ratio,
    decimal_row_result,
)

comptime ROW_NUMERIC_DTYPES = (
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
comptime ROW_INTEGER_DTYPES = (
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
)


def test_scalar_row_model_has_literal_selection_and_range_anchors() raises:
    assert_equal(half_even_ratio(125, 2), Int128(62))
    assert_equal(half_even_ratio(375, 10), Int128(38))
    assert_equal(half_even_ratio(-125, 2), Int128(-62))
    assert_equal(decimal_row_result(2, 125, 50, 2, 2), Int128(6250))
    assert_equal(decimal_row_result(3, 75, 200, 2, 2), Int128(38))
    assert_equal(expected_selection(9, 2, 0), [1, 4, 7])
    assert_equal(expected_selection(9, 2, 1), [1, 4])
    assert_equal(expected_selection(9, 1, 2), List[Int]())
    assert_equal(integer_bounds[DType.int8](), (Int128(-128), Int128(127)))
    assert_equal(
        integer_bounds[DType.uint64](),
        (Int128(0), (Int128(1) << Int128(64)) - 1),
    )
    assert_equal(checked_row_integer[DType.int8](0, -128, 1), Int128(-127))
    assert_equal(checked_row_integer[DType.uint64](2, 3, 7), Int128(21))
    with assert_raises():
        _ = checked_row_integer[DType.uint64](
            2, (Int128(1) << Int128(64)) - 1, (Int128(1) << Int128(64)) - 1
        )
    with assert_raises():
        _ = checked_row_integer[DType.int8](0, 127, 1)
    with assert_raises():
        _ = checked_row_integer[DType.uint8](1, 0, 1)


def test_bitmap_filter_compaction_matches_scalar_rows_at_every_density() raises:
    for n in [0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129]:
        for offset in [0, 1, 7]:
            for density in range(5):
                for validity in range(3):
                    var flags = List[Bool](length=offset, fill=True)
                    var valid = List[Bool](length=offset, fill=True)
                    for row in range(n):
                        flags.append(row_selected(row, n, density))
                        valid.append(row_valid(row, validity))
                    var mask = BoolColumn(flags^, valid^).slice(offset, n)
                    var expected = expected_selection(n, density, validity)
                    assert_equal(true_rows(mask), expected)
                    # Directly invoke the specialized 64-bit bitmap scan:
                    # its route is guaranteed, including a partial word.
                    var job = _MaskJob(mask, 0, n)
                    job.run()
                    assert_equal(job^.into_rows(), expected)
                    # Partition starts need not coincide with a bitmap byte.
                    for first in [0, min(1, n), min(7, n)]:
                        var partition = _MaskJob(mask, first, n)
                        partition.run()
                        var partial = List[Int]()
                        for row in expected:
                            if row >= first:
                                partial.append(row)
                        assert_equal(partition^.into_rows(), partial)


def scalar_column[
    D: DType
](n: Int, offset: Int, mode: Int) raises -> Column[Scalar[D]]:
    var values = List[Scalar[D]]()
    var valid = List[Bool]()
    for row in range(n + offset):
        var present = row_valid(row, mode)
        values.append(row_payload[D](row, present))
        valid.append(present)
    return Column[Scalar[D]](values^, valid^).slice(offset, n)


def assert_gathered[
    D: DType
](actual: Column[Scalar[D]], indices: List[Int], offset: Int, mode: Int) raises:
    assert_equal(len(actual), len(indices))
    for row in range(len(indices)):
        var present = indices[row] >= 0 and row_valid(
            indices[row] + offset, mode
        )
        assert_equal(actual.is_null(row), not present)
        if present:
            comptime if D.is_floating_point():
                assert_float(
                    Float64(actual.value(row)),
                    Float64(row_payload[D](indices[row] + offset, True)),
                )
            else:
                assert_equal(
                    actual.value(row),
                    row_payload[D](indices[row] + offset, True),
                )


def gather_numeric[D: DType]() raises:
    for n in [0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var source = scalar_column[D](n, offset, mode)
                var ascending = List[Int]()
                var descending = List[Int]()
                var repeated = List[Int]()
                for row in range(n):
                    ascending.append(row)
                    descending.append(n - 1 - row)
                    repeated.append(row // 2)
                var patterns: List[List[Int]] = [
                    ascending^,
                    descending^,
                    repeated^,
                ]
                for indices in patterns:
                    # Column.take is the scalar loop; gather_scalars copies
                    # consecutive runs in bulk and gathers validity separately.
                    assert_gathered[D](
                        source.take(indices), indices, offset, mode
                    )
                    assert_gathered[D](
                        gather_scalars[D](source, indices, False),
                        indices,
                        offset,
                        mode,
                    )
                var missing = List[Int]()
                for row in range(n):
                    missing.append(-1 if row % 3 == 0 else row)
                missing.append(-1)
                assert_gathered[D](
                    source.take_or_null(missing, Scalar[D](0)),
                    missing,
                    offset,
                    mode,
                )
                assert_gathered[D](
                    gather_scalars[D](source, missing, True),
                    missing,
                    offset,
                    mode,
                )
                var shifted: List[Int] = [999]
                for row in missing:
                    shifted.append(row + 101 if row >= 0 else -1)
                shifted.append(999)
                assert_gathered[D](
                    gather_scalars[D](
                        source,
                        shifted,
                        True,
                        first=1,
                        last=len(shifted) - 1,
                        base=101,
                    ),
                    missing,
                    offset,
                    mode,
                )
                for bad in [n, -2]:
                    with assert_raises():
                        _ = gather_scalars[D](source, [bad], True)
                with assert_raises():
                    _ = gather_scalars[D](source, [-1], False)


def test_numeric_gather_bulk_runs_and_scalar_fallback_match_typed_rows() raises:
    comptime for k in range(len(ROW_NUMERIC_DTYPES)):
        gather_numeric[ROW_NUMERIC_DTYPES[k]]()
    gather_numeric[DType.int128]()


def integer_row_kernels[D: DType, operation: Int]() raises:
    comptime op = ADD if operation == 0 else SUB if operation == 1 else MUL if operation == 2 else FLOORDIV if operation == 3 else MOD if operation == 4 else POW
    for n in [0, 1, 15, 16, 17, 31, 32, 33, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                for broadcast in range(3):
                    var a = List[Scalar[D]]()
                    var b = List[Scalar[D]]()
                    var av = List[Bool]()
                    var bv = List[Bool]()
                    for row in range(n + offset):
                        var left_valid = row_valid(row, mode)
                        var right_valid = row_valid(row + 1, mode)
                        var left_value = (row % 3 + 1) + (
                            3 if operation == 1 else 0
                        )
                        var right_value = row % 2 + 1
                        a.append(
                            Scalar[D](left_value) if left_valid else Scalar[
                                D
                            ].MAX
                        )
                        b.append(
                            Scalar[D](right_value) if right_valid else Scalar[
                                D
                            ].MAX
                        )
                        av.append(left_valid)
                        bv.append(right_valid)
                    var left = Column[Scalar[D]](a^, av).slice(offset, n)
                    var right = Column[Scalar[D]](b^, bv).slice(offset, n)
                    if broadcast == 1:
                        left = Column[Scalar[D]](
                            [Scalar[D](4 if operation == 1 else 2)]
                        )
                    if broadcast == 2:
                        right = Column[Scalar[D]]([Scalar[D](1)])
                    for guarded in [False, True]:
                        var mask = List[Bool]()
                        if guarded:
                            for row in range(n):
                                mask.append(row % 3 != 1)
                        var result = _numeric_int[op, D](left, right, mask)
                        assert_true(result.dtype() == DataType.of(D))
                        assert_equal(len(result), n)
                        for row in range(n):
                            var lrow = row + offset
                            var rrow = row + offset
                            var present = (
                                (broadcast == 1 or row_valid(lrow, mode))
                                and (
                                    broadcast == 2 or row_valid(rrow + 1, mode)
                                )
                                and (not guarded or row % 3 != 1)
                            )
                            assert_equal(result.get(row).is_null(), not present)
                            if present:
                                var x = Int128(
                                    4 if operation == 1 else 2
                                ) if broadcast == 1 else Int128(
                                    (lrow % 3 + 1)
                                    + (3 if operation == 1 else 0)
                                )
                                var y = Int128(1) if broadcast == 2 else Int128(
                                    rrow % 2 + 1
                                )
                                assert_equal(
                                    result.get(row)
                                    .numeric[D]()
                                    .cast[DType.int128](),
                                    checked_row_integer[D](operation, x, y),
                                )
                    comptime if operation < 3 and (
                        D == DType.int8 or D == DType.int16 or D == DType.int32
                    ):
                        if mode == 0 and n > 0:
                            # This is exactly _numeric_int's eligibility for
                            # the 16-lane specialization. Invoke it directly,
                            # then force the fallback with an explicit mask.
                            assert_equal(left.null_count(), 0)
                            assert_equal(right.null_count(), 0)
                            var vector = _vector_int[op, D](left, right, n)
                            var fallback = _numeric_int[op, D](
                                left, right, List[Bool](length=n, fill=True)
                            )
                            for row in range(n):
                                var lrow = row + offset
                                var x = Int128(
                                    4 if operation == 1 else 2
                                ) if broadcast == 1 else Int128(
                                    (lrow % 3 + 1)
                                    + (3 if operation == 1 else 0)
                                )
                                var y = Int128(1) if broadcast == 2 else Int128(
                                    lrow % 2 + 1
                                )
                                var expected = checked_row_integer[D](
                                    operation, x, y
                                )
                                assert_equal(
                                    vector.get(row)
                                    .numeric[D]()
                                    .cast[DType.int128](),
                                    expected,
                                )
                                assert_equal(
                                    fallback.get(row)
                                    .numeric[D]()
                                    .cast[DType.int128](),
                                    expected,
                                )


def test_integer_arithmetic_vector_and_guarded_scalar_routes_match_checked_model() raises:
    comptime for k in range(len(ROW_INTEGER_DTYPES)):
        comptime for operation in range(6):
            integer_row_kernels[ROW_INTEGER_DTYPES[k], operation]()


def integer_overflow_and_poison[D: DType, operation: Int]() raises:
    comptime op = ADD if operation == 0 else SUB if operation == 1 else MUL
    var limits = integer_bounds[D]()
    var x = limits[1] if operation != 1 else limits[0]
    var y = Int128(2) if operation == 2 else Int128(1)
    with assert_raises():
        _ = checked_row_integer[D](operation, x, y)
    for at in [0, 15, 16, 31, 32]:
        var a = List[Scalar[D]](length=33, fill=Scalar[D](3))
        var b = List[Scalar[D]](length=33, fill=Scalar[D](1))
        a[at] = x.cast[D]()
        b[at] = y.cast[D]()
        var left = Column[Scalar[D]](a.copy())
        var right = Column[Scalar[D]](b.copy())
        with assert_raises():
            _ = _numeric_int[op, D](left, right, List[Bool]())
        comptime if D == DType.int8 or D == DType.int16 or D == DType.int32:
            with assert_raises():
                _ = _vector_int[op, D](left, right, 33)
        var mask = List[Bool](length=33, fill=True)
        mask[at] = False
        var guarded = _numeric_int[op, D](left, right, mask)
        assert_true(guarded.get(at).is_null())
        var valid = List[Bool](length=33, fill=True)
        valid[at] = False
        var hidden = _numeric_int[op, D](
            Column[Scalar[D]](a^, valid), right, List[Bool]()
        )
        assert_true(hidden.get(at).is_null())
        for row in range(33):
            if row != at:
                var expected = checked_row_integer[D](operation, 3, 1)
                assert_equal(
                    guarded.get(row).numeric[D]().cast[DType.int128](), expected
                )
                assert_equal(
                    hidden.get(row).numeric[D]().cast[DType.int128](), expected
                )


def test_integer_overflow_is_checked_per_lane_and_never_reads_inactive_payloads() raises:
    comptime for k in range(len(ROW_INTEGER_DTYPES)):
        comptime for operation in range(3):
            integer_overflow_and_poison[ROW_INTEGER_DTYPES[k], operation]()


def float_row_kernels[D: DType, operation: Int, width: Int]() raises:
    comptime op = ADD if operation == 0 else SUB if operation == 1 else MUL if operation == 2 else DIV
    for n in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                for broadcast in range(3):
                    var left = scalar_column[D](n, offset, mode)
                    var right = scalar_column[D](n, offset + 1, mode)
                    if broadcast == 1:
                        left = Column[Scalar[D]]([Scalar[D](2)])
                    if broadcast == 2:
                        right = Column[Scalar[D]]([Scalar[D](4)])
                    var actual = _numeric_float[op, width, D](left, right)
                    assert_equal(len(actual), n)
                    assert_true(actual.dtype() == DataType.of(D))
                    for row in range(n):
                        var valid = (
                            broadcast == 1 or row_valid(row + offset, mode)
                        ) and (
                            broadcast == 2 or row_valid(row + offset + 1, mode)
                        )
                        assert_equal(actual.get(row).is_null(), not valid)
                        if valid:
                            var a = Scalar[D](
                                2
                            ) if broadcast == 1 else row_payload[D](
                                row + offset, True
                            )
                            var b = Scalar[D](
                                4
                            ) if broadcast == 2 else row_payload[D](
                                row + offset + 1, True
                            )
                            var expected: Scalar[D]
                            comptime if operation == 0:
                                expected = a + b
                            elif operation == 1:
                                expected = a - b
                            elif operation == 2:
                                expected = a * b
                            else:
                                expected = a / b
                            assert_float(
                                Float64(actual.get(row).numeric[D]()),
                                Float64(expected),
                            )


def test_float_arithmetic_simd_widths_and_tails_match_scalar_ieee_results() raises:
    comptime for k in range(2):
        comptime D = DType.float32 if k == 0 else DType.float64
        comptime for operation in range(4):
            float_row_kernels[D, operation, 1]()
            float_row_kernels[D, operation, 4]()
            float_row_kernels[D, operation, 8]()


def choose_numeric_rows[D: DType]() raises:
    for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                for broadcast in range(3):
                    var left = scalar_column[D](n, offset, mode)
                    var right = scalar_column[D](n, offset + 1, mode)
                    if broadcast == 1:
                        left = Column[Scalar[D]]([Scalar[D](2)], [mode != 2])
                    if broadcast == 2:
                        right = Column[Scalar[D]]([Scalar[D](4)], [mode != 2])
                    for density in range(5):
                        var selected = List[Bool]()
                        for row in range(n):
                            selected.append(row_selected(row, n, density))
                        var specialized = _choose_numeric[D](
                            selected, left, right
                        )
                        var fallback = _choose[Scalar[D]](selected, left, right)
                        var dispatched = choose(
                            selected,
                            Series("a", left.copy()),
                            Series("b", right.copy()),
                        )
                        assert_true(dispatched.dtype() == DataType.of(D))
                        assert_equal(len(dispatched), n)
                        for row in range(n):
                            var take_left = row_selected(row, n, density)
                            var scalar = (
                                broadcast == 1 if take_left else broadcast == 2
                            )
                            var valid = mode != 2 if scalar else row_valid(
                                row + offset + (0 if take_left else 1), mode
                            )
                            var expected = Scalar[D](
                                2 if take_left else 4
                            ) if scalar else row_payload[D](
                                row + offset + (0 if take_left else 1), True
                            )
                            assert_equal(specialized.is_null(row), not valid)
                            assert_equal(fallback.is_null(row), not valid)
                            assert_equal(
                                dispatched.get(row).is_null(), not valid
                            )
                            if valid:
                                comptime if D.is_floating_point():
                                    assert_float(
                                        Float64(specialized.value(row)),
                                        Float64(expected),
                                    )
                                    assert_float(
                                        Float64(fallback.value(row)),
                                        Float64(expected),
                                    )
                                    assert_float(
                                        Float64(
                                            dispatched.get(row).numeric[D]()
                                        ),
                                        Float64(expected),
                                    )
                                else:
                                    assert_equal(
                                        specialized.value(row), expected
                                    )
                                    assert_equal(fallback.value(row), expected)
                                    assert_equal(
                                        dispatched.get(row).numeric[D](),
                                        expected,
                                    )


def test_numeric_conditional_simd_and_fallback_choose_only_selected_validity() raises:
    comptime for k in range(len(ROW_NUMERIC_DTYPES)):
        choose_numeric_rows[ROW_NUMERIC_DTYPES[k]]()


def compare_numeric_rows[D: DType, op: Int]() raises:
    for n in [0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var left = scalar_column[D](n, offset, mode)
                var right = scalar_column[D](n, offset + 1, mode)
                var specialized = _compare_bits[op, D](left, right)
                var dispatched = binary[op](
                    Series("left", left.copy()), Series("right", right.copy())
                )
                # An explicit all-true evaluation mask disables the packed
                # comparison route and invokes the row numeric fallback.
                var fallback = binary[op](
                    Series("left", left.copy()),
                    Series("right", right.copy()),
                    List[Bool](length=n, fill=True),
                )
                assert_true(fallback.dtype() == DataType.BOOL)
                assert_true(specialized.dtype() == DataType.BOOL)
                assert_true(dispatched.dtype() == DataType.BOOL)
                for row in range(n):
                    var present = row_valid(row + offset, mode) and row_valid(
                        row + offset + 1, mode
                    )
                    assert_equal(specialized.get(row).is_null(), not present)
                    assert_equal(dispatched.get(row).is_null(), not present)
                    assert_equal(fallback.get(row).is_null(), not present)
                    if present:
                        var x = row_payload[D](row + offset, True)
                        var y = row_payload[D](row + offset + 1, True)
                        var expected: Bool
                        comptime if op == EQ:
                            expected = x == y
                        elif op == LT:
                            expected = x < y
                        else:
                            expected = x > y
                        assert_equal(specialized.get(row).bool(), expected)
                        assert_equal(dispatched.get(row).bool(), expected)
                        assert_equal(fallback.get(row).bool(), expected)


def test_numeric_comparison_bitmap_routes_preserve_validity_and_tail_bits() raises:
    comptime for k in range(len(ROW_NUMERIC_DTYPES)):
        compare_numeric_rows[ROW_NUMERIC_DTYPES[k], EQ]()
        compare_numeric_rows[ROW_NUMERIC_DTYPES[k], LT]()
        compare_numeric_rows[ROW_NUMERIC_DTYPES[k], GT]()


def float_special_rows[D: DType, op: Int, width: Int]() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var xs: List[Float64] = [0.0, -0.0, inf, -inf, nan, 3, 1, -2]
    var ys: List[Float64] = [-0.0, 0.0, 1, -1, 2, 0, inf, 0]
    for n in [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var av = List[Bool]()
                var bv = List[Bool]()
                var a = List[Scalar[D]]()
                var b = List[Scalar[D]]()
                for row in range(n + offset):
                    av.append(row_valid(row, mode))
                    bv.append(row_valid(row + 1, mode))
                    a.append(Scalar[D](xs[row % 8]))
                    b.append(Scalar[D](ys[row % 8]))
                var left = Column[Scalar[D]](a^, av).slice(offset, n)
                var right = Column[Scalar[D]](b^, bv).slice(offset, n)
                var actual = binary[op, width](
                    Series("a", left.copy()), Series("b", right.copy())
                )
                assert_true(
                    actual.dtype()
                    == (
                        DataType.BOOL if op == EQ
                        or op == GT else DataType.of(D)
                    )
                )
                assert_equal(len(actual), n)
                for row in range(n):
                    var original = row + offset
                    var present = row_valid(original, mode) and row_valid(
                        original + 1, mode
                    )
                    assert_equal(actual.get(row).is_null(), not present)
                    if present:
                        var x = Scalar[D](xs[original % 8])
                        var y = Scalar[D](ys[original % 8])
                        comptime if op == EQ or op == GT:
                            assert_true(actual.dtype() == DataType.BOOL)
                            assert_equal(
                                actual.get(row).bool(),
                                x == y if op == EQ else x > y,
                            )
                        else:
                            var expected: Scalar[D]
                            comptime if op == ADD:
                                expected = x + y
                            elif op == SUB:
                                expected = x - y
                            elif op == MUL:
                                expected = x * y
                            else:
                                expected = x / y
                            assert_float(
                                Float64(actual.get(row).numeric[D]()),
                                Float64(expected),
                            )


def test_float_special_values_keep_ieee_comparisons_and_signed_zero_at_all_widths() raises:
    comptime for k in range(2):
        comptime D = DType.float32 if k == 0 else DType.float64
        comptime for i in range(6):
            comptime op = ADD if i == 0 else SUB if i == 1 else MUL if i == 2 else DIV if i == 3 else EQ if i == 4 else GT
            float_special_rows[D, op, 1]()
            float_special_rows[D, op, 4]()
            float_special_rows[D, op, 8]()


def test_boolean_and_string_row_kernels_preserve_selected_payload_and_validity() raises:
    for n in [0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var bool_a = List[Bool]()
                var bool_b = List[Bool]()
                var text_a = List[String]()
                var text_b = List[String]()
                var valid_a = List[Bool]()
                var valid_b = List[Bool]()
                for row in range(n + offset):
                    bool_a.append(row % 2 == 0)
                    bool_b.append(row % 3 == 0)
                    text_a.append("λ\x00" + String(row))
                    text_b.append("β" + String(row))
                    valid_a.append(row_valid(row, mode))
                    valid_b.append(row_valid(row + 1, mode))
                var a = Series("a", BoolColumn(bool_a^, valid_a.copy())).slice(
                    offset, n
                )
                var b = Series("b", BoolColumn(bool_b^, valid_b.copy())).slice(
                    offset, n
                )
                var ta = Series("ta", Column[String](text_a^, valid_a^)).slice(
                    offset, n
                )
                var tb = Series("tb", Column[String](text_b^, valid_b^)).slice(
                    offset, n
                )
                for broadcast in range(3):
                    var left = a.copy()
                    var right = b.copy()
                    var text_left = ta.copy()
                    var text_right = tb.copy()
                    if broadcast == 1:
                        left = Series("a", BoolColumn([True], [mode != 2]))
                        text_left = Series(
                            "a", Column[String](["scalar λ"], [mode != 2])
                        )
                    if broadcast == 2:
                        right = Series("b", BoolColumn([False], [mode != 2]))
                        text_right = Series(
                            "b", Column[String](["scalar β"], [mode != 2])
                        )
                    for density in range(5):
                        var selected = List[Bool]()
                        for row in range(n):
                            selected.append(row_selected(row, n, density))
                        var actual_bool = choose(selected, left, right)
                        var actual_text = choose(
                            selected, text_left, text_right
                        )
                        assert_true(actual_bool.dtype() == DataType.BOOL)
                        assert_true(actual_text.dtype() == DataType.STRING)
                        for row in range(n):
                            var first = row_selected(row, n, density)
                            var scalar = (
                                broadcast == 1 if first else broadcast == 2
                            )
                            var present = mode != 2 if scalar else row_valid(
                                row + offset + (0 if first else 1), mode
                            )
                            assert_equal(
                                actual_bool.get(row).is_null(), not present
                            )
                            assert_equal(
                                actual_text.get(row).is_null(), not present
                            )
                            if present:
                                var boolean = (
                                    first if scalar else (row + offset) % 2
                                    == 0 if first else (row + offset) % 3
                                    == 0
                                )
                                var text = (
                                    String("scalar λ") if first else String(
                                        "scalar β"
                                    )
                                ) if scalar else (
                                    String("λ\x00") if first else String("β")
                                ) + String(
                                    row + offset
                                )
                                assert_equal(
                                    actual_bool.get(row).bool(), boolean
                                )
                                assert_equal(
                                    actual_text.get(row).string(), text
                                )
                var indices = List[Int]()
                for row in range(n):
                    indices.append(n - row - 1)
                var gathered_bool = a.take(indices)
                var gathered_text = ta.take(indices)
                for row in range(n):
                    var original = n - row - 1 + offset
                    var present = row_valid(original, mode)
                    assert_equal(gathered_bool.get(row).is_null(), not present)
                    assert_equal(gathered_text.get(row).is_null(), not present)
                    if present:
                        assert_equal(
                            gathered_bool.get(row).bool(), original % 2 == 0
                        )
                        assert_equal(
                            gathered_text.get(row).string(),
                            "λ\x00" + String(original),
                        )


def series_for_row_storage[
    D: DType
](column: Column[Scalar[D]]) raises -> Series:
    comptime if D == DType.int128:
        return Series._wrap("x", column.copy()).with_dtype(
            DataType.decimal(38, 0)
        )
    else:
        return Series("x", column.copy())


def chunk_row_gathers[D: DType]() raises:
    for n in [0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var source = series_for_row_storage[D](
                    scalar_column[D](n, offset, mode)
                )
                var cut1 = min(1, n)
                var cut2 = min(7, n)
                var chunked = Series._from_chunks(
                    [
                        source.slice(0, cut1),
                        source.slice(cut1, cut2 - cut1),
                        source.slice(cut2, n - cut2),
                    ]
                )
                if n > 1:
                    assert_true(chunked.is_chunked())
                    assert_true(
                        can_filter_aligned_chunks(
                            [chunked.copy(), chunked.copy()]
                        )
                    )
                    assert_true(
                        not can_filter_aligned_chunks(
                            [source.copy(), chunked.copy()]
                        )
                    )
                for density in range(5):
                    var indices = expected_selection(n, density, 0)
                    # workers=1 fixes execution policy; these calls still
                    # exercise chunk-local gather and generic scatter paths.
                    var sorted = take_sorted_chunked(
                        [chunked.copy()], indices.copy(), workers=1
                    )
                    var generic = take_parallel(
                        [chunked.copy()], indices.copy(), workers=1
                    )
                    for actual in [sorted[0].copy(), generic[0].copy()]:
                        assert_true(actual.dtype() == source.dtype())
                        var contiguous = actual.rechunk()
                        assert_gathered[D](
                            contiguous._data[Column[Scalar[D]]],
                            indices,
                            offset,
                            mode,
                        )
                    var repeated = List[Int]()
                    for index in indices:
                        repeated.append(index)
                        repeated.append(index)
                    var repeated_actual = take_sorted_chunked(
                        [chunked.copy()],
                        repeated.copy(),
                        workers=1,
                        allow_repeats=True,
                    )
                    var contiguous = repeated_actual[0].rechunk()
                    assert_gathered[D](
                        contiguous._data[Column[Scalar[D]]],
                        repeated,
                        offset,
                        mode,
                    )
                var outer: List[Int] = [-1]
                for row in range(n):
                    outer.append(n - 1 - row)
                    if row % 3 == 0:
                        outer.append(-1)
                var null_extended = take_parallel(
                    [chunked.copy()], outer.copy(), workers=1, or_null=True
                )
                var contiguous = null_extended[0].rechunk()
                assert_true(contiguous.dtype() == source.dtype())
                assert_gathered[D](
                    contiguous._data[Column[Scalar[D]]], outer, offset, mode
                )


def test_chunk_local_sorted_and_generic_gathers_preserve_all_storage_widths() raises:
    comptime for k in range(len(ROW_NUMERIC_DTYPES)):
        chunk_row_gathers[ROW_NUMERIC_DTYPES[k]]()
    chunk_row_gathers[DType.int128]()


def test_float_filter_direct_selection_matches_materialized_bitmap_and_scalar_rows() raises:
    var inf = Float64(1) / Float64(0)
    var nan = Float64(0) / Float64(0)
    var pattern: List[Float64] = [-inf, -2, -0.0, 0.0, 2, inf, nan]
    for n in [0, 1, 7, 8, 9, 63, 64, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var values = List[Float64]()
                var valid = List[Bool]()
                for row in range(n + offset):
                    values.append(pattern[row % 7])
                    valid.append(row_valid(row, mode))
                var source = Series(
                    "x", Column[Float64](values^, valid^)
                ).slice(offset, n)
                var cut = min(3, n)
                var chunked = Series._from_chunks(
                    [source.slice(0, cut), source.slice(cut, n - cut)]
                )
                for actual_source in [source.copy(), chunked.copy()]:
                    for literal in [-1.0, 0.0, 1.0, inf, nan]:
                        var expected_gt = List[Int]()
                        var expected_eq = List[Int]()
                        for row in range(n):
                            var value = pattern[(row + offset) % 7]
                            if row_valid(row + offset, mode):
                                if value > literal:
                                    expected_gt.append(row)
                                if value == literal:
                                    expected_eq.append(row)
                        assert_equal(
                            float_compare_rows(actual_source, GT, literal),
                            expected_gt,
                        )
                        assert_equal(
                            float_compare_rows(actual_source, EQ, literal),
                            expected_eq,
                        )
                        var operand = Series("lit", Column[Float64]([literal]))
                        var gt = binary[GT](actual_source, operand)
                        var eq = binary[EQ](actual_source, operand)
                        assert_equal(true_rows(gt.bool()), expected_gt)
                        assert_equal(true_rows(eq.bool()), expected_eq)


def integer_division_boundaries[D: DType]() raises:
    for op_index in range(2):
        # Zero divisors become null; poison zero at an already-null row
        # must follow exactly the same validity rule without evaluating it.
        var left = Column[Scalar[D]](
            [Scalar[D](5), Scalar[D].MAX, Scalar[D](7)], [True, False, True]
        )
        var right = Column[Scalar[D]](
            [Scalar[D](0), Scalar[D](0), Scalar[D](2)]
        )
        var actual = _numeric_int[FLOORDIV, D](
            left, right, List[Bool]()
        ) if op_index == 0 else _numeric_int[MOD, D](left, right, List[Bool]())
        assert_true(actual.get(0).is_null())
        assert_true(actual.get(1).is_null())
        assert_equal(
            actual.get(2).numeric[D]().cast[DType.int128](),
            Int128(3 if op_index == 0 else 1),
        )
    comptime if D.is_signed():
        var xs: List[Int128] = [-7, 7, -7, 7]
        var ys: List[Int128] = [3, -3, -3, 3]
        var a = List[Scalar[D]]()
        var b = List[Scalar[D]]()
        for row in range(4):
            a.append(xs[row].cast[D]())
            b.append(ys[row].cast[D]())
        var left = Column[Scalar[D]](a^)
        var right = Column[Scalar[D]](b^)
        var quotient = _numeric_int[FLOORDIV, D](left, right, List[Bool]())
        var remainder = _numeric_int[MOD, D](left, right, List[Bool]())
        var expected_q: List[Int128] = [-3, -3, 2, 2]
        var expected_r: List[Int128] = [2, -2, -1, 1]
        for row in range(4):
            assert_equal(
                quotient.get(row).numeric[D]().cast[DType.int128](),
                expected_q[row],
            )
            assert_equal(
                remainder.get(row).numeric[D]().cast[DType.int128](),
                expected_r[row],
            )
        with assert_raises():
            _ = _numeric_int[FLOORDIV, D](
                Column[Scalar[D]]([Scalar[D].MIN]),
                Column[Scalar[D]]([Scalar[D](-1)]),
                List[Bool](),
            )
        var mod_min = _numeric_int[MOD, D](
            Column[Scalar[D]]([Scalar[D].MIN]),
            Column[Scalar[D]]([Scalar[D](-1)]),
            List[Bool](),
        )
        assert_equal(mod_min.get(0).numeric[D](), Scalar[D](0))
        with assert_raises():
            _ = _numeric_int[POW, D](
                Column[Scalar[D]]([Scalar[D](2)]),
                Column[Scalar[D]]([Scalar[D](-1)]),
                List[Bool](),
            )


def test_integer_division_remainder_and_power_boundaries_have_checked_semantics() raises:
    comptime for k in range(len(ROW_INTEGER_DTYPES)):
        integer_division_boundaries[ROW_INTEGER_DTYPES[k]]()


def decimal_series_for_rows[
    width: Int
](n: Int, offset: Int, scale: Int, mode: Int, right: Bool) raises -> Series:
    comptime D = DType.int32 if width == 32 else DType.int64 if width == 64 else DType.int128
    var values = List[Scalar[D]]()
    var valid = List[Bool]()
    var xs: List[Int128] = [125, 375, -125, -375, 2000, 1, 100, -100]
    var ys: List[Int128] = [50, 10, 50, 10, 300, 2, 0, 0]
    for row in range(n + offset):
        var present = row_valid(row + (1 if right else 0), mode)
        var value = ys[row % 8] if right else xs[row % 8]
        values.append(value.cast[D]() if present else Scalar[D].MAX)
        valid.append(present)
    var dtype = DataType.decimal(
        9 if width == 32 else 18 if width == 64 else 38, scale, width
    )
    return (
        Series._wrap("d", Column[Scalar[D]](values^, valid^))
        .with_dtype(dtype)
        .slice(offset, n)
    )


def duckdb_decimal_type(
    operation: Int, left: DataType, right: DataType
) raises -> DataType:
    """DuckDB's result type: a sum or difference at the larger scale with
    one more integer digit, a product at the scales added, a quotient as
    DOUBLE; at most 38 digits (18 for 64-bit inputs), stored at the width
    the precision needs."""
    if operation == 3:
        return DataType.FLOAT64
    var scale: Int
    var precision: Int
    if operation == 2:
        scale = left.scale() + right.scale()
        precision = min(38, left.precision() + right.precision())
    else:
        scale = max(left.scale(), right.scale())
        precision = min(
            38,
            max(
                left.precision() - left.scale(),
                right.precision() - right.scale(),
            )
            + scale
            + 1,
        )
    # Inputs that fit 64 bits keep a 64-bit result (a product only when
    # its scale is under 18).
    if (
        precision > 18
        and max(left.precision(), right.precision()) <= 18
        and (operation != 2 or scale < 18)
    ):
        precision = 18
    var width = 32 if precision <= 9 else 64 if precision <= 18 else 128
    return DataType.decimal(precision, scale, width)


def decimal_row_kernels[
    left_width: Int, right_width: Int, operation: Int
]() raises:
    comptime op = ADD if operation == 0 else SUB if operation == 1 else MUL if operation == 2 else DIV
    comptime A = DType.int64 if left_width != 128 else DType.int128
    comptime B = DType.int64 if right_width != 128 else DType.int128
    var xs: List[Int128] = [125, 375, -125, -375, 2000, 1, 100, -100]
    var ys: List[Int128] = [50, 10, 50, 10, 300, 2, 0, 0]
    var scale_pairs: List[Tuple[Int, Int]] = [(0, 0), (2, 2), (2, 3), (3, 2)]
    for scales in scale_pairs:
        for n in [0, 1, 7, 8, 9, 16, 17, 65]:
            for offset in [0, 1, 7]:
                for mode in range(3):
                    var left = decimal_series_for_rows[left_width](
                        n, offset, scales[0], mode, False
                    )
                    var right = decimal_series_for_rows[right_width](
                        n, offset, scales[1], mode, True
                    )
                    var actual = binary[op](left, right)
                    var fallback = binary[op](
                        left, right, List[Bool](length=n, fill=True)
                    )
                    var expected_type = duckdb_decimal_type(
                        operation, left.dtype(), right.dtype()
                    )
                    assert_true(actual.dtype() == expected_type)
                    assert_true(fallback.dtype() == expected_type)
                    assert_equal(len(actual), n)
                    for row in range(n):
                        var position = row + offset
                        var present = (
                            row_valid(position, mode)
                            and row_valid(position + 1, mode)
                            and (operation != 3 or ys[position % 8] != 0)
                        )
                        assert_equal(actual.get(row).is_null(), not present)
                        assert_equal(fallback.get(row).is_null(), not present)
                        if present and operation == 3:
                            # Quotients are DOUBLE, from each side's value.
                            var quotient = (
                                Float64(xs[position % 8])
                                / Float64(decimal_power(scales[0]))
                            ) / (
                                Float64(ys[position % 8])
                                / Float64(decimal_power(scales[1]))
                            )
                            assert_equal(actual.get(row).float64(), quotient)
                            assert_equal(fallback.get(row).float64(), quotient)
                        elif present:
                            var expected = decimal_row_result(
                                operation,
                                xs[position % 8],
                                ys[position % 8],
                                scales[0],
                                scales[1],
                            )
                            assert_equal(actual._decimal_raw(row), expected)
                            assert_equal(fallback._decimal_raw(row), expected)
                    comptime if operation < 3:
                        if mode == 0:
                            var a = left._decimal64()
                            var b = right._decimal64()
                            assert_equal(a.null_count(), 0)
                            assert_equal(b.null_count(), 0)
                            var dense = _dense_decimal[op, A, B](
                                a._data[Column[Scalar[A]]],
                                b._data[Column[Scalar[B]]],
                                a.dtype(),
                                b.dtype(),
                                expected_type,
                            )
                            for row in range(n):
                                var position = row + offset
                                assert_equal(
                                    dense._decimal_raw(row),
                                    decimal_row_result(
                                        operation,
                                        xs[position % 8],
                                        ys[position % 8],
                                        scales[0],
                                        scales[1],
                                    ),
                                )


def test_decimal_storage_widths_dense_and_masked_arithmetic_match_scaled_integer_model() raises:
    comptime widths = (32, 64, 128)
    comptime for left in range(3):
        comptime for right in range(3):
            comptime for operation in range(4):
                decimal_row_kernels[widths[left], widths[right], operation]()


def test_decimal_precision_overflow_never_evaluates_null_or_inactive_slots() raises:
    var dtype = DataType.decimal(38, 0)
    var largest = decimal_power(38) - 1
    for at in [0, 7, 8, 15, 16]:
        var values = List[Int128](length=17, fill=2)
        values[at] = largest
        var left = Series._wrap("a", Column[Int128](values.copy())).with_dtype(
            dtype
        )
        var right = Series._wrap(
            "b", Column[Int128](List[Int128](length=17, fill=1))
        ).with_dtype(dtype)
        with assert_raises():
            _ = binary[ADD](left, right)
        var active = List[Bool](length=17, fill=True)
        active[at] = False
        var guarded = binary[ADD](left, right, active)
        assert_true(guarded.get(at).is_null())
        var present = List[Bool](length=17, fill=True)
        present[at] = False
        values[at] = Int128.MAX
        var hidden = Series._wrap(
            "a", Column[Int128](values^, present^)
        ).with_dtype(dtype)
        var masked = binary[ADD](hidden, right)
        assert_true(masked.get(at).is_null())
        for row in range(17):
            if row != at:
                assert_equal(guarded._decimal_raw(row), Int128(3))
                assert_equal(masked._decimal_raw(row), Int128(3))


def decimal_choose_rows[left_width: Int, right_width: Int]() raises:
    var xs: List[Int128] = [125, 375, -125, -375, 2000, 1, 100, -100]
    var ys: List[Int128] = [50, 10, 50, 10, 300, 2, 0, 0]
    for n in [0, 1, 7, 8, 9, 16, 17, 65]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                for broadcast in range(3):
                    var left = decimal_series_for_rows[left_width](
                        n, offset, 2, mode, False
                    )
                    var right = decimal_series_for_rows[right_width](
                        n, offset, 2, mode, True
                    )
                    if broadcast == 1:
                        left = decimal_series_for_rows[left_width](
                            1, 0, 2, mode, False
                        )
                    if broadcast == 2:
                        right = decimal_series_for_rows[right_width](
                            1, 0, 2, mode, True
                        )
                    for density in range(5):
                        var selected = List[Bool]()
                        for row in range(n):
                            selected.append(row_selected(row, n, density))
                        var actual = choose(selected, left, right)
                        # One scale: the wider precision, at its width.
                        var expected_type = DataType.decimal(
                            max(
                                left.dtype().precision(),
                                right.dtype().precision(),
                            ),
                            2,
                            max(left_width, right_width),
                        )
                        assert_true(actual.dtype() == expected_type)
                        assert_equal(len(actual), n)
                        for row in range(n):
                            var first = row_selected(row, n, density)
                            var scalar = (
                                broadcast == 1 if first else broadcast == 2
                            )
                            var position = 0 if scalar else row + offset
                            var present = row_valid(
                                position + (0 if first else 1), mode
                            )
                            assert_equal(actual.get(row).is_null(), not present)
                            if present:
                                assert_equal(
                                    actual._decimal_raw(row),
                                    xs[position % 8] if first else ys[
                                        position % 8
                                    ],
                                )
                var bad_scale = decimal_series_for_rows[right_width](
                    n, offset, 3, mode, True
                )
                # Another scale: both at the larger one, exactly (DuckDB's
                # common type), so the left values gain a digit.
                var rescaled = choose(
                    List[Bool](length=n, fill=True),
                    decimal_series_for_rows[left_width](
                        n, offset, 2, mode, False
                    ),
                    bad_scale,
                )
                assert_equal(rescaled.dtype().scale(), 3)
                for row in range(n):
                    if row_valid(row + offset, mode):
                        assert_equal(
                            rescaled._decimal_raw(row),
                            xs[(row + offset) % 8] * 10,
                        )


def test_decimal_conditional_storage_and_common_type_follow_selected_branch() raises:
    comptime widths = (32, 64, 128)
    comptime for left in range(3):
        comptime for right in range(3):
            decimal_choose_rows[widths[left], widths[right]]()
