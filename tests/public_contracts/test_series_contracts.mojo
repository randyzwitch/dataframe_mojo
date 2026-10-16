"""Standalone Series APIs obey positional, typed scalar and validity contracts."""
# contract-group: series.api
from std.math import sqrt
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import AnyValue, Column, DataType, Series, col, lit
from tests.support.arithmetic_reference import integer_result, float_result
from tests.support.coverage_compare import assert_float, assert_value
from tests.support.numeric_model import INTEGER_DTYPES, FLOAT_DTYPES


def _binary(left: Series, right: Series, op: Int) raises -> Series:
    if op == 0:
        return left + right
    if op == 1:
        return left - right
    if op == 2:
        return left * right
    if op == 3:
        return left // right
    if op == 4:
        return left % right
    if op == 5:
        return left**right
    if op == 6:
        return left / right
    if op == 7:
        return left < right
    if op == 8:
        return left <= right
    if op == 9:
        return left > right
    if op == 10:
        return left >= right
    if op == 11:
        return left.eq(right)
    return left.ne(right)


def _numeric[D: DType]() raises:
    var xs: List[Int128] = [4, 3, 2, 5, 2]
    var ys: List[Int128] = [1, 2, 0, 3, 2]
    var lv: List[Bool] = [True, True, False, True, True]
    var rv: List[Bool] = [True, False, True, True, True]
    var x = List[Scalar[D]]()
    var y = List[Scalar[D]]()
    for row in range(5):
        x.append(xs[row].cast[D]())
        y.append(ys[row].cast[D]())
    for same_name in [False, True]:
        var left = Series("left", Column[Scalar[D]](x.copy(), lv.copy()))
        var right = Series(
            "left" if same_name else "right",
            Column[Scalar[D]](y.copy(), rv.copy()),
        )
        for op in range(13):
            var actual = _binary(left, right, op)
            var dtype = (
                DataType.BOOL if op >= 7 else DataType.FLOAT64 if op == 6
                and D != DType.float32 else DataType.of(D)
            )
            assert_true(actual.dtype() == dtype)
            assert_equal(actual.name(), "left")
            assert_equal(len(actual), 5)
            for row in range(5):
                assert_equal(
                    actual.get(row).is_null(), not (lv[row] and rv[row])
                )
                if not lv[row] or not rv[row]:
                    continue
                if op >= 7:
                    var a = xs[row]
                    var b = ys[row]
                    var expected = (
                        a
                        < b if op
                        == 7 else a
                        <= b if op
                        == 8 else a
                        > b if op
                        == 9 else a
                        >= b if op
                        == 10 else a
                        == b if op
                        == 11 else a
                        != b
                    )
                    assert_equal(actual.get(row).bool(), expected)
                elif op == 6:
                    comptime if D == DType.float32:
                        assert_float(
                            Float64(actual.get(row).numeric[D]()),
                            Float64(Float32(xs[row]) / Float32(ys[row])),
                        )
                    else:
                        assert_float(
                            actual.get(row).float64(),
                            Float64(xs[row]) / Float64(ys[row]),
                        )
                else:
                    comptime if D.is_floating_point():
                        assert_float(
                            Float64(actual.get(row).numeric[D]()),
                            float_result(
                                Float64(xs[row]), Float64(ys[row]), op
                            ),
                        )
                    else:
                        var expected = integer_result[D](op, xs[row], ys[row])
                        assert_true(Bool(expected))
                        assert_equal(
                            Int128(actual.get(row).numeric[D]()),
                            expected.value(),
                        )
        var scalar = left + lit(Scalar[D](1))
        assert_equal(scalar.name(), "left")
        assert_true(scalar.dtype() == DataType.of(D))
        for row in range(5):
            assert_equal(scalar.get(row).is_null(), not lv[row])
            if lv[row]:
                assert_equal(
                    scalar.get(row).numeric[D](), (xs[row] + 1).cast[D]()
                )
        with assert_raises(contains="Series lengths differ"):
            _ = left + right.head(2)


def test_series_numeric_operators_pair_positions_and_preserve_left_names() raises:
    comptime for i in range(len(INTEGER_DTYPES)):
        _numeric[INTEGER_DTYPES[i]]()
    comptime for i in range(len(FLOAT_DTYPES)):
        _numeric[FLOAT_DTYPES[i]]()
    with assert_raises(contains="requires matching dtypes"):
        _ = Series("x", Column[Int64]([1])) + Series("y", Column[Float64]([1]))


def test_series_reductions_return_correct_tagged_values_and_empty_identities() raises:
    var source = Series(
        "x",
        Column[Int64](
            [2, -1, 9, 2, 4, 0], [True, True, False, True, True, False]
        ),
    )
    # Valid values [2,-1,2,4] have exact total7 and centered square total12.75.
    var integers = [
        source.sum(),
        source.min(),
        source.max(),
        source.first(),
        source.last(),
    ]
    for index in range(len(integers)):
        assert_true(integers[index].dtype() == DataType.INT64)
        assert_equal(integers[index].is_null(), index == 4)
        if index < 4:
            assert_equal(integers[index].int64(), Int64([7, -1, 4, 2][index]))
    assert_equal(source.count(), 4)
    assert_equal(source.n_unique(), 4)
    var floats = [
        source.mean(),
        source.median(),
        source.quantile(0),
        source.var(0),
        source.var(1),
        source.std(1),
    ]
    var expected: List[Float64] = [
        1.75,
        2,
        -1,
        12.75 / 4,
        12.75 / 3,
        sqrt(Float64(12.75 / 3)),
    ]
    for index in range(len(floats)):
        assert_true(floats[index].dtype() == DataType.FLOAT64)
        assert_float(floats[index].float64(), expected[index])
    for s in [source.slice(0, 0), Series.full_null("x", DataType.INT64, 3)]:
        assert_equal(s.sum().int64(), Int64(0))
        assert_equal(s.count(), 0)
        for result in [
            s.min(),
            s.max(),
            s.first(),
            s.last(),
            s.mean(),
            s.median(),
            s.var(),
            s.std(),
        ]:
            assert_true(result.is_null())
    var valid: List[Bool] = [True, False, True]
    var flags = Series("b", Column[Bool]([True, False, True], valid^))
    assert_true(flags.any().dtype() == DataType.BOOL)
    assert_true(flags.any().bool())
    assert_true(flags.all().bool())
    assert_true(flags.all(ignore_nulls=False).is_null())
    assert_true(flags.any(ignore_nulls=False).bool())
    var empty = flags.slice(0, 0)
    assert_true(empty.all().bool())
    assert_true(not empty.any().bool())


def test_series_elementwise_access_and_apply_preserve_names_validity_and_bounds() raises:
    var source = Series("x", Column[Int64]([-3, 9, 2], [True, False, True]))
    var results = [
        source.abs(),
        -source,
        source.fill_null(lit(Int64(7))),
        source.apply(col("x") * 2),
    ]
    var expected: List[List[Int64]] = [
        [3, 0, 2],
        [3, 0, -2],
        [-3, 7, 2],
        [-6, 0, 4],
    ]
    for operation in range(len(results)):
        assert_equal(results[operation].name(), "x")
        assert_true(results[operation].dtype() == DataType.INT64)
        for row in range(3):
            var absent = row == 1 and operation != 2
            assert_equal(results[operation].get(row).is_null(), absent)
            if not absent:
                assert_equal(
                    results[operation].get(row).int64(),
                    expected[operation][row],
                )
    assert_equal(source[-1].int64(), Int64(2))
    assert_equal(source[-3].int64(), Int64(-3))
    assert_true(source[1].is_null())
    for index in [-4, 3]:
        with assert_raises():
            _ = source[index]
    var values = source.to_values()
    assert_equal(len(values), 3)
    for row in range(3):
        assert_value(values[row].copy(), source.get(row))
    for row in range(3):
        assert_equal(source.is_null().get(row).bool(), row == 1)
        assert_equal(source.is_not_null().get(row).bool(), row != 1)


def test_series_boolean_operators_follow_kleene_truth_tables_positionally() raises:
    for left in range(3):
        for right in range(3):
            var a = Series("a", Column[Bool]([left == 1], [left != 2]))
            var b = Series("a", Column[Bool]([right == 1], [right != 2]))
            var outputs = [a & b, a | b, a ^ b, ~a]
            var expected: List[Int] = [
                0 if left == 0
                or right == 0 else 2 if left == 2
                or right == 2 else 1,
                1 if left == 1
                or right == 1 else 2 if left == 2
                or right == 2 else 0,
                2 if left == 2 or right == 2 else Int(left != right),
                2 if left == 2 else 1 - left,
            ]
            for operation in range(4):
                assert_true(outputs[operation].dtype() == DataType.BOOL)
                assert_equal(outputs[operation].name(), "a")
                assert_equal(len(outputs[operation]), 1)
                assert_equal(
                    outputs[operation].get(0).is_null(),
                    expected[operation] == 2,
                )
                if expected[operation] != 2:
                    assert_equal(
                        outputs[operation].get(0).bool(),
                        expected[operation] == 1,
                    )


def test_series_sort_unique_and_value_counts_return_complete_typed_rows() raises:
    var source = Series(
        "x", Column[Int64]([2, -1, 2, 4, 99], [True, True, True, True, False])
    )
    var ascending = source.sort()
    var descending = source.sort(descending=True, nulls_last=False)
    var distinct = source.unique(maintain_order=True)
    var expected: List[List[Int64]] = [
        [-1, 2, 2, 4, 0],
        [0, 4, 2, 2, -1],
        [2, -1, 4, 0],
    ]
    for operation in range(3):
        var actual = (
            ascending.copy() if operation
            == 0 else descending.copy() if operation
            == 1 else distinct.copy()
        )
        assert_true(actual.dtype() == DataType.INT64)
        assert_equal(actual.name(), "x")
        assert_equal(len(actual), 4 if operation == 2 else 5)
        for row in range(len(actual)):
            var absent = row == 0 if operation == 1 else row == len(actual) - 1
            assert_equal(actual.get(row).is_null(), absent)
            if not absent:
                assert_equal(actual.get(row).int64(), expected[operation][row])
    for sort in [False, True]:
        var actual = source.value_counts(sort=sort, name="frequency")
        assert_equal(actual.columns(), ["x", "frequency"])
        assert_equal(actual.dtypes(), [DataType.INT64, DataType.INT64])
        assert_equal(actual.height(), 4)
        for row in range(4):
            assert_equal(actual.item(row, "x").is_null(), row == 3)
            if row != 3:
                assert_equal(
                    actual.item(row, "x").int64(), Int64([2, -1, 4][row])
                )
            assert_equal(
                actual.item(row, "frequency").int64(),
                Int64(2 if row == 0 else 1),
            )
