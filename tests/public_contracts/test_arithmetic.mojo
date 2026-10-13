"""Numeric element semantics independent of executor/kernel equivalence.

Generators partition mathematical cases into values, null results, and errors;
errors are then exercised as valid cells and as hidden null payloads. Integer
expectations use exact Int128 references, never floating-point range checks.
"""
from std.math import abs, isnan, isinf
from std.memory import bitcast
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, DataType, Series, Expr, col, lit
from tests.support.numeric_model import (
    INTEGER_DTYPES,
    FLOAT_DTYPES,
    integer_bounds,
)
from tests.support.arithmetic_reference import (
    float_result,
    infinite_divisor_remainder,
    reference_floor,
    reference_ceil,
    ADD,
    SUB,
    MUL,
    FLOOR_DIV,
    MOD,
    POW,
    TRUE_DIV,
    integer_result,
    checked_product,
    floor_quotient,
)
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_frame, assert_float


def _operation(op: Int, left: Expr, right: Expr) -> Expr:
    if op == ADD:
        return left + right
    if op == SUB:
        return left - right
    if op == MUL:
        return left * right
    if op == FLOOR_DIV:
        return left // right
    if op == MOD:
        return left % right
    if op == POW:
        return left.pow(right)
    return left / right


def _layout(frame: DataFrame, mode: Int) raises -> DataFrame:
    return (
        frame.copy() if mode
        == 0 else offset_layout(frame) if mode
        == 1 else chunk_layout(frame)
    )


def _integer_values[D: DType]() -> List[Int128]:
    var bounds = integer_bounds[D]()
    var values: List[Int128] = [
        bounds[0],
        bounds[0] + 1,
        0,
        1,
        2,
        7,
        bounds[1] - 1,
        bounds[1],
    ]
    comptime if D.is_signed():
        values.append(-1)
        values.append(-7)
    return values^


def _integer_operator[D: DType](op: Int) raises:
    var values = _integer_values[D]()
    var rights = values.copy()
    if op == POW:
        rights = [0, 1, 2, 3, 7, 63, 64]
        comptime if D.is_signed():
            rights.append(-1)
    var left_payload = List[Scalar[D]]()
    var right_payload = List[Scalar[D]]()
    var expected_payload = List[Scalar[D]]()
    var expected_valid = List[Bool]()
    var input_valid = List[Bool]()
    var bad_left = List[Scalar[D]]()
    var bad_right = List[Scalar[D]]()
    for left in values:
        for right in rights:
            var result: Optional[Int128] = None
            var bad = False
            try:
                result = integer_result[D](op, left, right)
            except:
                bad = True
            left_payload.append(left.cast[D]())
            right_payload.append(right.cast[D]())
            expected_payload.append(
                result.value().cast[D]() if result else Scalar[D](0)
            )
            input_valid.append(not bad)
            expected_valid.append(not bad and Bool(result))
            if bad:
                bad_left.append(left.cast[D]())
                bad_right.append(right.cast[D]())
    var source = DataFrame(
        [
            Series("a", Column[Scalar[D]](left_payload^, input_valid.copy())),
            Series("b", Column[Scalar[D]](right_payload^, input_valid.copy())),
        ]
    )
    var expected = DataFrame(
        [
            Series(
                "result", Column[Scalar[D]](expected_payload^, expected_valid^)
            )
        ]
    )
    for layout in range(3):
        var frame = _layout(source, layout)
        assert_frame(
            frame.select(_operation(op, col("a"), col("b")).alias("result")),
            expected,
        )
        assert_frame(
            frame.clear().select(
                _operation(op, col("a"), col("b")).alias("result")
            ),
            expected.clear(),
        )
    # Every mathematical error is proved with that same payload made valid.
    # A null-filled companion ensures errors cannot be inferred from totals.
    for row in range(len(bad_left)):
        var bad = DataFrame(
            [
                Series(
                    "a",
                    Column[Scalar[D]](
                        [Scalar[D](0), bad_left[row], Scalar[D](0)],
                        [False, True, False],
                    ),
                ),
                Series(
                    "b",
                    Column[Scalar[D]](
                        [Scalar[D](0), bad_right[row], Scalar[D](0)],
                        [False, True, False],
                    ),
                ),
            ]
        )
        with assert_raises():
            _ = bad.select(_operation(op, col("a"), col("b")))
    var nulls = DataFrame(
        [
            Series(
                "a",
                Column[Scalar[D]]([Scalar[D](0), Scalar[D](1)], [False, False]),
            ),
            Series(
                "b",
                Column[Scalar[D]]([Scalar[D](0), Scalar[D](0)], [False, False]),
            ),
        ]
    )
    var actual = nulls.select(_operation(op, col("a"), col("b")))
    assert_true(actual.column("a").dtype() == DataType.of(D))
    assert_true(actual.item(0, "a").is_null())
    assert_true(actual.item(1, "a").is_null())


def _integer_true_division[D: DType]() raises:
    var values = _integer_values[D]()
    var left = List[Scalar[D]]()
    var right = List[Scalar[D]]()
    var expected = List[Float64]()
    var valid = List[Bool]()
    for a in values:
        for b in values:
            left.append(a.cast[D]())
            right.append(b.cast[D]())
            expected.append(Float64(a) / Float64(b))
            valid.append(True)
    left.append(integer_bounds[D]()[1].cast[D]())
    right.append(Scalar[D](0))
    valid.append(False)
    expected.append(0)
    var source = DataFrame(
        [
            Series("a", Column[Scalar[D]](left^, valid.copy())),
            Series("b", Column[Scalar[D]](right^)),
        ]
    )
    var answer = DataFrame(
        [Series("r", Column[Float64](expected^, valid.copy()))]
    )
    for layout in range(3):
        var frame = _layout(source, layout)
        assert_frame(frame.select((col("a") / col("b")).alias("r")), answer)
        assert_frame(
            frame.clear().select((col("a") / col("b")).alias("r")),
            answer.clear(),
        )


def test_integer_reference_has_literal_bounds_floor_modulo_and_power_anchors() raises:
    assert_equal(floor_quotient(-7, 3), Int128(-3))
    assert_equal(floor_quotient(7, -3), Int128(-3))
    assert_equal(floor_quotient(-7, -3), Int128(2))
    assert_equal(integer_result[DType.int8](MOD, -7, 3).value(), Int128(2))
    assert_equal(integer_result[DType.int8](MOD, 7, -3).value(), Int128(-2))
    assert_equal(integer_result[DType.int8](POW, -2, 7).value(), Int128(-128))
    assert_equal(
        integer_result[DType.uint64](ADD, Int128(UInt64.MAX) - 1, 1).value(),
        Int128(UInt64.MAX),
    )
    assert_true(not integer_result[DType.int8](FLOOR_DIV, 1, 0))
    with assert_raises():
        _ = checked_product[DType.uint64](
            Int128(UInt64.MAX), Int128(UInt64.MAX)
        )
    with assert_raises():
        _ = integer_result[DType.int8](FLOOR_DIV, -128, -1)
    with assert_raises():
        _ = integer_result[DType.uint8](SUB, 0, 1)
    with assert_raises():
        _ = integer_result[DType.int64](POW, 2, -1)


def test_integer_width_operator_matrix_checks_every_valid_cell_and_null_poison() raises:
    comptime for index in range(len(INTEGER_DTYPES)):
        for op in range(6):
            _integer_operator[INTEGER_DTYPES[index]](op)
        _integer_true_division[INTEGER_DTYPES[index]]()


def _integer_directions[D: DType]() raises:
    var values: List[Int128] = [0, 1, 2, 7]
    comptime if D.is_signed():
        values.append(-1)
        values.append(-7)
    var payloads = List[Scalar[D]]()
    for value in values:
        payloads.append(value.cast[D]())
    var frame = DataFrame([Series("x", Column[Scalar[D]](payloads^))])
    var expressions: List[Expr] = [
        col("x") + 2,
        2 + col("x"),
        col("x") - 2,
        2 - col("x"),
        col("x") * 2,
        2 * col("x"),
        col("x") // 2,
        2 // col("x"),
        col("x") % 2,
        2 % col("x"),
        col("x").pow(2),
        2 ** col("x"),
        col("x") / 2,
        2 / col("x"),
    ]
    var ops: List[Int] = [
        ADD,
        ADD,
        SUB,
        SUB,
        MUL,
        MUL,
        FLOOR_DIV,
        FLOOR_DIV,
        MOD,
        MOD,
        POW,
        POW,
        TRUE_DIV,
        TRUE_DIV,
    ]
    for expression in range(len(expressions)):
        var safe_indices = List[Int]()
        var expected_values = List[Scalar[D]]()
        var expected_valid = List[Bool]()
        for row in range(len(values)):
            var left = values[row] if expression % 2 == 0 else Int128(2)
            var right = Int128(2) if expression % 2 == 0 else values[row]
            if ops[expression] == TRUE_DIV:
                continue
            try:
                var expected = integer_result[D](ops[expression], left, right)
                safe_indices.append(row)
                expected_values.append(
                    expected.value().cast[D]() if expected else Scalar[D](0)
                )
                expected_valid.append(Bool(expected))
            except:
                with assert_raises():
                    _ = frame.take([row]).select(expressions[expression])
        if ops[expression] == TRUE_DIV:
            var actual = frame.select(expressions[expression].alias("r"))
            assert_true(actual.column("r").dtype() == DataType.FLOAT64)
            for row in range(len(values)):
                var left = Float64(
                    values[row]
                ) if expression % 2 == 0 else Float64(2)
                var right = Float64(2) if expression % 2 == 0 else Float64(
                    values[row]
                )
                assert_float(actual.item(row, "r").float64(), left / right)
        else:
            assert_frame(
                frame.take(safe_indices).select(
                    expressions[expression].alias("r")
                ),
                DataFrame(
                    [
                        Series(
                            "r",
                            Column[Scalar[D]](
                                expected_values^, expected_valid^
                            ),
                        )
                    ]
                ),
            )
    # Typed literals preserve the primitive width, while untyped integers adopt.
    assert_frame(
        frame.select(col("x") + 0), frame.select(col("x") + lit(Scalar[D](0)))
    )


def test_untyped_integer_adoption_and_every_scalar_operator_direction() raises:
    comptime for index in range(len(INTEGER_DTYPES)):
        _integer_directions[INTEGER_DTYPES[index]]()


def _integer_comparison_unary[D: DType]() raises:
    var values = _integer_values[D]()
    var a = List[Scalar[D]]()
    var b = List[Scalar[D]]()
    var valid = List[Bool]()
    var bv = List[Bool]()
    for left in values:
        for right in values:
            a.append(left.cast[D]())
            b.append(right.cast[D]())
            valid.append(True)
            bv.append(True)
    # Preserve all equal/unequal extrema pairs, then add independent nulls.
    for row in range(2):
        a.append(values[0].cast[D]())
        b.append(values[0].cast[D]())
        valid.append(row != 0)
        bv.append(row == 0)
    var source = DataFrame(
        [
            Series("a", Column[Scalar[D]](a.copy(), valid.copy())),
            Series("b", Column[Scalar[D]](b.copy(), bv.copy())),
        ]
    )
    for layout in range(3):
        var frame = _layout(source, layout)
        var actual = frame.select_exprs(
            [
                (col("a") < col("b")).alias("lt"),
                (col("a") <= col("b")).alias("le"),
                (col("a") > col("b")).alias("gt"),
                (col("a") >= col("b")).alias("ge"),
                col("a").eq(col("b")).alias("eq"),
                col("a").ne(col("b")).alias("ne"),
            ]
        )
        assert_equal(actual.columns(), ["lt", "le", "gt", "ge", "eq", "ne"])
        assert_equal(actual.height(), len(a))
        for row in range(len(a)):
            var left = a[row].cast[DType.int128]()
            var right = b[row].cast[DType.int128]()
            var expected: List[Bool] = [
                left < right,
                left <= right,
                left > right,
                left >= right,
                left == right,
                left != right,
            ]
            for index in range(6):
                var cell = actual.item(row, actual.columns()[index])
                assert_equal(cell.is_null(), not (valid[row] and bv[row]))
                if valid[row] and bv[row]:
                    assert_equal(cell.bool(), expected[index])
        assert_frame(
            frame.select_exprs(
                [
                    col("a").floor().alias("floor"),
                    col("a").ceil().alias("ceil"),
                    col("a").round(-2).alias("round"),
                ]
            ),
            DataFrame(
                [
                    frame.column("a").renamed("floor"),
                    frame.column("a").renamed("ceil"),
                    frame.column("a").renamed("round"),
                ]
            ),
        )
    for value in values:
        var single = DataFrame(
            [Series("a", Column[Scalar[D]]([value.cast[D]()]))]
        )
        var abs_valid = value >= 0 or -value <= integer_bounds[D]()[1]
        var neg_valid = (
            -value >= integer_bounds[D]()[0]
            and -value <= integer_bounds[D]()[1]
        )
        if abs_valid:
            assert_equal(
                single.select(col("a").abs())
                .item()
                .numeric[D]()
                .cast[DType.int128](),
                -value if value < 0 else value,
            )
        else:
            with assert_raises():
                _ = single.select(col("a").abs())
        if neg_valid:
            assert_equal(
                single.select(-col("a"))
                .item()
                .numeric[D]()
                .cast[DType.int128](),
                -value,
            )
        else:
            with assert_raises():
                _ = single.select(-col("a"))
        var hidden = DataFrame(
            [Series("a", Column[Scalar[D]]([value.cast[D]()], [False]))]
        )
        assert_true(hidden.select(-col("a")).item().is_null())
        assert_true(hidden.select(col("a").abs()).item().is_null())

    # Clip is exact at every integer width and math outputs widen to Float64.
    var small = DataFrame(
        [
            Series(
                "x",
                Column[Scalar[D]](
                    [Scalar[D](0), Scalar[D](1), Scalar[D](4), Scalar[D](7)],
                    [True, True, True, False],
                ),
            )
        ]
    )
    var math = small.select_exprs(
        [
            col("x").sqrt().alias("sqrt"),
            col("x").exp().alias("exp"),
            col("x").log().alias("log"),
            col("x").clip(1, 2).alias("clip"),
        ]
    )
    var sqrt_expected: List[Float64] = [0, 1, 2]
    var exp_expected: List[Float64] = [
        1,
        2.7182818284590452354,
        54.59815003314423908,
    ]
    var log_expected: List[Float64] = [
        -Float64(1) / Float64(0),
        0,
        1.3862943611198906194,
    ]
    var clip_expected: List[Int128] = [1, 1, 2]
    for row in range(3):
        assert_true(math.column("sqrt").dtype() == DataType.FLOAT64)
        assert_true(math.column("exp").dtype() == DataType.FLOAT64)
        assert_true(math.column("log").dtype() == DataType.FLOAT64)
        assert_true(math.column("clip").dtype() == DataType.of(D))
        _close(math.item(row, "sqrt").float64(), sqrt_expected[row], 5e-12)
        _close(math.item(row, "exp").float64(), exp_expected[row], 5e-12)
        _close(math.item(row, "log").float64(), log_expected[row], 5e-12)
        assert_equal(
            math.item(row, "clip").numeric[D]().cast[DType.int128](),
            clip_expected[row],
        )
    for name in math.columns():
        assert_true(math.item(3, name).is_null())


def test_all_integer_comparisons_unary_extrema_and_rounding_identity() raises:
    comptime for index in range(len(INTEGER_DTYPES)):
        _integer_comparison_unary[INTEGER_DTYPES[index]]()


# IEEE arithmetic and preserved values compare signed zero exactly. The API
# defines round/modulo numerically without specifying a zero sign, so their
# comparisons explicitly accept either sign. Transcendentals use a 5e-12
# Float64 relative/absolute envelope: std.math exp(4) has measured 1.7e-12
# relative error on this toolchain, consistent with the documented ~1e-12.
def _close(
    actual: Float64,
    expected: Float64,
    tolerance: Float64,
    context: String = "",
    zero_sign: Bool = True,
) raises:
    try:
        if expected == 0 and not zero_sign:
            assert_equal(actual, expected)
        elif isnan(expected) or isinf(expected) or expected == 0:
            assert_float(actual, expected)
        else:
            assert_true(
                abs(actual - expected)
                <= tolerance * max(abs(expected), Float64(1))
            )
    except error:
        raise Error(
            "numeric comparison "
            + context
            + ": actual="
            + String(actual)
            + " expected="
            + String(expected)
            + "; "
            + String(error)
        )


def _float_binary[D: DType]() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var left: List[Float64] = [
        -7.5,
        7.5,
        0.0,
        -0.0,
        inf,
        -inf,
        nan,
        2,
        -1,
        4,
        1.25,
        -0.0,
    ]
    var right: List[Float64] = [2, -2, 0.0, 0.0, 2, 2, 3, 3, 0.5, 0, 0.5, -2]
    var maximum = Float64(
        3.4028234663852886e38
    ) if D == DType.float32 else Float64(1.7976931348623157e308)
    var minimum_normal = Float64(
        1.1754943508222875e-38
    ) if D == DType.float32 else Float64(2.2250738585072014e-308)
    for value in [
        maximum,
        maximum,
        minimum_normal,
        Float64(1),
        Float64(-1),
        Float64(1),
    ]:
        left.append(value)
    for value in [Float64(2), maximum, Float64(2), nan, inf, -inf]:
        right.append(value)
    var a = List[Scalar[D]]()
    var b = List[Scalar[D]]()
    for row in range(len(left)):
        a.append(left[row].cast[D]())
        b.append(right[row].cast[D]())
    for mode in range(3):
        var av = List[Bool]()
        var bv = List[Bool]()
        for row in range(len(left)):
            av.append(mode == 0 or (mode == 1 and row % 3 != 0))
            bv.append(mode == 0 or (mode == 1 and row % 3 != 1))
        var source = DataFrame(
            [
                Series("a", Column[Scalar[D]](a.copy(), av.copy())),
                Series("b", Column[Scalar[D]](b.copy(), bv.copy())),
            ]
        )
        for layout in range(3):
            var frame = _layout(source, layout)
            for op in range(7):
                var actual = frame.select(
                    _operation(op, col("a"), col("b")).alias("r")
                )
                assert_true(actual.column("r").dtype() == DataType.of(D))
                assert_equal(actual.height(), len(left))
                for row in range(len(left)):
                    assert_equal(
                        actual.item(row, "r").is_null(),
                        not (av[row] and bv[row]),
                    )
                    if av[row] and bv[row]:
                        # The selected Python infinite-divisor policy has
                        # focused strict regression cases below; keep its
                        # known failure separate from the passing matrix.
                        if (
                            op == MOD
                            and isinf(right[row])
                            and not isinf(left[row])
                            and not isnan(left[row])
                        ):
                            continue
                        var expected = float_result(
                            left[row], right[row], op
                        ).cast[D]()
                        _close(
                            Float64(actual.item(row, "r").numeric[D]()),
                            Float64(expected),
                            1e-5 if D == DType.float32 else 5e-12,
                            String(D)
                            + " binary op="
                            + String(op)
                            + " row="
                            + String(row),
                            zero_sign=op != MOD,
                        )
                assert_true(
                    frame.clear()
                    .select(_operation(op, col("a"), col("b")))
                    .column("a")
                    .dtype()
                    == DataType.of(D)
                )
    # Untyped floating scalars adopt either float width, in both directions.
    var base = DataFrame(
        [
            Series(
                "x",
                Column[Scalar[D]]([Scalar[D](-2), Scalar[D](0), Scalar[D](2)]),
            )
        ]
    )
    var expressions: List[Expr] = [
        col("x") + 2.5,
        2.5 + col("x"),
        col("x") - 2.5,
        2.5 - col("x"),
        col("x") * 2.5,
        2.5 * col("x"),
        col("x") // 2.5,
        2.5 // col("x"),
        col("x") % 2.5,
        2.5 % col("x"),
        col("x").pow(2.5),
        2.5 ** col("x"),
        col("x") / 2.5,
        2.5 / col("x"),
    ]
    var ops: List[Int] = [
        ADD,
        ADD,
        SUB,
        SUB,
        MUL,
        MUL,
        FLOOR_DIV,
        FLOOR_DIV,
        MOD,
        MOD,
        POW,
        POW,
        TRUE_DIV,
        TRUE_DIV,
    ]
    for expression in range(len(expressions)):
        var actual = base.select(expressions[expression].alias("r"))
        assert_true(actual.column("r").dtype() == DataType.of(D))
        for row in range(3):
            var value = Float64(row * 2 - 2)
            var l = value if expression % 2 == 0 else Float64(2.5)
            var r = Float64(2.5) if expression % 2 == 0 else value
            _close(
                Float64(actual.item(row, "r").numeric[D]()),
                Float64(float_result(l, r, ops[expression]).cast[D]()),
                1e-5 if D == DType.float32 else 5e-12,
                zero_sign=ops[expression] != MOD,
            )


def test_float_width_operators_ieee_specials_and_scalar_directions() raises:
    comptime for index in range(len(FLOAT_DTYPES)):
        _float_binary[FLOAT_DTYPES[index]]()


def _float_comparisons[D: DType]() raises:
    var inf = Float64(1) / Float64(0)
    var nan = Float64(0) / Float64(0)
    var values: List[Float64] = [-inf, -1, -0.0, 0.0, 1, inf, nan]
    var a = List[Scalar[D]]()
    var b = List[Scalar[D]]()
    var av = List[Bool]()
    var bv = List[Bool]()
    for left in values:
        for right in values:
            a.append(left.cast[D]())
            b.append(right.cast[D]())
            av.append(True)
            bv.append(True)
    # Separate trailing nulls preserve every valid IEEE pair above.
    a.append(nan.cast[D]())
    b.append(nan.cast[D]())
    av.append(False)
    bv.append(True)
    a.append(Float64(1).cast[D]())
    b.append(Float64(1).cast[D]())
    av.append(True)
    bv.append(False)
    var source = DataFrame(
        [
            Series("a", Column[Scalar[D]](a.copy(), av.copy())),
            Series("b", Column[Scalar[D]](b.copy(), bv.copy())),
        ]
    )
    for layout in range(3):
        var actual = _layout(source, layout).select_exprs(
            [
                (col("a") < col("b")).alias("lt"),
                (col("a") <= col("b")).alias("le"),
                (col("a") > col("b")).alias("gt"),
                (col("a") >= col("b")).alias("ge"),
                col("a").eq(col("b")).alias("eq"),
                col("a").ne(col("b")).alias("ne"),
            ]
        )
        assert_equal(actual.columns(), ["lt", "le", "gt", "ge", "eq", "ne"])
        assert_equal(actual.height(), len(a))
        for row in range(len(a)):
            var expected: List[Bool] = [
                a[row] < b[row],
                a[row] <= b[row],
                a[row] > b[row],
                a[row] >= b[row],
                a[row] == b[row],
                a[row] != b[row],
            ]
            for index in range(6):
                var cell = actual.item(row, actual.columns()[index])
                assert_true(cell.dtype() == DataType.BOOL)
                assert_equal(cell.is_null(), not (av[row] and bv[row]))
                if av[row] and bv[row]:
                    assert_equal(cell.bool(), expected[index])


def test_float_comparisons_use_ieee_equality_and_independent_null_validity() raises:
    comptime for index in range(len(FLOAT_DTYPES)):
        _float_comparisons[FLOAT_DTYPES[index]]()


def _float_transforms[D: DType]() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var values: List[Float64] = [
        -2.5,
        -1.5,
        -0.5,
        -0.0,
        0.5,
        1.5,
        2.5,
        nan,
        inf,
        -inf,
        15,
        -15,
        1.125,
        -1.125,
        123,
    ]
    var payload = List[Scalar[D]]()
    for value in values:
        payload.append(value.cast[D]())
    var valid = List[Bool](length=len(values), fill=True)
    valid[len(values) - 1] = False
    var source = DataFrame(
        [Series("x", Column[Scalar[D]](payload^, valid.copy()))]
    )
    var rounded: List[Float64] = [
        -3,
        -2,
        -1,
        -0.0,
        1,
        2,
        3,
        nan,
        inf,
        -inf,
        15,
        -15,
        1,
        -1,
        0,
    ]
    var tenths: List[Float64] = [
        -0.0,
        -0.0,
        -0.0,
        -0.0,
        0,
        0,
        0,
        nan,
        inf,
        -inf,
        20,
        -20,
        0,
        -0.0,
        0,
    ]
    var hundredths: List[Float64] = [
        -2.5,
        -1.5,
        -0.5,
        -0.0,
        0.5,
        1.5,
        2.5,
        nan,
        inf,
        -inf,
        15,
        -15,
        1.13,
        -1.13,
        0,
    ]
    for layout in range(3):
        var frame = _layout(source, layout)
        var actual = frame.select_exprs(
            [
                (-col("x")).alias("neg"),
                col("x").abs().alias("abs"),
                col("x").floor().alias("floor"),
                col("x").ceil().alias("ceil"),
                col("x").round().alias("round"),
                col("x").round(-1).alias("tens"),
                col("x").round(2).alias("hundredths"),
                col("x").clip(-1, 1).alias("clip"),
                col("x").clip_min(nan).alias("nan_low"),
                col("x").clip_max(nan).alias("nan_high"),
            ]
        )
        for row in range(len(values)):
            var expected: List[Float64] = [
                -values[row],
                abs(values[row]),
                reference_floor(values[row]),
                reference_ceil(values[row]),
                rounded[row],
                tenths[row],
                hundredths[row],
                max(Float64(-1), min(Float64(1), values[row])) if not isnan(
                    values[row]
                ) else nan,
                values[row],
                values[row],
            ]
            for index in range(len(expected)):
                var cell = actual.item(row, actual.columns()[index])
                assert_true(cell.dtype() == DataType.of(D))
                assert_equal(cell.is_null(), not valid[row])
                if valid[row]:
                    _close(
                        Float64(cell.numeric[D]()),
                        Float64(expected[index].cast[D]()),
                        1e-5 if D == DType.float32 else 5e-12,
                        String(D)
                        + " transform="
                        + actual.columns()[index]
                        + " row="
                        + String(row),
                        zero_sign=index < 4 or index > 6,
                    )
        assert_equal(frame.clear().select(col("x").sqrt()).height(), 0)
    # Mathematical anchors avoid comparing transcendental kernels to themselves.
    var x = DataFrame(
        [
            Series(
                "x",
                Column[Scalar[D]](
                    [
                        Scalar[D](-1),
                        Scalar[D](0),
                        Scalar[D](1),
                        Scalar[D](4),
                        Scalar[D](123),
                    ],
                    [True, True, True, True, False],
                ),
            )
        ]
    )
    var math = x.select_exprs(
        [
            col("x").sqrt().alias("sqrt"),
            col("x").exp().alias("exp"),
            col("x").log().alias("log"),
        ]
    )
    var sqrt_expected: List[Float64] = [nan, 0, 1, 2]
    var exp_expected: List[Float64] = [
        0.3678794411714423216,
        1,
        2.7182818284590452354,
        54.59815003314423908,
    ]
    var log_expected: List[Float64] = [nan, -inf, 0, 1.3862943611198906194]
    for row in range(4):
        _close(
            Float64(math.item(row, "sqrt").numeric[D]()),
            sqrt_expected[row],
            1e-5 if D == DType.float32 else 5e-12,
        )
        _close(
            Float64(math.item(row, "exp").numeric[D]()),
            exp_expected[row],
            1e-5 if D == DType.float32 else 5e-12,
        )
        _close(
            Float64(math.item(row, "log").numeric[D]()),
            log_expected[row],
            1e-5 if D == DType.float32 else 5e-12,
        )
    assert_true(math.item(4, "sqrt").is_null())
    assert_true(math.item(4, "exp").is_null())
    assert_true(math.item(4, "log").is_null())


def test_float_unary_math_clip_and_half_away_rounding() raises:
    comptime for index in range(len(FLOAT_DTYPES)):
        _float_transforms[FLOAT_DTYPES[index]]()


def test_numeric_binding_rejects_mixed_typed_widths_and_unrepresentable_adoption() raises:
    var frame = DataFrame(
        [
            Series("i", Column[Int8]([1])),
            Series("u", Column[UInt8]([1])),
            Series("f", Column[Float32]([1])),
            Series("b", Column[Bool]([True])),
            Series("s", Column[String](["1"])),
        ]
    )
    for source in [frame.copy(), frame.clear()]:
        for invalid in [
            col("i") + 128,
            col("i") + -129,
            col("u") - -1,
            col("u") + 256,
            col("i") + 1.0,
            col("i") + lit(Int64(1)),
            col("f") + lit(Float64(1)),
            col("b") + col("b"),
            col("s").abs(),
            col("i") / col("f"),
            col("i") == lit(UInt8(1)),
            col("i").clip_min(lit(Int64(0))),
        ]:
            with assert_raises():
                _ = source.select(invalid)


def test_decimal_arithmetic_uses_exact_scale_half_even_and_null_zero_divisor() raises:
    var left: List[Int128] = [125, 75, -125, -75, 250, 100, 300, 625, 375, 999]
    var right: List[Int128] = [50, 50, 50, 50, 0, 800, 800, 100, 100, 0]
    var valid: List[Bool] = [
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
    ]
    var results: List[List[Int128]] = [
        [175, 125, -75, -25, 250, 900, 1100, 725, 475, 0],
        [75, 25, -175, -125, 250, -700, -500, 525, 275, 0],
        [62, 38, -62, -38, 0, 800, 2400, 625, 375, 0],
        [250, 150, -250, -150, 0, 12, 38, 625, 375, 0],
    ]
    for bits in [32, 64, 128]:
        var source = DataFrame(
            [
                Series("a", Column[Int128](left.copy(), valid.copy()))
                .with_dtype(DataType.decimal(8, 2))
                .cast(DataType.decimal(8, 2, bits)),
                Series("b", Column[Int128](right.copy()))
                .with_dtype(DataType.decimal(8, 2))
                .cast(DataType.decimal(8, 2, bits)),
            ]
        )
        for layout in range(3):
            var frame = _layout(source, layout)
            var expressions: List[Expr] = [
                col("a") + col("b"),
                col("a") - col("b"),
                col("a") * col("b"),
                col("a") / col("b"),
            ]
            for op in range(4):
                var result_valid = valid.copy()
                if op == 3:
                    result_valid[4] = False
                var expected = DataFrame(
                    [
                        Series(
                            "r",
                            Column[Int128](results[op].copy(), result_valid^),
                        ).with_dtype(DataType.decimal(38, 2))
                    ]
                )
                assert_frame(frame.select(expressions[op].alias("r")), expected)
                assert_frame(
                    frame.clear().select(expressions[op].alias("r")),
                    expected.clear(),
                )
    # Different scales align exactly and choose the larger output scale.
    var mixed = DataFrame(
        [
            Series("a", Column[Int128]([125])).with_dtype(
                DataType.decimal(4, 2)
            ),
            Series("b", Column[Int128]([375])).with_dtype(
                DataType.decimal(4, 3)
            ),
        ]
    )
    var output = mixed.select_exprs(
        [
            (col("a") + col("b")).alias("add"),
            (col("a") * col("b")).alias("mul"),
            (col("a") / col("b")).alias("div"),
        ]
    )
    assert_frame(
        output,
        DataFrame(
            [
                Series("add", Column[Int128]([1625])).with_dtype(
                    DataType.decimal(38, 3)
                ),
                Series("mul", Column[Int128]([469])).with_dtype(
                    DataType.decimal(38, 3)
                ),
                Series("div", Column[Int128]([3333])).with_dtype(
                    DataType.decimal(38, 3)
                ),
            ]
        ),
    )
    # Exact integers above 2^53 distinguish decimal arithmetic from float detours.
    var large = DataFrame(
        [
            Series("a", Column[Int128]([9007199254740993])).with_dtype(
                DataType.decimal(20, 0)
            ),
            Series("b", Column[Int128]([1])).with_dtype(
                DataType.decimal(20, 0)
            ),
        ]
    )
    assert_equal(
        String(large.select(col("a") + col("b")).item()), "9007199254740994"
    )


def test_decimal_precision_overflow_is_per_valid_row_and_scalars_keep_decimal_type() raises:
    var maximum = Int128(1)
    for _ in range(38):
        maximum *= 10
    maximum -= 1
    var dtype = DataType.decimal(38, 0)
    var frame = DataFrame(
        [
            Series(
                "a", Column[Int128]([maximum, -maximum, 7], [True, True, False])
            ).with_dtype(dtype)
        ]
    )
    var one = lit("1").cast(dtype)
    var two = lit("2").cast(dtype)
    with assert_raises():
        _ = frame.take([0]).select(col("a") + one)
    with assert_raises():
        _ = frame.take([1]).select(col("a") - one)
    with assert_raises():
        _ = frame.take([0]).select(col("a") * two)
    var poison = DataFrame(
        [
            Series(
                "a", Column[Int128]([maximum, -maximum], [False, False])
            ).with_dtype(dtype)
        ]
    )
    for expression in [col("a") + one, col("a") - one, col("a") * two]:
        var actual = poison.select(expression.alias("r"))
        assert_true(actual.column("r").dtype() == dtype)
        assert_true(actual.item(0, "r").is_null())
        assert_true(actual.item(1, "r").is_null())
    var small = DataFrame(
        [Series("a", Column[Int128]([125])).with_dtype(DataType.decimal(4, 2))]
    )
    var scalar = lit("0.50").cast(DataType.decimal(4, 2))
    var expressions: List[Expr] = [
        col("a") + scalar,
        scalar + col("a"),
        col("a") - scalar,
        scalar - col("a"),
        col("a") * scalar,
        scalar * col("a"),
        col("a") / scalar,
        scalar / col("a"),
    ]
    var expected: List[Int128] = [175, 175, 75, -75, 62, 62, 250, 40]
    for direction in range(len(expressions)):
        assert_frame(
            small.select(expressions[direction].alias("r")),
            DataFrame(
                [
                    Series(
                        "r", Column[Int128]([expected[direction]])
                    ).with_dtype(DataType.decimal(38, 2))
                ]
            ),
        )


def _nullable_clip[D: DType]() raises:
    var source_values: List[Int] = [0, 1, 2, 3, 7, 0, 3]
    var expected_min: List[Int] = [1, 1, 2, 3, 0, 0, 3]
    var expected_max: List[Int] = [0, 1, 2, 2, 0, 0, 0]
    var expected_both: List[Int] = [1, 1, 2, 2, 0, 0, 0]
    var x = List[Scalar[D]]()
    var lo = List[Scalar[D]]()
    var hi = List[Scalar[D]]()
    var min_payload = List[Scalar[D]]()
    var max_payload = List[Scalar[D]]()
    var both_payload = List[Scalar[D]]()
    for row in range(7):
        x.append(Scalar[D](source_values[row]))
        lo.append(Scalar[D](1))
        hi.append(Scalar[D](2))
        min_payload.append(Scalar[D](expected_min[row]))
        max_payload.append(Scalar[D](expected_max[row]))
        both_payload.append(Scalar[D](expected_both[row]))
    var source = DataFrame(
        [
            Series(
                "x",
                Column[Scalar[D]](
                    x^, [True, True, True, True, False, True, True]
                ),
            ),
            Series(
                "low",
                Column[Scalar[D]](
                    lo^, [True, True, True, True, True, False, True]
                ),
            ),
            Series(
                "high",
                Column[Scalar[D]](
                    hi^, [True, True, True, True, True, True, False]
                ),
            ),
        ]
    )
    var expected = DataFrame(
        [
            Series(
                "min",
                Column[Scalar[D]](
                    min_payload^, [True, True, True, True, False, False, True]
                ),
            ),
            Series(
                "max",
                Column[Scalar[D]](
                    max_payload^, [True, True, True, True, False, True, False]
                ),
            ),
            Series(
                "both",
                Column[Scalar[D]](
                    both_payload^, [True, True, True, True, False, False, False]
                ),
            ),
        ]
    )
    var expressions: List[Expr] = [
        col("x").clip_min(col("low")).alias("min"),
        col("x").clip_max(col("high")).alias("max"),
        col("x").clip(col("low"), col("high")).alias("both"),
    ]
    for layout in range(3):
        var frame = _layout(source, layout)
        assert_frame(frame.select_exprs(expressions), expected)
        assert_frame(frame.clear().select_exprs(expressions), expected.clear())


def test_numeric_clip_propagates_only_bounds_used_by_each_operation() raises:
    comptime for index in range(len(INTEGER_DTYPES)):
        _nullable_clip[INTEGER_DTYPES[index]]()
    comptime for index in range(len(FLOAT_DTYPES)):
        _nullable_clip[FLOAT_DTYPES[index]]()


def _infinite_divisor_modulo[D: DType]() raises:
    var inf = Float64(1) / Float64(0)
    var numbers: List[Float64] = [2.0, -2.0, 0.0, -0.0]
    for divisor in [inf, -inf]:
        for mode in range(3):
            var av = List[Bool](length=4, fill=mode != 2)
            var bv = List[Bool](length=4, fill=mode != 2)
            if mode == 1:
                av[0] = False
                bv[1] = False
            var a = List[Scalar[D]]()
            var b = List[Scalar[D]]()
            for number in numbers:
                a.append(number.cast[D]())
                b.append(divisor.cast[D]())
            var input = DataFrame(
                [
                    Series("a", Column[Scalar[D]](a^, av.copy())),
                    Series("b", Column[Scalar[D]](b^, bv.copy())),
                ]
            )
            for layout in range(3):
                var frame = _layout(input, layout)
                var expressions: List[Expr] = [
                    (col("a") % col("b")).alias("both"),
                    (col("a") % lit(divisor.cast[D]())).alias("right"),
                ]
                for expression in expressions:
                    var actual = frame.select(expression)
                    assert_true(
                        actual.column(expression._name).dtype()
                        == DataType.of(D)
                    )
                    assert_equal(actual.height(), 4)
                    for row in range(4):
                        var valid = av[row] and (
                            bv[row] if expression._name == "both" else True
                        )
                        assert_equal(
                            actual.item(row, expression._name).is_null(),
                            not valid,
                        )
                        if valid:
                            var expected = infinite_divisor_remainder(
                                numbers[row], divisor
                            )
                            var value = Float64(
                                actual.item(row, expression._name).numeric[D]()
                            )
                            assert_true(
                                not isnan(value),
                                "finite modulo infinite divisor must follow Python remainder, not NaN",
                            )
                            assert_float(value, expected)
                for number in numbers:
                    var actual = frame.select(
                        (lit(number.cast[D]()) % col("b")).alias("left")
                    )
                    for row in range(4):
                        assert_equal(
                            actual.item(row, "left").is_null(), not bv[row]
                        )
                        if bv[row]:
                            var value = Float64(
                                actual.item(row, "left").numeric[D]()
                            )
                            assert_true(
                                not isnan(value),
                                "finite modulo infinite divisor must follow Python remainder, not NaN",
                            )
                            assert_float(
                                value,
                                infinite_divisor_remainder(number, divisor),
                            )
                assert_equal(
                    frame.clear().select_exprs(expressions).height(), 0
                )


def test_float32_modulo_infinite_divisor_matches_selected_python_policy() raises:
    _infinite_divisor_modulo[DType.float32]()


def test_float64_modulo_infinite_divisor_matches_selected_python_policy() raises:
    _infinite_divisor_modulo[DType.float64]()


def _transcendental_exceptional_values[D: DType]() raises:
    var nan = Float64(0) / Float64(0)
    var inf = Float64(1) / Float64(0)
    var upper = Float64(88.72283172607422) if D == DType.float32 else Float64(
        709.782712893384
    )
    var lower = Float64(-103.97207641601562) if D == DType.float32 else Float64(
        -745.1332191019411
    )
    var values: List[Float64] = [
        nan,
        inf,
        -inf,
        -0.0,
        0,
        1,
        -1,
        upper,
        lower,
        1,
        nan,
    ]
    var input = List[Scalar[D]]()
    for value in values:
        input.append(value.cast[D]())
    var frame = DataFrame(
        [
            Series(
                "x",
                Column[Scalar[D]](
                    input^,
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
                    ],
                ),
            )
        ]
    )
    for streaming in [False, True]:
        var actual = (
            frame.lazy()
            .select_exprs(
                [col("x").exp().alias("e"), col("x").log().alias("l")]
            )
            .collect(engine="cpu", streaming=streaming)
        )
        var e = actual.column("e").numeric[D]()
        var l = actual.column("l").numeric[D]()
        assert_true(isnan(e._get(0)))
        assert_true(isnan(l._get(0)))
        assert_true(isinf(e._get(1)) and e._get(1) > 0)
        assert_true(isinf(l._get(1)) and l._get(1) > 0)
        assert_equal(e._get(2), Scalar[D](0))
        assert_true(isnan(l._get(2)))
        assert_equal(e._get(3), Scalar[D](1))
        assert_equal(e._get(4), Scalar[D](1))
        assert_true(isinf(l._get(3)) and l._get(3) < 0)
        assert_true(isinf(l._get(4)) and l._get(4) < 0)
        assert_equal(l._get(5), Scalar[D](0))
        assert_true(isnan(l._get(6)))
        assert_true(not isinf(e._get(7)) and not isnan(e._get(7)))
        assert_true(
            e._get(7)
            > Scalar[D](3.4027e38 if D == DType.float32 else 1.7976e308)
        )
        comptime if D == DType.float32:
            assert_equal(bitcast[DType.uint32](e._get(8)), UInt32(1))
        else:
            assert_equal(bitcast[DType.uint64](e._get(8)), UInt64(1))
        assert_true(not e.is_valid(10) and not l.is_valid(10))
        var ordinary = (
            DataFrame([Series("x", Column[Scalar[D]]([10, 0.5, 2]))])
            .lazy()
            .select(col("x").log())
            .collect(engine="cpu", streaming=streaming)
        )
        var anchors: List[Float64] = [
            2.302585092994046,
            -0.6931471805599453,
            0.6931471805599453,
        ]
        for row in range(3):
            _close(
                Float64(ordinary.column("x").numeric[D]()._get(row)),
                anchors[row],
                Float64(1e-5) if D == DType.float32 else Float64(5e-12),
            )


def test_transcendental_nan_infinity_and_boundary_classifications() raises:
    _transcendental_exceptional_values[DType.float32]()
    _transcendental_exceptional_values[DType.float64]()
