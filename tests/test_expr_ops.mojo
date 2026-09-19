"""Arithmetic, comparison, and math operators against scalar references."""
from std.math import sqrt, exp, log, pow, isnan
from std.testing import (
    TestSuite,
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
    assert_almost_equal,
)
from dataframe import Column, DataFrame, Expr, Series, col, lit
from dataframe.binding import bind
from dataframe.execution import evaluate

comptime MIN = Int64(-9223372036854775807) - 1
comptime MAX = Int64(9223372036854775807)


def nan() -> Float64:
    return Float64(0) / Float64(0)


def inf() -> Float64:
    return Float64(1) / Float64(0)


def ints() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "a",
                Column[Int64](
                    [7, -7, 7, -7, 0, MIN, MAX, 5, 9],
                    [True, True, True, True, True, True, True, False, True],
                ),
            ),
            Series(
                "b",
                Column[Int64](
                    [2, 2, -2, -2, 3, 1, 1, 1, 0],
                    [True, True, True, True, True, True, True, True, True],
                ),
            ),
        ]
    )


def floats() raises -> DataFrame:
    return DataFrame(
        [
            Series(
                "x",
                Column[Float64](
                    [7.5, -7.5, 0.0, -0.0, nan(), inf(), -inf(), 2.0, 1.0],
                    [True, True, True, True, True, True, True, True, False],
                ),
            ),
            Series(
                "y",
                Column[Float64](
                    [2.0, 2.0, 0.0, 1.0, 1.0, 2.0, 2.0, nan(), 1.0],
                    [True, True, True, True, True, True, True, True, True],
                ),
            ),
        ]
    )


def run[
    width: Int = 4
](frame: DataFrame, expr: Expr, batch_size: Int = 3) raises -> Series:
    return evaluate[width](
        bind(expr, frame._columns),
        frame._columns,
        frame.height(),
        batch_size=batch_size,
    )


def test_int_division_modulo_follow_floor_semantics() raises:
    var frame = ints().head(5)
    var fdiv = run(frame, col("a") // col("b")).int64()
    var mod = run(frame, col("a") % col("b")).int64()
    var expected_div: List[Int64] = [3, -4, -4, 3, 0]
    var expected_mod: List[Int64] = [1, 1, -1, -1, 0]
    for i in range(5):
        assert_equal(fdiv.value(i), expected_div[i])
        assert_equal(mod.value(i), expected_mod[i])
        # The floor-division identity holds for every nonzero divisor.
        var a = frame.column("a").int64().value(i)
        var b = frame.column("b").int64().value(i)
        assert_equal(fdiv.value(i) * b + mod.value(i), a)
    var zero = ints().slice(8)
    assert_true(run(zero, col("a") // col("b")).int64().is_null(0))
    assert_true(run(zero, col("a") % col("b")).int64().is_null(0))
    var edge = DataFrame(
        [
            Series("a", Column[Int64]([MIN, MIN])),
            Series("b", Column[Int64]([-1, 2])),
        ]
    )
    assert_equal(run(edge, col("a") % col("b")).int64().value(0), Int64(0))
    with assert_raises(contains="floor division overflow"):
        _ = run(edge, col("a") // col("b"))
    var truediv = run(frame, col("a") / col("b")).float64()
    assert_equal(run(frame, col("a") / col("b")).dtype(), "float64")
    assert_equal(truediv.value(0), Float64(3.5))
    assert_equal(truediv.value(1), Float64(-3.5))
    assert_equal(truediv.value(4), Float64(0))
    var by_zero = run(zero, col("a") / col("b")).float64()
    assert_equal(by_zero.value(0), inf())
    assert_true(run(ints(), col("a") / col("b")).float64().is_null(7))


def test_float_division_modulo_and_special_values() raises:
    var frame = floats()
    var div = run(frame, col("x") / col("y")).float64()
    var fdiv = run(frame, col("x") // col("y")).float64()
    var mod = run(frame, col("x") % col("y")).float64()
    assert_equal(div.value(0), Float64(3.75))
    assert_equal(fdiv.value(0), Float64(3))
    assert_equal(fdiv.value(1), Float64(-4))
    assert_equal(mod.value(0), Float64(1.5))
    assert_equal(mod.value(1), Float64(0.5))
    assert_true(isnan(div.value(2)))
    assert_true(isnan(mod.value(2)))
    assert_true(isnan(fdiv.value(4)))
    assert_true(isnan(mod.value(5)))
    assert_equal(div.value(5), inf())
    assert_equal(div.value(6), -inf())
    assert_true(isnan(div.value(7)))
    assert_true(div.is_null(8))
    assert_true(fdiv.is_null(8))
    assert_true(mod.is_null(8))


def test_pow() raises:
    var frame = DataFrame(
        [
            Series("a", Column[Int64]([2, -3, 0, 5, 10])),
            Series("b", Column[Int64]([10, 3, 0, 0, 18])),
        ]
    )
    var result = run(frame, col("a") ** col("b")).int64()
    var expected: List[Int64] = [1024, -27, 1, 1, 1000000000000000000]
    for i in range(5):
        assert_equal(result.value(i), expected[i])
    with assert_raises(contains="multiplication overflow"):
        _ = run(frame, col("a").pow(lit(Int64(64))))
    with assert_raises(contains="nonnegative exponent"):
        _ = run(frame, col("a") ** lit(Int64(-1)))
    assert_equal(
        run(frame, lit(Int64(-2)) ** lit(Int64(63))).int64().value(0), MIN
    )
    var f = floats()
    var powered = run(f, col("x") ** lit(Float64(2))).float64()
    assert_almost_equal(powered.value(0), 56.25, atol=1e-9)
    assert_almost_equal(
        run(f, lit(Float64(2)) ** lit(Float64(0.5))).float64().value(0),
        sqrt(Float64(2)),
        rtol=1e-12,
    )
    assert_true(powered.is_null(8))


def test_comparisons_all_dtypes() raises:
    var ops: List[String] = ["lt", "le", "ge", "gt", "eq", "ne"]
    var frame = floats()
    for op in ops:
        var e: Expr
        if op == "lt":
            e = col("x") < col("y")
        elif op == "le":
            e = col("x") <= col("y")
        elif op == "ge":
            e = col("x") >= col("y")
        elif op == "gt":
            e = col("x") > col("y")
        elif op == "eq":
            e = col("x").eq(col("y"))
        else:
            e = col("x").ne(col("y"))
        var result = run(frame, e).bool()
        for i in range(8):
            var x = frame.column("x").float64().value(i)
            var y = frame.column("y").float64().value(i)
            var expected: Bool
            if op == "lt":
                expected = x < y
            elif op == "le":
                expected = x <= y
            elif op == "ge":
                expected = x >= y
            elif op == "gt":
                expected = x > y
            elif op == "eq":
                expected = x == y
            else:
                expected = x != y
            assert_equal(
                result.value(i), expected, msg=op + " row " + String(i)
            )
        assert_true(result.is_null(8))
    var mixed = DataFrame(
        [
            Series(
                "s",
                Column[String](
                    ["apple", "b", "", "é"], [True, True, True, False]
                ),
            ),
            Series("t", Column[String](["banana", "b", "a", "e"])),
            Series("p", Column[Bool]([False, True, True, False])),
            Series("q", Column[Bool]([True, True, False, False])),
        ]
    )
    var lt = run(mixed, col("s") < col("t")).bool()
    assert_true(lt.value(0))
    assert_false(lt.value(1))
    assert_true(lt.value(2))
    assert_true(lt.is_null(3))
    assert_true(run(mixed, col("s") >= col("t")).bool().value(1))
    assert_true(run(mixed, col("s").ne(col("t"))).bool().value(0))
    var blt = run(mixed, col("p") < col("q")).bool()
    assert_true(blt.value(0))
    assert_false(blt.value(1))
    assert_false(blt.value(2))
    assert_true(run(mixed, col("p") > col("q")).bool().value(2))
    assert_true(run(mixed, col("p") <= col("q")).bool().value(3))
    var ints_cmp = run(ints(), col("a") <= col("b")).bool()
    assert_false(ints_cmp.value(0))
    assert_true(ints_cmp.value(1))
    assert_true(ints_cmp.value(5))
    assert_true(ints_cmp.is_null(7))


def test_unary_math() raises:
    var frame = floats()
    var neg = run(frame, -col("x")).float64()
    assert_equal(neg.value(0), Float64(-7.5))
    assert_equal(neg.value(5), -inf())
    var absolute = run(frame, col("x").abs()).float64()
    assert_equal(absolute.value(1), Float64(7.5))
    assert_equal(absolute.value(6), inf())
    assert_true(absolute.is_null(8))
    var roots = run(frame, col("y").sqrt()).float64()
    assert_almost_equal(roots.value(0), sqrt(Float64(2)), rtol=1e-12)
    assert_true(isnan(run(frame, col("x").sqrt()).float64().value(1)))
    var exps = run(frame, col("y").exp()).float64()
    assert_almost_equal(exps.value(3), exp(Float64(1)), rtol=1e-12)
    var logs = run(frame, col("y").log()).float64()
    assert_almost_equal(logs.value(0), log(Float64(2)), rtol=1e-12)
    assert_equal(run(frame, col("x").log()).float64().value(2), -inf())
    var floors = run(frame, col("x").floor()).float64()
    var ceils = run(frame, col("x").ceil()).float64()
    assert_equal(floors.value(0), Float64(7))
    assert_equal(floors.value(1), Float64(-8))
    assert_equal(ceils.value(0), Float64(8))
    assert_equal(ceils.value(1), Float64(-7))
    assert_true(isnan(floors.value(4)))
    var i = ints()
    assert_equal(run(i.slice(6), -col("a")).int64().value(0), -MAX)
    assert_equal(run(i.head(5), col("a").abs()).int64().value(1), Int64(7))
    with assert_raises(contains="negation overflow"):
        _ = run(i, -col("a"))
    with assert_raises(contains="abs overflow"):
        _ = run(i, col("a").abs())
    var int_roots = run(i.head(1), col("b").sqrt())
    assert_equal(int_roots.dtype(), "float64")
    assert_equal(run(i, col("a").floor()).int64().value(5), MIN)
    assert_equal(run(i, col("a").round(2)).int64().value(0), Int64(7))


def test_round_half_away_from_zero() raises:
    var frame = DataFrame(
        [
            Series(
                "v",
                Column[Float64](
                    [2.5, -2.5, 0.125, 1234.5678, 1e300, -0.4, 3.0, 12.5]
                ),
            )
        ]
    )
    var zero = run(frame, col("v").round()).float64()
    assert_equal(zero.value(0), Float64(3))
    assert_equal(zero.value(1), Float64(-3))
    assert_equal(zero.value(4), Float64(1e300))
    assert_equal(zero.value(5), Float64(-0.0))
    var two = run(frame, col("v").round(2)).float64()
    assert_almost_equal(two.value(2), 0.13, atol=1e-12)
    assert_almost_equal(two.value(3), 1234.57, atol=1e-9)
    var negative = run(frame, col("v").round(-1)).float64()
    assert_equal(negative.value(3), Float64(1230))
    assert_equal(negative.value(7), Float64(10))


def test_clip() raises:
    var frame = floats()
    var clipped = run(
        frame, col("x").clip(lit(Float64(-1)), lit(Float64(1)))
    ).float64()
    assert_equal(clipped.value(0), Float64(1))
    assert_equal(clipped.value(1), Float64(-1))
    assert_equal(clipped.value(3), Float64(-0.0))
    assert_true(isnan(clipped.value(4)))
    assert_equal(clipped.value(5), Float64(1))
    assert_equal(clipped.value(6), Float64(-1))
    assert_true(clipped.is_null(8))
    var i = ints()
    var low = run(i, col("a").clip_min(lit(Int64(0)))).int64()
    assert_equal(low.value(1), Int64(0))
    assert_equal(low.value(6), MAX)
    var high = run(i, col("a").clip_max(col("b"))).int64()
    assert_equal(high.value(0), Int64(2))
    assert_equal(high.value(1), Int64(-7))
    assert_true(high.is_null(7))


def test_bind_errors_name_operator_and_dtypes() raises:
    var frame = DataFrame(
        [
            Series("s", Column[String](["a"])),
            Series("i", Column[Int64]([1])),
            Series("f", Column[Float64]([1])),
            Series("b", Column[Bool]([True])),
        ]
    )
    with assert_raises(
        contains="/ requires matching dtypes, found int64 and float64"
    ):
        _ = frame.select(col("i") / col("f"))
    with assert_raises(contains="// requires numeric operands, found string"):
        _ = frame.select(col("s") // col("s"))
    with assert_raises(contains="sqrt requires a numeric operand, found bool"):
        _ = frame.select(col("b").sqrt())
    with assert_raises(contains="negation requires a numeric operand"):
        _ = frame.select(-col("s"))
    with assert_raises(contains="comparison requires matching dtypes"):
        _ = frame.select(col("s") < lit(Int64(1)))
    # Validation happens before execution, even on empty frames.
    with assert_raises():
        _ = frame.clear().select(col("s") % col("s"))


def test_scalar_broadcast_and_simd_width_agreement() raises:
    var values = List[Float64]()
    var valid = List[Bool]()
    for i in range(37):
        values.append(Float64(i) * 0.75 - 9)
        valid.append(i % 5 != 2)
    var frame = DataFrame([Series("x", Column[Float64](values^, valid^))])
    var e = (lit(Float64(100)) / (col("x").abs() + lit(Float64(1)))).clip(
        lit(Float64(2)), lit(Float64(50))
    ).round(3) - (col("x") // lit(Float64(4))) % lit(Float64(3))
    var reference = run[1](frame, e, 1)
    var sizes: List[Int] = [1, 3, 4, 7, 8, 64]
    for size in sizes:
        assert_true(run[1](frame, e, size).equals(reference))
        assert_true(run[4](frame, e, size).equals(reference))
        assert_true(run[8](frame, e, size).equals(reference))
    var comparisons = col("x") >= lit(Float64(0))
    var ref_cmp = run[1](frame, comparisons, 1)
    assert_true(run[8](frame, comparisons, 5).equals(ref_cmp))
    var empty = frame.clear()
    assert_equal(len(run(empty, e)), 0)
    var scalar = DataFrame([], height=4).select(
        (lit(Int64(7)) // lit(Int64(2))).alias("q")
    )
    assert_equal(scalar.height(), 1)
    assert_equal(scalar.column("q").int64().value(0), Int64(3))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
