"""Fused Float64 kernels match the unfused evaluator exactly."""
from std.testing import TestSuite, assert_equal, assert_true, assert_false
from dataframe import (
    DataType,
    Column,
    DataFrame,
    Expr,
    Series,
    col,
    lit,
    concat,
)
from dataframe.binding import bind
from dataframe.execution import evaluate


def nan() -> Float64:
    return Float64(0) / Float64(0)


def inf() -> Float64:
    return Float64(1) / Float64(0)


def frame(rows: Int) raises -> DataFrame:
    var specials: List[Float64] = [
        0.0,
        -0.0,
        1.5,
        -2.25,
        nan(),
        inf(),
        -inf(),
        1e300,
        3.0,
    ]
    var x = List[Float64]()
    var y = List[Float64]()
    var xv = List[Bool]()
    var yv = List[Bool]()
    for i in range(rows):
        x.append(specials[i % len(specials)])
        y.append(specials[(i * 5 + 3) % len(specials)])
        xv.append(i % 4 != 1)
        yv.append(i % 7 != 2)
    return DataFrame(
        [
            Series("x", Column[Float64](x^, xv^)),
            Series("y", Column[Float64](y^, yv^)),
            Series("i", Column[Int64](List[Int64](length=rows, fill=3))),
        ]
    )


def run[
    width: Int
](df: DataFrame, expr: Expr, fuse: Bool, batch: Int) raises -> Series:
    return evaluate[width](
        bind(expr, df._columns, fuse),
        df._columns,
        df.height(),
        batch_size=batch,
    )


def bitwise_equal(a: Series, b: Series) raises -> Bool:
    """Structural equality plus identical float bit patterns (so -0.0 and
    0.0 are distinguished)."""
    if not a.equals(b):
        return False
    if a.dtype() == DataType.FLOAT64:
        var x = a.float64()
        var y = b.float64()
        for i in range(len(x)):
            if x._valid(i):
                var p = x._get(i)
                var q = y._get(i)
                if p == 0 and q == 0 and String(p) != String(q):
                    return False
    return True


def test_fused_matches_unfused() raises:
    var exprs: List[Expr] = [
        (col("x") + lit(Float64(3)))
        * (col("y") - lit(Float64(2)))
        / lit(Float64(4)),
        col("x") * col("x") - col("y"),
        col("x") / col("y"),
        lit(Float64(-0.0)) * col("x"),
        (col("x") + col("y")) > lit(Float64(1)),
        col("x").ne(col("y")),
        col("x").eq(col("x")),
        (col("x") * lit(Float64(2))) <= col("y"),
        lit(Float64(1)) + lit(Float64(2)),
        ((col("x") + lit(Float64(1))) * lit(Float64(2))).sum(),
        (col("x") - col("y")).abs() + col("y"),
        col("i") + lit(Int64(1)),
    ]
    for rows in [0, 1, 3, 5, 9, 37]:
        var df = frame(rows)
        for e in exprs:
            for batch in [1, 4, 64]:
                var reference = run[4](df, e, False, batch)
                assert_true(
                    bitwise_equal(run[4](df, e, True, batch), reference)
                )
                assert_true(
                    bitwise_equal(run[1](df, e, True, batch), reference)
                )
                assert_true(
                    bitwise_equal(run[8](df, e, True, batch), reference)
                )


def test_fused_chunked_inputs_match_contiguous() raises:
    var pieces = List[DataFrame]()
    for _ in range(20):
        pieces.append(frame(25))
    var chunked = concat(pieces)
    var contiguous = chunked.rechunk()
    for expr in [
        (col("x") + lit(Float64(3))) * (col("y") - lit(Float64(2))),
        col("x") > col("y"),
        (col("x") + col("y")).sum(),
    ]:
        for batch in [1, 64, 1024]:
            var expected = run[4](contiguous, expr, True, batch)
            assert_true(
                bitwise_equal(run[4](chunked, expr, True, batch), expected)
            )


def test_fused_packed_validity_handles_bit_offsets_and_all_valid_sources() raises:
    var values = List[Float64]()
    var valid = List[Bool]()
    for i in range(40):
        values.append(Float64(i) - 20.0)
        valid.append(i % 5 != 0)
    var x = Column[Float64](values.copy(), valid^).slice(3, 29)
    var y = Column[Float64](values=values^, bits=List[UInt8]()).slice(2, 29)
    var df = DataFrame([Series("x", x^), Series("y", y^)])
    var expr = (col("x") + col("y")) > lit(Float64(0))
    for batch in [1, 3, 8, 64]:
        var expected = run[4](df, expr, False, batch)
        assert_true(bitwise_equal(run[4](df, expr, True, batch), expected))
        assert_true(bitwise_equal(run[8](df, expr, True, batch), expected))


def test_fusible_analysis() raises:
    var df = frame(4)
    var chain = bind((col("x") + lit(Float64(1))) * col("y"), df._columns)
    for flag in chain.fusible:
        assert_true(flag)
    var mixed = bind((col("x") + lit(Float64(1))).abs(), df._columns)
    assert_true(mixed.fusible[2])
    assert_false(mixed.fusible[3])
    var ints = bind(col("i") + lit(Int64(1)), df._columns)
    for flag in ints.fusible:
        assert_false(flag)
    var off = bind(col("x") + col("y"), df._columns, False)
    for flag in off.fusible:
        assert_false(flag)
    # Null payloads are zero in fused output, as in the unfused kernels.
    var out = run[4](df, col("x") + col("y"), True, 64).float64()
    assert_true(out.is_null(1))
    assert_equal(out._get(1), 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
