"""The packed-bit comparison and Kleene kernels (#327) against the scalar
reference kernels they replaced, on random inputs with nulls, NaN, one-row
(broadcast) operands and windows whose first row is not on a byte
boundary."""
from std.testing import TestSuite, assert_equal, assert_true

from dataframe import Column, DataType, Series
from dataframe.bool_column import BoolColumn
from dataframe.expr import AND, EQ, GE, GT, LE, LT, NE, OR, XOR, NOT
from dataframe.expr_kernels import (
    _compare,
    _compare_bits,
    _logical,
    _logical_bits,
    _not_bits,
)


def same(got: Series, want: Series, label: String) raises:
    """Equal validity everywhere and equal values where valid."""
    ref a = got._data[BoolColumn]
    ref b = want._data[BoolColumn]
    assert_equal(len(a), len(b), label)
    for i in range(len(a)):
        assert_equal(
            a._valid(i), b._valid(i), label + " validity row " + String(i)
        )
        if b._valid(i):
            assert_equal(
                a._get(i), b._get(i), label + " value row " + String(i)
            )
        else:
            assert_true(
                not a._get(i), label + " null value bit row " + String(i)
            )


struct Rng:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int(self.state >> 33)


def floats(mut rng: Rng, n: Int) -> Column[Float64]:
    var values = List[Float64](capacity=n)
    var valid = List[Bool](capacity=n)
    for _ in range(n):
        var r = rng.next() % 20
        var v = Float64(rng.next() % 7) - 3
        if r == 0:
            v = Float64(0) / Float64(0)
        elif r == 1:
            v = -0.0
        values.append(v)
        valid.append(rng.next() % 6 != 0)
    try:
        return Column[Float64](values^, valid^)
    except:
        return Column[Float64](List[Float64]())


def ints(mut rng: Rng, n: Int, nulls: Bool) -> Column[Int64]:
    var values = List[Int64](capacity=n)
    var valid = List[Bool](capacity=n)
    for _ in range(n):
        values.append(Int64(rng.next() % 9) - 4)
        valid.append(not nulls or rng.next() % 5 != 0)
    try:
        return Column[Int64](values^, valid^)
    except:
        return Column[Int64](List[Int64]())


def check_compare[
    op: Int, D: DType
](left: Column[Scalar[D]], right: Column[Scalar[D]], label: String) raises:
    same(
        _compare_bits[op, D](left, right),
        _compare[op, Scalar[D]](left, right),
        label,
    )


def check_all[
    D: DType
](left: Column[Scalar[D]], right: Column[Scalar[D]], label: String) raises:
    check_compare[GT, D](left, right, label + " gt")
    check_compare[LT, D](left, right, label + " lt")
    check_compare[GE, D](left, right, label + " ge")
    check_compare[LE, D](left, right, label + " le")
    check_compare[EQ, D](left, right, label + " eq")
    check_compare[NE, D](left, right, label + " ne")


def test_comparisons_match_the_scalar_kernel() raises:
    var rng = Rng(7)
    for n in [0, 1, 7, 8, 9, 63, 64, 65, 203]:
        var a = floats(rng, n + 13)
        var b = floats(rng, n + 13)
        # Windows starting at rows 5 and 11: neither is byte-aligned.
        var x = a.slice(5, n)
        var y = b.slice(11, n)
        check_all[DType.float64](x, y, "float n=" + String(n))
        var i = ints(rng, n + 13, True)
        var j = ints(rng, n + 13, n % 2 == 0)
        check_all[DType.int64](
            i.slice(3, n), j.slice(9, n), "int n=" + String(n)
        )
        if n > 0:
            # One-row operands broadcast, including a null one.
            check_all[DType.float64](x, b.slice(2, 1), "float scalar right")
            check_all[DType.float64](a.slice(4, 1), y, "float scalar left")
            check_all[DType.int64](i.slice(3, n), j.slice(1, 1), "int scalar")


def bools(mut rng: Rng, n: Int) -> BoolColumn:
    var values = List[Bool](capacity=n)
    var valid = List[Bool](capacity=n)
    for _ in range(n):
        values.append(rng.next() % 2 == 0)
        valid.append(rng.next() % 4 != 0)
    try:
        return BoolColumn(values^, valid^)
    except:
        return BoolColumn(List[Bool]())


def test_kleene_logic_matches_the_scalar_kernel() raises:
    var rng = Rng(11)
    for n in [0, 1, 5, 8, 9, 64, 71, 200]:
        var a = bools(rng, n + 10).slice(3, n)
        var b = bools(rng, n + 10).slice(6, n)
        same(_logical_bits[AND](a, b), _logical[AND](a, b), "and")
        same(_logical_bits[OR](a, b), _logical[OR](a, b), "or")
        same(_logical_bits[XOR](a, b), _logical[XOR](a, b), "xor")
        var expected = List[Bool](capacity=n)
        var valid = List[Bool](capacity=n)
        for r in range(n):
            valid.append(a._valid(r))
            expected.append(a._valid(r) and not a._get(r))
        same(
            _not_bits(a),
            Series("", BoolColumn(expected^, valid^)),
            "not",
        )
        if n > 0:
            var one = bools(rng, 4).slice(2, 1)
            same(_logical_bits[AND](a, one), _logical[AND](a, one), "and one")
            same(_logical_bits[OR](one, b), _logical[OR](one, b), "or one")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
