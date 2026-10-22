"""Boolean fill and null retention use independent scalar state expectations."""
# contract-group: kernel.row_ops
from std.testing import assert_equal
from dataframe import Column, DataFrame, DataType, Series, col, lit
from dataframe.bool_column import BoolColumn
from dataframe.expr_kernels import (
    _fill_null_bools,
    _keep_nulls_bits,
    _keep_nulls_bools,
)
from tests.support.boolean_word_reference import truth_column, truth_state


def test_boolean_fill_and_numeric_membership_null_retention_match_scalar_states_at_every_bit_alignment() raises:
    for n in [0, 1, 7, 8, 9, 63, 64, 65, 129, 257]:
        for lo in range(8):
            for ro in [0, 3, 7]:
                for lm in range(3):
                    for rm in range(3):
                        for broadcast in [0, 1, 2]:
                            if n == 0 and broadcast != 0:
                                continue
                            var ln = 1 if broadcast == 1 else n
                            var rn = 1 if broadcast == 2 else n
                            var left = truth_column(ln, lo, lm, 0)
                            var right = truth_column(rn, ro, rm, 1)
                            var filled = _fill_null_bools(left, right)
                            var integers = List[Int64]()
                            var strings = List[String]()
                            var valid = List[Bool]()
                            for row in range(lo + ln):
                                integers.append(Int64.MAX)
                                strings.append("unused external string payload")
                                valid.append(truth_state(row, lm, 0) >= 0)
                            var numeric = Series(
                                "number", Column[Int64](integers^, valid.copy())
                            ).slice(lo, ln)
                            var text = Series(
                                "text", Column[String](strings^, valid)
                            ).slice(lo, ln)
                            var numeric_keep = _keep_nulls_bits(numeric, right)
                            # A scalar nonnumeric validity fallback does not
                            # currently broadcast; the public regression below
                            # isolates that defect from every numeric route.
                            var text_keep = (
                                _keep_nulls_bits(text, right) if broadcast
                                != 1 else numeric_keep.copy()
                            )
                            var mask = List[Bool]()
                            for i in range(ln):
                                mask.append(truth_state(lo + i, lm, 0) >= 0)
                            var scalar_keep = _keep_nulls_bools(mask, right)
                            for i in range(n):
                                var a = truth_state(
                                    lo + (0 if broadcast == 1 else i), lm, 0
                                )
                                var b = truth_state(
                                    ro + (0 if broadcast == 2 else i), rm, 1
                                )
                                var expected = a if a >= 0 else b
                                assert_equal(filled._valid(i), expected >= 0)
                                assert_equal(filled._get(i), expected == 1)
                                var present = a >= 0 and b >= 0
                                assert_equal(numeric_keep._valid(i), present)
                                assert_equal(text_keep._valid(i), present)
                                assert_equal(scalar_keep._valid(i), present)
                                assert_equal(
                                    numeric_keep._get(i), present and b == 1
                                )
                                assert_equal(
                                    text_keep._get(i), present and b == 1
                                )
                                if present:
                                    assert_equal(scalar_keep._get(i), b == 1)


def test_scalar_string_membership_broadcast_preserves_nonnull_literal_validity() raises:
    var words: List[String] = ["a", "b", "a", "b", "a", "a", "b"]
    var frame = DataFrame([Series("candidate", Column[String](words.copy()))])
    var actual = frame.select(
        lit("a").is_in([col("candidate")]).alias("member")
    )
    assert_equal(actual.height(), 7)
    assert_equal(actual.column("member").dtype(), DataType.BOOL)
    var missing = 0
    for row in range(7):
        var cell = actual.item(row, "member")
        if cell.is_null():
            missing += 1
        else:
            assert_equal(cell.bool(), words[row] == "a")
    assert_equal(
        missing,
        0,
        "scalar string membership must broadcast valid literal state",
    )
