"""Arithmetic boundary results and error messages share an Int128 oracle."""
# contract-group: expr.arithmetic
from std.testing import assert_equal
from dataframe.kernels.binary import checked_add, _checked_sub, _checked_mul
from dataframe.time.temporal import (
    checked_add as temporal_add,
    checked_sub as temporal_sub,
    checked_mul as temporal_mul,
)


def test_checked_int64_operations_keep_numeric_and_temporal_error_policies() raises:
    var values: List[Int64] = [
        Int64.MIN,
        Int64.MIN + 1,
        -4611686018427387904,
        -3037000500,
        -3037000499,
        -2,
        -1,
        0,
        1,
        2,
        3037000499,
        3037000500,
        4611686018427387904,
        Int64.MAX - 1,
        Int64.MAX,
    ]
    var messages: List[String] = [
        "Int64 sum overflow",
        "Int64 subtraction overflow",
        "Int64 multiplication overflow",
    ]
    for a in values:
        for b in values:
            for op in range(3):
                var expected = (
                    Int128(a) + Int128(b) if op
                    == 0 else Int128(a) - Int128(b) if op
                    == 1 else Int128(a) * Int128(b)
                )
                var overflow = expected < Int128(
                    Int64.MIN
                ) or expected > Int128(Int64.MAX)
                for temporal in [False, True]:
                    var message = String()
                    var result = Int64(0)
                    try:
                        if temporal:
                            result = (
                                temporal_add(a, b) if op
                                == 0 else temporal_sub(a, b) if op
                                == 1 else temporal_mul(a, b)
                            )
                        else:
                            result = (
                                checked_add(a, b) if op
                                == 0 else _checked_sub(a, b) if op
                                == 1 else _checked_mul(a, b)
                            )
                    except error:
                        message = String(error)
                    if overflow:
                        assert_equal(
                            message,
                            "temporal value overflows Int64" if temporal else messages[
                                op
                            ],
                        )
                    else:
                        assert_equal(message, "")
                        assert_equal(result, Int64(expected))
