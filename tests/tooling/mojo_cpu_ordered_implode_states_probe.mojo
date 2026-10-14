"""CPU-only literal phase anchors for prospective post-state IMPLODE parity."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataType, Expr, Series, col, when
from tests.execution_modes.test_mojo_gpu_ordered_implode_states import (
    _frame,
    _cast,
    _power,
    _state,
    _time,
)


def main() raises:
    var frame = _frame()
    for cast_first in [False, True]:
        var row = _cast() + _power() if cast_first else _power() + _cast()
        with assert_raises(
            contains="cast from int64 to int8 failed" if cast_first else "nonnegative exponent"
        ):
            _ = (
                frame.lazy()
                .select((_state() + row).implode())
                .collect(engine="cpu", optimize=False, streaming=False)
            )
    var bad_state = col("d").sum().over("k").cast(DataType.INT64)
    for state_first in [False, True]:
        var row = (
            bad_state
            + (_cast() + _power()) if state_first else (_cast() + _power())
            + bad_state
        )
        with assert_raises(contains="decimal precision"):
            _ = (
                frame.lazy()
                .select(row.implode())
                .head(0)
                .collect(engine="cpu", optimize=False, streaming=False)
            )
    var strict_state = col("x").cast(DataType.INT8).sum().over("k")
    for strict_first in [False, True]:
        var states = (
            strict_state
            + bad_state if strict_first else bad_state
            + strict_state
        )
        with assert_raises(
            contains="cast from int64 to int8 failed" if strict_first else "decimal precision"
        ):
            _ = (
                frame.lazy()
                .select((states + (_cast() + _power())).implode())
                .collect(engine="cpu", optimize=False, streaming=False)
            )
    with assert_raises(contains="decimal precision"):
        _ = (
            frame.lazy()
            .select(
                when(col("keep"))
                .then(bad_state + (_cast() + _power()))
                .otherwise(col("v"))
                .implode()
            )
            .collect(engine="cpu", optimize=False, streaming=False)
        )
    var first = (_state() + (_cast() + _power())).implode().alias("first")
    var later = bad_state.implode().alias("later")
    for later_first in [False, True]:
        var fields = List[Expr]()
        fields.append(later.copy() if later_first else first.copy())
        fields.append(first.copy() if later_first else later.copy())
        with assert_raises(
            contains="decimal precision" if later_first else "cast from int64 to int8 failed"
        ):
            _ = (
                frame.lazy()
                .select_exprs(fields^)
                .head(0)
                .collect(engine="cpu", optimize=False, streaming=False)
            )
    frame = frame.with_column(Series("x", Column[Int64]([1, 2]))).with_column(
        Series("t", Column[Int64]([-1, 1])).with_dtype(DataType.duration("ms"))
    )
    var state = (col("v").sum() + col("v").min()).over("k")
    var expression = (state + (_cast() + _time())).implode()
    var expected = Series("item", Column[Int64]([-999989, 1000012]))
    assert_true(
        frame.lazy()
        .select(expression.copy())
        .collect(engine="cpu", optimize=False, streaming=False)
        .item()
        .list()
        .equals(expected)
    )
    assert_equal(
        len(
            frame.clear()
            .lazy()
            .select(expression.copy())
            .collect(engine="cpu", optimize=False, streaming=False)
            .item()
            .list()
        ),
        0,
    )
    print(
        "PASS CPU post-state AST/state/projection order, eager WHEN states, composed state literals and empty list"
    )
