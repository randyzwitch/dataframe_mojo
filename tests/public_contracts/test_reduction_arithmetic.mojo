"""Reducing an expression cannot bypass its per-row arithmetic contract."""
from std.testing import assert_equal, assert_true, assert_raises
from dataframe import Column, DataFrame, Expr, Series, col, lit
from tests.support.numeric_model import integer_bounds
from tests.support.coverage_generators import offset_layout, chunk_layout
from tests.support.coverage_compare import assert_float


def _expressions[D: DType](constant: Int64) -> List[Expr]:
    var scalar = lit(constant.cast[D]())
    return [
        col("x") + scalar,
        scalar + col("x"),
        col("x") - scalar,
        scalar - col("x"),
        col("x") * scalar,
        scalar * col("x"),
    ]


def _operation(value: Int128, constant: Int128, op: Int) -> Int128:
    return (
        value + constant if op
        < 2 else value - constant if op
        == 2 else constant - value if op
        == 3 else value * constant
    )


def _matrix[D: DType]() raises:
    var bounds = integer_bounds[D]()
    comptime SUM_D = DType.int32 if D == DType.int32 else DType.int64
    var sum_bounds = integer_bounds[SUM_D]()
    var values: List[Int128] = [bounds[0], bounds[1], -3, 0, 3]
    for constant in [-2, -1, 0, 1, 2]:
        for mode in range(3):
            var payload = List[Scalar[D]]()
            var valid = List[Bool]()
            for row in range(len(values)):
                payload.append(values[row].cast[D]())
                valid.append(mode == 0 or mode == 1 and row >= 2)
            var source = DataFrame(
                [Series("x", Column[Scalar[D]](payload^, valid.copy()))]
            )
            for layout in range(3):
                var frame = (
                    source.copy() if layout
                    == 0 else offset_layout(source) if layout
                    == 1 else chunk_layout(source)
                )
                var expressions = _expressions[D](Int64(constant))
                for op in range(len(expressions)):
                    var total = Int128(0)
                    var count = 0
                    var per_row_error = False
                    for row in range(len(values)):
                        if not valid[row]:
                            continue
                        var value = _operation(
                            values[row], Int128(constant), op
                        )
                        per_row_error = (
                            per_row_error
                            or value < bounds[0]
                            or value > bounds[1]
                        )
                        total += value
                        count += 1
                    for streaming in [False, True]:
                        for batch in [1, 7, 64]:
                            var sum_plan = frame.lazy().select(
                                expressions[op].sum()
                            )
                            var mean_plan = frame.lazy().select(
                                expressions[op].mean()
                            )
                            if per_row_error:
                                with assert_raises(contains="overflow"):
                                    _ = sum_plan.collect(
                                        optimize=False,
                                        streaming=streaming,
                                        batch_size=batch,
                                        engine="cpu",
                                    )
                                with assert_raises(contains="overflow"):
                                    _ = mean_plan.collect(
                                        optimize=False,
                                        streaming=streaming,
                                        batch_size=batch,
                                        engine="cpu",
                                    )
                            else:
                                var mean = mean_plan.collect(
                                    optimize=False,
                                    streaming=streaming,
                                    batch_size=batch,
                                    engine="cpu",
                                ).item()
                                assert_equal(mean.is_null(), count == 0)
                                if count:
                                    assert_float(
                                        mean.float64(),
                                        Float64(total) / Float64(count),
                                    )
                                if (
                                    total < sum_bounds[0]
                                    or total > sum_bounds[1]
                                ):
                                    with assert_raises(contains="overflow"):
                                        _ = sum_plan.collect(
                                            optimize=False,
                                            streaming=streaming,
                                            batch_size=batch,
                                            engine="cpu",
                                        )
                                else:
                                    assert_equal(
                                        sum_plan.collect(
                                            optimize=False,
                                            streaming=streaming,
                                            batch_size=batch,
                                            engine="cpu",
                                        )
                                        .item()
                                        .numeric[SUM_D]()
                                        .cast[DType.int128](),
                                        total,
                                    )


def test_reducing_integer_scalar_expressions_preserves_row_errors_nulls_and_wide_finalization() raises:
    _matrix[DType.int8]()
    _matrix[DType.int32]()
    _matrix[DType.int64]()


def test_reduction_arithmetic_has_literal_cancellation_and_empty_identity_anchors() raises:
    var source = DataFrame(
        [Series("x", Column[Int64]([Int64.MAX, -Int64.MAX]))]
    )
    # The exact final sum would be two and its mean one; the first row's
    # checked addition must still raise before either reduction can finalize.
    for reduction in [(col("x") + 1).sum(), (col("x") + 1).mean()]:
        with assert_raises(contains="overflow"):
            _ = source.select(reduction)
    var empty = source.clear()
    assert_equal(empty.select((col("x") + 1).sum()).item().int64(), Int64(0))
    assert_true(empty.select((col("x") + 1).mean()).item().is_null())
