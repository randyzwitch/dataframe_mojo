"""Functions without device kernels stay on the CPU."""
# contract-group: planner.routing
from std.testing import assert_equal, assert_true
from dataframe import DataFrame, Series, Column, col


def test_math_functions_product_and_cum_prod_stay_on_the_cpu() raises:
    # No device kernels exist for these (#791), so engine="auto" runs them
    # on the CPU.
    from dataframe import Expr, arctan2

    var frame = DataFrame(
        [
            Series("k", Column[Int32]([1, 2, 1])),
            Series("x", Column[Float32]([0.5, 2, 4])),
        ]
    )
    var expressions: List[Expr] = [
        col("x").log10(),
        col("x").log(3.0),
        col("x").log1p(),
        col("x").sign(),
        col("x").cbrt(),
        col("x").truncate(),
        col("x").sin(),
        col("x").cot(),
        col("x").arctanh(),
        col("x").degrees(),
        arctan2(col("x"), col("x")),
        col("x").product(),
        col("x").cum_prod(),
    ]
    for expression in expressions:
        var query = frame.lazy().select(expression)
        assert_true(
            query.collect(engine="auto").equals(query.collect(engine="cpu"))
        )
    var automatic = (
        frame.lazy().select(col("x").cum_prod()).collect(engine="auto")
    )
    assert_equal(automatic.item(2, "x").numeric[DType.float32](), 4)
