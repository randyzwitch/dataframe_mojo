"""Reduction kernels agree with a scalar accumulator across layouts and tails."""
# contract-group: reduce.global
# contract-group: kernel.reduce
from std.testing import assert_equal, assert_true
from dataframe import Expr, col
from tests.support.coverage_generators import frame_from_model, chunk_layout
from tests.support.reference_model import generated_model


def test_integer_reduction_matches_scalar_reference() raises:
    for n in [0, 1, 7, 8, 9, 63, 64, 65, 257]:
        for offset in [0, 1, 7]:
            for mode in range(3):
                var model = generated_model(n, 3, mode)
                var contiguous = frame_from_model(model, offset)
                for layout in range(2):
                    var frame = (
                        contiguous.copy() if layout
                        == 0 else chunk_layout(contiguous)
                    )
                    var total = model.int_sum()
                    var count = model.int_count()
                    var expressions: List[Expr] = [
                        col("i").sum().alias("sum"),
                        col("i").count().alias("count"),
                    ]
                    var out = frame.select_exprs(expressions)
                    assert_equal(out.item(0, "sum").int64(), Int64(total))
                    assert_equal(out.item(0, "count").int64(), count)
                    assert_true(
                        frame.select(col("i").sum(min_count=Int(count) + 1))
                        .item()
                        .is_null()
                    )
