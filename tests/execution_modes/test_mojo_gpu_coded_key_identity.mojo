"""Metal groups a coded key by label, as the CPU does (#943)."""
# contract-group: execution.backends
from std.testing import assert_equal
from dataframe import DataFrame, Expr, col
from dataframe.categorical import decode
from tests.execution_modes.test_backend_metal import require_metal
from tests.public_contracts.test_coded_key_identity import (
    _categorical,
    _coded,
    _frame,
    _summary,
)


def _same_as_cpu(frame: DataFrame, aggregates: List[Expr]) raises:
    for ordered in [False, True]:
        var query = (
            frame.lazy()
            .group_by("k", maintain_order=ordered)
            .agg(aggregates.copy())
        )
        var cpu = _summary(query.collect(engine="cpu", optimize=False))
        assert_equal(len(cpu), 3)
        var metal = _summary(query.collect(accelerator=require_metal()))
        assert_equal(metal, cpu)


# requires: metal
def test_mojo_gpu_two_codes_one_label_group_as_on_the_cpu() raises:
    var codes: List[UInt32] = [0, 1, 0, 1, 2, 0, 0, 1]
    var valid: List[Bool] = [True, True, True, True, True, False, False, True]
    var x: List[Int64] = [1, 2, 3, 4, 5, 6, 7, 8]
    var key = _categorical(codes, valid)
    var sum = col("x").sum().alias("s")
    var both: List[Expr] = [sum.copy(), col("x").max().alias("m")]
    for input in [key.copy(), _coded(key), decode(key)]:
        var frame = _frame(input, x)
        # The coded sum fast path, then the general grouped route.
        _same_as_cpu(frame, [sum.copy()])
        _same_as_cpu(frame, both)
