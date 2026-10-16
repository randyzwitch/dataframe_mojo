"""Unexecuted diagnostic: original streaming morsels versus scalar ANY scopes.

This is an evidence probe, not a passing parity claim. It prints every case
before failing on a CPU/GPU discrepancy, so batch geometry remains visible.
"""
from std.testing import assert_equal, assert_true
from dataframe.metal import MetalRuntime
from tests.tooling.iana_scalar_any_proofs import _frame, _expression


def main() raises:
    var runtime = MetalRuntime()
    var mismatches = 0
    for batch in List[Int]([4096, 8191, 8192, 8193, 16384]):
        # The pairs below/above 8192 share an eager Frame evaluation batch,
        # but straddle the requested streaming morsel when batch != 8192.
        var gap_row = (
            batch - 1 if batch < 8192 else 8192 if batch == 8193 else 0
        )
        var active_row = gap_row + 1
        for all_false in [False, True]:
            var source = _frame(16386, gap_row, -1 if all_false else active_row)
            var query = (
                source.lazy().with_columns(_expression(False)).sort("out")
            )
            var eager = String("")
            try:
                _ = query.collect(engine="cpu", streaming=False, optimize=False)
            except error:
                eager = String(error)
            if all_false:
                assert_equal(eager, "")
            else:
                assert_true("non-existent" in eager)
            var cpu = String("")
            var gpu = String("")
            try:
                _ = query.collect(
                    engine="cpu",
                    streaming=True,
                    batch_size=batch,
                    optimize=False,
                )
            except error:
                cpu = String(error)
            try:
                _ = query.collect(
                    accelerator=runtime,
                    streaming=True,
                    batch_size=batch,
                    optimize=False,
                )
            except error:
                gpu = String(error)
            print(
                "mode6 batch="
                + String(batch)
                + " gap="
                + String(gap_row)
                + " active="
                + String(-1 if all_false else active_row)
                + " eager=["
                + eager
                + "] CPU=["
                + cpu
                + "] GPU=["
                + gpu
                + "]",
                flush=True,
            )
            mismatches += Int(cpu != gpu)
    assert_equal(mismatches, 0)
