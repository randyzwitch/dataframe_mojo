"""Metal returns the CPU's outcome for a query, including the CPU's error."""
from std.testing import assert_raises, assert_true
from dataframe import DataFrame
from dataframe.lazy import LazyFrame
from dataframe_metal.metal import MetalRuntime


def same_outcome(query: LazyFrame) raises:
    """The forced-Metal result equals CPU's, or both raise CPU's error.

    Metal decides a data fault on the CPU (accelerator/faults.mojo), so a
    failing query raises exactly the CPU message.
    """
    var expected: DataFrame
    try:
        expected = query.collect(engine="cpu")
    except cpu_error:
        with assert_raises(contains=String(cpu_error)):
            _ = query.collect(accelerator=MetalRuntime())
        return
    assert_true(query.collect(accelerator=MetalRuntime()).equals(expected))


def overflowed(query: LazyFrame, runtime: MetalRuntime) raises -> Bool:
    """Whether the CPU raised an Int64 overflow, which Metal must then raise.

    Returns False, with nothing asserted, when the CPU produced a result; the
    caller compares the two results. Any other CPU error propagates.
    """
    try:
        _ = query.collect(engine="cpu", streaming=False)
    except cpu_error:
        if "overflow" not in String(cpu_error):
            raise cpu_error^
        with assert_raises(contains=String(cpu_error)):
            _ = query.collect(accelerator=runtime)
        return True
    return False
