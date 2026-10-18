"""Metal returns the CPU's outcome for a query, including the CPU's error."""
from std.testing import assert_raises, assert_true
from dataframe import DataFrame
from dataframe.lazy import LazyFrame
from dataframe.metal import MetalRuntime


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
